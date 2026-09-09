# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch-scoped trace collection and the on-row trace payload format.

Traces travel with the data as a JSON string in the ``_nrl_trace`` column. A
string column is the safest payload Ray Data can carry: Arrow represents it
natively, and every operator in the graph preserves unknown columns because
they copy the frame or rebuild rows from ``row.to_dict()`` rather than
declaring fixed output schemas.

Rows are matched between an operator's input and output by ``source_id``
(``{path}_{page_number}``), falling back to ``path`` for stages that run before
:class:`~nemo_retriever.operators.extract.pdf.split.PDFSplitActor` assigns page
identity. That fallback is what lets freshly split page rows inherit the
document-level spans recorded during format conversion.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import time
from typing import Any, Iterator

import pandas as pd

from nemo_retriever.common.tracing.runtime import (
    TRACE_COLUMN,
    model_identity,
    registered_models,
    tracing_enabled,
    worker_label,
)
from nemo_retriever.common.tracing.spans import (
    Span,
    new_span_id,
    pop_span_id,
    push_span_id,
    set_current_collector,
)

logger = logging.getLogger(__name__)

PAYLOAD_VERSION = 1


def _normalize_key(value: Any) -> str | None:
    """Return *value* as a non-empty string, or ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    if value is pd.NA:
        return None
    text = str(value)
    if not text or text in {"nan", "None", "<NA>"}:
        return None
    return text


def encode_payload(spans: list[dict[str, Any]], models: list[dict[str, Any]]) -> str:
    """Serialize a row's accumulated spans and model descriptors."""
    payload: dict[str, Any] = {"v": PAYLOAD_VERSION, "spans": spans}
    if models:
        payload["models"] = models
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def decode_payload(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a row's trace payload into ``(spans, models)``.

    Malformed payloads yield empty results rather than failing the pipeline;
    tracing must never be the reason an ingest run dies.
    """
    text = _normalize_key(raw)
    if text is None:
        return [], []
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        logger.debug("Discarding unparseable page trace payload.")
        return [], []
    if not isinstance(payload, dict):
        return [], []
    spans = payload.get("spans")
    models = payload.get("models")
    return (
        [item for item in spans if isinstance(item, dict)] if isinstance(spans, list) else [],
        [item for item in models if isinstance(item, dict)] if isinstance(models, list) else [],
    )


class BatchTraceCollector:
    """Accumulates spans for one operator invocation over one batch of rows."""

    def __init__(self, operator_name: str, frame: pd.DataFrame) -> None:
        self.operator_name = operator_name
        self.batch_size = int(len(frame))
        self._spans: list[Span] = []
        self._accumulated: dict[tuple[Any, ...], Span] = {}
        self._inherited_by_key: dict[str, list[dict[str, Any]]] = {}
        self._inherited_by_path: dict[str, list[dict[str, Any]]] = {}
        self._inherited_models: list[dict[str, Any]] = []
        self._single_input_spans: list[dict[str, Any]] | None = None
        self._read_input(frame)

    def _read_input(self, frame: pd.DataFrame) -> None:
        if TRACE_COLUMN not in frame.columns:
            return

        raw_traces = frame[TRACE_COLUMN].tolist()
        source_ids = frame["source_id"].tolist() if "source_id" in frame.columns else [None] * len(raw_traces)
        paths = frame["path"].tolist() if "path" in frame.columns else [None] * len(raw_traces)

        seen_models: set[tuple[Any, ...]] = set()
        for raw, source_id, path in zip(raw_traces, source_ids, paths):
            spans, models = decode_payload(raw)
            for descriptor in models:
                identity = model_identity(descriptor)
                if identity not in seen_models:
                    seen_models.add(identity)
                    self._inherited_models.append(descriptor)
            if not spans:
                continue
            key = _normalize_key(source_id)
            if key is not None:
                self._inherited_by_key.setdefault(key, []).extend(spans)
                continue
            # Only rows that predate page identity define document-level
            # lineage. Indexing page rows by path too would let one page
            # inherit every sibling page's spans on a fallback lookup.
            path_key = _normalize_key(path)
            if path_key is not None:
                bucket = self._inherited_by_path.setdefault(path_key, [])
                known = {item.get("span_id") for item in bucket}
                bucket.extend(item for item in spans if item.get("span_id") not in known)

        if len(raw_traces) == 1:
            spans, _ = decode_payload(raw_traces[0])
            self._single_input_spans = spans

    def add(self, span: Span) -> None:
        """Record a completed span."""
        self._spans.append(span)

    def accumulated_span(self, key: tuple[Any, ...], factory: Any) -> Span:
        """Return the rolled-up span for *key*, creating it on first use.

        Helpers called once per image crop would otherwise emit hundreds of
        spans per page. Folding them into one span per operator invocation
        keeps the artifact small while still reporting where the time went.
        Spans are serialized at :meth:`finish`, so mutating the returned object
        afterwards is safe.
        """
        span = self._accumulated.get(key)
        if span is None:
            span = factory()
            self._accumulated[key] = span
            self._spans.append(span)
        return span

    def make_operator_span(self, span_id: str, *, status: str = "ok", error: str | None = None) -> Span:
        """Create the batch-scoped span representing this operator invocation."""
        return Span(
            span_id=span_id,
            name=self.operator_name,
            category="operator",
            operator=self.operator_name,
            batch_size=self.batch_size,
            status=status,
            error=error,
            worker=worker_label(),
        )

    def _inherited_for(self, key: str | None, path: str | None) -> list[dict[str, Any]]:
        if key is not None:
            inherited = self._inherited_by_key.get(key)
            if inherited:
                return inherited
        if path is not None:
            inherited = self._inherited_by_path.get(path)
            if inherited:
                return inherited
        # A stage that reshapes one input row into new identities (format
        # conversion, page splitting) leaves nothing to match on. With a single
        # input row the lineage is unambiguous, so carry its spans forward.
        if self.batch_size == 1 and self._single_input_spans:
            return self._single_input_spans
        return []

    def finish(self, frame: Any) -> Any:
        """Write accumulated spans onto *frame* and return it."""
        if not isinstance(frame, pd.DataFrame):
            return frame
        if frame.empty:
            # Still declare the column so Ray Data does not have to reconcile
            # blocks whose schemas differ only by the trace column.
            if TRACE_COLUMN not in frame.columns:
                frame[TRACE_COLUMN] = pd.Series([], dtype=object)
            return frame

        page_spans: dict[str, list[dict[str, Any]]] = {}
        batch_spans: list[dict[str, Any]] = []
        for span in self._spans:
            if span.source_id:
                page_spans.setdefault(span.source_id, []).append(span.to_dict())
            else:
                batch_spans.append(span.to_dict())

        source_ids = frame["source_id"].tolist() if "source_id" in frame.columns else [None] * len(frame)
        paths = frame["path"].tolist() if "path" in frame.columns else [None] * len(frame)
        output_keys = {key for key in (_normalize_key(value) for value in source_ids) if key is not None}

        # Page-scoped spans whose page vanished from the output would otherwise
        # drop off the trace entirely. Demote them to batch scope so document
        # totals still account for the time spent.
        for scoped_key, spans in page_spans.items():
            if scoped_key not in output_keys:
                batch_spans.extend(spans)
        page_spans = {key: spans for key, spans in page_spans.items() if key in output_keys}

        models = list(self._inherited_models)
        known_models = {model_identity(descriptor) for descriptor in models}
        for descriptor in registered_models():
            identity = model_identity(descriptor)
            if identity not in known_models:
                known_models.add(identity)
                models.append(descriptor)

        payloads: list[str] = []
        for source_id, path in zip(source_ids, paths):
            key = _normalize_key(source_id)
            path_key = _normalize_key(path)
            combined: list[dict[str, Any]] = []
            seen: set[str] = set()
            for span in (
                *self._inherited_for(key, path_key),
                *batch_spans,
                *(page_spans.get(key, ()) if key is not None else ()),
            ):
                span_id = span.get("span_id")
                if span_id in seen:
                    continue
                seen.add(span_id)
                combined.append(span)
            payloads.append(encode_payload(combined, models))

        # Assign the list positionally. Building a Series first would align on
        # the frame index, which stages that rebuild rows can leave duplicated.
        frame[TRACE_COLUMN] = payloads
        return frame


class _NullTracer:
    """Tracer used when tracing is disabled or the payload is not a frame."""

    __slots__ = ()

    def finish(self, frame: Any) -> Any:
        return frame

    def abort(self, _exc: BaseException) -> None:
        return


NULL_TRACER = _NullTracer()


class _OperatorTracer:
    """Times one operator invocation and writes the result onto the output."""

    __slots__ = ("_collector", "_span_id", "_start", "_recorded")

    def __init__(self, collector: BatchTraceCollector) -> None:
        self._collector = collector
        # Reserve the operator span id up front and make it the active parent,
        # so hot-spot spans recorded inside the operator nest underneath it and
        # their time can be subtracted to give the operator's self time.
        self._span_id = new_span_id()
        push_span_id(self._span_id)
        self._start = (time.time(), time.perf_counter())
        self._recorded = False

    def _record(self, *, status: str = "ok", error: str | None = None) -> None:
        if self._recorded:
            return
        self._recorded = True

        pop_span_id(self._span_id)
        start_wall, start_perf = self._start
        duration_ms = (time.perf_counter() - start_perf) * 1000.0
        span = self._collector.make_operator_span(self._span_id, status=status, error=error)
        span.start_ms = start_wall * 1000.0
        span.duration_ms = duration_ms
        span.end_ms = span.start_ms + duration_ms
        self._collector.add(span)

    def finish(self, frame: Any) -> Any:
        """Close the operator span and stamp the trace column onto *frame*."""
        self._record()
        try:
            return self._collector.finish(frame)
        except Exception:
            logger.warning("Failed to attach page trace spans for %s.", self._collector.operator_name, exc_info=True)
            return frame

    def abort(self, exc: BaseException) -> None:
        """Close the operator span as failed."""
        self._record(status="error", error=f"{type(exc).__name__}: {exc}"[:400])

    def release(self) -> None:
        """Free the reserved parent slot if the invocation never closed it."""
        if not self._recorded:
            self._recorded = True
            pop_span_id(self._span_id)


def operator_name_for(operator: Any) -> str:
    """Return the trace label for *operator*.

    ``UDFOperator`` instances carry a descriptive ``name`` set at graph build
    time (``"DedupImages"``, ``"ExplodeContentToRows"``), which is far more
    useful in a trace than the shared wrapper class name.
    """
    name = getattr(operator, "name", None)
    if isinstance(name, str) and name:
        return name
    return type(operator).__name__


@contextmanager
def operator_trace(operator: Any, data: Any) -> Iterator[Any]:
    """Install a collector for one operator invocation over *data*.

    Yields a tracer whose ``finish`` must be called with the operator's output
    frame. Non-DataFrame payloads and disabled tracing yield a no-op tracer.
    """
    if not tracing_enabled() or not isinstance(data, pd.DataFrame) or data.empty:
        yield NULL_TRACER
        return

    try:
        collector = BatchTraceCollector(operator_name_for(operator), data)
    except Exception:
        logger.warning("Failed to initialize page trace collector; continuing untraced.", exc_info=True)
        yield NULL_TRACER
        return

    tracer = _OperatorTracer(collector)
    previous = set_current_collector(collector)
    try:
        yield tracer
    except BaseException as exc:
        tracer.abort(exc)
        raise
    finally:
        tracer.release()
        set_current_collector(previous)
