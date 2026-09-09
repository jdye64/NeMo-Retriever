# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate on-row page traces into document-level trace artifacts.

Rows arriving here may carry duplicate spans, because stages that fan a page
row out into element or chunk rows deep-copy the trace payload along with
everything else. Spans are therefore deduped by ``span_id``.

Time attribution
----------------
An operator processes a batch, so one operator span covers several pages. Each
emitted record reports both the exact measured ``duration_ms`` and an
``amortized_ms`` equal to ``duration_ms`` divided by the number of distinct
pages the span applies to. Amortized values are the ones safe to sum: summing
them across pages reproduces the exact measured duration, so page rollups and
document totals reconcile.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable
import uuid

import pandas as pd

from nemo_retriever.common.tracing.collector import decode_payload
from nemo_retriever.common.tracing.runtime import (
    TRACE_COLUMN,
    TRACE_SCHEMA_VERSION,
    model_identity,
)
from nemo_retriever.common.tracing.spans import SPAN_CATEGORIES

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _epoch_ms_to_iso(value: float | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def default_document_id(source_path: str) -> str:
    """Return a filesystem-safe, collision-resistant id for *source_path*."""
    name = Path(source_path).name or "document"
    slug = _SLUG_RE.sub("_", name).strip("._-") or "document"
    digest = hashlib.sha1(source_path.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{slug}-{digest}"


def _source_type(source_path: str) -> str:
    suffix = Path(source_path).suffix.lstrip(".").lower()
    return suffix or "unknown"


def _page_number_from_source_id(source_id: str | None) -> int | None:
    """Derive a page number from a ``source_id`` shaped ``{path}_{page}``."""
    if not source_id:
        return None
    tail = source_id.rpartition("_")[2]
    if tail.isdigit():
        return int(tail)
    return None


class _DocumentAccumulator:
    """Collects deduped spans and page identities for one source document."""

    __slots__ = ("source_path", "spans", "span_pages", "pages", "models", "model_keys")

    def __init__(self, source_path: str) -> None:
        self.source_path = source_path
        self.spans: dict[str, dict[str, Any]] = {}
        self.span_pages: dict[str, set[int]] = defaultdict(set)
        self.pages: dict[int, str] = {}
        self.models: list[dict[str, Any]] = []
        self.model_keys: set[tuple[Any, ...]] = set()

    def add_row(self, spans: Iterable[dict[str, Any]], source_id: str | None, page_number: int | None) -> None:
        if page_number is not None and source_id:
            self.pages.setdefault(page_number, source_id)
        for span in spans:
            span_id = span.get("span_id")
            if not span_id:
                continue
            self.spans.setdefault(span_id, span)
            if page_number is not None:
                self.span_pages[span_id].add(page_number)

    def add_models(self, models: Iterable[dict[str, Any]]) -> None:
        for descriptor in models:
            identity = model_identity(descriptor)
            if identity in self.model_keys:
                continue
            self.model_keys.add(identity)
            self.models.append(descriptor)


def _child_duration_totals(spans: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Return, per span id, the summed duration of its direct children."""
    totals: dict[str, float] = defaultdict(float)
    for span in spans.values():
        parent = span.get("parent_span_id")
        if parent:
            totals[parent] += float(span.get("duration_ms") or 0.0)
    return totals


def _empty_category_totals() -> dict[str, float]:
    return {f"{category}_ms": 0.0 for category in SPAN_CATEGORIES}


def _round_mapping(mapping: dict[str, float], digits: int = 3) -> dict[str, float]:
    return {key: round(value, digits) for key, value in sorted(mapping.items(), key=lambda item: -item[1])}


def _build_document_trace(
    accumulator: _DocumentAccumulator,
    *,
    document_id: str,
    run: dict[str, Any],
    version_info: dict[str, Any],
    pipeline: dict[str, Any],
    span_fanout: dict[str, int],
) -> dict[str, Any]:
    spans = accumulator.spans
    child_totals = _child_duration_totals(spans)

    # Batch-scoped spans apply to every page in their batch; page-scoped spans
    # to exactly one. A span observed with no page at all (document-level work
    # before splitting) is charged to every page the document produced.
    all_pages = sorted(accumulator.pages)
    span_records: list[dict[str, Any]] = []
    page_operator_ms: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    page_category_ms: dict[int, dict[str, float]] = defaultdict(_empty_category_totals)
    page_bounds: dict[int, list[float]] = {}
    page_span_counts: dict[int, int] = defaultdict(int)

    document_operator_ms: dict[str, float] = defaultdict(float)
    document_operator_self_ms: dict[str, float] = defaultdict(float)
    document_operator_calls: dict[str, int] = defaultdict(int)
    document_category_ms = _empty_category_totals()
    document_model_ms: dict[str, float] = defaultdict(float)

    start_bounds: list[float] = []
    end_bounds: list[float] = []
    error_count = 0

    for span_id, span in spans.items():
        pages = sorted(accumulator.span_pages.get(span_id) or ())
        if not pages:
            pages = all_pages
        # Divide by every page the span covered across the whole batch, not
        # just the ones belonging to this document. A batch can mix documents,
        # and this document is only owed its share.
        fanout = span_fanout.get(span_id) or len(pages) or 1

        duration_ms = float(span.get("duration_ms") or 0.0)
        self_ms = max(0.0, duration_ms - child_totals.get(span_id, 0.0))
        amortized_ms = duration_ms / fanout
        amortized_self_ms = self_ms / fanout
        category = str(span.get("category") or "cpu")
        operator = span.get("operator") or span.get("name") or "unknown"
        status = str(span.get("status") or "ok")
        if status != "ok":
            error_count += 1

        start_ms = float(span.get("start_ms") or 0.0)
        end_ms = float(span.get("end_ms") or 0.0)
        if start_ms:
            start_bounds.append(start_ms)
        if end_ms:
            end_bounds.append(end_ms)

        # Document rollups accumulate amortized time per page below, not the
        # raw duration. A batch can mix pages from several documents, so
        # charging each document the whole batch duration would inflate every
        # one of them. Amortizing keeps document totals equal to the sum of
        # their page summaries.
        if category == "operator":
            document_operator_calls[operator] += 1
        category_field = f"{category}_ms"
        model_key = span.get("model_key")

        for page_number in pages:
            page_span_counts[page_number] += 1
            if category == "operator":
                page_operator_ms[page_number][operator] += amortized_ms
                document_operator_ms[operator] += amortized_ms
                document_operator_self_ms[operator] += amortized_self_ms
            if category_field in page_category_ms[page_number]:
                page_category_ms[page_number][category_field] += amortized_ms
                document_category_ms[category_field] += amortized_ms
            if model_key:
                document_model_ms[str(model_key)] += amortized_ms
            if start_ms and end_ms:
                bounds = page_bounds.get(page_number)
                if bounds is None:
                    page_bounds[page_number] = [start_ms, end_ms]
                else:
                    bounds[0] = min(bounds[0], start_ms)
                    bounds[1] = max(bounds[1], end_ms)

            record = {
                "document_id": document_id,
                "source_path": accumulator.source_path,
                "page_number": page_number,
                "source_id": accumulator.pages.get(page_number),
                "span_id": span_id,
                "parent_span_id": span.get("parent_span_id"),
                "name": span.get("name"),
                "category": category,
                "operator": operator,
                "model_key": model_key,
                "start_ms": span.get("start_ms"),
                "end_ms": span.get("end_ms"),
                "duration_ms": round(duration_ms, 3),
                "self_ms": round(self_ms, 3),
                "amortized_ms": round(amortized_ms, 3),
                "amortized_self_ms": round(amortized_self_ms, 3),
                "page_fanout": fanout,
                "batch_size": span.get("batch_size", 1),
                "status": status,
                "error": span.get("error"),
                "worker": span.get("worker"),
                "attrs": span.get("attrs") or {},
            }
            span_records.append(record)

    span_records.sort(key=lambda item: (item["page_number"], item["start_ms"] or 0.0))

    page_summaries = []
    for page_number in all_pages:
        operator_ms = dict(page_operator_ms.get(page_number, {}))
        bounds = page_bounds.get(page_number)
        page_summaries.append(
            {
                "page_number": page_number,
                "source_id": accumulator.pages.get(page_number),
                "total_ms": round(sum(operator_ms.values()), 3),
                "wall_ms": round(bounds[1] - bounds[0], 3) if bounds else 0.0,
                "span_count": page_span_counts.get(page_number, 0),
                "by_operator": _round_mapping(operator_ms),
                "by_category": _round_mapping(dict(page_category_ms.get(page_number, _empty_category_totals()))),
            }
        )

    document_total_ms = sum(document_operator_ms.values())
    by_operator = [
        {
            "operator": operator,
            "calls": document_operator_calls[operator],
            "total_ms": round(total, 3),
            "self_ms": round(document_operator_self_ms[operator], 3),
            "pct_of_total": round(100.0 * total / document_total_ms, 2) if document_total_ms else 0.0,
        }
        for operator, total in sorted(document_operator_ms.items(), key=lambda item: -item[1])
    ]

    wall_ms = round(max(end_bounds) - min(start_bounds), 3) if start_bounds and end_bounds else 0.0

    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "nemo_retriever": version_info,
        "run": run,
        "document": {
            "document_id": document_id,
            "source_path": accumulator.source_path,
            "source_type": _source_type(accumulator.source_path),
            "page_count": len(all_pages),
            "status": "error" if error_count else "completed",
            "error_span_count": error_count,
            "wall_ms": wall_ms,
            "started_at": _epoch_ms_to_iso(min(start_bounds) if start_bounds else None),
            "completed_at": _epoch_ms_to_iso(max(end_bounds) if end_bounds else None),
        },
        "pipeline": pipeline,
        "models": accumulator.models,
        "document_summary": {
            "total_ms": round(document_total_ms, 3),
            "wall_ms": wall_ms,
            "span_count": len(spans),
            "by_operator": by_operator,
            "by_category": _round_mapping(document_category_ms),
            "by_model": _round_mapping(dict(document_model_ms)),
        },
        "page_summaries": page_summaries,
        "spans": span_records,
    }


def aggregate_document_traces(
    frame: Any,
    *,
    run_mode: str = "inprocess",
    run_id: str | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    pipeline_operators: list[str] | None = None,
    params: dict[str, Any] | None = None,
    document_id: str | None = None,
    otel_trace_id: str | None = None,
) -> list[dict[str, Any]]:
    """Build one document trace artifact per source document in *frame*.

    Parameters
    ----------
    frame
        Final pipeline DataFrame, still carrying the ``_nrl_trace`` column.
    document_id
        Override for the generated document id. Service mode passes the
        server-side document identifier so trace files line up with result
        files.

    Returns
    -------
    list of dict
        Trace documents matching schema version
        :data:`~nemo_retriever.common.tracing.runtime.TRACE_SCHEMA_VERSION`.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty or TRACE_COLUMN not in frame.columns:
        return []

    from nemo_retriever.version import get_version_info

    accumulators: dict[str, _DocumentAccumulator] = {}

    raw_traces = frame[TRACE_COLUMN].tolist()
    paths = frame["path"].tolist() if "path" in frame.columns else [None] * len(raw_traces)
    source_ids = frame["source_id"].tolist() if "source_id" in frame.columns else [None] * len(raw_traces)
    page_numbers = frame["page_number"].tolist() if "page_number" in frame.columns else [None] * len(raw_traces)

    for raw, path, source_id, page_number in zip(raw_traces, paths, source_ids, page_numbers):
        spans, models = decode_payload(raw)
        if not spans and not models:
            continue

        source_id_text = None if source_id is None else str(source_id)
        if source_id_text in {"", "nan", "None", "<NA>"}:
            source_id_text = None

        path_text = None if path is None else str(path)
        if path_text in {"", "nan", "None", "<NA>"}:
            path_text = None
        if path_text is None and source_id_text:
            path_text = source_id_text.rpartition("_")[0] or source_id_text
        if path_text is None:
            path_text = "unknown"

        resolved_page: int | None
        try:
            resolved_page = int(page_number) if page_number is not None and page_number == page_number else None
        except (TypeError, ValueError):
            resolved_page = None
        if resolved_page is None:
            resolved_page = _page_number_from_source_id(source_id_text)

        accumulator = accumulators.get(path_text)
        if accumulator is None:
            accumulator = _DocumentAccumulator(path_text)
            accumulators[path_text] = accumulator
        accumulator.add_row(spans, source_id_text, resolved_page)
        accumulator.add_models(models)

    if not accumulators:
        return []

    version_info = get_version_info()
    run = {
        "run_id": run_id or uuid.uuid4().hex,
        "run_mode": run_mode,
        "started_at": started_at,
        "completed_at": completed_at or _utc_now_iso(),
        "otel_trace_id": otel_trace_id,
    }
    pipeline = {
        "operators": list(pipeline_operators or []),
        "params": dict(params or {}),
    }

    span_fanout: dict[str, int] = defaultdict(int)
    for accumulator in accumulators.values():
        for span_id, pages in accumulator.span_pages.items():
            span_fanout[span_id] += len(pages)

    traces: list[dict[str, Any]] = []
    for path_text, accumulator in accumulators.items():
        resolved_document_id = document_id if document_id and len(accumulators) == 1 else default_document_id(path_text)
        try:
            traces.append(
                _build_document_trace(
                    accumulator,
                    document_id=resolved_document_id,
                    run=run,
                    version_info=version_info,
                    pipeline=pipeline,
                    span_fanout=span_fanout,
                )
            )
        except Exception:
            logger.warning("Failed to build page trace for %s.", path_text, exc_info=True)
    return traces


def trace_filename(document_id: str, *, compression: str | None = None) -> str:
    """Return the on-disk filename for a document trace artifact."""
    suffix = ".trace.json.gz" if compression == "gzip" else ".trace.json"
    return f"{document_id}{suffix}"


def write_document_trace(
    trace: dict[str, Any],
    output_directory: str | Path,
    *,
    compression: str | None = None,
) -> Path:
    """Write one document trace to *output_directory* and return its path."""
    directory = Path(output_directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    document_id = str(trace.get("document", {}).get("document_id") or "document")
    out_path = directory / trace_filename(document_id, compression=compression)
    payload = json.dumps(trace, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    if compression == "gzip":
        with gzip.open(out_path, "wb") as handle:
            handle.write(payload)
    else:
        out_path.write_bytes(payload)
    return out_path


def write_document_traces(
    traces: Iterable[dict[str, Any]],
    output_directory: str | Path,
    *,
    compression: str | None = None,
) -> list[Path]:
    """Write several document traces, returning the paths written."""
    return [write_document_trace(trace, output_directory, compression=compression) for trace in traces]


def strip_trace_column(frame: Any) -> Any:
    """Drop the internal trace column so user-facing results stay clean."""
    if isinstance(frame, pd.DataFrame) and TRACE_COLUMN in frame.columns:
        return frame.drop(columns=[TRACE_COLUMN])
    return frame
