# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Span primitives for page-level pipeline tracing.

A span is one timed operation applied to a page or a batch of pages. Spans are
recorded into the :class:`~nemo_retriever.common.tracing.collector.BatchTraceCollector`
that the enclosing operator installed; with no collector installed every helper
here degrades to a no-op so untraced runs pay nothing beyond a pointer check.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import secrets
import threading
import time
from typing import TYPE_CHECKING, Any, Final, Iterator, Literal

from nemo_retriever.common.tracing.runtime import (
    full_detail_enabled,
    record_model,
    worker_label,
)

if TYPE_CHECKING:
    from nemo_retriever.common.tracing.collector import BatchTraceCollector

SpanCategory = Literal["operator", "network", "gpu", "cpu", "io"]

#: Categories rolled up in ``by_category`` summaries.
SPAN_CATEGORIES: Final[tuple[str, ...]] = ("operator", "network", "gpu", "cpu", "io")

_ID_PREFIX: Final = secrets.token_hex(3)
_id_lock = threading.Lock()
_id_counter = 0

# One operator runs at a time per worker process, in both the Ray actor and the
# in-process executor. A module-level current collector therefore stays correct
# while also remaining visible to helper threads an operator may spawn
# internally, which a ContextVar would not be.
_collector_lock = threading.Lock()
_current: BatchTraceCollector | None = None

# Parent chains and page scoping are per-thread so concurrent helper threads do
# not interleave into each other's span hierarchies.
_local = threading.local()


def new_span_id() -> str:
    """Return a compact process-unique span identifier."""
    global _id_counter

    with _id_lock:
        _id_counter += 1
        counter = _id_counter
    return f"{_ID_PREFIX}{counter:x}"


@dataclass(slots=True)
class Span:
    """One timed operation recorded against a page or batch of pages."""

    span_id: str
    name: str
    category: str = "cpu"
    parent_span_id: str | None = None
    operator: str | None = None
    model_key: str | None = None
    source_id: str | None = None
    start_ms: float = 0.0
    end_ms: float = 0.0
    duration_ms: float = 0.0
    batch_size: int = 1
    status: str = "ok"
    error: str | None = None
    worker: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dict, omitting empty optional fields."""
        payload: dict[str, Any] = {
            "span_id": self.span_id,
            "name": self.name,
            "category": self.category,
            "start_ms": round(self.start_ms, 3),
            "end_ms": round(self.end_ms, 3),
            "duration_ms": round(self.duration_ms, 3),
            "batch_size": self.batch_size,
        }
        if self.parent_span_id:
            payload["parent_span_id"] = self.parent_span_id
        if self.operator:
            payload["operator"] = self.operator
        if self.model_key:
            payload["model_key"] = self.model_key
        if self.source_id:
            payload["source_id"] = self.source_id
        if self.status != "ok":
            payload["status"] = self.status
        if self.error:
            payload["error"] = self.error
        if self.worker:
            payload["worker"] = self.worker
        if self.attrs:
            payload["attrs"] = self.attrs
        return payload


class _NullSpan:
    """Stand-in returned when tracing is disabled."""

    __slots__ = ()

    span_id = ""

    def set_attr(self, key: str, value: Any) -> None:  # noqa: D102 - no-op
        return

    def set_attrs(self, **values: Any) -> None:  # noqa: D102 - no-op
        return

    def set_model(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: D102 - no-op
        return


class SpanHandle:
    """Mutable handle allowing a caller to annotate an in-flight span."""

    __slots__ = ("_span",)

    def __init__(self, span: Span) -> None:
        self._span = span

    @property
    def span_id(self) -> str:
        return self._span.span_id

    def set_attr(self, key: str, value: Any) -> None:
        """Attach a single attribute to the span."""
        if value is not None:
            self._span.attrs[key] = value

    def set_attrs(self, **values: Any) -> None:
        """Attach several attributes, skipping ``None`` values."""
        for key, value in values.items():
            if value is not None:
                self._span.attrs[key] = value

    def set_model(
        self,
        model_key: str,
        *,
        name: str | None = None,
        version: str | None = None,
        backend: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        """Associate the span with a model and register that model's identity."""
        self._span.model_key = model_key
        record_model(model_key, name=name, version=version, backend=backend, endpoint=endpoint)


NULL_SPAN: Final = _NullSpan()


def _current_collector() -> BatchTraceCollector | None:
    with _collector_lock:
        return _current


def set_current_collector(collector: BatchTraceCollector | None) -> BatchTraceCollector | None:
    """Install *collector* as the active recorder and return the previous one."""
    global _current

    with _collector_lock:
        previous = _current
        _current = collector
    return previous


def _stack() -> list[str]:
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = []
        _local.stack = stack
    return stack


def push_span_id(span_id: str) -> None:
    """Make *span_id* the parent for spans opened next on this thread."""
    _stack().append(span_id)


def pop_span_id(span_id: str) -> None:
    """Remove *span_id* from this thread's parent chain."""
    stack = _stack()
    if stack and stack[-1] == span_id:
        stack.pop()
    elif span_id in stack:
        stack.remove(span_id)


def current_parent_span_id() -> str | None:
    """Return the span that would parent a span opened right now."""
    stack = _stack()
    return stack[-1] if stack else None


def current_page_scope() -> str | None:
    """Return the ``source_id`` spans are currently attributed to, if any."""
    return getattr(_local, "page", None)


@contextmanager
def page_scope(source_id: str | None) -> Iterator[None]:
    """Attribute spans recorded in this block to a single page.

    Operators that loop over pages inside one batch use this so per-page work is
    charged to the right page instead of amortized across the batch.
    """
    previous = getattr(_local, "page", None)
    _local.page = source_id
    try:
        yield
    finally:
        _local.page = previous


@contextmanager
def span(
    name: str,
    *,
    category: str = "cpu",
    model_key: str | None = None,
    batch_size: int | None = None,
    attrs: dict[str, Any] | None = None,
    detail: str = "operator",
    parent_span_id: str | None = None,
    source_id: str | None = None,
) -> Iterator[Any]:
    """Record a timed span for *name* against the active collector.

    Spans measure entry and exit wall time around the call and nothing more.
    They deliberately do not synchronize CUDA or drive a profiler, so tracing
    stays cheap enough to leave on and never changes how the pipeline runs.
    Reach for Nsight Systems when you need true device-level attribution.

    Parameters
    ----------
    name
        Span label, for example ``"nim.infer"`` or an operator class name.
    category
        One of :data:`SPAN_CATEGORIES`; drives the ``by_category`` rollups.
    model_key
        Key of the model this operation used, linking the span to a registered
        model version.
    batch_size
        Number of pages or items the operation covered.
    detail
        Minimum detail level at which this span is recorded. Curated hot-spot
        spans pass ``"full"``; operator spans use the default.
    parent_span_id
        Explicit parent, for work handed to a worker thread. Parent chains are
        thread-local, so a span opened in a pool thread would otherwise be
        recorded as a root and its time would not be subtracted from the
        enclosing operator's self time.
    source_id
        Charge this span to one page directly, instead of inheriting the
        enclosing :func:`page_scope`. Convenient for operators that already
        have the page's ``source_id`` in hand while looping over a batch.
    """
    collector = _current_collector()
    if collector is None or (detail == "full" and not full_detail_enabled()):
        yield NULL_SPAN
        return

    stack = _stack()
    resolved_source_id = source_id or current_page_scope()
    if batch_size is not None:
        resolved_batch_size = int(batch_size)
    else:
        # A span charged to a single page covers one page by definition;
        # otherwise it covers whatever the operator was handed.
        resolved_batch_size = 1 if resolved_source_id else collector.batch_size
    record = Span(
        span_id=new_span_id(),
        name=name,
        category=category,
        parent_span_id=parent_span_id or (stack[-1] if stack else None),
        operator=collector.operator_name,
        model_key=model_key,
        source_id=resolved_source_id,
        batch_size=resolved_batch_size,
        worker=worker_label(),
        attrs=dict(attrs) if attrs else {},
    )
    handle = SpanHandle(record)

    stack.append(record.span_id)
    start_wall = time.time()
    start = time.perf_counter()
    try:
        yield handle
    except BaseException as exc:
        record.status = "error"
        record.error = f"{type(exc).__name__}: {exc}"[:400]
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        record.start_ms = start_wall * 1000.0
        record.duration_ms = duration_ms
        record.end_ms = record.start_ms + duration_ms
        stack.pop()
        collector.add(record)


@contextmanager
def accumulate(
    name: str,
    *,
    category: str = "cpu",
    source_id: str | None = None,
    detail: str = "full",
) -> Iterator[Any]:
    """Fold repeated calls to a hot helper into a single rolled-up span.

    Use this instead of :func:`span` for work invoked many times per page, such
    as per-crop image encoding. The resulting span reports the summed duration
    and a ``calls`` attribute rather than one record per invocation, which
    keeps trace artifacts small enough to stay useful.
    """
    collector = _current_collector()
    if collector is None or (detail == "full" and not full_detail_enabled()):
        yield NULL_SPAN
        return

    resolved_source_id = source_id or current_page_scope()
    stack = _stack()
    parent = stack[-1] if stack else None
    key = (name, resolved_source_id, parent)

    def _factory() -> Span:
        return Span(
            span_id=new_span_id(),
            name=name,
            category=category,
            parent_span_id=parent,
            operator=collector.operator_name,
            source_id=resolved_source_id,
            batch_size=1 if resolved_source_id else collector.batch_size,
            worker=worker_label(),
            attrs={"calls": 0, "rolled_up": True},
        )

    record = collector.accumulated_span(key, _factory)
    start_wall = time.time()
    start = time.perf_counter()
    try:
        yield SpanHandle(record)
    except BaseException as exc:
        record.status = "error"
        record.error = f"{type(exc).__name__}: {exc}"[:400]
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        if not record.start_ms:
            record.start_ms = start_wall * 1000.0
        record.duration_ms += duration_ms
        record.end_ms = max(record.end_ms, start_wall * 1000.0 + duration_ms)
        record.attrs["calls"] = int(record.attrs.get("calls", 0)) + 1


@contextmanager
def model_span(
    model_key: str,
    *,
    name: str | None = None,
    version: str | None = None,
    backend: str | None = None,
    endpoint: str | None = None,
    category: str = "network",
    span_name: str | None = None,
    batch_size: int | None = None,
    attrs: dict[str, Any] | None = None,
    detail: str = "full",
    parent_span_id: str | None = None,
) -> Iterator[Any]:
    """Record a model invocation span and register the model's identity.

    Model registration happens even when the span itself is filtered out by the
    detail level, so traces always report which model version ran.
    """
    record_model(model_key, name=name, version=version, backend=backend, endpoint=endpoint)
    with span(
        span_name or f"model.{model_key}",
        category=category,
        model_key=model_key,
        batch_size=batch_size,
        attrs=attrs,
        detail=detail,
        parent_span_id=parent_span_id,
    ) as handle:
        yield handle
