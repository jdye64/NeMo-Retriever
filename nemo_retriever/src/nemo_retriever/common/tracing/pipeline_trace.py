# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-page span collection for the inprocess ingest run mode.

A :class:`PipelineTrace` is the payload the ingestor threads through an
inprocess pipeline run. Executors open one span per graph stage and
instrumented operators open nested spans around per-page and per-batch work.
Because pages are also described by the characteristics recorded in
:attr:`PipelineTrace.page_features`, the returned payload answers both
"which stage is the hotspot" and "what is different about the pages that took
longest".

Operators reach the active trace through a :mod:`contextvars` handle rather
than a changed call signature, so an uninstrumented operator costs nothing and
an instrumented one costs a single ``ContextVar.get`` when tracing is off::

    from nemo_retriever.common.tracing import trace_span

    with trace_span("pdf_extract.render_page", page=(path, page_number)) as span:
        span.set(dpi=dpi)

Timing uses :func:`time.perf_counter`, so every ``*_s`` value is a wall-clock
duration in seconds measured from the start of the run.
"""

from __future__ import annotations

import contextvars
import itertools
import json
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "PageIndex",
    "PageKey",
    "PipelineTrace",
    "Span",
    "UNATTRIBUTED_PAGE",
    "UNKNOWN_SOURCE",
    "active_trace",
    "activate_trace",
    "observe_frame",
    "page_index",
    "page_key",
    "page_key_from_row",
    "page_keys_from_frame",
    "stage_span",
    "trace_span",
]

#: A page is addressed by its source document and 1-indexed page number.
#: Page ``0`` means the span covered a whole document rather than one page,
#: which is the case for stages that run before a document is split.
PageKey = tuple[str, int]

UNKNOWN_SOURCE = "<unknown>"

#: Bucket for span time that covered no identifiable page.
UNATTRIBUTED_PAGE: "PageKey" = ("<unattributed>", -1)

_DOCUMENT_LEVEL_PAGE = 0

# Columns inspected to describe a page. Every one is optional: a frame is
# described by whichever of these it happens to carry at that point in the run.
_SOURCE_COLUMNS = ("path", "source_path", "source_name")

_CURRENT_TRACE: contextvars.ContextVar["PipelineTrace | None"] = contextvars.ContextVar(
    "nemo_retriever_pipeline_trace", default=None
)
_CURRENT_SPAN: contextvars.ContextVar["Span | None"] = contextvars.ContextVar(
    "nemo_retriever_pipeline_span", default=None
)


def page_key(source: Any, page_number: Any = _DOCUMENT_LEVEL_PAGE) -> PageKey:
    """Normalize a source identifier and page number into a :data:`PageKey`."""
    text = str(source).strip() if source is not None else ""
    try:
        number = int(page_number)
    except (TypeError, ValueError):
        number = _DOCUMENT_LEVEL_PAGE
    return (text or UNKNOWN_SOURCE, number)


def _row_value(row: Any, name: str) -> Any:
    """Read one field from a mapping, pandas Series, or ``itertuples`` row."""
    if isinstance(row, Mapping):
        return row.get(name)
    getter = getattr(row, "get", None)
    if callable(getter):
        try:
            return getter(name)
        except Exception:
            return None
    return getattr(row, name, None)


def page_key_from_row(row: Any) -> PageKey:
    """Derive a :data:`PageKey` from a pipeline row.

    Falls back through ``path``, ``source_path``, ``source_name``, and
    ``metadata["source_path"]`` so the key stays stable across the stages that
    rename or drop the original path column.
    """
    source: Any = None
    for column in _SOURCE_COLUMNS:
        value = _row_value(row, column)
        if value is not None and str(value).strip():
            source = value
            break
    if source is None:
        metadata = _row_value(row, "metadata")
        if isinstance(metadata, Mapping):
            source = metadata.get("source_path") or metadata.get("source_name")
    if source is None:
        source = _row_value(row, "source_id")
    return page_key(source, _row_value(row, "page_number"))


def _row_page_keys(frame: Any) -> tuple[PageKey, ...]:
    """Return one page key per frame row, in row order."""
    if frame is None or not hasattr(frame, "columns"):
        return ()
    try:
        if len(frame.index) == 0:
            return ()
        columns = [name for name in (*_SOURCE_COLUMNS, "source_id", "metadata", "page_number") if name in frame.columns]
        if not columns:
            return ()
        return tuple(page_key_from_row(row) for row in frame.loc[:, columns].to_dict("records"))
    except Exception:
        return ()


def page_keys_from_frame(frame: Any) -> tuple[PageKey, ...]:
    """Return the distinct page keys addressed by a pandas frame, in order."""
    return tuple(dict.fromkeys(_row_page_keys(frame)))


class PageIndex:
    """Per-row page-key lookup used by operators that batch rows together.

    Instances are created through :func:`page_index`, which returns an empty
    index when no trace is active so an operator's tracing calls cost nothing
    beyond a ``ContextVar.get``.
    """

    __slots__ = ("_keys",)

    def __init__(self, keys: tuple[PageKey, ...] = ()) -> None:
        self._keys = keys

    def __call__(self, row_index: int) -> PageKey | None:
        """Return the page key for one row position, or ``None`` if unknown."""
        if 0 <= row_index < len(self._keys):
            return self._keys[row_index]
        return None

    def many(self, row_indices: Iterable[int]) -> tuple[PageKey, ...]:
        """Return the distinct page keys covering several row positions."""
        keys = (self(int(index)) for index in row_indices)
        return tuple(dict.fromkeys(key for key in keys if key is not None))

    def all(self) -> tuple[PageKey, ...]:
        """Return every row's page key, in row order."""
        return self._keys


_EMPTY_PAGE_INDEX = PageIndex()


def page_index(frame: Any) -> PageIndex:
    """Return a per-row page-key lookup for *frame*, inert when tracing is off."""
    if _CURRENT_TRACE.get() is None:
        return _EMPTY_PAGE_INDEX
    return PageIndex(_row_page_keys(frame))


@dataclass
class Span:
    """One timed region of a pipeline run.

    A span may cover a single page (``page_keys`` of length one), a batch of
    pages sent to a model in one request, or a whole stage. Rollups divide a
    span's self time evenly across its pages, so a batched model call charges
    each page in the batch its share of the request.
    """

    span_id: int
    parent_id: int | None
    name: str
    stage: str | None
    depth: int
    start_s: float
    end_s: float | None = None
    page_keys: tuple[PageKey, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_s(self) -> float:
        """Inclusive wall-clock duration; ``0.0`` while the span is still open."""
        if self.end_s is None:
            return 0.0
        return max(self.end_s - self.start_s, 0.0)

    @property
    def page_count(self) -> int:
        return len(self.page_keys)

    def set(self, **attributes: Any) -> "Span":
        """Attach descriptive attributes to the span."""
        self.attributes.update(attributes)
        return self

    def add_pages(self, pages: Any) -> "Span":
        """Extend the span's page coverage, preserving order and uniqueness."""
        merged: dict[PageKey, None] = dict.fromkeys(self.page_keys)
        for key in _normalize_pages(pages):
            merged.setdefault(key, None)
        self.page_keys = tuple(merged)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "stage": self.stage,
            "depth": self.depth,
            "start_s": round(self.start_s, 6),
            "end_s": None if self.end_s is None else round(self.end_s, 6),
            "duration_s": round(self.duration_s, 6),
            "pages": [list(key) for key in self.page_keys],
            "attributes": dict(self.attributes),
            "error": self.error,
        }


class _NullSpan:
    """Stand-in returned by :func:`trace_span` when no trace is active."""

    __slots__ = ()

    duration_s = 0.0
    page_count = 0

    def set(self, **attributes: Any) -> "_NullSpan":
        return self

    def add_pages(self, pages: Any) -> "_NullSpan":
        return self


class _NullSpanContext:
    """Reusable no-op context manager, safe to enter concurrently."""

    __slots__ = ()

    def __enter__(self) -> _NullSpan:
        return _NULL_SPAN

    def __exit__(self, *exc_info: Any) -> bool:
        return False


_NULL_SPAN = _NullSpan()
_NULL_SPAN_CONTEXT = _NullSpanContext()


def _normalize_pages(pages: Any) -> tuple[PageKey, ...]:
    """Coerce loosely specified page references into :data:`PageKey` tuples."""
    if pages is None:
        return ()
    if isinstance(pages, tuple) and len(pages) == 2 and not isinstance(pages[0], (tuple, list)):
        return (page_key(pages[0], pages[1]),)
    if isinstance(pages, (str, bytes)):
        return (page_key(pages),)
    if isinstance(pages, Mapping):
        return (page_key_from_row(pages),)
    if isinstance(pages, Iterable):
        keys: list[PageKey] = []
        for item in pages:
            keys.extend(_normalize_pages(item))
        return tuple(dict.fromkeys(keys))
    return (page_key(pages),)


def _page_coverage(spans: "tuple[Span, ...]") -> dict[int, tuple[PageKey, ...]]:
    """Resolve which pages each span's time belongs to.

    A span that names no page of its own borrows the coverage of its nearest
    ancestor that does. Operators only have to label the span that knows the
    page — ``pdf_extract.page``, say — and its ``render`` and ``text`` children
    are charged to that same page instead of being written off as
    unattributable.
    """
    by_id = {span.span_id: span for span in spans}
    resolved: dict[int, tuple[PageKey, ...]] = {}

    def resolve(span: Span) -> tuple[PageKey, ...]:
        cached = resolved.get(span.span_id)
        if cached is not None:
            return cached
        # Placeholder first: a malformed parent chain must not recurse forever.
        resolved[span.span_id] = ()
        keys = span.page_keys
        if not keys and span.parent_id is not None:
            parent = by_id.get(span.parent_id)
            if parent is not None:
                keys = resolve(parent)
        resolved[span.span_id] = keys
        return keys

    for span in spans:
        resolve(span)
    return resolved


class PipelineTrace:
    """Span and page-characteristic payload carried through one ingest run.

    The ingestor creates the trace, the executor threads it through every
    stage, and the caller receives it once execution completes. Nothing in the
    payload references Ray, OpenTelemetry, or any other optional dependency, so
    it can be pickled, serialized to JSON, or inspected directly.
    """

    def __init__(self, *, run_mode: str = "inprocess", capture_page_features: bool = True) -> None:
        self.run_mode = run_mode
        self.capture_page_features = capture_page_features
        self.notes: list[str] = []
        self.started_at = time.time()
        self._origin_s = time.perf_counter()
        self._finished_s: float | None = None
        self._spans: list[Span] = []
        self._page_features: dict[PageKey, dict[str, Any]] = {}
        self._page_order: list[PageKey] = []
        self._stage_order: list[str] = []
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def elapsed_s(self) -> float:
        """Seconds since the trace was created."""
        return time.perf_counter() - self._origin_s

    @property
    def total_s(self) -> float:
        """Total run duration, measured to :meth:`finish` when it was called."""
        return self._finished_s if self._finished_s is not None else self.elapsed_s()

    @property
    def spans(self) -> tuple[Span, ...]:
        with self._lock:
            return tuple(self._spans)

    @property
    def page_features(self) -> dict[PageKey, dict[str, Any]]:
        with self._lock:
            return {key: dict(self._page_features[key]) for key in self._page_order}

    @property
    def stage_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._stage_order)

    def finish(self) -> "PipelineTrace":
        """Freeze :attr:`total_s` at the current elapsed time."""
        self._finished_s = self.elapsed_s()
        return self

    def note(self, message: str) -> "PipelineTrace":
        """Record a human-readable caveat about the run."""
        self.notes.append(str(message))
        return self

    @contextmanager
    def span(
        self,
        name: str,
        *,
        stage: str | None = None,
        page: Any = None,
        pages: Any = None,
        **attributes: Any,
    ) -> Iterator[Span]:
        """Open a span that nests under whichever span is currently active."""
        parent = _CURRENT_SPAN.get()
        keys = _normalize_pages(page) + _normalize_pages(pages)
        record = Span(
            span_id=next(self._ids),
            parent_id=None if parent is None else parent.span_id,
            name=str(name),
            stage=stage if stage is not None else (parent.stage if parent is not None else None),
            depth=0 if parent is None else parent.depth + 1,
            start_s=self.elapsed_s(),
            page_keys=tuple(dict.fromkeys(keys)),
            attributes=dict(attributes),
        )
        with self._lock:
            self._spans.append(record)
            if record.stage and record.stage not in self._stage_order:
                self._stage_order.append(record.stage)
        token = _CURRENT_SPAN.set(record)
        try:
            yield record
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _CURRENT_SPAN.reset(token)
            record.end_s = self.elapsed_s()

    def record_page_features(self, key: PageKey, **features: Any) -> None:
        """Merge page characteristics, keeping the most recent non-null value."""
        with self._lock:
            existing = self._page_features.get(key)
            if existing is None:
                existing = {"source": key[0], "page_number": key[1]}
                self._page_features[key] = existing
                self._page_order.append(key)
            for name, value in features.items():
                if value is not None:
                    existing[name] = value

    def observe_frame(self, frame: Any, *, stage: str | None = None) -> tuple[PageKey, ...]:
        """Describe every page in a pipeline frame and return their page keys.

        Called by the executor between stages so page characteristics are
        captured from whichever columns exist at that point: input size before
        rendering, raster dimensions after it, detection counts after page
        element detection, and so on.
        """
        if not self.capture_page_features:
            return page_keys_from_frame(frame)
        return _describe_frame(self, frame, stage=stage)

    # ------------------------------------------------------------------
    # Rollups
    # ------------------------------------------------------------------

    def self_durations(self) -> dict[int, float]:
        """Return each span's exclusive time (inclusive minus direct children).

        Exclusive time is what makes per-page attribution add up: a stage span
        that wraps instrumented per-page spans is only charged the portion of
        its runtime that those children did not already account for.
        """
        spans = self.spans
        child_totals: dict[int, float] = {}
        for span in spans:
            if span.parent_id is not None:
                child_totals[span.parent_id] = child_totals.get(span.parent_id, 0.0) + span.duration_s
        return {span.span_id: max(span.duration_s - child_totals.get(span.span_id, 0.0), 0.0) for span in spans}

    def page_attribution(self, *, by: str = "stage") -> dict[PageKey, dict[str, float]]:
        """Attribute exclusive span time to pages, split evenly within a span.

        Parameters
        ----------
        by
            ``"stage"`` groups a page's time by graph stage; ``"name"`` groups
            it by span name so nested operator spans stay distinct.

        Returns
        -------
        dict
            Mapping of page key to ``{label: seconds}``. Span time that covers
            no known page lands under the synthetic
            :data:`UNATTRIBUTED_PAGE` key.
        """
        if by not in ("stage", "name"):
            raise ValueError(f"by must be 'stage' or 'name', got {by!r}")
        spans = self.spans
        self_durations = self.self_durations()
        coverage = _page_coverage(spans)
        attributed: dict[PageKey, dict[str, float]] = {}
        for span in spans:
            seconds = self_durations.get(span.span_id, 0.0)
            if seconds <= 0.0:
                continue
            keys = coverage[span.span_id] or (UNATTRIBUTED_PAGE,)
            share = seconds / len(keys)
            label = span.name if by == "name" else (span.stage or span.name)
            for key in keys:
                bucket = attributed.setdefault(key, {})
                bucket[label] = bucket.get(label, 0.0) + share
        return attributed

    def stage_table(self) -> Any:
        """Per-stage rollup: call counts, inclusive and exclusive seconds, share.

        Returns
        -------
        pandas.DataFrame
            One row per graph stage, ordered by exclusive time descending.
        """
        import pandas as pd

        self_durations = self.self_durations()
        totals: dict[str, dict[str, Any]] = {}
        for span in self.spans:
            label = span.stage or span.name
            entry = totals.setdefault(
                label,
                {"stage": label, "spans": 0, "pages": set(), "total_s": 0.0, "self_s": 0.0, "errors": 0},
            )
            entry["spans"] += 1
            entry["pages"].update(span.page_keys)
            entry["self_s"] += self_durations.get(span.span_id, 0.0)
            if span.depth == 0:
                entry["total_s"] += span.duration_s
            if span.error:
                entry["errors"] += 1

        total_self = sum(entry["self_s"] for entry in totals.values()) or 1.0
        rows = []
        for entry in totals.values():
            pages = len(entry["pages"])
            rows.append(
                {
                    "stage": entry["stage"],
                    "spans": entry["spans"],
                    "pages": pages,
                    "total_s": entry["total_s"],
                    "self_s": entry["self_s"],
                    "self_pct": 100.0 * entry["self_s"] / total_self,
                    "mean_s_per_page": entry["self_s"] / pages if pages else 0.0,
                    "errors": entry["errors"],
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values("self_s", ascending=False, ignore_index=True)

    def span_table(self) -> Any:
        """Per-span-name rollup including nested operator spans.

        Returns
        -------
        pandas.DataFrame
            One row per span name with call count and duration distribution.
        """
        import pandas as pd

        self_durations = self.self_durations()
        grouped: dict[tuple[str | None, str], list[Span]] = {}
        for span in self.spans:
            grouped.setdefault((span.stage, span.name), []).append(span)

        rows = []
        for (stage, name), spans in grouped.items():
            durations = sorted(span.duration_s for span in spans)
            count = len(durations)
            rows.append(
                {
                    "stage": stage,
                    "name": name,
                    "depth": min(span.depth for span in spans),
                    "calls": count,
                    "total_s": sum(durations),
                    "self_s": sum(self_durations.get(span.span_id, 0.0) for span in spans),
                    "mean_s": sum(durations) / count,
                    "p50_s": durations[count // 2],
                    "p95_s": durations[min(int(count * 0.95), count - 1)],
                    "max_s": durations[-1],
                    "errors": sum(1 for span in spans if span.error),
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values("self_s", ascending=False, ignore_index=True)

    def page_table(self) -> Any:
        """Per-page rollup joining attributed time to page characteristics.

        Returns
        -------
        pandas.DataFrame
            One row per page with ``total_s``, one ``<stage>_s`` column per
            stage, and every captured page characteristic. Ordered slowest
            first, which is the view for finding page-level hotspots.
        """
        import pandas as pd

        attribution = self.page_attribution()
        features = self.page_features
        keys: dict[PageKey, None] = dict.fromkeys(features)
        for key in attribution:
            keys.setdefault(key, None)

        rows = []
        for key in keys:
            per_stage = attribution.get(key, {})
            row: dict[str, Any] = {
                "source": key[0],
                "page_number": key[1],
                "total_s": sum(per_stage.values()),
            }
            row.update({f"{stage}_s": seconds for stage, seconds in per_stage.items()})
            row.update({name: value for name, value in features.get(key, {}).items() if name not in row})
            rows.append(row)

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values("total_s", ascending=False, ignore_index=True)

    def page_stage_table(self, *, by: str = "stage") -> Any:
        """Long-form page attribution, one row per page and label.

        Parameters
        ----------
        by
            ``"stage"`` (default) attributes to graph stages; ``"name"``
            attributes to individual span names, which is the view for
            questions like "where did page 12 spend its time".

        Returns
        -------
        pandas.DataFrame
            Columns ``source``, ``page_number``, ``stage``, ``seconds``,
            ordered slowest first.
        """
        import pandas as pd

        rows = [
            {"source": key[0], "page_number": key[1], "stage": label, "seconds": seconds}
            for key, per_label in self.page_attribution(by=by).items()
            for label, seconds in per_label.items()
        ]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values("seconds", ascending=False, ignore_index=True)

    def hotspots(self, limit: int = 5) -> list[tuple[str, float, float]]:
        """Return the ``limit`` slowest span names as ``(name, self_s, pct)``."""
        frame = self.span_table()
        if getattr(frame, "empty", True):
            return []
        total = float(frame["self_s"].sum()) or 1.0
        return [
            (str(row["name"]), float(row["self_s"]), 100.0 * float(row["self_s"]) / total)
            for row in frame.head(limit).to_dict("records")
        ]

    def slowest_pages(self, limit: int = 5) -> list[tuple[PageKey, float]]:
        """Return the ``limit`` slowest pages as ``(page_key, seconds)``."""
        attribution = self.page_attribution()
        ranked = sorted(
            ((key, sum(per_stage.values())) for key, per_stage in attribution.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:limit]

    def feature_correlation(self, metric: str = "total_s", *, min_pages: int = 3) -> Any:
        """Correlate numeric page characteristics with per-page time.

        This is the direct answer to "what makes a page slow": a positive
        coefficient means pages scoring higher on that characteristic tend to
        take longer. Correlation is not causation and small page counts are
        noisy, so ``min_pages`` guards against reading meaning into a handful
        of rows.

        Parameters
        ----------
        metric
            Column from :meth:`page_table` to correlate against, typically
            ``"total_s"`` or a single ``"<stage>_s"`` column.
        min_pages
            Minimum number of pages required before any coefficient is
            computed.

        Returns
        -------
        pandas.Series
            Coefficients indexed by characteristic, ordered by absolute value
            descending. Empty when there is not enough data.
        """
        import pandas as pd

        frame = self.page_table()
        if getattr(frame, "empty", True) or metric not in frame.columns or len(frame.index) < min_pages:
            return pd.Series(dtype="float64")

        numeric = frame.select_dtypes(include="number")
        # ``page_number`` is an identifier, not a characteristic, and ``*_s``
        # columns are the timings being explained rather than explanations.
        candidates = [
            column
            for column in numeric.columns
            if column not in (metric, "page_number") and not column.endswith("_s") and numeric[column].nunique() > 1
        ]
        if not candidates:
            return pd.Series(dtype="float64")
        coefficients = numeric[candidates].corrwith(numeric[metric]).dropna()
        return coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the whole payload."""
        return {
            "run_mode": self.run_mode,
            "started_at": self.started_at,
            "total_s": round(self.total_s, 6),
            "stages": list(self.stage_names),
            "notes": list(self.notes),
            "spans": [span.to_dict() for span in self.spans],
            "pages": [
                {
                    "source": key[0],
                    "page_number": key[1],
                    "attributed_s": round(sum(per_stage.values()), 6),
                    "by_stage": {stage: round(seconds, 6) for stage, seconds in per_stage.items()},
                    "features": features,
                }
                for key, features, per_stage in self._page_records()
            ],
        }

    def _page_records(self) -> Iterator[tuple[PageKey, dict[str, Any], dict[str, float]]]:
        attribution = self.page_attribution()
        features = self.page_features
        keys: dict[PageKey, None] = dict.fromkeys(features)
        for key in attribution:
            keys.setdefault(key, None)
        for key in keys:
            yield key, features.get(key, {"source": key[0], "page_number": key[1]}), attribution.get(key, {})

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize :meth:`to_dict` to JSON, coercing unserializable values."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path, *, indent: int | None = 2) -> Path:
        """Write :meth:`to_json` to *path*, creating parent directories."""
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=indent), encoding="utf-8")
        return target

    def jsonl_records(self, **run_fields: Any) -> Iterator[dict[str, Any]]:
        """Yield one flat record per span-and-page pair.

        This is the long ("tidy") form of the trace: every record answers "this
        span charged this page this many seconds", with the page's
        characteristics repeated on the row. One table then answers both
        questions the trace exists for — group by ``name`` or ``stage`` for
        operation hotspots, by ``source`` and ``page_number`` for page
        hotspots, and correlate ``seconds`` against any characteristic column.

        Extra keyword arguments are copied onto every record, which is how the
        writer stamps run-level identifiers such as ``run_id``.
        """
        spans = self.spans
        self_durations = self.self_durations()
        coverage = _page_coverage(spans)
        features = self.page_features
        for span in spans:
            self_s = self_durations.get(span.span_id, 0.0)
            keys = coverage[span.span_id] or (UNATTRIBUTED_PAGE,)
            share = self_s / len(keys)
            for key in keys:
                record: dict[str, Any] = dict(run_fields)
                record.update(
                    run_mode=self.run_mode,
                    source=key[0],
                    page_number=key[1],
                    stage=span.stage or span.name,
                    name=span.name,
                    depth=span.depth,
                    span_id=span.span_id,
                    parent_span_id=span.parent_id,
                    seconds=round(share, 9),
                    span_self_s=round(self_s, 9),
                    span_total_s=round(span.duration_s, 9),
                    span_pages=len(keys),
                    error=span.error,
                    attributes=dict(span.attributes),
                )
                page_features = features.get(key)
                if page_features:
                    # ``source``/``page_number`` are already set from the key.
                    record.update(
                        {name: value for name, value in page_features.items() if name not in ("source", "page_number")}
                    )
                yield record

    def to_jsonl(self, **run_fields: Any) -> str:
        """Serialize :meth:`jsonl_records` as newline-delimited JSON."""
        return "".join(json.dumps(record, default=str) + "\n" for record in self.jsonl_records(**run_fields))

    def save_jsonl(self, path: str | Path, **run_fields: Any) -> Path:
        """Write :meth:`to_jsonl` to *path*, creating parent directories.

        The file loads directly with ``pandas.read_json(path, lines=True)``.
        """
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for record in self.jsonl_records(**run_fields):
                handle.write(json.dumps(record, default=str))
                handle.write("\n")
        return target

    def summary(self, *, limit: int = 5) -> str:
        """Render a short text report of stage and page hotspots."""
        lines = [
            f"Pipeline trace ({self.run_mode}): {self.total_s:.3f}s, "
            f"{len(self.spans)} spans, {len(self._page_order)} pages"
        ]
        for note in self.notes:
            lines.append(f"  note: {note}")
        lines.append("  slowest operations:")
        for name, seconds, pct in self.hotspots(limit=limit):
            lines.append(f"    {name}: {seconds:.3f}s ({pct:.1f}%)")
        lines.append("  slowest pages:")
        for (source, number), seconds in self.slowest_pages(limit=limit):
            lines.append(f"    {source} p{number}: {seconds:.3f}s")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PipelineTrace(run_mode={self.run_mode!r}, total_s={self.total_s:.3f}, "
            f"spans={len(self._spans)}, pages={len(self._page_order)})"
        )


# ----------------------------------------------------------------------
# Ambient access used by executors and operators
# ----------------------------------------------------------------------


def active_trace() -> PipelineTrace | None:
    """Return the trace collecting spans on this call stack, if any."""
    return _CURRENT_TRACE.get()


@contextmanager
def activate_trace(trace: PipelineTrace | None) -> Iterator[PipelineTrace | None]:
    """Make *trace* the ambient trace for the duration of the block.

    Passing ``None`` is supported and leaves tracing disabled, which lets
    executors run the same code path whether or not a caller asked for spans.
    """
    if trace is None:
        yield None
        return
    trace_token = _CURRENT_TRACE.set(trace)
    span_token = _CURRENT_SPAN.set(None)
    try:
        yield trace
    finally:
        _CURRENT_SPAN.reset(span_token)
        _CURRENT_TRACE.reset(trace_token)


def trace_span(name: str, **kwargs: Any) -> Any:
    """Open a span on the active trace, or a no-op context when tracing is off.

    Operators call this directly; it is the only tracing API they need::

        with trace_span("ocr.page", page=key) as span:
            span.set(crops=len(crops))
    """
    trace = _CURRENT_TRACE.get()
    if trace is None:
        return _NULL_SPAN_CONTEXT
    return trace.span(name, **kwargs)


def stage_span(stage: str, **kwargs: Any) -> Any:
    """Open a top-level span representing one graph stage."""
    return trace_span(stage, stage=stage, **kwargs)


def observe_frame(frame: Any, *, stage: str | None = None) -> tuple[PageKey, ...]:
    """Describe *frame* on the active trace and return its page keys."""
    trace = _CURRENT_TRACE.get()
    if trace is None:
        return ()
    return trace.observe_frame(frame, stage=stage)


# ----------------------------------------------------------------------
# Page characteristics
# ----------------------------------------------------------------------


def _length(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _describe_frame(trace: PipelineTrace, frame: Any, *, stage: str | None) -> tuple[PageKey, ...]:
    """Extract page characteristics from whichever known columns *frame* has."""
    if frame is None or not hasattr(frame, "columns"):
        return ()
    try:
        if len(frame.index) == 0:
            return ()
        available = set(frame.columns)
    except Exception:
        return ()

    wanted = (
        *_SOURCE_COLUMNS,
        "source_id",
        "page_number",
        "metadata",
        "bytes",
        "text",
        "page_image",
        "page_elements_v3_num_detections",
        "page_elements_v3_counts_by_label",
        "ocr_v1_num_detections",
        "table",
        "chart",
        "infographic",
        "images",
    )
    columns = [name for name in wanted if name in available]
    if not columns:
        return ()

    try:
        records = frame.loc[:, columns].to_dict("records")
    except Exception:
        return ()

    row_counts: dict[PageKey, int] = {}
    ordered: dict[PageKey, None] = {}
    for record in records:
        key = page_key_from_row(record)
        ordered[key] = None
        row_counts[key] = row_counts.get(key, 0) + 1
        trace.record_page_features(key, **_page_features_from_record(record))

    for key, count in row_counts.items():
        trace.record_page_features(key, rows=count, last_stage=stage)
    return tuple(ordered)


def _page_features_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map one row's known columns to page characteristics."""
    features: dict[str, Any] = {}

    payload = record.get("bytes")
    if isinstance(payload, (bytes, bytearray, memoryview)):
        features["source_bytes"] = len(payload)

    text = record.get("text")
    if isinstance(text, str):
        features["text_chars"] = len(text)
        features["text_words"] = len(text.split())

    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        for name in ("has_text", "needs_ocr_for_text", "dpi"):
            if name in metadata:
                features[name] = metadata[name]
        features["has_error"] = bool(metadata.get("error"))

    image = record.get("page_image")
    if isinstance(image, Mapping):
        shape = image.get("orig_shape_hw")
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            try:
                height, width = int(shape[0]), int(shape[1])
            except (TypeError, ValueError):
                height = width = 0
            if height and width:
                features["image_height"] = height
                features["image_width"] = width
                features["image_megapixels"] = round(height * width / 1e6, 4)
        encoded = image.get("image_b64")
        if isinstance(encoded, str):
            features["image_b64_chars"] = len(encoded)

    detections = record.get("page_elements_v3_num_detections")
    if isinstance(detections, (int, float)):
        features["num_detections"] = int(detections)

    counts = record.get("page_elements_v3_counts_by_label")
    if isinstance(counts, Mapping):
        for label, count in counts.items():
            try:
                features[f"detected_{label}"] = int(count)
            except (TypeError, ValueError):
                continue

    ocr_detections = record.get("ocr_v1_num_detections")
    if isinstance(ocr_detections, (int, float)):
        features["ocr_detections"] = int(ocr_detections)

    for column, name in (
        ("table", "num_tables"),
        ("chart", "num_charts"),
        ("infographic", "num_infographics"),
        ("images", "num_images"),
    ):
        length = _length(record.get(column))
        if length is not None:
            features[name] = length

    return features
