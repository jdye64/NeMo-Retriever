# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-page pipeline tracing.

Covers the span/collector primitives, survival of trace payloads through a
synthetic multi-operator graph including row fan-out, aggregation and
amortization arithmetic, and the pandas round-trip of the emitted artifact.
"""

from __future__ import annotations

import gzip
import json
from typing import Any

import pandas as pd
import pytest

from nemo_retriever.common.tracing import (
    TRACE_COLUMN,
    TRACE_SCHEMA_VERSION,
    aggregate_document_traces,
    decode_payload,
    default_document_id,
    load_traces,
    normalize_detail,
    page_summaries_dataframe,
    record_model,
    span,
    spans_dataframe,
    strip_trace_column,
    trace_filename,
    write_document_traces,
)
from nemo_retriever.common.tracing import runtime as trace_runtime
from nemo_retriever.common.tracing.collector import BatchTraceCollector, operator_trace
from nemo_retriever.common.tracing.load import TraceFileError
from nemo_retriever.common.tracing.spans import accumulate, model_span, page_scope
from nemo_retriever.operators.abstract_operator import AbstractOperator

DOC_A = "/data/a.pdf"
DOC_B = "/data/b.pdf"


@pytest.fixture(autouse=True)
def _isolated_trace_runtime() -> Any:
    """Keep the process-local detail override and model registry per-test."""
    trace_runtime.reset_detail()
    trace_runtime.clear_models()
    yield
    trace_runtime.reset_detail()
    trace_runtime.clear_models()


def _page_frame(path: str, pages: range | list[int], *, doc: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "path": doc or path,
                "source_id": f"{path}_{page}",
                "page_number": page,
                "content": f"page {page}",
            }
            for page in pages
        ]
    )


# ---------------------------------------------------------------------------
# Detail-level gating
# ---------------------------------------------------------------------------


def test_normalize_detail_accepts_levels_and_boolean_spellings() -> None:
    assert normalize_detail("full") == "full"
    assert normalize_detail("OPERATOR") == "operator"
    assert normalize_detail("false") == "off"
    assert normalize_detail("verbose") == "full"
    # Unset and unrecognized values both fall back to the documented default.
    assert normalize_detail(None) == "operator"
    assert normalize_detail("banana") == "operator"


def test_detail_reads_environment_when_no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(trace_runtime.TRACE_DETAIL_ENV_VAR, "full")
    assert trace_runtime.get_detail() == "full"
    assert trace_runtime.full_detail_enabled()

    trace_runtime.set_detail("operator")
    # An explicit process-local override wins over the environment.
    assert trace_runtime.get_detail() == "operator"
    assert not trace_runtime.full_detail_enabled()


def test_tracing_disabled_is_a_no_op() -> None:
    trace_runtime.set_detail("off")
    frame = _page_frame(DOC_A, [1, 2])

    class Op:
        pass

    with operator_trace(Op(), frame) as tracer:
        with span("should-not-record", category="cpu"):
            pass
        result = tracer.finish(frame)

    assert TRACE_COLUMN not in result.columns
    assert aggregate_document_traces(result) == []


def test_span_outside_an_operator_is_a_no_op() -> None:
    trace_runtime.set_detail("full")
    # No collector installed: the context manager must still be usable.
    with span("orphan", category="network") as handle:
        handle.set_attr("endpoint", "http://example/v1")
    assert handle.span_id == ""


def test_full_detail_spans_are_filtered_at_operator_detail() -> None:
    trace_runtime.set_detail("operator")
    frame = _page_frame(DOC_A, [1])

    class Op:
        pass

    with operator_trace(Op(), frame) as tracer:
        with span("hot-spot", category="network", detail="full"):
            pass
        result = tracer.finish(frame)

    spans, _ = decode_payload(result[TRACE_COLUMN].iloc[0])
    assert [item["name"] for item in spans] == ["Op"]


def test_tracing_never_synchronizes_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spans record entry and exit times without perturbing execution.

    Synchronizing to attribute device time would make the traced pipeline run
    differently from the untraced one, which defeats leaving tracing on. Use
    Nsight Systems for kernel-level attribution instead.
    """
    import sys
    import types

    calls: list[str] = []
    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(
        synchronize=lambda *args, **kwargs: calls.append("synchronize"),
        is_available=lambda: True,
    )
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    trace_runtime.set_detail("full")
    frame = _page_frame(DOC_A, [1])

    class Op:
        pass

    with operator_trace(Op(), frame) as tracer:
        with model_span("nemotron-ocr", category="gpu", span_name="gpu.nemotron-ocr"):
            pass
        result = tracer.finish(frame)

    assert calls == []
    spans, _ = decode_payload(result[TRACE_COLUMN].iloc[0])
    assert "gpu.nemotron-ocr" in [item["name"] for item in spans]


# ---------------------------------------------------------------------------
# Collector behavior
# ---------------------------------------------------------------------------


def test_operator_span_nests_child_spans_and_records_models() -> None:
    trace_runtime.set_detail("full")
    frame = _page_frame(DOC_A, [1, 2, 3, 4])

    class OCRActor:
        pass

    with operator_trace(OCRActor(), frame) as tracer:
        with model_span("ocr", name="nvidia/nemoretriever-ocr-v1", version="v2", backend="nim-grpc"):
            pass
        result = tracer.finish(frame)

    spans, models = decode_payload(result[TRACE_COLUMN].iloc[0])
    by_name = {item["name"]: item for item in spans}
    assert set(by_name) == {"OCRActor", "model.ocr"}
    assert by_name["model.ocr"]["parent_span_id"] == by_name["OCRActor"]["span_id"]
    assert by_name["OCRActor"]["batch_size"] == 4
    assert models == [
        {
            "model_key": "ocr",
            "name": "nvidia/nemoretriever-ocr-v1",
            "version": "v2",
            "backend": "nim-grpc",
            "endpoint": None,
        }
    ]


def test_page_scoped_span_is_charged_to_one_page() -> None:
    trace_runtime.set_detail("full")
    frame = _page_frame(DOC_A, [1, 2])

    class ExtractActor:
        pass

    with operator_trace(ExtractActor(), frame) as tracer:
        with page_scope(f"{DOC_A}_2"):
            with span("pdfium.render", category="cpu", detail="full"):
                pass
        result = tracer.finish(frame)

    page_one, _ = decode_payload(result[TRACE_COLUMN].iloc[0])
    page_two, _ = decode_payload(result[TRACE_COLUMN].iloc[1])
    assert "pdfium.render" not in {item["name"] for item in page_one}
    render = next(item for item in page_two if item["name"] == "pdfium.render")
    # A span charged to a single page covers exactly that page.
    assert render["batch_size"] == 1
    assert render["source_id"] == f"{DOC_A}_2"


def test_accumulated_spans_fold_repeated_calls() -> None:
    trace_runtime.set_detail("full")
    frame = _page_frame(DOC_A, [1])

    class EncodeActor:
        pass

    with operator_trace(EncodeActor(), frame) as tracer:
        for _ in range(25):
            with accumulate("image.encode", category="cpu"):
                pass
        result = tracer.finish(frame)

    spans, _ = decode_payload(result[TRACE_COLUMN].iloc[0])
    encodes = [item for item in spans if item["name"] == "image.encode"]
    assert len(encodes) == 1
    assert encodes[0]["attrs"]["calls"] == 25
    assert encodes[0]["attrs"]["rolled_up"] is True


def test_failing_operator_records_an_error_span() -> None:
    trace_runtime.set_detail("operator")
    frame = _page_frame(DOC_A, [1])

    class BoomActor:
        pass

    with pytest.raises(RuntimeError):
        with operator_trace(BoomActor(), frame) as tracer:
            raise RuntimeError("boom")

    # The collector recorded the failure; assert via a fresh collector run that
    # the parent stack was released so later spans are not orphaned to it.
    with operator_trace(BoomActor(), frame) as tracer:
        with span("after", category="cpu"):
            pass
        result = tracer.finish(frame)
    spans, _ = decode_payload(result[TRACE_COLUMN].iloc[0])
    after = next(item for item in spans if item["name"] == "after")
    parent = next(item for item in spans if item["name"] == "BoomActor")
    assert after["parent_span_id"] == parent["span_id"]


def test_empty_frame_still_declares_the_trace_column() -> None:
    trace_runtime.set_detail("operator")
    frame = _page_frame(DOC_A, [1])
    collector = BatchTraceCollector("Op", frame)
    empty = frame.iloc[0:0]

    result = collector.finish(empty)
    assert TRACE_COLUMN in result.columns


def test_pages_dropped_from_the_output_keep_their_time_at_batch_scope() -> None:
    trace_runtime.set_detail("full")
    frame = _page_frame(DOC_A, [1, 2])

    class FilterActor:
        pass

    with operator_trace(FilterActor(), frame) as tracer:
        with page_scope(f"{DOC_A}_2"):
            with span("dropped.work", category="cpu", detail="full"):
                pass
        result = tracer.finish(frame.iloc[[0]].copy())

    spans, _ = decode_payload(result[TRACE_COLUMN].iloc[0])
    # Page 2 vanished, but its cost must not vanish with it.
    assert "dropped.work" in {item["name"] for item in spans}


# ---------------------------------------------------------------------------
# Survival through a synthetic multi-operator graph
# ---------------------------------------------------------------------------


class _StubOperator(AbstractOperator):
    """Operator base with the pass-through hooks the real stages inherit."""

    def preprocess(self, data: Any, **_: Any) -> Any:
        return data

    def postprocess(self, data: Any, **_: Any) -> Any:
        return data


class _SplitOperator(_StubOperator):
    """Stands in for PDFSplitActor: one document row becomes N page rows."""

    def __init__(self, pages: int) -> None:
        self._pages = pages

    def process(self, data: pd.DataFrame, **_: Any) -> pd.DataFrame:
        rows = []
        for _, row in data.iterrows():
            for page in range(1, self._pages + 1):
                rows.append(
                    {
                        "path": row["path"],
                        "source_id": f"{row['path']}_{page}",
                        "page_number": page,
                        "content": f"page {page}",
                    }
                )
        return pd.DataFrame(rows)


class _NetworkOperator(_StubOperator):
    """Stands in for a NIM-backed stage: one network span per batch."""

    def process(self, data: pd.DataFrame, **_: Any) -> pd.DataFrame:
        record_model("ocr", name="nvidia/nemoretriever-ocr-v1", version="v2", backend="nim-http")
        with span("nim.infer", category="network", model_key="ocr", detail="full"):
            pass
        return data.copy()


class _ExplodeOperator(_StubOperator):
    """Stands in for ExplodeContentToRows: each page row fans out to elements."""

    def __init__(self, elements: int) -> None:
        self._elements = elements

    def process(self, data: pd.DataFrame, **_: Any) -> pd.DataFrame:
        rows = []
        for _, row in data.iterrows():
            for element in range(self._elements):
                new_row = row.to_dict()
                new_row["element_index"] = element
                rows.append(new_row)
        return pd.DataFrame(rows)


def _run_synthetic_graph(*, pages: int = 3, elements: int = 4) -> pd.DataFrame:
    frame = pd.DataFrame([{"path": DOC_A, "content": "whole document"}])
    for operator in (_SplitOperator(pages), _NetworkOperator(), _ExplodeOperator(elements)):
        frame = operator.run(frame)
    return frame


def test_trace_survives_fan_out_and_dedupes_by_span_id() -> None:
    trace_runtime.set_detail("full")
    frame = _run_synthetic_graph(pages=3, elements=4)

    assert len(frame) == 12
    assert TRACE_COLUMN in frame.columns

    # Fan-out deep-copies the payload, so sibling element rows share spans.
    duplicated_ids = [
        span_id
        for raw in frame[TRACE_COLUMN]
        for span_id in (item["span_id"] for item in decode_payload(raw)[0])
    ]
    assert len(duplicated_ids) > len(set(duplicated_ids))

    (trace,) = aggregate_document_traces(frame, run_mode="inprocess")
    emitted = [record["span_id"] for record in trace["spans"]]
    # One record per (span, page) pair; no page repeats a span.
    per_page: dict[int, set[str]] = {}
    for record in trace["spans"]:
        page_spans = per_page.setdefault(record["page_number"], set())
        assert record["span_id"] not in page_spans
        page_spans.add(record["span_id"])
    assert set(emitted) == set(duplicated_ids)
    assert trace["document"]["page_count"] == 3


def test_result_frames_can_be_stripped_of_the_trace_column() -> None:
    trace_runtime.set_detail("operator")
    frame = _run_synthetic_graph(pages=2, elements=1)
    assert TRACE_COLUMN in frame.columns
    assert TRACE_COLUMN not in strip_trace_column(frame).columns


def test_document_level_spans_are_inherited_by_split_pages() -> None:
    trace_runtime.set_detail("full")

    class ConvertOperator(_StubOperator):
        def process(self, data: pd.DataFrame, **_: Any) -> pd.DataFrame:
            with span("libreoffice.convert", category="io", detail="full"):
                pass
            return data.copy()

    frame = pd.DataFrame([{"path": DOC_A, "content": "whole document"}])
    frame = ConvertOperator().run(frame)
    frame = _SplitOperator(3).run(frame)

    (trace,) = aggregate_document_traces(frame)
    converts = [record for record in trace["spans"] if record["name"] == "libreoffice.convert"]
    # The conversion predates page identity, so every page inherits it.
    assert {record["page_number"] for record in converts} == {1, 2, 3}
    assert len({record["span_id"] for record in converts}) == 1


# ---------------------------------------------------------------------------
# Aggregation and amortization
# ---------------------------------------------------------------------------


def _synthetic_trace_frame(spans_by_row: list[tuple[str, int, list[dict[str, Any]]]]) -> pd.DataFrame:
    from nemo_retriever.common.tracing.collector import encode_payload

    return pd.DataFrame(
        [
            {
                "path": path,
                "source_id": f"{path}_{page}",
                "page_number": page,
                TRACE_COLUMN: encode_payload(spans, []),
            }
            for path, page, spans in spans_by_row
        ]
    )


def _operator_span(span_id: str, name: str, duration_ms: float, *, batch_size: int) -> dict[str, Any]:
    return {
        "span_id": span_id,
        "name": name,
        "operator": name,
        "category": "operator",
        "start_ms": 1_000.0,
        "end_ms": 1_000.0 + duration_ms,
        "duration_ms": duration_ms,
        "batch_size": batch_size,
    }


def test_amortization_divides_batch_cost_across_the_pages_it_covered() -> None:
    shared = _operator_span("s1", "ExtractActor", 300.0, batch_size=3)
    frame = _synthetic_trace_frame([(DOC_A, page, [shared]) for page in (1, 2, 3)])

    (trace,) = aggregate_document_traces(frame)

    for record in trace["spans"]:
        # Each page reports the exact measured duration plus its own share.
        assert record["duration_ms"] == 300.0
        assert record["amortized_ms"] == 100.0
        assert record["page_fanout"] == 3
    assert trace["document_summary"]["total_ms"] == 300.0


def test_page_summaries_sum_to_the_document_total() -> None:
    shared = _operator_span("s1", "ExtractActor", 300.0, batch_size=3)
    per_page = [_operator_span(f"p{page}", "EmbedActor", 30.0, batch_size=1) for page in (1, 2, 3)]
    frame = _synthetic_trace_frame(
        [(DOC_A, page, [shared, per_page[page - 1]]) for page in (1, 2, 3)]
    )

    (trace,) = aggregate_document_traces(frame)

    page_total = sum(summary["total_ms"] for summary in trace["page_summaries"])
    assert page_total == pytest.approx(trace["document_summary"]["total_ms"])
    assert trace["document_summary"]["total_ms"] == pytest.approx(390.0)
    by_operator = {entry["operator"]: entry for entry in trace["document_summary"]["by_operator"]}
    assert by_operator["ExtractActor"]["total_ms"] == pytest.approx(300.0)
    assert by_operator["EmbedActor"]["total_ms"] == pytest.approx(90.0)


def test_batches_mixing_documents_do_not_over_attribute_time() -> None:
    # One operator span covered two pages of A and one page of B. Charging each
    # document the whole batch duration would inflate both.
    shared = _operator_span("s1", "ExtractActor", 300.0, batch_size=3)
    frame = _synthetic_trace_frame(
        [(DOC_A, 1, [shared]), (DOC_A, 2, [shared]), (DOC_B, 1, [shared])]
    )

    traces = {trace["document"]["source_path"]: trace for trace in aggregate_document_traces(frame)}

    assert traces[DOC_A]["document_summary"]["total_ms"] == pytest.approx(200.0)
    assert traces[DOC_B]["document_summary"]["total_ms"] == pytest.approx(100.0)
    combined = sum(trace["document_summary"]["total_ms"] for trace in traces.values())
    assert combined == pytest.approx(300.0)


def test_self_time_subtracts_child_spans() -> None:
    parent = _operator_span("s1", "ExtractActor", 100.0, batch_size=1)
    child = {
        "span_id": "s2",
        "parent_span_id": "s1",
        "name": "nim.infer",
        "operator": "ExtractActor",
        "category": "network",
        "start_ms": 1_010.0,
        "end_ms": 1_080.0,
        "duration_ms": 70.0,
        "batch_size": 1,
    }
    frame = _synthetic_trace_frame([(DOC_A, 1, [parent, child])])

    (trace,) = aggregate_document_traces(frame)

    by_operator = {entry["operator"]: entry for entry in trace["document_summary"]["by_operator"]}
    assert by_operator["ExtractActor"]["total_ms"] == pytest.approx(100.0)
    assert by_operator["ExtractActor"]["self_ms"] == pytest.approx(30.0)
    assert trace["document_summary"]["by_category"]["network_ms"] == pytest.approx(70.0)


def test_aggregate_returns_nothing_without_a_trace_column() -> None:
    assert aggregate_document_traces(pd.DataFrame([{"path": DOC_A}])) == []
    assert aggregate_document_traces(pd.DataFrame()) == []
    assert aggregate_document_traces("not a frame") == []


def test_run_and_version_metadata_is_attached() -> None:
    frame = _synthetic_trace_frame([(DOC_A, 1, [_operator_span("s1", "ExtractActor", 10.0, batch_size=1)])])

    (trace,) = aggregate_document_traces(
        frame,
        run_mode="batch",
        run_id="run-123",
        pipeline_operators=["ExtractActor"],
        params={"page_trace_detail": "full"},
    )

    assert trace["schema_version"] == TRACE_SCHEMA_VERSION
    assert trace["run"]["run_id"] == "run-123"
    assert trace["run"]["run_mode"] == "batch"
    assert trace["run"]["completed_at"]
    assert trace["pipeline"] == {"operators": ["ExtractActor"], "params": {"page_trace_detail": "full"}}
    assert trace["nemo_retriever"]["version"]
    assert trace["document"]["source_type"] == "pdf"


def test_document_id_is_filesystem_safe_and_path_specific() -> None:
    first = default_document_id("/data/reports/q1 report.pdf")
    second = default_document_id("/archive/reports/q1 report.pdf")
    assert first != second
    assert "/" not in first and " " not in first
    assert first.startswith("q1_report.pdf-")


# ---------------------------------------------------------------------------
# Artifact round-trip and pandas compatibility
# ---------------------------------------------------------------------------


def test_write_and_load_round_trip(tmp_path: Any) -> None:
    trace_runtime.set_detail("full")
    frame = _run_synthetic_graph(pages=2, elements=2)
    traces = aggregate_document_traces(frame)

    paths = write_document_traces(traces, tmp_path)
    assert [path.name for path in paths] == [trace_filename(traces[0]["document"]["document_id"])]

    # Directory, explicit file, and glob all resolve to the same artifact.
    assert load_traces(tmp_path) == traces
    assert load_traces(paths[0]) == traces
    assert load_traces(str(tmp_path / "*.trace.json")) == traces


def test_gzip_round_trip(tmp_path: Any) -> None:
    trace_runtime.set_detail("operator")
    traces = aggregate_document_traces(_run_synthetic_graph(pages=1, elements=1))

    (path,) = write_document_traces(traces, tmp_path, compression="gzip")
    assert path.name.endswith(".trace.json.gz")
    with gzip.open(path, "rb") as handle:
        assert json.loads(handle.read().decode("utf-8")) == traces[0]
    assert load_traces(tmp_path) == traces


def test_loading_a_non_trace_file_reports_a_useful_error(tmp_path: Any) -> None:
    stray = tmp_path / "notes.trace.json"
    stray.write_text(json.dumps({"hello": "world"}), encoding="utf-8")

    with pytest.raises(TraceFileError, match="not a NeMo Retriever page trace artifact"):
        load_traces(stray)

    with pytest.raises(TraceFileError, match="No trace files matched"):
        load_traces(str(tmp_path / "missing-*.trace.json"))


def test_spans_dataframe_is_flat_and_summable() -> None:
    trace_runtime.set_detail("full")
    traces = aggregate_document_traces(_run_synthetic_graph(pages=3, elements=2))

    spans = spans_dataframe(traces)
    assert not spans.empty
    for column in ("page_number", "operator", "category", "duration_ms", "amortized_ms", "library_version"):
        assert column in spans.columns
    # json_normalize must not leave nested objects behind in the numeric columns.
    assert spans["amortized_ms"].dtype.kind == "f"

    operator_spans = spans[spans.category == "operator"]
    amortized_total = operator_spans.amortized_ms.sum()
    document_total = traces[0]["document_summary"]["total_ms"]
    # Each per-page record is rounded to microseconds before it is written, so
    # the summed value can drift from the document total by the rounding error
    # of one record per page.
    assert amortized_total == pytest.approx(document_total, abs=0.001 * len(operator_spans))


def test_page_summaries_dataframe_has_one_row_per_page() -> None:
    trace_runtime.set_detail("operator")
    traces = aggregate_document_traces(_run_synthetic_graph(pages=4, elements=1))

    pages = page_summaries_dataframe(traces)
    assert sorted(pages.page_number) == [1, 2, 3, 4]
    assert pages.document_id.nunique() == 1


def test_empty_dataframes_have_stable_columns() -> None:
    assert list(spans_dataframe([]).columns)
    assert list(page_summaries_dataframe([]).columns)
