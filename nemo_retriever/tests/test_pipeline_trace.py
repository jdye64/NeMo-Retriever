# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for per-page span collection in the inprocess run mode."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from nemo_retriever.common.params import ExtractParams, IngestExecuteParams
from nemo_retriever.common.tracing import (
    DEFAULT_TRACE_DIR,
    PipelineTrace,
    activate_trace,
    active_trace,
    new_run_id,
    observe_frame,
    page_index,
    page_key,
    page_key_from_row,
    page_keys_from_frame,
    save_ingest_trace,
    stage_span,
    trace_file_path,
    trace_span,
)
from nemo_retriever.common.tracing.pipeline_trace import UNATTRIBUTED_PAGE, UNKNOWN_SOURCE
from nemo_retriever.graph.executor import InprocessExecutor
from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator


class _PageProducingOperator(AbstractOperator, CPUOperator):
    """Turn one document row into ``pages`` page rows, like the PDF splitter."""

    def __init__(self, pages: int = 2) -> None:
        super().__init__(pages=pages)
        self.pages = pages

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        rows = [
            {"path": row["path"], "page_number": number + 1, "text": "x" * (number + 1)}
            for _, row in data.iterrows()
            for number in range(self.pages)
        ]
        return pd.DataFrame(rows)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class _PerPageSpanOperator(AbstractOperator, CPUOperator):
    """Emit one nested span per page, as instrumented operators do."""

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        pages = page_index(data)
        for position in range(len(data.index)):
            with trace_span("fake.page", page=pages(position)) as span:
                span.set(work=position)
        return data

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


def _linear_graph(*operators: AbstractOperator) -> Graph:
    graph = Graph()
    nodes = [
        Node(
            operator,
            name=type(operator).__name__,
            operator_class=type(operator),
            operator_kwargs=operator.get_constructor_kwargs(),
        )
        for operator in operators
    ]
    for left, right in zip(nodes, nodes[1:]):
        left >> right
    graph.add_root(nodes[0])
    return graph


def _run_graph_ingest_with_result(ingestor: GraphIngestor, result: Any, monkeypatch, **ingest_kwargs: Any) -> Any:
    """Run ``ingest`` with graph execution stubbed out to *result*."""
    if not ingestor._documents:
        ingestor.files(["document.pdf"])
    monkeypatch.setattr(ingestor, "_plan_default_extraction_branches", lambda: None)
    monkeypatch.setattr(
        ingestor,
        "_resolve_effective_extraction_inputs",
        lambda: SimpleNamespace(extraction_mode="pdf"),
    )
    monkeypatch.setattr(
        ingestor,
        "_execute_single_graph",
        lambda effective_extraction, *, post_extract_order: result,
    )
    return ingestor.ingest(**ingest_kwargs)


# ---------------------------------------------------------------------------
# Page keys
# ---------------------------------------------------------------------------


def test_page_key_normalizes_missing_source_and_page() -> None:
    assert page_key(None) == (UNKNOWN_SOURCE, 0)
    assert page_key("  ", "nope") == (UNKNOWN_SOURCE, 0)
    assert page_key("a.pdf", "3") == ("a.pdf", 3)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"path": "a.pdf", "page_number": 2}, ("a.pdf", 2)),
        ({"source_path": "b.pdf", "page_number": 1}, ("b.pdf", 1)),
        ({"metadata": {"source_path": "c.pdf"}, "page_number": 7}, ("c.pdf", 7)),
        ({"source_id": "d.pdf_4"}, ("d.pdf_4", 0)),
        ({}, (UNKNOWN_SOURCE, 0)),
    ],
)
def test_page_key_from_row_falls_back_across_source_columns(row: dict, expected: tuple) -> None:
    assert page_key_from_row(row) == expected


def test_page_key_from_row_reads_itertuples_rows() -> None:
    frame = pd.DataFrame([{"path": "a.pdf", "page_number": 5}])
    row = next(frame.itertuples(index=False))
    assert page_key_from_row(row) == ("a.pdf", 5)


def test_page_keys_from_frame_deduplicates_in_order() -> None:
    frame = pd.DataFrame(
        [
            {"path": "a.pdf", "page_number": 1},
            {"path": "a.pdf", "page_number": 1},
            {"path": "a.pdf", "page_number": 2},
        ]
    )
    assert page_keys_from_frame(frame) == (("a.pdf", 1), ("a.pdf", 2))


def test_page_keys_from_frame_tolerates_empty_and_unknown_frames() -> None:
    assert page_keys_from_frame(pd.DataFrame()) == ()
    assert page_keys_from_frame(pd.DataFrame({"unrelated": [1]})) == ()
    assert page_keys_from_frame(None) == ()


# ---------------------------------------------------------------------------
# Inert behavior when no trace is active
# ---------------------------------------------------------------------------


def test_trace_span_is_inert_without_an_active_trace() -> None:
    assert active_trace() is None
    with trace_span("nothing", page=("a.pdf", 1)) as span:
        span.set(anything=1).add_pages([("a.pdf", 2)])
    assert observe_frame(pd.DataFrame([{"path": "a.pdf", "page_number": 1}])) == ()


def test_page_index_is_empty_without_an_active_trace() -> None:
    frame = pd.DataFrame([{"path": "a.pdf", "page_number": 1}])
    index = page_index(frame)
    assert index.all() == ()
    assert index(0) is None
    assert index.many([0]) == ()


def test_activate_trace_accepts_none_and_restores_previous_state() -> None:
    trace = PipelineTrace()
    with activate_trace(trace):
        with activate_trace(None):
            assert active_trace() is trace
        assert active_trace() is trace
    assert active_trace() is None


# ---------------------------------------------------------------------------
# Span recording and rollups
# ---------------------------------------------------------------------------


def test_spans_nest_and_self_time_excludes_children() -> None:
    trace = PipelineTrace()
    with activate_trace(trace):
        with stage_span("Stage") as outer:
            outer.add_pages([("a.pdf", 1)])
            with trace_span("child"):
                pass

    outer_span, child_span = trace.spans
    assert child_span.parent_id == outer_span.span_id
    assert child_span.depth == 1
    # The child inherits its parent's stage without being told about it.
    assert child_span.stage == "Stage"

    self_durations = trace.self_durations()
    assert self_durations[outer_span.span_id] == pytest.approx(outer_span.duration_s - child_span.duration_s, abs=1e-9)
    assert self_durations[child_span.span_id] == pytest.approx(child_span.duration_s, abs=1e-9)


def test_span_records_exception_message_and_reraises() -> None:
    trace = PipelineTrace()
    with activate_trace(trace):
        with pytest.raises(RuntimeError, match="boom"):
            with trace_span("failing"):
                raise RuntimeError("boom")

    (span,) = trace.spans
    assert span.error == "RuntimeError: boom"
    assert span.end_s is not None


def test_batched_span_time_splits_evenly_across_its_pages() -> None:
    trace = PipelineTrace()
    with activate_trace(trace):
        with trace_span("batch", pages=[("a.pdf", 1), ("a.pdf", 2), ("a.pdf", 3)], stage="Detect"):
            pass

    (span,) = trace.spans
    attribution = trace.page_attribution()
    assert set(attribution) == {("a.pdf", 1), ("a.pdf", 2), ("a.pdf", 3)}
    for per_stage in attribution.values():
        assert per_stage["Detect"] == pytest.approx(span.duration_s / 3, abs=1e-9)


def test_span_without_pages_is_reported_as_unattributed() -> None:
    trace = PipelineTrace()
    with activate_trace(trace):
        with trace_span("orphan", stage="Stage"):
            pass

    assert UNATTRIBUTED_PAGE in trace.page_attribution()


def test_page_attribution_by_name_keeps_nested_spans_distinct() -> None:
    trace = PipelineTrace()
    with activate_trace(trace):
        with stage_span("Stage") as outer:
            outer.add_pages([("a.pdf", 1)])
            with trace_span("inner", page=("a.pdf", 1)):
                pass

    by_stage = trace.page_attribution()[("a.pdf", 1)]
    by_name = trace.page_attribution(by="name")[("a.pdf", 1)]
    assert set(by_stage) == {"Stage"}
    assert set(by_name) == {"Stage", "inner"}
    assert sum(by_name.values()) == pytest.approx(sum(by_stage.values()), abs=1e-9)


def test_nested_spans_inherit_their_ancestor_page() -> None:
    """A child span that names no page is charged to its ancestor's page."""
    trace = PipelineTrace()
    with activate_trace(trace):
        with stage_span("Extract") as stage:
            stage.add_pages([("a.pdf", 1), ("a.pdf", 2)])
            with trace_span("extract.page", page=("a.pdf", 1)):
                # Neither child names a page; both belong to page 1.
                with trace_span("extract.render"):
                    pass
                with trace_span("extract.text"):
                    pass

    attribution = trace.page_attribution(by="name")
    assert UNATTRIBUTED_PAGE not in attribution
    assert {"extract.render", "extract.text"} <= set(attribution[("a.pdf", 1)])
    assert "extract.render" not in attribution.get(("a.pdf", 2), {})


def test_page_attribution_falls_back_to_the_stage_page_set() -> None:
    """With no per-page span, a stage's children spread over its pages."""
    trace = PipelineTrace()
    with activate_trace(trace):
        with stage_span("Embed") as stage:
            stage.add_pages([("a.pdf", 1), ("a.pdf", 2)])
            with trace_span("embed.local_inference"):
                pass

    attribution = trace.page_attribution(by="name")
    first = attribution[("a.pdf", 1)]["embed.local_inference"]
    second = attribution[("a.pdf", 2)]["embed.local_inference"]
    assert first == pytest.approx(second, abs=1e-9)


def test_page_attribution_survives_a_broken_parent_chain() -> None:
    trace = PipelineTrace()
    with activate_trace(trace):
        with trace_span("orphaned", stage="Stage"):
            pass
    # Simulate a parent reference the trace never recorded.
    trace.spans[0].parent_id = 999
    assert UNATTRIBUTED_PAGE in trace.page_attribution()


def test_page_attribution_rejects_unknown_grouping() -> None:
    with pytest.raises(ValueError, match="by must be 'stage' or 'name'"):
        PipelineTrace().page_attribution(by="page")


def test_add_pages_preserves_order_without_duplicates() -> None:
    trace = PipelineTrace()
    with activate_trace(trace):
        with trace_span("span", page=("a.pdf", 1)) as span:
            span.add_pages([("a.pdf", 2), ("a.pdf", 1)])

    (recorded,) = trace.spans
    assert recorded.page_keys == (("a.pdf", 1), ("a.pdf", 2))
    assert recorded.page_count == 2


@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        # A bare key, as ``pdf_split`` passes for a document-level fallback.
        (page_key("a.pdf"), (("a.pdf", 0),)),
        # A generator, as ``pdf_split`` passes for the pages it produced.
        ((page_key("a.pdf", n) for n in (1, 2)), (("a.pdf", 1), ("a.pdf", 2))),
        ("a.pdf", (("a.pdf", 0),)),
        ({"path": "a.pdf", "page_number": 3}, (("a.pdf", 3),)),
        (None, ()),
    ],
)
def test_add_pages_accepts_loose_page_references(pages: Any, expected: tuple[Any, ...]) -> None:
    trace = PipelineTrace()
    with activate_trace(trace):
        with trace_span("span") as span:
            span.add_pages(pages)

    assert trace.spans[0].page_keys == expected


# ---------------------------------------------------------------------------
# Page characteristics
# ---------------------------------------------------------------------------


def test_observe_frame_captures_page_characteristics() -> None:
    trace = PipelineTrace()
    frame = pd.DataFrame(
        [
            {
                "path": "a.pdf",
                "page_number": 1,
                "bytes": b"1234",
                "text": "two words",
                "metadata": {"has_text": True, "needs_ocr_for_text": False, "dpi": 200, "error": None},
                "page_image": {"image_b64": "abcd", "orig_shape_hw": (1000, 500)},
                "page_elements_v3_num_detections": 3,
                "page_elements_v3_counts_by_label": {"table": 1, "text": 2},
                "ocr_v1_num_detections": 5,
                "table": [{}],
                "chart": [],
                "images": [{}, {}],
            }
        ]
    )
    with activate_trace(trace):
        assert observe_frame(frame, stage="Extract") == (("a.pdf", 1),)

    features = trace.page_features[("a.pdf", 1)]
    assert features["source_bytes"] == 4
    assert features["text_chars"] == 9
    assert features["text_words"] == 2
    assert features["has_text"] is True
    assert features["needs_ocr_for_text"] is False
    assert features["dpi"] == 200
    assert features["has_error"] is False
    assert features["image_height"] == 1000
    assert features["image_width"] == 500
    assert features["image_megapixels"] == pytest.approx(0.5)
    assert features["image_b64_chars"] == 4
    assert features["num_detections"] == 3
    assert features["detected_table"] == 1
    assert features["detected_text"] == 2
    assert features["ocr_detections"] == 5
    assert features["num_tables"] == 1
    assert features["num_charts"] == 0
    assert features["num_images"] == 2
    assert features["rows"] == 1
    assert features["last_stage"] == "Extract"


def test_page_features_keep_the_latest_non_null_value() -> None:
    trace = PipelineTrace()
    trace.record_page_features(("a.pdf", 1), text_chars=10, dpi=200)
    trace.record_page_features(("a.pdf", 1), text_chars=None, dpi=300)
    features = trace.page_features[("a.pdf", 1)]
    assert features["text_chars"] == 10
    assert features["dpi"] == 300


def test_observe_frame_counts_rows_per_page_after_row_expansion() -> None:
    trace = PipelineTrace()
    frame = pd.DataFrame(
        [
            {"path": "a.pdf", "page_number": 1, "text": "one"},
            {"path": "a.pdf", "page_number": 1, "text": "two"},
        ]
    )
    with activate_trace(trace):
        observe_frame(frame, stage="Embed")
    assert trace.page_features[("a.pdf", 1)]["rows"] == 2


def test_observe_frame_is_skipped_when_feature_capture_is_disabled() -> None:
    trace = PipelineTrace(capture_page_features=False)
    frame = pd.DataFrame([{"path": "a.pdf", "page_number": 1, "text": "hello"}])
    with activate_trace(trace):
        assert observe_frame(frame, stage="Extract") == (("a.pdf", 1),)
    assert trace.page_features == {}


# ---------------------------------------------------------------------------
# Tables and reporting
# ---------------------------------------------------------------------------


def _trace_with_two_pages() -> PipelineTrace:
    trace = PipelineTrace()
    with activate_trace(trace):
        with stage_span("Detect") as stage:
            stage.add_pages([("a.pdf", 1), ("a.pdf", 2)])
            with trace_span("detect.page", page=("a.pdf", 1)):
                pass
        observe_frame(
            pd.DataFrame(
                [
                    {"path": "a.pdf", "page_number": 1, "page_elements_v3_num_detections": 12},
                    {"path": "a.pdf", "page_number": 2, "page_elements_v3_num_detections": 1},
                ]
            ),
            stage="Detect",
        )
    return trace.finish()


def test_stage_table_reports_one_row_per_stage() -> None:
    table = _trace_with_two_pages().stage_table()
    assert list(table["stage"]) == ["Detect"]
    row = table.iloc[0]
    assert row["spans"] == 2
    assert row["pages"] == 2
    assert row["self_pct"] == pytest.approx(100.0)
    assert row["errors"] == 0


def test_span_table_reports_one_row_per_span_name() -> None:
    table = _trace_with_two_pages().span_table()
    assert set(table["name"]) == {"Detect", "detect.page"}
    assert set(table["stage"]) == {"Detect"}
    assert all(table["calls"] == 1)


def test_page_table_joins_time_to_characteristics_slowest_first() -> None:
    table = _trace_with_two_pages().page_table()
    assert list(table.columns[:3]) == ["source", "page_number", "total_s"]
    assert "Detect_s" in table.columns
    assert "num_detections" in table.columns
    assert list(table["total_s"]) == sorted(table["total_s"], reverse=True)
    assert set(table["page_number"]) == {1, 2}


def test_page_stage_table_is_long_form() -> None:
    table = _trace_with_two_pages().page_stage_table()
    assert list(table.columns) == ["source", "page_number", "stage", "seconds"]
    by_name = _trace_with_two_pages().page_stage_table(by="name")
    assert "detect.page" in set(by_name["stage"])


def test_tables_are_empty_for_an_empty_trace() -> None:
    trace = PipelineTrace()
    assert trace.stage_table().empty
    assert trace.span_table().empty
    assert trace.page_table().empty
    assert trace.page_stage_table().empty
    assert trace.hotspots() == []
    assert trace.slowest_pages() == []
    assert trace.feature_correlation().empty


def test_hotspots_and_slowest_pages_are_ranked() -> None:
    trace = _trace_with_two_pages()
    hotspots = trace.hotspots(limit=1)
    assert len(hotspots) == 1
    name, seconds, percent = hotspots[0]
    assert name in {"Detect", "detect.page"}
    assert seconds >= 0.0
    assert 0.0 <= percent <= 100.0

    pages = trace.slowest_pages(limit=2)
    assert [key for key, _ in pages] == [("a.pdf", 1), ("a.pdf", 2)]
    assert pages[0][1] >= pages[1][1]


def test_feature_correlation_ranks_characteristics_by_absolute_strength() -> None:
    trace = PipelineTrace()
    for number in range(1, 5):
        trace.record_page_features(("a.pdf", number), num_detections=number, text_chars=100)
        with activate_trace(trace):
            with trace_span("work", page=("a.pdf", number), stage="Stage"):
                # Later pages do proportionally more work.
                for _ in range(number * 2000):
                    pass

    coefficients = trace.feature_correlation()
    assert "num_detections" in coefficients.index
    # A constant characteristic carries no signal and is dropped.
    assert "text_chars" not in coefficients.index
    assert "page_number" not in coefficients.index
    assert coefficients["num_detections"] > 0.5


def test_feature_correlation_requires_enough_pages() -> None:
    trace = PipelineTrace()
    trace.record_page_features(("a.pdf", 1), num_detections=1)
    assert trace.feature_correlation(min_pages=3).empty


def test_feature_correlation_requires_a_known_metric() -> None:
    assert _trace_with_two_pages().feature_correlation(metric="missing_s", min_pages=1).empty


def test_summary_lists_notes_stages_and_pages() -> None:
    trace = _trace_with_two_pages()
    trace.note("partial run")
    summary = trace.summary(limit=2)
    assert "Pipeline trace (inprocess)" in summary
    assert "note: partial run" in summary
    assert "Detect" in summary
    assert "a.pdf p1" in summary


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_to_dict_round_trips_through_json() -> None:
    trace = _trace_with_two_pages()
    payload = json.loads(trace.to_json())
    assert payload["run_mode"] == "inprocess"
    assert payload["stages"] == ["Detect"]
    assert {span["name"] for span in payload["spans"]} == {"Detect", "detect.page"}
    page = next(entry for entry in payload["pages"] if entry["page_number"] == 1)
    assert page["source"] == "a.pdf"
    assert page["by_stage"]["Detect"] >= 0.0
    assert page["features"]["num_detections"] == 12


def test_save_writes_json_and_creates_parent_directories(tmp_path) -> None:
    target = _trace_with_two_pages().save(tmp_path / "nested" / "trace.json")
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8"))["run_mode"] == "inprocess"


def test_to_dict_coerces_unserializable_attribute_values() -> None:
    trace = PipelineTrace()
    with activate_trace(trace):
        with trace_span("span") as span:
            span.set(model=object())
    assert isinstance(json.loads(trace.to_json())["spans"][0]["attributes"]["model"], str)


# ---------------------------------------------------------------------------
# JSONL export
# ---------------------------------------------------------------------------


def test_jsonl_records_emit_one_row_per_span_and_page() -> None:
    records = list(_trace_with_two_pages().jsonl_records())

    # The stage span covers two pages and the nested span covers one.
    assert len(records) == 3
    stage_rows = [row for row in records if row["name"] == "Detect"]
    assert {row["page_number"] for row in stage_rows} == {1, 2}
    assert all(row["span_pages"] == 2 for row in stage_rows)
    nested = next(row for row in records if row["name"] == "detect.page")
    assert nested["page_number"] == 1
    assert nested["stage"] == "Detect"
    assert nested["depth"] == 1


def test_jsonl_records_denormalize_page_characteristics() -> None:
    record = next(row for row in _trace_with_two_pages().jsonl_records() if row["page_number"] == 1)
    assert record["source"] == "a.pdf"
    assert record["num_detections"] == 12
    assert record["last_stage"] == "Detect"


def test_jsonl_record_seconds_match_page_attribution() -> None:
    trace = _trace_with_two_pages()
    attributed = sum(sum(per_stage.values()) for per_stage in trace.page_attribution().values())
    exported = sum(row["seconds"] for row in trace.jsonl_records())
    assert exported == pytest.approx(attributed, abs=1e-6)


def test_jsonl_records_stamp_run_fields_on_every_row() -> None:
    records = list(_trace_with_two_pages().jsonl_records(run_id="run-1", n_documents=2))
    assert all(row["run_id"] == "run-1" and row["n_documents"] == 2 for row in records)


def test_save_jsonl_is_loadable_with_pandas(tmp_path) -> None:
    target = _trace_with_two_pages().save_jsonl(tmp_path / "nested" / "trace.jsonl", run_id="run-1")

    frame = pd.read_json(target, lines=True)
    assert list(frame["run_id"].unique()) == ["run-1"]
    # The two questions the export exists to answer.
    by_operation = frame.groupby("name")["seconds"].sum()
    assert set(by_operation.index) == {"Detect", "detect.page"}
    by_page = frame.groupby("page_number")["seconds"].sum()
    assert set(by_page.index) == {1, 2}


def test_save_jsonl_writes_an_empty_file_for_an_empty_trace(tmp_path) -> None:
    target = PipelineTrace().save_jsonl(tmp_path / "empty.jsonl")
    assert target.read_text(encoding="utf-8") == ""


def test_save_ingest_trace_uses_the_default_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = save_ingest_trace(_trace_with_two_pages(), documents=3)

    assert target.parent == Path(DEFAULT_TRACE_DIR)
    assert target.name.startswith("ingest-trace-")
    assert target.suffix == ".jsonl"
    assert all(row["n_documents"] == 3 for row in _read_jsonl(target))


def test_save_ingest_trace_honors_an_explicit_directory(tmp_path) -> None:
    target = save_ingest_trace(_trace_with_two_pages(), trace_dir=tmp_path / "traces", run_id="run-7")

    assert target == tmp_path / "traces" / "ingest-trace-run-7.jsonl"
    assert all(row["run_id"] == "run-7" for row in _read_jsonl(target))


def test_trace_file_path_sanitizes_the_run_id(tmp_path) -> None:
    target = trace_file_path("../../etc/passwd", trace_dir=tmp_path)
    # Path separators are replaced, so the run id cannot escape the directory.
    assert target.parent == tmp_path
    assert target.name == "ingest-trace-..-..-etc-passwd.jsonl"


def test_new_run_id_is_sortable_and_filesystem_safe() -> None:
    run_id = new_run_id(now=datetime(2026, 9, 10, 13, 14, 15, tzinfo=timezone.utc))
    assert run_id.startswith("20260910T131415Z-")
    assert trace_file_path(run_id).name == f"ingest-trace-{run_id}.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# Executor integration
# ---------------------------------------------------------------------------


def test_inprocess_executor_records_one_span_per_stage() -> None:
    trace = PipelineTrace()
    graph = _linear_graph(_PageProducingOperator(pages=2), _PerPageSpanOperator())
    executor = InprocessExecutor(graph, show_progress=False, trace=trace)

    result = executor.ingest(pd.DataFrame([{"path": "a.pdf", "bytes": b"12"}]))

    assert len(result.index) == 2
    stage_spans = [span for span in trace.spans if span.depth == 0]
    assert [span.name for span in stage_spans] == ["_PageProducingOperator", "_PerPageSpanOperator"]
    producer, consumer = stage_spans
    assert producer.attributes["rows_in"] == 1
    assert producer.attributes["rows_out"] == 2
    assert producer.attributes["operator"] == "_PageProducingOperator"
    # The producing stage covers both the document it read and the pages it made.
    assert ("a.pdf", 1) in producer.page_keys
    assert consumer.page_keys == (("a.pdf", 1), ("a.pdf", 2))


def test_inprocess_executor_collects_nested_operator_spans() -> None:
    trace = PipelineTrace()
    graph = _linear_graph(_PageProducingOperator(pages=3), _PerPageSpanOperator())
    InprocessExecutor(graph, show_progress=False, trace=trace).ingest(pd.DataFrame([{"path": "a.pdf", "bytes": b"12"}]))

    nested = [span for span in trace.spans if span.name == "fake.page"]
    assert len(nested) == 3
    assert {span.page_keys for span in nested} == {(("a.pdf", number),) for number in (1, 2, 3)}
    assert all(span.stage == "_PerPageSpanOperator" for span in nested)


def test_inprocess_executor_captures_page_characteristics_between_stages() -> None:
    trace = PipelineTrace()
    graph = _linear_graph(_PageProducingOperator(pages=2))
    InprocessExecutor(graph, show_progress=False, trace=trace).ingest(
        pd.DataFrame([{"path": "a.pdf", "bytes": b"12345"}])
    )

    features = trace.page_features
    assert features[("a.pdf", 0)]["source_bytes"] == 5
    assert features[("a.pdf", 1)]["text_chars"] == 1
    assert features[("a.pdf", 2)]["text_chars"] == 2


def test_inprocess_executor_leaves_no_active_trace_behind() -> None:
    trace = PipelineTrace()
    graph = _linear_graph(_PageProducingOperator())
    InprocessExecutor(graph, show_progress=False, trace=trace).ingest(pd.DataFrame([{"path": "a.pdf", "bytes": b"1"}]))
    assert active_trace() is None


def test_inprocess_executor_runs_untraced_by_default() -> None:
    graph = _linear_graph(_PageProducingOperator(pages=2), _PerPageSpanOperator())
    result = InprocessExecutor(graph, show_progress=False).ingest(pd.DataFrame([{"path": "a.pdf", "bytes": b"1"}]))
    assert len(result.index) == 2
    assert active_trace() is None


def test_inprocess_executor_span_records_a_failing_stage() -> None:
    class _FailingOperator(AbstractOperator, CPUOperator):
        def preprocess(self, data: Any, **kwargs: Any) -> Any:
            return data

        def process(self, data: Any, **kwargs: Any) -> Any:
            raise RuntimeError("stage failed")

        def postprocess(self, data: Any, **kwargs: Any) -> Any:
            return data

    trace = PipelineTrace()
    graph = _linear_graph(_FailingOperator())
    with pytest.raises(RuntimeError, match="stage failed"):
        InprocessExecutor(graph, show_progress=False, trace=trace).ingest(
            pd.DataFrame([{"path": "a.pdf", "bytes": b"1"}])
        )

    (span,) = trace.spans
    assert span.error == "RuntimeError: stage failed"
    assert active_trace() is None


# ---------------------------------------------------------------------------
# GraphIngestor return contract
# ---------------------------------------------------------------------------


def test_ingest_returns_only_the_dataframe_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor = GraphIngestor(run_mode="inprocess", show_progress=False).extract(ExtractParams(extract_text=True))
    result = _run_graph_ingest_with_result(ingestor, pd.DataFrame([{"path": "a.pdf"}]), monkeypatch)
    assert isinstance(result, pd.DataFrame)
    assert ingestor.last_trace is None


def test_ingest_with_return_traces_returns_the_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor = GraphIngestor(run_mode="inprocess", show_progress=False).extract(ExtractParams(extract_text=True))
    result, trace = _run_graph_ingest_with_result(
        ingestor,
        pd.DataFrame([{"path": "a.pdf"}]),
        monkeypatch,
        return_traces=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert isinstance(trace, PipelineTrace)
    assert trace.run_mode == "inprocess"
    assert trace.total_s >= 0.0
    assert ingestor.last_trace is trace


def test_ingest_with_both_flags_returns_result_failures_and_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor = GraphIngestor(run_mode="inprocess", show_progress=False).extract(ExtractParams(extract_text=True))
    result, failures, trace = _run_graph_ingest_with_result(
        ingestor,
        pd.DataFrame([{"path": "a.pdf"}]),
        monkeypatch,
        return_failures=True,
        return_traces=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert failures == []
    assert isinstance(trace, PipelineTrace)


def test_ingest_reads_return_traces_from_execute_params(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor = GraphIngestor(run_mode="inprocess", show_progress=False).extract(ExtractParams(extract_text=True))
    result, trace = _run_graph_ingest_with_result(
        ingestor,
        pd.DataFrame([{"path": "a.pdf"}]),
        monkeypatch,
        params=IngestExecuteParams(return_traces=True),
    )
    assert isinstance(result, pd.DataFrame)
    assert isinstance(trace, PipelineTrace)


def test_ingest_kwarg_overrides_return_traces_from_params(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor = GraphIngestor(run_mode="inprocess", show_progress=False).extract(ExtractParams(extract_text=True))
    result = _run_graph_ingest_with_result(
        ingestor,
        pd.DataFrame([{"path": "a.pdf"}]),
        monkeypatch,
        params=IngestExecuteParams(return_traces=True),
        return_traces=False,
    )
    assert isinstance(result, pd.DataFrame)


def test_inprocess_execution_passes_the_trace_to_the_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _RecordingExecutor:
        def __init__(self, graph: Any, *, show_progress: bool = True, trace: Any = None) -> None:
            captured["trace"] = trace

        def ingest(self, data: Any, **kwargs: Any) -> Any:
            return pd.DataFrame([{"path": "a.pdf"}])

    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.InprocessExecutor", _RecordingExecutor)
    ingestor = GraphIngestor(run_mode="inprocess", show_progress=False).files(["a.pdf"]).extract()
    ingestor._begin_trace(True)

    ingestor._execute_single_graph_inprocess(
        SimpleNamespace(
            extraction_mode="pdf",
            extract_params=ingestor._extract_params,
            text_params=None,
            html_params=None,
            audio_chunk_params=None,
            asr_params=None,
            video_frame_params=None,
            video_text_dedup_params=None,
            av_fuse_params=None,
        ),
        post_extract_order=(),
    )

    assert captured["trace"] is ingestor.last_trace
    assert isinstance(captured["trace"], PipelineTrace)


def test_batch_run_mode_returns_an_empty_trace_with_a_note(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor = GraphIngestor(run_mode="batch").files(["a.pdf"]).extract(ExtractParams(extract_text=True))
    monkeypatch.setattr(ingestor, "_plan_default_extraction_branches", lambda: None)
    monkeypatch.setattr(
        ingestor,
        "_resolve_effective_extraction_inputs",
        lambda: SimpleNamespace(extraction_mode="pdf"),
    )
    monkeypatch.setattr(
        ingestor,
        "_execute_single_graph",
        lambda effective_extraction, *, post_extract_order: pd.DataFrame([{"path": "a.pdf"}]),
    )

    _result, trace = ingestor.ingest(return_traces=True)
    assert trace.run_mode == "batch"
    assert trace.spans == ()
    assert any("run_mode='inprocess'" in note for note in trace.notes)


def test_blank_corpus_short_circuit_still_returns_a_trace() -> None:
    """The early return for a blank corpus must honor the trace contract."""
    ingestor = GraphIngestor(run_mode="inprocess", show_progress=False).texts(["   "])
    result, trace = ingestor.ingest(return_traces=True)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(trace, PipelineTrace)
    assert trace.spans == ()
    assert trace.total_s >= 0.0
