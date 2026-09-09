# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-page work counters.

The heavy stages batch their inference across pages, so one duration covers the
whole batch and amortizing it gives every page in that batch an identical time.
Work counters exist to stay meaningful in exactly that case: they are counted on
each page individually, whatever the batch size, so an expensive page remains
identifiable when its timings are averages.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from nemo_retriever.common.tracing import (
    TRACE_COLUMN,
    aggregate_document_traces,
    page_summaries_dataframe,
    record_page_work,
)
from nemo_retriever.common.tracing import runtime as trace_runtime
from nemo_retriever.common.tracing.collector import decode_payload_full, operator_trace
from nemo_retriever.common.tracing.spans import page_scope
from nemo_retriever.cli.trace import report
from nemo_retriever.operators.abstract_operator import AbstractOperator

DOC_A = "/data/a.pdf"
DOC_B = "/data/b.pdf"


@pytest.fixture(autouse=True)
def _isolated_trace_runtime() -> Any:
    trace_runtime.reset_detail()
    trace_runtime.clear_models()
    yield
    trace_runtime.reset_detail()
    trace_runtime.clear_models()


def _page_frame(path: str, pages: range | list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "path": path,
                "source_id": f"{path}_{page}",
                "page_number": page,
                "content": f"page {page}",
            }
            for page in pages
        ]
    )


class _StubOperator(AbstractOperator):
    def preprocess(self, data: Any, **_: Any) -> Any:
        return data

    def postprocess(self, data: Any, **_: Any) -> Any:
        return data


class _OCRActor(_StubOperator):
    """Batches inference across pages; only the crop count is per page."""

    def __init__(self, crops_by_page: dict[int, int]) -> None:
        self._crops = crops_by_page

    def process(self, data: pd.DataFrame, **_: Any) -> pd.DataFrame:
        for row in data.itertuples(index=False):
            record_page_work(row.source_id, crops=self._crops[int(row.page_number)])
        return data.copy()


class _ExplodeOperator(_StubOperator):
    """Fans each page row out into element rows, copying the trace payload."""

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


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_recording_outside_an_operator_is_a_no_op() -> None:
    trace_runtime.set_detail("operator")
    record_page_work(f"{DOC_A}_1", crops=5)  # No collector installed.


def test_recording_is_a_no_op_when_tracing_is_disabled() -> None:
    trace_runtime.set_detail("off")
    frame = _page_frame(DOC_A, [1])

    class _Operator:
        pass

    with operator_trace(_Operator(), frame) as tracer:
        record_page_work(f"{DOC_A}_1", crops=5)
        out = tracer.finish(frame)

    assert TRACE_COLUMN not in out.columns


def test_counters_are_namespaced_by_operator_and_charged_to_one_page() -> None:
    trace_runtime.set_detail("operator")
    frame = _OCRActor({1: 2, 2: 34}).run(_page_frame(DOC_A, [1, 2]))

    (trace,) = aggregate_document_traces(frame, run_mode="inprocess")
    work_by_page = {summary["page_number"]: summary["work"] for summary in trace["page_summaries"]}

    assert work_by_page[1] == {"_OCRActor.crops": 2.0}
    assert work_by_page[2] == {"_OCRActor.crops": 34.0}


def test_page_scope_supplies_the_page_when_omitted() -> None:
    trace_runtime.set_detail("operator")
    frame = _page_frame(DOC_A, [1, 2])

    class _Operator:
        pass

    with operator_trace(_Operator(), frame) as tracer:
        for row in frame.itertuples(index=False):
            with page_scope(row.source_id):
                record_page_work(detections=int(row.page_number))
        out = tracer.finish(frame)

    (trace,) = aggregate_document_traces(out, run_mode="inprocess")
    work_by_page = {summary["page_number"]: summary["work"] for summary in trace["page_summaries"]}
    assert work_by_page[1]["_Operator.detections"] == 1.0
    assert work_by_page[2]["_Operator.detections"] == 2.0


def test_repeated_counts_for_one_page_accumulate() -> None:
    trace_runtime.set_detail("operator")
    frame = _page_frame(DOC_A, [1])

    class _Operator:
        pass

    with operator_trace(_Operator(), frame) as tracer:
        record_page_work(f"{DOC_A}_1", crops=3)
        record_page_work(f"{DOC_A}_1", crops=4)
        out = tracer.finish(frame)

    (trace,) = aggregate_document_traces(out, run_mode="inprocess")
    assert trace["page_summaries"][0]["work"]["_Operator.crops"] == 7.0


@pytest.mark.parametrize("value", [None, True, False, "many", object()])
def test_non_numeric_counts_are_ignored(value: Any) -> None:
    trace_runtime.set_detail("operator")
    frame = _page_frame(DOC_A, [1])

    class _Operator:
        pass

    with operator_trace(_Operator(), frame) as tracer:
        record_page_work(f"{DOC_A}_1", crops=value)
        out = tracer.finish(frame)

    (trace,) = aggregate_document_traces(out, run_mode="inprocess")
    assert trace["page_summaries"][0]["work"] == {}


def test_work_for_a_page_missing_from_the_output_is_not_misattributed() -> None:
    trace_runtime.set_detail("operator")
    frame = _page_frame(DOC_A, [1, 2])

    class _Operator:
        pass

    with operator_trace(_Operator(), frame) as tracer:
        record_page_work(f"{DOC_A}_1", crops=9)
        record_page_work(f"{DOC_A}_2", crops=4)
        # Page 1 is filtered out of the output.
        out = tracer.finish(frame[frame["page_number"] == 2].copy())

    (trace,) = aggregate_document_traces(out, run_mode="inprocess")
    (summary,) = trace["page_summaries"]
    assert summary["page_number"] == 2
    assert summary["work"] == {"_Operator.crops": 4.0}


# ---------------------------------------------------------------------------
# Survival through fan-out
# ---------------------------------------------------------------------------


def test_counts_survive_fan_out_without_double_counting() -> None:
    trace_runtime.set_detail("operator")
    crops = {1: 2, 2: 3, 3: 34}
    frame = _OCRActor(crops).run(_page_frame(DOC_A, [1, 2, 3]))
    frame = _ExplodeOperator(4).run(frame)

    assert len(frame) == 12
    # Sibling element rows really do carry duplicate work records.
    carried = [record for raw in frame[TRACE_COLUMN] for record in decode_payload_full(raw)[2]]
    assert len(carried) > 3

    (trace,) = aggregate_document_traces(frame, run_mode="inprocess")
    work_by_page = {summary["page_number"]: summary["work"]["_OCRActor.crops"] for summary in trace["page_summaries"]}
    assert work_by_page == {1: 2.0, 2: 3.0, 3: 34.0}
    assert trace["document_summary"]["work_totals"]["_OCRActor.crops"] == float(sum(crops.values()))


def test_counts_from_several_operators_accumulate_per_page() -> None:
    trace_runtime.set_detail("operator")

    class _EmbedActor(_StubOperator):
        def process(self, data: pd.DataFrame, **_: Any) -> pd.DataFrame:
            for row in data.itertuples(index=False):
                record_page_work(row.source_id, text_chars=100 * int(row.page_number))
            return data.copy()

    frame = _OCRActor({1: 2, 2: 5}).run(_page_frame(DOC_A, [1, 2]))
    frame = _EmbedActor().run(frame)

    (trace,) = aggregate_document_traces(frame, run_mode="inprocess")
    work_by_page = {summary["page_number"]: summary["work"] for summary in trace["page_summaries"]}
    assert work_by_page[1] == {"_OCRActor.crops": 2.0, "_EmbedActor.text_chars": 100.0}
    assert work_by_page[2] == {"_OCRActor.crops": 5.0, "_EmbedActor.text_chars": 200.0}


def test_documents_sharing_a_batch_keep_their_own_counts() -> None:
    trace_runtime.set_detail("operator")
    frame = pd.concat([_page_frame(DOC_A, [1]), _page_frame(DOC_B, [1, 2])], ignore_index=True)

    class _Operator:
        pass

    with operator_trace(_Operator(), frame) as tracer:
        record_page_work(f"{DOC_A}_1", crops=11)
        record_page_work(f"{DOC_B}_1", crops=2)
        record_page_work(f"{DOC_B}_2", crops=3)
        out = tracer.finish(frame)

    traces = {trace["document"]["source_path"]: trace for trace in aggregate_document_traces(out)}
    assert traces[DOC_A]["document_summary"]["work_totals"]["_Operator.crops"] == 11.0
    assert traces[DOC_B]["document_summary"]["work_totals"]["_Operator.crops"] == 5.0


# ---------------------------------------------------------------------------
# Telling measured timings from batch averages
# ---------------------------------------------------------------------------


def test_a_single_page_batch_reports_measured_timings() -> None:
    trace_runtime.set_detail("operator")
    operator = _OCRActor({1: 1, 2: 1})
    frame = pd.concat([operator.run(_page_frame(DOC_A, [page])) for page in (1, 2)], ignore_index=True)

    (trace,) = aggregate_document_traces(frame, run_mode="batch")
    for summary in trace["page_summaries"]:
        assert summary["timing_source"] == "measured"
        assert summary["max_page_fanout"] == 1


def test_a_multi_page_batch_reports_amortized_timings_but_exact_work() -> None:
    trace_runtime.set_detail("operator")
    frame = _OCRActor({1: 2, 2: 3, 3: 34}).run(_page_frame(DOC_A, [1, 2, 3]))

    (trace,) = aggregate_document_traces(frame, run_mode="inprocess")
    summaries = trace["page_summaries"]

    # Timings are identical across pages, which is precisely why work matters.
    assert {summary["timing_source"] for summary in summaries} == {"amortized"}
    assert {summary["max_page_fanout"] for summary in summaries} == {3}
    assert len({summary["total_ms"] for summary in summaries}) == 1
    assert len({summary["work"]["_OCRActor.crops"] for summary in summaries}) == 3


def test_page_summaries_dataframe_exposes_work_and_timing_source() -> None:
    trace_runtime.set_detail("operator")
    frame = _OCRActor({1: 2, 2: 34}).run(_page_frame(DOC_A, [1, 2]))
    traces = aggregate_document_traces(frame, run_mode="inprocess")

    pages = page_summaries_dataframe(traces)
    assert "work._OCRActor.crops" in pages.columns
    assert list(pages["timing_source"]) == ["amortized", "amortized"]
    assert pages.nlargest(1, "work._OCRActor.crops")["page_number"].tolist() == [2]


# ---------------------------------------------------------------------------
# CLI rollup
# ---------------------------------------------------------------------------


def test_rollup_ranks_pages_by_their_most_extreme_cost_driver() -> None:
    trace_runtime.set_detail("operator")

    class _EmbedActor(_StubOperator):
        def process(self, data: pd.DataFrame, **_: Any) -> pd.DataFrame:
            for row in data.itertuples(index=False):
                record_page_work(row.source_id, text_chars=10)
            return data.copy()

    # Page 2 is extreme by crops while text volume is identical everywhere. The
    # uniform counter must not dilute the skewed one, and the two use different
    # units so no cross-metric sum could rank them.
    frame = _OCRActor({1: 1, 2: 40, 3: 2}).run(_page_frame(DOC_A, [1, 2, 3]))
    frame = _EmbedActor().run(frame)

    rollup = report.summarize(aggregate_document_traces(frame, run_mode="inprocess"))

    assert rollup["timing_resolution"] == "amortized"
    assert rollup["heaviest_pages"][0]["page_number"] == 2
    # 40 crops against a 43/3 per-page average.
    assert rollup["heaviest_pages"][0]["work_index"] == pytest.approx(2.791, abs=1e-3)
    # A counter with the same value on every page scores exactly typical.
    assert [page["work_index"] for page in rollup["heaviest_pages"][1:]] == [1.0, 1.0]
    assert rollup["work_metrics"][0] == "_OCRActor.crops"
    assert rollup["work_totals"]["_OCRActor.crops"] == 43.0


def test_rollup_reports_measured_resolution_for_single_page_batches() -> None:
    trace_runtime.set_detail("operator")
    operator = _OCRActor({1: 1, 2: 1})
    frame = pd.concat([operator.run(_page_frame(DOC_A, [page])) for page in (1, 2)], ignore_index=True)

    rollup = report.summarize(aggregate_document_traces(frame, run_mode="batch"))
    assert rollup["timing_resolution"] == "measured"


def test_summary_warns_when_timings_are_batch_averages() -> None:
    trace_runtime.set_detail("operator")
    frame = _OCRActor({1: 2, 2: 34}).run(_page_frame(DOC_A, [1, 2]))
    rollup = report.summarize(aggregate_document_traces(frame, run_mode="inprocess"))

    from rich.console import Console

    console = Console(record=True, width=120)
    report.render_summary(console, rollup, top=5)
    output = console.export_text()

    assert "batch averages" in output
    assert "Heaviest pages by work volume" in output
    assert "run_mode='batch'" in output


def test_summary_does_not_warn_when_every_page_was_measured() -> None:
    trace_runtime.set_detail("operator")
    operator = _OCRActor({1: 1, 2: 1})
    frame = pd.concat([operator.run(_page_frame(DOC_A, [page])) for page in (1, 2)], ignore_index=True)
    rollup = report.summarize(aggregate_document_traces(frame, run_mode="batch"))

    from rich.console import Console

    console = Console(record=True, width=120)
    report.render_summary(console, rollup, top=5)
    output = console.export_text()

    assert "batch averages" not in output


def test_page_view_shows_work_and_labels_amortized_timings() -> None:
    trace_runtime.set_detail("operator")
    frame = _OCRActor({1: 2, 2: 34}).run(_page_frame(DOC_A, [1, 2]))
    (trace,) = aggregate_document_traces(frame, run_mode="inprocess")

    from rich.console import Console

    console = Console(record=True, width=120)
    report.render_page(console, report.page_detail(trace, 2))
    output = console.export_text()

    assert "Work recorded for this page" in output
    assert "amortized across 2 pages" in output
    assert "34" in output
