# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-surface tests for page tracing on ``GraphIngestor``.

These exercise the result-finalization seam directly rather than running a
real pipeline, so they cover trace aggregation, ``_nrl_trace`` stripping, and
the return/persist contract without needing models or Ray.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nemo_retriever.common.params import IngestExecuteParams
from nemo_retriever.common.tracing import TRACE_COLUMN, TRACE_DETAIL_ENV_VAR
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor


_EPOCH_MS = 1_760_000_000_000.0


def _span(source_id: str, name: str, duration_ms: float, *, start_offset_ms: float = 0.0) -> dict[str, Any]:
    start_ms = _EPOCH_MS + start_offset_ms
    return {
        "span_id": f"{source_id}:{name}",
        "parent_span_id": None,
        "page_number": int(source_id.rsplit("_", 1)[1]),
        "source_id": source_id,
        "name": name,
        "category": "operator",
        "operator": name,
        "model_key": None,
        "start_ms": start_ms,
        "end_ms": start_ms + duration_ms,
        "duration_ms": duration_ms,
        "batch_size": 1,
        "amortized_ms": duration_ms,
        "status": "ok",
        "attrs": {},
    }


def _traced_frame(path: str = "/data/report.pdf", pages: int = 2) -> pd.DataFrame:
    rows = []
    for page in range(1, pages + 1):
        source_id = f"{path}_{page}"
        rows.append(
            {
                "source_id": source_id,
                "source_name": path,
                "document_type": "text",
                "metadata": {"content": f"page {page}"},
                TRACE_COLUMN: json.dumps(
                    {
                        "spans": [
                            _span(source_id, "PDFSplitActor", 5.0),
                            _span(source_id, "PDFExtractionActor", 40.0, start_offset_ms=5.0),
                        ]
                    }
                ),
            }
        )
    return pd.DataFrame(rows)


def _ingestor(**kwargs: Any) -> GraphIngestor:
    return GraphIngestor(documents=["/data/report.pdf"], show_progress=False, **kwargs)


def _finalize(ingestor: GraphIngestor, frame: pd.DataFrame, **kwargs: Any) -> Any:
    ingestor._resolve_page_trace_options(None, kwargs)
    return ingestor._finalize_ingest_result(frame, return_failures=kwargs.get("return_failures", False))


def test_results_never_carry_the_internal_trace_column() -> None:
    ingestor = _ingestor()

    result = _finalize(ingestor, _traced_frame())

    assert TRACE_COLUMN not in result.columns
    assert list(result["source_id"]) == ["/data/report.pdf_1", "/data/report.pdf_2"]


def test_traces_are_aggregated_even_when_not_requested() -> None:
    ingestor = _ingestor()

    result = _finalize(ingestor, _traced_frame())

    # A bare ingest() returns just the frame, but the trace stays reachable.
    assert isinstance(result, pd.DataFrame)
    assert len(ingestor.page_traces) == 1
    assert ingestor.page_traces[0]["document"]["page_count"] == 2


def test_return_page_traces_appends_the_traces_to_the_result() -> None:
    ingestor = _ingestor()

    result, traces = _finalize(ingestor, _traced_frame(), return_page_traces=True)

    assert TRACE_COLUMN not in result.columns
    assert [trace["document"]["source_path"] for trace in traces] == ["/data/report.pdf"]
    assert traces == ingestor.page_traces


def test_failures_and_traces_are_returned_in_a_stable_order() -> None:
    ingestor = _ingestor(error_policy="collect")

    result, failures, traces = _finalize(
        ingestor,
        _traced_frame(),
        return_failures=True,
        return_page_traces=True,
    )

    assert isinstance(result, pd.DataFrame)
    assert failures == []
    assert len(traces) == 1


def test_page_rollups_sum_to_the_document_total() -> None:
    ingestor = _ingestor()

    _finalize(ingestor, _traced_frame(pages=3))
    trace = ingestor.page_traces[0]

    page_total = sum(page["total_ms"] for page in trace["page_summaries"])
    assert page_total == pytest.approx(trace["document_summary"]["total_ms"], abs=0.01)
    assert [page["page_number"] for page in trace["page_summaries"]] == [1, 2, 3]
    # Each page here has its own spans, so wall time is the span extent.
    assert all(page["wall_ms"] == pytest.approx(45.0) for page in trace["page_summaries"])


def test_save_page_traces_writes_one_file_per_document(tmp_path: Path) -> None:
    ingestor = _ingestor().save_page_traces(output_directory=str(tmp_path))

    _finalize(ingestor, _traced_frame())

    written = sorted(tmp_path.glob("*.trace.json"))
    assert len(written) == 1
    revived = json.loads(written[0].read_text())
    assert revived["schema_version"] == "1.0"
    assert revived["document"]["source_path"] == "/data/report.pdf"
    assert revived["nemo_retriever"]["version"]
    assert len(revived["spans"]) == 4


def test_gzip_compression_writes_a_readable_gz_file(tmp_path: Path) -> None:
    ingestor = _ingestor().save_page_traces(output_directory=str(tmp_path), compression="gzip")

    _finalize(ingestor, _traced_frame())

    written = sorted(tmp_path.glob("*.trace.json.gz"))
    assert len(written) == 1
    with gzip.open(written[0], "rt", encoding="utf-8") as handle:
        assert json.load(handle)["document"]["page_count"] == 2


def test_save_page_traces_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="output_directory is required"):
        _ingestor().save_page_traces(output_directory="")
    with pytest.raises(ValueError, match="only None or 'gzip'"):
        _ingestor().save_page_traces(output_directory="/tmp/x", compression="zstd")


def test_detail_off_skips_aggregation_entirely() -> None:
    ingestor = _ingestor()

    result = _finalize(ingestor, _traced_frame(), page_trace_detail="off")

    assert TRACE_COLUMN not in result.columns
    assert ingestor.page_traces == []


def test_requesting_traces_overrides_a_disabled_default(monkeypatch) -> None:
    monkeypatch.setenv(TRACE_DETAIL_ENV_VAR, "off")
    ingestor = _ingestor()

    # Asking for traces with tracing switched off would otherwise return
    # nothing, so the explicit request wins.
    _, traces = _finalize(ingestor, _traced_frame(), return_page_traces=True)

    assert ingestor._page_trace_detail == "operator"
    assert len(traces) == 1


def test_a_configured_output_directory_also_overrides_off(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(TRACE_DETAIL_ENV_VAR, "off")
    ingestor = _ingestor().save_page_traces(output_directory=str(tmp_path))

    _finalize(ingestor, _traced_frame())

    assert sorted(path.name for path in tmp_path.glob("*.trace.json"))


def test_execute_params_carry_the_trace_options() -> None:
    ingestor = _ingestor()
    params = IngestExecuteParams(page_trace_detail="full", return_page_traces=True)

    ingestor._resolve_page_trace_options(params, {})

    assert ingestor._page_trace_detail == "full"
    assert ingestor._return_page_traces is True


def test_kwargs_win_over_execute_params() -> None:
    ingestor = _ingestor()
    params = IngestExecuteParams(page_trace_detail="full")

    ingestor._resolve_page_trace_options(params, {"page_trace_detail": "operator"})

    assert ingestor._page_trace_detail == "operator"


def test_untraced_frames_finalize_without_producing_a_trace() -> None:
    ingestor = _ingestor()
    frame = pd.DataFrame([{"source_id": "/data/report.pdf_1", "metadata": {}}])

    result = _finalize(ingestor, frame)

    assert list(result.columns) == ["source_id", "metadata"]
    assert ingestor.page_traces == []


def test_aggregation_failure_does_not_fail_the_ingest(monkeypatch) -> None:
    from nemo_retriever.ingestor import graph_ingestor as module

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("aggregation bug")

    monkeypatch.setattr(module, "aggregate_document_traces", explode)
    ingestor = _ingestor()

    # Tracing is a diagnostic, so a bug in it must never cost the user results.
    result = _finalize(ingestor, _traced_frame())

    assert TRACE_COLUMN not in result.columns
    assert ingestor.page_traces == []
