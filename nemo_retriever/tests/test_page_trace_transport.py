# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport and client-surface tests for page traces.

The service path is the fragile one: result rows are sanitized before they
leave the worker, and ``_sanitize_result_value`` truncates long strings. A
trace that rode inside ``result_data`` would arrive as unparseable JSON, so
these tests pin the artifact to its own transport channel.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.common.schemas.responses import DocumentStatusResponse, JobStatusResponse
from nemo_retriever.common.tracing import TRACE_SCHEMA_VERSION, load_traces, trace_filename
from nemo_retriever.ingestor.results import dataframe_to_transport_records
from nemo_retriever.service.services import worker_result_store
from nemo_retriever.service.services.job_tracker import JobTracker


def _page_trace(*, pages: int = 3, endpoint_padding: int = 4000) -> dict[str, Any]:
    """A trace whose span attributes exceed the transport truncation limit."""
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "nemo_retriever": {"version": "26.1.0", "full_version": "26.1.0+abc1234"},
        "run": {"run_id": "run-1", "run_mode": "service"},
        "document": {
            "document_id": "report.pdf-abc12345",
            "source_path": "/data/report.pdf",
            "source_type": "pdf",
            "page_count": pages,
            "status": "completed",
            "wall_ms": 900.0,
        },
        "pipeline": {"operators": ["PDFExtractionActor"], "params": {}},
        "models": [{"model_key": "ocr", "name": "nvidia/nemoretriever-ocr-v1", "version": "v2"}],
        "document_summary": {"total_ms": 900.0, "by_operator": [], "by_category": {}, "by_model": {}},
        "page_summaries": [
            {"page_number": page, "total_ms": 300.0, "by_operator": {}, "by_category": {}}
            for page in range(1, pages + 1)
        ],
        "spans": [
            {
                "span_id": f"s{page}",
                "page_number": page,
                "name": "nim.infer",
                "category": "network",
                "duration_ms": 300.0,
                "amortized_ms": 300.0,
                # Long enough that row sanitization would truncate it.
                "attrs": {"endpoint": "http://ocr:8000/v1/infer?q=" + "x" * endpoint_padding},
            }
            for page in range(1, pages + 1)
        ],
    }


@pytest.fixture(autouse=True)
def _clean_result_store() -> Any:
    worker_result_store.clear_for_tests()
    yield
    worker_result_store.clear_for_tests()


# ---------------------------------------------------------------------------
# The truncation trap
# ---------------------------------------------------------------------------


def test_row_sanitization_would_corrupt_an_embedded_trace() -> None:
    """Documents why the trace does not ride inside ``result_data``."""
    trace = _page_trace()
    serialized = json.dumps(trace)
    frame = pd.DataFrame([{"source_id": "doc-1", "page_trace": serialized}])

    (record,) = dataframe_to_transport_records(frame)

    assert len(record["page_trace"]) < len(serialized)
    with pytest.raises(json.JSONDecodeError):
        json.loads(record["page_trace"])


def test_worker_result_store_round_trips_a_trace_in_memory() -> None:
    trace = _page_trace()
    worker_result_store.store_page_trace("doc-1", trace)

    fetched = worker_result_store.get_page_trace("doc-1")
    assert fetched == trace
    # Long attribute values must survive intact.
    assert len(fetched["spans"][0]["attrs"]["endpoint"]) == len(trace["spans"][0]["attrs"]["endpoint"])
    # Reading does not consume, so a retrying client can fetch again.
    assert worker_result_store.get_page_trace("doc-1") == trace


def test_worker_result_store_round_trips_a_trace_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_WORKER_RESULTS_DIR", str(tmp_path))
    worker_result_store.clear_for_tests()
    trace = _page_trace()

    worker_result_store.store_page_trace("doc-1", trace)
    assert worker_result_store.get_page_trace("doc-1") == trace


def test_worker_result_store_ignores_empty_traces() -> None:
    worker_result_store.store_page_trace("doc-1", None)
    worker_result_store.store_page_trace("doc-1", {})
    worker_result_store.store_page_trace("", _page_trace())
    assert worker_result_store.get_page_trace("doc-1") is None


def test_trace_and_result_data_are_stored_independently() -> None:
    worker_result_store.store_result_data("doc-1", [{"source_id": "doc-1"}])
    worker_result_store.store_page_trace("doc-1", _page_trace())

    worker_result_store.discard_local_result_data("doc-1")
    # Dropping rows must not leave a trace behind that outlives them.
    assert worker_result_store.get_result_data("doc-1") is None
    assert worker_result_store.get_page_trace("doc-1") is None


# ---------------------------------------------------------------------------
# Job tracker and response models
# ---------------------------------------------------------------------------


def test_job_tracker_retains_the_trace_on_the_document_record() -> None:
    tracker = JobTracker()
    trace = _page_trace()
    tracker.register_job("job-1", expected_documents=1)
    tracker.register_document("doc-1", job_id="job-1", filename="report.pdf")

    tracker.mark_completed("doc-1", result_rows=7, page_trace=trace)

    record = tracker.get_document("doc-1")
    assert record is not None
    assert record.page_trace == trace
    # A stored copy, so a later mutation of the caller's dict cannot leak in.
    trace["document"]["page_count"] = 999
    assert record.page_trace["document"]["page_count"] == 3


def _document_response(**overrides: Any) -> DocumentStatusResponse:
    fields: dict[str, Any] = {
        "document_id": "doc-1",
        "attempt_id": "attempt-1",
        "job_id": "job-1",
        "status": "completed",
        "submitted_at": "2026-01-01T00:00:00+00:00",
    }
    fields.update(overrides)
    return DocumentStatusResponse(**fields)


def _job_response(**overrides: Any) -> JobStatusResponse:
    fields: dict[str, Any] = {
        "id": "job-1",
        "status": "completed",
        "submitted_at": "2026-01-01T00:00:00+00:00",
    }
    fields.update(overrides)
    return JobStatusResponse(**fields)


def test_document_and_job_responses_carry_the_trace() -> None:
    trace = _page_trace()

    assert _document_response(page_trace=trace).page_trace == trace
    assert _job_response(page_trace=trace).page_trace == trace
    # The field is optional and additive, so omitting it stays valid.
    assert _document_response(status="processing").page_trace is None
    assert _job_response(status="processing").page_trace is None


def test_response_trace_survives_a_json_round_trip() -> None:
    trace = _page_trace()
    response = _document_response(page_trace=trace)

    revived = DocumentStatusResponse.model_validate(json.loads(response.model_dump_json()))
    assert revived.page_trace == trace


# ---------------------------------------------------------------------------
# Pipeline spec plumbing
# ---------------------------------------------------------------------------


def test_pipeline_spec_defaults_to_no_tracing_and_stays_empty() -> None:
    spec = PipelineSpec()
    # Service tracing is opt-in: the artifact rides every status response for
    # the document, so a client that did not ask must not pay for one. An
    # untouched spec therefore still short-circuits to the baked pipeline.
    assert spec.page_trace_detail == "off"
    assert spec.is_empty()


def test_requesting_a_trace_makes_the_spec_non_empty() -> None:
    assert not PipelineSpec(page_trace_detail="operator").is_empty()
    assert not PipelineSpec(page_trace_detail="full").is_empty()


def test_pipeline_spec_rejects_an_unknown_detail_level() -> None:
    with pytest.raises(Exception):
        PipelineSpec(page_trace_detail="verbose")


def test_trace_detail_is_a_benign_override_under_policy() -> None:
    from nemo_retriever.common.policy import validate_pipeline_spec
    from nemo_retriever.service.config import PipelineOverridesConfig

    # Tracing is a diagnostic knob, so a spec carrying only page_trace_detail
    # must be accepted even where per-request overrides are rejected outright.
    reject_all = PipelineOverridesConfig(mode="reject")
    spec = PipelineSpec(page_trace_detail="full")

    assert validate_pipeline_spec(spec, reject_all) == spec


# ---------------------------------------------------------------------------
# Client-side persistence
# ---------------------------------------------------------------------------


def test_service_client_writes_the_trace_named_by_document(tmp_path: Path) -> None:
    from nemo_retriever.service.service_ingestor import ServiceIngestor

    ingestor = ServiceIngestor(base_url="http://localhost:7670").save_page_traces(output_directory=tmp_path)
    trace = _page_trace()

    written = ingestor._write_page_trace_to_disk("attempt-42", trace)

    assert written.name == trace_filename("attempt-42")
    assert json.loads(written.read_text(encoding="utf-8")) == trace
    assert load_traces(tmp_path) == [trace]


def test_service_client_honors_gzip_compression(tmp_path: Path) -> None:
    from nemo_retriever.service.service_ingestor import ServiceIngestor

    ingestor = ServiceIngestor(base_url="http://localhost:7670").save_page_traces(
        output_directory=tmp_path, compression="gzip"
    )
    trace = _page_trace()

    written = ingestor._write_page_trace_to_disk("attempt-42", trace)

    assert written.name == trace_filename("attempt-42", compression="gzip")
    with gzip.open(written, "rb") as handle:
        assert json.loads(handle.read().decode("utf-8")) == trace


def test_service_client_requires_save_page_traces_before_writing(tmp_path: Path) -> None:
    from nemo_retriever.service.service_ingestor import ServiceIngestor

    ingestor = ServiceIngestor(base_url="http://localhost:7670")
    with pytest.raises(RuntimeError, match="save_page_traces"):
        ingestor._write_page_trace_to_disk("attempt-42", _page_trace())


def test_page_traces_property_is_populated_when_only_saving_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``page_traces`` must not depend on how the caller asked for traces.

    The local run modes populate it whenever tracing produced something, so a
    caller that switches ``run_mode`` to ``"service"`` would otherwise find the
    property silently empty.
    """
    from nemo_retriever.service.service_ingestor import ServiceIngestor

    trace = _page_trace()
    ingestor = ServiceIngestor(base_url="http://localhost:7670").save_page_traces(output_directory=tmp_path)
    ingestor._apply_page_trace_flags(None, {})
    monkeypatch.setattr(
        ServiceIngestor,
        "_fetch_document_status_body",
        lambda self, document_id, client=None: {"result_data": [], "page_trace": trace},
    )

    ingestor._materialize_completed_document("doc-1", return_results=False)

    # Saving to disk alone did not request the trace on the return value.
    assert ingestor.page_traces == [trace]
    assert load_traces(tmp_path) == [trace]


def test_service_client_sends_the_requested_detail_on_the_spec() -> None:
    from nemo_retriever.service.service_ingestor import ServiceIngestor

    ingestor = ServiceIngestor(base_url="http://localhost:7670")
    assert ingestor._apply_page_trace_flags(None, {"page_trace_detail": "full"}) is False
    assert ingestor._page_trace_detail_active == "full"

    payload = ingestor._pipeline_payload(
        result_schema="legacy", return_embeddings=False, return_images=False
    )
    assert payload is not None
    assert payload["page_trace_detail"] == "full"


def test_service_client_does_not_request_traces_it_will_not_read() -> None:
    from nemo_retriever.service.service_ingestor import ServiceIngestor

    ingestor = ServiceIngestor(base_url="http://localhost:7670")
    ingestor._apply_page_trace_flags(None, {})

    # A run that neither returns nor saves traces must not ask the worker for
    # one, because it would ride back on every status response for free.
    assert ingestor._page_trace_detail_active == "off"
    assert (
        ingestor._pipeline_payload(result_schema="legacy", return_embeddings=False, return_images=False)
        is None
    )


def test_service_client_opts_in_when_returning_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_retriever.service.service_ingestor import ServiceIngestor

    monkeypatch.delenv("NEMO_RETRIEVER_PAGE_TRACE_DETAIL", raising=False)
    ingestor = ServiceIngestor(base_url="http://localhost:7670")

    assert ingestor._apply_page_trace_flags(None, {"return_page_traces": True}) is True
    assert ingestor._page_trace_detail_active == "operator"

    payload = ingestor._pipeline_payload(result_schema="legacy", return_embeddings=False, return_images=False)
    assert payload is not None
    assert payload["page_trace_detail"] == "operator"


def test_service_client_opts_in_when_saving_traces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_retriever.service.service_ingestor import ServiceIngestor

    monkeypatch.delenv("NEMO_RETRIEVER_PAGE_TRACE_DETAIL", raising=False)
    ingestor = ServiceIngestor(base_url="http://localhost:7670").save_page_traces(output_directory=tmp_path)

    ingestor._apply_page_trace_flags(None, {})

    assert ingestor._page_trace_detail_active == "operator"


def test_worker_treats_an_absent_spec_key_as_tracing_off() -> None:
    from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec as Spec

    # The wire default has to be the neutral value, otherwise every plain
    # service ingest would carry a non-empty spec and lose the worker's
    # short-circuit to the baked-at-startup pipeline.
    assert Spec().page_trace_detail == "off"
    assert Spec(page_trace_detail="off").is_empty()


def test_saving_traces_implies_tracing_even_when_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_retriever.service.service_ingestor import ServiceIngestor

    monkeypatch.setenv("NEMO_RETRIEVER_PAGE_TRACE_DETAIL", "off")
    ingestor = ServiceIngestor(base_url="http://localhost:7670").save_page_traces(output_directory=tmp_path)

    ingestor._apply_page_trace_flags(None, {})

    # Asking for a trace file while the environment says off would otherwise
    # write nothing, so the explicit request wins.
    assert ingestor._page_trace_detail_active == "operator"


def test_execute_params_accept_the_page_trace_fields() -> None:
    from nemo_retriever.common.params import IngestExecuteParams

    default = IngestExecuteParams()
    # Unset by default so the environment variable still governs.
    assert default.page_trace_detail is None
    assert default.return_page_traces is False

    explicit = IngestExecuteParams(page_trace_detail="full", return_page_traces=True)
    assert explicit.page_trace_detail == "full"
    assert explicit.return_page_traces is True

    with pytest.raises(Exception):
        IngestExecuteParams(page_trace_detail="loud")
