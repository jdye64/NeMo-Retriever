# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plumbing tests for the ingest-side page trace flags.

Covers the path from ``retriever ingest`` CLI options through the resolved
plan to the ingestor calls, for both graph and service run modes.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest
import typer.main
from typer.testing import CliRunner

from nemo_retriever.ingest import execution as ingest_execution
from nemo_retriever.ingest import plan as ingest_plan
from nemo_retriever.ingest import service as ingest_service
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor

# Imported lazily so the CLI's own deferred sub-app registration runs first,
# matching tests/test_root_cli_workflow.py.
cli_main = importlib.import_module("nemo_retriever.cli.main")

RUNNER = CliRunner()


def _fake_ingestor() -> Any:
    fake = create_autospec(GraphIngestor, instance=True, spec_set=True)
    for method in ("files", "extract", "dedup", "caption", "embed", "store", "vdb_upload", "save_page_traces"):
        getattr(fake, method).return_value = fake
    fake.ingest.return_value = [{"status": "ok"}]
    return fake


def _document(tmp_path: Path, name: str = "traced.pdf") -> Path:
    document = tmp_path / name
    document.write_bytes(b"%PDF-1.4\n")
    return document


def _plan(tmp_path: Path, **trace_kwargs: Any) -> ingest_plan.ResolvedIngestPlan:
    return ingest_plan.resolve_ingest_plan(
        ingest_plan.IngestPlanRequest(
            source=ingest_plan.IngestSourceOptions(documents=[str(_document(tmp_path))], input_type="pdf"),
            trace=ingest_plan.IngestTraceOptions(**trace_kwargs),
        )
    )


def test_trace_options_default_to_unset(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    # Unset detail means the environment variable (and then the "operator"
    # default) decides, rather than the plan pinning a value.
    assert plan.page_trace_dir is None
    assert plan.page_trace_detail is None


def test_trace_options_reach_the_resolved_plan(tmp_path: Path) -> None:
    plan = _plan(tmp_path, page_trace_dir=str(tmp_path / "traces"), page_trace_detail="operator")

    assert plan.page_trace_dir == str(tmp_path / "traces")
    assert plan.page_trace_detail == "operator"


def test_unknown_trace_detail_is_rejected_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="page_trace_detail"):
        _plan(tmp_path, page_trace_detail="verbose")


def test_pipeline_build_enables_trace_persistence(monkeypatch, tmp_path: Path) -> None:
    fake = _fake_ingestor()
    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake)

    ingest_execution.build_ingest_pipeline(_plan(tmp_path, page_trace_dir=str(tmp_path / "traces")))

    fake.save_page_traces.assert_called_once_with(output_directory=str(tmp_path / "traces"))


def test_pipeline_build_skips_persistence_when_no_directory(monkeypatch, tmp_path: Path) -> None:
    fake = _fake_ingestor()
    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake)

    ingest_execution.build_ingest_pipeline(_plan(tmp_path))

    fake.save_page_traces.assert_not_called()


def test_execute_forwards_trace_detail_to_ingest(monkeypatch, tmp_path: Path) -> None:
    fake = _fake_ingestor()
    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_a, **_k: 3)

    ingest_execution.execute_ingest_plan(_plan(tmp_path, page_trace_detail="operator"))

    assert fake.ingest.call_args.kwargs == {"page_trace_detail": "operator"}


def test_execute_omits_trace_detail_when_unset(monkeypatch, tmp_path: Path) -> None:
    fake = _fake_ingestor()
    monkeypatch.setattr(ingest_execution, "create_ingestor", lambda **_kwargs: fake)
    monkeypatch.setattr(ingest_execution, "_count_lancedb_rows", lambda *_a, **_k: 3)

    ingest_execution.execute_ingest_plan(_plan(tmp_path))

    # Passing nothing keeps the env var in charge of the effective detail.
    assert "page_trace_detail" not in fake.ingest.call_args.kwargs


def _option_names(*command_path: str) -> set[str]:
    """Declared option strings for a CLI command.

    Read off the Click command rather than ``--help`` text, since Rich
    truncates long option names when it renders the help table.
    """
    command = typer.main.get_command(cli_main.app)
    for name in command_path:
        command = command.commands[name]  # type: ignore[attr-defined]
    return {opt for param in command.params for opt in param.opts}


@pytest.mark.parametrize("command", ["local", "batch", "service"])
def test_ingest_commands_expose_the_trace_flags(command: str) -> None:
    options = _option_names("ingest", command)

    assert "--page-trace-dir" in options
    assert "--page-trace-detail" in options


def test_graph_command_forwards_the_trace_flags(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def capture(request: ingest_plan.IngestPlanRequest) -> Any:
        captured["trace"] = request.trace
        raise SystemExit(0)

    from nemo_retriever.cli.ingest import graph_commands

    monkeypatch.setattr(graph_commands, "resolve_ingest_plan", capture)

    RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            "local",
            str(_document(tmp_path)),
            "--page-trace-dir",
            str(tmp_path / "traces"),
            "--page-trace-detail",
            "operator",
        ],
    )

    assert captured["trace"] == ingest_plan.IngestTraceOptions(
        page_trace_dir=str(tmp_path / "traces"),
        page_trace_detail="operator",
    )


def _service_plan_request(tmp_path: Path, **trace_kwargs: Any) -> Any:
    return ingest_service.ServiceIngestPlanRequest(
        source=ingest_service.ServiceIngestSourceOptions(
            documents=[str(_document(tmp_path))],
            input_type="pdf",
        ),
        trace=ingest_plan.IngestTraceOptions(**trace_kwargs),
    )


def test_service_request_resolution_carries_trace_options(tmp_path: Path) -> None:
    request = ingest_service.resolve_service_ingest_request(
        _service_plan_request(
            tmp_path,
            page_trace_dir=str(tmp_path / "traces"),
            page_trace_detail="operator",
        )
    )

    assert request.trace.page_trace_dir == str(tmp_path / "traces")
    assert request.trace.page_trace_detail == "operator"


def test_service_request_rejects_unknown_trace_detail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="page_trace_detail"):
        ingest_service.resolve_service_ingest_request(_service_plan_request(tmp_path, page_trace_detail="loud"))
