# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``retriever trace`` command group."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from nemo_retriever.common.tracing import TRACE_SCHEMA_VERSION, write_document_trace

RUNNER = CliRunner()
cli_main = importlib.import_module("nemo_retriever.cli.main")


def _span(
    span_id: str,
    *,
    page: int,
    name: str,
    category: str,
    duration_ms: float,
    amortized_ms: float,
    parent: str | None = None,
    model_key: str | None = None,
    attrs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "document_id": "report.pdf-abc12345",
        "source_path": "/data/report.pdf",
        "page_number": page,
        "source_id": f"/data/report.pdf_{page}",
        "span_id": span_id,
        "parent_span_id": parent,
        "name": name,
        "category": category,
        "operator": "ExtractActor" if category != "operator" else name,
        "model_key": model_key,
        "start_ms": 1_000.0,
        "end_ms": 1_000.0 + duration_ms,
        "duration_ms": duration_ms,
        "self_ms": duration_ms,
        "amortized_ms": amortized_ms,
        "amortized_self_ms": amortized_ms,
        "page_fanout": 2,
        "batch_size": 2,
        "status": "ok",
        "error": None,
        "worker": "host:1",
        "attrs": attrs or {},
    }


def _fixture_trace() -> dict[str, Any]:
    """A two-page document trace with one operator and one nested model span."""
    spans = []
    for page in (1, 2):
        spans.append(
            _span(
                "op1",
                page=page,
                name="ExtractActor",
                category="operator",
                duration_ms=200.0,
                amortized_ms=100.0,
            )
        )
        spans.append(
            _span(
                "net1",
                page=page,
                name="nim.infer",
                category="network",
                duration_ms=120.0,
                amortized_ms=60.0,
                parent="op1",
                model_key="ocr",
                attrs={"endpoint": "http://ocr:8000/v1/infer", "protocol": "http"},
            )
        )
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "nemo_retriever": {"version": "26.1.0", "full_version": "26.1.0+abc1234"},
        "run": {"run_id": "run-1", "run_mode": "batch", "completed_at": "2026-01-01T00:00:00+00:00"},
        "document": {
            "document_id": "report.pdf-abc12345",
            "source_path": "/data/report.pdf",
            "source_type": "pdf",
            "page_count": 2,
            "status": "completed",
            "wall_ms": 200.0,
        },
        "pipeline": {"operators": ["ExtractActor"], "params": {}},
        "models": [
            {
                "model_key": "ocr",
                "name": "nvidia/nemoretriever-ocr-v1",
                "version": "v2",
                "backend": "nim-http",
                "endpoint": "http://ocr:8000/v1/infer",
            }
        ],
        "document_summary": {
            "total_ms": 200.0,
            "wall_ms": 200.0,
            "span_count": 2,
            "by_operator": [
                {
                    "operator": "ExtractActor",
                    "calls": 1,
                    "total_ms": 200.0,
                    "self_ms": 80.0,
                    "pct_of_total": 100.0,
                }
            ],
            "by_category": {"operator_ms": 200.0, "network_ms": 120.0, "gpu_ms": 0.0, "cpu_ms": 0.0, "io_ms": 0.0},
            "by_model": {"ocr": 120.0},
        },
        "page_summaries": [
            {
                "page_number": 1,
                "source_id": "/data/report.pdf_1",
                "total_ms": 100.0,
                "wall_ms": 100.0,
                "span_count": 2,
                "by_operator": {"ExtractActor": 100.0},
                "by_category": {"operator_ms": 100.0, "network_ms": 60.0, "gpu_ms": 0.0, "cpu_ms": 0.0, "io_ms": 0.0},
            },
            {
                "page_number": 2,
                "source_id": "/data/report.pdf_2",
                "total_ms": 100.0,
                "wall_ms": 100.0,
                "span_count": 2,
                "by_operator": {"ExtractActor": 100.0},
                "by_category": {"operator_ms": 100.0, "network_ms": 60.0, "gpu_ms": 0.0, "cpu_ms": 0.0, "io_ms": 0.0},
            },
        ],
        "spans": spans,
    }


def _second_trace() -> dict[str, Any]:
    """A distinct single-page document so multi-file behavior is exercised."""
    trace = json.loads(json.dumps(_fixture_trace()))
    trace["document"]["document_id"] = "memo.pdf-def67890"
    trace["document"]["source_path"] = "/data/memo.pdf"
    trace["document"]["page_count"] = 1
    trace["page_summaries"] = trace["page_summaries"][:1]
    trace["spans"] = [span for span in trace["spans"] if span["page_number"] == 1]
    for span in trace["spans"]:
        span["document_id"] = "memo.pdf-def67890"
        span["source_path"] = "/data/memo.pdf"
    return trace


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "traces"
    write_document_trace(_fixture_trace(), directory)
    return directory


@pytest.fixture
def multi_trace_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "many"
    write_document_trace(_fixture_trace(), directory)
    write_document_trace(_second_trace(), directory)
    return directory


def test_trace_group_is_registered_on_the_root_app() -> None:
    result = RUNNER.invoke(cli_main.app, ["--help"])
    assert result.exit_code == 0
    assert "trace" in result.output


def test_bare_trace_path_runs_the_summary_report(trace_dir: Path) -> None:
    result = RUNNER.invoke(cli_main.app, ["trace", str(trace_dir)])
    assert result.exit_code == 0, result.output
    assert "Time by operator" in result.output
    assert "ExtractActor" in result.output
    assert "Slowest pages" in result.output
    assert "nvidia/nemoretriever-ocr-v1" in result.output


def test_summary_json_matches_the_rendered_rollup(trace_dir: Path) -> None:
    result = RUNNER.invoke(cli_main.app, ["trace", str(trace_dir), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload["document_count"] == 1
    assert payload["page_count"] == 2
    assert payload["total_ms"] == pytest.approx(200.0)
    assert payload["ms_per_page"] == pytest.approx(100.0)
    assert payload["by_operator"][0]["operator"] == "ExtractActor"
    # The operator category is the denominator for the others, not a peer row.
    categories = {entry["category"] for entry in payload["by_category"]}
    assert "operator" not in categories
    network = next(entry for entry in payload["by_category"] if entry["category"] == "network")
    assert network["pct_of_total"] == pytest.approx(60.0)


def test_summary_accepts_globs_and_multiple_files(multi_trace_dir: Path) -> None:
    result = RUNNER.invoke(cli_main.app, ["trace", str(multi_trace_dir / "*.trace.json"), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["document_count"] == 2
    assert payload["page_count"] == 3
    assert "Documents" not in payload  # rollup keys are lowercase snake_case


def test_summary_top_limits_ranked_rows(multi_trace_dir: Path) -> None:
    result = RUNNER.invoke(cli_main.app, ["trace", str(multi_trace_dir), "--top", "1"])
    assert result.exit_code == 0, result.output
    assert "Slowest pages (top 1)" in result.output


def test_summary_hints_when_only_operator_spans_were_recorded(tmp_path: Path) -> None:
    trace = _fixture_trace()
    trace["document_summary"]["by_category"] = {
        "operator_ms": 200.0,
        "network_ms": 0.0,
        "gpu_ms": 0.0,
        "cpu_ms": 0.0,
        "io_ms": 0.0,
    }
    directory = tmp_path / "operator-only"
    write_document_trace(trace, directory)

    result = RUNNER.invoke(cli_main.app, ["trace", str(directory)])
    assert result.exit_code == 0, result.output
    assert "--page-trace-detail full" in result.output


def test_page_renders_the_span_waterfall(trace_dir: Path) -> None:
    result = RUNNER.invoke(cli_main.app, ["trace", "page", str(trace_dir), "--page", "2"])
    assert result.exit_code == 0, result.output
    assert "Span waterfall" in result.output
    assert "nim.infer" in result.output
    assert "model=ocr" in result.output


def test_page_json_orders_children_under_their_parent(trace_dir: Path) -> None:
    result = RUNNER.invoke(cli_main.app, ["trace", "page", str(trace_dir), "-p", "1", "--json"])
    assert result.exit_code == 0, result.output
    detail = json.loads(result.output)

    assert detail["page"]["page_number"] == 1
    names = [span["name"] for span in detail["spans"]]
    depths = [span["depth"] for span in detail["spans"]]
    assert names == ["ExtractActor", "nim.infer"]
    assert depths == [0, 1]


def test_page_rejects_a_page_the_trace_does_not_contain(trace_dir: Path) -> None:
    result = RUNNER.invoke(cli_main.app, ["trace", "page", str(trace_dir), "--page", "99"])
    assert result.exit_code != 0
    assert "Traced pages" in result.output


def test_page_requires_a_document_when_several_traces_load(multi_trace_dir: Path) -> None:
    ambiguous = RUNNER.invoke(cli_main.app, ["trace", "page", str(multi_trace_dir), "--page", "1"])
    assert ambiguous.exit_code != 0
    assert "--document" in ambiguous.output

    chosen = RUNNER.invoke(
        cli_main.app,
        ["trace", "page", str(multi_trace_dir), "--page", "1", "--document", "memo.pdf-def67890", "--json"],
    )
    assert chosen.exit_code == 0, chosen.output
    assert json.loads(chosen.output)["document"]["document_id"] == "memo.pdf-def67890"


def test_page_accepts_a_source_path_as_the_document_selector(multi_trace_dir: Path) -> None:
    result = RUNNER.invoke(
        cli_main.app,
        ["trace", "page", str(multi_trace_dir), "--page", "1", "-d", "/data/memo.pdf", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["document"]["source_path"] == "/data/memo.pdf"


@pytest.mark.parametrize("extension", ["csv", "jsonl", "parquet"])
def test_export_writes_the_flat_span_table(trace_dir: Path, tmp_path: Path, extension: str) -> None:
    destination = tmp_path / f"spans.{extension}"
    result = RUNNER.invoke(cli_main.app, ["trace", "export", str(trace_dir), "--output", str(destination)])
    assert result.exit_code == 0, result.output
    assert destination.exists()

    if extension == "csv":
        frame = pd.read_csv(destination)
    elif extension == "jsonl":
        frame = pd.read_json(destination, lines=True)
    else:
        frame = pd.read_parquet(destination)

    assert len(frame) == 4
    assert set(frame.page_number) == {1, 2}
    assert "amortized_ms" in frame.columns


def test_export_format_override_wins_over_the_extension(trace_dir: Path, tmp_path: Path) -> None:
    destination = tmp_path / "spans.out"
    result = RUNNER.invoke(
        cli_main.app,
        ["trace", "export", str(trace_dir), "-o", str(destination), "--format", "csv"],
    )
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(destination)) == 4


def test_export_rejects_an_unknown_format(trace_dir: Path, tmp_path: Path) -> None:
    unknown_extension = RUNNER.invoke(
        cli_main.app, ["trace", "export", str(trace_dir), "-o", str(tmp_path / "spans.xyz")]
    )
    assert unknown_extension.exit_code != 0
    assert "Could not infer a format" in unknown_extension.output

    bad_format = RUNNER.invoke(
        cli_main.app,
        ["trace", "export", str(trace_dir), "-o", str(tmp_path / "spans.csv"), "--format", "avro"],
    )
    assert bad_format.exit_code != 0
    assert "Unsupported format" in bad_format.output


def test_export_creates_missing_parent_directories(trace_dir: Path, tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "deeper" / "spans.csv"
    result = RUNNER.invoke(cli_main.app, ["trace", "export", str(trace_dir), "-o", str(destination)])
    assert result.exit_code == 0, result.output
    assert destination.exists()


def test_missing_paths_fail_with_a_clear_message(tmp_path: Path) -> None:
    result = RUNNER.invoke(cli_main.app, ["trace", str(tmp_path / "nope-*.trace.json")])
    assert result.exit_code != 0
    assert "No trace files matched" in result.output


def test_non_trace_json_is_rejected(tmp_path: Path) -> None:
    stray = tmp_path / "config.trace.json"
    stray.write_text(json.dumps({"unrelated": True}), encoding="utf-8")

    result = RUNNER.invoke(cli_main.app, ["trace", str(stray)])
    assert result.exit_code != 0
    assert "page trace artifact" in result.output
