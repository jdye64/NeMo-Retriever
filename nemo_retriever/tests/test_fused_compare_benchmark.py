# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fused-versus-staged benchmark helpers.

The measurement loop needs a GPU and the optional fused package, so these
cover the pure logic that decides whether a reported speedup is trustworthy.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import typer

from nemo_retriever.tools.benchmark.fused_compare import (
    OutputSummary,
    PathResult,
    build_embed_params,
    build_extract_params,
    discover_pdfs,
    parity_problems,
    render_report,
    results_payload,
    summarize_output,
)


def _summary(rows: int = 10, embeddings: int = 10, dim: int | None = 2048, **mix: int) -> OutputSummary:
    return OutputSummary(
        rows=rows,
        embeddings=embeddings,
        embedding_dim=dim,
        rows_by_type=dict(mix) or {"text": rows},
    )


def _result(label: str, times: list[float], summary: OutputSummary | None = None) -> PathResult:
    return PathResult(
        label=label,
        method=label,
        model_load_seconds=1.0,
        first_pass_seconds=2.0,
        iterations=times,
        peak_gpu_bytes=1024**3,
        summary=summary if summary is not None else _summary(),
    )


class TestCorpusDiscovery:
    def test_finds_pdfs_recursively_in_sorted_order(self, tmp_path) -> None:
        (tmp_path / "nested").mkdir()
        (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "nested" / "c.pdf").write_bytes(b"%PDF-1.4")

        found = discover_pdfs(tmp_path)

        assert [path.name for path in found] == ["a.pdf", "b.pdf", "c.pdf"]
        assert found == sorted(found)

    def test_ignores_non_pdf_files(self, tmp_path) -> None:
        (tmp_path / "keep.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "skip.txt").write_text("not a pdf")

        assert [path.name for path in discover_pdfs(tmp_path)] == ["keep.pdf"]

    def test_rejects_empty_directory(self, tmp_path) -> None:
        with pytest.raises(typer.BadParameter, match="no PDF files"):
            discover_pdfs(tmp_path)

    def test_rejects_a_file_argument(self, tmp_path) -> None:
        target = tmp_path / "single.pdf"
        target.write_bytes(b"%PDF-1.4")

        with pytest.raises(typer.BadParameter, match="must be a directory"):
            discover_pdfs(target)


class TestPathsDifferOnlyByMethod:
    """A speedup is meaningless if the two paths were configured differently."""

    def test_extract_params_match_except_method(self) -> None:
        shared = {"extract_tables": True, "extract_charts": True, "extract_infographics": False}
        fused = build_extract_params(method="fused", use_table_structure=True, dpi=200, **shared)
        staged = build_extract_params(method="pdfium_hybrid", use_table_structure=True, dpi=200, **shared)

        fused_fields = fused.model_dump()
        staged_fields = staged.model_dump()
        differing = {key for key in fused_fields if fused_fields[key] != staged_fields[key]}

        assert differing == {"method"}

    def test_embed_params_pin_a_model_name_so_warmup_applies(self) -> None:
        """``build_warmup_spec`` only warms the embedder when a model name is set."""
        params = build_embed_params(model_name="nvidia/llama-nemotron-embed-vl-1b-v2", granularity="page")

        assert params.model_name == "nvidia/llama-nemotron-embed-vl-1b-v2"
        assert params.embed_granularity == "page"


class TestOutputSummary:
    def test_counts_rows_and_embeddings_from_columns(self) -> None:
        params = build_embed_params(model_name="m", granularity="page")
        frame = pd.DataFrame(
            {
                params.has_embedding_column: [True, True, False],
                params.embedding_dim_column: [2048, 2048, 0],
                "document_type": ["text", "structured", "text"],
            }
        )

        summary = summarize_output(frame, params)

        assert summary.rows == 3
        assert summary.embeddings == 2
        assert summary.embedding_dim == 2048
        assert summary.rows_by_type == {"text": 2, "structured": 1}

    def test_falls_back_to_metadata_embeddings(self) -> None:
        params = build_embed_params(model_name="m", granularity="page")
        frame = pd.DataFrame({"metadata": [{"embedding": [0.1]}, {"embedding": None}, {}]})

        assert summarize_output(frame, params).embeddings == 1

    def test_reports_no_dimension_when_nothing_embedded(self) -> None:
        params = build_embed_params(model_name="m", granularity="page")
        frame = pd.DataFrame({params.has_embedding_column: [False], params.embedding_dim_column: [0]})

        summary = summarize_output(frame, params)

        assert summary.embeddings == 0
        assert summary.embedding_dim is None


class TestParityChecking:
    def test_identical_output_has_no_problems(self) -> None:
        assert parity_problems(_result("fused", [1.0]), _result("staged", [2.0])) == []

    def test_row_count_mismatch_is_reported(self) -> None:
        problems = parity_problems(
            _result("fused", [1.0], _summary(rows=5)),
            _result("staged", [2.0], _summary(rows=10)),
        )

        assert any("row count differs" in problem for problem in problems)

    def test_missing_embeddings_is_reported(self) -> None:
        """The headline failure mode: fused looks fast because it skipped embedding."""
        problems = parity_problems(
            _result("fused", [1.0], _summary(embeddings=0, dim=None)),
            _result("staged", [2.0], _summary(embeddings=10)),
        )

        assert any("embedding count differs" in problem for problem in problems)
        assert any("embedding dimension differs" in problem for problem in problems)

    def test_element_mix_mismatch_is_reported(self) -> None:
        problems = parity_problems(
            _result("fused", [1.0], _summary(text=10)),
            _result("staged", [2.0], _summary(text=8, structured=2)),
        )

        assert any("element mix differs" in problem for problem in problems)

    def test_a_failed_path_is_never_comparable(self) -> None:
        broken = PathResult(label="fused", method="fused", error="boom")

        assert parity_problems(broken, _result("staged", [2.0])) != []


class TestReport:
    def test_reports_speedup_when_fused_is_faster(self) -> None:
        report = render_report(
            [_result("fused", [1.0, 1.0, 1.0]), _result("staged", [2.0, 2.0, 2.0])],
            documents=2,
            pages=10,
            run_mode="inprocess",
            granularity="page",
            iterations=3,
        )

        assert "2.00x faster" in report
        assert "both paths agree" in report

    def test_reports_slower_without_pretending_otherwise(self) -> None:
        report = render_report(
            [_result("fused", [4.0, 4.0]), _result("staged", [2.0, 2.0])],
            documents=1,
            pages=4,
            run_mode="inprocess",
            granularity="page",
            iterations=2,
        )

        assert "0.50x slower" in report

    def test_flags_unstable_iterations(self) -> None:
        report = render_report(
            [_result("fused", [1.0, 5.0, 3.0]), _result("staged", [2.0, 2.0, 2.0])],
            documents=1,
            pages=4,
            run_mode="inprocess",
            granularity="page",
            iterations=3,
        )

        assert "varied by" in report

    def test_surfaces_parity_failure_instead_of_a_speedup_claim(self) -> None:
        report = render_report(
            [
                _result("fused", [1.0], _summary(embeddings=0, dim=None)),
                _result("staged", [2.0], _summary(embeddings=10)),
            ],
            documents=1,
            pages=4,
            run_mode="inprocess",
            granularity="page",
            iterations=1,
        )

        assert "NOT COMPARABLE" in report

    def test_failed_path_is_shown(self) -> None:
        report = render_report(
            [PathResult(label="fused", method="fused", error="ImportError: missing"), _result("staged", [2.0])],
            documents=1,
            pages=4,
            run_mode="inprocess",
            granularity="page",
            iterations=1,
        )

        assert "FAILED" in report


class TestJsonPayload:
    def test_payload_is_serializable_and_carries_the_speedup(self) -> None:
        payload = results_payload(
            [_result("fused", [1.0, 1.0]), _result("staged", [2.0, 2.0])],
            documents=2,
            pages=10,
            run_mode="inprocess",
            granularity="page",
        )

        assert payload["speedup_fused_over_staged"] == pytest.approx(2.0)
        assert payload["corpus"] == {"documents": 2, "pages": 10}
        assert payload["parity_problems"] == []
        assert {entry["label"] for entry in payload["paths"]} == {"fused", "staged"}
        json.dumps(payload)

    def test_payload_records_pages_per_second(self) -> None:
        payload = results_payload(
            [_result("fused", [2.0])],
            documents=1,
            pages=10,
            run_mode="inprocess",
            granularity="page",
        )

        assert payload["paths"][0]["pages_per_second"] == pytest.approx(5.0)
