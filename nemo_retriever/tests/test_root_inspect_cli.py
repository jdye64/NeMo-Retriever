# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

lancedb = pytest.importorskip("lancedb")
pa = pytest.importorskip("pyarrow")

RUNNER = CliRunner()
cli_main = importlib.import_module("nemo_retriever.cli.main")


def _write_table(uri: str, table_name: str) -> None:
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("path", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("content_type", pa.string()),
        ]
    )
    rows = [
        {
            "vector": [0.0, 1.0],
            "text": "Giraffe driving a car at the beach.",
            "metadata": json.dumps({"page_number": 1, "type": "text"}),
            "source": json.dumps({"source_id": "data/multimodal_test.pdf"}),
            "filename": "multimodal_test.pdf",
            "path": "data/multimodal_test.pdf",
            "page_number": 1,
            "content_type": "text",
        },
        {
            "vector": [1.0, 1.0],
            "text": "Safety checklist for warehouse robots.",
            "metadata": json.dumps({"page_number": 2, "type": "text"}),
            "source": json.dumps({"source_id": "docs/safety.pdf"}),
            "filename": "safety.pdf",
            "path": "docs/safety.pdf",
            "page_number": 2,
            "content_type": "text",
        },
    ]
    lancedb.connect(uri).create_table(table_name, data=rows, schema=schema, mode="overwrite")


def test_root_help_lists_inspect() -> None:
    result = RUNNER.invoke(cli_main.app, ["--help"])

    assert result.exit_code == 0
    assert "inspect" in result.output
    assert "ingest" in result.output
    assert "query" in result.output


def test_inspect_pretty_prints_documents(tmp_path: Path) -> None:
    uri = str(tmp_path / "lancedb")
    _write_table(uri, "nemo-retriever")

    result = RUNNER.invoke(cli_main.app, ["inspect", "--lancedb-uri", uri])

    assert result.exit_code == 0, result.output
    assert "multimodal_test.pdf" in result.output
    assert "safety.pdf" in result.output
    assert "text 2" in result.output


def test_inspect_json_format(tmp_path: Path) -> None:
    uri = str(tmp_path / "lancedb")
    _write_table(uri, "nemo-retriever")

    result = RUNNER.invoke(
        cli_main.app,
        ["inspect", "--lancedb-uri", uri, "--format", "json", "--preview-limit", "1"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["table_name"] == "nemo-retriever"
    assert payload["document_count"] == 2
    assert len(payload["previews"]) == 1


def test_inspect_html_requires_output_or_open(tmp_path: Path) -> None:
    uri = str(tmp_path / "lancedb")
    _write_table(uri, "nemo-retriever")

    result = RUNNER.invoke(cli_main.app, ["inspect", "--lancedb-uri", uri, "--format", "html"])

    assert result.exit_code == 1
    assert "--output or --open" in result.output


def test_inspect_writes_html(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uri = str(tmp_path / "lancedb")
    html_path = tmp_path / "gallery.html"
    _write_table(uri, "nemo-retriever")
    opened: list[str] = []
    monkeypatch.setattr("nemo_retriever.cli.inspect.app.webbrowser.open", opened.append)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "inspect",
            "--lancedb-uri",
            uri,
            "--format",
            "html",
            "--output",
            str(html_path),
            "--open",
        ],
    )

    assert result.exit_code == 0, result.output
    assert html_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    assert "What did ingest store?" in html_path.read_text(encoding="utf-8")
    assert opened and opened[0].startswith("file:")


def test_inspect_list_tables(tmp_path: Path) -> None:
    uri = str(tmp_path / "lancedb")
    _write_table(uri, "docs")

    result = RUNNER.invoke(cli_main.app, ["inspect", "--lancedb-uri", uri, "--list-tables"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "docs"


def test_inspect_missing_table(tmp_path: Path) -> None:
    uri = str(tmp_path / "lancedb")
    _write_table(uri, "docs")

    result = RUNNER.invoke(cli_main.app, ["inspect", "--lancedb-uri", uri, "--table-name", "missing"])

    assert result.exit_code == 1
    assert "Available tables: docs" in result.output
