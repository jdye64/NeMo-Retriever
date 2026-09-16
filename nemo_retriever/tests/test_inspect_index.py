# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

lancedb = pytest.importorskip("lancedb")
pa = pytest.importorskip("pyarrow")

from nemo_retriever.inspect import (  # noqa: E402
    render_index_html,
    render_index_pretty,
    summarize_index,
    summary_to_dict,
)


def _write_table(uri: str, table_name: str, *, with_vector: bool = True) -> None:
    fields = []
    if with_vector:
        fields.append(pa.field("vector", pa.list_(pa.float32(), 2)))
    fields.extend(
        [
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("path", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("content_type", pa.string()),
        ]
    )
    schema = pa.schema(
        fields,
        metadata={b"nemo_retriever.embedding_model_name": b"nvidia/test-embed"},
    )
    rows = [
        {
            "text": "Giraffe driving a car at the beach.",
            "metadata": json.dumps({"page_number": 1, "type": "text"}),
            "source": json.dumps({"source_id": "data/multimodal_test.pdf"}),
            "filename": "multimodal_test.pdf",
            "path": "data/multimodal_test.pdf",
            "page_number": 1,
            "content_type": "text",
        },
        {
            "text": "| Animal | Activity |\n| Cat | Jumping |",
            "metadata": json.dumps({"page_number": 1, "_content_type": "table"}),
            "source": json.dumps({"source_id": "data/multimodal_test.pdf"}),
            "filename": "multimodal_test.pdf",
            "path": "data/multimodal_test.pdf",
            "page_number": 1,
            "content_type": "table",
        },
        {
            "text": "Safety checklist for warehouse robots.",
            "metadata": json.dumps({"page_number": 2, "type": "text"}),
            "source": json.dumps({"source_id": "docs/safety.pdf"}),
            "filename": "safety.pdf",
            "path": "docs/safety.pdf",
            "page_number": 2,
            "content_type": "text",
        },
    ]
    if with_vector:
        for index, row in enumerate(rows):
            row["vector"] = [float(index), 1.0]
    lancedb.connect(uri).create_table(table_name, data=rows, schema=schema, mode="overwrite")


def test_summarize_index_groups_documents_and_modalities(tmp_path: Path) -> None:
    uri = str(tmp_path / "lancedb")
    _write_table(uri, "nemo-retriever")

    summary = summarize_index(uri, "nemo-retriever", preview_limit=10, max_text_chars=40)

    assert summary.row_count == 3
    assert summary.document_count == 2
    assert summary.page_count == 2
    assert summary.content_types == {"table": 1, "text": 2}
    assert summary.embedding_model_name == "nvidia/test-embed"
    assert summary.has_vector is True
    assert {document.filename for document in summary.documents} == {"multimodal_test.pdf", "safety.pdf"}
    assert summary.previews[0].text.startswith("Giraffe driving")
    assert summary.previews[0].text.endswith("…") or len(summary.previews[0].text) <= 40


def test_summarize_index_missing_table_lists_available(tmp_path: Path) -> None:
    uri = str(tmp_path / "lancedb")
    _write_table(uri, "docs")

    with pytest.raises(FileNotFoundError, match="Available tables: docs"):
        summarize_index(uri, "nemo-retriever")


def test_render_index_html_includes_search_and_payload(tmp_path: Path) -> None:
    uri = str(tmp_path / "lancedb")
    _write_table(uri, "nemo-retriever")
    html = render_index_html(summarize_index(uri, "nemo-retriever"))

    assert "What did ingest store?" in html
    assert 'id="q"' in html
    assert "multimodal_test.pdf" in html
    assert "Giraffe driving" in html
    assert "<script>" not in json.dumps({"text": "</script>"}) or "<\\/" in html or "Giraffe" in html


def test_render_index_pretty_mentions_gallery_follow_up(tmp_path: Path) -> None:
    uri = str(tmp_path / "lancedb")
    _write_table(uri, "nemo-retriever")
    pretty = render_index_pretty(summarize_index(uri, "nemo-retriever"))

    assert "nemo-retriever" in pretty
    assert "multimodal_test.pdf" in pretty
    assert "retriever inspect --format html --open" in pretty
    assert "retriever query" in pretty


def test_summary_to_dict_is_json_serializable(tmp_path: Path) -> None:
    uri = str(tmp_path / "lancedb")
    _write_table(uri, "nemo-retriever")
    payload = summary_to_dict(summarize_index(uri, "nemo-retriever", preview_limit=1, max_text_chars=12))
    encoded = json.dumps(payload)
    assert "nemo-retriever" in encoded
    assert payload["previews"][0]["text"].endswith("…")
