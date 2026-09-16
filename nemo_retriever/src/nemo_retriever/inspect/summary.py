# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize a local LanceDB table without loading embedding vectors."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from nemo_retriever.common.vdb.lancedb_capabilities import inspect_lancedb_table_object
from nemo_retriever.common.vdb.records import normalize_content_type

_EMBEDDING_MODEL_METADATA_KEY = b"nemo_retriever.embedding_model_name"
_UNKNOWN_SOURCE = "(unknown)"


@dataclass(frozen=True)
class ChunkPreview:
    source: str
    filename: str
    page_number: int | None
    content_type: str
    text: str


@dataclass(frozen=True)
class DocumentSummary:
    source: str
    filename: str
    page_count: int
    chunk_count: int
    content_types: dict[str, int]


@dataclass(frozen=True)
class IndexSummary:
    lancedb_uri: str
    table_name: str
    row_count: int
    document_count: int
    page_count: int
    retrieval_mode: str
    embedding_model_name: str | None
    has_vector: bool
    has_fts: bool
    content_types: dict[str, int]
    empty_text_count: int
    documents: list[DocumentSummary]
    previews: list[ChunkPreview]
    available_tables: list[str] = field(default_factory=list)


def summary_to_dict(summary: IndexSummary) -> dict[str, Any]:
    """Return a JSON-serializable mapping of ``summary``."""
    return asdict(summary)


def summarize_index(
    lancedb_uri: str,
    table_name: str,
    *,
    preview_limit: int = 50,
    max_text_chars: int = 280,
) -> IndexSummary:
    """Scan a LanceDB table and return document, modality, and preview stats.

    Embedding / vector columns are dropped before the scan so inspect stays
    usable on local GPU-hosted indexes.
    """
    if preview_limit < 0:
        raise ValueError("preview_limit must be greater than or equal to 0.")
    if max_text_chars < 0:
        raise ValueError("max_text_chars must be greater than or equal to 0.")

    import lancedb

    db = lancedb.connect(lancedb_uri)
    available_tables = _table_names(db)
    if table_name not in available_tables:
        available = ", ".join(available_tables) if available_tables else "(none)"
        raise FileNotFoundError(
            f"LanceDB table {table_name!r} was not found at {lancedb_uri!r}. "
            f"Available tables: {available}."
        )

    table = db.open_table(table_name)
    capabilities = inspect_lancedb_table_object(table)
    schema = table.schema() if callable(table.schema) else table.schema
    embedding_model_name = _embedding_model_name(schema)
    drop_columns = {capabilities.vector_column} if capabilities.vector_column else set()
    rows = _load_rows_without_vectors(table, drop_columns=drop_columns)

    type_counts: Counter[str] = Counter()
    empty_text_count = 0
    pages_by_source: dict[str, set[int | None]] = defaultdict(set)
    types_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    filenames: dict[str, str] = {}
    previews: list[ChunkPreview] = []

    for row in rows:
        source, filename = _source_and_filename(row)
        page_number = _page_number(row)
        content_type = _content_type(row)
        text = _text(row)
        type_counts[content_type] += 1
        types_by_source[source][content_type] += 1
        pages_by_source[source].add(page_number)
        filenames.setdefault(source, filename)
        if not text.strip():
            empty_text_count += 1
        if len(previews) < preview_limit:
            previews.append(
                ChunkPreview(
                    source=source,
                    filename=filename,
                    page_number=page_number,
                    content_type=content_type,
                    text=_truncate(text, max_text_chars),
                )
            )

    documents = [
        DocumentSummary(
            source=source,
            filename=filenames.get(source, Path(source).name or _UNKNOWN_SOURCE),
            page_count=len({page for page in pages if page is not None}),
            chunk_count=sum(types_by_source[source].values()),
            content_types=dict(sorted(types_by_source[source].items())),
        )
        for source, pages in sorted(pages_by_source.items(), key=lambda item: item[0])
    ]
    unique_pages = {
        (source, page)
        for source, pages in pages_by_source.items()
        for page in pages
        if page is not None
    }

    return IndexSummary(
        lancedb_uri=lancedb_uri,
        table_name=table_name,
        row_count=len(rows),
        document_count=len(documents),
        page_count=len(unique_pages),
        retrieval_mode=capabilities.retrieval_mode,
        embedding_model_name=embedding_model_name,
        has_vector=capabilities.has_vector,
        has_fts=capabilities.has_fts,
        content_types=dict(sorted(type_counts.items())),
        empty_text_count=empty_text_count,
        documents=documents,
        previews=previews,
        available_tables=available_tables,
    )


def _table_names(db: Any) -> list[str]:
    list_tables = getattr(db, "list_tables", None)
    if callable(list_tables):
        listing = list_tables()
        tables = getattr(listing, "tables", None)
        if tables is not None:
            return sorted(str(name) for name in tables)
    names = db.table_names()
    return sorted(str(name) for name in names)


def _embedding_model_name(schema: Any) -> str | None:
    metadata = getattr(schema, "metadata", None) or {}
    raw = metadata.get(_EMBEDDING_MODEL_METADATA_KEY)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        value = raw.decode("utf-8", errors="replace").strip()
    else:
        value = str(raw).strip()
    return value or None


def _load_rows_without_vectors(table: Any, *, drop_columns: set[str]) -> list[dict[str, Any]]:
    schema = table.schema() if callable(table.schema) else table.schema
    keep = [field.name for field in schema if field.name not in drop_columns]
    lance_ds = getattr(table, "to_lance", None)
    if callable(lance_ds):
        try:
            arrow = lance_ds().to_table(columns=keep)
            return list(arrow.to_pylist())
        except Exception:
            pass
    to_arrow = getattr(table, "to_arrow", None)
    if callable(to_arrow):
        arrow = to_arrow()
        names = [name for name in arrow.column_names if name not in drop_columns]
        return list(arrow.select(names).to_pylist())
    frame = table.to_pandas()
    drop = [column for column in drop_columns if column in frame.columns]
    if drop:
        frame = frame.drop(columns=drop)
    return list(frame.to_dict(orient="records"))


def _source_and_filename(row: Mapping[str, Any]) -> tuple[str, str]:
    filename = _first_str(row.get("filename"), row.get("pdf_basename"))
    path = _first_str(row.get("path"), row.get("source_id"))
    source_payload = _decode_json_maybe(row.get("source"))
    if isinstance(source_payload, Mapping):
        path = _first_str(source_payload.get("source_id"), path)
    source = path or filename or _UNKNOWN_SOURCE
    if not filename:
        filename = Path(source).name or source
    return source, filename


def _page_number(row: Mapping[str, Any]) -> int | None:
    metadata = _decode_json_maybe(row.get("metadata"))
    candidates = [row.get("page_number")]
    if isinstance(metadata, Mapping):
        candidates.append(metadata.get("page_number"))
    for value in candidates:
        parsed = _optional_int(value)
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def _content_type(row: Mapping[str, Any]) -> str:
    metadata = _decode_json_maybe(row.get("metadata"))
    candidates = [row.get("content_type")]
    if isinstance(metadata, Mapping):
        candidates.extend(
            [
                metadata.get("_content_type"),
                metadata.get("type"),
            ]
        )
    for value in candidates:
        normalized = normalize_content_type(value)
        if normalized:
            return normalized
    return "unknown"


def _text(row: Mapping[str, Any]) -> str:
    value = row.get("text")
    return value if isinstance(value, str) else ""


def _truncate(text: str, max_text_chars: int) -> str:
    collapsed = " ".join(text.split())
    if max_text_chars == 0:
        return ""
    if len(collapsed) <= max_text_chars:
        return collapsed
    return collapsed[:max_text_chars] + "…"


def _decode_json_maybe(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "{[":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_str(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
