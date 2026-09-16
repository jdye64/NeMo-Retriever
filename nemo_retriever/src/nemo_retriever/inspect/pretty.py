# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal rendering for ``retriever inspect``."""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nemo_retriever.inspect.summary import IndexSummary


def render_index_pretty(summary: IndexSummary) -> str:
    """Return a Rich-formatted summary of a local Retriever index."""
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=100, color_system=None)

    header = Text.assemble(
        ("Retriever index  ", "bold"),
        (summary.table_name, "bold cyan"),
        (f"  @ {summary.lancedb_uri}", "dim"),
    )
    stats = Table.grid(padding=(0, 2))
    stats.add_row(
        "Rows",
        str(summary.row_count),
        "Documents",
        str(summary.document_count),
        "Pages",
        str(summary.page_count),
    )
    stats.add_row(
        "Mode",
        summary.retrieval_mode,
        "Vector",
        "yes" if summary.has_vector else "no",
        "FTS",
        "yes" if summary.has_fts else "no",
    )
    stats.add_row(
        "Embed",
        summary.embedding_model_name or "(not recorded)",
        "Empty text",
        str(summary.empty_text_count),
        "Types",
        _format_counts(summary.content_types) or "(none)",
    )
    console.print(Panel(stats, title=header, border_style="green"))

    if summary.documents:
        docs = Table(title="Documents", show_lines=False)
        docs.add_column("File", overflow="fold")
        docs.add_column("Pages", justify="right")
        docs.add_column("Chunks", justify="right")
        docs.add_column("Content types", overflow="fold")
        for document in summary.documents:
            docs.add_row(
                document.filename,
                str(document.page_count),
                str(document.chunk_count),
                _format_counts(document.content_types),
            )
        console.print(docs)
    else:
        console.print("No documents in this table.")

    if summary.previews:
        samples = Table(title="Chunk preview")
        samples.add_column("#", justify="right")
        samples.add_column("File")
        samples.add_column("Page", justify="right")
        samples.add_column("Type")
        samples.add_column("Text", overflow="fold")
        for index, preview in enumerate(summary.previews, start=1):
            samples.add_row(
                str(index),
                preview.filename,
                "" if preview.page_number is None else str(preview.page_number),
                preview.content_type,
                preview.text,
            )
        console.print(samples)

    console.print(
        "Open a searchable gallery with `retriever inspect --format html --open`. "
        "Then run `retriever query \"your question\"` against the same table."
    )
    return buffer.getvalue()


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{name} {value}" for name, value in counts.items())
