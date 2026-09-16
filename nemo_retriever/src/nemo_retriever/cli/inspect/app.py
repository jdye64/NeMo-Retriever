# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional

import typer

from nemo_retriever.cli.query import options as query_opts
from nemo_retriever.cli.shared import ROOT_CLI_ERRORS
from nemo_retriever.inspect import (
    render_index_html,
    render_index_pretty,
    summarize_index,
    summary_to_dict,
)

_FORMATS = ("pretty", "json", "html")


def inspect_command(
    lancedb_uri: query_opts.LanceDbUriOption = "lancedb",
    table_name: query_opts.TableNameOption = "nemo-retriever",
    output_format: str = typer.Option(
        "pretty",
        "--format",
        help="pretty (default): terminal summary. json: machine-readable stats. html: searchable gallery.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write HTML to this path. Required for --format html unless --open is set.",
    ),
    open_gallery: bool = typer.Option(
        False,
        "--open",
        help="Write an HTML gallery and open it in your default browser.",
    ),
    preview_limit: int = typer.Option(
        50,
        "--preview-limit",
        min=0,
        help="Maximum number of chunk previews to include.",
    ),
    max_text_chars: int = typer.Option(
        280,
        "--max-text-chars",
        min=0,
        help="Truncate each preview to this many characters.",
    ),
    list_tables: bool = typer.Option(
        False,
        "--list-tables",
        help="Print table names at --lancedb-uri and exit.",
    ),
) -> None:
    """Browse a local LanceDB index produced by retriever ingest."""
    if output_format not in _FORMATS:
        typer.echo(
            f"Error: unknown --format {output_format!r} (use 'pretty', 'json', or 'html').",
            err=True,
        )
        raise typer.Exit(1)
    if output_format == "html" and output is None and not open_gallery:
        typer.echo("Error: --format html requires --output or --open.", err=True)
        raise typer.Exit(1)
    if output is not None and output_format != "html":
        typer.echo("Error: --output is only valid with --format html.", err=True)
        raise typer.Exit(1)

    try:
        if list_tables:
            _print_tables(lancedb_uri)
            return
        summary = summarize_index(
            lancedb_uri,
            table_name,
            preview_limit=preview_limit,
            max_text_chars=max_text_chars,
        )
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except ROOT_CLI_ERRORS as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if output_format == "json" and not open_gallery:
        typer.echo(json.dumps(summary_to_dict(summary), indent=2, sort_keys=True))
        return

    if output_format == "html" or open_gallery:
        html = render_index_html(summary)
        html_path = output
        if html_path is None:
            handle = tempfile.NamedTemporaryFile(
                prefix="retriever-inspect-",
                suffix=".html",
                delete=False,
            )
            html_path = Path(handle.name)
            handle.write(html.encode("utf-8"))
            handle.close()
        else:
            html_path.write_text(html, encoding="utf-8")
        typer.echo(str(html_path))
        if open_gallery:
            webbrowser.open(html_path.resolve().as_uri())
        return

    typer.echo(render_index_pretty(summary), nl=False)


def _print_tables(lancedb_uri: str) -> None:
    import lancedb

    names = sorted(str(name) for name in lancedb.connect(lancedb_uri).table_names())
    if not names:
        typer.echo(f"No LanceDB tables found at {lancedb_uri!r}.")
        return
    for name in names:
        typer.echo(name)
