# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``retriever trace`` — analyze page trace artifacts produced by ingest.

``retriever trace FILES...`` runs the ``summary`` report, mirroring the
default-subcommand behavior of ``retriever ingest``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import click
import typer
from rich.console import Console
from typer.core import TyperCommand, TyperGroup

from nemo_retriever.cli.trace import report
from nemo_retriever.common.tracing import TraceFileError, load_traces, spans_dataframe

_DEFAULT_COMMAND = "summary"
_GROUP_OPTIONS = {"-h", "--help"}

_EXPORT_FORMATS = ("parquet", "csv", "jsonl")
_EXPORT_SUFFIXES = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".json": "jsonl",
}


class DefaultSummaryTraceGroup(TyperGroup):
    """Treat a leading non-command argument as a path for ``summary``."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in _GROUP_OPTIONS:
            args = [_DEFAULT_COMMAND, *args]
        return super().parse_args(ctx, args)


class PublicDefaultTraceContext(typer.Context):
    @property
    def command_path(self) -> str:
        return self.parent.command_path if self.parent is not None else super().command_path


class DefaultSummaryTraceCommand(TyperCommand):
    context_class = PublicDefaultTraceContext


app = typer.Typer(
    cls=DefaultSummaryTraceGroup,
    help=(
        "Analyze per-page pipeline traces written by ingest.\n\n"
        "Use retriever trace FILES... for the default summary report. FILES accepts "
        "trace files, directories of traces, and globs.\n\n"
        "Produce traces with retriever ingest --page-trace-dir DIR, or from Python with "
        "Ingestor.save_page_traces(DIR)."
    ),
    no_args_is_help=True,
)

_FILES_ARGUMENT = typer.Argument(
    ...,
    metavar="FILES...",
    help="Trace files, directories containing *.trace.json[.gz], or globs.",
)


_PIPED_WIDTH = 120


def _console(*, to_stderr: bool = False) -> Console:
    console = Console(stderr=to_stderr)
    if not console.is_terminal:
        # Rich falls back to 80 columns when the output is redirected, which
        # truncates operator names and endpoints out of the tables.
        console = Console(stderr=to_stderr, width=_PIPED_WIDTH)
    return console


def _load(files: List[str]) -> list[dict[str, Any]]:
    try:
        traces = load_traces(files)
    except TraceFileError as exc:
        raise typer.BadParameter(str(exc), param_hint="FILES") from exc
    if not traces:
        raise typer.BadParameter("No trace documents were loaded.", param_hint="FILES")
    return traces


def _emit_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command(
    "summary",
    cls=DefaultSummaryTraceCommand,
    help=(
        "Report where pipeline time went across one or more traces: totals by "
        "operator, model versions, and the slowest pages."
    ),
)
def summary_command(
    files: List[str] = _FILES_ARGUMENT,
    top: int = typer.Option(10, "--top", "-n", min=1, help="Rows to show in the ranked tables."),
    as_json: bool = typer.Option(False, "--json", help="Emit the rollup as JSON instead of tables."),
) -> None:
    traces = _load(files)
    rollup = report.summarize(traces)
    if as_json:
        _emit_json(rollup)
        return
    report.render_summary(_console(), rollup, top=top)


@app.command(
    "page",
    help=(
        "Show the span waterfall for a single page, including per-call model and "
        "endpoint detail. Pass --document when several traces are loaded."
    ),
)
def page_command(
    files: List[str] = _FILES_ARGUMENT,
    page: int = typer.Option(..., "--page", "-p", min=1, help="Page number to inspect (1-based)."),
    document: Optional[str] = typer.Option(
        None,
        "--document",
        "-d",
        help="Document id or source path to inspect when the paths hold several traces.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the page detail as JSON instead of tables."),
) -> None:
    traces = _load(files)
    try:
        trace = report.resolve_trace_for_document(traces, document)
        detail = report.page_detail(trace, page)
    except KeyError as exc:
        # KeyError stringifies with quotes, which reads badly in CLI output.
        raise typer.BadParameter(exc.args[0] if exc.args else str(exc)) from exc
    if as_json:
        _emit_json(detail)
        return
    report.render_page(_console(), detail)


@app.command(
    "export",
    help=(
        "Write the flat span table from one or more traces to Parquet, CSV, or JSONL "
        "for analysis in pandas or a notebook."
    ),
)
def export_command(
    files: List[str] = _FILES_ARGUMENT,
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Destination file. Format is inferred from the extension unless --format is given.",
    ),
    export_format: Optional[str] = typer.Option(
        None,
        "--format",
        help=f"Override the output format. One of: {', '.join(_EXPORT_FORMATS)}.",
    ),
) -> None:
    traces = _load(files)
    resolved_format = _resolve_export_format(output, export_format)
    frame = spans_dataframe(traces)

    destination = output.expanduser()
    if destination.parent != Path(""):
        destination.parent.mkdir(parents=True, exist_ok=True)

    if resolved_format == "parquet":
        try:
            frame.to_parquet(destination, index=False)
        except (ImportError, ValueError) as exc:
            raise typer.BadParameter(
                f"Could not write Parquet ({exc}). Install pyarrow, or export with --format csv.",
                param_hint="--output",
            ) from exc
    elif resolved_format == "csv":
        frame.to_csv(destination, index=False)
    else:
        # Attribute dicts do not survive a CSV round-trip, so JSONL keeps them.
        frame.to_json(destination, orient="records", lines=True)

    console = _console(to_stderr=True)
    console.print(f"Wrote {len(frame):,} spans from {len(traces)} trace(s) to {destination} ({resolved_format}).")


def _resolve_export_format(output: Path, requested: str | None) -> str:
    if requested:
        candidate = requested.strip().lower()
        if candidate not in _EXPORT_FORMATS:
            raise typer.BadParameter(
                f"Unsupported format '{requested}'. Choose one of: {', '.join(_EXPORT_FORMATS)}.",
                param_hint="--format",
            )
        return candidate
    inferred = _EXPORT_SUFFIXES.get(output.suffix.lower())
    if inferred is None:
        raise typer.BadParameter(
            f"Could not infer a format from '{output.name}'. "
            f"Use a {'/'.join(_EXPORT_FORMATS)} extension or pass --format.",
            param_hint="--output",
        )
    return inferred
