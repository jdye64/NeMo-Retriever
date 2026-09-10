#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingest a directory with page tracing on and report where the time went.

Run this against a corpus to find the pages that dominate a slow ingest::

    python scripts/trace_ingest.py /path/to/pdfs

The run mode decides whether per-page timings can distinguish pages at all.
``batch`` hands each operator a single page, so every page timing is measured
on that page. ``inprocess`` hands each operator the whole document at once, so
one duration covers every page and gets divided evenly among them, leaving the
per-page timings identical. The report labels which case each page is in, and
ranks pages by work volume as well, which is counted per page in both modes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from rich.console import Console

from nemo_retriever import create_ingestor
from nemo_retriever.cli.trace import report
from nemo_retriever.common.tracing import TraceFileError, load_traces


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a directory with page tracing enabled and summarize the traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/trace_ingest.py ~/datasets/bo20\n"
            "  python scripts/trace_ingest.py ~/datasets/bo20 --run-mode inprocess\n"
            "  python scripts/trace_ingest.py ~/datasets/bo20 --trace-dir /tmp/t --keep-traces --top 25\n"
        ),
    )
    parser.add_argument(
        "input",
        help="Directory of documents to ingest, a single file, or a glob.",
    )
    parser.add_argument(
        "--trace-dir",
        default="traces",
        help="Directory to write {document_id}.trace.json files into. Default: traces",
    )
    parser.add_argument(
        "--run-mode",
        choices=("batch", "inprocess"),
        default="batch",
        help=(
            "batch gives measured per-page timings, one page per operator call. "
            "inprocess processes the whole document at once, so per-page timings "
            "become batch averages. Default: batch"
        ),
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Also run the embedding stage. Requires an embedding model to be reachable.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Rows to show in each ranked table. Default: 10",
    )
    parser.add_argument(
        "--keep-traces",
        action="store_true",
        help="Keep any trace files already in --trace-dir instead of clearing it first.",
    )
    parser.add_argument(
        "--allow-no-gpu",
        action="store_true",
        help="Permit the run to proceed on a host without a usable GPU.",
    )
    return parser.parse_args(argv)


def _resolve_input(raw: str) -> str | None:
    """Return an ingestable path spec, or ``None`` when the path is missing.

    ``.files()`` takes files and globs, not bare directories, so a directory is
    expanded into a recursive glob rather than rejected.
    """
    if any(character in raw for character in "*?["):
        return raw
    path = Path(raw).expanduser()
    if path.is_dir():
        return str(path / "**" / "*")
    return str(path) if path.exists() else None


def _prepare_trace_dir(trace_dir: Path, *, keep: bool) -> None:
    """Clear stale traces so the report describes only this run."""
    if trace_dir.exists() and not keep:
        for stale in (*trace_dir.glob("*.trace.json"), *trace_dir.glob("*.trace.json.gz")):
            stale.unlink()
    trace_dir.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    console = Console()

    source = _resolve_input(args.input)
    if source is None:
        console.print(f"[red]Input path does not exist:[/red] {Path(args.input).expanduser()}")
        return 2

    trace_dir = Path(args.trace_dir).expanduser()
    _prepare_trace_dir(trace_dir, keep=args.keep_traces)

    console.print(f"Ingesting [bold]{source}[/bold] in [bold]{args.run_mode}[/bold] mode.")
    if args.run_mode == "inprocess":
        console.print(
            "[yellow]inprocess mode processes each document in one call per operator, so "
            "per-page timings will be batch averages.[/yellow] Work counters stay per page."
        )

    ingestor = create_ingestor(run_mode=args.run_mode, allow_no_gpu=args.allow_no_gpu)
    ingestor = ingestor.files(source).extract()
    if args.embed:
        ingestor = ingestor.embed()
    ingestor = ingestor.save_page_traces(output_directory=str(trace_dir))

    started = time.perf_counter()
    try:
        results = ingestor.ingest()
    except Exception as exc:  # Surface the failure without a traceback wall.
        console.print(f"[red]Ingest failed:[/red] {type(exc).__name__}: {exc}")
        return 1
    elapsed_s = time.perf_counter() - started

    console.print(f"Ingest finished in [bold]{elapsed_s:,.1f} s[/bold] producing {len(results):,} rows.")

    try:
        traces = load_traces(trace_dir)
    except TraceFileError as exc:
        console.print(f"[red]No traces to report:[/red] {exc}")
        console.print("Tracing may have been disabled via NEMO_RETRIEVER_PAGE_TRACE_DETAIL=off.")
        return 1

    console.print(f"Wrote {len(traces)} trace file(s) to [bold]{trace_dir}[/bold].\n")
    report.render_summary(console, report.summarize(traces), top=args.top)

    console.print(
        f"\nRe-render this report with: [bold]retriever trace {trace_dir}[/bold]\n"
        f"Inspect one page with:      [bold]retriever trace page {trace_dir} "
        "--page N --document DOCUMENT_ID[/bold]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
