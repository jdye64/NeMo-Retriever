# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read page trace artifacts back into Python and pandas.

The ``spans`` array in a trace document is deliberately flat, so
:func:`spans_dataframe` is a thin wrapper over ``pandas.json_normalize``. Sum
``amortized_ms`` rather than ``duration_ms``: a batched operator span reports
its exact measured duration on every page it covered, while the amortized
value divides that duration across those pages and therefore stays additive.
"""

from __future__ import annotations

from glob import glob
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

TRACE_GLOB_SUFFIXES = ("*.trace.json", "*.trace.json.gz")


class TraceFileError(ValueError):
    """Raised when a path is not a readable page trace artifact."""


def load_trace_file(path: str | Path) -> dict[str, Any]:
    """Load one trace document, transparently handling gzip."""
    file_path = Path(path).expanduser()
    try:
        if file_path.suffix == ".gz":
            with gzip.open(file_path, "rb") as handle:
                payload = json.loads(handle.read().decode("utf-8"))
        else:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TraceFileError(f"Trace file not found: {file_path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceFileError(f"Could not read trace file {file_path}: {exc}") from exc

    if not isinstance(payload, dict) or "spans" not in payload:
        raise TraceFileError(
            f"{file_path} is not a NeMo Retriever page trace artifact "
            "(expected a JSON object with a 'spans' array)."
        )
    return payload


def expand_trace_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Resolve files, directories, and globs into a sorted list of trace files."""
    resolved: list[Path] = []
    seen: set[Path] = set()

    def _append(candidate: Path) -> None:
        if candidate not in seen:
            seen.add(candidate)
            resolved.append(candidate)

    for entry in paths:
        candidate = Path(str(entry)).expanduser()
        if candidate.is_dir():
            for suffix in TRACE_GLOB_SUFFIXES:
                for match in sorted(candidate.glob(suffix)):
                    _append(match)
            continue
        if candidate.is_file():
            _append(candidate)
            continue
        matches = sorted(glob(str(candidate), recursive=True))
        if not matches:
            raise TraceFileError(f"No trace files matched: {entry}")
        for match in matches:
            _append(Path(match))

    if not resolved:
        raise TraceFileError("No trace files were found.")
    return resolved


def load_traces(paths: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load every trace document addressed by *paths*.

    Accepts a single file, a directory of trace files, a glob, or a sequence of
    any of those.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return [load_trace_file(path) for path in expand_trace_paths(paths)]


def spans_dataframe(traces: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Return every span across *traces* as one flat DataFrame."""
    records: list[dict[str, Any]] = []
    for trace in traces:
        run = trace.get("run") or {}
        library = trace.get("nemo_retriever") or {}
        for span in trace.get("spans") or []:
            record = dict(span)
            record.setdefault("run_id", run.get("run_id"))
            record.setdefault("run_mode", run.get("run_mode"))
            record.setdefault("library_version", library.get("version"))
            records.append(record)
    if not records:
        return pd.DataFrame(
            columns=[
                "document_id",
                "source_path",
                "page_number",
                "span_id",
                "name",
                "category",
                "operator",
                "duration_ms",
                "amortized_ms",
            ]
        )
    return pd.json_normalize(records)


def page_summaries_dataframe(traces: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Return per-page rollups across *traces* as one flat DataFrame."""
    records: list[dict[str, Any]] = []
    for trace in traces:
        document = trace.get("document") or {}
        for summary in trace.get("page_summaries") or []:
            record = {
                "document_id": document.get("document_id"),
                "source_path": document.get("source_path"),
                "page_number": summary.get("page_number"),
                "source_id": summary.get("source_id"),
                "total_ms": summary.get("total_ms"),
                "wall_ms": summary.get("wall_ms"),
                "span_count": summary.get("span_count"),
            }
            for key, value in (summary.get("by_category") or {}).items():
                record[f"category.{key}"] = value
            for key, value in (summary.get("by_operator") or {}).items():
                record[f"operator.{key}"] = value
            records.append(record)
    if not records:
        return pd.DataFrame(columns=["document_id", "page_number", "total_ms", "wall_ms"])
    return pd.DataFrame(records)
