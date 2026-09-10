# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where and how a run's :class:`PipelineTrace` is written to disk.

One ingest job produces one JSONL file. Keeping the naming and directory
policy here means the CLI, the SDK, and tests all agree on where traces land
and what a saved trace is called.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from nemo_retriever.common.tracing.pipeline_trace import PipelineTrace

#: Directory used when tracing is requested without an explicit destination.
#: Listed in the repository ``.gitignore`` so saved traces are never committed.
DEFAULT_TRACE_DIR = ".ingest_traces"

_UNSAFE_RUN_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def resolve_trace_dir(trace_dir: str | os.PathLike[str] | None) -> Path:
    """Return the directory traces are written to, expanding ``~``."""
    return Path(trace_dir if trace_dir else DEFAULT_TRACE_DIR).expanduser()


def new_run_id(*, now: datetime | None = None) -> str:
    """Return a sortable, filesystem-safe identifier for one ingest job."""
    moment = now or datetime.now(timezone.utc)
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"


def trace_file_path(run_id: str, *, trace_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the JSONL path for one run inside the resolved trace directory."""
    safe_run_id = _UNSAFE_RUN_ID_CHARS.sub("-", run_id).strip("-") or "run"
    return resolve_trace_dir(trace_dir) / f"ingest-trace-{safe_run_id}.jsonl"


def save_ingest_trace(
    trace: PipelineTrace,
    *,
    trace_dir: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    documents: int | None = None,
) -> Path:
    """Write one job's trace as JSONL and return the file that was written.

    Parameters
    ----------
    trace
        The payload returned by ``ingest(return_traces=True)``.
    trace_dir
        Destination directory. Defaults to :data:`DEFAULT_TRACE_DIR`.
    run_id
        Identifier stamped on every record. Generated when omitted.
    documents
        Number of input documents in the job, stamped on every record so
        traces from several jobs can be concatenated and still compared.
    """
    resolved_run_id = run_id or new_run_id()
    run_fields: dict[str, object] = {"run_id": resolved_run_id}
    if documents is not None:
        run_fields["n_documents"] = int(documents)
    return trace.save_jsonl(trace_file_path(resolved_run_id, trace_dir=trace_dir), **run_fields)
