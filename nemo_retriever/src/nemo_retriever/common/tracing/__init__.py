# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free span collection for inprocess ingest runs."""

from __future__ import annotations

from nemo_retriever.common.tracing.pipeline_trace import (
    PageIndex,
    PageKey,
    PipelineTrace,
    Span,
    UNATTRIBUTED_PAGE,
    UNKNOWN_SOURCE,
    activate_trace,
    active_trace,
    observe_frame,
    page_index,
    page_key,
    page_key_from_row,
    page_keys_from_frame,
    stage_span,
    trace_span,
)
from nemo_retriever.common.tracing.trace_output import (
    DEFAULT_TRACE_DIR,
    new_run_id,
    resolve_trace_dir,
    save_ingest_trace,
    trace_file_path,
)

__all__ = [
    "DEFAULT_TRACE_DIR",
    "PageIndex",
    "PageKey",
    "PipelineTrace",
    "Span",
    "UNATTRIBUTED_PAGE",
    "UNKNOWN_SOURCE",
    "activate_trace",
    "active_trace",
    "new_run_id",
    "observe_frame",
    "page_index",
    "page_key",
    "page_key_from_row",
    "page_keys_from_frame",
    "resolve_trace_dir",
    "save_ingest_trace",
    "stage_span",
    "trace_file_path",
    "trace_span",
]
