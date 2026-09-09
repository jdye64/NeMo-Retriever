# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-page pipeline tracing.

Every operator invocation is timed and recorded against the pages it touched,
producing one document-level JSON artifact per source document that retains
per-page detail.

How sharp that per-page detail is depends on how the run batched its work. One
operator span covers a whole batch, and its cost is divided evenly across those
pages, so per-page timings only distinguish pages when a span covered a single
page. ``run_mode='batch'`` does that by default, while ``inprocess`` and
``service`` hand each operator the whole document at once. Alongside timings,
operators record per-page ``work`` counters (crops, detections, characters) via
:func:`record_page_work`; those are measured per page in every run mode, so an
expensive page stays identifiable even when its timings are batch averages.

This package is independent of the legacy control-message tracing under
``common/api/internal/primitives/tracing``, which the current operator graph
does not use.

Typical use from Python::

    from nemo_retriever.common.tracing import load_traces, page_summaries_dataframe

    pages = page_summaries_dataframe(load_traces("traces/"))
    pages.nlargest(10, "work.OCRActor.crops")
"""

from __future__ import annotations

from nemo_retriever.common.tracing.aggregate import (
    aggregate_document_traces,
    default_document_id,
    strip_trace_column,
    trace_filename,
    write_document_trace,
    write_document_traces,
)
from nemo_retriever.common.tracing.collector import (
    BatchTraceCollector,
    decode_payload,
    decode_payload_full,
    encode_payload,
    operator_trace,
)
from nemo_retriever.common.tracing.load import (
    TraceFileError,
    expand_trace_paths,
    load_trace_file,
    load_traces,
    page_summaries_dataframe,
    spans_dataframe,
)
from nemo_retriever.common.tracing.runtime import (
    DEFAULT_TRACE_DETAIL,
    TRACE_COLUMN,
    TRACE_DETAIL_ENV_VAR,
    TRACE_SCHEMA_VERSION,
    VALID_TRACE_DETAILS,
    TraceDetail,
    get_detail,
    normalize_detail,
    record_model,
    registered_models,
    reset_detail,
    set_detail,
    tracing_enabled,
)
from nemo_retriever.common.tracing.spans import (
    Span,
    page_scope,
    record_page_work,
    span,
)

__all__ = [
    "BatchTraceCollector",
    "DEFAULT_TRACE_DETAIL",
    "Span",
    "TRACE_COLUMN",
    "TRACE_DETAIL_ENV_VAR",
    "TRACE_SCHEMA_VERSION",
    "TraceDetail",
    "TraceFileError",
    "VALID_TRACE_DETAILS",
    "aggregate_document_traces",
    "decode_payload",
    "decode_payload_full",
    "default_document_id",
    "encode_payload",
    "expand_trace_paths",
    "get_detail",
    "load_trace_file",
    "load_traces",
    "normalize_detail",
    "operator_trace",
    "page_scope",
    "page_summaries_dataframe",
    "record_model",
    "record_page_work",
    "registered_models",
    "reset_detail",
    "set_detail",
    "span",
    "spans_dataframe",
    "strip_trace_column",
    "trace_filename",
    "tracing_enabled",
    "write_document_trace",
    "write_document_traces",
]
