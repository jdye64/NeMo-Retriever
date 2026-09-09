# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-page pipeline tracing.

Every operator invocation is timed and recorded against the pages it touched,
producing one document-level JSON artifact per source document that retains
full per-page detail. Operator-level timing is always on; curated hot-spot
spans around network calls, model invocations, and heavy dependency work are
recorded at ``full`` detail.

This package is independent of the legacy control-message tracing under
``common/api/internal/primitives/tracing``, which the current operator graph
does not use.

Typical use from Python::

    from nemo_retriever.common.tracing import load_traces, spans_dataframe

    traces = load_traces("traces/")
    spans = spans_dataframe(traces)
    spans.groupby("operator").amortized_ms.sum().sort_values(ascending=False)
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
    full_detail_enabled,
    get_detail,
    normalize_detail,
    record_model,
    registered_models,
    reset_detail,
    set_detail,
    tracing_enabled,
)
from nemo_retriever.common.tracing.spans import (
    SPAN_CATEGORIES,
    Span,
    accumulate,
    model_span,
    page_scope,
    span,
)

__all__ = [
    "BatchTraceCollector",
    "DEFAULT_TRACE_DETAIL",
    "SPAN_CATEGORIES",
    "Span",
    "TRACE_COLUMN",
    "TRACE_DETAIL_ENV_VAR",
    "TRACE_SCHEMA_VERSION",
    "TraceDetail",
    "TraceFileError",
    "VALID_TRACE_DETAILS",
    "accumulate",
    "aggregate_document_traces",
    "decode_payload",
    "default_document_id",
    "encode_payload",
    "expand_trace_paths",
    "full_detail_enabled",
    "get_detail",
    "load_trace_file",
    "load_traces",
    "model_span",
    "normalize_detail",
    "operator_trace",
    "page_scope",
    "page_summaries_dataframe",
    "record_model",
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
