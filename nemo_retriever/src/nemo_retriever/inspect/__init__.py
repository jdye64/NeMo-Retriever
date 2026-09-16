# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect local LanceDB indexes produced by ``retriever ingest``."""

from nemo_retriever.inspect.html import render_index_html
from nemo_retriever.inspect.pretty import render_index_pretty
from nemo_retriever.inspect.summary import (
    ChunkPreview,
    DocumentSummary,
    IndexSummary,
    summarize_index,
    summary_to_dict,
)

__all__ = [
    "ChunkPreview",
    "DocumentSummary",
    "IndexSummary",
    "render_index_html",
    "render_index_pretty",
    "summarize_index",
    "summary_to_dict",
]
