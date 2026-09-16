# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-request pipeline configuration for ingest job requests.

``PipelineSpec`` records stage overrides such as extract, embed, and storage
parameters. Fields left ``None`` keep caller or runtime defaults.
``stage_order`` controls post-extraction stage ordering only; extraction is
always first.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import ConfigDict, Field

from nemo_retriever.common.schemas.base import RichModel


ExtractionMode = Literal["pdf", "image", "auto", "text", "html", "audio"]
StageName = Literal["extract", "dedup", "caption", "embed", "store", "filter", "webhook"]


class PdfSplitSpec(RichModel):
    """Per-request PDF chunking config (``pages_per_chunk`` only for now)."""

    model_config = ConfigDict(extra="forbid")

    pages_per_chunk: int = Field(default=32, ge=1, le=4096)


class PipelineSpec(RichModel):
    """Representation of fluent pipeline state.

    Each ``*_params`` field is an opaque dict matching the corresponding
    Pydantic params model (``ExtractParams``, ``EmbedParams``, and so on).
    Fields are intentionally permissive so the format does not need to
    track every params-model field change in lock-step.
    """

    model_config = ConfigDict(extra="forbid")

    # Extraction stage selector (mirrors GraphIngestor._extraction_mode).
    extraction_mode: ExtractionMode = "auto"

    extract_params: Optional[dict[str, Any]] = None
    embed_params: Optional[dict[str, Any]] = None
    dedup_params: Optional[dict[str, Any]] = None
    caption_params: Optional[dict[str, Any]] = None
    store_params: Optional[dict[str, Any]] = None
    vdb_upload_params: Optional[dict[str, Any]] = None
    webhook_params: Optional[dict[str, Any]] = None

    split_config: Optional[dict[str, Any]] = None
    pdf_split: Optional[PdfSplitSpec] = None

    stage_order: list[StageName] = Field(default_factory=list)
    result_schema: Literal["legacy", "compact"] = Field(
        default="legacy",
        description=(
            "Result row schema for service return_results/save_to_disk. "
            "'legacy' preserves GraphIngestor.ingest() columns with bulky values stripped; "
            "'compact' returns the future compact public schema."
        ),
    )
    return_embeddings: bool = Field(
        default=False,
        description="Include embedding payload values in transport rows.",
    )
    return_images: bool = Field(
        default=False,
        description="Include raw image payload values in legacy transport rows.",
    )

    def is_empty(self) -> bool:
        """``True`` when the client supplied no overrides and no stage_order.

        Used by the worker to short-circuit to the legacy
        baked-at-startup pipeline path.
        """
        return (
            self.extraction_mode in ("pdf", "auto")
            and self.extract_params is None
            and self.embed_params is None
            and self.dedup_params is None
            and self.caption_params is None
            and self.store_params is None
            and self.vdb_upload_params is None
            and self.webhook_params is None
            and self.split_config is None
            and self.pdf_split is None
            and not self.stage_order
            and self.result_schema == "legacy"
            and not self.return_embeddings
            and not self.return_images
        )
