# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace configuration for the fused NeMo Retriever model.

`PretrainedConfig` serialises to flat JSON, while the pipeline is configured
with nested dataclasses. This class holds the flat form for `config.json` and
converts to `FusedPipelineConfig` in `to_pipeline_config`.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from transformers import PretrainedConfig

from nemo_retriever_fused.config import (
    EmbedConfig,
    ExecutionConfig,
    FusedPipelineConfig,
    OCRConfig,
    PageElementsConfig,
    RerankConfig,
    TableStructureConfig,
)


class NemoRetrieverFusedHFConfig(PretrainedConfig):
    """Configuration for `NemoRetrieverFusedForExtraction`.

    Parameters
    ----------
    page_elements, table_structure, ocr, embed, rerank, execution:
        Per-stage settings as plain dicts. Omitted keys fall back to the
        dataclass defaults, which mirror the production pipeline.
    extract_tables, extract_charts, extract_infographics, extract_text:
        Stage gating, matching the production `ExtractParams` flags.
    enable_embed, enable_rerank:
        Whether to load and run the embedding and reranking stages.
    **kwargs:
        Forwarded to `PretrainedConfig`.
    """

    model_type = "nemo_retriever_fused"

    def __init__(
        self,
        page_elements: dict[str, Any] | None = None,
        table_structure: dict[str, Any] | None = None,
        ocr: dict[str, Any] | None = None,
        embed: dict[str, Any] | None = None,
        rerank: dict[str, Any] | None = None,
        execution: dict[str, Any] | None = None,
        extract_tables: bool = True,
        extract_charts: bool = True,
        extract_infographics: bool = True,
        extract_text: bool = True,
        enable_embed: bool = True,
        enable_rerank: bool = False,
        **kwargs: Any,
    ) -> None:
        self.page_elements = page_elements or asdict(PageElementsConfig())
        self.table_structure = table_structure or asdict(TableStructureConfig())
        self.ocr = ocr or asdict(OCRConfig())
        self.embed = embed or asdict(EmbedConfig())
        self.rerank = rerank or asdict(RerankConfig())
        self.execution = execution or asdict(ExecutionConfig())

        self.extract_tables = extract_tables
        self.extract_charts = extract_charts
        self.extract_infographics = extract_infographics
        self.extract_text = extract_text
        self.enable_embed = enable_embed
        self.enable_rerank = enable_rerank

        super().__init__(**kwargs)

    def to_pipeline_config(self) -> FusedPipelineConfig:
        """Return the nested `FusedPipelineConfig` this configuration describes.

        JSON round-trips tuples as lists, so tuple-typed fields are restored
        before the dataclasses are constructed.
        """
        return FusedPipelineConfig(
            page_elements=PageElementsConfig(**_restore_tuples(self.page_elements, ("input_size",))),
            table_structure=TableStructureConfig(**_restore_tuples(self.table_structure, ("input_size",))),
            ocr=OCRConfig(**_restore_tuples(self.ocr, ("pad_color",))),
            embed=EmbedConfig(**dict(self.embed)),
            rerank=RerankConfig(**dict(self.rerank)),
            execution=ExecutionConfig(**dict(self.execution)),
            extract_tables=self.extract_tables,
            extract_charts=self.extract_charts,
            extract_infographics=self.extract_infographics,
            extract_text=self.extract_text,
            enable_embed=self.enable_embed,
            enable_rerank=self.enable_rerank,
        )


def _restore_tuples(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Return *payload* with the named list-valued keys converted back to tuples."""
    restored = dict(payload)
    for key in keys:
        if isinstance(restored.get(key), list):
            restored[key] = tuple(restored[key])
    return restored


__all__ = ["NemoRetrieverFusedHFConfig"]
