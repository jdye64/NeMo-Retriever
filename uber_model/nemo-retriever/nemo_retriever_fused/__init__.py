# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused, GPU-resident NeMo Retriever extraction and embedding model.

One invocation replaces the four sequential model calls the production pipeline
makes per page. See `nemo_retriever_fused.pipeline.NemoRetrieverFusedModel`.
"""

from __future__ import annotations

from nemo_retriever_fused.config import (
    EmbedConfig,
    ExecutionConfig,
    FusedPipelineConfig,
    OCRConfig,
    PageElementsConfig,
    RerankConfig,
    TableStructureConfig,
)
from nemo_retriever_fused.gpu_image import DeviceImageBatch
from nemo_retriever_fused.results import ElementResult, FusedResult, PageResult, StageTiming

__all__ = [
    "DeviceImageBatch",
    "ElementResult",
    "EmbedConfig",
    "ExecutionConfig",
    "FusedPipelineConfig",
    "FusedResult",
    "NemoRetrieverFusedModel",
    "OCRConfig",
    "PageElementsConfig",
    "PageResult",
    "RerankConfig",
    "StageTiming",
    "TableStructureConfig",
    "__version__",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> object:
    """Import the pipeline lazily so `import nemo_retriever_fused` stays cheap.

    `NemoRetrieverFusedModel` pulls in torchvision and the upstream model
    packages; deferring that keeps configuration-only imports fast and lets the
    config and result types be used on machines without CUDA.
    """
    if name == "NemoRetrieverFusedModel":
        from nemo_retriever_fused.pipeline import NemoRetrieverFusedModel

        return NemoRetrieverFusedModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
