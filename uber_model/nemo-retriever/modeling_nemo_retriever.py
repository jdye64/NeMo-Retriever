# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace entry point for the fused NeMo Retriever model.

This module exists so the directory can be published as a HuggingFace repo and
loaded with `AutoModel.from_pretrained(..., trust_remote_code=True)`. The
implementation lives in the `nemo_retriever_fused` package; this file only
adapts it to the `PreTrainedModel` surface that `transformers` expects.

The wrapped pipeline is not a differentiable graph. It composes four pretrained
models with non-differentiable postprocessing between them, so `forward`
returns extraction results rather than logits and the model is inference only.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from configuration_nemo_retriever import NemoRetrieverFusedHFConfig
from nemo_retriever_fused.pipeline import NemoRetrieverFusedModel
from nemo_retriever_fused.results import FusedResult

logger = logging.getLogger(__name__)


class NemoRetrieverFusedOutput(ModelOutput):
    """Output of `NemoRetrieverFusedForExtraction.forward`.

    Attributes
    ----------
    page_embeddings:
        ``[num_pages, D]`` device-resident page embeddings, or None when the
        embed stage is disabled.
    element_embeddings:
        ``[num_elements, D]`` device-resident element embeddings, or None.
    result:
        The full `FusedResult`, including detections, text, and stage timings.
    """

    page_embeddings: torch.Tensor | None = None
    element_embeddings: torch.Tensor | None = None
    result: FusedResult | None = None


class NemoRetrieverFusedForExtraction(PreTrainedModel):
    """Fused extraction and embedding model with a `transformers` surface.

    Examples
    --------
    >>> from transformers import AutoModel
    >>> model = AutoModel.from_pretrained("./nemo-retriever", trust_remote_code=True)
    >>> output = model(pages=[page_b64])
    >>> output.result.pages[0].text
    'Quarterly revenue ...'
    """

    config_class = NemoRetrieverFusedHFConfig
    base_model_prefix = "nemo_retriever_fused"
    main_input_name = "pages"
    supports_gradient_checkpointing = False
    _supports_sdpa = True

    def __init__(self, config: NemoRetrieverFusedHFConfig) -> None:
        super().__init__(config)
        self._pipeline = NemoRetrieverFusedModel(config.to_pipeline_config())
        self._loaded = False

    def _init_weights(self, module: torch.nn.Module) -> None:
        """No-op: every stage loads pretrained weights through the pipeline loader."""

    def load_stages(self) -> None:
        """Load the stage weights. Idempotent, so it is safe to call repeatedly."""
        if not self._loaded:
            self._pipeline.load()
            self._loaded = True

    @property
    def pipeline(self) -> NemoRetrieverFusedModel:
        """Return the underlying fused pipeline for direct access to `rerank`."""
        self.load_stages()
        return self._pipeline

    def forward(  # type: ignore[override]
        self,
        pages: Sequence[str | bytes] | None = None,
        *,
        page_ids: Sequence[str] | None = None,
        embed_elements: bool = False,
        return_dict: bool | None = None,
        **_unused: Any,
    ) -> NemoRetrieverFusedOutput | tuple[Any, ...]:
        """Run every enabled stage over *pages* in one invocation.

        Parameters
        ----------
        pages:
            Base64 page images or raw encoded image bytes.
        page_ids:
            Optional page identifiers carried through to the results.
        embed_elements:
            Also embed each detected region, not just the whole page.
        return_dict:
            Return a `NemoRetrieverFusedOutput` instead of a tuple. Defaults to
            the value in the model configuration.

        Raises
        ------
        ValueError
            If *pages* is None or empty.
        """
        if not pages:
            raise ValueError("pages must contain at least one page image")

        self.load_stages()
        result = self._pipeline.forward(pages, page_ids=page_ids, embed_elements=embed_elements)

        use_dict = self.config.use_return_dict if return_dict is None else return_dict
        if not use_dict:
            return result.page_embeddings, result.element_embeddings, result

        return NemoRetrieverFusedOutput(
            page_embeddings=result.page_embeddings,
            element_embeddings=result.element_embeddings,
            result=result,
        )


__all__ = [
    "NemoRetrieverFusedForExtraction",
    "NemoRetrieverFusedOutput",
]
