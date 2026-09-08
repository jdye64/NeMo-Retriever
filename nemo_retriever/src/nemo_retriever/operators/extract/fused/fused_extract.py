# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd

from nemo_retriever.common.modality.fused.shared import (
    _error_payload,
    fused_extract_pages,
    load_fused_model,
)
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.gpu_operator import GPUOperator

__all__ = ["FusedExtractionActor"]


@designer_component(
    name="Fused Extraction",
    category="Detection & OCR",
    compute="gpu",
    description="Runs page elements, table structure, OCR, and embedding as one GPU-resident model",
    category_color="#76b900",
)
class FusedExtractionActor(AbstractOperator, GPUOperator):
    """Run page elements, table structure, OCR, and embed in a single call.

    This operator replaces the ``PageElementDetectionActor →
    TableStructureActor → OCRActor → _BatchEmbedActor`` chain with one node.
    The four staged actors exchange page rasters as base64 in a DataFrame
    column, so each one decodes the same page again and copies it across PCIe
    again; the fused model decodes once into device memory and every stage
    reads the same resident tensor.

    There is no CPU variant and no NIM variant: the whole point is a single
    resident model, so ``ExtractParams`` rejects per-stage endpoint overrides
    when ``method='fused'``. That keeps this operator off the archetype path
    that the staged operators use to choose between local and remote.

    Use with Ray Data::

        ds = ds.map_batches(
            FusedExtractionActor,
            fn_constructor_kwargs={...},
            batch_format="pandas",
        )
    """

    def __init__(self, **fused_kwargs: Any) -> None:
        super().__init__(**fused_kwargs)
        self.fused_kwargs = dict(fused_kwargs)
        # Loading is deferred to the first batch so graph construction stays
        # cheap and importable on hosts without the optional package or a GPU.
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = load_fused_model()
        return self._model

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return fused_extract_pages(
            data,
            model=self._ensure_model(),
            **self.fused_kwargs,
            **kwargs,
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, pages_df: Any, **override_kwargs: Any) -> Any:
        try:
            return self.run(pages_df, **override_kwargs)
        except Exception as exc:
            if isinstance(pages_df, pd.DataFrame):
                out = pages_df.copy()
                payload = _error_payload(stage="actor_call", exc=exc)
                out["page_elements_v3"] = [payload for _ in range(len(out.index))]
                out["page_elements_v3_num_detections"] = [0 for _ in range(len(out.index))]
                out["page_elements_v3_counts_by_label"] = [{} for _ in range(len(out.index))]
                return out
            return [{"page_elements_v3": _error_payload(stage="actor_call", exc=exc)}]
