# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Result types returned by a single fused invocation.

The production pipeline hands DataFrame columns between stages, so each stage
writes host-side Python dicts and lists. The fused model keeps its intermediates
on device and converts to host types once, at the boundary the caller asks for.

`PageResult.to_metadata` produces the same shape the existing pipeline writes
into its `page_elements_v3`, `table_structure_v1`, `table`, `chart`,
`infographic`, and `text` columns, so a consumer can adopt the fused model
without changing its downstream schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class ElementResult:
    """One detected page element and whatever was extracted from it.

    Attributes
    ----------
    label:
        Page element class name, one of the six page-element labels.
    score:
        Detection confidence.
    bbox_xyxy_norm:
        ``(x1, y1, x2, y2)`` in 0-1 page coordinates.
    text:
        OCR text for the region, when OCR ran over it.
    table_markdown:
        Reconstructed table when the element is a table and table structure ran.
    embedding:
        Device-resident embedding for the region, when the embed stage ran.
    """

    label: str
    score: float
    bbox_xyxy_norm: tuple[float, float, float, float]
    text: str | None = None
    table_markdown: str | None = None
    embedding: torch.Tensor | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Return the host-side dict shape the existing pipeline columns use."""
        payload: dict[str, Any] = {
            "label_name": self.label,
            "score": self.score,
            "bbox_xyxy_norm": list(self.bbox_xyxy_norm),
        }
        if self.text is not None:
            payload["text"] = self.text
        if self.table_markdown is not None:
            payload["table_markdown"] = self.table_markdown
        return payload


@dataclass(slots=True)
class StageTiming:
    """Wall-clock milliseconds for one stage of one invocation.

    Recorded with CUDA events rather than `time.perf_counter` so the numbers
    reflect device time without forcing a synchronisation per stage.
    """

    name: str
    milliseconds: float
    items: int = 0

    def per_item(self) -> float:
        """Return milliseconds per item, or the total when `items` is zero."""
        return self.milliseconds / self.items if self.items else self.milliseconds


@dataclass(slots=True)
class PageResult:
    """Everything the fused model produced for one page.

    Attributes
    ----------
    page_id:
        Caller-supplied page identifier.
    height, width:
        Original page pixel dimensions.
    elements:
        Detected page elements with their extracted content.
    text:
        Concatenated page text from the OCR stage.
    embedding:
        Device-resident page-level embedding, when the embed stage ran.
    """

    page_id: str
    height: int
    width: int
    elements: list[ElementResult] = field(default_factory=list)
    text: str = ""
    embedding: torch.Tensor | None = None

    def elements_of(self, label: str) -> list[ElementResult]:
        """Return the elements whose class name is *label*."""
        return [element for element in self.elements if element.label == label]

    def to_metadata(self) -> dict[str, Any]:
        """Return a dict matching the existing pipeline's per-row payload."""
        return {
            "page_id": self.page_id,
            "page_elements_v3": {
                "detections": [element.to_metadata() for element in self.elements],
            },
            "table": [element.to_metadata() for element in self.elements if element.label == "table"],
            "chart": [element.to_metadata() for element in self.elements if element.label == "chart"],
            "infographic": [element.to_metadata() for element in self.elements if element.label == "infographic"],
            "text": self.text,
        }


@dataclass(slots=True)
class FusedResult:
    """The complete output of one `NemoRetrieverFusedModel` invocation.

    Attributes
    ----------
    pages:
        One `PageResult` per input page, in input order.
    page_embeddings:
        ``[num_pages, D]`` device-resident page embeddings, when embed ran.
    element_embeddings:
        ``[num_elements, D]`` device-resident element embeddings, when embed ran
        with `embed_elements` enabled.
    timings:
        Per-stage device timings for this invocation.
    """

    pages: list[PageResult]
    page_embeddings: torch.Tensor | None = None
    element_embeddings: torch.Tensor | None = None
    timings: list[StageTiming] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pages)

    def total_milliseconds(self) -> float:
        """Return the summed stage time for this invocation."""
        return sum(timing.milliseconds for timing in self.timings)

    def timing_table(self) -> str:
        """Return the per-stage timings as a fixed-width table for logging."""
        if not self.timings:
            return "no timings recorded"
        width = max(len(timing.name) for timing in self.timings)
        lines = [f"{'stage'.ljust(width)}  {'ms':>9}  {'items':>6}  {'ms/item':>9}"]
        for timing in self.timings:
            lines.append(
                f"{timing.name.ljust(width)}  {timing.milliseconds:9.2f}  "
                f"{timing.items:6d}  {timing.per_item():9.3f}"
            )
        lines.append(f"{'total'.ljust(width)}  {self.total_milliseconds():9.2f}")
        return "\n".join(lines)

    def to_metadata(self) -> list[dict[str, Any]]:
        """Return one host-side metadata dict per page.

        This is the single device-to-host conversion point for detection and
        text results. Embeddings are left on device; call `.cpu()` on
        `page_embeddings` only if the consumer cannot take a device buffer.
        """
        return [page.to_metadata() for page in self.pages]
