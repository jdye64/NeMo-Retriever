# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the fused NeMo Retriever model.

Every default mirrors the value the corresponding stage uses in the production
`nemo_retriever` pipeline so that the fused model is numerically comparable to
the sequence of individual NIM calls it replaces. Where a default differs
deliberately, the field docstring says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Both detectors are YOLOX with identical geometry (depth 1.0, width 1.0,
# in_channels [256, 512, 1024]) and the same 1024x1024 letterbox. The shared
# constant lets one preprocessing kernel serve both stages.
YOLOX_INPUT_SIZE: tuple[int, int] = (1024, 1024)
YOLOX_PAD_VALUE: float = 114.0

# Detector input resolution used by nemotron-ocr v1/v2.
OCR_INFER_LENGTH: int = 1024

# Tile side length for the VL embed/rerank vision tower.
VL_TILE_SIZE: int = 448

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
SIGLIP_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
SIGLIP_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)

PageElementLabel = Literal["table", "chart", "title", "infographic", "text", "header_footer"]
TableStructureLabel = Literal["border", "cell", "row", "column", "header"]

PAGE_ELEMENT_LABELS: tuple[str, ...] = (
    "table",
    "chart",
    "title",
    "infographic",
    "text",
    "header_footer",
)

TABLE_STRUCTURE_LABELS: tuple[str, ...] = (
    "border",
    "cell",
    "row",
    "column",
    "header",
)


@dataclass(slots=True)
class PageElementsConfig:
    """Layout detection settings, matching `nemotron_page_elements_v3.Exp`."""

    repo_id: str = "nvidia/nemotron-page-elements-v3"
    input_size: tuple[int, int] = YOLOX_INPUT_SIZE
    conf_thresh: float = 0.01
    iou_thresh: float = 0.5
    class_agnostic: bool = True
    min_bbox_size: int = 0
    thresholds_per_class: dict[str, float] = field(
        default_factory=lambda: {
            "table": 0.1,
            "chart": 0.01,
            "infographic": 0.01,
            "title": 0.1,
            "text": 0.1,
            "header_footer": 0.1,
        }
    )
    # Weighted box fusion runs on device in this implementation; upstream runs
    # the equivalent pass in numpy on the host.
    wbf_iou_thresh: float = 0.55


@dataclass(slots=True)
class TableStructureConfig:
    """Table cell/row/column detection, matching `nemotron_table_structure_v1.Exp`."""

    repo_id: str = "nvidia/nemotron-table-structure-v1"
    input_size: tuple[int, int] = YOLOX_INPUT_SIZE
    conf_thresh: float = 0.01
    iou_thresh: float = 0.25
    class_agnostic: bool = False
    min_bbox_size: int = 0
    threshold: float = 0.05
    # Score gate the production pipeline applies via YOLOX_TABLE_MIN_SCORE.
    min_score: float = 0.1


@dataclass(slots=True)
class OCRConfig:
    """OCR settings, matching `nemotron_ocr.inference.pipeline_v2`."""

    repo_id: str = "nvidia/nemotron-ocr-v2"
    version: Literal["v1", "v2"] = "v2"
    lang: str = "multi"
    merge_level: Literal["word", "sentence", "paragraph"] = "paragraph"
    infer_length: int = OCR_INFER_LENGTH
    detector_max_batch_size: int = 8
    recognizer_chunk_size: int = 128
    relational_chunk_size: int = 128
    use_prefilter: bool = True
    pad_how: Literal["bottom_right", "center"] = "bottom_right"
    pad_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    include_invalid: bool = False


@dataclass(slots=True)
class EmbedConfig:
    """Multimodal embedding settings for the VL embedder."""

    repo_id: str = "nvidia/llama-nemotron-embed-vl-1b-v2"
    output_dimension: int = 2048
    tile_size: int = VL_TILE_SIZE
    min_tiles: int = 1
    max_tiles: int = 6
    use_thumbnail: bool = True
    norm_type: Literal["imagenet", "siglip"] = "imagenet"
    query_prefix: str = "query:"
    document_prefix: str = "passage:"
    normalize: bool = True
    batch_size: int = 8


@dataclass(slots=True)
class RerankConfig:
    """Optional reranking settings for the VL cross-encoder."""

    repo_id: str = "nvidia/llama-nemotron-rerank-vl-1b-v2"
    tile_size: int = VL_TILE_SIZE
    min_tiles: int = 1
    max_tiles: int = 6
    use_thumbnail: bool = True
    norm_type: Literal["imagenet", "siglip"] = "imagenet"
    batch_size: int = 8


@dataclass(slots=True)
class ExecutionConfig:
    """Device, dtype, and scheduling knobs for the fused pipeline.

    Attributes
    ----------
    device:
        CUDA device the whole pipeline pins itself to. All four stages share it
        so that no intermediate ever leaves device memory.
    detector_dtype:
        Autocast dtype for the two YOLOX detectors.
    ocr_dtype:
        nemotron-ocr runs its detector in float16 upstream; keep parity.
    embed_dtype:
        The VL tower ships bfloat16 weights.
    overlap_detectors:
        Run page-elements and table-structure on separate CUDA streams. They
        are independent once the page tensor exists, and the two YOLOX graphs
        have identical shape so they interleave cleanly.
    decode_on_device:
        Decode JPEG page images with nvJPEG on the GPU instead of PIL on the
        host. Falls back to host decode for formats nvJPEG cannot handle.
    pinned_staging:
        Stage host->device copies through pinned memory so the transfers are
        asynchronous and overlap with compute.
    graph_capture:
        Capture the detector forward passes as CUDA graphs. Requires static
        batch shape, so it only engages when `static_batch_size` is set.
    static_batch_size:
        Pad every detector batch to this size to keep kernel shapes constant.
    """

    device: str = "cuda:0"
    detector_dtype: str = "float16"
    ocr_dtype: str = "float16"
    embed_dtype: str = "bfloat16"
    overlap_detectors: bool = True
    decode_on_device: bool = True
    pinned_staging: bool = True
    graph_capture: bool = False
    static_batch_size: int | None = None
    # Emit NVTX ranges so the fused pipeline shows up cleanly under Nsight.
    nvtx: bool = True
    # Single explicit synchronisation at the end of `forward` instead of the
    # implicit sync each stage boundary performs today.
    single_sync_point: bool = True


@dataclass(slots=True)
class FusedPipelineConfig:
    """Top-level configuration for `NemoRetrieverFusedModel`."""

    page_elements: PageElementsConfig = field(default_factory=PageElementsConfig)
    table_structure: TableStructureConfig = field(default_factory=TableStructureConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Stage toggles. Disabling a stage skips loading its weights entirely.
    extract_tables: bool = True
    extract_charts: bool = True
    extract_infographics: bool = True
    extract_text: bool = True
    enable_embed: bool = True
    enable_rerank: bool = False

    def stages_enabled(self) -> tuple[str, ...]:
        """Return the ordered names of the stages this configuration runs."""
        stages: list[str] = ["page_elements"]
        if self.extract_tables:
            stages.append("table_structure")
        if self.extract_text or self.extract_tables or self.extract_charts or self.extract_infographics:
            stages.append("ocr")
        if self.enable_embed:
            stages.append("embed")
        if self.enable_rerank:
            stages.append("rerank")
        return tuple(stages)
