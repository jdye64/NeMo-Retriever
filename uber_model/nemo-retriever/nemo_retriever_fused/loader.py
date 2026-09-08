# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight loading for the fused model.

Every stage's weights come from the same pinned HuggingFace revisions the
production pipeline resolves, so the fused model and the multi-NIM pipeline run
identical parameters. Revisions live in `REVISIONS`, mirroring
`nemo_retriever.models.hf_model_registry.HF_MODEL_REVISIONS`.

The loader builds the two YOLOX detectors from the raw checkpoint rather than
through the upstream `define_model` helper. `define_model` returns a
`YoloXWrapper` whose `forward` performs box scaling with host-side arithmetic
and calls the packaged `postprocess`; the fused pipeline replaces both with
device-resident equivalents in `gpu_ops`, so it needs the bare `nn.Module`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from nemo_retriever_fused.config import (
    PAGE_ELEMENT_LABELS,
    TABLE_STRUCTURE_LABELS,
    FusedPipelineConfig,
)
from nemo_retriever_fused.stages.detectors import FusedYoloxDetector

logger = logging.getLogger(__name__)

# Pinned revisions, kept in step with the in-repo registry so that a fused run
# and a production run load byte-identical weights.
REVISIONS: dict[str, str] = {
    "nvidia/nemotron-page-elements-v3": "df62dbb631502575ac4d43b44d700b1674ab1d56",
    "nvidia/nemotron-table-structure-v1": "9350162faa1110320af62699105780b0c87b73ad",
    "nvidia/nemotron-ocr-v1": "8657d08d3279f4864002d5fd3fdcd47ad8c96bcb",
    "nvidia/nemotron-ocr-v2": "0e83e83f17943524b90afa6c0fd82ac2bc1a40ca",
    "nvidia/llama-nemotron-embed-vl-1b-v2": "582e3bf72aee355e3c59ed89de53543c5b0657ee",
    "nvidia/llama-nemotron-rerank-vl-1b-v2": "9c20c4aedf9ec87b6b7346c3bc4754ea030dab35",
}

# Environment variable pointing at the local mirror produced by
# `uber_model/download_models.py`. When set, weights load from disk instead of
# resolving through the Hub.
MIRROR_ENV = "NEMO_RETRIEVER_FUSED_MIRROR"

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    """Return the `torch.dtype` for a configuration dtype string."""
    try:
        return _DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}; choose from {sorted(_DTYPES)}") from exc


def mirror_dir_for(repo_id: str) -> Path | None:
    """Return the local mirror directory for *repo_id*, when one exists."""
    root = os.environ.get(MIRROR_ENV)
    if not root:
        return None
    candidate = Path(root) / repo_id.replace("/", "--")
    return candidate if candidate.is_dir() else None


def _resolve_file(repo_id: str, filename: str) -> str:
    """Return a local path to *filename* from *repo_id*, preferring the mirror."""
    mirror = mirror_dir_for(repo_id)
    if mirror is not None:
        local = mirror / filename
        if local.is_file():
            return str(local)
        logger.debug("mirror %s has no %s; falling back to the Hub", mirror, filename)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, revision=REVISIONS.get(repo_id))


def _build_yolox(num_classes: int, module_prefix: str) -> torch.nn.Module:
    """Construct an untrained YOLOX with the geometry both detectors share."""
    yolox_module = __import__(f"{module_prefix}.yolox.yolox", fromlist=["YOLOX"])
    pafpn_module = __import__(f"{module_prefix}.yolox.yolo_pafpn", fromlist=["YOLOPAFPN"])
    head_module = __import__(f"{module_prefix}.yolox.yolo_head", fromlist=["YOLOXHead"])

    in_channels = [256, 512, 1024]
    backbone = pafpn_module.YOLOPAFPN(1.00, 1.00, in_channels=in_channels, act="silu")
    head = head_module.YOLOXHead(num_classes, 1.00, in_channels=in_channels, act="silu")
    model = yolox_module.YOLOX(backbone, head)

    # Upstream adjusts BatchNorm epsilon and momentum for inference; matching it
    # keeps the fused detector numerically identical to the packaged one.
    for submodule in model.modules():
        if isinstance(submodule, torch.nn.BatchNorm2d):
            submodule.eps = 1e-3
            submodule.momentum = 0.03

    return model


def load_detector(
    repo_id: str,
    *,
    module_prefix: str,
    weights_filename: str,
    label_names: tuple[str, ...],
    input_size: tuple[int, int],
    conf_thresh: float,
    iou_thresh: float,
    class_agnostic: bool,
    min_bbox_size: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> FusedYoloxDetector:
    """Load one YOLOX detector onto *device* and wrap it for fused inference.

    Parameters
    ----------
    repo_id:
        HuggingFace repo holding the checkpoint.
    module_prefix:
        Importable package that provides the YOLOX classes, either
        ``nemotron_page_elements_v3`` or ``nemotron_table_structure_v1``.
    weights_filename:
        Checkpoint path within the repo.
    label_names, input_size, conf_thresh, iou_thresh, class_agnostic, min_bbox_size:
        Stage parameters forwarded to `FusedYoloxDetector`.
    device:
        Target CUDA device.
    autocast_dtype:
        Autocast dtype for the forward pass.
    """
    model = _build_yolox(len(label_names), module_prefix)

    weights_path = _resolve_file(repo_id, weights_filename)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)

    model = model.eval().to(device)
    logger.info("loaded %s onto %s", repo_id, device)

    return FusedYoloxDetector(
        model,
        num_classes=len(label_names),
        label_names=label_names,
        input_size=input_size,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        class_agnostic=class_agnostic,
        min_bbox_size=min_bbox_size,
        device=device,
        autocast_dtype=autocast_dtype,
    )


def load_page_elements_detector(config: FusedPipelineConfig, device: torch.device) -> FusedYoloxDetector:
    """Load the page-elements detector described by *config*."""
    stage = config.page_elements
    return load_detector(
        stage.repo_id,
        module_prefix="nemotron_page_elements_v3",
        weights_filename="nemotron_page_elements_v3/weights.pth",
        label_names=PAGE_ELEMENT_LABELS,
        input_size=stage.input_size,
        conf_thresh=stage.conf_thresh,
        iou_thresh=stage.iou_thresh,
        class_agnostic=stage.class_agnostic,
        min_bbox_size=stage.min_bbox_size,
        device=device,
        autocast_dtype=resolve_dtype(config.execution.detector_dtype),
    )


def load_table_structure_detector(config: FusedPipelineConfig, device: torch.device) -> FusedYoloxDetector:
    """Load the table-structure detector described by *config*."""
    stage = config.table_structure
    return load_detector(
        stage.repo_id,
        module_prefix="nemotron_table_structure_v1",
        weights_filename="nemotron_table_structure_v1/weights.pth",
        label_names=TABLE_STRUCTURE_LABELS,
        input_size=stage.input_size,
        conf_thresh=stage.conf_thresh,
        iou_thresh=stage.iou_thresh,
        class_agnostic=stage.class_agnostic,
        min_bbox_size=stage.min_bbox_size,
        device=device,
        autocast_dtype=resolve_dtype(config.execution.detector_dtype),
    )


def load_ocr_pipeline(config: FusedPipelineConfig, device: torch.device) -> Any:
    """Load the upstream nemotron-ocr pipeline for the configured version.

    The fused OCR stage calls this object's phase methods directly, so the
    upstream pipeline is loaded unmodified. Only its host-side entry point is
    bypassed.
    """
    stage = config.ocr
    if device.type != "cuda":
        raise RuntimeError("nemotron-ocr requires a CUDA device")

    mirror = mirror_dir_for(stage.repo_id)
    model_dir = str(mirror) if mirror is not None else None

    if stage.version == "v2":
        from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

        pipeline = NemotronOCRV2(
            detector_max_batch_size=stage.detector_max_batch_size,
            recognizer_chunk_size=stage.recognizer_chunk_size,
            relational_chunk_size=stage.relational_chunk_size,
            use_prefilter=stage.use_prefilter,
            pad_color=list(stage.pad_color),
            pad_how=stage.pad_how,
            infer_length=stage.infer_length,
            **({"model_dir": model_dir} if model_dir else {}),
            **({"lang": stage.lang} if stage.lang else {}),
        )
    else:
        from nemotron_ocr.inference.pipeline import NemotronOCR

        pipeline = NemotronOCR(**({"model_dir": model_dir} if model_dir else {}))

    logger.info("loaded %s (%s) onto %s", stage.repo_id, stage.version, device)
    return pipeline


def load_vl_model(repo_id: str, *, device: torch.device, dtype: torch.dtype) -> tuple[Any, Any]:
    """Load a VL embed or rerank model and its tokenizer onto *device*.

    Returns
    -------
    tuple[Any, Any]
        The model and its tokenizer.
    """
    from transformers import AutoModel, AutoTokenizer

    mirror = mirror_dir_for(repo_id)
    source = str(mirror) if mirror is not None else repo_id
    revision = None if mirror is not None else REVISIONS.get(repo_id)

    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        source,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.eval().to(device)
    logger.info("loaded %s onto %s as %s", repo_id, device, dtype)
    return model, tokenizer
