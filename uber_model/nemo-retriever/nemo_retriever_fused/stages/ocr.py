# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident OCR stage.

This stage is the largest single win in the fused pipeline. The upstream
`nemotron_ocr` v2 pipeline is well optimised internally: preprocess, detector,
prefilter, NMS, rectify plus grid sample, recogniser, and the relational model
all run on the GPU. The problem is its entry point.

`NemotronOCRV2._load_image_to_tensor_uint8` begins with::

    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()

so handing it a CUDA tensor copies the image to the host, and
`_preprocess_batch` then copies it straight back with
``tensor.to("cuda", non_blocking=True)``. The NRL wrapper makes this worse: its
`_tensor_to_png_b64` path goes device tensor to host, to numpy, to PIL, to PNG
bytes, to base64 text, and then the upstream pipeline base64-decodes and
torchvision-decodes it back onto the device.

`GpuOcrStage` skips `_load_image_to_tensor_uint8` and `_preprocess_batch`
entirely. It builds the detector input on device with
`gpu_ops.resize_and_pad_for_ocr` and calls the upstream phase methods directly,
so a crop produced by the page-elements stage never leaves device memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from nemo_retriever_fused import gpu_ops
from nemo_retriever_fused.config import OCRConfig

logger = logging.getLogger(__name__)


class _UpstreamOcrPipeline(Protocol):
    """The subset of `nemotron_ocr.inference.pipeline_v2.NemotronOCRV2` used here.

    Depending on these phase methods rather than `__call__` is what lets the
    fused stage supply an already-device-resident detector batch.
    """

    infer_length: int

    def _run_detector_batched(self, batch: torch.Tensor) -> tuple[Any, Any, Any]: ...

    def _prefilter_detections(self, conf: Any, rboxes: Any) -> Any: ...

    def _run_nms(self, conf: Any, rboxes: Any) -> Any: ...

    def _run_rectify_and_sample(self, quads: Any, counts: Any, features: Any) -> tuple[Any, Any, Any, Any]: ...

    def _run_recognizer_chunked(self, quads: Any) -> tuple[Any, Any, Any]: ...

    def _decode_with_fallback(self, batch: dict[str, Any]) -> Any: ...


@dataclass(slots=True)
class TextRegion:
    """One recognised text region in normalized crop coordinates.

    Attributes
    ----------
    text:
        Recognised text at the configured merge level.
    confidence:
        Recogniser confidence for the region.
    left, right, upper, lower:
        Axis-aligned bounds in 0-1 coordinates relative to the crop that was
        passed to OCR, matching the production pipeline's contract.
    quad:
        The four corner points in the same normalized space.
    """

    text: str
    confidence: float
    left: float
    right: float
    upper: float
    lower: float
    quad: list[list[float]]


@dataclass(slots=True)
class OcrResult:
    """OCR output for one crop."""

    regions: list[TextRegion]

    @property
    def text(self) -> str:
        """Return the regions joined into a single string in emission order."""
        return "\n".join(region.text for region in self.regions if region.text)


class GpuOcrStage:
    """Run nemotron-ocr from device tensors, bypassing its host entry point.

    Parameters
    ----------
    pipeline:
        A loaded upstream `NemotronOCRV2` (or v1) instance whose weights are on
        *device*.
    config:
        OCR settings.
    device:
        Device the pipeline's weights live on.
    dtype:
        Detector compute dtype. Upstream uses float16.
    """

    def __init__(
        self,
        pipeline: _UpstreamOcrPipeline,
        config: OCRConfig,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self._pipeline = pipeline
        self._config = config
        self._device = device
        self._dtype = dtype

    def run(self, crops: list[torch.Tensor]) -> list[OcrResult]:
        """Recognise text in every crop, keeping the inputs on device.

        Parameters
        ----------
        crops:
            ``[C, h, w]`` CUDA tensors in 0-255 space, produced by cropping the
            resident page tensor.

        Returns
        -------
        list[OcrResult]
            One result per crop, index-aligned with *crops*.
        """
        if not crops:
            return []

        results: list[OcrResult] = []
        chunk = max(1, self._config.detector_max_batch_size)
        for start in range(0, len(crops), chunk):
            results.extend(self._run_chunk(crops[start : start + chunk]))
        return results

    def _run_chunk(self, crops: list[torch.Tensor]) -> list[OcrResult]:
        """Run one detector-sized chunk through the upstream phase methods."""
        batch, original_shapes, padded_lengths = gpu_ops.resize_and_pad_for_ocr(
            crops,
            infer_length=self._config.infer_length,
            pad_how=self._config.pad_how,
            dtype=self._dtype,
        )

        with torch.inference_mode():
            det_conf, det_rboxes, det_features = self._pipeline._run_detector_batched(batch)
            del batch

            if self._config.use_prefilter:
                det_conf = self._pipeline._prefilter_detections(det_conf, det_rboxes)

            nms_result = self._pipeline._run_nms(det_conf, det_rboxes)
            del det_conf, det_rboxes

            if nms_result is None:
                return [OcrResult(regions=[]) for _ in crops]

            quads, confidence, region_counts, _ = nms_result

            quads_cuda, rec_quads, _rel_quads, _counts_cpu = self._pipeline._run_rectify_and_sample(
                quads, region_counts, det_features
            )
            del det_features

            rec_ids, rec_probs, _rec_features = self._pipeline._run_recognizer_chunked(rec_quads)
            del rec_quads

            # Rescale quads from detector space back to crop pixels. Upstream
            # does the same arithmetic; keeping it here means the scale factors
            # are built on device from `padded_lengths` without a host trip.
            scale_factors = torch.as_tensor(padded_lengths, dtype=torch.float32, device=quads_cuda.device) / float(
                self._config.infer_length
            )
            counts_long = region_counts.to(dtype=torch.long, device=quads_cuda.device)
            scale_per_region = torch.repeat_interleave(scale_factors, counts_long, dim=0)
            quads_scaled = quads_cuda * scale_per_region.view(-1, 1, 1)

            text_confidence = _sequence_confidence(rec_ids, rec_probs)

        # The decoder is a host-side text assembler, so this is the single
        # unavoidable device-to-host transfer for the OCR stage. Batching every
        # tensor into one `_decode_with_fallback` call means one sync per chunk
        # rather than one per crop.
        decoded = self._pipeline._decode_with_fallback(
            {
                "sequences": rec_ids.cpu(),
                "sequence_probs": rec_probs.cpu(),
                "text_confidence": text_confidence.cpu(),
                "region_counts": region_counts.cpu(),
                "quads": quads_scaled.cpu(),
                "confidence": confidence.cpu(),
            }
        )

        return [_to_result(example, original_shapes[index]) for index, example in enumerate(decoded)]


def _sequence_confidence(sequence_ids: torch.Tensor, sequence_probs: torch.Tensor) -> torch.Tensor:
    """Return per-region text confidence as the geometric mean of token probs.

    Mirrors the confidence computation in the upstream `_process_batch`: tokens
    at or after the first end-of-sequence marker and padding tokens are excluded.
    """
    before_eos = (sequence_ids == 1).cumsum(dim=1) == 0
    real = (sequence_ids != 0) & before_eos
    counts = real.sum(dim=1).clamp(min=1).float()
    log_sum = (torch.log(sequence_probs.clamp(min=1e-8)) * real.float()).sum(dim=1)
    return torch.exp(log_sum / counts)


def _to_result(example: Any, crop_hw: tuple[int, int]) -> OcrResult:
    """Convert one decoded example into normalized `TextRegion` entries."""
    import numpy as np

    height, width = crop_hw
    regions: list[TextRegion] = []

    for text_region in example:
        vertices = text_region.region.vertices
        vertices = vertices.cpu().numpy() if hasattr(vertices, "cpu") else np.asarray(vertices)
        normalized = vertices.astype("float64", copy=True)
        normalized[:, 0] /= max(width, 1)
        normalized[:, 1] /= max(height, 1)

        regions.append(
            TextRegion(
                text=text_region.text,
                confidence=float(text_region.confidence),
                left=float(normalized[:, 0].min()),
                right=float(normalized[:, 0].max()),
                upper=float(normalized[:, 1].max()),
                lower=float(normalized[:, 1].min()),
                quad=normalized.tolist(),
            )
        )

    return OcrResult(regions=regions)
