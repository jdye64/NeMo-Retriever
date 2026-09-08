# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused YOLOX detector stages for page elements and table structure.

Both `nvidia/nemotron-page-elements-v3` and `nvidia/nemotron-table-structure-v1`
are YOLOX models with identical geometry: depth 1.0, width 1.0,
`in_channels=[256, 512, 1024]`, and a 1024x1024 letterboxed input. Only the head
class count (6 versus 5) and the NMS parameters differ. That lets one detector
implementation and one preprocessing kernel serve both stages, and lets the two
forward passes overlap on separate CUDA streams once the page tensor exists.

The production path instead calls each detector through its own wrapper, each of
which runs `resize_pad` per image on the host side of the call and copies the
prediction tensors to the host before postprocessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from nemo_retriever_fused import gpu_ops
from nemo_retriever_fused.config import (
    PAGE_ELEMENT_LABELS,
    TABLE_STRUCTURE_LABELS,
    PageElementsConfig,
    TableStructureConfig,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Detections:
    """Detection output for one image, kept on device.

    Attributes
    ----------
    boxes:
        ``[N, 4]`` normalized ``(x1, y1, x2, y2)`` in 0-1 page coordinates.
    scores:
        ``[N]`` confidence scores.
    labels:
        ``[N]`` integer class indices into the stage's label tuple.
    """

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def select(self, label_names: tuple[str, ...], wanted: set[str]) -> Detections:
        """Return the subset of detections whose class name is in *wanted*."""
        if not len(self):
            return self
        wanted_ids = [index for index, name in enumerate(label_names) if name in wanted]
        if not wanted_ids:
            empty = self.boxes.new_zeros((0, 4))
            return Detections(empty, self.scores.new_zeros((0,)), self.labels.new_zeros((0,)))
        keep = torch.isin(self.labels, torch.tensor(wanted_ids, device=self.labels.device))
        return Detections(self.boxes[keep], self.scores[keep], self.labels[keep])


class FusedYoloxDetector:
    """A single YOLOX detector wrapped for device-resident inference.

    Parameters
    ----------
    module:
        The loaded YOLOX `nn.Module`, already on the target device in eval mode.
        This is the raw model, not the upstream `YoloXWrapper`, because the
        wrapper's `forward` embeds host-side box scaling that this class does on
        device instead.
    num_classes:
        Head class count.
    label_names:
        Class names indexed by class id.
    input_size:
        Letterbox target ``(height, width)``.
    conf_thresh, iou_thresh, class_agnostic, min_bbox_size:
        NMS and filtering parameters for this detector.
    device:
        Device the module lives on.
    autocast_dtype:
        Autocast dtype for the forward pass.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        num_classes: int,
        label_names: tuple[str, ...],
        input_size: tuple[int, int],
        conf_thresh: float,
        iou_thresh: float,
        class_agnostic: bool,
        min_bbox_size: int,
        device: torch.device,
        autocast_dtype: torch.dtype,
    ) -> None:
        self._module = module
        self._num_classes = num_classes
        self._label_names = label_names
        self._input_size = input_size
        self._conf_thresh = conf_thresh
        self._iou_thresh = iou_thresh
        self._class_agnostic = class_agnostic
        self._min_bbox_size = min_bbox_size
        self._device = device
        self._autocast_dtype = autocast_dtype

    @property
    def label_names(self) -> tuple[str, ...]:
        """Return the class names indexed by class id."""
        return self._label_names

    @property
    def input_size(self) -> tuple[int, int]:
        """Return the letterbox target ``(height, width)``."""
        return self._input_size

    def preprocess(self, images: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Letterbox *images* into a single device batch.

        Parameters
        ----------
        images:
            ``[C, H, W]`` CUDA tensors in 0-255 space, sizes may differ.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The ``[B, C, h, w]`` batch and the ``[B, 2]`` original sizes.
        """
        return gpu_ops.letterbox_batch(images, self._input_size)

    def forward(self, batch: torch.Tensor, orig_hw: torch.Tensor) -> list[Detections]:
        """Run the detector and decode boxes without leaving the device.

        Parameters
        ----------
        batch:
            ``[B, C, h, w]`` letterboxed float batch on the detector's device.
        orig_hw:
            ``[B, 2]`` int tensor of original ``(height, width)`` per image.

        Returns
        -------
        list[Detections]
            One entry per image, with normalized boxes on device.
        """
        with torch.inference_mode():
            with torch.autocast(device_type=self._device.type, dtype=self._autocast_dtype):
                raw = self._module(batch)

        # The head emits float16 under autocast; NMS and the box arithmetic want
        # float32 for stable IoU, and the cast is cheap relative to the forward.
        predictions = gpu_ops.yolox_nms(
            raw.to(torch.float32),
            self._num_classes,
            self._conf_thresh,
            self._iou_thresh,
            class_agnostic=self._class_agnostic,
        )

        scale = gpu_ops.letterbox_scale(orig_hw, self._input_size)
        results: list[Detections] = []

        for index, detection in enumerate(predictions):
            if detection is None or detection.shape[0] == 0:
                empty = batch.new_zeros((0, 4), dtype=torch.float32)
                results.append(
                    Detections(
                        boxes=empty,
                        scores=batch.new_zeros((0,), dtype=torch.float32),
                        labels=torch.zeros((0,), dtype=torch.int64, device=batch.device),
                    )
                )
                continue

            height = orig_hw[index, 0].to(torch.float32)
            width = orig_hw[index, 1].to(torch.float32)

            # Undo the letterbox scale, clamp to the page, drop degenerate
            # boxes, and normalize. All of this stays on device; the upstream
            # wrapper does the same arithmetic but the caller then copies the
            # result to the host before postprocessing.
            boxes = detection[:, :4] / scale[index]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0).clamp(max=width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0).clamp(max=height)

            keep = ((boxes[:, 2] - boxes[:, 0]) > self._min_bbox_size) & (
                (boxes[:, 3] - boxes[:, 1]) > self._min_bbox_size
            )
            boxes = boxes[keep]
            detection = detection[keep]

            boxes[:, [0, 2]] /= width
            boxes[:, [1, 3]] /= height

            results.append(
                Detections(
                    boxes=boxes,
                    scores=detection[:, 4] * detection[:, 5],
                    labels=detection[:, 6].to(torch.int64),
                )
            )

        return results


class PageElementsStage:
    """Layout detection with device-resident weighted box fusion.

    The production pipeline runs NMS on the GPU, copies boxes, scores, and
    labels to the host, then applies weighted box fusion and the per-class score
    gate in numpy. This stage keeps all three passes on device, which removes
    the synchronisation that the copy forces at the end of every page.
    """

    def __init__(self, detector: FusedYoloxDetector, config: PageElementsConfig) -> None:
        self._detector = detector
        self._config = config
        self._thresholds = torch.tensor(
            [config.thresholds_per_class[name] for name in PAGE_ELEMENT_LABELS],
            dtype=torch.float32,
            device=detector._device,
        )

    @property
    def label_names(self) -> tuple[str, ...]:
        """Return the page element class names."""
        return PAGE_ELEMENT_LABELS

    def run(self, images: list[torch.Tensor]) -> list[Detections]:
        """Detect page elements for every image in *images*.

        Parameters
        ----------
        images:
            ``[C, H, W]`` CUDA tensors in 0-255 space.

        Returns
        -------
        list[Detections]
            Fused and thresholded detections, one entry per image, on device.
        """
        batch, orig_hw = self._detector.preprocess(images)
        raw = self._detector.forward(batch, orig_hw)

        fused: list[Detections] = []
        for detections in raw:
            if not len(detections):
                fused.append(detections)
                continue
            boxes, scores, labels = gpu_ops.weighted_box_fusion(
                detections.boxes,
                detections.scores,
                detections.labels,
                iou_thresh=self._config.wbf_iou_thresh,
            )
            boxes, scores, labels = gpu_ops.apply_per_class_thresholds(boxes, scores, labels, self._thresholds)
            fused.append(Detections(boxes=boxes, scores=scores, labels=labels))

        return fused


class TableStructureStage:
    """Cell, row, and column detection over table crops.

    The production pipeline crops each table by decoding the page's base64,
    cropping with PIL, converting to numpy, and building a fresh torch tensor.
    This stage crops directly from the resident page tensor and batches every
    table on the page into one detector call. The upstream local wrapper also
    discards all but the first batch element in `invoke`, so batching there was
    not possible at all.
    """

    def __init__(self, detector: FusedYoloxDetector, config: TableStructureConfig) -> None:
        self._detector = detector
        self._config = config

    @property
    def label_names(self) -> tuple[str, ...]:
        """Return the table structure class names."""
        return TABLE_STRUCTURE_LABELS

    def run(self, crops: list[torch.Tensor]) -> list[Detections]:
        """Detect table structure for every crop in *crops*.

        Parameters
        ----------
        crops:
            ``[C, h, w]`` CUDA tensors, one per detected table region.

        Returns
        -------
        list[Detections]
            Structure detections in crop-normalized coordinates, on device.
        """
        if not crops:
            return []

        batch, orig_hw = self._detector.preprocess(crops)
        raw = self._detector.forward(batch, orig_hw)

        gated: list[Detections] = []
        for detections in raw:
            if not len(detections):
                gated.append(detections)
                continue
            keep = detections.scores > self._config.min_score
            gated.append(
                Detections(
                    boxes=detections.boxes[keep],
                    scores=detections.scores[keep],
                    labels=detections.labels[keep],
                )
            )
        return gated
