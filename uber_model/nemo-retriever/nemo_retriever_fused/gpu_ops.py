# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident replacements for the pipeline's host-side image and box ops.

Each function here has a counterpart in the production pipeline that runs on the
host via PIL, numpy, or base64 round-trips. The versions in this module take and
return CUDA tensors so a page image can be decoded once and stay in device
memory through layout detection, table structure, OCR, and embedding.

The numerics intentionally track the host implementations:

* `letterbox_batch` reproduces `nemotron_page_elements_v3.model.resize_pad`,
  including the 114.0 pad value and the uint8 requantisation the production
  wrapper applies to match NIM preprocessing.
* `weighted_box_fusion` reproduces `remove_overlapping_boxes_using_wbf` from
  `nemo_retriever.models.nim.primitives.model_interface.yolox`.
* `tile_for_vl_tower` reproduces `dynamic_preprocess` plus `build_transform`
  from the VL model's `processing_llama_nemotron_vl` module.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms, nms

from nemo_retriever_fused.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SIGLIP_MEAN,
    SIGLIP_STD,
    YOLOX_PAD_VALUE,
)

__all__ = [
    "letterbox_batch",
    "letterbox_scale",
    "crop_regions",
    "pad_to_square",
    "resize_and_pad_for_ocr",
    "yolox_nms",
    "weighted_box_fusion",
    "apply_per_class_thresholds",
    "denormalize_boxes",
    "normalize_boxes",
    "tile_for_vl_tower",
]


def letterbox_scale(orig_hw: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Return the per-image letterbox scale factor.

    Parameters
    ----------
    orig_hw:
        ``[B, 2]`` int tensor of original ``(height, width)`` per image.
    target_hw:
        Target ``(height, width)`` after letterboxing.

    Returns
    -------
    torch.Tensor
        ``[B]`` float tensor of ``min(target_h / h, target_w / w)``.
    """
    target_h, target_w = target_hw
    heights = orig_hw[:, 0].to(torch.float32)
    widths = orig_hw[:, 1].to(torch.float32)
    return torch.minimum(target_h / heights, target_w / widths)


def letterbox_batch(
    images: list[torch.Tensor],
    target_hw: tuple[int, int],
    *,
    pad_value: float = YOLOX_PAD_VALUE,
    requantize_uint8: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Letterbox a ragged list of device images into one padded batch.

    This replaces the production path of one `resize_pad` call per page followed
    by a host-side `torch.stack`, and keeps everything on the images' device.

    Parameters
    ----------
    images:
        List of ``[C, H, W]`` float or uint8 CUDA tensors in 0-255 space. Sizes
        may differ between entries.
    target_hw:
        Target ``(height, width)``, normally ``(1024, 1024)``.
    pad_value:
        Fill value for the padded region. YOLOX expects 114.0 in 0-255 space.
    requantize_uint8:
        Clamp to ``[0, 255]`` and round-trip through uint8 after interpolation.
        The production wrapper does this to match NIM preprocessing bit for bit;
        turning it off saves two elementwise kernels at a small numeric cost.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``[B, C, target_h, target_w]`` float32 batch, and ``[B, 2]`` int32
        tensor of the original ``(height, width)`` needed to unscale boxes.
    """
    if not images:
        raise ValueError("letterbox_batch requires at least one image")

    device = images[0].device
    target_h, target_w = target_hw
    channels = images[0].shape[0]

    batch = torch.full(
        (len(images), channels, target_h, target_w),
        pad_value,
        dtype=torch.float32,
        device=device,
    )
    orig_hw = torch.empty((len(images), 2), dtype=torch.int32, device=device)

    for index, image in enumerate(images):
        if image.device != device:
            raise ValueError(f"letterbox_batch expects every image on {device}, got {image.device} at index {index}")
        _, height, width = image.shape
        orig_hw[index, 0] = height
        orig_hw[index, 1] = width

        scale = min(target_h / height, target_w / width)
        new_h = int(height * scale)
        new_w = int(width * scale)

        resized = F.interpolate(
            image.unsqueeze(0).to(torch.float32),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        resized = resized.clamp_(0.0, 255.0)
        if requantize_uint8:
            resized = resized.to(torch.uint8).to(torch.float32)

        # Writing into the pre-filled buffer avoids a separate F.pad allocation
        # and the copy that comes with it.
        batch[index, :, :new_h, :new_w] = resized

    return batch, orig_hw


def crop_regions(
    image: torch.Tensor,
    boxes_norm: torch.Tensor,
    *,
    min_side: int = 1,
) -> list[torch.Tensor]:
    """Crop normalized boxes out of a device image without leaving the GPU.

    Replaces the production path of base64 decode, PIL crop, numpy conversion,
    and PNG re-encode per region.

    Parameters
    ----------
    image:
        ``[C, H, W]`` CUDA tensor.
    boxes_norm:
        ``[N, 4]`` tensor of ``(x1, y1, x2, y2)`` in 0-1 normalized coordinates.
    min_side:
        Minimum crop height and width in pixels. Degenerate boxes are widened
        rather than dropped so that the returned list stays index-aligned with
        `boxes_norm`.

    Returns
    -------
    list[torch.Tensor]
        One ``[C, h, w]`` view or narrow copy per box, on the image's device.
    """
    _, height, width = image.shape
    if boxes_norm.numel() == 0:
        return []

    scale = torch.tensor([width, height, width, height], dtype=torch.float32, device=boxes_norm.device)
    pixels = (boxes_norm.to(torch.float32) * scale).round().to(torch.int64)

    x1 = pixels[:, 0].clamp(0, width - min_side)
    y1 = pixels[:, 1].clamp(0, height - min_side)
    x2 = pixels[:, 2].clamp(min_side, width)
    y2 = pixels[:, 3].clamp(min_side, height)
    x2 = torch.maximum(x2, x1 + min_side)
    y2 = torch.maximum(y2, y1 + min_side)

    # The bounds must be Python ints to slice, which is the one unavoidable
    # device-to-host read in this function. Doing it as a single `.tolist()`
    # costs one transfer of 4N integers instead of 4N scalar `.item()` syncs.
    bounds = torch.stack([x1, y1, x2, y2], dim=1).tolist()
    return [image[:, top:bottom, left:right] for left, top, right, bottom in bounds]


def pad_to_square(
    image: torch.Tensor, target_length: int, *, how: str = "bottom_right", value: float = 1.0
) -> torch.Tensor:
    """Pad a ``[C, H, W]`` device tensor to ``target_length`` on both sides.

    Mirrors `nemotron_ocr.inference.pre_processing.pad_to_square`.
    """
    _, height, width = image.shape
    if how == "center":
        pad_h = (target_length - height) // 2
        pad_w = (target_length - width) // 2
        padding = (pad_w, target_length - width - pad_w, pad_h, target_length - height - pad_h)
    elif how == "bottom_right":
        padding = (0, target_length - width, 0, target_length - height)
    else:
        raise ValueError(f"Unsupported padding method: {how}")
    return F.pad(image, padding, value=value)


def resize_and_pad_for_ocr(
    crops: list[torch.Tensor],
    *,
    infer_length: int,
    pad_how: str = "bottom_right",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, list[tuple[int, int]], list[int]]:
    """Build the OCR detector input batch directly from device crops.

    This is the fused replacement for `NemotronOCRV2._preprocess_batch`, which
    calls `_load_image_to_tensor_uint8` and therefore forces every input through
    `.detach().cpu()` before copying it back to the GPU. Starting from CUDA
    tensors skips that round-trip entirely.

    Parameters
    ----------
    crops:
        List of ``[C, H, W]`` CUDA tensors in 0-255 space (uint8 or float).
    infer_length:
        Detector input resolution, normally 1024.
    pad_how:
        Square-padding placement, matching the upstream `pad_how` setting.
    dtype:
        Compute dtype for the detector. Upstream uses float16.

    Returns
    -------
    tuple
        ``[N, C, infer_length, pad_width]`` batch, the original ``(H, W)`` per
        crop, and the square-padded side length per crop. The latter two are
        needed to rescale detected quads back to crop coordinates.
    """
    if not crops:
        raise ValueError("resize_and_pad_for_ocr requires at least one crop")

    device = crops[0].device
    # Upstream pads the detector width up to a multiple of 128.
    pad_width = -(-infer_length // 128) * 128

    batch = torch.empty((len(crops), 3, infer_length, pad_width), dtype=dtype, device=device)
    original_shapes: list[tuple[int, int]] = []
    padded_lengths: list[int] = []

    for index, crop in enumerate(crops):
        _, height, width = crop.shape
        original_shapes.append((height, width))
        padded_length = max(height, width)
        padded_lengths.append(padded_length)

        # Normalise in the target dtype so the divide and the cast fuse into one
        # pass rather than materialising an intermediate float32 tensor.
        scaled = crop.to(dtype).div_(255.0) if crop.dtype != dtype else crop.div(255.0)
        squared = pad_to_square(scaled, padded_length, how=pad_how, value=1.0)
        resized = F.interpolate(
            squared.unsqueeze(0),
            size=(infer_length, infer_length),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

        batch[index, :, :, :infer_length] = resized
        if pad_width > infer_length:
            batch[index, :, :, infer_length:] = 0.0

    return batch, original_shapes, padded_lengths


def yolox_nms(
    predictions: torch.Tensor,
    num_classes: int,
    conf_thresh: float,
    iou_thresh: float,
    *,
    class_agnostic: bool,
) -> list[torch.Tensor | None]:
    """Run YOLOX decode plus NMS on device.

    Equivalent to `nemotron_page_elements_v3.yolox.boxes.postprocess`, kept here
    so the fused pipeline can share one implementation between both detectors
    instead of importing two near-identical copies.

    Parameters
    ----------
    predictions:
        ``[B, A, 5 + num_classes]`` raw head output with boxes in
        ``(cx, cy, w, h)`` form.
    num_classes:
        Class count for the detector head.
    conf_thresh:
        Objectness times class-score gate applied before NMS.
    iou_thresh:
        NMS IoU threshold.
    class_agnostic:
        When True, suppress across classes; page-elements uses True and
        table-structure uses False.

    Returns
    -------
    list[torch.Tensor | None]
        Per image, a ``[N, 7]`` tensor of
        ``(x1, y1, x2, y2, obj_conf, class_conf, class_index)`` or None.
    """
    # Convert centre form to corner form in place on a single clone.
    box_corner = predictions.new_empty(predictions.shape[:2] + (4,))
    box_corner[..., 0] = predictions[..., 0] - predictions[..., 2] / 2
    box_corner[..., 1] = predictions[..., 1] - predictions[..., 3] / 2
    box_corner[..., 2] = predictions[..., 0] + predictions[..., 2] / 2
    box_corner[..., 3] = predictions[..., 1] + predictions[..., 3] / 2

    outputs: list[torch.Tensor | None] = []
    for image_pred, boxes in zip(predictions, box_corner):
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], dim=1, keepdim=True)
        keep = (image_pred[:, 4] * class_conf.squeeze(1)) >= conf_thresh
        if not bool(keep.any()):
            outputs.append(None)
            continue

        detections = torch.cat(
            (boxes, image_pred[:, 4:5], class_conf, class_pred.to(boxes.dtype)),
            dim=1,
        )[keep]

        scores = detections[:, 4] * detections[:, 5]
        if class_agnostic:
            kept = nms(detections[:, :4], scores, iou_thresh)
        else:
            kept = batched_nms(detections[:, :4], scores, detections[:, 6].to(torch.int64), iou_thresh)
        outputs.append(detections[kept])

    return outputs


def weighted_box_fusion(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    iou_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse overlapping same-class boxes by score-weighted averaging.

    The production pipeline performs this pass in numpy on the host after
    pulling NMS survivors off the GPU. Running it on device keeps the detection
    tensors resident and removes the synchronisation that the host copy forces.

    Parameters
    ----------
    boxes:
        ``[N, 4]`` tensor of ``(x1, y1, x2, y2)``.
    scores:
        ``[N]`` confidence scores.
    labels:
        ``[N]`` integer class indices.
    iou_thresh:
        Boxes of the same class overlapping above this IoU are fused.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Fused ``boxes``, ``scores``, and ``labels``, ordered by descending
        score within each class.
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    fused_boxes: list[torch.Tensor] = []
    fused_scores: list[torch.Tensor] = []
    fused_labels: list[torch.Tensor] = []

    for label in torch.unique(labels):
        mask = labels == label
        class_boxes = boxes[mask]
        class_scores = scores[mask]

        order = torch.argsort(class_scores, descending=True)
        class_boxes = class_boxes[order]
        class_scores = class_scores[order]

        # Pairwise IoU for this class, computed once as a matrix rather than in
        # the host-side double loop the numpy implementation uses.
        areas = (class_boxes[:, 2] - class_boxes[:, 0]).clamp(min=0) * (class_boxes[:, 3] - class_boxes[:, 1]).clamp(
            min=0
        )
        lt = torch.maximum(class_boxes[:, None, :2], class_boxes[None, :, :2])
        rb = torch.minimum(class_boxes[:, None, 2:], class_boxes[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = areas[:, None] + areas[None, :] - inter
        iou = inter / union.clamp(min=1e-9)

        claimed = torch.zeros(class_boxes.shape[0], dtype=torch.bool, device=boxes.device)
        overlaps = iou >= iou_thresh

        for index in range(class_boxes.shape[0]):
            if bool(claimed[index]):
                continue
            group = overlaps[index] & ~claimed
            group[index] = True
            claimed |= group

            weights = class_scores[group]
            weight_sum = weights.sum().clamp(min=1e-9)
            fused_boxes.append((class_boxes[group] * weights[:, None]).sum(dim=0) / weight_sum)
            fused_scores.append(weights.max())
            fused_labels.append(label)

    return (
        torch.stack(fused_boxes),
        torch.stack(fused_scores),
        torch.stack(fused_labels).to(labels.dtype),
    )


def apply_per_class_thresholds(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    thresholds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gate detections by a per-class score threshold, entirely on device.

    Replaces `postprocess_preds_page_element`, which copies boxes, labels, and
    scores to the host and builds the threshold vector with a Python list
    comprehension.

    Parameters
    ----------
    boxes, scores, labels:
        Detection tensors as returned by `weighted_box_fusion`.
    thresholds:
        ``[num_classes]`` tensor indexed by class id.
    """
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep = scores > thresholds[labels.to(torch.int64)]
    return boxes[keep], scores[keep], labels[keep]


def denormalize_boxes(boxes_norm: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Scale 0-1 normalized ``(x1, y1, x2, y2)`` boxes to pixel coordinates."""
    scale = torch.tensor([width, height, width, height], dtype=boxes_norm.dtype, device=boxes_norm.device)
    return boxes_norm * scale


def normalize_boxes(boxes_px: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Scale pixel ``(x1, y1, x2, y2)`` boxes into 0-1 normalized coordinates."""
    scale = torch.tensor([width, height, width, height], dtype=boxes_px.dtype, device=boxes_px.device)
    return boxes_px / scale


def _closest_tile_ratio(
    aspect_ratio: float,
    width: int,
    height: int,
    tile_size: int,
    min_tiles: int,
    max_tiles: int,
) -> tuple[int, int]:
    """Pick the tile grid for `tile_for_vl_tower`.

    This is the scalar search from `find_closest_aspect_ratio` in the VL
    processor. It stays on the host because it only reads two integers and
    returns two more; moving it to the device would cost a synchronisation to
    read the result back.
    """
    candidates = sorted(
        {
            (i, j)
            for n in range(min_tiles, max_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_tiles <= i * j <= max_tiles
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )

    area = width * height
    best_ratio = (1, 1)
    best_factor = float("-inf")
    for ratio in candidates:
        target_aspect = ratio[0] / ratio[1]
        area_ratio = (ratio[0] * ratio[1] * tile_size * tile_size) / area
        factor = min(area_ratio, 0.6) * min(target_aspect / aspect_ratio, aspect_ratio / target_aspect)
        if factor > best_factor:
            best_factor = factor
            best_ratio = ratio
    return best_ratio


def tile_for_vl_tower(
    image: torch.Tensor,
    *,
    tile_size: int,
    min_tiles: int,
    max_tiles: int,
    use_thumbnail: bool,
    norm_type: str = "imagenet",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Tile and normalize a device image for the VL vision tower.

    Fused replacement for `dynamic_preprocess` followed by `build_transform`
    in the VL model's processor. Upstream performs a PIL bicubic resize, up to
    `max_tiles` PIL crops, and one `ToTensor` plus `Normalize` per tile, all on
    the host. This version does the resize with `F.interpolate`, produces the
    tiles as a strided view, and folds the normalisation into a single kernel.

    Parameters
    ----------
    image:
        ``[C, H, W]`` CUDA tensor in 0-255 space.
    tile_size:
        Tile side length, 448 for this model family.
    min_tiles, max_tiles:
        Bounds on the tile count, matching the processor's `min_num`/`max_num`.
    use_thumbnail:
        Append a whole-image thumbnail tile when the grid has more than one
        tile, matching upstream behaviour.
    norm_type:
        ``"imagenet"`` or ``"siglip"`` normalisation constants.
    dtype:
        Output dtype. The VL tower ships bfloat16 weights.

    Returns
    -------
    torch.Tensor
        ``[num_tiles, C, tile_size, tile_size]`` normalized pixel values.
    """
    channels, height, width = image.shape
    cols, rows = _closest_tile_ratio(width / height, width, height, tile_size, min_tiles, max_tiles)
    target_w = tile_size * cols
    target_h = tile_size * rows

    resized = F.interpolate(
        image.unsqueeze(0).to(torch.float32),
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)

    # Reshape into tiles instead of cropping. This is a view plus one permute
    # copy rather than `rows * cols` separate crop allocations.
    tiles = (
        resized.reshape(channels, rows, tile_size, cols, tile_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(rows * cols, channels, tile_size, tile_size)
    )

    if use_thumbnail and tiles.shape[0] != 1:
        thumbnail = F.interpolate(
            image.unsqueeze(0).to(torch.float32),
            size=(tile_size, tile_size),
            mode="bicubic",
            align_corners=False,
        )
        tiles = torch.cat([tiles, thumbnail], dim=0)

    mean_values, std_values = (IMAGENET_MEAN, IMAGENET_STD) if norm_type == "imagenet" else (SIGLIP_MEAN, SIGLIP_STD)
    mean = torch.tensor(mean_values, dtype=torch.float32, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(std_values, dtype=torch.float32, device=image.device).view(1, 3, 1, 1)

    # ToTensor scales to 0-1, then Normalize subtracts and divides. Folding all
    # three into one expression keeps it to a single elementwise pass.
    tiles = tiles.clamp_(0.0, 255.0).div_(255.0).sub_(mean).div_(std)
    return tiles.to(dtype)
