# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the device-resident image and box operations.

These run on CPU tensors so they execute in CI without a GPU. The operations are
device-agnostic torch, so correctness here implies correctness on CUDA; the
CUDA-specific behaviour (stream overlap, nvJPEG) is covered separately in tests
marked `requires_cuda`.
"""

from __future__ import annotations

import pytest
import torch

from nemo_retriever_fused import gpu_ops
from nemo_retriever_fused.config import YOLOX_PAD_VALUE

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


class TestLetterbox:
    def test_preserves_aspect_ratio_and_pads_remainder(self) -> None:
        # A 100x200 image letterboxed to 64x64 should occupy 32x64 and leave the
        # bottom 32 rows at the pad value.
        image = torch.full((3, 100, 200), 200.0)
        batch, orig_hw = gpu_ops.letterbox_batch([image], (64, 64))

        assert batch.shape == (1, 3, 64, 64)
        assert orig_hw.tolist() == [[100, 200]]
        assert torch.allclose(batch[0, :, :32, :], torch.full((3, 32, 64), 200.0))
        assert torch.allclose(batch[0, :, 32:, :], torch.full((3, 32, 64), YOLOX_PAD_VALUE))

    def test_batches_ragged_sizes(self) -> None:
        images = [
            torch.full((3, 100, 200), 10.0),
            torch.full((3, 300, 150), 20.0),
            torch.full((3, 64, 64), 30.0),
        ]
        batch, orig_hw = gpu_ops.letterbox_batch(images, (128, 128))

        assert batch.shape == (3, 3, 128, 128)
        assert orig_hw.tolist() == [[100, 200], [300, 150], [64, 64]]

    def test_scale_matches_upstream_formula(self) -> None:
        orig_hw = torch.tensor([[100, 200], [300, 150]], dtype=torch.int32)
        scale = gpu_ops.letterbox_scale(orig_hw, (1024, 1024))

        assert scale[0].item() == pytest.approx(1024 / 200)
        assert scale[1].item() == pytest.approx(1024 / 300)

    def test_requires_at_least_one_image(self) -> None:
        with pytest.raises(ValueError, match="at least one image"):
            gpu_ops.letterbox_batch([], (64, 64))

    def test_requantization_can_be_disabled(self) -> None:
        image = torch.full((3, 10, 10), 127.5)
        quantized, _ = gpu_ops.letterbox_batch([image], (10, 10), requantize_uint8=True)
        exact, _ = gpu_ops.letterbox_batch([image], (10, 10), requantize_uint8=False)

        assert quantized[0, 0, 0, 0].item() == 127.0
        assert exact[0, 0, 0, 0].item() == pytest.approx(127.5)


class TestCropRegions:
    def test_crops_match_normalized_boxes(self) -> None:
        image = torch.arange(3 * 100 * 200, dtype=torch.float32).reshape(3, 100, 200)
        boxes = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])

        crops = gpu_ops.crop_regions(image, boxes)

        assert len(crops) == 2
        assert crops[0].shape == (3, 50, 100)
        assert crops[1].shape == (3, 50, 100)
        assert torch.equal(crops[0], image[:, 0:50, 0:100])

    def test_empty_boxes_produce_no_crops(self) -> None:
        image = torch.zeros((3, 10, 10))
        assert gpu_ops.crop_regions(image, torch.zeros((0, 4))) == []

    def test_degenerate_box_is_widened_not_dropped(self) -> None:
        # Index alignment with the detection tensor matters, so a zero-area box
        # must still yield a crop.
        image = torch.zeros((3, 100, 100))
        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

        crops = gpu_ops.crop_regions(image, boxes)

        assert len(crops) == 1
        assert crops[0].shape[1] >= 1 and crops[0].shape[2] >= 1

    def test_out_of_range_box_is_clamped_to_the_page(self) -> None:
        image = torch.zeros((3, 50, 50))
        boxes = torch.tensor([[-0.5, -0.5, 1.5, 1.5]])

        crops = gpu_ops.crop_regions(image, boxes)

        assert crops[0].shape == (3, 50, 50)


class TestOcrPreprocessing:
    def test_builds_padded_detector_batch(self) -> None:
        crops = [torch.full((3, 40, 90), 255.0), torch.full((3, 70, 20), 128.0)]

        batch, shapes, padded = gpu_ops.resize_and_pad_for_ocr(crops, infer_length=256, dtype=torch.float32)

        # 256 rounds up to 256 for the width, since 256 is already a multiple of 128.
        assert batch.shape == (2, 3, 256, 256)
        assert shapes == [(40, 90), (70, 20)]
        assert padded == [90, 70]

    def test_pads_width_to_multiple_of_128(self) -> None:
        crops = [torch.full((3, 10, 10), 255.0)]

        batch, _, _ = gpu_ops.resize_and_pad_for_ocr(crops, infer_length=200, dtype=torch.float32)

        assert batch.shape == (1, 3, 200, 256)
        # The pad columns beyond infer_length are zero, matching upstream.
        assert torch.all(batch[0, :, :, 200:] == 0.0)

    def test_normalizes_to_unit_range(self) -> None:
        crops = [torch.full((3, 32, 32), 255.0)]

        batch, _, _ = gpu_ops.resize_and_pad_for_ocr(crops, infer_length=128, dtype=torch.float32)

        assert batch.max().item() == pytest.approx(1.0)


class TestPadToSquare:
    def test_bottom_right_padding(self) -> None:
        image = torch.zeros((3, 4, 8))
        padded = gpu_ops.pad_to_square(image, 8, how="bottom_right", value=1.0)

        assert padded.shape == (3, 8, 8)
        assert torch.all(padded[:, 4:, :] == 1.0)

    def test_center_padding_splits_evenly(self) -> None:
        image = torch.zeros((3, 4, 8))
        padded = gpu_ops.pad_to_square(image, 8, how="center", value=1.0)

        assert padded.shape == (3, 8, 8)
        assert torch.all(padded[:, :2, :] == 1.0)
        assert torch.all(padded[:, 6:, :] == 1.0)

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="Unsupported padding method"):
            gpu_ops.pad_to_square(torch.zeros((3, 4, 4)), 8, how="corner")


class TestWeightedBoxFusion:
    def test_fuses_overlapping_same_class_boxes(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5]])
        scores = torch.tensor([0.9, 0.7])
        labels = torch.tensor([0, 0])

        fused_boxes, fused_scores, fused_labels = gpu_ops.weighted_box_fusion(boxes, scores, labels, iou_thresh=0.5)

        assert fused_boxes.shape == (1, 4)
        # Score-weighted average of the two corners.
        assert fused_boxes[0, 0].item() == pytest.approx((0.0 * 0.9 + 0.5 * 0.7) / 1.6)
        assert fused_scores[0].item() == pytest.approx(0.9)
        assert fused_labels.tolist() == [0]

    def test_keeps_different_classes_separate(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 1])

        fused_boxes, _, fused_labels = gpu_ops.weighted_box_fusion(boxes, scores, labels, iou_thresh=0.5)

        assert fused_boxes.shape == (2, 4)
        assert sorted(fused_labels.tolist()) == [0, 1]

    def test_keeps_disjoint_boxes_separate(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 5.0, 5.0], [50.0, 50.0, 55.0, 55.0]])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 0])

        fused_boxes, _, _ = gpu_ops.weighted_box_fusion(boxes, scores, labels, iou_thresh=0.5)

        assert fused_boxes.shape == (2, 4)

    def test_empty_input_round_trips(self) -> None:
        empty_boxes = torch.zeros((0, 4))
        boxes, scores, labels = gpu_ops.weighted_box_fusion(
            empty_boxes, torch.zeros((0,)), torch.zeros((0,), dtype=torch.int64), iou_thresh=0.5
        )
        assert boxes.shape == (0, 4)
        assert scores.numel() == 0
        assert labels.numel() == 0


class TestPerClassThresholds:
    def test_gates_by_class_specific_threshold(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 3)
        scores = torch.tensor([0.05, 0.5, 0.05])
        labels = torch.tensor([0, 0, 1])
        # Class 0 needs 0.1, class 1 needs 0.01.
        thresholds = torch.tensor([0.1, 0.01])

        kept_boxes, kept_scores, kept_labels = gpu_ops.apply_per_class_thresholds(boxes, scores, labels, thresholds)

        assert kept_boxes.shape == (2, 4)
        assert kept_scores.tolist() == pytest.approx([0.5, 0.05])
        assert kept_labels.tolist() == [0, 1]

    def test_empty_input_round_trips(self) -> None:
        boxes, scores, labels = gpu_ops.apply_per_class_thresholds(
            torch.zeros((0, 4)),
            torch.zeros((0,)),
            torch.zeros((0,), dtype=torch.int64),
            torch.tensor([0.1]),
        )
        assert boxes.shape == (0, 4)


class TestBoxScaling:
    def test_denormalize_and_normalize_are_inverses(self) -> None:
        boxes = torch.tensor([[0.1, 0.2, 0.7, 0.9]])
        pixels = gpu_ops.denormalize_boxes(boxes, height=200, width=100)

        assert pixels[0].tolist() == pytest.approx([10.0, 40.0, 70.0, 180.0])
        assert gpu_ops.normalize_boxes(pixels, height=200, width=100)[0].tolist() == pytest.approx(boxes[0].tolist())


class TestVlTiling:
    def test_tile_count_matches_grid_plus_thumbnail(self) -> None:
        image = torch.full((3, 896, 448), 255.0)

        tiles = gpu_ops.tile_for_vl_tower(
            image,
            tile_size=448,
            min_tiles=1,
            max_tiles=6,
            use_thumbnail=True,
            dtype=torch.float32,
        )

        assert tiles.shape[1:] == (3, 448, 448)
        # A grid plus one thumbnail, and the thumbnail only when the grid is
        # larger than a single tile.
        assert tiles.shape[0] >= 2

    def test_single_tile_grid_omits_thumbnail(self) -> None:
        image = torch.full((3, 448, 448), 255.0)

        tiles = gpu_ops.tile_for_vl_tower(
            image,
            tile_size=448,
            min_tiles=1,
            max_tiles=1,
            use_thumbnail=True,
            dtype=torch.float32,
        )

        assert tiles.shape[0] == 1

    def test_imagenet_normalization_is_applied(self) -> None:
        # A pure white image maps to (1.0 - mean) / std per channel.
        image = torch.full((3, 448, 448), 255.0)

        tiles = gpu_ops.tile_for_vl_tower(
            image,
            tile_size=448,
            min_tiles=1,
            max_tiles=1,
            use_thumbnail=False,
            norm_type="imagenet",
            dtype=torch.float32,
        )

        assert tiles[0, 0].mean().item() == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-4)

    def test_siglip_normalization_differs_from_imagenet(self) -> None:
        image = torch.full((3, 448, 448), 255.0)
        kwargs = {
            "tile_size": 448,
            "min_tiles": 1,
            "max_tiles": 1,
            "use_thumbnail": False,
            "dtype": torch.float32,
        }

        imagenet = gpu_ops.tile_for_vl_tower(image, norm_type="imagenet", **kwargs)
        siglip = gpu_ops.tile_for_vl_tower(image, norm_type="siglip", **kwargs)

        assert siglip[0, 0].mean().item() == pytest.approx(1.0, abs=1e-4)
        assert not torch.allclose(imagenet, siglip)

    def test_tiles_are_contiguous_slices_of_the_resize(self) -> None:
        # Build an image whose left and right halves differ so the tiling order
        # is observable.
        image = torch.zeros((3, 448, 896))
        image[:, :, 448:] = 255.0

        tiles = gpu_ops.tile_for_vl_tower(
            image,
            tile_size=448,
            min_tiles=2,
            max_tiles=2,
            use_thumbnail=False,
            norm_type="siglip",
            dtype=torch.float32,
        )

        assert tiles.shape[0] == 2
        assert tiles[0].mean().item() < tiles[1].mean().item()


class TestYoloxNms:
    def _predictions(self) -> torch.Tensor:
        # Two anchors, three classes. Anchor 0 is a confident class-1 box and
        # anchor 1 heavily overlaps it with a lower score.
        predictions = torch.zeros((1, 2, 8))
        predictions[0, 0, :4] = torch.tensor([50.0, 50.0, 20.0, 20.0])
        predictions[0, 0, 4] = 0.9
        predictions[0, 0, 6] = 0.8
        predictions[0, 1, :4] = torch.tensor([51.0, 51.0, 20.0, 20.0])
        predictions[0, 1, 4] = 0.5
        predictions[0, 1, 6] = 0.4
        return predictions

    def test_converts_center_to_corner_form(self) -> None:
        results = gpu_ops.yolox_nms(
            self._predictions(), num_classes=3, conf_thresh=0.01, iou_thresh=0.9, class_agnostic=True
        )

        assert results[0] is not None
        # cx=50, w=20 becomes x1=40, x2=60.
        assert results[0][0, 0].item() == pytest.approx(40.0)
        assert results[0][0, 2].item() == pytest.approx(60.0)

    def test_suppresses_overlapping_boxes(self) -> None:
        results = gpu_ops.yolox_nms(
            self._predictions(), num_classes=3, conf_thresh=0.01, iou_thresh=0.3, class_agnostic=True
        )

        assert results[0] is not None
        assert results[0].shape[0] == 1

    def test_confidence_gate_can_drop_every_box(self) -> None:
        results = gpu_ops.yolox_nms(
            self._predictions(), num_classes=3, conf_thresh=0.99, iou_thresh=0.5, class_agnostic=True
        )

        assert results[0] is None

    def test_class_aware_nms_keeps_distinct_classes(self) -> None:
        predictions = self._predictions()
        # Make anchor 1 a confident class-2 box at the same location.
        predictions[0, 1, 5:] = torch.tensor([0.0, 0.0, 0.9])
        predictions[0, 1, 4] = 0.9

        agnostic = gpu_ops.yolox_nms(predictions, num_classes=3, conf_thresh=0.01, iou_thresh=0.3, class_agnostic=True)
        aware = gpu_ops.yolox_nms(predictions, num_classes=3, conf_thresh=0.01, iou_thresh=0.3, class_agnostic=False)

        assert agnostic[0].shape[0] == 1
        assert aware[0].shape[0] == 2


@requires_cuda
class TestCudaResidency:
    def test_letterbox_output_stays_on_device(self) -> None:
        image = torch.full((3, 100, 200), 200.0, device="cuda")
        batch, orig_hw = gpu_ops.letterbox_batch([image], (256, 256))

        assert batch.is_cuda
        assert orig_hw.is_cuda

    def test_mixed_devices_are_rejected(self) -> None:
        images = [torch.zeros((3, 10, 10), device="cuda"), torch.zeros((3, 10, 10))]

        with pytest.raises(ValueError, match="expects every image on"):
            gpu_ops.letterbox_batch(images, (32, 32))

    def test_crops_are_views_of_the_page(self) -> None:
        image = torch.zeros((3, 100, 100), device="cuda")
        crops = gpu_ops.crop_regions(image, torch.tensor([[0.0, 0.0, 0.5, 0.5]], device="cuda"))

        # A slice shares storage with the page, so cropping allocates nothing.
        assert crops[0].data_ptr() == image.data_ptr()

    def test_vl_tiling_stays_on_device(self) -> None:
        image = torch.full((3, 448, 896), 255.0, device="cuda")
        tiles = gpu_ops.tile_for_vl_tower(image, tile_size=448, min_tiles=2, max_tiles=2, use_thumbnail=False)

        assert tiles.is_cuda
