# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pipeline plumbing that does not need model weights.

Covered here: the device image batch, configuration round-tripping, table
assembly, result serialisation, and the transfer accounting used by the
benchmark. The stage adapters that need real weights are exercised by the
CUDA integration tests.
"""

from __future__ import annotations

import base64
import io

import pytest
import torch

from nemo_retriever_fused.config import (
    TABLE_STRUCTURE_LABELS,
    EmbedConfig,
    FusedPipelineConfig,
    OCRConfig,
)
from nemo_retriever_fused.gpu_image import DeviceImageBatch, decode_base64_image
from nemo_retriever_fused.results import ElementResult, FusedResult, PageResult, StageTiming
from nemo_retriever_fused.stages.detectors import Detections
from nemo_retriever_fused.stages.ocr import OcrResult, TextRegion, _sequence_confidence
from nemo_retriever_fused.table_assembly import assemble_table_markdown


def _png_bytes(width: int = 32, height: int = 16, value: int = 200) -> bytes:
    from PIL import Image

    import numpy as np

    buffer = io.BytesIO()
    Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes(width: int = 32, height: int = 16, value: int = 200) -> bytes:
    from PIL import Image

    import numpy as np

    buffer = io.BytesIO()
    Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)).save(buffer, format="JPEG")
    return buffer.getvalue()


class TestDecode:
    def test_decodes_base64_png(self) -> None:
        payload = base64.b64encode(_png_bytes()).decode("ascii")
        image = decode_base64_image(payload, device=torch.device("cpu"))

        assert image.shape == (3, 16, 32)
        assert image.dtype == torch.uint8

    def test_decodes_base64_jpeg(self) -> None:
        payload = base64.b64encode(_jpeg_bytes()).decode("ascii")
        image = decode_base64_image(payload, device=torch.device("cpu"))

        assert image.shape == (3, 16, 32)

    def test_strips_data_url_prefix(self) -> None:
        body = base64.b64encode(_png_bytes()).decode("ascii")
        image = decode_base64_image(f"data:image/png;base64,{body}", device=torch.device("cpu"))

        assert image.shape == (3, 16, 32)

    def test_accepts_raw_bytes(self) -> None:
        image = decode_base64_image(_png_bytes(), device=torch.device("cpu"))

        assert image.shape == (3, 16, 32)

    def test_rejects_empty_payload(self) -> None:
        with pytest.raises(ValueError, match="zero bytes"):
            decode_base64_image("", device=torch.device("cpu"))


class TestDeviceImageBatch:
    def test_builds_from_base64(self) -> None:
        payloads = [base64.b64encode(_png_bytes(w, h)).decode("ascii") for w, h in [(32, 16), (64, 48)]]
        batch = DeviceImageBatch.from_base64(payloads, device="cpu")

        assert len(batch) == 2
        assert batch.shapes == [(16, 32), (48, 64)]
        assert batch.page_ids == ["0", "1"]

    def test_carries_page_ids(self) -> None:
        payload = base64.b64encode(_png_bytes()).decode("ascii")
        batch = DeviceImageBatch.from_base64([payload], device="cpu", page_ids=["doc-7-p3"])

        assert batch.page_ids == ["doc-7-p3"]

    def test_accepts_channel_last_tensors(self) -> None:
        image = torch.zeros((16, 32, 3), dtype=torch.uint8)
        batch = DeviceImageBatch.from_tensors([image], device="cpu")

        assert batch.images[0].shape == (3, 16, 32)

    def test_rejects_mismatched_page_id_count(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            DeviceImageBatch(images=[torch.zeros((3, 4, 4))], page_ids=["a", "b"], device=torch.device("cpu"))

    def test_rejects_non_three_channel_images(self) -> None:
        with pytest.raises(ValueError, match=r"must be \[3, H, W\]"):
            DeviceImageBatch(images=[torch.zeros((1, 4, 4))], page_ids=["a"], device=torch.device("cpu"))

    def test_as_float_preserves_range(self) -> None:
        image = torch.full((3, 4, 4), 255, dtype=torch.uint8)
        batch = DeviceImageBatch.from_tensors([image], device="cpu")

        assert batch.as_float()[0].max().item() == 255.0

    def test_nbytes_reports_device_footprint(self) -> None:
        batch = DeviceImageBatch.from_tensors([torch.zeros((3, 10, 10), dtype=torch.uint8)], device="cpu")

        assert batch.nbytes() == 300


class TestDetections:
    def test_select_filters_by_label_name(self) -> None:
        detections = Detections(
            boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]]),
            scores=torch.tensor([0.9, 0.8]),
            labels=torch.tensor([0, 2]),
        )

        selected = detections.select(("table", "chart", "title"), {"title"})

        assert len(selected) == 1
        assert selected.labels.tolist() == [2]

    def test_select_with_no_matching_label_returns_empty(self) -> None:
        detections = Detections(
            boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([0]),
        )

        assert len(detections.select(("table",), {"chart"})) == 0


class TestTableAssembly:
    def _structure(self) -> Detections:
        row_id = TABLE_STRUCTURE_LABELS.index("row")
        column_id = TABLE_STRUCTURE_LABELS.index("column")
        # Two rows stacked vertically, two columns side by side.
        return Detections(
            boxes=torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.5, 1.0, 1.0],
                    [0.0, 0.0, 0.5, 1.0],
                    [0.5, 0.0, 1.0, 1.0],
                ]
            ),
            scores=torch.tensor([0.9, 0.9, 0.9, 0.9]),
            labels=torch.tensor([row_id, row_id, column_id, column_id]),
        )

    def _region(self, text: str, left: float, right: float, low: float, high: float) -> TextRegion:
        return TextRegion(
            text=text,
            confidence=0.9,
            left=left,
            right=right,
            upper=high,
            lower=low,
            quad=[[left, low], [right, low], [right, high], [left, high]],
        )

    def test_places_text_into_the_right_cells(self) -> None:
        ocr = OcrResult(
            regions=[
                self._region("Name", 0.1, 0.3, 0.1, 0.2),
                self._region("Total", 0.6, 0.8, 0.1, 0.2),
                self._region("Widget", 0.1, 0.3, 0.6, 0.7),
                self._region("42", 0.6, 0.8, 0.6, 0.7),
            ]
        )

        markdown = assemble_table_markdown(self._structure(), ocr, label_names=TABLE_STRUCTURE_LABELS)

        lines = markdown.splitlines()
        assert lines[0] == "| Name | Total |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| Widget | 42 |"

    def test_escapes_pipes_in_cell_text(self) -> None:
        ocr = OcrResult(regions=[self._region("a|b", 0.1, 0.3, 0.1, 0.2)])

        markdown = assemble_table_markdown(self._structure(), ocr, label_names=TABLE_STRUCTURE_LABELS)

        assert "a\\|b" in markdown

    def test_falls_back_to_raw_text_without_structure(self) -> None:
        empty = Detections(
            boxes=torch.zeros((0, 4)),
            scores=torch.zeros((0,)),
            labels=torch.zeros((0,), dtype=torch.int64),
        )
        ocr = OcrResult(regions=[self._region("orphan", 0.1, 0.3, 0.1, 0.2)])

        assert assemble_table_markdown(empty, ocr, label_names=TABLE_STRUCTURE_LABELS) == "orphan"

    def test_header_row_can_be_disabled(self) -> None:
        ocr = OcrResult(regions=[self._region("Name", 0.1, 0.3, 0.1, 0.2)])

        markdown = assemble_table_markdown(
            self._structure(), ocr, label_names=TABLE_STRUCTURE_LABELS, header_first_row=False
        )

        assert "| --- |" not in markdown


class TestOcrConfidence:
    def test_excludes_padding_and_post_eos_tokens(self) -> None:
        # Token id 1 is end-of-sequence and 0 is padding.
        sequence_ids = torch.tensor([[5, 6, 1, 0]])
        sequence_probs = torch.tensor([[0.9, 0.81, 0.5, 0.1]])

        confidence = _sequence_confidence(sequence_ids, sequence_probs)

        # Geometric mean of only the two real tokens.
        assert confidence.item() == pytest.approx((0.9 * 0.81) ** 0.5, abs=1e-5)

    def test_all_padding_does_not_divide_by_zero(self) -> None:
        confidence = _sequence_confidence(torch.tensor([[0, 0]]), torch.tensor([[0.5, 0.5]]))

        assert torch.isfinite(confidence).all()


class TestOcrResult:
    def test_text_joins_regions_in_order(self) -> None:
        result = OcrResult(
            regions=[
                TextRegion("first", 0.9, 0.0, 0.1, 0.1, 0.0, []),
                TextRegion("", 0.9, 0.0, 0.1, 0.1, 0.0, []),
                TextRegion("second", 0.9, 0.0, 0.1, 0.1, 0.0, []),
            ]
        )

        assert result.text == "first\nsecond"


class TestResults:
    def _result(self) -> FusedResult:
        page = PageResult(page_id="p1", height=100, width=80, text="body text")
        page.elements = [
            ElementResult("table", 0.9, (0.1, 0.1, 0.5, 0.5), text="cells", table_markdown="| a |"),
            ElementResult("text", 0.8, (0.1, 0.6, 0.9, 0.7), text="body text"),
        ]
        return FusedResult(
            pages=[page],
            timings=[
                StageTiming("page_elements", 12.0, items=1),
                StageTiming("ocr", 30.0, items=2),
            ],
        )

    def test_metadata_shape_matches_pipeline_columns(self) -> None:
        metadata = self._result().to_metadata()[0]

        assert set(metadata) == {
            "page_id",
            "page_elements_v3",
            "table",
            "chart",
            "infographic",
            "text",
        }
        assert len(metadata["page_elements_v3"]["detections"]) == 2
        assert len(metadata["table"]) == 1
        assert metadata["chart"] == []

    def test_elements_of_filters_by_label(self) -> None:
        page = self._result().pages[0]

        assert len(page.elements_of("table")) == 1
        assert page.elements_of("chart") == []

    def test_timing_totals_and_per_item(self) -> None:
        result = self._result()

        assert result.total_milliseconds() == pytest.approx(42.0)
        assert result.timings[1].per_item() == pytest.approx(15.0)

    def test_timing_table_renders_every_stage(self) -> None:
        rendered = self._result().timing_table()

        assert "page_elements" in rendered
        assert "ocr" in rendered
        assert "total" in rendered

    def test_timing_table_handles_no_timings(self) -> None:
        assert "no timings" in FusedResult(pages=[]).timing_table()


class TestConfig:
    def test_stage_order_matches_the_production_graph(self) -> None:
        config = FusedPipelineConfig()

        assert config.stages_enabled() == (
            "page_elements",
            "table_structure",
            "ocr",
            "embed",
        )

    def test_disabling_tables_removes_the_structure_stage(self) -> None:
        config = FusedPipelineConfig(extract_tables=False)

        assert "table_structure" not in config.stages_enabled()

    def test_enabling_rerank_appends_it_last(self) -> None:
        config = FusedPipelineConfig(enable_rerank=True)

        assert config.stages_enabled()[-1] == "rerank"

    def test_ocr_drops_out_when_nothing_needs_text(self) -> None:
        config = FusedPipelineConfig(
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
            extract_text=False,
        )

        assert "ocr" not in config.stages_enabled()

    def test_defaults_track_the_upstream_model_parameters(self) -> None:
        config = FusedPipelineConfig()

        # Page elements: conf 0.01, iou 0.5, class-agnostic NMS.
        assert config.page_elements.conf_thresh == 0.01
        assert config.page_elements.iou_thresh == 0.5
        assert config.page_elements.class_agnostic is True
        # Table structure: iou 0.25, class-aware NMS.
        assert config.table_structure.iou_thresh == 0.25
        assert config.table_structure.class_agnostic is False
        # Both detectors share the 1024x1024 letterbox.
        assert config.page_elements.input_size == config.table_structure.input_size == (1024, 1024)
        # OCR detector resolution.
        assert config.ocr.infer_length == 1024

    def test_ocr_defaults_to_v2_paragraph_merge(self) -> None:
        ocr = OCRConfig()

        assert ocr.version == "v2"
        assert ocr.merge_level == "paragraph"

    def test_embed_defaults_to_2048_dimensions(self) -> None:
        assert EmbedConfig().output_dimension == 2048


class TestTransferAccounting:
    def test_counts_host_to_device_and_back(self) -> None:
        from nemo_retriever_fused.benchmark import transfer_accounting

        # Without CUDA there is nothing to count, but the wrapper must still
        # install and restore cleanly.
        original_to = torch.Tensor.to
        with transfer_accounting() as account:
            torch.zeros((4, 4)).to(torch.float64)
        assert torch.Tensor.to is original_to
        assert account.host_to_device == 0

    def test_summary_is_human_readable(self) -> None:
        from nemo_retriever_fused.benchmark import TransferAccount

        account = TransferAccount(
            host_to_device=2_000_000,
            device_to_host=500_000,
            host_to_device_calls=3,
            device_to_host_calls=1,
        )

        summary = account.summary()
        assert "H2D" in summary and "D2H" in summary
        assert "2.00 MB" in summary
