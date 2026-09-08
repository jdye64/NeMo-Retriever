# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused extraction: graph shape, embed absorption, tuning, and warmup."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from nemo_retriever.common.params import EmbedParams, ExtractParams
from nemo_retriever.graph.ingestor_runtime import (
    build_graph,
    default_concurrency_node_names,
    fused_absorbs_embed,
    is_fused_extraction,
)
from nemo_retriever.models.warmup_registry import build_warmup_spec

STAGED_NODES = {"PageElementDetectionActor", "TableStructureActor", "OCRActor"}


def _graph_node_names(graph: Any) -> list[str]:
    names: list[str] = []

    def visit(node: Any) -> None:
        names.append(getattr(node.operator, "name", node.name))
        for child in node.children:
            visit(child)

    for root in graph.roots:
        visit(root)
    return names


def _pdf_graph_names(extract_params: ExtractParams, embed_params: EmbedParams | None) -> list[str]:
    return _graph_node_names(
        build_graph(
            extraction_mode="pdf",
            extract_params=extract_params,
            embed_params=embed_params,
            stage_order=("extract", "embed") if embed_params is not None else ("extract",),
        )
    )


class TestGraphShape:
    def test_fused_replaces_the_three_staged_extraction_actors(self) -> None:
        names = _pdf_graph_names(ExtractParams(method="fused", use_table_structure=True), None)

        assert "FusedExtractionActor" in names
        assert STAGED_NODES.isdisjoint(names)

    def test_staged_method_still_builds_the_staged_actors(self) -> None:
        names = _pdf_graph_names(ExtractParams(method="pdfium_hybrid", use_table_structure=True), None)

        assert "FusedExtractionActor" not in names
        assert STAGED_NODES.issubset(names)

    def test_fused_still_renders_pages_upstream(self) -> None:
        names = _pdf_graph_names(ExtractParams(method="fused"), None)

        assert names.index("PDFExtractionActor") < names.index("FusedExtractionActor")


class TestEmbedAbsorption:
    def test_page_granularity_embed_is_absorbed(self) -> None:
        names = _pdf_graph_names(
            ExtractParams(method="fused", use_table_structure=True),
            EmbedParams(embed_granularity="page"),
        )

        assert "FusedExtractionActor" in names
        assert "_BatchEmbedActor" not in names
        # The reshape only exists to feed the embed stage.
        assert "CollapseContentToPageRows" not in names

    def test_element_granularity_embed_keeps_the_embed_stage(self) -> None:
        """Element rows do not exist yet when the fused stage runs."""
        names = _pdf_graph_names(
            ExtractParams(method="fused", use_table_structure=True),
            EmbedParams(embed_granularity="element"),
        )

        assert "FusedExtractionActor" in names
        assert "_BatchEmbedActor" in names
        assert "ExplodeContentToRows" in names

    def test_staged_method_never_absorbs_embed(self) -> None:
        names = _pdf_graph_names(
            ExtractParams(method="pdfium_hybrid"),
            EmbedParams(embed_granularity="page"),
        )

        assert "_BatchEmbedActor" in names

    @pytest.mark.parametrize("granularity,expected", [("page", True), ("element", False)])
    def test_absorption_predicate(self, granularity: str, expected: bool) -> None:
        assert (
            fused_absorbs_embed(
                ExtractParams(method="fused"),
                EmbedParams(embed_granularity=granularity),
            )
            is expected
        )

    def test_absorption_requires_fused_and_embed(self) -> None:
        assert fused_absorbs_embed(ExtractParams(method="fused"), None) is False
        assert fused_absorbs_embed(ExtractParams(method="pdfium"), EmbedParams(embed_granularity="page")) is False
        assert is_fused_extraction(None) is False


class TestParamsValidation:
    @pytest.mark.parametrize(
        "endpoint",
        ["page_elements_invoke_url", "ocr_invoke_url", "table_structure_invoke_url"],
    )
    def test_fused_rejects_per_stage_nim_endpoints(self, endpoint: str) -> None:
        with pytest.raises(ValueError, match="cannot delegate a stage to a NIM"):
            ExtractParams(method="fused", **{endpoint: "http://nim.invalid/v1"})

    def test_fused_rejects_disabled_page_elements(self) -> None:
        with pytest.raises(ValueError, match="detects page elements inside the fused model"):
            ExtractParams(method="fused", use_page_elements=False)

    def test_fused_forces_page_raster(self) -> None:
        assert ExtractParams(method="fused", extract_page_as_image=False).extract_page_as_image is True

    def test_staged_methods_are_unaffected(self) -> None:
        params = ExtractParams(method="pdfium", ocr_invoke_url="http://nim.invalid/v1")

        assert params.method == "pdfium"
        assert params.ocr_invoke_url == "http://nim.invalid/v1"


class TestConcurrencyDefaults:
    def test_fused_does_not_claim_staged_worker_pools(self) -> None:
        """Fused workers come from FusedTuningParams, which always has a value."""
        names = default_concurrency_node_names(ExtractParams(method="fused"), None, None, None)

        assert names == set()

    def test_staged_pools_still_report_defaults(self) -> None:
        names = default_concurrency_node_names(ExtractParams(method="pdfium_hybrid"), None, None, None)

        assert "PageElementDetectionActor" in names


class TestWarmupSpec:
    def test_fused_warms_one_model_instead_of_three(self) -> None:
        spec = build_warmup_spec({"method": "fused", "use_table_structure": True}, None, None)

        assert spec is not None
        assert spec["stages"] == ["fused"]

    def test_staged_warms_the_individual_stages(self) -> None:
        spec = build_warmup_spec({"method": "pdfium_hybrid", "use_table_structure": True}, None, None)

        assert spec is not None
        assert set(spec["stages"]) == {"page_elements", "ocr", "table_structure"}


class TestServiceConfigPlumbing:
    """``local_models.extract.method`` must reach ``ExtractParams``."""

    @staticmethod
    def _extract_params(method: str, **nim_overrides: Any) -> Any:
        from nemo_retriever.service.config import LocalModelsConfig, NimEndpointsConfig
        from nemo_retriever.service.services.pipeline_executor import build_extract_params

        local = LocalModelsConfig(enabled=True, extract={"enabled": True, "method": method})
        return build_extract_params(NimEndpointsConfig(**nim_overrides), local)

    def test_fused_method_reaches_extract_params(self) -> None:
        assert self._extract_params("fused").method == "fused"

    def test_default_method_is_unchanged(self) -> None:
        assert self._extract_params("pdfium").method == "pdfium"

    def test_configured_nim_endpoint_wins_over_fused(self) -> None:
        """NIM endpoints take precedence, so fused must yield rather than raise."""
        params = self._extract_params("fused", ocr_invoke_url="http://nim.invalid/v1")

        assert params.method != "fused"
        assert params.ocr_invoke_url == "http://nim.invalid/v1"


class TestFusedCompute:
    """The column contract the fused stage must reproduce."""

    @staticmethod
    def _pages_frame() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "path": "a.pdf",
                    "page_number": 1,
                    "source_id": "a",
                    "text": "",
                    "page_image": {"image_b64": "Zm9v", "encoding": "jpeg", "orig_shape_hw": (100, 80)},
                    "metadata": {"error": None},
                }
            ]
        )

    def test_missing_page_raster_yields_empty_payloads(self) -> None:
        from nemo_retriever.common.modality.fused.shared import fused_extract_pages

        frame = self._pages_frame()
        frame.at[0, "page_image"] = None

        out = fused_extract_pages(frame, model=object())

        assert out.loc[0, "page_elements_v3"]["detections"] == []
        assert out.loc[0, "page_elements_v3_num_detections"] == 0

    def test_model_failure_is_recorded_not_raised(self) -> None:
        from nemo_retriever.common.modality.fused.shared import fused_extract_pages

        def explode(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("cuda oom")

        out = fused_extract_pages(self._pages_frame(), model=explode)

        error = out.loc[0, "page_elements_v3"]["error"]
        assert error["type"] == "RuntimeError"
        assert error["stage"] == "fused_invoke"

    def test_results_are_written_to_the_staged_columns(self) -> None:
        from nemo_retriever.common.modality.fused.shared import fused_extract_pages

        class _Element:
            def __init__(self, label: str, text: str) -> None:
                self.label = label
                self.score = 0.9
                self.bbox_xyxy_norm = (0.1, 0.2, 0.3, 0.4)
                self.text = text
                self.table_markdown = "| a |" if label == "table" else None

        class _Page:
            page_id = "a:1"
            height = 100
            width = 80
            text = "page text"
            elements = [_Element("table", "t"), _Element("chart", "c")]

        class _Result:
            pages = [_Page()]
            page_embeddings = None

        out = fused_extract_pages(self._pages_frame(), model=lambda *_a, **_k: _Result())

        assert out.loc[0, "text"] == "page text"
        assert out.loc[0, "page_elements_v3_num_detections"] == 2
        assert out.loc[0, "page_elements_v3_counts_by_label"] == {"table": 1, "chart": 1}
        assert out.loc[0, "table"] == [{"bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4], "text": "| a |"}]
        assert out.loc[0, "chart"] == [{"bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4], "text": "c"}]
        assert out.loc[0, "ocr_v1_num_detections"] == 2
