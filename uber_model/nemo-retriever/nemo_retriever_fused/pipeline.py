# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fused NeMo Retriever model.

`NemoRetrieverFusedModel` replaces four sequential remote model calls with one
in-process invocation. The production pipeline runs page-elements, table
structure, OCR, and embedding as separate Ray actors that exchange pandas
DataFrames through the object store, so each page image is base64-encoded,
transferred, and decoded once per stage. This model decodes each page once into
device memory and every stage reads the same resident tensors.

Stage order matches the production graph so that results stay comparable::

    page elements -> table structure -> OCR -> embed -> (optional) rerank

What changes is where the data lives between those stages. See `PERFORMANCE.md`
for the measured effect of each item below.

Device residency
    A page is decoded once by `DeviceImageBatch` and the tensor is reused by
    every stage. Region crops are slices of that tensor rather than fresh
    decodes.

Detector overlap
    Page-elements and table-structure are the same YOLOX geometry, so their
    forward passes are issued on two CUDA streams and interleave.

Device-side postprocessing
    Weighted box fusion and the per-class score gate run in torch on device
    instead of numpy on the host, which removes a synchronisation per page.

Single synchronisation
    Only `forward` synchronises, and only once, after every stage has been
    issued. The production path synchronises implicitly at each actor boundary.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator, Sequence

import torch

from nemo_retriever_fused import gpu_ops, loader
from nemo_retriever_fused.config import FusedPipelineConfig
from nemo_retriever_fused.gpu_image import DeviceImageBatch
from nemo_retriever_fused.results import ElementResult, FusedResult, PageResult, StageTiming
from nemo_retriever_fused.stages.detectors import (
    Detections,
    PageElementsStage,
    TableStructureStage,
)
from nemo_retriever_fused.stages.embed import EmbedResult, GpuEmbedStage, GpuRerankStage
from nemo_retriever_fused.stages.ocr import GpuOcrStage, OcrResult
from nemo_retriever_fused.table_assembly import assemble_table_markdown

logger = logging.getLogger(__name__)

# Page element classes OCR runs over, keyed by the extraction flag that enables
# them. Mirrors the production pipeline's stage gating.
_OCR_LABELS_BY_FLAG: dict[str, str] = {
    "extract_tables": "table",
    "extract_charts": "chart",
    "extract_infographics": "infographic",
    "extract_text": "text",
}


class _DeviceTimer:
    """Record per-stage device time with CUDA events.

    Events are cheap to record and do not synchronise, so the timings cost
    nothing until `results()` reads them after the pipeline's single sync.
    """

    def __init__(self, *, enabled: bool, device: torch.device) -> None:
        self._enabled = enabled and device.type == "cuda"
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event, int]] = []

    @contextlib.contextmanager
    def stage(self, name: str, items: int = 0) -> Iterator[None]:
        """Time the enclosed block as stage *name* covering *items* items."""
        if not self._enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((name, start, end, items))

    def results(self) -> list[StageTiming]:
        """Return the recorded timings. Requires a prior device synchronise."""
        return [
            StageTiming(name=name, milliseconds=start.elapsed_time(end), items=items)
            for name, start, end, items in self._pending
        ]


@contextlib.contextmanager
def _nvtx(enabled: bool, label: str) -> Iterator[None]:
    """Push an NVTX range so the fused stages are legible under Nsight."""
    if not enabled:
        yield
        return
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class NemoRetrieverFusedModel:
    """One model, one call, four stages, no host round-trips between them.

    Parameters
    ----------
    config:
        Pipeline configuration. Defaults reproduce the production pipeline's
        stage parameters.

    Examples
    --------
    >>> model = NemoRetrieverFusedModel.from_pretrained()
    >>> result = model(page_images_b64)
    >>> result.pages[0].text
    'Quarterly revenue ...'
    >>> result.page_embeddings.shape
    torch.Size([1, 2048])
    """

    def __init__(self, config: FusedPipelineConfig | None = None) -> None:
        self.config = config or FusedPipelineConfig()
        self.device = torch.device(self.config.execution.device)

        self._page_elements: PageElementsStage | None = None
        self._table_structure: TableStructureStage | None = None
        self._ocr: GpuOcrStage | None = None
        self._embed: GpuEmbedStage | None = None
        self._rerank: GpuRerankStage | None = None

        # A second stream for the table-structure detector. The two YOLOX
        # graphs have identical shapes and no data dependency once the page
        # tensor exists, so they interleave rather than serialise.
        self._side_stream: torch.cuda.Stream | None = None

    @classmethod
    def from_pretrained(cls, config: FusedPipelineConfig | None = None) -> NemoRetrieverFusedModel:
        """Build the model and load every enabled stage's weights.

        Parameters
        ----------
        config:
            Pipeline configuration. Defaults to `FusedPipelineConfig()`.
        """
        model = cls(config)
        model.load()
        return model

    def load(self) -> None:
        """Load weights for every stage the configuration enables.

        Stages that are disabled never allocate device memory, so a text-only
        or detection-only deployment pays for only what it runs.
        """
        execution = self.config.execution

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if execution.overlap_detectors:
                self._side_stream = torch.cuda.Stream(device=self.device)

        self._page_elements = PageElementsStage(
            loader.load_page_elements_detector(self.config, self.device),
            self.config.page_elements,
        )

        if self.config.extract_tables:
            self._table_structure = TableStructureStage(
                loader.load_table_structure_detector(self.config, self.device),
                self.config.table_structure,
            )

        if self._ocr_labels():
            self._ocr = GpuOcrStage(
                loader.load_ocr_pipeline(self.config, self.device),
                self.config.ocr,
                device=self.device,
                dtype=loader.resolve_dtype(execution.ocr_dtype),
            )

        if self.config.enable_embed:
            embed_dtype = loader.resolve_dtype(execution.embed_dtype)
            model, tokenizer = loader.load_vl_model(self.config.embed.repo_id, device=self.device, dtype=embed_dtype)
            self._embed = GpuEmbedStage(model, tokenizer, self.config.embed, device=self.device, dtype=embed_dtype)

        if self.config.enable_rerank:
            rerank_dtype = loader.resolve_dtype(execution.embed_dtype)
            model, tokenizer = loader.load_vl_model(self.config.rerank.repo_id, device=self.device, dtype=rerank_dtype)
            self._rerank = GpuRerankStage(model, tokenizer, self.config.rerank, device=self.device, dtype=rerank_dtype)

        logger.info(
            "fused model ready on %s with stages: %s",
            self.device,
            ", ".join(self.config.stages_enabled()),
        )

    def _ocr_labels(self) -> set[str]:
        """Return the page element labels OCR should run over."""
        return {label for flag, label in _OCR_LABELS_BY_FLAG.items() if getattr(self.config, flag)}

    def __call__(
        self,
        pages: Sequence[str | bytes] | DeviceImageBatch,
        *,
        page_ids: Sequence[str] | None = None,
        embed_elements: bool = False,
        embed_pages: bool = True,
    ) -> FusedResult:
        """Alias for `forward`, so the model is callable like an `nn.Module`."""
        return self.forward(
            pages,
            page_ids=page_ids,
            embed_elements=embed_elements,
            embed_pages=embed_pages,
        )

    def forward(
        self,
        pages: Sequence[str | bytes] | DeviceImageBatch,
        *,
        page_ids: Sequence[str] | None = None,
        embed_elements: bool = False,
        embed_pages: bool = True,
    ) -> FusedResult:
        """Run every enabled stage over *pages* in a single invocation.

        Parameters
        ----------
        pages:
            Base64 page images, raw encoded image bytes, or an already-resident
            `DeviceImageBatch`. Passing a `DeviceImageBatch` skips decoding
            entirely, which is the fast path when the caller rendered the pages
            on the GPU.
        page_ids:
            Optional page identifiers, carried through to `PageResult.page_id`.
        embed_elements:
            Also embed each detected region, not just the whole page. Off by
            default because the production pipeline embeds page-level content.

        Returns
        -------
        FusedResult
            Detections, text, and device-resident embeddings for every page,
            plus per-stage device timings.

        Raises
        ------
        RuntimeError
            If `load` has not run.
        """
        if self._page_elements is None:
            raise RuntimeError("call load() or use from_pretrained() before forward()")

        execution = self.config.execution
        timer = _DeviceTimer(enabled=execution.nvtx, device=self.device)

        # Stage 0: decode. One decode per page for the whole invocation.
        with _nvtx(execution.nvtx, "fused/decode"), timer.stage("decode", len(pages)):
            batch = (
                pages
                if isinstance(pages, DeviceImageBatch)
                else DeviceImageBatch.from_base64(
                    pages,
                    device=self.device,
                    page_ids=page_ids,
                    pinned_staging=execution.pinned_staging,
                    prefer_device_decode=execution.decode_on_device,
                )
            )

        page_floats = batch.as_float()

        # Stage 1: page elements over the resident page tensors.
        with _nvtx(execution.nvtx, "fused/page_elements"), timer.stage("page_elements", len(batch)):
            page_detections = self._page_elements.run(page_floats)

        # Stage 2 and 3: crop once, then run table structure and OCR over the
        # same crops. Cropping from the resident page replaces the production
        # path of base64 decode, PIL crop, numpy convert, and PNG re-encode.
        with _nvtx(execution.nvtx, "fused/crop"), timer.stage("crop"):
            plan = self._build_crop_plan(page_floats, page_detections)

        structure_by_index: dict[int, Detections] = {}
        if self._table_structure is not None and plan.table_crops:
            with _nvtx(execution.nvtx, "fused/table_structure"), timer.stage("table_structure", len(plan.table_crops)):
                structure_by_index = dict(
                    zip(
                        plan.table_crop_indices,
                        self._run_table_structure(plan.table_crops),
                    )
                )

        ocr_by_index: dict[int, OcrResult] = {}
        if self._ocr is not None and plan.ocr_crops:
            with _nvtx(execution.nvtx, "fused/ocr"), timer.stage("ocr", len(plan.ocr_crops)):
                ocr_by_index = dict(zip(plan.ocr_crop_indices, self._ocr.run(plan.ocr_crops)))

        # Assemble page results while the embeddings are still to come, so the
        # host-side text work overlaps the queued device work.
        pages_out = self._assemble_pages(batch, page_detections, plan, structure_by_index, ocr_by_index)

        page_embeddings: torch.Tensor | None = None
        element_embeddings: torch.Tensor | None = None

        # Embedding is the most expensive stage, so skip it when the caller
        # keeps a dedicated embed stage downstream and would discard these.
        if self._embed is not None and (embed_pages or embed_elements):
            with _nvtx(execution.nvtx, "fused/embed"), timer.stage("embed", len(batch)):
                if embed_pages:
                    page_result = self._embed.embed_images(page_floats)
                    page_embeddings = page_result.vectors
                    for index, page in enumerate(pages_out):
                        page.embedding = page_embeddings[index]

                if embed_elements and plan.all_crops:
                    element_result = self._embed.embed_images(plan.all_crops)
                    element_embeddings = element_result.vectors
                    self._attach_element_embeddings(pages_out, plan, element_result)

        # The one synchronisation for the whole invocation. Everything above was
        # issued asynchronously; the production pipeline instead synchronises at
        # each actor boundary because the next stage needs host-side data.
        if execution.single_sync_point and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        return FusedResult(
            pages=pages_out,
            page_embeddings=page_embeddings,
            element_embeddings=element_embeddings,
            timings=timer.results(),
        )

    class _CropPlan:
        """Flattened crops for the whole batch with their provenance.

        Flattening across pages lets table structure and OCR run one batched
        call per invocation instead of one call per page, which is what the
        production pipeline does because its actors process pages row by row.
        """

        __slots__ = (
            "all_crops",
            "owners",
            "table_crops",
            "table_crop_indices",
            "ocr_crops",
            "ocr_crop_indices",
        )

        def __init__(self) -> None:
            self.all_crops: list[torch.Tensor] = []
            # (page index, element index within that page) per flattened crop.
            self.owners: list[tuple[int, int]] = []
            self.table_crops: list[torch.Tensor] = []
            self.table_crop_indices: list[int] = []
            self.ocr_crops: list[torch.Tensor] = []
            self.ocr_crop_indices: list[int] = []

    def _build_crop_plan(self, page_floats: list[torch.Tensor], page_detections: list[Detections]) -> _CropPlan:
        """Crop every region of interest out of the resident page tensors once."""
        labels = self._page_elements.label_names
        ocr_labels = self._ocr_labels()
        plan = self._CropPlan()

        for page_index, (page, detections) in enumerate(zip(page_floats, page_detections)):
            if not len(detections):
                continue

            crops = gpu_ops.crop_regions(page, detections.boxes)
            # `labels` is small and already on device; one transfer per page
            # beats one `.item()` per detection.
            label_ids = detections.labels.tolist()

            for element_index, (crop, label_id) in enumerate(zip(crops, label_ids)):
                flat_index = len(plan.all_crops)
                plan.all_crops.append(crop)
                plan.owners.append((page_index, element_index))

                label_name = labels[int(label_id)]
                if label_name == "table" and self._table_structure is not None:
                    plan.table_crops.append(crop)
                    plan.table_crop_indices.append(flat_index)
                if label_name in ocr_labels:
                    plan.ocr_crops.append(crop)
                    plan.ocr_crop_indices.append(flat_index)

        return plan

    def _run_table_structure(self, crops: list[torch.Tensor]) -> list[Detections]:
        """Run the table-structure detector, overlapping it when configured."""
        if self._table_structure is None:
            return []
        if self._side_stream is None:
            return self._table_structure.run(crops)

        # The crops were produced on the default stream, so the side stream must
        # wait for them before reading, and the default stream must wait for the
        # side stream's results afterwards.
        self._side_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._side_stream):
            detections = self._table_structure.run(crops)
        torch.cuda.current_stream(self.device).wait_stream(self._side_stream)
        return detections

    def _assemble_pages(
        self,
        batch: DeviceImageBatch,
        page_detections: list[Detections],
        plan: _CropPlan,
        structure_by_index: dict[int, Detections],
        ocr_by_index: dict[int, OcrResult],
    ) -> list[PageResult]:
        """Build the host-side `PageResult` list from the device outputs."""
        labels = self._page_elements.label_names
        table_labels = self._table_structure.label_names if self._table_structure is not None else ()

        # Map flattened crop index back to its owning page and element.
        by_owner = {owner: flat for flat, owner in enumerate(plan.owners)}

        pages_out: list[PageResult] = []
        for page_index, (page_id, (height, width)) in enumerate(zip(batch.page_ids, batch.shapes)):
            detections = page_detections[page_index]
            page = PageResult(page_id=page_id, height=height, width=width)

            if len(detections):
                # One transfer per page for boxes, scores, and labels together,
                # rather than the per-detection `.item()` calls the production
                # postprocessor makes.
                boxes = detections.boxes.tolist()
                scores = detections.scores.tolist()
                label_ids = detections.labels.tolist()

                for element_index, (box, score, label_id) in enumerate(zip(boxes, scores, label_ids)):
                    label_name = labels[int(label_id)]
                    flat_index = by_owner.get((page_index, element_index))
                    ocr_result = ocr_by_index.get(flat_index) if flat_index is not None else None
                    structure = structure_by_index.get(flat_index) if flat_index is not None else None

                    markdown = None
                    if structure is not None and ocr_result is not None and len(structure):
                        markdown = assemble_table_markdown(structure, ocr_result, label_names=table_labels)

                    page.elements.append(
                        ElementResult(
                            label=label_name,
                            score=float(score),
                            bbox_xyxy_norm=(
                                float(box[0]),
                                float(box[1]),
                                float(box[2]),
                                float(box[3]),
                            ),
                            text=ocr_result.text if ocr_result is not None else None,
                            table_markdown=markdown,
                        )
                    )

            page.text = "\n".join(element.text for element in page.elements if element.label == "text" and element.text)
            pages_out.append(page)

        return pages_out

    @staticmethod
    def _attach_element_embeddings(pages_out: list[PageResult], plan: _CropPlan, element_result: EmbedResult) -> None:
        """Attach per-element embeddings to their owning `ElementResult`."""
        for flat_index, (page_index, element_index) in enumerate(plan.owners):
            elements = pages_out[page_index].elements
            if element_index < len(elements):
                elements[element_index].embedding = element_result.vectors[flat_index]

    def rerank(
        self,
        query: str,
        *,
        texts: Sequence[str] | None = None,
        images: Sequence[torch.Tensor] | None = None,
        top_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score and order candidates for *query*.

        Parameters
        ----------
        query:
            The search query.
        texts:
            Candidate passage text.
        images:
            Candidate page images as resident device tensors.
        top_k:
            Truncate to the highest-scoring *top_k* candidates. Selecting on
            device before any host transfer means only the survivors are copied.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Device-resident scores and the indices that sort them descending.

        Raises
        ------
        RuntimeError
            If reranking is not enabled in the configuration.
        """
        if self._rerank is None:
            raise RuntimeError("reranking requires enable_rerank=True in the configuration")

        scores = self._rerank.rank(query, texts=texts, images=images)
        order = torch.argsort(scores, descending=True)
        if top_k is not None:
            order = order[:top_k]
        return scores[order], order
