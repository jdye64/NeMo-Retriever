# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU PDF engine: PDFium parse, raster, and JPEG or PNG encode on the host."""

from __future__ import annotations

from typing import Optional

from nemo_retriever.common.api.util.pdf.engine import (
    ImageFormat,
    PDFEngine,
    PageImage,
    PdfEngineBackend,
    PdfSource,
    RenderMode,
)
from nemo_retriever.common.api.util.pdf.pdfium_document import PdfiumDocumentMixin
from nemo_retriever.common.api.util.pdf.render import render_pdfium_page_cpu


class CpuPDFEngine(PdfiumDocumentMixin, PDFEngine):
    """PDFium-only engine matching the current ``pdf_extraction`` raster path."""

    @property
    def backend(self) -> PdfEngineBackend:
        return "cpu"

    @property
    def device(self) -> str:
        return "cpu"

    def load(self, source: PdfSource) -> "CpuPDFEngine":
        self.load_document(source)
        return self

    def rasterize_page(
        self,
        page_index: int,
        *,
        dpi: int = 200,
        render_mode: RenderMode = "fit_to_model",
        image_format: ImageFormat = "jpeg",
        jpeg_quality: int = 100,
        keep_on_device: Optional[bool] = None,
    ) -> PageImage:
        del keep_on_device  # CPU engine always returns host bytes.
        doc = self._require_doc()
        n_pages = len(doc)
        if page_index < 0 or page_index >= n_pages:
            raise IndexError(f"page_index {page_index} is out of range for document with {n_pages} pages")
        page = None
        try:
            page = doc.get_page(page_index)
            _arr, encoded, fmt, (height, width) = render_pdfium_page_cpu(
                page,
                dpi=dpi,
                render_mode=render_mode,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
                encode=True,
            )
        finally:
            if page is not None and hasattr(page, "close"):
                try:
                    page.close()
                except Exception:
                    pass
        assert encoded is not None
        return PageImage(
            page_index=page_index,
            width=width,
            height=height,
            nbytes=len(encoded),
            encoding=fmt,
            host_bytes=encoded,
            device_tensor=None,
        )

    def close(self) -> None:
        self.close_document()
