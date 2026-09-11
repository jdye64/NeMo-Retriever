# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDFium document load helpers shared by CPU and GPU PDF engines."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from nemo_retriever.common.api.util.pdf.engine import PageInfo, PdfSource

try:
    import pypdfium2 as pdfium
except Exception as exc:  # pragma: no cover
    pdfium = None  # type: ignore[assignment]
    _PDFIUM_IMPORT_ERROR = exc
else:
    _PDFIUM_IMPORT_ERROR = None


def require_pdfium() -> Any:
    if pdfium is None:
        raise RuntimeError("pypdfium2 is required for PDF engines") from _PDFIUM_IMPORT_ERROR
    return pdfium


def open_pdfium_document(source: PdfSource) -> Any:
    """Open a PDFium document from a path or in-memory bytes."""
    lib = require_pdfium()
    if isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
        try:
            return lib.PdfDocument(payload)
        except Exception:
            return lib.PdfDocument(BytesIO(payload))
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"PDF path does not exist: {path}")
    return lib.PdfDocument(str(path))


class PdfiumDocumentMixin:
    """Holds a loaded PDFium document and exposes page-count / page-info."""

    def __init__(self) -> None:
        self._doc: Any = None

    def _require_doc(self) -> Any:
        if self._doc is None:
            raise RuntimeError("No PDF is loaded. Call load() first.")
        return self._doc

    def load_document(self, source: PdfSource) -> None:
        self.close_document()
        self._doc = open_pdfium_document(source)

    def page_count(self) -> int:
        return len(self._require_doc())

    def page_info(self, page_index: Optional[int] = None) -> PageInfo | list[PageInfo]:
        doc = self._require_doc()
        if page_index is not None:
            return self._page_info(int(page_index))
        return [self._page_info(i) for i in range(len(doc))]

    def _page_info(self, page_index: int) -> PageInfo:
        doc = self._require_doc()
        n_pages = len(doc)
        if page_index < 0 or page_index >= n_pages:
            raise IndexError(f"page_index {page_index} is out of range for document with {n_pages} pages")
        page = None
        try:
            page = doc.get_page(page_index)
            rotation = 0
            try:
                rotation = int(page.get_rotation())
            except Exception:
                rotation = 0
            return PageInfo(
                page_index=page_index,
                width_pt=float(page.get_width()),
                height_pt=float(page.get_height()),
                rotation=rotation,
            )
        finally:
            if page is not None and hasattr(page, "close"):
                try:
                    page.close()
                except Exception:
                    pass

    def close_document(self) -> None:
        if self._doc is None:
            return
        try:
            self._doc.close()
        except Exception:
            pass
        self._doc = None
