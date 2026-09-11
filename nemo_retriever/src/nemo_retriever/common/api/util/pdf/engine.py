# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDF engine interface for load, metadata, and page rasterization.

``PDFEngine`` is the shared contract for PDF operations used by extraction and
by the CPU versus GPU raster benchmark. Concrete engines load documents with
PDFium. The CPU engine rasterizes and encodes entirely on the host, matching
the current ``pdf_extraction`` path. The GPU engine still uses PDFium for
document load, page metadata, and the CPU bitmap render, then completes color
conversion and JPEG encoding on the GPU and retains tensors for page-elements
inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Sequence, Union

PdfSource = Union[str, Path, bytes, bytearray, memoryview]
RenderMode = Literal["full_dpi", "fit_to_model"]
ImageFormat = Literal["jpeg", "png"]
PdfEngineBackend = Literal["cpu", "gpu"]


@dataclass(frozen=True)
class PageInfo:
    """Geometry for one PDF page in PDF points (1/72 inch)."""

    page_index: int
    width_pt: float
    height_pt: float
    rotation: int


@dataclass
class PageImage:
    """Raster of one PDF page plus byte and dimension metadata.

    ``host_bytes`` holds the encoded JPEG or PNG payload. ``device_tensor`` is
    a CHW uint8 RGB tensor when the engine keeps the raster on device for
    later page-elements invocation. ``nbytes`` is the encoded payload size
    when ``host_bytes`` is present, otherwise the device tensor size.
    """

    page_index: int
    width: int
    height: int
    nbytes: int
    encoding: str
    host_bytes: Optional[bytes] = None
    device_tensor: Any = None


class PDFEngine(ABC):
    """Interface for loading PDFs, reading page info, and rasterizing pages."""

    def __enter__(self) -> "PDFEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    @abstractmethod
    def backend(self) -> PdfEngineBackend:
        """Engine backend name: ``cpu`` or ``gpu``."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Device string for rasters, for example ``cpu`` or ``cuda:0``."""

    @abstractmethod
    def load(self, source: PdfSource) -> "PDFEngine":
        """Open a PDF from a filesystem path or in-memory bytes."""

    @abstractmethod
    def page_count(self) -> int:
        """Return the number of pages in the loaded document."""

    @abstractmethod
    def page_info(self, page_index: Optional[int] = None) -> Union[PageInfo, list[PageInfo]]:
        """Return geometry for one page, or for every page when ``page_index`` is omitted."""

    @abstractmethod
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
        """Rasterize one page and return encoded bytes plus pixel dimensions."""

    def rasterize_pages(
        self,
        page_indices: Optional[Sequence[int]] = None,
        *,
        dpi: int = 200,
        render_mode: RenderMode = "fit_to_model",
        image_format: ImageFormat = "jpeg",
        jpeg_quality: int = 100,
        keep_on_device: Optional[bool] = None,
    ) -> list[PageImage]:
        """Rasterize the selected pages, or every page when ``page_indices`` is omitted."""
        indices: Sequence[int]
        if page_indices is None:
            indices = range(self.page_count())
        else:
            indices = page_indices
        return [
            self.rasterize_page(
                int(idx),
                dpi=dpi,
                render_mode=render_mode,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
                keep_on_device=keep_on_device,
            )
            for idx in indices
        ]

    @abstractmethod
    def close(self) -> None:
        """Release the loaded document and any device rasters."""

    def release_device_images(self) -> None:
        """Drop retained device tensors. CPU engines no-op."""
        return None

    def iter_page_info(self) -> Iterator[PageInfo]:
        """Yield geometry for every page."""
        infos = self.page_info()
        if isinstance(infos, PageInfo):
            yield infos
            return
        yield from infos


def create_pdf_engine(
    backend: PdfEngineBackend = "cpu",
    *,
    device: Optional[str] = None,
    keep_on_device: bool = True,
) -> PDFEngine:
    """Construct a CPU or GPU PDF engine.

    Parameters
    ----------
    backend:
        ``cpu`` uses PDFium for parse, render, and JPEG or PNG encode.
        ``gpu`` uses PDFium for parse, page info, and the CPU bitmap, then
        rasterizes JPEG on the GPU and keeps tensors on device.
    device:
        Torch device for the GPU engine. Ignored for the CPU engine.
        Defaults to ``cuda`` when CUDA is available, otherwise ``cpu``.
    keep_on_device:
        When True, the GPU engine retains CHW uint8 RGB tensors after
        rasterize so page-elements inference can consume them without a host
        round-trip.
    """
    if backend == "cpu":
        from nemo_retriever.common.api.util.pdf.cpu_engine import CpuPDFEngine

        return CpuPDFEngine()
    if backend == "gpu":
        from nemo_retriever.common.api.util.pdf.gpu_engine import GpuPDFEngine

        return GpuPDFEngine(device=device, keep_on_device=keep_on_device)
    raise ValueError(f"Unsupported PDF engine backend: {backend!r}")
