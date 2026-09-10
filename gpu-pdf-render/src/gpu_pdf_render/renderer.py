# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np

from gpu_pdf_render.encode.jpeg import encode_jpeg
from gpu_pdf_render.encode.png import encode_png
from gpu_pdf_render.gpu.device import resolve_backend
from gpu_pdf_render.gpu.raster import rasterize
from gpu_pdf_render.pdf import load_pdf
from gpu_pdf_render.pdf.compiler import compile_document
from gpu_pdf_render.pdf.ir import CompiledDocument
from gpu_pdf_render.types import RenderBackend

PdfSource = Union[str, Path, bytes]


def _as_numpy(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)


@dataclass
class RenderedDocument:
    """Full-document raster kept as a batched page tensor.

    When ``backend='cuda'``, ``device_pixels`` is a CuPy ndarray in GPU memory.
    """

    device_pixels: Any
    page_widths: np.ndarray
    page_heights: np.ndarray
    dpi: float
    backend: RenderBackend
    compiled: CompiledDocument

    @property
    def page_count(self) -> int:
        return int(self.device_pixels.shape[0])

    def page_array(self, page_index: int) -> Any:
        self._check_page(page_index)
        h = int(self.page_heights[page_index])
        w = int(self.page_widths[page_index])
        return self.device_pixels[page_index, :h, :w, :]

    def to_numpy(self, page_index: int) -> np.ndarray:
        return _as_numpy(self.page_array(page_index))

    def to_png(self, page_index: int = 0) -> bytes:
        return encode_png(self.to_numpy(page_index))

    def to_jpeg(self, page_index: int = 0, quality: int = 90) -> bytes:
        pixels = self.page_array(page_index)
        return encode_jpeg(pixels, quality=quality)

    def save(self, output_dir: str | Path, image_format: str = "png", quality: int = 90) -> list[Path]:
        fmt = image_format.lower().lstrip(".")
        if fmt in {"jpg", "jpeg"}:
            fmt = "jpeg"
            suffix = ".jpg"
        elif fmt == "png":
            suffix = ".png"
        else:
            raise ValueError("image_format must be 'png' or 'jpeg'")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        width = max(1, int(np.ceil(np.log10(self.page_count + 1))))
        for index in range(self.page_count):
            path = out / f"page-{index + 1:0{width}d}{suffix}"
            data = self.to_jpeg(index, quality=quality) if fmt == "jpeg" else self.to_png(index)
            path.write_bytes(data)
            written.append(path)
        return written

    def _check_page(self, page_index: int) -> None:
        if page_index < 0 or page_index >= self.page_count:
            raise IndexError(f"page_index {page_index} out of range for {self.page_count} pages")


class GpuPdfRenderer:
    """Rasterize every page of a PDF into a GPU-resident image batch."""

    def __init__(self, dpi: float = 144.0, backend: RenderBackend | str = "cuda") -> None:
        self.dpi = float(dpi)
        self.backend = resolve_backend(backend)

    def render_document(self, source: PdfSource) -> RenderedDocument:
        pages = load_pdf(source)
        compiled = compile_document(pages, dpi=self.dpi)
        pixels = rasterize(compiled, backend=self.backend)
        return RenderedDocument(
            device_pixels=pixels,
            page_widths=compiled.page_widths,
            page_heights=compiled.page_heights,
            dpi=compiled.dpi,
            backend=self.backend,
            compiled=compiled,
        )
