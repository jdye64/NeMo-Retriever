# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpu_pdf_render import GpuPdfRenderer, GpuUnavailableError
from gpu_pdf_render.gpu.device import cuda_available
from tests.pdf_fixtures import write_rect_pdf, write_text_pdf


def test_full_document_png_and_jpeg(tmp_path: Path) -> None:
    pdf = write_rect_pdf([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
    renderer = GpuPdfRenderer(dpi=144, backend="reference")
    document = renderer.render_document(pdf)
    assert document.page_count == 2
    assert document.backend == "reference"

    red = document.to_numpy(0)
    blue = document.to_numpy(1)
    assert red.shape[0] == 144
    assert red.shape[1] == 144
    # Rectangle is 10,10,20,20 pt in a 72pt page at 144 dpi (2 px/pt), Y flipped.
    sample_y, sample_x = 100, 40
    np.testing.assert_allclose(red[sample_y, sample_x, :3], (255, 0, 0), atol=1)
    np.testing.assert_allclose(blue[sample_y, sample_x, :3], (0, 0, 255), atol=1)
    np.testing.assert_allclose(red[4, 4, :3], (255, 255, 255), atol=1)

    png = document.to_png(0)
    jpeg = document.to_jpeg(1, quality=90)
    assert png.startswith(b"\x89PNG")
    assert jpeg[:2] == b"\xff\xd8"

    paths = document.save(tmp_path, image_format="png")
    assert [path.name for path in paths] == ["page-1.png", "page-2.png"]
    jpeg_paths = document.save(tmp_path / "jpg", image_format="jpeg")
    assert jpeg_paths[0].suffix == ".jpg"


def test_text_and_rect_document_does_not_crash() -> None:
    renderer = GpuPdfRenderer(dpi=72, backend="reference")
    document = renderer.render_document(write_text_pdf())
    assert document.page_count == 1
    pixels = document.to_numpy(0)
    assert pixels[..., :3].min() < 255


def test_cuda_backend_requires_gpu() -> None:
    if cuda_available():
        pytest.skip("CUDA is available in this environment")
    with pytest.raises(GpuUnavailableError):
        GpuPdfRenderer(backend="cuda")


@pytest.mark.cuda
def test_cuda_backend_keeps_device_array() -> None:
    if not cuda_available():
        pytest.skip("CUDA is required")
    import cupy as cp

    renderer = GpuPdfRenderer(dpi=72, backend="cuda")
    document = renderer.render_document(write_rect_pdf([(0.0, 1.0, 0.0)]))
    assert isinstance(document.device_pixels, cp.ndarray)
    assert document.to_png(0).startswith(b"\x89PNG")
