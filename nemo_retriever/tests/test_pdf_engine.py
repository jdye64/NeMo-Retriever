# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PDFEngine CPU/GPU backends and the pdf-engine benchmark helper."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

pdfium = pytest.importorskip("pypdfium2")

from nemo_retriever.common.api.util.pdf.engine import (  # noqa: E402
    PDFEngine,
    PageImage,
    PageInfo,
    create_pdf_engine,
)
from nemo_retriever.tools.benchmark.pdf_engine import (  # noqa: E402
    benchmark_pdf_engine,
    collect_pdf_paths,
)


def _make_blank_pdf_bytes(width: float = 612, height: float = 792, pages: int = 1) -> bytes:
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(width, height)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _write_pdf(tmp_path: Path, name: str = "sample.pdf", pages: int = 1) -> Path:
    path = tmp_path / name
    path.write_bytes(_make_blank_pdf_bytes(pages=pages))
    return path


def test_cpu_engine_load_info_and_jpeg_bytes(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path, pages=2)
    engine = create_pdf_engine("cpu")
    try:
        engine.load(pdf_path)
        assert engine.backend == "cpu"
        assert engine.device == "cpu"
        assert engine.page_count() == 2
        infos = engine.page_info()
        assert isinstance(infos, list) and len(infos) == 2
        assert isinstance(infos[0], PageInfo)
        assert infos[0].width_pt == pytest.approx(612)
        assert infos[0].height_pt == pytest.approx(792)

        images = engine.rasterize_pages(dpi=72, render_mode="full_dpi", image_format="jpeg")
        assert len(images) == 2
        for image in images:
            assert isinstance(image, PageImage)
            assert image.encoding == "jpeg"
            assert image.host_bytes is not None and image.host_bytes[:2] == b"\xff\xd8"
            assert image.nbytes == len(image.host_bytes)
            assert image.width > 0 and image.height > 0
            assert image.device_tensor is None
    finally:
        engine.close()


def test_cpu_engine_matches_extraction_helper_dimensions() -> None:
    from nemo_retriever.operators.extract.pdf.extract import _render_page_to_base64

    payload = _make_blank_pdf_bytes()
    doc = pdfium.PdfDocument(payload)
    page = doc.get_page(0)
    try:
        render_info = _render_page_to_base64(page, dpi=200, image_format="jpeg", render_mode="fit_to_model")
    finally:
        page.close()
        doc.close()

    engine = create_pdf_engine("cpu")
    try:
        engine.load(payload)
        image = engine.rasterize_page(0, dpi=200, render_mode="fit_to_model", image_format="jpeg")
    finally:
        engine.close()

    orig_h, orig_w = render_info["orig_shape_hw"]
    assert image.height == orig_h
    assert image.width == orig_w
    assert image.nbytes > 0


def test_gpu_engine_keeps_device_tensor_for_page_elements() -> None:
    torch = pytest.importorskip("torch")
    payload = _make_blank_pdf_bytes()
    engine = create_pdf_engine("gpu", device="cpu", keep_on_device=True)
    try:
        assert isinstance(engine, PDFEngine)
        engine.load(payload)
        image = engine.rasterize_page(0, dpi=72, render_mode="full_dpi", image_format="jpeg")
        assert image.host_bytes is not None and image.host_bytes[:2] == b"\xff\xd8"
        assert image.device_tensor is not None
        assert tuple(image.device_tensor.shape) == (3, image.height, image.width)
        assert image.device_tensor.dtype == torch.uint8
        batch = engine.page_elements_batch()
        assert batch.ndim == 4
        assert batch.shape[0] == 1
        assert batch.shape[1] == 3
    finally:
        engine.close()


def test_create_pdf_engine_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported PDF engine backend"):
        create_pdf_engine("tpu")  # type: ignore[arg-type]


def test_benchmark_cpu_engine_on_directory(tmp_path: Path) -> None:
    _write_pdf(tmp_path, "a.pdf", pages=1)
    _write_pdf(tmp_path, "b.pdf", pages=2)
    paths = collect_pdf_paths(tmp_path, recursive=False)
    assert len(paths) == 2
    summary, files = benchmark_pdf_engine(
        paths,
        backend="cpu",
        dpi=72,
        render_mode="full_dpi",
        image_format="jpeg",
        jpeg_quality=80,
        warmup=False,
        keep_on_device=False,
    )
    assert summary.backend == "cpu"
    assert summary.files == 2
    assert summary.pages == 3
    assert summary.failed_files == 0
    assert summary.encoded_bytes > 0
    assert len(files) == 2
    assert all(row.error is None for row in files)
