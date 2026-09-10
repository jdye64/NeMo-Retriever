# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from gpu_pdf_render.pdf import load_pdf
from gpu_pdf_render.pdf.compiler import compile_document
from tests.pdf_fixtures import write_rect_pdf


def test_load_pdf_reads_every_page() -> None:
    data = write_rect_pdf([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    pages = load_pdf(data)
    assert len(pages) == 3
    assert pages[0].width_pt == 72
    assert pages[2].index == 2
    assert any(op == "re" for _, op in pages[0].operations)


def test_compile_emits_rects_for_all_pages() -> None:
    pages = load_pdf(write_rect_pdf([(1, 0, 0), (0, 0, 1)]))
    compiled = compile_document(pages, dpi=144)
    assert compiled.page_count == 2
    assert compiled.rects.shape[0] == 2
    assert set(compiled.rects[:, 0].astype(int).tolist()) == {0, 1}
