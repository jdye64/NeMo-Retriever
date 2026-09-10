<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# gpu-pdf-render

Research-and-development library for rendering a **full PDF document** (every page, not a single-page crop) into GPU-resident page images, then encoding those images as **PNG** or **JPEG**.

This package is intentionally standalone. NeMo Retriever does not import it yet.

## Why this is hard

A production PDF engine is a sequential interpreter: content streams, resource dictionaries, Form XObjects, font programs, blend modes, and soft masks. Almost none of that state machine maps cleanly onto data-parallel GPU work.

What *does* map onto the GPU — and what this library keeps on device — is the expensive pixel work after a page has been compiled into a display list:

1. **Parse and compile on the host.** `pypdf` walks the document once, including every page, and lowers operators into GPU-friendly arrays (rects, path edges, image blits, glyph quads).
2. **Rasterize on the device.** A batched RGBA framebuffer `(pages, height, width, 4)` lives in GPU memory. Fill, winding-number path coverage, and image/glyph blits run as CUDA kernels (CuPy `RawKernel`).
3. **Color convert and transform on the device.** JPEG YCbCr, 8×8 DCT, and quantization run as array kernels on the same buffer. PNG filter-0 scanlines are packed on device.
4. **Leave pixels on the GPU until encode.** `RenderedDocument.device_pixels` is a device array. Host copies happen only when you ask for bytes or files.

JPEG Huffman coding and PNG DEFLATE are still partly sequential; the CUDA path runs the dense linear algebra on GPU and finishes the bitstream on the host. The `reference` backend implements the same algorithms in NumPy so the compiler and codecs can be tested without a GPU.

## Install

```bash
pip install -e .
# CUDA raster path
pip install -e ".[cuda]"
```

Requires an NVIDIA GPU and a working CuPy build for `backend="cuda"`.

## Python API

```python
from gpu_pdf_render import GpuPdfRenderer

renderer = GpuPdfRenderer(dpi=144, backend="cuda")
doc = renderer.render_document("paper.pdf")  # all pages, GPU-resident

png = doc.to_png(page_index=0)
jpeg = doc.to_jpeg(page_index=0, quality=90)
doc.save("out_pages", image_format="png")
```

`device_pixels` stays on the GPU when CuPy is active. Index it like a batch of page tensors.

## CLI

```bash
gpu-pdf-render render paper.pdf --out ./out --format png --dpi 144 --backend cuda
gpu-pdf-render info paper.pdf
```

## Backends

| Backend | Role |
|---------|------|
| `cuda` | Default. Framebuffer, fills, blits, YCbCr/DCT live in GPU memory. |
| `reference` | Host implementation of the same kernels for tests and algorithm comparison. |

## Scope (research subset)

The interpreter covers a practical subset of PDF 1.x drawing: graphics state, CTM, colors (`rg`/`g`/`k`), rectangles, paths (including cubics), fills, strokes, `Do` on Image and Form XObjects, and `Tj`/`TJ` text via an 8×8 atlas. Transparency groups, ICC, Type 3 fonts, JBIG2, and full clipping paths are out of scope for this R&D slice.
