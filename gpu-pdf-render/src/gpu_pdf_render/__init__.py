# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Research GPU PDF rasterizer: full-document pages to GPU-resident PNG/JPEG."""

from gpu_pdf_render.renderer import GpuPdfRenderer, RenderedDocument
from gpu_pdf_render.types import GpuUnavailableError, RenderBackend

__version__ = "0.1.0"

__all__ = [
    "GpuPdfRenderer",
    "GpuUnavailableError",
    "RenderBackend",
    "RenderedDocument",
    "__version__",
]
