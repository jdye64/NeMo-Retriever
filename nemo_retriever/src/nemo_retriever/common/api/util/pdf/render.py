# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared PDFium page-render helpers used by extraction and PDF engines."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Literal, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from nemo_retriever.common.api.util.pdf.engine import RenderMode
from nemo_retriever.common.api.util.pdf.pdfium import convert_bitmap_to_corrected_numpy

# Default model input size used by page-element detection.
MODEL_INPUT_SIZE: Tuple[int, int] = (1024, 1024)


def compute_page_render_scale(
    page: Any,
    *,
    dpi: int = 200,
    render_mode: RenderMode = "fit_to_model",
    target_wh: Tuple[int, int] = MODEL_INPUT_SIZE,
) -> float:
    """Compute the PDFium render scale for a page.

    ``full_dpi`` uses ``dpi / 72``. ``fit_to_model`` caps that scale so the
    raster fits inside ``target_wh``. The formula matches
    ``PDFExtractionActor`` / ``_compute_fit_to_model_scale``.
    """
    base_scale = max(float(dpi) / 72.0, 0.01)
    if render_mode != "fit_to_model":
        return base_scale
    target_w, target_h = target_wh
    page_w = float(page.get_width())
    page_h = float(page.get_height())
    if page_w <= 0 or page_h <= 0 or target_w <= 0 or target_h <= 0:
        return base_scale
    fit_scale = max(min(target_w / page_w, target_h / page_h), 1e-3)
    return min(base_scale, fit_scale)


def encode_rgb_image(
    arr: np.ndarray,
    *,
    image_format: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 100,
) -> bytes:
    """Encode an RGB (or RGBA) uint8 HWC array as JPEG or PNG bytes."""
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    fmt = image_format.lower()
    if cv2 is not None:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if fmt == "jpeg":
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            if not ok:
                raise RuntimeError("cv2.imencode failed for JPEG")
            return buf.tobytes()
        ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise RuntimeError("cv2.imencode failed for PNG")
        return buf.tobytes()

    from PIL import Image as _PILImage

    pil_img = _PILImage.fromarray(arr)
    buf = BytesIO()
    if fmt == "jpeg":
        pil_img = pil_img.convert("RGB")
        pil_img.save(buf, format="JPEG", quality=int(jpeg_quality))
    else:
        pil_img.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def render_pdfium_page_cpu(
    page: Any,
    *,
    dpi: int = 200,
    render_mode: RenderMode = "fit_to_model",
    image_format: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 100,
    encode: bool = True,
) -> tuple[np.ndarray, Optional[bytes], str, Tuple[int, int]]:
    """Render a PDFium page on CPU the same way ``pdf_extraction`` does.

    Returns ``(rgb_array, encoded_bytes_or_none, encoding, (height, width))``.
    """
    render_scale = compute_page_render_scale(page, dpi=dpi, render_mode=render_mode)
    bitmap = page.render(scale=render_scale)
    arr = convert_bitmap_to_corrected_numpy(bitmap)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    orig_h, orig_w = int(arr.shape[0]), int(arr.shape[1])
    fmt = image_format.lower()
    encoded = encode_rgb_image(arr, image_format=fmt, jpeg_quality=jpeg_quality) if encode else None
    return arr, encoded, fmt, (orig_h, orig_w)
