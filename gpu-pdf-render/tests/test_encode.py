# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from gpu_pdf_render.encode.jpeg import encode_jpeg
from gpu_pdf_render.encode.png import encode_png


def test_png_roundtrip_signature_and_size() -> None:
    pixels = np.zeros((16, 24, 4), dtype=np.uint8)
    pixels[..., 0] = 200
    pixels[..., 3] = 255
    encoded = encode_png(pixels)
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
    image = Image.open(BytesIO(encoded))
    assert image.size == (24, 16)


def test_jpeg_roundtrip_opens() -> None:
    pixels = np.full((16, 24, 4), 255, dtype=np.uint8)
    pixels[..., 1] = 40
    encoded = encode_jpeg(pixels, quality=85)
    assert encoded[:2] == b"\xff\xd8"
    assert encoded[-2:] == b"\xff\xd9"
    image = Image.open(BytesIO(encoded))
    assert image.size == (24, 16)
    arr = np.asarray(image.convert("RGB"))
    assert arr[8, 12, 0] > 180
    assert arr[8, 12, 1] < 80
