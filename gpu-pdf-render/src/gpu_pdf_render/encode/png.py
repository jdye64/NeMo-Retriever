# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
import zlib

import numpy as np


def encode_png(rgba: np.ndarray) -> bytes:
    """Encode an HxWx3 or HxWx4 uint8 image as PNG.

    Scanline packing uses filter 0. DEFLATE of IDAT is sequential, so it runs on the
    host after pixels are copied off the GPU.
    """
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError("encode_png expects HxWx3 or HxWx4 uint8")
    image = np.ascontiguousarray(rgba, dtype=np.uint8)
    height, width, channels = image.shape
    color_type = 2 if channels == 3 else 6
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))
    compressed = zlib.compress(raw, 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
