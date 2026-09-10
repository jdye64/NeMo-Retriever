# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline JPEG encoder. YCbCr + 8x8 DCT + quantization can run on CuPy arrays."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

_ZIGZAG = np.array(
    [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44,
        51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ],
    dtype=np.int32,
)

_LUMA_Q = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)

_CHROMA_Q = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float32,
)

# ITU T.81 Annex K standard Huffman tables.
_DC_L_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
_DC_L_VAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
_AC_L_BITS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
_AC_L_VAL = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
]
_DC_C_BITS = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
_DC_C_VAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
_AC_C_BITS = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119]
_AC_C_VAL = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
]


def _scale_q(table: np.ndarray, quality: int) -> np.ndarray:
    q = int(np.clip(quality, 1, 100))
    scale = 5000 / q if q < 50 else 200 - q * 2
    out = np.floor((table * scale + 50) / 100)
    return np.clip(out, 1, 255).astype(np.float32)


def _array_module(array: Any):
    if isinstance(array, np.ndarray):
        return np
    try:
        import cupy as cp
    except Exception:
        return np
    if isinstance(array, cp.ndarray):
        return cp
    return np


def _dct_matrix(xp):
    x = xp.arange(8, dtype=xp.float32)
    u = x.reshape(8, 1)
    cu = xp.where(u == 0, 1.0 / xp.sqrt(xp.asarray(2.0, dtype=xp.float32)), xp.asarray(1.0, dtype=xp.float32))
    return 0.5 * cu * xp.cos((2.0 * x + 1.0) * u * xp.pi / 16.0)


def composite_on_white(rgba: Any) -> Any:
    xp = _array_module(rgba)
    img = rgba.astype(xp.float32)
    if img.shape[-1] == 3:
        return img
    alpha = img[..., 3:4] / 255.0
    return img[..., :3] * alpha + 255.0 * (1.0 - alpha)


def quantize_blocks(rgb: Any, quality: int = 90) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Device-capable YCbCr DCT + quantization. Returns host zigzag blocks for Huffman."""
    xp = _array_module(rgb)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    ph = (h + 7) // 8 * 8
    pw = (w + 7) // 8 * 8
    padded = xp.ones((ph, pw, 3), dtype=xp.float32) * 128.0
    padded[:h, :w] = rgb
    r, g, b = padded[..., 0], padded[..., 1], padded[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b - 128.0
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    cmat = _dct_matrix(xp)
    luma_q = xp.asarray(_scale_q(_LUMA_Q, quality))
    chroma_q = xp.asarray(_scale_q(_CHROMA_Q, quality))

    def blocks(plane, qtable):
        tiles = plane.reshape(ph // 8, 8, pw // 8, 8).transpose(0, 2, 1, 3)
        dct = xp.einsum("ui,bcij,vj->bcuv", cmat, tiles, cmat)
        return xp.clip(xp.rint(dct / qtable), -1023, 1023)

    y_b = blocks(y, luma_q)
    cb_b = blocks(cb, chroma_q)
    cr_b = blocks(cr, chroma_q)
    to_host = np.asarray
    return (
        to_host(y_b).astype(np.int32),
        to_host(cb_b).astype(np.int32),
        to_host(cr_b).astype(np.int32),
        np.asarray(luma_q).astype(np.uint8),
        np.asarray(chroma_q).astype(np.uint8),
    )


def encode_jpeg(rgba: Any, quality: int = 90) -> bytes:
    rgb = composite_on_white(rgba)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    y_b, cb_b, cr_b, lq, cq = quantize_blocks(rgb, quality=quality)
    return _huffman_scan(y_b, cb_b, cr_b, lq, cq, width, height)


def _build_huffman(bits: list[int], values: list[int]) -> dict[int, tuple[int, int]]:
    table: dict[int, tuple[int, int]] = {}
    code = 0
    k = 0
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            table[values[k]] = (code, length)
            k += 1
            code += 1
        code <<= 1
    return table


class _Bits:
    def __init__(self) -> None:
        self._buf = bytearray()
        self._acc = 0
        self._n = 0

    def write(self, code: int, length: int) -> None:
        for i in range(length - 1, -1, -1):
            self._acc = (self._acc << 1) | ((code >> i) & 1)
            self._n += 1
            if self._n == 8:
                self._flush_byte()

    def write_amp(self, value: int, size: int) -> None:
        if size == 0:
            return
        if value < 0:
            value = value + ((1 << size) - 1)
        self.write(value, size)

    def _flush_byte(self) -> None:
        byte = self._acc & 0xFF
        self._buf.append(byte)
        if byte == 0xFF:
            self._buf.append(0x00)
        self._acc = 0
        self._n = 0

    def finish(self) -> bytes:
        if self._n:
            self._acc <<= 8 - self._n
            self._n = 8
            self._flush_byte()
        return bytes(self._buf)


def _category(value: int) -> int:
    if value == 0:
        return 0
    v = abs(int(value))
    size = 0
    while v:
        v >>= 1
        size += 1
    return size


def _encode_block(block: np.ndarray, prev_dc: int, dc_tab, ac_tab, bits: _Bits) -> int:
    flat = block.reshape(64)
    zz = flat[_ZIGZAG]
    dc = int(zz[0])
    diff = dc - prev_dc
    size = _category(diff)
    code, length = dc_tab[size]
    bits.write(code, length)
    bits.write_amp(diff, size)
    run = 0
    for i in range(1, 64):
        ac = int(zz[i])
        if ac == 0:
            run += 1
            continue
        while run >= 16:
            eob_code, eob_len = ac_tab[0xF0]
            bits.write(eob_code, eob_len)
            run -= 16
        size = _category(ac)
        rs = (run << 4) | size
        code, length = ac_tab[rs]
        bits.write(code, length)
        bits.write_amp(ac, size)
        run = 0
    if run:
        code, length = ac_tab[0x00]
        bits.write(code, length)
    return dc


def _dqt_segment(table: np.ndarray, dest: int) -> bytes:
    zz = table.reshape(64)[_ZIGZAG].astype(np.uint8)
    return b"\xff\xdb" + struct.pack(">H", 67) + bytes([dest]) + zz.tobytes()


def _dht_segment(bits: list[int], values: list[int], cls_ident: int) -> bytes:
    payload = bytes([cls_ident]) + bytes(bits) + bytes(values)
    return b"\xff\xc4" + struct.pack(">H", 2 + len(payload)) + payload


def _huffman_scan(
    y_b: np.ndarray,
    cb_b: np.ndarray,
    cr_b: np.ndarray,
    lq: np.ndarray,
    cq: np.ndarray,
    width: int,
    height: int,
) -> bytes:
    dc_l = _build_huffman(_DC_L_BITS, _DC_L_VAL)
    ac_l = _build_huffman(_AC_L_BITS, _AC_L_VAL)
    dc_c = _build_huffman(_DC_C_BITS, _DC_C_VAL)
    ac_c = _build_huffman(_AC_C_BITS, _AC_C_VAL)
    bits = _Bits()
    prev = [0, 0, 0]
    tiles_y, tiles_x = y_b.shape[0], y_b.shape[1]
    for by in range(tiles_y):
        for bx in range(tiles_x):
            prev[0] = _encode_block(y_b[by, bx], prev[0], dc_l, ac_l, bits)
            prev[1] = _encode_block(cb_b[by, bx], prev[1], dc_c, ac_c, bits)
            prev[2] = _encode_block(cr_b[by, bx], prev[2], dc_c, ac_c, bits)

    sof = (
        b"\xff\xc0"
        + struct.pack(">H", 17)
        + bytes([8])
        + struct.pack(">HH", height, width)
        + bytes([3, 1, 0x11, 0, 2, 0x11, 1, 3, 0x11, 1])
    )
    sos = b"\xff\xda" + struct.pack(">H", 12) + bytes([3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0])
    header = (
        b"\xff\xd8"
        + _dqt_segment(lq, 0)
        + _dqt_segment(cq, 1)
        + sof
        + _dht_segment(_DC_L_BITS, _DC_L_VAL, 0x00)
        + _dht_segment(_AC_L_BITS, _AC_L_VAL, 0x10)
        + _dht_segment(_DC_C_BITS, _DC_C_VAL, 0x01)
        + _dht_segment(_AC_C_BITS, _AC_C_VAL, 0x11)
        + sos
    )
    return header + bits.finish() + b"\xff\xd9"
