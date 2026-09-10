# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np

from gpu_pdf_render.pdf.ir import CompiledDocument


def rasterize(compiled: CompiledDocument, backend: str):
    if backend == "cuda":
        return _rasterize_cuda(compiled)
    return _rasterize_reference(compiled)


def _empty_fb(compiled: CompiledDocument, xp) -> np.ndarray:
    fb = xp.full((compiled.page_count, compiled.max_height, compiled.max_width, 4), 255, dtype=xp.uint8)
    return fb


def _rasterize_reference(compiled: CompiledDocument) -> np.ndarray:
    fb = np.asarray(_empty_fb(compiled, np))
    _fill_rects_numpy(fb, compiled.rects)
    for path in compiled.paths:
        _fill_path_numpy(fb, path.page, path.edges, path.rgba, path.even_odd)
    for blit in compiled.blits:
        _blit_numpy(fb, blit.page, blit.dst, blit.rgba)
    return fb


def _rasterize_cuda(compiled: CompiledDocument):
    import cupy as cp

    from gpu_pdf_render.gpu import kernels

    fb = _empty_fb(compiled, cp)
    n, h, w = compiled.page_count, compiled.max_height, compiled.max_width
    if compiled.rects.shape[0]:
        rects = cp.asarray(compiled.rects, dtype=cp.float32)
        nrects = int(compiled.rects.shape[0])
        max_area = 0
        for row in compiled.rects:
            bw = max(1, int(math.ceil(row[3]) - math.floor(row[1])))
            bh = max(1, int(math.ceil(row[4]) - math.floor(row[2])))
            max_area = max(max_area, bw * bh)
        fill = cp.RawKernel(kernels.FILL_RECTS, "fill_rects")
        threads = 256
        blocks_x = (max_area + threads - 1) // threads
        fill((blocks_x, nrects), (threads,), (fb, n, h, w, rects, nrects))
    fill_path = cp.RawKernel(kernels.FILL_PATH, "fill_path")
    blit = cp.RawKernel(kernels.BLIT, "blit_image")
    threads = 256
    for path in compiled.paths:
        edges = cp.asarray(path.edges, dtype=cp.float32)
        npix = h * w
        blocks = (npix + threads - 1) // threads
        r, g, b, a = [float(v) for v in path.rgba]
        fill_path(
            (blocks,),
            (threads,),
            (fb, n, h, w, int(path.page), edges, int(path.edges.shape[0]), r, g, b, a, 1 if path.even_odd else 0),
        )
    for item in compiled.blits:
        src = cp.asarray(item.rgba)
        x0, y0, x1, y1 = [int(round(v)) for v in item.dst]
        x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
        area = max(1, (x1 - x0) * (y1 - y0))
        blocks = (area + threads - 1) // threads
        sh, sw = int(src.shape[0]), int(src.shape[1])
        blit((blocks,), (threads,), (fb, n, h, w, int(item.page), x0, y0, x1, y1, src, sh, sw))
    return fb


def _fill_rects_numpy(fb: np.ndarray, rects: np.ndarray) -> None:
    _, h, w, _ = fb.shape
    for row in rects:
        page = int(row[0])
        x0 = max(0, int(math.floor(row[1])))
        y0 = max(0, int(math.floor(row[2])))
        x1 = min(w, int(math.ceil(row[3])))
        y1 = min(h, int(math.ceil(row[4])))
        if x1 <= x0 or y1 <= y0:
            continue
        a = float(row[8])
        if a <= 0:
            continue
        color = np.array([row[5] * 255.0, row[6] * 255.0, row[7] * 255.0], dtype=np.float32)
        dest = fb[page, y0:y1, x0:x1].astype(np.float32)
        dest[..., :3] = color * a + dest[..., :3] * (1.0 - a)
        dest[..., 3] = (a + dest[..., 3] / 255.0 * (1.0 - a)) * 255.0
        fb[page, y0:y1, x0:x1] = np.clip(dest, 0, 255).astype(np.uint8)


def _fill_path_numpy(fb: np.ndarray, page: int, edges: np.ndarray, rgba: np.ndarray, even_odd: bool) -> None:
    _, h, w, _ = fb.shape
    ys, xs = np.mgrid[0:h, 0:w]
    px = xs.astype(np.float32) + 0.5
    py = ys.astype(np.float32) + 0.5
    wind = np.zeros((h, w), dtype=np.int32)
    cross = np.zeros((h, w), dtype=np.int32)
    for x0, y0, x1, y1 in edges:
        above0 = y0 > py
        above1 = y1 > py
        mask = above0 != above1
        t = (py - y0) / ((y1 - y0) + 1e-12)
        xsamp = x0 + t * (x1 - x0)
        hit = mask & (xsamp >= px)
        wind[hit] += np.where(y1 > y0, 1, -1)
        cross[hit] += 1
    inside = (cross % 2 == 1) if even_odd else (wind != 0)
    if not np.any(inside):
        return
    a = float(rgba[3])
    color = np.array([rgba[0] * 255.0, rgba[1] * 255.0, rgba[2] * 255.0], dtype=np.float32)
    dest = fb[page].astype(np.float32)
    dest[inside, :3] = color * a + dest[inside, :3] * (1.0 - a)
    dest[inside, 3] = (a + dest[inside, 3] / 255.0 * (1.0 - a)) * 255.0
    fb[page] = np.clip(dest, 0, 255).astype(np.uint8)


def _blit_numpy(fb: np.ndarray, page: int, dst: np.ndarray, src: np.ndarray) -> None:
    _, h, w, _ = fb.shape
    x0, y0, x1, y1 = [int(round(v)) for v in dst]
    x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
    if x1 <= x0 or y1 <= y0:
        return
    sh, sw = src.shape[:2]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    u = ((xx - x0).astype(np.float32) + 0.5) / float(x1 - x0)
    v = ((yy - y0).astype(np.float32) + 0.5) / float(y1 - y0)
    sx = np.clip((u * sw).astype(np.int32), 0, sw - 1)
    sy = np.clip((v * sh).astype(np.int32), 0, sh - 1)
    sampled = src[sy, sx]
    a = sampled[..., 3:4].astype(np.float32) / 255.0
    dest = fb[page, y0:y1, x0:x1].astype(np.float32)
    dest[..., :3] = sampled[..., :3].astype(np.float32) * a + dest[..., :3] * (1.0 - a)
    dest[..., 3:4] = (a + dest[..., 3:4] / 255.0 * (1.0 - a)) * 255.0
    fb[page, y0:y1, x0:x1] = np.clip(dest, 0, 255).astype(np.uint8)
