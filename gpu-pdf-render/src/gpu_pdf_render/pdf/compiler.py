# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pypdf.generic import ArrayObject

from gpu_pdf_render.glyphs import glyph_bitmap
from gpu_pdf_render.pdf import resolve_name
from gpu_pdf_render.pdf.ir import CompiledBlit, CompiledDocument, CompiledPath, PageIR

_MAX_FORM_DEPTH = 16


def compile_document(pages: list[PageIR], dpi: float) -> CompiledDocument:
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    scale = dpi / 72.0
    widths = np.array([max(1, int(math.ceil(p.width_pt * scale))) for p in pages], dtype=np.int32)
    heights = np.array([max(1, int(math.ceil(p.height_pt * scale))) for p in pages], dtype=np.int32)
    compiler = _Compiler(scale=scale)
    for page in pages:
        compiler.compile_page(page)
    rects = np.asarray(compiler.rects, dtype=np.float32) if compiler.rects else np.zeros((0, 9), dtype=np.float32)
    return CompiledDocument(
        page_count=len(pages),
        page_widths=widths,
        page_heights=heights,
        max_width=int(widths.max()),
        max_height=int(heights.max()),
        dpi=float(dpi),
        rects=rects,
        paths=compiler.paths,
        blits=compiler.blits,
    )


@dataclass
class _GraphicsState:
    ctm: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    fill: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    stroke: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    line_width: float = 1.0
    font_name: str | None = None
    font_size: float = 12.0
    text_matrix: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    text_line_matrix: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    leading: float = 0.0
    char_spacing: float = 0.0
    word_spacing: float = 0.0
    horiz_scale: float = 1.0

    def clone(self) -> _GraphicsState:
        return _GraphicsState(**self.__dict__)


@dataclass
class _Compiler:
    scale: float
    rects: list[list[float]] = field(default_factory=list)
    paths: list[CompiledPath] = field(default_factory=list)
    blits: list[CompiledBlit] = field(default_factory=list)

    def compile_page(self, page: PageIR) -> None:
        state = _GraphicsState()
        stack: list[_GraphicsState] = []
        path: list[tuple[float, float]] = []
        subpath_start: tuple[float, float] | None = None
        current: tuple[float, float] | None = None
        in_text = False
        self._run_ops(
            page=page,
            operations=page.operations,
            resources=page.resources,
            state=state,
            stack=stack,
            path=path,
            subpath_start=subpath_start,
            current=current,
            in_text=in_text,
            form_depth=0,
        )

    def _run_ops(
        self,
        page: PageIR,
        operations: list[tuple[list[Any], str]],
        resources: Any,
        state: _GraphicsState,
        stack: list[_GraphicsState],
        path: list[tuple[float, float]],
        subpath_start: tuple[float, float] | None,
        current: tuple[float, float] | None,
        in_text: bool,
        form_depth: int,
    ) -> None:
        for operands, op in operations:
            if op == "q":
                stack.append(state.clone())
            elif op == "Q":
                if stack:
                    state.__dict__.update(stack.pop().__dict__)
            elif op == "cm" and len(operands) >= 6:
                extra = tuple(float(v) for v in operands[:6])
                state.ctm = _mul(state.ctm, extra)
            elif op == "w" and operands:
                state.line_width = float(operands[0])
            elif op == "rg" and len(operands) >= 3:
                state.fill = (float(operands[0]), float(operands[1]), float(operands[2]), 1.0)
            elif op == "RG" and len(operands) >= 3:
                state.stroke = (float(operands[0]), float(operands[1]), float(operands[2]), 1.0)
            elif op == "g" and operands:
                v = float(operands[0])
                state.fill = (v, v, v, 1.0)
            elif op == "G" and operands:
                v = float(operands[0])
                state.stroke = (v, v, v, 1.0)
            elif op == "k" and len(operands) >= 4:
                c, m, y, k = (float(v) for v in operands[:4])
                state.fill = _cmyk_to_rgb(c, m, y, k)
            elif op == "K" and len(operands) >= 4:
                c, m, y, k = (float(v) for v in operands[:4])
                state.stroke = _cmyk_to_rgb(c, m, y, k)
            elif op == "m" and len(operands) >= 2:
                current = (float(operands[0]), float(operands[1]))
                subpath_start = current
                path.append(current)
            elif op == "l" and len(operands) >= 2 and current is not None:
                nxt = (float(operands[0]), float(operands[1]))
                path.extend([current, nxt])
                current = nxt
            elif op in {"c", "v", "y"} and current is not None:
                pts = _cubic_points(op, current, operands)
                flat = _flatten_cubic(current, pts[0], pts[1], pts[2])
                for a, b in zip(flat[:-1], flat[1:]):
                    path.extend([a, b])
                current = pts[2]
            elif op == "h" and current is not None and subpath_start is not None:
                path.extend([current, subpath_start])
                current = subpath_start
            elif op == "re" and len(operands) >= 4:
                x, y, w, h = (float(v) for v in operands[:4])
                path.extend(_rect_edges(x, y, w, h))
                current = (x, y)
                subpath_start = current
            elif op in {"n"}:
                path.clear()
                current = None
                subpath_start = None
            elif op in {"f", "F", "f*"}:
                self._fill_path(page, state, path, even_odd=(op == "f*"))
                path.clear()
                current = None
                subpath_start = None
            elif op in {"S", "s"}:
                if op == "s" and current is not None and subpath_start is not None:
                    path.extend([current, subpath_start])
                self._stroke_path(page, state, path)
                path.clear()
                current = None
                subpath_start = None
            elif op in {"B", "B*", "b", "b*"}:
                if op in {"b", "b*"} and current is not None and subpath_start is not None:
                    path.extend([current, subpath_start])
                self._fill_path(page, state, path, even_odd=op.endswith("*"))
                self._stroke_path(page, state, path)
                path.clear()
                current = None
                subpath_start = None
            elif op == "Do" and operands:
                self._do_xobject(page, resources, state, operands[0], form_depth)
            elif op == "BT":
                in_text = True
                state.text_matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                state.text_line_matrix = state.text_matrix
            elif op == "ET":
                in_text = False
            elif op == "Td" and len(operands) >= 2:
                tx, ty = float(operands[0]), float(operands[1])
                state.text_line_matrix = _mul(state.text_line_matrix, (1.0, 0.0, 0.0, 1.0, tx, ty))
                state.text_matrix = state.text_line_matrix
            elif op == "TD" and len(operands) >= 2:
                tx, ty = float(operands[0]), float(operands[1])
                state.leading = -ty
                state.text_line_matrix = _mul(state.text_line_matrix, (1.0, 0.0, 0.0, 1.0, tx, ty))
                state.text_matrix = state.text_line_matrix
            elif op == "Tm" and len(operands) >= 6:
                state.text_matrix = tuple(float(v) for v in operands[:6])  # type: ignore[assignment]
                state.text_line_matrix = state.text_matrix
            elif op == "T*":
                state.text_line_matrix = _mul(state.text_line_matrix, (1.0, 0.0, 0.0, 1.0, 0.0, -state.leading))
                state.text_matrix = state.text_line_matrix
            elif op == "TL" and operands:
                state.leading = float(operands[0])
            elif op == "Tc" and operands:
                state.char_spacing = float(operands[0])
            elif op == "Tw" and operands:
                state.word_spacing = float(operands[0])
            elif op == "Tz" and operands:
                state.horiz_scale = float(operands[0]) / 100.0
            elif op == "Tf" and len(operands) >= 2:
                state.font_name = str(operands[0])
                state.font_size = float(operands[1])
            elif op in {"Tj", "'", '"'} and in_text and operands:
                if op == "'":
                    state.text_line_matrix = _mul(state.text_line_matrix, (1.0, 0.0, 0.0, 1.0, 0.0, -state.leading))
                    state.text_matrix = state.text_line_matrix
                self._show_text(page, state, _pdf_string(operands[0]))
            elif op == "TJ" and in_text and operands:
                self._show_text_array(page, state, operands[0])
            elif op in {"W", "W*"}:
                # Research subset: clipping paths are ignored (pixels still rasterize).
                path.clear()
                current = None
                subpath_start = None

        # silence unused
        _ = in_text

    def _fill_path(self, page: PageIR, state: _GraphicsState, path: list[tuple[float, float]], even_odd: bool) -> None:
        if len(path) < 2:
            return
        rect = _try_axis_aligned_rect(path)
        if rect is not None and not even_odd:
            x, y, w, h = rect
            self._emit_rect(page, state, x, y, w, h, state.fill)
            return
        edges = []
        for i in range(0, len(path) - 1, 2):
            x0, y0 = self._to_px(page, state.ctm, path[i][0], path[i][1])
            x1, y1 = self._to_px(page, state.ctm, path[i + 1][0], path[i + 1][1])
            edges.append([x0, y0, x1, y1])
        if not edges:
            return
        self.paths.append(
            CompiledPath(
                page=page.index,
                edges=np.asarray(edges, dtype=np.float32),
                rgba=np.asarray(state.fill, dtype=np.float32),
                even_odd=even_odd,
            )
        )

    def _stroke_path(self, page: PageIR, state: _GraphicsState, path: list[tuple[float, float]]) -> None:
        width = max(state.line_width, 0.25)
        for i in range(0, len(path) - 1, 2):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            self._emit_stroke_segment(page, state, x0, y0, x1, y1, width)

    def _emit_stroke_segment(
        self,
        page: PageIR,
        state: _GraphicsState,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        width: float,
    ) -> None:
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy / length * width / 2.0, dx / length * width / 2.0
        quad = [
            (x0 + nx, y0 + ny),
            (x1 + nx, y1 + ny),
            (x1 - nx, y1 - ny),
            (x0 - nx, y0 - ny),
            (x0 + nx, y0 + ny),
        ]
        edges = []
        for a, b in zip(quad[:-1], quad[1:]):
            p0 = self._to_px(page, state.ctm, a[0], a[1])
            p1 = self._to_px(page, state.ctm, b[0], b[1])
            edges.append([p0[0], p0[1], p1[0], p1[1]])
        self.paths.append(
            CompiledPath(
                page=page.index,
                edges=np.asarray(edges, dtype=np.float32),
                rgba=np.asarray(state.stroke, dtype=np.float32),
                even_odd=False,
            )
        )

    def _emit_rect(
        self,
        page: PageIR,
        state: _GraphicsState,
        x: float,
        y: float,
        w: float,
        h: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        pts = [self._to_px(page, state.ctm, px, py) for px, py in corners]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.rects.append(
            [
                float(page.index),
                min(xs),
                min(ys),
                max(xs),
                max(ys),
                rgba[0],
                rgba[1],
                rgba[2],
                rgba[3],
            ]
        )

    def _do_xobject(self, page: PageIR, resources: Any, state: _GraphicsState, name: Any, form_depth: int) -> None:
        xobj = resolve_name(resources, "/XObject", name)
        if xobj is None:
            return
        subtype = str(xobj.get("/Subtype", ""))
        if subtype.endswith("Image"):
            image = _decode_image_xobject(xobj)
            if image is None:
                return
            # PDF images occupy the unit square in user space, mapped by CTM.
            p00 = self._to_px(page, state.ctm, 0.0, 0.0)
            p10 = self._to_px(page, state.ctm, 1.0, 0.0)
            p01 = self._to_px(page, state.ctm, 0.0, 1.0)
            xs = [p00[0], p10[0], p01[0]]
            ys = [p00[1], p10[1], p01[1]]
            self.blits.append(
                CompiledBlit(
                    page=page.index,
                    rgba=image,
                    dst=np.asarray([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32),
                    multiply=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                )
            )
            return
        if subtype.endswith("Form") and form_depth < _MAX_FORM_DEPTH:
            matrix = xobj.get("/Matrix")
            form_state = state.clone()
            if matrix is not None:
                vals = [float(v) for v in list(matrix)[:6]]
                form_state.ctm = _mul(state.ctm, tuple(vals))  # type: ignore[arg-type]
            form_res = xobj.get("/Resources") or resources
            if form_res is not None:
                form_res = form_res.get_object()
            from pypdf.generic import ContentStream

            stream = ContentStream(xobj, page.reader)
            ops = []
            for operands, operator in stream.operations:
                op = operator.decode("latin1") if isinstance(operator, (bytes, bytearray)) else str(operator)
                ops.append((list(operands), op))
            self._run_ops(
                page=page,
                operations=ops,
                resources=form_res,
                state=form_state,
                stack=[],
                path=[],
                subpath_start=None,
                current=None,
                in_text=False,
                form_depth=form_depth + 1,
            )

    def _show_text_array(self, page: PageIR, state: _GraphicsState, arr: Any) -> None:
        items = list(arr) if isinstance(arr, (list, ArrayObject)) else [arr]
        for item in items:
            if isinstance(item, (int, float)) or type(item).__name__ in {"NumberObject", "FloatObject"}:
                try:
                    adj = float(item)
                except Exception:
                    continue
                dx = -adj / 1000.0 * state.font_size
                state.text_matrix = _mul(state.text_matrix, (1.0, 0.0, 0.0, 1.0, dx, 0.0))
            else:
                self._show_text(page, state, _pdf_string(item))

    def _show_text(self, page: PageIR, state: _GraphicsState, text: str) -> None:
        for ch in text:
            code = ord(ch) & 255
            extra = state.word_spacing if ch == " " else 0.0
            glyph_w = state.font_size * 0.6 * state.horiz_scale
            combined = _mul(state.ctm, state.text_matrix)
            origin = self._to_px(page, combined, 0.0, 0.0)
            x1y = self._to_px(page, combined, glyph_w, 0.0)
            xy1 = self._to_px(page, combined, 0.0, state.font_size)
            xs = [origin[0], x1y[0], xy1[0]]
            ys = [origin[1], x1y[1], xy1[1]]
            mask = glyph_bitmap(code)
            h, w = mask.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgb = np.clip(np.asarray(state.fill[:3]) * 255.0, 0, 255).astype(np.uint8)
            rgba[..., 0] = rgb[0]
            rgba[..., 1] = rgb[1]
            rgba[..., 2] = rgb[2]
            rgba[..., 3] = mask
            self.blits.append(
                CompiledBlit(
                    page=page.index,
                    rgba=rgba,
                    dst=np.asarray([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32),
                    multiply=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                )
            )
            adv = glyph_w + state.char_spacing + extra
            state.text_matrix = _mul(state.text_matrix, (1.0, 0.0, 0.0, 1.0, adv, 0.0))

    def _to_px(
        self, page: PageIR, ctm: tuple[float, float, float, float, float, float], x: float, y: float
    ) -> tuple[float, float]:
        a, b, c, d, e, f = ctm
        ux = a * x + c * y + e
        uy = b * x + d * y + f
        px = ux * self.scale
        py = (page.height_pt - uy) * self.scale
        return px, py


def _mul(
    m1: tuple[float, float, float, float, float, float],
    m2: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _cmyk_to_rgb(c: float, m: float, y: float, k: float) -> tuple[float, float, float, float]:
    return ((1.0 - c) * (1.0 - k), (1.0 - m) * (1.0 - k), (1.0 - y) * (1.0 - k), 1.0)


def _pdf_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("latin1", errors="replace")
    text = getattr(value, "original_bytes", None)
    if isinstance(text, bytes):
        return text.decode("latin1", errors="replace")
    return str(value)


def _rect_edges(x: float, y: float, w: float, h: float) -> list[tuple[float, float]]:
    x1, y1 = x + w, y + h
    return [
        (x, y),
        (x1, y),
        (x1, y),
        (x1, y1),
        (x1, y1),
        (x, y1),
        (x, y1),
        (x, y),
    ]


def _try_axis_aligned_rect(path: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    pts = path
    if len(pts) == 8:
        xs = sorted({round(p[0], 5) for p in pts})
        ys = sorted({round(p[1], 5) for p in pts})
        if len(xs) == 2 and len(ys) == 2:
            return xs[0], ys[0], xs[1] - xs[0], ys[1] - ys[0]
    return None


def _cubic_points(op: str, current: tuple[float, float], operands: list[Any]) -> list[tuple[float, float]]:
    vals = [float(v) for v in operands]
    if op == "c" and len(vals) >= 6:
        return [(vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5])]
    if op == "v" and len(vals) >= 4:
        return [current, (vals[0], vals[1]), (vals[2], vals[3])]
    if op == "y" and len(vals) >= 4:
        return [(vals[0], vals[1]), (vals[2], vals[3]), (vals[2], vals[3])]
    return [current, current, current]


def _flatten_cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    depth: int = 0,
) -> list[tuple[float, float]]:
    if depth > 8 or _flat_enough(p0, p1, p2, p3):
        return [p0, p3]
    m01 = _mid(p0, p1)
    m12 = _mid(p1, p2)
    m23 = _mid(p2, p3)
    m012 = _mid(m01, m12)
    m123 = _mid(m12, m23)
    m = _mid(m012, m123)
    return _flatten_cubic(p0, m01, m012, m, depth + 1)[:-1] + _flatten_cubic(m, m123, m23, p3, depth + 1)


def _mid(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _flat_enough(p0, p1, p2, p3) -> bool:
    d1 = abs((p1[0] - p0[0]) * (p3[1] - p0[1]) - (p1[1] - p0[1]) * (p3[0] - p0[0]))
    d2 = abs((p2[0] - p0[0]) * (p3[1] - p0[1]) - (p2[1] - p0[1]) * (p3[0] - p0[0]))
    return (d1 + d2) < 0.35 * (abs(p3[0] - p0[0]) + abs(p3[1] - p0[1]) + 1.0)


def _decode_image_xobject(xobj: Any) -> np.ndarray | None:
    try:
        width = int(xobj.get("/Width"))
        height = int(xobj.get("/Height"))
        data = xobj.get_data()
    except Exception:
        return None
    if not data or width <= 0 or height <= 0:
        return None
    if data[:2] == b"\xff\xd8":
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(data)).convert("RGBA")
        return np.asarray(img, dtype=np.uint8)
    n = width * height
    if len(data) >= n * 3:
        rgb = np.frombuffer(data[: n * 3], dtype=np.uint8).reshape(height, width, 3)
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        return rgba
    if len(data) >= n:
        gray = np.frombuffer(data[:n], dtype=np.uint8).reshape(height, width)
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = gray
        rgba[..., 1] = gray
        rgba[..., 2] = gray
        rgba[..., 3] = 255
        return rgba
    return None
