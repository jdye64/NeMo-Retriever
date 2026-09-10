# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class PageIR:
    index: int
    width_pt: float
    height_pt: float
    operations: Sequence[tuple[list[Any], str]]
    resources: Any
    reader: Any


@dataclass
class CompiledPath:
    page: int
    edges: np.ndarray  # (E, 4) x0,y0,x1,y1 in pixels
    rgba: np.ndarray  # (4,) float 0..1
    even_odd: bool = False


@dataclass
class CompiledBlit:
    page: int
    rgba: np.ndarray  # (h, w, 4) uint8 host texture (uploaded at raster time)
    dst: np.ndarray  # (4,) x0,y0,x1,y1 pixels
    multiply: np.ndarray  # (4,) fill color multiplier


@dataclass
class CompiledDocument:
    page_count: int
    page_widths: np.ndarray
    page_heights: np.ndarray
    max_width: int
    max_height: int
    dpi: float
    rects: np.ndarray  # (R, 9) page,x0,y0,x1,y1,r,g,b,a
    paths: list[CompiledPath] = field(default_factory=list)
    blits: list[CompiledBlit] = field(default_factory=list)
