# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

RenderBackend = Literal["cuda", "reference"]


class GpuUnavailableError(RuntimeError):
    """Raised when the CUDA backend is requested but no usable GPU/CuPy is present."""
