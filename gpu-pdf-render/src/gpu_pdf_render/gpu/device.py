# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from gpu_pdf_render.types import GpuUnavailableError, RenderBackend


def resolve_backend(backend: RenderBackend | str) -> RenderBackend:
    name = str(backend).lower()
    if name not in {"cuda", "reference"}:
        raise ValueError(f"Unknown backend {backend!r}; expected 'cuda' or 'reference'")
    if name == "cuda":
        _require_cuda()
    return name  # type: ignore[return-value]


def cuda_available() -> bool:
    try:
        _require_cuda()
        return True
    except GpuUnavailableError:
        return False


def _require_cuda() -> None:
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - import error path
        raise GpuUnavailableError(
            "CUDA backend requires CuPy. Install with: pip install 'gpu-pdf-render[cuda]'"
        ) from exc
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            raise GpuUnavailableError("No CUDA device is visible")
        cp.cuda.Device(0).use()
        cp.zeros(1, dtype=cp.uint8)
    except GpuUnavailableError:
        raise
    except Exception as exc:
        raise GpuUnavailableError(f"CUDA is not usable: {exc}") from exc
