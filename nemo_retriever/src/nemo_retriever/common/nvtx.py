# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight NVTX helpers for GPU inference profiling.

Usage::

    from nemo_retriever.common.nvtx import gpu_inference_range

    with gpu_inference_range("NemotronOCRv1", batch_size=8):
        result = self._model(input_data)

When ``nsys`` is launched with ``--capture-range=nvtx --nvtx-capture=gpu_inference``,
only the code inside these blocks is captured.  When no profiler is attached the
overhead is near-zero (a pair of C-level push/pop calls).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch.cuda.nvtx as _nvtx

from nemo_retriever.common.tracing.spans import model_span


@contextmanager
def gpu_inference_range(model_name: str, batch_size: int = -1, **extra):
    """Push a nested NVTX range pair around GPU inference.

    The outer range is always named ``gpu_inference`` so that
    ``nsys --nvtx-capture=gpu_inference`` can trigger on it.
    The inner range carries the human-readable label visible in
    the Nsight Systems timeline (e.g. ``NemotronOCRv1 | bs=8``).

    Every local GPU model already wraps its forward pass in this helper, so it
    doubles as the page trace hook for on-device inference. The trace span
    records only entry and exit times around the call; it does not synchronize
    CUDA, so the pipeline runs the same whether or not tracing is on. Attach
    Nsight Systems to the NVTX ranges when you need device-level attribution.
    """
    parts = [model_name]
    if batch_size >= 0:
        parts.append(f"bs={batch_size}")
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    label = " | ".join(parts)
    _nvtx.range_push("gpu_inference")
    _nvtx.range_push(label)
    try:
        with model_span(
            model_name,
            name=model_name,
            backend="local-gpu",
            category="gpu",
            span_name=f"gpu.{model_name}",
            batch_size=batch_size if batch_size >= 0 else None,
            attrs={key: str(value) for key, value in extra.items()},
        ):
            yield
    finally:
        _nvtx.range_pop()
        _nvtx.range_pop()
