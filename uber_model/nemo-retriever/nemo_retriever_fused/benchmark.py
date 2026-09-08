# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the fused pipeline and account for its transfer traffic.

Two things are measured:

Stage time
    Device milliseconds per stage, from the CUDA events the pipeline records.
    Because the pipeline synchronises once, these add up to the real wall time
    rather than to a sum inflated by per-stage syncs.

Transfer traffic
    Bytes crossing PCIe, counted by instrumenting the tensor copies. This is the
    number the fused design is built to reduce, and it is the one that does not
    vary with GPU model, so it is the more portable comparison against the
    production pipeline.

Usage::

    python -m nemo_retriever_fused.benchmark --pages page1.jpg page2.jpg
    python -m nemo_retriever_fused.benchmark --synthetic 16 --iterations 5
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from nemo_retriever_fused.config import FusedPipelineConfig
from nemo_retriever_fused.results import FusedResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TransferAccount:
    """Host-to-device and device-to-host byte counts for one run."""

    host_to_device: int = 0
    device_to_host: int = 0
    host_to_device_calls: int = 0
    device_to_host_calls: int = 0

    def add(self, other: TransferAccount) -> None:
        """Accumulate *other* into this account."""
        self.host_to_device += other.host_to_device
        self.device_to_host += other.device_to_host
        self.host_to_device_calls += other.host_to_device_calls
        self.device_to_host_calls += other.device_to_host_calls

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        return (
            f"H2D {self.host_to_device / 1e6:8.2f} MB in {self.host_to_device_calls:5d} copies | "
            f"D2H {self.device_to_host / 1e6:8.2f} MB in {self.device_to_host_calls:5d} copies"
        )


class transfer_accounting:  # noqa: N801 - used as a context manager, not a class API
    """Count bytes crossing PCIe inside the enclosing block.

    Wraps `torch.Tensor.to` and `torch.Tensor.cpu` so that every copy between
    host and device is attributed. This is instrumentation for benchmarking, so
    it is deliberately not active on the normal inference path.

    Examples
    --------
    >>> with transfer_accounting() as account:
    ...     model(pages)
    >>> print(account.summary())
    """

    def __init__(self) -> None:
        self.account = TransferAccount()
        self._original_to = torch.Tensor.to
        self._original_cpu = torch.Tensor.cpu

    def __enter__(self) -> TransferAccount:
        account = self.account
        original_to = self._original_to
        original_cpu = self._original_cpu

        def counted_to(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            result = original_to(tensor, *args, **kwargs)
            if isinstance(result, torch.Tensor) and result.device != tensor.device:
                nbytes = tensor.element_size() * tensor.numel()
                if tensor.device.type == "cpu" and result.device.type == "cuda":
                    account.host_to_device += nbytes
                    account.host_to_device_calls += 1
                elif tensor.device.type == "cuda" and result.device.type == "cpu":
                    account.device_to_host += nbytes
                    account.device_to_host_calls += 1
            return result

        def counted_cpu(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            if tensor.device.type == "cuda":
                account.device_to_host += tensor.element_size() * tensor.numel()
                account.device_to_host_calls += 1
            return original_cpu(tensor, *args, **kwargs)

        torch.Tensor.to = counted_to  # type: ignore[method-assign]
        torch.Tensor.cpu = counted_cpu  # type: ignore[method-assign]
        return account

    def __exit__(self, *exc_info: object) -> None:
        torch.Tensor.to = self._original_to  # type: ignore[method-assign]
        torch.Tensor.cpu = self._original_cpu  # type: ignore[method-assign]


@dataclass(slots=True)
class BenchmarkReport:
    """Aggregated results across benchmark iterations."""

    pages: int
    iterations: int
    wall_milliseconds: list[float] = field(default_factory=list)
    stage_milliseconds: dict[str, list[float]] = field(default_factory=dict)
    transfers: TransferAccount = field(default_factory=TransferAccount)

    def record(self, result: FusedResult, wall_ms: float) -> None:
        """Record one iteration's timings."""
        self.wall_milliseconds.append(wall_ms)
        for timing in result.timings:
            self.stage_milliseconds.setdefault(timing.name, []).append(timing.milliseconds)

    def render(self) -> str:
        """Return the report as a fixed-width table."""
        if not self.wall_milliseconds:
            return "no iterations recorded"

        lines = [
            f"pages/iteration: {self.pages}    iterations: {self.iterations}",
            "",
            f"{'stage':<18}{'mean ms':>10}{'p50 ms':>10}{'ms/page':>10}",
        ]
        for name, samples in self.stage_milliseconds.items():
            mean = statistics.fmean(samples)
            lines.append(
                f"{name:<18}{mean:>10.2f}{statistics.median(samples):>10.2f}" f"{mean / max(self.pages, 1):>10.3f}"
            )

        wall_mean = statistics.fmean(self.wall_milliseconds)
        lines.extend(
            [
                "",
                f"{'wall':<18}{wall_mean:>10.2f}"
                f"{statistics.median(self.wall_milliseconds):>10.2f}"
                f"{wall_mean / max(self.pages, 1):>10.3f}",
                f"throughput: {self.pages * 1000.0 / wall_mean:.2f} pages/s",
                "",
                "transfers per iteration:",
                f"  {self.transfers.summary()}",
            ]
        )
        return "\n".join(lines)


def synthetic_page(width: int = 1700, height: int = 2200) -> str:
    """Return a base64 JPEG page of *width* by *height* for smoke benchmarks.

    The content is a light background with darker bands, which gives the
    detectors and OCR something non-degenerate to work on without needing a
    fixture corpus checked into the repository.
    """
    import numpy as np
    from PIL import Image

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    for top in range(120, height - 120, 180):
        canvas[top : top + 40, 100 : width - 100] = 40
    for left in range(150, width - 150, 260):
        canvas[height // 2 : height // 2 + 400, left : left + 4] = 90

    buffer = io.BytesIO()
    Image.fromarray(canvas).save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def run_benchmark(
    pages: list[str],
    *,
    config: FusedPipelineConfig | None = None,
    iterations: int = 3,
    warmup: int = 1,
) -> BenchmarkReport:
    """Benchmark the fused pipeline over *pages*.

    Parameters
    ----------
    pages:
        Base64 page images.
    config:
        Pipeline configuration. Defaults to `FusedPipelineConfig()`.
    iterations:
        Timed iterations to run.
    warmup:
        Untimed iterations to run first, so that cuDNN autotuning and any lazy
        allocation are not attributed to the measurement.
    """
    from nemo_retriever_fused.pipeline import NemoRetrieverFusedModel

    model = NemoRetrieverFusedModel.from_pretrained(config)

    for _ in range(warmup):
        model(pages)

    report = BenchmarkReport(pages=len(pages), iterations=iterations)
    for iteration in range(iterations):
        # Account transfers only on the first timed iteration; the wrapper adds
        # per-call Python overhead that would distort the timing samples.
        if iteration == 0:
            with transfer_accounting() as account:
                start = time.perf_counter()
                result = model(pages)
                wall_ms = (time.perf_counter() - start) * 1000.0
            report.transfers.add(account)
        else:
            start = time.perf_counter()
            result = model(pages)
            wall_ms = (time.perf_counter() - start) * 1000.0

        report.record(result, wall_ms)
        logger.info("iteration %d: %.2f ms", iteration + 1, wall_ms)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pages", nargs="+", type=Path, help="Page image files to benchmark.")
    source.add_argument("--synthetic", type=int, help="Benchmark N generated pages instead.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--no-overlap", action="store_true", help="Disable detector stream overlap.")
    parser.add_argument("--no-device-decode", action="store_true", help="Disable nvJPEG decoding.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.synthetic:
        pages = [synthetic_page() for _ in range(args.synthetic)]
    else:
        pages = [base64.b64encode(path.read_bytes()).decode("ascii") for path in args.pages]

    config = FusedPipelineConfig()
    if args.no_overlap:
        config.execution.overlap_detectors = False
    if args.no_device_decode:
        config.execution.decode_on_device = False

    report = run_benchmark(pages, config=config, iterations=args.iterations, warmup=args.warmup)
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
