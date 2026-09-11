# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare CPU (PDFium) and GPU PDF engine rasterization on a directory of PDFs."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import typer

from nemo_retriever.common.api.util.pdf.engine import (
    PdfEngineBackend,
    RenderMode,
    create_pdf_engine,
)

app = typer.Typer(help="Benchmark CPU vs GPU PDF engines on a directory of PDFs.")


@dataclass
class EngineFileResult:
    backend: str
    device: str
    path: str
    pages: int
    load_seconds: float
    info_seconds: float
    raster_seconds: float
    total_seconds: float
    encoded_bytes: int
    pages_per_second: float
    error: Optional[str] = None


@dataclass
class EngineSummary:
    backend: str
    device: str
    files: int
    pages: int
    load_seconds: float
    info_seconds: float
    raster_seconds: float
    total_seconds: float
    encoded_bytes: int
    pages_per_second: float
    failed_files: int


def collect_pdf_paths(input_dir: Path, *, recursive: bool) -> List[Path]:
    root = input_dir.expanduser().resolve()
    if not root.is_dir():
        raise typer.BadParameter(f"Input directory does not exist: {root}")
    pattern = "**/*.pdf" if recursive else "*.pdf"
    paths = sorted({p.resolve() for p in root.glob(pattern) if p.is_file()})
    if not paths:
        raise typer.BadParameter(f"No PDF files found in {root}")
    return paths


def _parse_backends(value: str) -> List[PdfEngineBackend]:
    out: List[PdfEngineBackend] = []
    for part in str(value or "").split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token not in {"cpu", "gpu"}:
            raise typer.BadParameter(f"backends must be cpu, gpu, or both, got {value!r}")
        if token not in out:
            out.append(token)  # type: ignore[arg-type]
    if not out:
        raise typer.BadParameter("backends cannot be empty")
    return out


def _gpu_memory_bytes() -> Optional[int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def benchmark_pdf_engine(
    paths: Sequence[Path],
    *,
    backend: PdfEngineBackend,
    dpi: int,
    render_mode: RenderMode,
    image_format: str,
    jpeg_quality: int,
    warmup: bool,
    keep_on_device: bool,
    device: Optional[str] = None,
) -> tuple[EngineSummary, List[EngineFileResult]]:
    engine = create_pdf_engine(backend, device=device, keep_on_device=keep_on_device)
    resolved_device = engine.device
    file_results: List[EngineFileResult] = []

    if warmup and paths:
        try:
            engine.load(paths[0])
            engine.page_info()
            engine.rasterize_pages(
                dpi=dpi,
                render_mode=render_mode,
                image_format=image_format,  # type: ignore[arg-type]
                jpeg_quality=jpeg_quality,
                keep_on_device=keep_on_device,
            )
            engine.release_device_images()
        except Exception:
            pass
        finally:
            engine.close()

    load_s = info_s = raster_s = 0.0
    encoded_bytes = 0
    pages = 0
    failed = 0

    for path in paths:
        t_file = time.perf_counter()
        try:
            t0 = time.perf_counter()
            engine.load(path)
            file_load = time.perf_counter() - t0

            t0 = time.perf_counter()
            infos = engine.page_info()
            file_info = time.perf_counter() - t0
            n_pages = len(infos) if isinstance(infos, list) else 1

            t0 = time.perf_counter()
            images = engine.rasterize_pages(
                dpi=dpi,
                render_mode=render_mode,
                image_format=image_format,  # type: ignore[arg-type]
                jpeg_quality=jpeg_quality,
                keep_on_device=keep_on_device,
            )
            if backend == "gpu":
                try:
                    import torch

                    if engine.device.startswith("cuda"):
                        torch.cuda.synchronize()
                except Exception:
                    pass
            file_raster = time.perf_counter() - t0
            file_encoded = sum(int(img.nbytes) for img in images)
            engine.release_device_images()
            engine.close()
            file_total = time.perf_counter() - t_file
            pps = (n_pages / file_raster) if file_raster > 0 else 0.0
            file_results.append(
                EngineFileResult(
                    backend=backend,
                    device=resolved_device,
                    path=str(path),
                    pages=n_pages,
                    load_seconds=file_load,
                    info_seconds=file_info,
                    raster_seconds=file_raster,
                    total_seconds=file_total,
                    encoded_bytes=file_encoded,
                    pages_per_second=pps,
                )
            )
            load_s += file_load
            info_s += file_info
            raster_s += file_raster
            encoded_bytes += file_encoded
            pages += n_pages
        except Exception as exc:
            failed += 1
            try:
                engine.close()
            except Exception:
                pass
            file_results.append(
                EngineFileResult(
                    backend=backend,
                    device=resolved_device,
                    path=str(path),
                    pages=0,
                    load_seconds=0.0,
                    info_seconds=0.0,
                    raster_seconds=0.0,
                    total_seconds=time.perf_counter() - t_file,
                    encoded_bytes=0,
                    pages_per_second=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    total_s = load_s + info_s + raster_s
    summary = EngineSummary(
        backend=backend,
        device=resolved_device,
        files=len(paths),
        pages=pages,
        load_seconds=load_s,
        info_seconds=info_s,
        raster_seconds=raster_s,
        total_seconds=total_s,
        encoded_bytes=encoded_bytes,
        pages_per_second=(pages / raster_s) if raster_s > 0 else 0.0,
        failed_files=failed,
    )
    return summary, file_results


def _format_seconds(value: float) -> str:
    return f"{value:.3f}"


def _print_summary_table(summaries: Iterable[EngineSummary], gpu_peak_bytes: Optional[int]) -> None:
    rows = list(summaries)
    typer.echo("")
    typer.echo("PDF engine comparison")
    typer.echo(
        f"{'backend':<8} {'device':<10} {'files':>6} {'pages':>6} "
        f"{'load_s':>8} {'info_s':>8} {'raster_s':>10} {'total_s':>8} "
        f"{'pages/s':>10} {'encoded_MiB':>12} {'failed':>7}"
    )
    for row in rows:
        typer.echo(
            f"{row.backend:<8} {row.device:<10} {row.files:>6d} {row.pages:>6d} "
            f"{_format_seconds(row.load_seconds):>8} {_format_seconds(row.info_seconds):>8} "
            f"{_format_seconds(row.raster_seconds):>10} {_format_seconds(row.total_seconds):>8} "
            f"{row.pages_per_second:>10.2f} {row.encoded_bytes / (1024 * 1024):>12.2f} "
            f"{row.failed_files:>7d}"
        )
    if len(rows) == 2 and rows[0].raster_seconds > 0 and rows[1].raster_seconds > 0:
        faster, slower = (rows[0], rows[1]) if rows[0].raster_seconds < rows[1].raster_seconds else (rows[1], rows[0])
        speedup = slower.raster_seconds / faster.raster_seconds
        typer.echo(
            f"Raster speedup: {faster.backend} is {speedup:.2f}x faster than {slower.backend} "
            f"({faster.pages_per_second:.2f} vs {slower.pages_per_second:.2f} pages/s)."
        )
    if gpu_peak_bytes is not None:
        typer.echo(f"Peak CUDA memory allocated: {gpu_peak_bytes / (1024 * 1024):.2f} MiB.")


@app.command("run")
def run(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory of PDF files to rasterize.",
    ),
    backends: str = typer.Option("cpu,gpu", "--backends", help="Comma-separated backends: cpu, gpu."),
    dpi: int = typer.Option(200, "--dpi", min=50, max=1200, help="Render DPI (capped by fit-to-model when enabled)."),
    render_mode: str = typer.Option(
        "fit_to_model",
        "--render-mode",
        help="full_dpi or fit_to_model (same modes as PDF extraction).",
    ),
    image_format: str = typer.Option("jpeg", "--image-format", help="jpeg or png."),
    jpeg_quality: int = typer.Option(100, "--jpeg-quality", min=1, max=100, help="JPEG quality."),
    recursive: bool = typer.Option(False, "--recursive/--no-recursive", help="Recurse into subdirectories."),
    warmup: bool = typer.Option(True, "--warmup/--no-warmup", help="Rasterize the first PDF once before timing."),
    keep_on_device: bool = typer.Option(
        True,
        "--keep-on-device/--no-keep-on-device",
        help="GPU engine retains CHW tensors for page-elements invoke.",
    ),
    device: Optional[str] = typer.Option(None, "--device", help="Torch device for the GPU engine, for example cuda:0."),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Write the comparison JSON here."),
    output_csv: Optional[Path] = typer.Option(None, "--output-csv", help="Write per-file CSV here."),
) -> None:
    if render_mode not in {"full_dpi", "fit_to_model"}:
        raise typer.BadParameter("--render-mode must be full_dpi or fit_to_model")
    if image_format not in {"jpeg", "png"}:
        raise typer.BadParameter("--image-format must be jpeg or png")

    pdf_paths = collect_pdf_paths(input_dir, recursive=recursive)
    selected = _parse_backends(backends)
    typer.echo(f"Benchmarking {len(pdf_paths)} PDF(s) from {input_dir} with backends={','.join(selected)}")

    summaries: List[EngineSummary] = []
    all_files: List[EngineFileResult] = []
    for backend in selected:
        try:
            summary, file_results = benchmark_pdf_engine(
                pdf_paths,
                backend=backend,
                dpi=int(dpi),
                render_mode=render_mode,  # type: ignore[arg-type]
                image_format=image_format,
                jpeg_quality=int(jpeg_quality),
                warmup=bool(warmup),
                keep_on_device=bool(keep_on_device),
                device=device,
            )
        except Exception as exc:
            typer.echo(f"{backend} engine failed to start: {type(exc).__name__}: {exc}", err=True)
            continue
        summaries.append(summary)
        all_files.extend(file_results)

    if not summaries:
        raise typer.Exit(code=1)

    gpu_peak = _gpu_memory_bytes() if any(row.backend == "gpu" for row in summaries) else None
    _print_summary_table(summaries, gpu_peak)

    if output_json is not None:
        payload = {
            "input_dir": str(input_dir.expanduser().resolve()),
            "dpi": int(dpi),
            "render_mode": render_mode,
            "image_format": image_format,
            "jpeg_quality": int(jpeg_quality),
            "keep_on_device": bool(keep_on_device),
            "gpu_peak_memory_bytes": gpu_peak,
            "summaries": [asdict(row) for row in summaries],
            "files": [asdict(row) for row in all_files],
        }
        output_json.expanduser().resolve().write_text(json.dumps(payload, indent=2) + "\n")
        typer.echo(f"Wrote JSON summary to {output_json}")

    if output_csv is not None:
        import csv

        csv_path = output_csv.expanduser().resolve()
        fieldnames = list(asdict(all_files[0]).keys()) if all_files else []
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_files:
                writer.writerow(asdict(row))
        typer.echo(f"Wrote per-file CSV to {csv_path}")
