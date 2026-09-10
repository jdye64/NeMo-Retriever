# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from gpu_pdf_render.pdf import load_pdf
from gpu_pdf_render.renderer import GpuPdfRenderer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="GPU full-document PDF rasterizer (research).")


@app.command()
def info(pdf: Path) -> None:
    """Print page count and media boxes for a PDF."""
    pages = load_pdf(pdf)
    typer.echo(f"pages={len(pages)}")
    for page in pages:
        typer.echo(f"{page.index}: {page.width_pt:.2f}x{page.height_pt:.2f} pt")


@app.command()
def render(
    pdf: Path,
    out: Path = typer.Option(..., "--out", "-o", help="Output directory for page images."),
    image_format: str = typer.Option("png", "--format", "-f", help="png or jpeg"),
    dpi: float = typer.Option(144.0, "--dpi", help="Raster resolution."),
    backend: str = typer.Option("cuda", "--backend", help="cuda or reference"),
    quality: int = typer.Option(90, "--quality", help="JPEG quality 1-100."),
    page: Optional[int] = typer.Option(None, "--page", help="Optional 1-based page to write; default is all pages."),
) -> None:
    """Render every page of a PDF to PNG or JPEG."""
    renderer = GpuPdfRenderer(dpi=dpi, backend=backend)
    document = renderer.render_document(pdf)
    if page is None:
        paths = document.save(out, image_format=image_format, quality=quality)
        typer.echo(f"wrote {len(paths)} pages to {out}")
        return
    index = page - 1
    out.mkdir(parents=True, exist_ok=True)
    fmt = image_format.lower()
    if fmt in {"jpg", "jpeg"}:
        data = document.to_jpeg(index, quality=quality)
        path = out / f"page-{page}.jpg"
    else:
        data = document.to_png(index)
        path = out / f"page-{page}.png"
    path.write_bytes(data)
    typer.echo(str(path))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
