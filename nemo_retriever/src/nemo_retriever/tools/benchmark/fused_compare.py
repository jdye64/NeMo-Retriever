# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end comparison of fused GPU-resident extraction against the staged pipeline.

Both paths run the same graph over the same PDFs with the same extraction
flags, differing only in ``ExtractParams.method``. The comparison is only
meaningful if three things hold, so this module enforces all three rather than
leaving them to the caller.

Weights are warmed before timing. Every local GPU actor resolves its model
through :mod:`nemo_retriever.models.warmup_registry`, and nothing else caches
models across graph builds, so an unwarmed run reloads every checkpoint on each
``ingest()``. Model load dominates a short corpus badly enough to invert the
result, so load time is measured once and reported on its own line.

The first pass after warmup is recorded separately. It still pays CUDA kernel
autotuning and allocator growth, so folding it into the steady-state numbers
would penalize whichever path ran first.

Output is compared, not just runtime. A path that emits fewer elements or skips
embeddings is not faster, so the report fails loudly when the two paths disagree
on row counts, element mix, or embedding coverage.
"""

from __future__ import annotations

import gc
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(help="Compare fused GPU-resident extraction against the staged local pipeline.")

FUSED_LABEL = "fused"
STAGED_LABEL = "staged"


@dataclass(frozen=True)
class OutputSummary:
    """What a path actually produced, used to reject unequal comparisons."""

    rows: int
    embeddings: int
    embedding_dim: Optional[int]
    rows_by_type: dict[str, int]


@dataclass
class PathResult:
    """Timings and output for one extraction path."""

    label: str
    method: str
    model_load_seconds: Optional[float] = None
    first_pass_seconds: Optional[float] = None
    iterations: list[float] = field(default_factory=list)
    peak_gpu_bytes: Optional[int] = None
    warmed_keys: list[str] = field(default_factory=list)
    absorbed_embed: bool = False
    summary: Optional[OutputSummary] = None
    error: Optional[str] = None

    @property
    def median_seconds(self) -> Optional[float]:
        return statistics.median(self.iterations) if self.iterations else None

    def pages_per_second(self, pages: int) -> Optional[float]:
        median = self.median_seconds
        if not median or pages <= 0:
            return None
        return pages / median


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    """Return the PDFs under *pdf_dir*, sorted so both paths see one order."""
    root = pdf_dir.expanduser().resolve()
    if not root.is_dir():
        raise typer.BadParameter(f"--pdf-dir must be a directory, got {root}")
    pdfs = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        raise typer.BadParameter(f"no PDF files found under {root}")
    return pdfs


def count_pages(pdfs: list[Path]) -> int:
    """Return the total page count, which is the unit throughput is reported in.

    Falls back to zero when pdfium cannot open a file, because a page count is
    only used to normalize timings and must never fail the benchmark.
    """
    import pypdfium2 as pdfium

    total = 0
    for path in pdfs:
        try:
            document = pdfium.PdfDocument(str(path))
        except Exception as exc:  # pragma: no cover - depends on corpus
            logger.warning("could not read page count from %s: %s", path.name, exc)
            continue
        try:
            total += len(document)
        finally:
            document.close()
    return total


def build_extract_params(*, method: str, use_table_structure: bool, dpi: int, **flags: Any) -> Any:
    """Build extract params that differ between paths only by *method*."""
    from nemo_retriever.common.params import ExtractParams

    return ExtractParams(
        method=method,
        use_page_elements=True,
        use_table_structure=use_table_structure,
        extract_page_as_image=True,
        dpi=dpi,
        **flags,
    )


def build_embed_params(*, model_name: str, granularity: str) -> Any:
    """Build embed params pinned to a local model.

    ``model_name`` is set explicitly because :func:`build_warmup_spec` only
    emits an embed warmup entry when a model name is present. Leaving it unset
    resolves the same default at actor construction but skips warmup, which
    would charge the staged path a model load on every iteration.
    """
    from nemo_retriever.common.params import EmbedParams

    return EmbedParams(model_name=model_name, embed_granularity=granularity)


def summarize_output(frame: Any, embed_params: Any) -> OutputSummary:
    """Describe *frame* so two paths can be checked for equivalent work."""
    rows = int(len(frame))
    output_column = getattr(embed_params, "output_column", "text_embeddings_1b_v2")
    has_embedding_column = getattr(embed_params, "has_embedding_column", f"{output_column}_has_embedding")
    dim_column = getattr(embed_params, "embedding_dim_column", f"{output_column}_dim")

    embeddings = 0
    if has_embedding_column in frame.columns:
        embeddings = int(frame[has_embedding_column].fillna(False).astype(bool).sum())
    elif "metadata" in frame.columns:
        embeddings = int(
            sum(1 for value in frame["metadata"] if isinstance(value, dict) and value.get("embedding") is not None)
        )

    embedding_dim: Optional[int] = None
    if dim_column in frame.columns:
        dims = {int(value) for value in frame[dim_column].fillna(0).tolist() if int(value) > 0}
        if len(dims) == 1:
            embedding_dim = dims.pop()
        elif dims:
            logger.warning("mixed embedding dimensions in output: %s", sorted(dims))

    rows_by_type: dict[str, int] = {}
    for column in ("document_type", "element_type"):
        if column in frame.columns:
            counts = frame[column].astype(str).value_counts().to_dict()
            rows_by_type = {str(key): int(value) for key, value in counts.items()}
            break

    return OutputSummary(rows=rows, embeddings=embeddings, embedding_dim=embedding_dim, rows_by_type=rows_by_type)


def _reset_peak_gpu_memory() -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is present on GPU hosts
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu_memory() -> Optional[int]:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is present on GPU hosts
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def _release_gpu_memory() -> None:
    """Drop warmed weights so the next path starts from a clean device."""
    from nemo_retriever.models.warmup_registry import clear_warmed_models

    clear_warmed_models()
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is present on GPU hosts
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def warm_path(extract_params: Any, embed_params: Any, *, absorbs_embed: bool) -> tuple[float, list[str]]:
    """Warm the models this path needs and return the load time and warmed keys.

    ``embed_params`` is withheld from the spec when the fused stage absorbs
    embedding, because the graph drops the separate embed actor in that case and
    warming an embedder nobody calls would inflate this path's reported VRAM.
    """
    from nemo_retriever.models.warmup_registry import build_warmup_spec, warm_local_models, warmed_model_keys

    spec = build_warmup_spec(
        extract_params.model_dump(),
        None if absorbs_embed else embed_params.model_dump(),
        None,
    )
    if spec is None:
        return 0.0, []

    start = time.perf_counter()
    warm_local_models(spec)
    elapsed = time.perf_counter() - start
    return elapsed, warmed_model_keys()


def run_once(
    documents: list[str],
    extract_params: Any,
    embed_params: Any,
    *,
    run_mode: str,
    error_policy: str,
) -> tuple[float, Any]:
    """Time one full ingest and return the elapsed seconds with the output frame."""
    from nemo_retriever.ingestor.graph_ingestor import GraphIngestor

    ingestor = GraphIngestor(
        run_mode=run_mode,
        show_progress=False,
        error_policy=error_policy,
    )
    ingestor = ingestor.files(documents).extract(extract_params).embed(embed_params)

    start = time.perf_counter()
    frame = ingestor.ingest()
    elapsed = time.perf_counter() - start
    return elapsed, frame


def run_path(
    label: str,
    method: str,
    documents: list[str],
    *,
    iterations: int,
    run_mode: str,
    error_policy: str,
    use_table_structure: bool,
    dpi: int,
    embed_model: str,
    granularity: str,
    extract_flags: dict[str, Any],
) -> PathResult:
    """Warm, measure, and summarize one extraction path."""
    from nemo_retriever.graph.ingestor_runtime import fused_absorbs_embed

    result = PathResult(label=label, method=method)
    extract_params = build_extract_params(
        method=method,
        use_table_structure=use_table_structure,
        dpi=dpi,
        **extract_flags,
    )
    embed_params = build_embed_params(model_name=embed_model, granularity=granularity)
    result.absorbed_embed = bool(fused_absorbs_embed(extract_params, embed_params))

    try:
        load_seconds, warmed = warm_path(extract_params, embed_params, absorbs_embed=result.absorbed_embed)
        result.model_load_seconds = load_seconds
        result.warmed_keys = warmed
        typer.echo(f"[{label}] warmed {warmed or 'nothing'} in {load_seconds:.1f}s")

        _reset_peak_gpu_memory()

        first, frame = run_once(documents, extract_params, embed_params, run_mode=run_mode, error_policy=error_policy)
        result.first_pass_seconds = first
        result.summary = summarize_output(frame, embed_params)
        typer.echo(f"[{label}] first pass {first:.2f}s rows={result.summary.rows}")
        del frame

        for index in range(iterations):
            elapsed, frame = run_once(
                documents, extract_params, embed_params, run_mode=run_mode, error_policy=error_policy
            )
            result.iterations.append(elapsed)
            typer.echo(f"[{label}] iteration {index + 1}/{iterations} {elapsed:.2f}s")
            del frame

        result.peak_gpu_bytes = _peak_gpu_memory()
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        logger.exception("%s path failed", label)
        typer.echo(f"[{label}] FAILED {result.error}")
    finally:
        _release_gpu_memory()

    return result


def _format_seconds(value: Optional[float]) -> str:
    return f"{value:.2f}s" if value is not None else "n/a"


def _format_gib(value: Optional[int]) -> str:
    return f"{value / (1024 ** 3):.2f} GiB" if value else "n/a"


def parity_problems(fused: PathResult, staged: PathResult) -> list[str]:
    """Return the reasons the two runs are not comparable.

    A runtime win only counts if both paths did the same work, so row counts,
    embedding coverage, and element mix are all checked.
    """
    if fused.summary is None or staged.summary is None:
        return ["one path produced no output, so runtimes are not comparable"]

    problems: list[str] = []
    if fused.summary.rows != staged.summary.rows:
        problems.append(f"row count differs: fused={fused.summary.rows} staged={staged.summary.rows}")
    if fused.summary.embeddings != staged.summary.embeddings:
        problems.append(f"embedding count differs: fused={fused.summary.embeddings} staged={staged.summary.embeddings}")
    if fused.summary.embedding_dim != staged.summary.embedding_dim:
        problems.append(
            f"embedding dimension differs: fused={fused.summary.embedding_dim} "
            f"staged={staged.summary.embedding_dim}"
        )
    if fused.summary.rows_by_type != staged.summary.rows_by_type:
        problems.append(f"element mix differs: fused={fused.summary.rows_by_type} staged={staged.summary.rows_by_type}")
    return problems


def render_report(
    results: list[PathResult],
    *,
    documents: int,
    pages: int,
    run_mode: str,
    granularity: str,
    iterations: int,
) -> str:
    """Render the comparison as a fixed-width report."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("Fused vs staged extraction")
    lines.append("=" * 78)
    lines.append(f"corpus        : {documents} document(s), {pages} page(s)")
    lines.append(f"run mode      : {run_mode}")
    lines.append(f"embedding     : {granularity} granularity")
    lines.append(f"iterations    : {iterations} timed pass(es) per path, after one warmup pass")
    lines.append("")

    header = f"{'path':<8} {'median':>9} {'min':>9} {'max':>9} {'pages/s':>9} {'load':>9} {'first':>9} {'peak GPU':>11}"
    lines.append(header)
    lines.append("-" * len(header))
    for result in results:
        if result.error:
            lines.append(f"{result.label:<8} FAILED  {result.error}")
            continue
        times = result.iterations
        lines.append(
            f"{result.label:<8} "
            f"{_format_seconds(result.median_seconds):>9} "
            f"{_format_seconds(min(times) if times else None):>9} "
            f"{_format_seconds(max(times) if times else None):>9} "
            f"{(f'{result.pages_per_second(pages):.2f}' if result.pages_per_second(pages) else 'n/a'):>9} "
            f"{_format_seconds(result.model_load_seconds):>9} "
            f"{_format_seconds(result.first_pass_seconds):>9} "
            f"{_format_gib(result.peak_gpu_bytes):>11}"
        )
    lines.append("")

    by_label = {result.label: result for result in results}
    fused = by_label.get(FUSED_LABEL)
    staged = by_label.get(STAGED_LABEL)

    if fused and staged and fused.median_seconds and staged.median_seconds:
        speedup = staged.median_seconds / fused.median_seconds
        verdict = "faster" if speedup > 1.0 else "slower"
        lines.append(f"fused is {speedup:.2f}x {verdict} than staged in steady state")
        if fused.peak_gpu_bytes and staged.peak_gpu_bytes:
            ratio = staged.peak_gpu_bytes / fused.peak_gpu_bytes
            lines.append(f"fused peak GPU memory is {ratio:.2f}x that of staged (higher than 1.0 favors fused)")
        lines.append(f"fused absorbed embedding into the extract stage: {fused.absorbed_embed}")

        for result in (fused, staged):
            if len(result.iterations) >= 2:
                spread = max(result.iterations) - min(result.iterations)
                relative = spread / result.median_seconds if result.median_seconds else 0.0
                if relative > 0.1:
                    lines.append(
                        f"warning: {result.label} varied by {relative * 100:.0f}% across iterations, "
                        f"so the median is not yet stable; raise --iterations"
                    )

        lines.append("")
        lines.append("Output parity")
        lines.append("-" * 78)
        problems = parity_problems(fused, staged)
        if problems:
            lines.append("NOT COMPARABLE. The paths did different amounts of work:")
            lines.extend(f"  - {problem}" for problem in problems)
        else:
            summary = fused.summary
            assert summary is not None
            lines.append(
                f"both paths agree: {summary.rows} rows, {summary.embeddings} embeddings, "
                f"dim={summary.embedding_dim}"
            )
            if summary.rows_by_type:
                mix = ", ".join(f"{key}={value}" for key, value in sorted(summary.rows_by_type.items()))
                lines.append(f"element mix: {mix}")
    elif fused and staged:
        lines.append("no speedup computed because at least one path did not complete a timed pass")

    lines.append("=" * 78)
    return "\n".join(lines)


def results_payload(
    results: list[PathResult],
    *,
    documents: int,
    pages: int,
    run_mode: str,
    granularity: str,
) -> dict[str, Any]:
    """Build the JSON payload for ``--output-json``."""
    by_label = {result.label: result for result in results}
    fused = by_label.get(FUSED_LABEL)
    staged = by_label.get(STAGED_LABEL)

    speedup: Optional[float] = None
    if fused and staged and fused.median_seconds and staged.median_seconds:
        speedup = staged.median_seconds / fused.median_seconds

    payload: dict[str, Any] = {
        "corpus": {"documents": documents, "pages": pages},
        "run_mode": run_mode,
        "embed_granularity": granularity,
        "speedup_fused_over_staged": speedup,
        "paths": [],
    }
    for result in results:
        entry: dict[str, Any] = {
            "label": result.label,
            "method": result.method,
            "model_load_seconds": result.model_load_seconds,
            "first_pass_seconds": result.first_pass_seconds,
            "iterations_seconds": result.iterations,
            "median_seconds": result.median_seconds,
            "pages_per_second": result.pages_per_second(pages),
            "peak_gpu_bytes": result.peak_gpu_bytes,
            "warmed_models": result.warmed_keys,
            "absorbed_embed": result.absorbed_embed,
            "error": result.error,
        }
        if result.summary is not None:
            entry["output"] = {
                "rows": result.summary.rows,
                "embeddings": result.summary.embeddings,
                "embedding_dim": result.summary.embedding_dim,
                "rows_by_type": result.summary.rows_by_type,
            }
        payload["paths"].append(entry)

    if fused and staged:
        payload["parity_problems"] = parity_problems(fused, staged)
    return payload


@app.command("run")
def run(
    pdf_dir: Path = typer.Option(..., "--pdf-dir", help="Directory of PDFs to ingest, searched recursively."),
    iterations: int = typer.Option(3, "--iterations", min=1, help="Timed passes per path after the warmup pass."),
    run_mode: str = typer.Option("inprocess", "--run-mode", help="Either 'inprocess' or 'batch'."),
    embed_granularity: str = typer.Option(
        "page",
        "--embed-granularity",
        help="Either 'page' or 'element'. Fused only absorbs embedding at page granularity.",
    ),
    staged_method: str = typer.Option(
        "pdfium_hybrid", "--staged-method", help="Extraction method for the staged baseline."
    ),
    embed_model: str = typer.Option(
        "nvidia/llama-nemotron-embed-vl-1b-v2", "--embed-model", help="Local embedding model for both paths."
    ),
    extract_tables: bool = typer.Option(True, "--extract-tables/--no-extract-tables", help="Extract tables."),
    extract_charts: bool = typer.Option(True, "--extract-charts/--no-extract-charts", help="Extract charts."),
    extract_infographics: bool = typer.Option(
        False, "--extract-infographics/--no-extract-infographics", help="Extract infographics."
    ),
    use_table_structure: bool = typer.Option(
        True, "--use-table-structure/--no-use-table-structure", help="Run table structure on detected tables."
    ),
    dpi: int = typer.Option(200, "--dpi", min=72, help="Page render DPI for both paths."),
    only: Optional[str] = typer.Option(None, "--only", help="Run just one path: 'fused' or 'staged'."),
    error_policy: str = typer.Option("raise", "--error-policy", help="Either 'raise' or 'collect'."),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Optional JSON summary path."),
) -> None:
    """Benchmark fused against staged extraction over a directory of PDFs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if run_mode not in {"inprocess", "batch"}:
        raise typer.BadParameter("--run-mode must be 'inprocess' or 'batch'")
    if embed_granularity not in {"page", "element"}:
        raise typer.BadParameter("--embed-granularity must be 'page' or 'element'")
    if only is not None and only not in {FUSED_LABEL, STAGED_LABEL}:
        raise typer.BadParameter("--only must be 'fused' or 'staged'")

    if run_mode == "batch":
        typer.echo(
            "warning: Ray actors are separate processes, so driver-side warmup does not reach them. "
            "In batch mode every timed pass reloads model weights, which measures cold start rather "
            "than steady state. Use --run-mode inprocess to compare compute."
        )
    if embed_granularity == "element":
        typer.echo(
            "note: at element granularity the fused stage does not absorb embedding, because element "
            "rows do not exist yet when it runs. Both paths will use the separate embed actor."
        )

    pdfs = discover_pdfs(pdf_dir)
    documents = [str(path) for path in pdfs]
    pages = count_pages(pdfs)
    typer.echo(f"benchmarking {len(documents)} document(s), {pages} page(s)")

    extract_flags = {
        "extract_tables": extract_tables,
        "extract_charts": extract_charts,
        "extract_infographics": extract_infographics,
    }

    plan = [(FUSED_LABEL, FUSED_LABEL), (STAGED_LABEL, staged_method)]
    if only is not None:
        plan = [item for item in plan if item[0] == only]

    results: list[PathResult] = []
    for label, method in plan:
        typer.echo("")
        typer.echo(f"--- {label} (method={method}) ---")
        results.append(
            run_path(
                label,
                method,
                documents,
                iterations=iterations,
                run_mode=run_mode,
                error_policy=error_policy,
                use_table_structure=use_table_structure,
                dpi=dpi,
                embed_model=embed_model,
                granularity=embed_granularity,
                extract_flags=extract_flags,
            )
        )

    typer.echo(
        render_report(
            results,
            documents=len(documents),
            pages=pages,
            run_mode=run_mode,
            granularity=embed_granularity,
            iterations=iterations,
        )
    )

    if output_json is not None:
        payload = results_payload(
            results,
            documents=len(documents),
            pages=pages,
            run_mode=run_mode,
            granularity=embed_granularity,
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"wrote {output_json}")

    if any(result.error for result in results):
        raise typer.Exit(code=1)
