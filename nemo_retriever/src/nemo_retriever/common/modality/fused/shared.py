# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused extraction: page elements, table structure, OCR, and embed in one call.

The staged pipeline hands page images between four actors as base64 in a
DataFrame column, so each stage decodes the same page again and copies it
across PCIe again. This module drives ``nemo_retriever_fused``, which decodes
each page once into device memory and lets every stage read the same resident
tensor, then writes the identical columns the four stages would have written.

The column contract is deliberately unchanged: downstream stages cannot tell
whether ``table``, ``chart``, ``infographic``, ``text``, ``page_elements_v3``,
and ``table_structure_v1`` came from four actors or from one.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Sequence

import pandas as pd

_logger = logging.getLogger(__name__)

FUSED_IMPORT_HINT = (
    "method='fused' requires the `nemo_retriever_fused` package, which is not installed. "
    "Install it from the in-repo project: `pip install ./uber_model/nemo-retriever`."
)


def _record_trace(event: str, duration_s: float, extra: Dict[str, Any] | None = None) -> None:
    """Append one JSON record per fused batch when tracing is enabled.

    Follows the same opt-in shape as the LanceDB timing hook: point
    ``NV_INGEST_FUSED_TRACE_PATH`` at a file and each batch appends one JSON
    object, so a run that dies partway still leaves the batches it finished.
    """
    trace_path = os.getenv("NV_INGEST_FUSED_TRACE_PATH")
    if not trace_path:
        return
    payload = {
        "event": event,
        "duration_s": duration_s,
        "timestamp_s": time.time(),
    }
    if extra:
        payload.update(extra)
    trace_dir = os.path.dirname(trace_path)
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
    with open(trace_path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    """Return the error shape the staged operators write on failure."""
    return {
        "detections": [],
        "error": {
            "stage": str(stage),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


def _counts_by_label(detections: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for detection in detections:
        label = str(detection.get("label_name") or "")
        if label:
            counts[label] = counts.get(label, 0) + 1
    return counts


def build_fused_config(*, ocr_version: str = "v2", ocr_lang: str | None = None) -> Any:
    """Return a fused pipeline config carrying the OCR selectors.

    ``ExtractParams`` exposes ``ocr_version`` and ``ocr_lang`` for the staged
    OCR actor, and the fused OCR stage accepts the same selectors, so they must
    be forwarded rather than silently defaulted. The v1 pipeline lives in its
    own repository, so the repo id moves with the version.
    """
    from nemo_retriever_fused.config import FusedPipelineConfig

    config = FusedPipelineConfig()
    config.ocr.version = ocr_version
    if ocr_version == "v1":
        config.ocr.repo_id = "nvidia/nemotron-ocr-v1"
    elif ocr_lang is not None:
        config.ocr.lang = ocr_lang
    return config


def load_fused_model(*, ocr_version: str = "v2", ocr_lang: str | None = None) -> Any:
    """Return a loaded fused model, preferring one warmed into this process.

    Raises
    ------
    ImportError
        When ``nemo_retriever_fused`` is not installed. The message names the
        install command rather than the missing symbol, because the package is
        an optional extra rather than a hard dependency.
    """
    from nemo_retriever.models.warmup_registry import get_warmed_model

    warmed = get_warmed_model("fused")
    if warmed is not None:
        return warmed

    try:
        from nemo_retriever_fused import NemoRetrieverFusedModel
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(FUSED_IMPORT_HINT) from exc

    return NemoRetrieverFusedModel.from_pretrained(build_fused_config(ocr_version=ocr_version, ocr_lang=ocr_lang))


def _page_image_b64(row: Any) -> str | None:
    """Return the page raster from the canonical ``page_image`` column."""
    page_image = getattr(row, "page_image", None)
    if not isinstance(page_image, dict):
        return None
    b64 = page_image.get("image_b64")
    return b64 if isinstance(b64, str) and b64 else None


def _element_entry(element: Any) -> Dict[str, Any]:
    """Return the ``{bbox, text}`` entry the OCR stage writes for content."""
    text = element.table_markdown or element.text or ""
    return {"bbox_xyxy_norm": list(element.bbox_xyxy_norm), "text": text}


def _detection_entry(element: Any) -> Dict[str, Any]:
    """Return the detection entry the page-elements stage writes."""
    return {
        "bbox_xyxy_norm": list(element.bbox_xyxy_norm),
        "label_name": element.label,
        "score": float(element.score),
    }


def fused_extract_pages(
    pages_df: Any,
    *,
    model: Any = None,
    extract_text: bool = True,
    extract_tables: bool = True,
    extract_charts: bool = True,
    extract_infographics: bool = False,
    use_table_structure: bool = True,
    embed_pages: bool = False,
    embed_output_column: str = "text_embeddings_1b_v2",
    embedding_dim_column: str = "text_embeddings_1b_v2_dim",
    has_embedding_column: str = "text_embeddings_1b_v2_has_embedding",
    **_ignored: Any,
) -> Any:
    """Run every fused stage over *pages_df* and write the staged columns.

    Parameters
    ----------
    pages_df
        One row per page, as produced by ``PDFExtractionActor``. The page
        raster is read from ``page_image["image_b64"]``.
    model
        A loaded fused model. When ``None`` one is loaded on first use.
    embed_pages
        When true, attach page-level embeddings so the separate batch embed
        stage can be skipped. Element-granularity embedding stays with the
        dedicated embed stage, which reshapes rows before embedding.

    Returns
    -------
    pandas.DataFrame
        *pages_df* with the page-elements, table-structure, OCR, and (when
        ``embed_pages``) embedding columns added.
    """
    if not isinstance(pages_df, pd.DataFrame) or pages_df.empty:
        return pages_df

    row_count = len(pages_df.index)
    detections_payloads: List[Dict[str, Any]] = [
        {"detections": [], "timing": None, "error": None} for _ in range(row_count)
    ]
    table_items: List[List[Dict[str, Any]]] = [[] for _ in range(row_count)]
    chart_items: List[List[Dict[str, Any]]] = [[] for _ in range(row_count)]
    infographic_items: List[List[Dict[str, Any]]] = [[] for _ in range(row_count)]
    page_texts: List[str | None] = [None] * row_count
    embeddings: List[List[float] | None] = [None] * row_count

    # Only pages with a raster reach the model; the rest keep their empty
    # payloads so the frame stays rectangular.
    pending_rows: List[int] = []
    pending_b64: List[str] = []
    pending_ids: List[str] = []
    for row_index, row in enumerate(pages_df.itertuples(index=False)):
        b64 = _page_image_b64(row)
        if b64 is None:
            continue
        pending_rows.append(row_index)
        pending_b64.append(b64)
        pending_ids.append(f"{getattr(row, 'source_id', None)}:{getattr(row, 'page_number', row_index)}")

    if not pending_rows:
        _logger.debug("Fused extraction skipped a batch of %d pages — no page rasters present", row_count)
        return _assign_columns(
            pages_df,
            detections_payloads=detections_payloads,
            table_items=table_items,
            chart_items=chart_items,
            infographic_items=infographic_items,
            page_texts=page_texts,
            embeddings=embeddings,
            extract_text=extract_text,
            extract_tables=extract_tables,
            extract_charts=extract_charts,
            extract_infographics=extract_infographics,
            use_table_structure=use_table_structure,
            embed_pages=embed_pages,
            embed_output_column=embed_output_column,
            embedding_dim_column=embedding_dim_column,
            has_embedding_column=has_embedding_column,
        )

    started = time.perf_counter()
    try:
        if model is None:
            model = load_fused_model()
        # Skip the embed stage inside the model when a downstream embed stage
        # owns embedding, so element granularity does not pay for it twice.
        result = model(pending_b64, page_ids=pending_ids, embed_pages=embed_pages)
    except Exception as exc:
        _logger.warning(
            "Fused extraction failed for %d pages: %s: %s",
            len(pending_rows),
            exc.__class__.__name__,
            exc,
        )
        _record_trace(
            "fused.extract_failed",
            time.perf_counter() - started,
            {
                "pages": len(pending_rows),
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )
        payload = _error_payload(stage="fused_invoke", exc=exc)
        for row_index in pending_rows:
            detections_payloads[row_index] = dict(payload)
        return _assign_columns(
            pages_df,
            detections_payloads=detections_payloads,
            table_items=table_items,
            chart_items=chart_items,
            infographic_items=infographic_items,
            page_texts=page_texts,
            embeddings=embeddings,
            extract_text=extract_text,
            extract_tables=extract_tables,
            extract_charts=extract_charts,
            extract_infographics=extract_infographics,
            use_table_structure=use_table_structure,
            embed_pages=embed_pages,
            embed_output_column=embed_output_column,
            embedding_dim_column=embedding_dim_column,
            has_embedding_column=has_embedding_column,
        )

    elapsed = time.perf_counter() - started
    page_embeddings = getattr(result, "page_embeddings", None)
    if embed_pages and page_embeddings is not None:
        # The single device-to-host copy for vectors in the fused path.
        host_embeddings = page_embeddings.detach().to("cpu").tolist()
    else:
        host_embeddings = None

    for offset, row_index in enumerate(pending_rows):
        page = result.pages[offset]
        detections_payloads[row_index] = {
            "detections": [_detection_entry(element) for element in page.elements],
            "timing": {"seconds": float(elapsed / len(pending_rows))},
            "error": None,
        }
        table_items[row_index] = [_element_entry(e) for e in page.elements if e.label == "table"]
        chart_items[row_index] = [_element_entry(e) for e in page.elements if e.label == "chart"]
        infographic_items[row_index] = [_element_entry(e) for e in page.elements if e.label == "infographic"]
        page_texts[row_index] = page.text
        if host_embeddings is not None:
            embeddings[row_index] = host_embeddings[offset]

    _logger.info(
        "Fused extraction completed %d pages in %.3fs (%.1f ms/page)",
        len(pending_rows),
        elapsed,
        1000.0 * elapsed / len(pending_rows),
    )

    # The model records per-stage device time with CUDA events, which is the
    # only per-stage breakdown available without attaching a profiler.
    stage_timings = getattr(result, "timings", None) or ()
    if stage_timings:
        _logger.info(
            "Fused stage device time: %s",
            ", ".join(f"{timing.name}={timing.milliseconds:.1f}ms" for timing in stage_timings),
        )

    _record_trace(
        "fused.extract",
        elapsed,
        {
            "pages": len(pending_rows),
            "ms_per_page": 1000.0 * elapsed / len(pending_rows),
            "embed_pages": bool(embed_pages),
            "stages_ms": {timing.name: float(timing.milliseconds) for timing in stage_timings},
        },
    )

    return _assign_columns(
        pages_df,
        detections_payloads=detections_payloads,
        table_items=table_items,
        chart_items=chart_items,
        infographic_items=infographic_items,
        page_texts=page_texts,
        embeddings=embeddings,
        extract_text=extract_text,
        extract_tables=extract_tables,
        extract_charts=extract_charts,
        extract_infographics=extract_infographics,
        use_table_structure=use_table_structure,
        embed_pages=embed_pages,
        embed_output_column=embed_output_column,
        embedding_dim_column=embedding_dim_column,
        has_embedding_column=has_embedding_column,
    )


def _assign_columns(
    pages_df: pd.DataFrame,
    *,
    detections_payloads: List[Dict[str, Any]],
    table_items: List[List[Dict[str, Any]]],
    chart_items: List[List[Dict[str, Any]]],
    infographic_items: List[List[Dict[str, Any]]],
    page_texts: List[str | None],
    embeddings: List[List[float] | None],
    extract_text: bool,
    extract_tables: bool,
    extract_charts: bool,
    extract_infographics: bool,
    use_table_structure: bool,
    embed_pages: bool,
    embed_output_column: str,
    embedding_dim_column: str,
    has_embedding_column: str,
) -> pd.DataFrame:
    """Write the staged pipeline's columns onto a copy of *pages_df*."""
    out = pages_df.copy()

    out["page_elements_v3"] = detections_payloads
    out["page_elements_v3_num_detections"] = [len(p.get("detections") or []) for p in detections_payloads]
    out["page_elements_v3_counts_by_label"] = [_counts_by_label(p.get("detections") or []) for p in detections_payloads]

    if extract_tables or "table" not in out.columns:
        out["table"] = table_items
    if extract_charts or "chart" not in out.columns:
        out["chart"] = chart_items
    if extract_infographics or "infographic" not in out.columns:
        out["infographic"] = infographic_items

    if extract_text:
        if "text" in out.columns:
            text_position = out.columns.get_loc("text")
            for row_index, text in enumerate(page_texts):
                if text is not None:
                    out.iat[row_index, text_position] = text
        else:
            out["text"] = [text if text is not None else "" for text in page_texts]

    # The fused model reconstructs tables inline, so the structure payload
    # carries the same regions the staged table actor would have published.
    if use_table_structure:
        out["table_structure_v1"] = [
            {
                "regions": [
                    {"bbox_xyxy_norm": entry["bbox_xyxy_norm"], "label_name": "table", "detections": []}
                    for entry in items
                ],
                "timing": None,
                "error": None,
            }
            for items in table_items
        ]
        out["table_structure_v1_num_detections"] = [len(items) for items in table_items]
        out["table_structure_v1_counts_by_label"] = [{"table": len(items)} if items else {} for items in table_items]

    out["ocr"] = [
        {
            "timing": payload.get("timing"),
            "error": payload.get("error"),
            "num_detections": len(payload.get("detections") or []),
            "counts_by_label": _counts_by_label(payload.get("detections") or []),
        }
        for payload in detections_payloads
    ]
    out["ocr_v1_num_detections"] = [meta["num_detections"] for meta in out["ocr"]]
    out["ocr_v1_counts_by_label"] = [meta["counts_by_label"] for meta in out["ocr"]]

    if embed_pages:
        out = _assign_embeddings(
            out,
            embeddings=embeddings,
            embed_output_column=embed_output_column,
            embedding_dim_column=embedding_dim_column,
            has_embedding_column=has_embedding_column,
        )

    return out


def _assign_embeddings(
    out: pd.DataFrame,
    *,
    embeddings: List[List[float] | None],
    embed_output_column: str,
    embedding_dim_column: str,
    has_embedding_column: str,
) -> pd.DataFrame:
    """Write embeddings in the shape the batch embed stage publishes."""
    out[embed_output_column] = [
        {"embedding": vector, "info_msg": None} if vector is not None else None for vector in embeddings
    ]
    out[embedding_dim_column] = [len(vector) if vector else 0 for vector in embeddings]
    out[has_embedding_column] = [bool(vector) for vector in embeddings]
    out["embedding_v1_num_detections"] = [int(bool(vector)) for vector in embeddings]
    out["embedding_v1_counts_by_label"] = [{"embedded": 1} if vector else {} for vector in embeddings]
    out["_contains_embeddings"] = [bool(vector) for vector in embeddings]

    # Downstream vector-DB upload reads the vector from row metadata.
    if "metadata" in out.columns:
        metadata_position = out.columns.get_loc("metadata")
        for row_index, vector in enumerate(embeddings):
            existing = out.iat[row_index, metadata_position]
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged["embedding"] = vector
            out.iat[row_index, metadata_position] = merged

    return out
