# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join table structure detections with OCR text into markdown.

This is the fused counterpart to
`nemo_retriever.common.modality.table_and_chart.join_table_structure_and_ocr_output`.
Table reconstruction is inherently a host-side string operation, so this module
runs on the host by design. What it avoids is the production path's habit of
re-deriving geometry from separately transferred tensors: the caller hands it
detections and OCR regions that were produced from the same resident crop, in
the same normalized coordinate space, so no rescaling or re-cropping is needed.

Assignment uses the row and column boxes as an implicit grid and places each OCR
region into the cell whose row and column bands contain its centre. That matches
the production behaviour for well-formed tables without pulling in the DBSCAN
reading-order pass, which only affects tables where structure detection failed
to produce usable rows or columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from nemo_retriever_fused.stages.detectors import Detections
from nemo_retriever_fused.stages.ocr import OcrResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Band:
    """A row or column band in normalized crop coordinates."""

    start: float
    end: float
    score: float

    @property
    def centre(self) -> float:
        """Return the midpoint of the band."""
        return (self.start + self.end) / 2.0

    def contains(self, position: float) -> bool:
        """Return whether *position* falls inside the band."""
        return self.start <= position <= self.end


def _bands(detections: Detections, label_names: tuple[str, ...], wanted: str, axis: int) -> list[_Band]:
    """Extract sorted row or column bands for the *wanted* class.

    Parameters
    ----------
    detections:
        Table structure detections in normalized crop coordinates.
    label_names:
        Class names indexed by class id.
    wanted:
        ``"row"`` or ``"column"``.
    axis:
        0 to read the x extent (columns), 1 to read the y extent (rows).
    """
    if not len(detections):
        return []

    try:
        wanted_id = label_names.index(wanted)
    except ValueError:
        return []

    # One transfer for the whole detection set rather than per-box reads.
    boxes = detections.boxes.tolist()
    scores = detections.scores.tolist()
    labels = detections.labels.tolist()

    bands = [
        _Band(start=box[axis], end=box[axis + 2], score=score)
        for box, score, label in zip(boxes, scores, labels)
        if int(label) == wanted_id
    ]
    bands.sort(key=lambda band: band.start)
    return bands


def _index_for(bands: list[_Band], position: float) -> int | None:
    """Return the index of the band containing *position*, or the nearest one."""
    if not bands:
        return None
    for index, band in enumerate(bands):
        if band.contains(position):
            return index
    # Fall back to the nearest band centre so text outside every detected band
    # still lands somewhere rather than being dropped.
    return min(range(len(bands)), key=lambda index: abs(bands[index].centre - position))


def _escape_cell(text: str) -> str:
    """Escape pipes and collapse newlines so a cell stays on one markdown row."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def assemble_table_markdown(
    structure: Detections,
    ocr: OcrResult,
    *,
    label_names: tuple[str, ...],
    header_first_row: bool = True,
) -> str:
    """Build a markdown table from structure detections and OCR regions.

    Parameters
    ----------
    structure:
        Row, column, and cell detections for one table crop, in normalized crop
        coordinates.
    ocr:
        OCR regions for the same crop, in the same normalized space.
    label_names:
        Table structure class names indexed by class id.
    header_first_row:
        Emit the first row as a markdown header with a separator beneath it.

    Returns
    -------
    str
        A markdown table, or the OCR text joined by newlines when structure
        detection produced no usable rows or columns.
    """
    rows = _bands(structure, label_names, "row", axis=1)
    columns = _bands(structure, label_names, "column", axis=0)

    if not rows or not columns:
        logger.debug(
            "table structure produced %d rows and %d columns; falling back to raw text",
            len(rows),
            len(columns),
        )
        return ocr.text

    grid: list[list[list[str]]] = [[[] for _ in columns] for _ in rows]

    for region in ocr.regions:
        if not region.text:
            continue
        # `upper` is the maximum y and `lower` the minimum, matching the
        # production pipeline's field naming, so the vertical centre is their
        # mean either way.
        centre_x = (region.left + region.right) / 2.0
        centre_y = (region.upper + region.lower) / 2.0

        row_index = _index_for(rows, centre_y)
        column_index = _index_for(columns, centre_x)
        if row_index is None or column_index is None:
            continue
        grid[row_index][column_index].append(region.text)

    cells = [[_escape_cell(" ".join(cell)) for cell in row] for row in grid]

    # Drop trailing rows that ended up empty; structure detection commonly emits
    # a row band below the last populated one.
    while cells and not any(cells[-1]):
        cells.pop()
    if not cells:
        return ocr.text

    lines = [f"| {' | '.join(cells[0])} |"]
    if header_first_row:
        lines.append(f"| {' | '.join(['---'] * len(columns))} |")
    for row in cells[1:]:
        lines.append(f"| {' | '.join(row)} |")

    return "\n".join(lines)
