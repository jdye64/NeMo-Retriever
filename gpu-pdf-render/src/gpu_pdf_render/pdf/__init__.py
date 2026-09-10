# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from pypdf import PdfReader
from pypdf.generic import ContentStream, DictionaryObject, NameObject

from gpu_pdf_render.pdf.ir import PageIR

PdfSource = Union[str, Path, bytes]


def load_pdf(source: PdfSource) -> list[PageIR]:
    """Load every page of a PDF into an interpreter-ready IR."""
    reader = _open_reader(source)
    pages: list[PageIR] = []
    for index, page in enumerate(reader.pages):
        if hasattr(page, "transfer_rotation_to_content"):
            page.transfer_rotation_to_content()
        box = page.mediabox
        width_pt = float(box.width)
        height_pt = float(box.height)
        operations = _page_operations(page, reader)
        resources = page.get("/Resources")
        if resources is not None:
            resources = resources.get_object()
        pages.append(
            PageIR(
                index=index,
                width_pt=width_pt,
                height_pt=height_pt,
                operations=operations,
                resources=resources,
                reader=reader,
            )
        )
    if not pages:
        raise ValueError("PDF contains no pages")
    return pages


def _open_reader(source: PdfSource) -> PdfReader:
    if isinstance(source, (bytes, bytearray)):
        from io import BytesIO

        return PdfReader(BytesIO(bytes(source)))
    return PdfReader(str(source))


def _page_operations(page: Any, reader: PdfReader) -> list[tuple[list[Any], str]]:
    contents = page.get_contents()
    if contents is None:
        return []
    stream = ContentStream(contents, reader)
    ops: list[tuple[list[Any], str]] = []
    for operands, operator in stream.operations:
        op = operator.decode("latin1") if isinstance(operator, (bytes, bytearray)) else str(operator)
        ops.append((list(operands), op))
    return ops


def resolve_name(resources: Any, category: str, name: Any) -> Any | None:
    if resources is None:
        return None
    bucket = resources.get(NameObject(category)) if isinstance(resources, DictionaryObject) else resources.get(category)
    if bucket is None:
        return None
    bucket = bucket.get_object()
    key = name if isinstance(name, NameObject) else NameObject(str(name) if str(name).startswith("/") else f"/{name}")
    value = bucket.get(key)
    if value is None:
        return None
    return value.get_object()
