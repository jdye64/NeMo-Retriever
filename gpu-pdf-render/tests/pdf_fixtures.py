# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from io import BytesIO

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject


def write_rect_pdf(colors: list[tuple[float, float, float]], width: float = 72.0, height: float = 72.0) -> bytes:
    """Build a multi-page PDF with one filled rectangle per page."""
    writer = PdfWriter()
    for red, green, blue in colors:
        page = writer.add_blank_page(width=width, height=height)
        stream = DecodedStreamObject()
        stream.set_data(f"{red} {green} {blue} rg 10 10 20 20 re f\n".encode("ascii"))
        page[NameObject("/Contents")] = stream
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def write_text_pdf() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=80)
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 30 Td (GPU) Tj ET\n1 0 0 rg 5 5 12 12 re f\n")
    page[NameObject("/Contents")] = stream
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
