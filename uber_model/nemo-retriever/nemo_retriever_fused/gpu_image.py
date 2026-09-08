# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident page image batches.

The production pipeline carries page images between stages as base64 strings in
a pandas column, so each stage pays a base64 decode, a PIL or torchvision
decode, and a host-to-device copy. `DeviceImageBatch` decodes once into device
memory and hands the same tensors to every downstream stage.

Decoding uses nvJPEG through `torchvision.io.decode_jpeg(device="cuda")` when
the payload is JPEG. PNG has no GPU decoder in torchvision, so PNG payloads
decode on the host and transfer through pinned memory; the transfer is still a
single copy per page instead of one per stage.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

logger = logging.getLogger(__name__)

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _strip_data_url(payload: str) -> str:
    """Return the base64 body of a ``data:...;base64,`` URL, or *payload*."""
    if payload.startswith("data:"):
        _, _, remainder = payload.partition(",")
        return remainder
    return payload


def decode_base64_image(
    payload: str | bytes,
    *,
    device: torch.device,
    pinned_staging: bool = True,
    prefer_device_decode: bool = True,
) -> torch.Tensor:
    """Decode a base64 image straight into device memory.

    Parameters
    ----------
    payload:
        Base64 text (optionally a ``data:`` URL) or raw encoded image bytes.
    device:
        CUDA device to land the decoded tensor on.
    pinned_staging:
        Stage the host-side fallback copy through pinned memory so the transfer
        is asynchronous with respect to compute on the default stream.
    prefer_device_decode:
        Use nvJPEG for JPEG payloads. Set False to force the host path, which
        is useful when comparing numerics against the production pipeline.

    Returns
    -------
    torch.Tensor
        ``[3, H, W]`` uint8 RGB tensor on *device*.

    Raises
    ------
    ValueError
        If the payload is not valid base64 or not a decodable image.
    """
    if isinstance(payload, str):
        try:
            raw = base64.b64decode(_strip_data_url(payload), validate=False)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("page image payload is not valid base64") from exc
    else:
        raw = payload

    if not raw:
        raise ValueError("page image payload decoded to zero bytes")

    is_jpeg = raw.startswith(_JPEG_MAGIC)

    if prefer_device_decode and is_jpeg and device.type == "cuda":
        try:
            from torchvision.io import ImageReadMode, decode_jpeg

            # nvJPEG wants the compressed bytes as a CPU uint8 tensor and does
            # the entropy decode plus colour conversion on the GPU, so the only
            # thing crossing PCIe is the compressed payload.
            buffer = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            return decode_jpeg(buffer, mode=ImageReadMode.RGB, device=device)
        except Exception as exc:  # noqa: BLE001 - nvJPEG availability varies by build
            logger.debug("nvJPEG decode unavailable, falling back to host decode: %s", exc)

    return _decode_on_host(raw, device=device, pinned_staging=pinned_staging)


def _decode_on_host(raw: bytes, *, device: torch.device, pinned_staging: bool) -> torch.Tensor:
    """Decode on the CPU and copy to *device* in one transfer."""
    if raw.startswith(_PNG_MAGIC) or raw.startswith(_JPEG_MAGIC):
        from torchvision.io import ImageReadMode, decode_image

        host = decode_image(torch.frombuffer(bytearray(raw), dtype=torch.uint8), mode=ImageReadMode.RGB)
    else:
        from PIL import Image

        import numpy as np

        with Image.open(io.BytesIO(raw)) as opened:
            converted = opened.convert("RGB")
            host = torch.from_numpy(np.asarray(converted, dtype=np.uint8)).permute(2, 0, 1).contiguous()

    if device.type != "cuda":
        return host
    if pinned_staging:
        host = host.pin_memory()
    return host.to(device, non_blocking=pinned_staging)


@dataclass(slots=True)
class DeviceImageBatch:
    """A batch of page images that lives in device memory for its whole life.

    Attributes
    ----------
    images:
        One ``[3, H, W]`` uint8 CUDA tensor per page. Sizes may differ, so this
        is a list rather than a stacked tensor; stages that need a rectangular
        batch letterbox on demand.
    page_ids:
        Caller-supplied identifiers, carried through so results can be joined
        back to the source documents without a second lookup.
    device:
        The device every tensor in `images` is on.
    """

    images: list[torch.Tensor]
    page_ids: list[str]
    device: torch.device
    _shapes: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.images) != len(self.page_ids):
            raise ValueError(
                f"images and page_ids must be the same length, got {len(self.images)} and {len(self.page_ids)}"
            )
        for index, image in enumerate(self.images):
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"page {index} must be [3, H, W], got {tuple(image.shape)}")
            if image.device != self.device:
                raise ValueError(f"page {index} is on {image.device}, expected {self.device}")
        self._shapes = [(int(image.shape[1]), int(image.shape[2])) for image in self.images]

    def __len__(self) -> int:
        return len(self.images)

    @property
    def shapes(self) -> list[tuple[int, int]]:
        """Return the ``(height, width)`` of each page."""
        return list(self._shapes)

    @classmethod
    def from_base64(
        cls,
        payloads: Sequence[str | bytes],
        *,
        device: torch.device | str,
        page_ids: Iterable[str] | None = None,
        pinned_staging: bool = True,
        prefer_device_decode: bool = True,
    ) -> DeviceImageBatch:
        """Build a batch by decoding base64 page images onto *device*.

        Parameters
        ----------
        payloads:
            Base64 page images, as produced by the extraction stage's
            `_render_page_to_base64`.
        device:
            Target CUDA device.
        page_ids:
            Optional identifiers. Defaults to the positional index.
        pinned_staging, prefer_device_decode:
            Forwarded to `decode_base64_image`.
        """
        resolved = torch.device(device)
        images = [
            decode_base64_image(
                payload,
                device=resolved,
                pinned_staging=pinned_staging,
                prefer_device_decode=prefer_device_decode,
            )
            for payload in payloads
        ]
        identifiers = list(page_ids) if page_ids is not None else [str(index) for index in range(len(images))]
        return cls(images=images, page_ids=identifiers, device=resolved)

    @classmethod
    def from_tensors(
        cls,
        images: Sequence[torch.Tensor],
        *,
        device: torch.device | str,
        page_ids: Iterable[str] | None = None,
    ) -> DeviceImageBatch:
        """Build a batch from tensors that are already on *device*.

        Accepts ``[3, H, W]`` or ``[H, W, 3]`` uint8 tensors and normalises to
        channel-first without copying when the layout already matches.
        """
        resolved = torch.device(device)
        normalized: list[torch.Tensor] = []
        for index, image in enumerate(images):
            if image.ndim != 3:
                raise ValueError(f"page {index} must be 3-dimensional, got {tuple(image.shape)}")
            if image.shape[0] != 3 and image.shape[2] == 3:
                image = image.permute(2, 0, 1)
            if image.dtype != torch.uint8:
                image = image.clamp(0, 255).to(torch.uint8)
            normalized.append(image.to(resolved, non_blocking=True).contiguous())

        identifiers = list(page_ids) if page_ids is not None else [str(index) for index in range(len(normalized))]
        return cls(images=normalized, page_ids=identifiers, device=resolved)

    def as_float(self) -> list[torch.Tensor]:
        """Return the pages as float32 tensors in 0-255 space.

        The detectors want float input; this keeps the conversion in one place
        so it happens once per page rather than once per stage.
        """
        return [image.to(torch.float32) for image in self.images]

    def nbytes(self) -> int:
        """Return the total device memory the batch occupies, in bytes."""
        return sum(image.element_size() * image.numel() for image in self.images)
