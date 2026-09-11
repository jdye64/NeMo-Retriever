# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU PDF engine: PDFium CPU parse/render, GPU JPEG raster, on-device tensors."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from nemo_retriever.common.api.util.pdf.engine import (
    ImageFormat,
    PDFEngine,
    PageImage,
    PdfEngineBackend,
    PdfSource,
    RenderMode,
)
from nemo_retriever.common.api.util.pdf.pdfium import convert_bitmap_to_corrected_numpy
from nemo_retriever.common.api.util.pdf.pdfium_document import PdfiumDocumentMixin
from nemo_retriever.common.api.util.pdf.render import compute_page_render_scale, encode_rgb_image


def _resolve_torch_device(device: Optional[str]) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The GPU PDF engine requires torch.") from exc

    if device:
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for GpuPDFEngine(device=%r)." % (device,))
        return resolved
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _bgra_bitmap_to_rgb_uint8(bitmap: Any, *, torch_mod: Any, device: Any) -> Any:
    """Upload a PDFium bitmap and convert BGRA/BGR to RGB uint8 CHW on ``device``."""
    img = bitmap.to_numpy()
    if not img.flags.writeable or not img.flags.c_contiguous:
        img = np.ascontiguousarray(img.copy())
    else:
        img = np.ascontiguousarray(img)
    tensor = torch_mod.from_numpy(img)
    if device.type == "cuda":
        tensor = tensor.to(device=device, non_blocking=True)
    else:
        tensor = tensor.clone()
    mode = getattr(bitmap, "mode", None)
    if tensor.ndim != 3:
        raise ValueError(f"Expected HWC bitmap tensor, got shape {tuple(tensor.shape)}")
    channels = int(tensor.shape[2])
    if mode in {"BGRA", "BGRX"} or channels == 4:
        rgb = tensor[:, :, [2, 1, 0]]
    elif mode == "BGR" or channels == 3:
        # PDFium BGR without rev_byteorder; swap to RGB. RGB bitmaps stay as-is
        # only when mode is already RGB (rare for PDFium).
        if mode in {None, "BGR"}:
            rgb = tensor[:, :, [2, 1, 0]]
        else:
            rgb = tensor
    else:
        raise ValueError(f"Unsupported PDFium bitmap mode={mode!r} shape={tuple(tensor.shape)}")
    return rgb.permute(2, 0, 1).contiguous()


def _encode_jpeg_from_chw(
    chw_uint8: Any,
    *,
    quality: int,
    torch_mod: Any,
    image_format: str,
) -> bytes:
    """Encode a CHW RGB uint8 tensor as JPEG, preferring GPU nvJPEG when possible."""
    fmt = image_format.lower()
    if fmt == "jpeg":
        encoded = _try_nvjpeg_encode(chw_uint8, quality=quality, torch_mod=torch_mod)
        if encoded is not None:
            return encoded
        encoded = _try_torchvision_encode_jpeg(chw_uint8, quality=quality)
        if encoded is not None:
            return encoded
    rgb_hwc = chw_uint8.detach().to("cpu").permute(1, 2, 0).contiguous().numpy()
    if rgb_hwc.dtype != np.uint8:
        rgb_hwc = rgb_hwc.astype(np.uint8, copy=False)
    return encode_rgb_image(rgb_hwc, image_format=fmt, jpeg_quality=quality)


def _try_torchvision_encode_jpeg(chw_uint8: Any, *, quality: int) -> Optional[bytes]:
    try:
        from torchvision.io import encode_jpeg
    except Exception:
        return None
    try:
        payload = encode_jpeg(chw_uint8.detach().to("cpu"), quality=int(quality))
    except Exception:
        return None
    if hasattr(payload, "cpu"):
        payload = payload.detach().cpu()
    if hasattr(payload, "numpy"):
        return bytes(payload.numpy().tobytes())
    return bytes(payload)


def _try_nvjpeg_encode(chw_uint8: Any, *, quality: int, torch_mod: Any) -> Optional[bytes]:
    """Encode JPEG on GPU with nvImageCodec when the tensor already lives on CUDA."""
    if getattr(chw_uint8, "device", None) is None or chw_uint8.device.type != "cuda":
        return None
    try:
        from nvidia import nvimgcodec
    except Exception:
        return None
    try:
        encoder = nvimgcodec.Encoder()
        hwc = chw_uint8.detach().permute(1, 2, 0).contiguous()
        cuda_array = torch_mod.as_tensor(hwc).detach()
        encoded = encoder.encode(
            cuda_array,
            codec="jpeg",
            jpeg_quality=int(quality),
        )
        if encoded is None:
            return None
        if isinstance(encoded, (bytes, bytearray)):
            return bytes(encoded)
        if hasattr(encoded, "__iter__") and not isinstance(encoded, (str, bytes)):
            first = next(iter(encoded), None)
            if first is None:
                return None
            if hasattr(first, "tobytes"):
                return bytes(first.tobytes())
            return bytes(first)
        if hasattr(encoded, "tobytes"):
            return bytes(encoded.tobytes())
        return bytes(encoded)
    except Exception:
        return None


class GpuPDFEngine(PdfiumDocumentMixin, PDFEngine):
    """PDFium for CPU PDF ops; GPU for JPEG rasterization and on-device tensors.

    PDFium still interprets the page and produces a bitmap because it has no
    CUDA raster backend. This engine uploads that bitmap, converts BGR to RGB
    on the GPU, encodes JPEG (nvJPEG / torchvision when available), and keeps
    CHW uint8 tensors on device for later ``NemotronPageElementsV3`` invoke.
    """

    def __init__(self, *, device: Optional[str] = None, keep_on_device: bool = True) -> None:
        super().__init__()
        self._torch_device = _resolve_torch_device(device)
        self._keep_on_device = bool(keep_on_device)
        self._device_tensors: List[Any] = []

    @property
    def backend(self) -> PdfEngineBackend:
        return "gpu"

    @property
    def device(self) -> str:
        dev = self._torch_device
        if getattr(dev, "type", None) == "cuda":
            index = getattr(dev, "index", None)
            return f"cuda:{index}" if index is not None else "cuda:0"
        return "cpu"

    def load(self, source: PdfSource) -> "GpuPDFEngine":
        self.release_device_images()
        self.load_document(source)
        return self

    def rasterize_page(
        self,
        page_index: int,
        *,
        dpi: int = 200,
        render_mode: RenderMode = "fit_to_model",
        image_format: ImageFormat = "jpeg",
        jpeg_quality: int = 100,
        keep_on_device: Optional[bool] = None,
    ) -> PageImage:
        import torch

        retain = self._keep_on_device if keep_on_device is None else bool(keep_on_device)
        doc = self._require_doc()
        n_pages = len(doc)
        if page_index < 0 or page_index >= n_pages:
            raise IndexError(f"page_index {page_index} is out of range for document with {n_pages} pages")

        page = None
        try:
            page = doc.get_page(page_index)
            render_scale = compute_page_render_scale(page, dpi=dpi, render_mode=render_mode)
            bitmap = page.render(scale=render_scale)
            try:
                chw = _bgra_bitmap_to_rgb_uint8(bitmap, torch_mod=torch, device=self._torch_device)
            except Exception:
                # Fall back to the proven host convert, then upload.
                arr = convert_bitmap_to_corrected_numpy(bitmap)
                if arr.ndim == 3 and arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                chw = (
                    torch.from_numpy(np.ascontiguousarray(arr))
                    .permute(2, 0, 1)
                    .contiguous()
                    .to(device=self._torch_device, non_blocking=(self._torch_device.type == "cuda"))
                )
            if chw.dtype != torch.uint8:
                chw = chw.to(dtype=torch.uint8)
            if self._torch_device.type == "cuda":
                torch.cuda.synchronize(self._torch_device)
            encoded = _encode_jpeg_from_chw(
                chw,
                quality=int(jpeg_quality),
                torch_mod=torch,
                image_format=image_format,
            )
            height, width = int(chw.shape[1]), int(chw.shape[2])
            device_tensor = chw if retain else None
            if retain:
                self._device_tensors.append(chw)
            return PageImage(
                page_index=page_index,
                width=width,
                height=height,
                nbytes=len(encoded),
                encoding=image_format.lower(),
                host_bytes=encoded,
                device_tensor=device_tensor,
            )
        finally:
            if page is not None and hasattr(page, "close"):
                try:
                    page.close()
                except Exception:
                    pass

    def device_tensors(self) -> List[Any]:
        """Return CHW uint8 RGB tensors retained from rasterize calls."""
        return list(self._device_tensors)

    def page_elements_batch(self) -> Any:
        """Stack retained device tensors for page-elements model invoke.

        Returns a BCHW uint8 tensor on the engine device. Callers pass this
        batch to ``NemotronPageElementsV3.preprocess`` / ``invoke``.
        """
        import torch

        if not self._device_tensors:
            raise RuntimeError("No on-device rasters. Call rasterize_page(s) with keep_on_device=True first.")
        return torch.stack(self._device_tensors, dim=0)

    def release_device_images(self) -> None:
        self._device_tensors.clear()
        try:
            import torch

            if self._torch_device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    def close(self) -> None:
        self.release_device_images()
        self.close_document()
