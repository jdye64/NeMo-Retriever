# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident multimodal embedding and reranking stages.

The production embed stage receives page and region images as base64 strings in
a pandas column, decodes each one with PIL, and hands the resulting
`PIL.Image` to the VL processor. The processor then runs `dynamic_preprocess`,
which does a PIL bicubic resize followed by up to `max_tiles` PIL crops, and
`build_transform`, which runs `ToTensor` and `Normalize` once per tile. For a
six-tile page that is one decode, one resize, seven crops, and seven
normalisation passes, all on the host, for an image that the OCR stage already
had resident on the GPU.

`GpuEmbedStage` takes the resident page or region tensor and produces the
`pixel_values` batch with `gpu_ops.tile_for_vl_tower`, so the tiling and
normalisation happen on device and the image is never re-encoded.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch

from nemo_retriever_fused import gpu_ops
from nemo_retriever_fused.config import EmbedConfig, RerankConfig

logger = logging.getLogger(__name__)


# Image placeholder tokens the VL processor splices into the prompt. The model
# replaces every IMG_CONTEXT position with one projected vision feature, so the
# count has to match the tiles handed to `forward` exactly.
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class _VisionLanguageModel(Protocol):
    """The subset of the VL model the fused stages depend on."""

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor: ...


@dataclass(slots=True)
class EmbedResult:
    """Embeddings for a batch of items.

    Attributes
    ----------
    vectors:
        ``[N, D]`` embeddings. These stay on device unless the caller asks for
        host tensors, which is the point of the fused pipeline: a downstream
        vector database writer that supports device buffers never needs the
        host copy.
    dimension:
        Embedding width, for callers that validate against a collection schema.
    """

    vectors: torch.Tensor
    dimension: int

    def to_host(self) -> torch.Tensor:
        """Return the vectors as a host float32 tensor.

        This is the one deliberate device-to-host transfer in the embed stage.
        Call it only at the boundary where results leave the pipeline.
        """
        return self.vectors.detach().to(torch.float32).cpu()


def _l2_normalize(vectors: torch.Tensor) -> torch.Tensor:
    """L2-normalize along the last dimension, on whatever device holds *vectors*."""
    return torch.nn.functional.normalize(vectors.to(torch.float32), p=2.0, dim=-1)


def _masked_average(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average the unpadded positions of a final hidden state.

    This is the ``avg`` pooling the VL retriever configures, and it is what
    turns the model's per-token hidden state into one vector per document.
    Averaging over the mask makes the result independent of padding side.
    """
    mask = attention_mask[..., None].bool()
    summed = hidden_states.masked_fill(~mask, 0.0).sum(dim=1)
    return summed / attention_mask.sum(dim=1)[..., None]


class GpuEmbedStage:
    """Embed resident image tensors and text without host round-trips.

    Parameters
    ----------
    model:
        A loaded VL embedding model on *device*.
    tokenizer:
        Tokenizer for the text path.
    config:
        Embedding settings.
    device:
        Device the model's weights live on.
    dtype:
        Vision tower dtype. The VL tower ships bfloat16.
    """

    def __init__(
        self,
        model: _VisionLanguageModel,
        tokenizer: Any,
        config: EmbedConfig,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._device = device
        self._dtype = dtype

    def build_pixel_values(self, images: Sequence[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
        """Tile and normalize resident images into one `pixel_values` batch.

        Parameters
        ----------
        images:
            ``[C, H, W]`` CUDA tensors in 0-255 space.

        Returns
        -------
        tuple[torch.Tensor, list[int]]
            The concatenated ``[sum(tiles), C, tile, tile]`` batch and the tile
            count per image, which the model needs to split features back apart.
        """
        tiles: list[torch.Tensor] = []
        counts: list[int] = []
        for image in images:
            tiled = gpu_ops.tile_for_vl_tower(
                image,
                tile_size=self._config.tile_size,
                min_tiles=self._config.min_tiles,
                max_tiles=self._config.max_tiles,
                use_thumbnail=self._config.use_thumbnail,
                norm_type=self._config.norm_type,
                dtype=self._dtype,
            )
            tiles.append(tiled)
            counts.append(int(tiled.shape[0]))
        return torch.cat(tiles, dim=0), counts

    def _processor(self) -> Any:
        """Return the model's own processor, which owns the prompt contract."""
        processor = getattr(self._model, "processor", None)
        if processor is None:
            raise RuntimeError(
                "the VL embedding model exposes no processor, so the document "
                "prompt contract cannot be reproduced; load it with "
                "trust_remote_code=True"
            )
        return processor

    def _prompt(self, *, text: str, tile_count: int) -> str:
        """Build the document prompt for one item.

        The vision tower alone does not produce a retrieval embedding: the
        model projects tiles into the language model's embedding space, splices
        them over the IMG_CONTEXT positions, and pools the language model's
        final hidden state. Reproducing that means reproducing the prompt the
        processor would have built, with one IMG_CONTEXT run per tile.
        """
        processor = self._processor()
        # The template lives in the model's remote code alongside the processor.
        template = sys.modules[type(processor).__module__].get_conv_template(processor.template)
        template.system_message = processor.system_message

        content = f"{self._config.document_prefix} <image> {text}".rstrip()
        template.messages.clear()
        template.append_message(template.roles[0], content)
        template.append_message(template.roles[1], None)

        image_tokens = (
            IMG_START_TOKEN + IMG_CONTEXT_TOKEN * int(self._model.num_image_token) * tile_count + IMG_END_TOKEN
        )
        return template.get_prompt().replace("<image>", image_tokens, 1)

    def _encode(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize *prompts* onto the model's device."""
        encoded = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=getattr(self._processor(), "p_max_length", None),
            return_tensors="pt",
        )
        return (
            encoded["input_ids"].to(self._device, non_blocking=True),
            encoded["attention_mask"].to(self._device, non_blocking=True),
        )

    def embed_images(
        self,
        images: Sequence[torch.Tensor],
        *,
        texts: Sequence[str] | None = None,
    ) -> EmbedResult:
        """Embed resident images, returning device-resident vectors.

        Parameters
        ----------
        images:
            ``[C, H, W]`` CUDA tensors in 0-255 space.
        texts:
            Optional page text to embed alongside each image, matching the
            multimodal document the production embed stage builds. Defaults to
            image-only documents.

        Returns
        -------
        EmbedResult
            ``[N, D]`` vectors, one per image.
        """
        if not images:
            empty = torch.zeros((0, self._config.output_dimension), device=self._device)
            return EmbedResult(vectors=empty, dimension=self._config.output_dimension)

        item_texts = list(texts) if texts is not None else [""] * len(images)
        if len(item_texts) != len(images):
            raise ValueError(f"texts has {len(item_texts)} entries for {len(images)} images")

        outputs: list[torch.Tensor] = []
        batch_size = max(1, self._config.batch_size)

        for start in range(0, len(images), batch_size):
            window = list(images[start : start + batch_size])
            window_texts = item_texts[start : start + batch_size]

            # Tiling stays on device; this is the work the production stage does
            # with PIL after a base64 decode.
            pixel_values, counts = self.build_pixel_values(window)
            prompts = [self._prompt(text=text, tile_count=count) for text, count in zip(window_texts, counts)]
            input_ids, attention_mask = self._encode(prompts)

            with torch.no_grad():
                outputs_batch = self._model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

            outputs.append(_masked_average(outputs_batch.hidden_states[-1], attention_mask))

        vectors = torch.cat(outputs, dim=0)
        if self._config.normalize:
            vectors = _l2_normalize(vectors)
        return EmbedResult(vectors=vectors, dimension=int(vectors.shape[-1]))

    def embed_texts(self, texts: Sequence[str], *, is_query: bool = False) -> EmbedResult:
        """Embed text passages or queries.

        Parameters
        ----------
        texts:
            Raw text, without the task prefix.
        is_query:
            Apply the query prefix instead of the passage prefix.
        """
        if not texts:
            empty = torch.zeros((0, self._config.output_dimension), device=self._device)
            return EmbedResult(vectors=empty, dimension=self._config.output_dimension)

        prefix = self._config.query_prefix if is_query else self._config.document_prefix
        prefixed = [f"{prefix} {text}" for text in texts]

        outputs: list[torch.Tensor] = []
        batch_size = max(1, self._config.batch_size)

        for start in range(0, len(prefixed), batch_size):
            window = prefixed[start : start + batch_size]
            input_ids, attention_mask = self._encode(window)

            # No pixel_values, so the language model runs over text alone. The
            # pooling is the same as the image path; only the prompt differs.
            with torch.no_grad():
                outputs_batch = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

            outputs.append(_masked_average(outputs_batch.hidden_states[-1], attention_mask))

        vectors = torch.cat(outputs, dim=0)
        if self._config.normalize:
            vectors = _l2_normalize(vectors)
        return EmbedResult(vectors=vectors, dimension=int(vectors.shape[-1]))


class GpuRerankStage:
    """Score query and document pairs from resident tensors.

    Reranking is a query-time stage, so it is optional in the ingest path. It
    shares the VL tiling work with `GpuEmbedStage`, which means a rerank run
    over pages already resident from ingestion pays no image preprocessing at
    all.

    Parameters
    ----------
    model:
        A loaded VL cross-encoder on *device*.
    tokenizer:
        Tokenizer for the query and passage text.
    config:
        Reranking settings.
    device:
        Device the model's weights live on.
    dtype:
        Vision tower dtype.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: RerankConfig,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._device = device
        self._dtype = dtype

    def rank(
        self,
        query: str,
        *,
        texts: Sequence[str] | None = None,
        images: Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return relevance logits for *query* against each candidate.

        Parameters
        ----------
        query:
            The search query.
        texts:
            Candidate passage text, one entry per candidate.
        images:
            Candidate page images as resident ``[C, H, W]`` CUDA tensors, one
            entry per candidate.

        Returns
        -------
        torch.Tensor
            ``[N]`` device-resident logits, higher is more relevant. Sort on
            device and slice the top-k before any host transfer.
        """
        candidate_count = len(texts or images or [])
        if candidate_count == 0:
            return torch.zeros((0,), device=self._device)

        pixel_values = None
        if images:
            tiles = [
                gpu_ops.tile_for_vl_tower(
                    image,
                    tile_size=self._config.tile_size,
                    min_tiles=self._config.min_tiles,
                    max_tiles=self._config.max_tiles,
                    use_thumbnail=self._config.use_thumbnail,
                    norm_type=self._config.norm_type,
                    dtype=self._dtype,
                )
                for image in images
            ]
            pixel_values = torch.cat(tiles, dim=0)

        pairs = [(query, text) for text in (texts or [""] * candidate_count)]
        encoded = self._tokenizer(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device, non_blocking=True) for key, value in encoded.items()}
        if pixel_values is not None:
            encoded["pixel_values"] = pixel_values

        with torch.inference_mode():
            output = self._model(**encoded)

        logits = output if isinstance(output, torch.Tensor) else output.logits
        return logits.reshape(-1).to(torch.float32)
