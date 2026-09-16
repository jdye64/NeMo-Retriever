# Pre-Requisites & Support Matrix

Before you begin using [NeMo Retriever Library](overview.md), confirm your software stack, deployment hardware, and advanced features you plan to enable (audio and video, Nemotron Parse, VLM image captioning, reranking, and `/v1/answer` generation) against the guidance on this page.

**Platform summary:** Supported **local GPU inference** requires **Linux** and CUDA 13. For **remote NIM inference**, the base Python package also installs on **Windows x64** and **macOS Apple Silicon (arm64)**; local GPU inference is not supported on those platforms. **macOS Intel (x86_64) is not supported** — `pip`/`uv` installs fail because Ray no longer publishes Intel Mac wheels.

> **Note — NVIDIA AI Enterprise (NVAIE) support**
>
> The NeMo Retriever Library, including its container image, is not supported under NVIDIA AI Enterprise (NVAIE), even though some NIM microservices and models it uses may be individually covered by NVAIE. For more information, refer to [NVIDIA AI Enterprise (NVAIE) support](overview.md#nvidia-ai-enterprise-nvaie-support).

## Software Requirements { #software-requirements }

- Linux operating systems (Ubuntu 22.04 or later recommended) for supported local GPU inference. For remote NIM inference, the base package can also be installed on Windows x64 and macOS Apple Silicon (arm64); local GPU inference is not supported on those platforms. macOS Intel (x86_64) is not supported: package installation fails because Ray `>=2.56.1` has no Intel Mac wheels (including in-process library mode).
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (local GPU inference only; NVIDIA Driver >= `580`, CUDA >= `13.0`)
- [Python](https://www.python.org/downloads/) `3.12` — required to install and run the NeMo Retriever Library Python API, CLI, and related packages from PyPI (for example `pip` or `uv`). Older Python versions will fail dependency resolution without a clear error.
- [UV Python package and environment manager](https://docs.astral.sh/uv/getting-started/installation/) (optional; recommended for creating isolated environments)
- For audio and video, `ffmpeg` and `ffprobe` must be on `PATH` (for example
  `sudo apt-get install -y --no-install-recommends ffmpeg` on Debian/Ubuntu).
  `ffmpeg-python` and `nemo-retriever[multimedia]` do not install these binaries.
  For container guidance, refer to [Audio and video](audio-video.md).
- For PDF extraction with `method="nemotron_parse"`, install the Nemotron Parse
  client dependencies with `uv pip install "nemo-retriever[nemotron-parse]"` (pulls
  `open-clip-torch`, which provides the `open_clip` module required by the Nemotron Parse
  NIM client). The base `nemo-retriever` install and `[local]` extra do not include this
  package. You can use the equivalent `pip install` command if you do not use UV.

> **Note**
>
> When you use UV, create the environment with Python 3.12 — for example, `uv venv --python 3.12`. This matches the `requires-python` metadata in the library packages.

## Hardware Requirements { #hardware-requirements }

The full ingestion pipeline is designed to consume significant CPU and memory resources to achieve maximal parallelism. 
Resource usage scales up to the limits of your deployed system.

For per-feature GPU memory, disk, and co-residency rules, refer to [Model hardware requirements](#model-hardware-requirements) below.


### Recommended Production Deployment Specifications

- **System Memory**: At least 256 GB RAM
- **CPU Cores**: At least 32 CPU cores
- **GPU**: NVIDIA GPU with at least 24 GB VRAM (for example, A100, H100, L40S, or equivalent)

> **Note**
>
> Using less powerful systems or lower resource limits is still viable, but performance will suffer.

### Resource Consumption Notes

- Batch mode sizes unspecified actor pools from the CPU and GPU resources that Ray reports as available at pipeline submission.
- Explicit batch worker counts and direct Ray node overrides must fit the available resource budget. The library rejects infeasible plans before submission.
- Memory usage can reach up to the full system capacity for large document processing
- CPU utilization scales with the number of concurrent processing tasks
- GPU is required for inference using HuggingFace models or NIMs
- GPU is NOT required for build.nvidia.com hosted inference

### Scaling Considerations

For production deployments processing large volumes of documents, consider:
- Higher memory configurations for processing large PDF files or image collections
- Additional CPU cores for improved parallel processing
- Multiple GPUs for distributed processing workloads

### Environment Requirements

Ensure your deployment environment meets these specifications before running the full pipeline. Resource-constrained environments may experience performance degradation.

## Core and Advanced Pipeline Features { #core-and-advanced-pipeline-features }

The NeMo Retriever Library extraction core pipeline features have a combined GPU memory footprint that fits on a single A10G or better GPU (about 4.8 GiB). That figure is model capacity. If you run four independent NIM containers without GPU sharing, plan for four GPU assignments.

Optional advanced features (audio and video transcription, Nemotron Parse, Omni image captioning, the VL reranker, and `/v1/answer` generation) are **not** part of that core footprint. Audio, video, Nemotron Parse, and Omni captioning each need **one or more additional dedicated GPUs** beyond the GPU running the four core NIMs. The VL reranker can share the core GPU when it has at least 80 GB VRAM. The default Super-49B answer LLM is also outside the core footprint. It needs **two additional physical GPUs**. Capacity requirements are listed in the **Additional Dedicated GPUs** rows of the [model hardware requirements](#model-hardware-requirements) table below.

<a id="optional-nims-not-auto-wired-by-default"></a>
<a id="optional-helm-nims-not-auto-wired-by-default"></a>

### Default NIMs { #default-nims }
<a id="default-helm-nims"></a>

> **Important — NVAIE support applies to individual NIMs only**
>
> A NIM or model listed in the default and optional NIM rows in the table below might be supported under NVIDIA AI Enterprise (NVAIE) as an individual product. That support does **not** cover its use through NeMo Retriever Library or extend to the library, its container image, or the end-to-end extraction workflow.

Four core NIMs make up the default extraction pipeline. Optional NIMs are used only when you opt in through the library, CLI, or service configuration.

| Component | NIM | Default image (`repository:tag`) | Role | Core pipeline |
|-----------|-----|----------------------------------|------|----------------|
| `page_elements` | [nemotron-page-elements-v3](https://build.nvidia.com/nvidia/nemotron-page-elements-v3) | `nvcr.io/nim/nvidia/nemotron-object-detection:2.0.1` | Page layout and element detection | Yes |
| `table_structure` | [nemotron-table-structure-v1](https://build.nvidia.com/nvidia/nemotron-table-structure-v1) | `nvcr.io/nim/nvidia/nemotron-object-detection:2.0.1` | Table structure extraction | Yes |
| `ocr` | [nemotron-ocr-v2](https://build.nvidia.com/nvidia/nemotron-ocr-v2) | `nvcr.io/nim/nvidia/nemotron-ocr-v2:2.0.1` | Image OCR | Yes |
| `vlm_embed` | [llama-nemotron-embed-vl-1b-v2](https://build.nvidia.com/nvidia/llama-nemotron-embed-vl-1b-v2) | `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.3.0` | Multimodal (VL) embedding | Yes |
| `rerankqa` | [llama-nemotron-rerank-vl-1b-v2](https://build.nvidia.com/nvidia/llama-nemotron-rerank-vl-1b-v2) | `nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:2.3.0` | Reranking for improved retrieval accuracy | No |
| `nemotron_parse` | [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse) | `nvcr.io/nim/nvidia/nemotron-parse-v1.2:1.7.0-variant` | Optional PDF `method="nemotron_parse"`. The Python `ExtractParams` default is `pdfium`; the CLI `auto` profile selects `pdfium_hybrid`. | No |
| `nemotron_3_nano_omni_30b_a3b_reasoning` | [nemotron-3-nano-omni-30b-a3b-reasoning](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) | `nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:2.0.4-variant` | Image captioning when you enable the caption stage. This VLM is also a supported configurable `/v1/answer` backend. Enabling captioning does not enable `/v1/answer`. Refer to [Answer generation](#answer-generation). | No |
| `audio` | [parakeet-1-1b-ctc-en-us](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/index.html) | `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:1.5.0` | [Audio and video](audio-video.md) transcription | No |
| `answer_llm` | [llama-3.3-nemotron-super-49b-v1.5](https://build.nvidia.com/nvidia/llama-3.3-nemotron-super-49b-v1.5) | `nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:2.0.5` | Optional `/v1/answer` generation. The generic slot defaults to Super-49B. You can override it to another OpenAI-compatible LLM or VLM, including Omni. Enabling this slot does not configure agentic retrieval. Refer to [Self-hosted Super-49B](workflow-agentic-retrieval.md#self-hosted-super-49b). Not part of the default extraction pipeline. | No |

The `page_elements` and `table_structure` services share the combined `nemotron-object-detection:2.0.1` image and select distinct models. For air-gapped, mirrored, or allowlisted deployments, pull that image once. Do not treat the older standalone `nemotron-page-elements-v3` or `nemotron-table-structure-v1` container images as the current defaults.

For self-hosted NIM GPU memory by SKU and precision, refer to the following product memory footprint tables.

- [Object detection supported hardware and memory footprint](https://docs.nvidia.com/nim/ingestion/object-detection/latest/support-matrix.html#supported-hardware-and-memory-footprint) covers `page_elements` and `table_structure`.
- [Image OCR supported hardware and memory footprint](https://docs.nvidia.com/nim/ingestion/image-ocr/latest/support-matrix.html#supported-hardware-and-memory-footprint) covers `ocr`.
- [Embedding NIM memory footprint](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html#memory-footprint) covers `embed` (`llama-nemotron-embed-vl-1b-v2`).
- [Reranking NIM memory footprint](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/support-matrix.html#memory-footprint) covers `rerank` (`llama-nemotron-rerank-vl-1b-v2`).

### Configure query reranking { #configure-query-reranking }

To rerank `retriever query` results, pass `--rerank` and a ranking endpoint or local reranker model. Refer to the [CLI reference](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/docs/cli).

<a id="nemotron-ocr-v2-language-mode"></a>

### Default NVCF endpoints { #default-nvcf-endpoints }

When you call [NVIDIA-hosted NIMs](deployment-options.md#when-to-use-nvidia-hosted-nims) from the Python library or CLI, these are the default remote endpoints the library uses when you do not set invoke URLs. Self-hosted NIMs use the HTTP URLs you configure instead.

| NIM | Default hosted endpoint | Notes |
|-----|-------------------------|-------|
| nemotron-page-elements-v3 | `https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3` | Core layout detection |
| nemotron-table-structure-v1 | `https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1` | Core table structure |
| nemotron-ocr-v2 | `https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2` | Default OCR SKU; library CPU actors default to this URL when no OCR invoke URL is set. **Local OCR language selectors (`--ocr-lang`, API `ocr_lang`) are not sent on remote requests** — hosted OCR v2 uses its own language behavior |
| llama-nemotron-embed-vl-1b-v2 | `https://integrate.api.nvidia.com/v1/embeddings` with model ID `nvidia/llama-nemotron-embed-vl-1b-v2` | Core multimodal embedding |
| llama-nemotron-rerank-vl-1b-v2 | `https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking` | Optional VL reranker |
| nemotron-parse | `https://integrate.api.nvidia.com/v1/chat/completions` with model ID `nvidia/nemotron-parse` | Optional `method="nemotron_parse"`. Hosted Build and self-hosted Parse v1.2 use different request contracts. Refer to [Nemotron Parse: hosted Build endpoint vs self-hosted NIM](#nemotron-parse-hosted-vs-self-hosted) |
| nemotron-3-nano-omni-30b-a3b-reasoning | `https://integrate.api.nvidia.com/v1/chat/completions` with model ID `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Optional image captioning. Also a supported configurable `/v1/answer` VLM backend when you point the service LLM settings at this endpoint. Enabling captioning does not enable `/v1/answer`. |
| llama-3.3-nemotron-super-49b-v1.5 | `https://integrate.api.nvidia.com/v1/chat/completions` with model ID `nvidia/llama-3.3-nemotron-super-49b-v1.5` | Default optional `/v1/answer` LLM and a supported OpenAI-compatible agentic RAG model. Self-hosted agentic use requires tool-call passthrough arguments. Hosted Build endpoints do not need that override. Agentic CLI and harness runs default to local in-process vLLM. Refer to [Answer generation](#answer-generation), [Self-hosted Super-49B](workflow-agentic-retrieval.md#self-hosted-super-49b), and [local in-process vLLM](workflow-agentic-retrieval.md#local-in-process-vllm). |
| parakeet-1-1b-ctc-en-us | `grpc.nvcf.nvidia.com:443` (function ID from [build.nvidia.com](https://build.nvidia.com/)) | Optional ASR; refer to [Parakeet hosted inference](audio-video.md#parakeet-hosted-inference-build-nvidia) |

<a id="nemotron-parse-hosted-vs-self-hosted"></a>

!!! note "Nemotron Parse: hosted Build and self-hosted NIM contracts"

    NVIDIA Build and self-hosted Nemotron Parse use distinct request contracts:

    - **Hosted Build** (`https://integrate.api.nvidia.com/v1/chat/completions`) resolves to model ID `nvidia/nemotron-parse` and uses an image-only tool-call contract.
    - **Self-hosted Parse v1.2** uses `nvidia/nemotron-parse-v1.2` and the tagged text-prompt contract. The default self-hosted image is `nvcr.io/nim/nvidia/nemotron-parse-v1.2:1.7.0-variant`.
    To use hosted Build, set `nemotron_parse_invoke_url` to the Build chat-completions URL and set `method="nemotron_parse"`. You can normally omit `nemotron_parse_model` so the library selects the model automatically. If you set `nemotron_parse_model` explicitly, it must match the endpoint contract.

    Each endpoint list must contain only hosted Build endpoints or only compatible self-hosted endpoints. The library rejects a list that mixes hosted Build and self-hosted endpoints because one extraction workflow uses one model ID and request contract. Setting `nemotron_parse_model` explicitly does not make a mixed list valid. To use both deployment types, configure separate ingestors or extraction workflows for each endpoint contract.

    For model/endpoint mismatch symptoms, refer to [Nemotron Parse model and endpoint mismatch](troubleshoot.md#nemotron-parse-model-endpoint-mismatch).

For local Hugging Face OCR language mode (`multi` vs `english`) and local model install, refer to [OCR and scanned documents](multimodal-extraction.md#ocr-and-scanned-documents) and [CLI — OCR language mode](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docs/cli/README.md#ocr-language-mode).

### Image captioning { #image-captioning }

Use **`nemotron_3_nano_omni_30b_a3b_reasoning`** when you enable the caption stage (hosted model ID `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`). The component is in the [Default NIMs](#default-nims) table above.

That caption path uses the Omni VLM for image captioning only. It does not enable `POST /v1/answer`. The same Omni model is a supported configurable answer-generation backend. Refer to [Answer generation](#answer-generation).

Optional features in the table above require GPU capacity **beyond the four default NIMs**. Audio and video transcription, Nemotron Parse, and Omni image captioning each need a **dedicated additional GPU** (or two, for Omni on L40S) separate from the core pipeline GPU. The VL reranker can share the core GPU only when that GPU has at least 80 GB of VRAM. Otherwise, treat the reranker as a standalone workload. Each optional feature also needs extra disk space and feature-specific system dependencies. The default Super-49B answer LLM requests two physical GPUs in addition to the core pipeline.

For published NIM model IDs and deployment-specific constraints, use the product support matrices linked under [Related Topics](#related-topics) below.

The Omni rows in the following table describe self-hosted NIM deployments. Direct local Hugging Face inference uses vLLM and also needs GPU memory for the KV cache and runtime allocations. For the local Omni BF16 profile, NeMo Retriever Library uses `gpu_memory_utilization=0.95` by default. On an 80 GB GPU, dedicate the GPU to local captioning. A checkpoint's approximately 62 GiB weight size alone does not establish local vLLM capacity.

### Answer generation { #answer-generation }

`POST /v1/answer` is optional and is not part of the default extraction pipeline. Enable it when you need grounded answers from retrieved VectorDB chunks.

The supported answer-generation model paths are:

- **Default LLM:** `nvidia/llama-3.3-nemotron-super-49b-v1.5`. The default self-hosted image is `nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:2.0.5`.
- **Configurable vision-language model (VLM):** `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`. Use this Omni NIM as the `/v1/answer` backend by pointing the service LLM `api_base` and `model` at an Omni chat-completions endpoint.

Captioning and `/v1/answer` are independent:

- Enabling Omni captioning does not enable `llm.enabled`.
- Enabling `/v1/answer` does not populate `agentic` settings and does not start Super-49B with tool-call parsers. Refer to [Self-hosted Super-49B](workflow-agentic-retrieval.md#self-hosted-super-49b).

If you enable caption Omni and the default Super-49B answer LLM as separate NIMs, GPU and disk requirements are additive. If Omni captioning is already running, you can reuse that NIM for `/v1/answer` without deploying Super-49B. Set `llm.enabled=true`, `llm.api_base` to the Omni OpenAI-compatible `/v1` base URL, and `llm.model` to `openai/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`.

The default Super-49B NIM is **in addition to** the four core NIMs. It is outside the one-physical-GPU core recipe. The default profile is BF16 tensor-parallel:

- Two physical GPUs.
- `NIM_TENSOR_PARALLEL_SIZE=2`.
- About 250 GB of model cache disk.
- GPU time-slice replicas cannot satisfy that two-GPU tensor-parallel request.

The [model hardware requirements](#model-hardware-requirements) table lists GPU SKUs that can run that default Super-49B profile. A100 40GB, A10G, L40S, and RTX PRO 4500 Blackwell are not supported for the default BF16 TP2 profile. The [NVIDIA NIM for Large Language Models support matrix](https://docs.nvidia.com/nim/large-language-models/latest/support-matrix.html) includes other Super-49B profiles with a different tensor-parallel size or precision.

Omni hardware for answer generation matches the Omni caption rows in the table below. Size a dedicated Omni answer LLM deployment from those rows. Do not add a second Omni GPU or cache when you reuse an already running caption Omni endpoint.

## Model Hardware Requirements { #model-hardware-requirements }

NeMo Retriever Library supports the following GPU hardware given system constraints in the table.

**Total GPUs** for Core Features is combined VRAM co-residency for the four default NIMs (one A10G or better). If you run four independent NIM containers without GPU sharing, plan for four GPU assignments.

**Additional Dedicated GPUs** counts VRAM required **in addition to** that core co-residency. For example, a library-mode or GPU-shared deployment that runs the core pipeline on one H100 and self-hosted Parakeet ASR needs **two GPUs total**. Enabling the default Super-49B profile adds two physical GPUs, not one. Time-slice replicas cannot satisfy that tensor-parallel request.

- **HF model weights** — approximate Hugging Face checkpoint footprint (files such as `model*.safetensors`, `weights.pth`, or other published weight bundles in the model repository). Values are rounded from the current public file listing and can change when the repository is updated.
- **NIM disk space** — approximate container and on-disk model cache for self-hosted NIM microservices (not the same as HF download size). For Nemotron 3 Nano Omni captioning, refer to the [NVIDIA NIM for Vision Language Models support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-3-nano-omni-30b-a3b-reasoning). For the default Super-49B answer LLM, plan about 250 GB of cache. Refer to the [NVIDIA NIM for Large Language Models support matrix](https://docs.nvidia.com/nim/large-language-models/latest/support-matrix.html) for other Super-49B profiles.
- **NIM GPU memory** — approximate self-hosted NIM GPU memory after startup. These values can differ from the Hugging Face weight sizes in the table and can increase with batch size, precision, and working buffers. Refer to the memory footprint tables after [Default NIMs](#default-nims).

Model repositories and NIM references are linked in [Core and Advanced Pipeline Features](#core-and-advanced-pipeline-features) above.

**B200, H200 NVL, and audio/video extraction:** The [audio and video](audio-video.md) transcription path (self-hosted Parakeet ASR) is **not supported on B200**, other Blackwell GPUs, or **H200 NVL**. Core PDF and multimodal extraction on those GPUs is unchanged. Refer to footnote ⁴ below.

| Feature | HF Model Weights | GPU Option | [RTX Pro 6000](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) | [B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) | [H200 NVL](https://www.nvidia.com/en-us/data-center/h200/) | [H100](https://www.nvidia.com/en-us/data-center/h100/) | [A100 80GB](https://www.nvidia.com/en-us/data-center/a100/) | A100 40GB | [A10G](https://aws.amazon.com/ec2/instance-types/g5/) | L40S | [RTX PRO 4500 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4500/) |
|---------|------------------|------------|--------|--------|--------|--------|--------|--------|--------|--------|------------------------|
| GPU | — | Memory | 96GB | 180GB | 141GB | 80GB | 80GB | 40GB | 24GB | 48GB | 32GB GDDR7 (GB203) |
| Core Features | ~4.8 GiB combined: embed VL 1b ~3.1 GiB; page-elements ~0.41 GiB; table-structure ~0.81 GiB; OCR ~0.51 GiB | Total GPUs⁵ | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Core Features | — | Total Disk Space | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB |
| Audio/video extraction (parakeet-1-1b-ctc-en-us) | ~4.0 GiB (`model.safetensors`; the repo also ships `parakeet-ctc-1.1b.nemo` of similar size—use one format to avoid roughly doubling disk use) | Additional Dedicated GPUs | Not supported⁴ | Not supported⁴ | Not supported⁴ | 1¹ | 1¹ | 1¹ | 1¹ | 1¹ | Not supported⁴ |
| | — | Additional Disk Space | Not supported⁴ | Not supported⁴ | Not supported⁴ | ~37GB¹ | ~37GB¹ | ~37GB¹ | ~37GB¹ | ~37GB¹ | Not supported⁴ |
| nemotron-parse | ~3.5 GiB | Additional Dedicated GPUs | Not supported | 1 | Not supported | 1 | 1 | 1 | 1 | 1 | 1 |
| nemotron-parse | — | Additional Disk Space | Not supported | ~16GB | Not supported | ~16GB | ~16GB | ~16GB | ~16GB | ~16GB | ~16GB |
| Omni caption or answer (nemotron-3-nano-omni-30b-a3b-reasoning) | ~62 GiB (BF16); ~33 GiB (FP8); ~21 GiB (NVFP4) | Additional Dedicated GPUs | 1 | 1 | 1 | 1 | 1 | Not supported | Not supported | 2 | Not supported³ |
| Omni caption or answer (nemotron-3-nano-omni-30b-a3b-reasoning) | — | Additional Disk Space (HF) | ~21–62GB | ~21–62GB | ~21–62GB | ~21–62GB | ~21–62GB | Not supported | Not supported | ~21–62GB | Not supported³ |
| Omni caption or answer (nemotron-3-nano-omni-30b-a3b-reasoning) | — | Additional Disk Space (NIM) | ~80GB | ~80GB | ~80GB | ~80GB | ~80GB | Not supported | Not supported | ~80GB | Not supported³ |
| Answer generation (llama-3.3-nemotron-super-49b-v1.5, default `answer_llm`) | ~98 GiB (BF16) | Additional Dedicated GPUs | 2⁶ | 2⁶ | 2⁶ | 2⁶ | 2⁶ | Not supported⁶ | Not supported⁶ | Not supported⁶ | Not supported⁶ |
| Answer generation (llama-3.3-nemotron-super-49b-v1.5, default `answer_llm`) | — | Additional Disk Space (NIM) | ~250GB⁶ | ~250GB⁶ | ~250GB⁶ | ~250GB⁶ | ~250GB⁶ | Not supported⁶ | Not supported⁶ | Not supported⁶ | Not supported⁶ |
| Reranker | ~3.1 GiB (llama-nemotron-rerank-vl-1b-v2) | With Core Pipeline | Yes | Yes | Yes | Yes | Yes | No* | No* | No* | No* |
| Reranker | — | Standalone (recall only) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

¹ On other supported GPUs, Parakeet ASR (`parakeet-1-1b-ctc-en-us:1.5.0`) may require a runtime TensorRT engine build (no prebuilt profile in the published image).

⁴ Self-hosted [audio/video extraction](audio-video.md) through Parakeet ASR (`parakeet-1-1b-ctc-en-us:1.5.0`) is **not supported** on **B200**, other **Blackwell** GPUs (compute capability 12.0), including RTX PRO 6000 Blackwell and RTX PRO 4500 Blackwell, or **H200 NVL**. Core PDF and multimodal extraction on those GPUs is unchanged. Video workflows that depend on Parakeet for speech transcription are affected the same way. A self-hosted Parakeet container may fail while building the Riva/TensorRT engine (for example ONNX Runtime IR version, cuDNN visibility, or FP8 tactic errors). Use a supported dedicated GPU (for example H100 or A100) or [hosted Parakeet on build.nvidia.com](audio-video.md#parakeet-hosted-inference-build-nvidia).

³ Opt-in Omni captioning uses the [nemotron-3-nano-omni-30b-a3b-reasoning](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning) NIM (`nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:2.0.4-variant`). The same NIM hardware applies when you use Omni as the configurable `/v1/answer` backend. BF16 requires at least 80 GB total GPU memory for this NIM deployment. Refer to the [VLM NIM support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-3-nano-omni-30b-a3b-reasoning). L40S requires two GPUs. A100 40GB, A10G, and RTX PRO 4500 are below the minimum. Do not add a second Omni GPU or NIM cache when `/v1/answer` reuses an already running caption Omni endpoint.

⁶ The default Super-49B answer NIM (`nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:2.0.5`) uses two physical GPUs and `NIM_TENSOR_PARALLEL_SIZE=2` for the bundled BF16 tensor-parallel profile. Those two GPUs are **in addition to** the core pipeline. Time-slice replicas cannot satisfy that request. Plan about 250 GB of model cache. A100 40GB (40 GB), A10G (24 GB), L40S (48 GB), and RTX PRO 4500 Blackwell (32 GB) are not supported for that default BF16 TP2 profile. Other Super-49B NIM profiles can run on additional SKUs. Refer to the [NVIDIA NIM for Large Language Models support matrix](https://docs.nvidia.com/nim/large-language-models/latest/support-matrix.html).

⁵ **Total GPUs** is combined VRAM co-residency for the four core models. If you run four independent NIM containers without GPU sharing, plan for four GPU assignments.

* GPUs with less than 80GB VRAM cannot run the reranker concurrently with the core pipeline.
To perform recall testing with the reranker on these GPUs, shut down the core pipeline NIM microservices
and run only the embedder, reranker, and your vector database.

## Related Topics { #related-topics }

- [Troubleshooting](troubleshoot.md)
- [Release Notes](releasenotes.md)
- [Deployment options](deployment-options.md) (local Python, hosted NIMs, and Docker)
- [NVIDIA NIM for Object Detection (supported hardware and memory footprint)](https://docs.nvidia.com/nim/ingestion/object-detection/latest/support-matrix.html#supported-hardware-and-memory-footprint)
- [NVIDIA NIM for Image OCR (supported hardware and memory footprint)](https://docs.nvidia.com/nim/ingestion/image-ocr/latest/support-matrix.html#supported-hardware-and-memory-footprint)
- [NVIDIA NeMo Retriever Embedding NIM (memory footprint)](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html#memory-footprint)
- [NVIDIA NeMo Retriever Reranking NIM (memory footprint)](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/support-matrix.html#memory-footprint)
- [NVIDIA NIM for Vision Language Models (support matrix)](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html)
- [NVIDIA NIM for Large Language Models (support matrix)](https://docs.nvidia.com/nim/large-language-models/latest/support-matrix.html)
- [NVIDIA Speech NIM Microservices (support matrix)](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/index.html)
