# Deployment options

Use this page to compare how you run NeMo Retriever, including when to use [NVIDIA-hosted NIMs](https://build.nvidia.com/) versus self-hosting inference on your own hardware.

Kubernetes Helm charts are no longer supported. Use the Python library, hosted NIMs, or the standalone Docker service image instead.

## Compare deployment options

Use the sections below to pick documentation and deployment options that match your goal.

### I want to run locally or embed the library

1. [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
2. [Use the Python API](nemo-retriever-api-reference.md) or [Use the CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/docs/cli) — install and run the [`nemo_retriever`](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever) package in your environment

### I want a standalone Docker service container

Build and run the NeMo Retriever service image with the [Docker service image guide](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docker.md). Use this for local or host-level service-container validation. Development Compose helpers live in [`nemo_retriever/dev/compose`](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/dev/compose/README.md).

### I want examples and notebooks

1. [Jupyter Notebooks](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/README.md)

### I need API details and keys

1. [Get your API key](api-keys.md)
2. [API reference — PDF pre-splitting](nemo-retriever-api-reference.md#pdf-pre-splitting-for-parallel-ingest) if applicable

### I am tuning performance or cost

1. [Evaluation and performance](evaluate-on-your-data.md)
2. [Throughput is dataset-dependent](multimodal-extraction.md#extraction-limitations-and-quality)
3. [Evaluate on your data](evaluate-on-your-data.md)

## When to use NVIDIA-hosted NIMs { #when-to-use-nvidia-hosted-nims }

[NVIDIA-hosted NIMs](https://build.nvidia.com/) run inference on NVIDIA-managed infrastructure. You call models with API keys (refer to [Get your API key](api-keys.md)) without operating GPU nodes yourself.

Consider hosted NIMs when:

- You want the fastest path to try models and iterate without installing drivers or containers on your own hosts.
- Latency to NVIDIA endpoints works for your region and use case.
- Your compliance and data policies allow document or query content in the hosted service (confirm with your security review).

**Also refer to:** [NVIDIA NIM catalog](https://build.nvidia.com/)

## When to self-host NIMs { #when-to-self-host-nims }

Self-hosted NIMs run on your GPUs or air-gapped hardware. Point the library, CLI, or Docker service at those OpenAI-compatible or NIM HTTP endpoints.

Consider self-hosting when:

- You need an air gap, strict data residency, or customer data must not leave your network.
- You run at large scale where dedicated capacity can cost less than hosted API usage.
- You must meet latency or locality requirements that hosted regions cannot satisfy.

**GPU sharing.** Combined core NIM VRAM fits on one A10G or better GPU. If you run four independent NIM containers without GPU sharing, plan for four GPU assignments. Refer to [Model hardware requirements](prerequisites-support-matrix.md#model-hardware-requirements).

## Air-gapped and disconnected deployment { #air-gapped-deployment }

The **default document extraction pipeline** (page elements, table structure, OCR, and VL embed) runs disconnected when you mirror images and models into a private registry and point the library or Docker service at those endpoints.

On a staging host with internet access, pull from NGC, retag to your private registry, then run in the enclave with registry and endpoint overrides.

!!! warning "Audio and video extraction"

    Audio and video workflows require `ffmpeg` and `ffprobe` on `PATH`. Runtime package installation is not suitable for air-gapped hosts. Refer to [Audio and video](audio-video.md). Skip this if you do not use audio or video.

For offline image captioning, deploy a self-hosted [Nemotron 3 Nano Omni](prerequisites-support-matrix.md#image-captioning) NIM and point your pipeline caption endpoint at that HTTP URL instead of `integrate.api.nvidia.com` or other hosted APIs.

**Related**

- [Docker service image](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docker.md)
- [Development Compose helpers](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/dev/compose/README.md)
- [About getting started](getting-started-about.md)
- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Audio and video](audio-video.md)
