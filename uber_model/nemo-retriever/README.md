# NeMo Retriever fused extraction model

One model, one invocation, four stages, no host round-trips between them.

This project fuses the four models the NeMo Retriever extraction pipeline calls
per page into a single in-process model that keeps every intermediate in GPU
memory:

```
page elements -> table structure -> OCR -> embed -> (optional) rerank
```

The production pipeline runs those stages as separate Ray actors that exchange
pandas DataFrames through the object store, so each page image is base64
encoded, transferred, and decoded once per stage. This model decodes each page
once and every stage reads the same resident tensor.

Stage order, model weights, and stage parameters match production, so results
are comparable. What changes is where the data lives in between.

## Status

Local development project. Not published to the Hub yet, though it is laid out
as a HuggingFace repo (`configuration_nemo_retriever.py`,
`modeling_nemo_retriever.py`) so it can be.

## Install

Requires a CUDA device, Python 3.12, and the three upstream model packages.

```bash
pip install -e '.[local]'
```

Then mirror the pinned weights locally so the model loads from disk rather than
resolving through the Hub on every start:

```bash
python ../download_models.py
export NEMO_RETRIEVER_FUSED_MIRROR="$(cd .. && pwd)/huggingface"
```

## Usage

```python
from nemo_retriever_fused import NemoRetrieverFusedModel

model = NemoRetrieverFusedModel.from_pretrained()

result = model(page_images_b64)          # one call, all four stages

for page in result.pages:
    print(page.page_id, page.text)
    for element in page.elements_of("table"):
        print(element.table_markdown)

result.page_embeddings.shape             # [num_pages, 2048], still on device
print(result.timing_table())             # per-stage device milliseconds
```

Pages that are already resident on the GPU skip decoding entirely:

```python
from nemo_retriever_fused import DeviceImageBatch

batch = DeviceImageBatch.from_tensors(rendered_pages, device="cuda:0")
result = model(batch)
```

### Configuration

```python
from nemo_retriever_fused import ExecutionConfig, FusedPipelineConfig

config = FusedPipelineConfig(
    extract_tables=True,
    extract_charts=True,
    enable_embed=True,
    enable_rerank=False,
    execution=ExecutionConfig(
        device="cuda:0",
        overlap_detectors=True,     # run the two YOLOX detectors on two streams
        decode_on_device=True,      # nvJPEG instead of PIL
        single_sync_point=True,     # one synchronise per invocation
    ),
)

model = NemoRetrieverFusedModel.from_pretrained(config)
```

Disabling a stage skips loading its weights, so a detection-only or text-only
deployment pays for only what it runs.

### Reranking

Reranking is a query-time stage and is off by default. When enabled, it shares
the VL tiling work with the embed stage, so reranking pages that are already
resident from ingestion pays no image preprocessing at all.

```python
config = FusedPipelineConfig(enable_rerank=True)
model = NemoRetrieverFusedModel.from_pretrained(config)

scores, order = model.rerank("quarterly revenue", texts=candidates, top_k=10)
```

`rerank` sorts and truncates on device, so only the surviving candidates are
copied to the host.

### HuggingFace surface

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("./nemo-retriever", trust_remote_code=True)
output = model(pages=[page_b64])
output.result.pages[0].text
```

The pipeline is inference only. It composes four pretrained models with
non-differentiable postprocessing between them, so `forward` returns extraction
results rather than logits.

## Models

Every stage loads the same pinned revision the production pipeline resolves, so
a fused run and a production run use byte-identical weights. The pins live in
`nemo_retriever_fused/loader.REVISIONS` and are kept in step with
`nemo_retriever.models.hf_model_registry.HF_MODEL_REVISIONS`.

| Stage | Model | Pinned revision |
|---|---|---|
| Page elements | `nvidia/nemotron-page-elements-v3` | `df62dbb6` |
| Table structure | `nvidia/nemotron-table-structure-v1` | `9350162f` |
| OCR | `nvidia/nemotron-ocr-v2` | `0e83e83f` |
| OCR (legacy) | `nvidia/nemotron-ocr-v1` | `8657d08d` |
| Embed | `nvidia/llama-nemotron-embed-vl-1b-v2` | `582e3bf7` |
| Rerank | `nvidia/llama-nemotron-rerank-vl-1b-v2` | `9c20c4ae` |

## Layout

```
nemo_retriever_fused/
  config.py            stage and execution configuration
  gpu_image.py         DeviceImageBatch, nvJPEG decode
  gpu_ops.py           letterbox, ROI crop, NMS, weighted box fusion, VL tiling
  loader.py            pinned-revision weight loading
  pipeline.py          NemoRetrieverFusedModel, the fused orchestrator
  results.py           FusedResult, PageResult, ElementResult
  table_assembly.py    structure plus OCR to markdown
  benchmark.py         stage timings and PCIe transfer accounting
  stages/
    detectors.py       shared YOLOX detector, page elements, table structure
    ocr.py             device-resident OCR, bypassing the host entry point
    embed.py           device-resident VL embedding and reranking
configuration_nemo_retriever.py   HuggingFace config
modeling_nemo_retriever.py        HuggingFace PreTrainedModel wrapper
tests/                            unit tests, CPU-only plus CUDA-marked
```

## Design

The individual models are already well optimised on the GPU. The cost this
project removes is at the seams between them. Four things carry most of the
benefit:

1. **OCR no longer round-trips through the host.** `nemotron_ocr`'s
   `_load_image_to_tensor_uint8` calls `.detach().cpu()` even on CUDA tensors,
   and the NRL wrapper feeds it a base64 PNG. `GpuOcrStage` builds the detector
   batch on device and calls the upstream phase methods directly.
2. **Pages decode once, not once per stage.** JPEG pages decode on the GPU with
   nvJPEG, so only the compressed bytes cross PCIe.
3. **Crops are slices, not re-encodes.** `gpu_ops.crop_regions` returns views
   that share storage with the page.
4. **Box postprocessing stays on device.** Weighted box fusion and the per-class
   score gate run in torch, removing a synchronisation per page.

`PERFORMANCE.md` documents each of these against the specific file and line in
the production pipeline or upstream package it addresses, plus the optimisations
that are identified but not yet implemented.

## Benchmark

```bash
export NEMO_RETRIEVER_FUSED_MIRROR=/path/to/uber_model/huggingface

python -m nemo_retriever_fused.benchmark --synthetic 16 --iterations 5
python -m nemo_retriever_fused.benchmark --synthetic 16 --no-overlap
python -m nemo_retriever_fused.benchmark --synthetic 16 --no-device-decode
```

Reports per-stage device milliseconds from CUDA events and host-to-device and
device-to-host byte counts. The byte counts are the more portable comparison
against the production pipeline, since they do not vary with GPU model.

## Tests

```bash
PYTHONPATH=. pytest tests/ -q
```

The unit tests run on CPU tensors so they execute without a GPU. The operations
are device-agnostic torch, so correctness there implies correctness on CUDA;
CUDA-specific behaviour is covered by tests marked `requires_cuda`, which skip
when no device is present.
