# uber_model

Workspace for the fused NeMo Retriever model.

| Path | Purpose |
|---|---|
| `download_models.py` | Mirrors the pinned HuggingFace snapshots the extraction pipeline uses |
| `huggingface/` | The mirror itself. Not tracked; regenerate with the script |
| `nemo-retriever/` | The fused model project. Laid out as a HuggingFace repo, kept local for now |

## Mirroring the models

Revisions come from `nemo_retriever/src/nemo_retriever/models/hf_model_registry.py`,
so the mirror always matches the pins the production pipeline resolves at
runtime.

```bash
python uber_model/download_models.py                 # all six repos, about 8.7 GB
python uber_model/download_models.py page_elements   # a subset
python uber_model/download_models.py --source-only   # modeling code only, no weights
```

The mirrored repos ship their full PyTorch source alongside the weights, which
is what the fused model was built against:

| Stage | Repo | Source of interest |
|---|---|---|
| Page elements | `nvidia/nemotron-page-elements-v3` | `nemotron_page_elements_v3/` — YOLOX, WBF postprocessing |
| Table structure | `nvidia/nemotron-table-structure-v1` | `nemotron_table_structure_v1/` — the same YOLOX geometry |
| OCR | `nvidia/nemotron-ocr-v2` | `nemotron-ocr/src/nemotron_ocr/inference/pipeline_v2.py` |
| OCR (legacy) | `nvidia/nemotron-ocr-v1` | `nemotron-ocr/src/nemotron_ocr/inference/pipeline.py` |
| Embed | `nvidia/llama-nemotron-embed-vl-1b-v2` | VL tower and processor |
| Rerank | `nvidia/llama-nemotron-rerank-vl-1b-v2` | `processing_llama_nemotron_vl.py` — tiling and normalisation |

Point the fused model at the mirror so it loads from disk instead of the Hub:

```bash
export NEMO_RETRIEVER_FUSED_MIRROR="$(pwd)/uber_model/huggingface"
```

## The fused model

See [`nemo-retriever/README.md`](nemo-retriever/README.md) for usage and
[`nemo-retriever/PERFORMANCE.md`](nemo-retriever/PERFORMANCE.md) for the
optimisation notes, each cited against the pipeline or upstream source it
addresses.
