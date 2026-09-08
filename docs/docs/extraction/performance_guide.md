# Performance Guide

This page is a starting point for NeMo Retriever Library performance tuning guidance.

## Scope

Use this guide to document practical recommendations for:

- Extraction throughput and latency tuning
- Task-level settings (for example `extract`, `caption`, and `embed`)
- Deployment-specific tuning for library mode and Kubernetes/Helm
- NIM endpoint sizing and concurrency settings
- Benchmarking methodology and repeatable test setups

## Find bottlenecks with page tracing { #find-bottlenecks-with-page-tracing }

Before you tune worker counts or batch sizes, measure where the time goes. NeMo Retriever Library records a per-page trace of every operator that runs, and aggregates those traces into one JSON file per document.

Operator-level tracing is on by default and adds negligible overhead. The following command writes traces for a batch ingest. Replace `/path/to/your/pdfs` with a directory of PDF files that you supply.

```bash
retriever ingest batch /path/to/your/pdfs --page-trace-dir traces/

retriever trace traces/
```

The summary ranks pipeline stages by total and per-page time, lists the slowest pages, and reports the model versions that ran. Use that ranking to decide whether to add extraction workers, add NIM replicas, or change batch sizes. For the Python API, the trace file schema, and how to aggregate span timings correctly, refer to [Page tracing](page-tracing.md).

## Batch resource sizing

In batch mode, NeMo Retriever Library sizes unspecified Ray actor pools from Ray CPU and GPU resources. The library uses the resources that Ray reports as available immediately before it submits the pipeline. This prevents default extraction, OCR, and embedding pools from reserving more resources than the cluster can schedule.

For filesystem inputs, the library reserves CPU capacity for each Ray Data `ReadBinary` source task before it sizes actor pools. This reservation lets the input stage start instead of being blocked by persistent extraction actors. Inputs that are already Ray datasets, such as inline text rows, do not require this reservation.

If you set `BatchTuningParams` worker counts or direct `node_overrides`, those requests and required source-task reservations must fit the available Ray CPU and GPU budget. The library validates the final plan before submitting work and raises an error when it is infeasible. Reduce `*_workers` or per-node concurrency, or wait for shared-cluster capacity before retrying.

### Override worker counts

The library does not read environment variables to set worker counts or CPU and GPU totals. `CUDA_VISIBLE_DEVICES` still controls which GPUs Ray can see. To limit GPU count, start a Ray cluster with a restricted GPU set, for example `CUDA_VISIBLE_DEVICES=0 ray start --head --num-gpus=1`.

Set explicit worker counts with batch-mode CLI flags or with `BatchTuningParams` on `.extract()` and `.embed()`.

The following CLI example sets worker counts for a batch ingest. Replace
`/path/to/your/pdfs` with a directory of PDF files that you supply.

```bash
retriever ingest batch /path/to/your/pdfs \
  --pdf-extract-workers 4 \
  --page-elements-workers 3 \
  --ocr-workers 3 \
  --embed-workers 2
```

The following Python example passes the same worker counts through `BatchTuningParams`:

```python
from pathlib import Path

from nemo_retriever import create_ingestor
from nemo_retriever.common.params import BatchTuningParams

documents = [str(Path("data/multimodal_test.pdf"))]

chunks = (
    create_ingestor(run_mode="batch")
    .files(documents)
    .extract(
        batch_tuning=BatchTuningParams(
            pdf_extract_workers=4,
            page_elements_workers=3,
            ocr_workers=3,
        )
    )
    .embed(
        batch_tuning=BatchTuningParams(
            embed_workers=2,
        )
    )
    .ingest()
)
```

Related batch-size, CPU, and GPU-per-actor flags are documented in the [CLI ingest options](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docs/cli/README.md).

Use the Ray dashboard to verify the available-resource snapshot and the planned worker allocation when you tune throughput.

### Fused extraction tuning { #fused-extraction-tuning }

`ExtractParams(method="fused")` runs page elements detection, table structure
reconstruction, OCR, and embedding as one GPU-resident stage. That stage holds
the whole model stack in one process, so you tune it as a single actor pool
instead of tuning the four stages separately with `BatchTuningParams`. The
architectural benefit is that the model decodes each page raster once and keeps
the intermediate tensors on the device, which removes the host-to-device and
device-to-host copies that separate stages require.

Tune the pool with `FusedTuningParams` on `ExtractParams.fused_tuning`. The
following fields are supported.

| Field | Default | Purpose |
| --- | --- | --- |
| `fused_workers` | `1` | Number of fused-stage actors. |
| `fused_batch_size` | `64` | Pages submitted to the fused model for each call. |
| `fused_cpus_per_actor` | `1` | CPUs reserved for each fused actor. |
| `fused_gpus_per_actor` | `1.0` | GPUs reserved for each fused actor. The default is a whole device because the resident page tensors and the embedding tower are not shareable with a co-located actor. |

The following Python example raises the fused batch size and runs two fused
actors on a two-GPU host:

```python
from pathlib import Path

from nemo_retriever import create_ingestor
from nemo_retriever.common.params import ExtractParams, FusedTuningParams

documents = [str(Path("data/multimodal_test.pdf"))]

result = (
    create_ingestor(run_mode="batch")
    .files(documents)
    .extract(
        ExtractParams(
            method="fused",
            fused_tuning=FusedTuningParams(
                fused_workers=2,
                fused_batch_size=128,
            ),
        )
    )
    .ingest()
)
```

The example omits `.embed()` because the fused model emits embeddings directly.
Refer to [Run fused GPU-resident PDF extraction](nemo-retriever-api-reference.md#fused-gpu-resident-extraction)
for the validation rules and the optional package requirement.

Size `fused_workers` multiplied by `fused_gpus_per_actor` to the GPU capacity
that Ray reports for the cluster. Because the default reserves a whole GPU for
each actor, a request for more fused actors than available GPUs cannot
schedule.

## Shared preflight for custom Ray Data graphs

`GraphIngestor` reserves source capacity automatically. For custom graphs, declare source capacity before calling `preflight_executors(...)`. Set `source_cpu_reservation=1` on each `RayDataExecutor` that will receive a filesystem path or glob. `source_cpu_reservation` must be a finite, non-negative CPU value. An executor that only receives an existing Ray dataset can omit the reservation.

```python
file_executor = RayDataExecutor(graph, source_cpu_reservation=1)
inline_executor = RayDataExecutor(graph)
preflight_executors([file_executor, inline_executor], cluster_resources)
```

The shared preflight records these reservations. NeMo Retriever Library rejects a later filesystem input when its executor lacks the required reservation. It rejects the input before it starts Ray work. Construct a new executor with `source_cpu_reservation=1`, and include it in a new shared preflight instead.
