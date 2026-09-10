# Performance Guide

This page is a starting point for NeMo Retriever Library performance tuning guidance.

## Scope

Use this guide to document practical recommendations for:

- Extraction throughput and latency tuning
- Task-level settings (for example `extract`, `caption`, and `embed`)
- Deployment-specific tuning for library mode and Kubernetes/Helm
- NIM endpoint sizing and concurrency settings
- Benchmarking methodology and repeatable test setups

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

## Per-page profiling with pipeline traces { #pipeline-traces }

`GraphIngestor.ingest()` can return a `PipelineTrace` that records where an inprocess run spent its time. The payload answers two questions: which stage is the hotspot, and what is different about the pages that took longest. It uses only the Python standard library and pandas, so it adds no dependency and no external collector.

Use a trace when you want to do any of the following:

- Rank pipeline stages and operators by the time they actually consumed.
- Find the slowest pages in a corpus and inspect what those pages contain.
- Correlate page characteristics, such as raster size or detection count, with per-page time.
- Save a run profile to JSON so you can compare configurations later.

!!! note "Traces are collected for inprocess runs only"

    Only `run_mode="inprocess"` collects spans. `run_mode="batch"` runs stages
    in Ray actors, so `return_traces=True` returns an empty `PipelineTrace`
    that carries an explanatory note in `trace.notes`.
    `ServiceIngestor.ingest()` also accepts `return_traces=True`, but service
    mode returns the raw server-sent event dictionaries observed during the
    job instead of a `PipelineTrace`.

### Collect a trace from an ingest run { #collect-a-pipeline-trace }

Pass `return_traces=True` to `.ingest()`. The following example extracts a PDF, embeds the results, and prints the trace summary. Replace `data/multimodal_test.pdf` with a file that you supply.

```python
from nemo_retriever import create_ingestor

result, trace = (
    create_ingestor(run_mode="inprocess")
    .files(["data/multimodal_test.pdf"])
    .extract()
    .embed()
    .ingest(return_traces=True)
)

print(trace.summary())
```

`.ingest()` returns `result` by default, `(result, trace)` with `return_traces=True`, and `(result, failures, trace)` when you also pass `return_failures=True`. Refer to [Graph ingest](nemo-retriever-api-reference.md#graph-ingestor) for the full return contract.

`summary()` reports total run time, span and page counts, recorded notes, the slowest operations, and the slowest pages. Pass `limit=` to change how many entries it lists in each ranking. The output has the following shape, with values that depend on your documents and hardware.

```text
Pipeline trace (inprocess): 21.482s, 486 spans, 24 pages
  slowest operations:
    ocr.remote_invoke: 7.311s (34.0%)
    page_elements.remote_invoke: 5.002s (23.3%)
    pdf_extract.render: 2.740s (12.8%)
  slowest pages:
    data/multimodal_test.pdf p7: 1.204s
    data/multimodal_test.pdf p3: 0.998s
```

You can also request the trace through `IngestExecuteParams`. A keyword argument passed to `.ingest()` takes precedence over the value in `params`.

```python
from nemo_retriever import create_ingestor
from nemo_retriever.common.params import IngestExecuteParams

ingestor = create_ingestor(run_mode="inprocess").files(["data/multimodal_test.pdf"]).extract()
result, trace = ingestor.ingest(IngestExecuteParams(return_traces=True))
```

The library caches the most recent requested trace on `ingestor.last_trace`, so code that does not unpack the tuple can still reach the payload. The attribute is `None` until a run requests a trace.

Three attributes give you the raw material behind the rollups. `trace.total_s` is the wall-clock duration of the run in seconds. `trace.spans` is the tuple of recorded spans. `trace.notes` lists caveats that the library recorded about the run, and `trace.note(message)` appends your own, which is useful for labeling the configuration you tested.

### Rollups attribute time to stages and pages { #pipeline-trace-rollups }

Spans nest. The executor opens one span for each graph stage, and instrumented operators open nested spans beneath it. Rollups use self time, which is a span's duration minus the duration of its children, so nested spans do not double count.

Three rules govern how time reaches a page:

- Page identity is the pair `(source, page_number)`, and page numbers are 1-indexed. Page number `0` marks work on a whole document, such as the split stage that runs before pages exist.
- A span that covers a batch of pages splits its self time evenly across those pages. Batched remote inference calls therefore still appear in per-page numbers.
- Time that covers no identifiable page lands on a synthetic page whose source is `<unattributed>` and whose page number is `-1`.

The trace exposes four rollups, each of which returns a pandas DataFrame.

| Method | One row per | Key columns |
| --- | --- | --- |
| `stage_table()` | Graph stage | `spans`, `pages`, `total_s`, `self_s`, `self_pct`, `mean_s_per_page`, `errors` |
| `span_table()` | Stage and span name | `calls`, `total_s`, `self_s`, `mean_s`, `p50_s`, `p95_s`, `max_s`, `errors` |
| `page_table()` | Page | `total_s`, one `<stage>_s` column for each stage, and every captured page characteristic |
| `page_stage_table()` | Page and stage | `source`, `page_number`, `stage`, `seconds` |

Every rollup is ordered slowest first. `page_stage_table(by="name")` attributes time to individual span names instead of graph stages, which is the view for the question "where did page 7 spend its time."

```python
print(trace.stage_table())
print(trace.span_table().head(10))
print(trace.page_stage_table(by="name").query("page_number == 7").head(10))
```

Stage names come from graph node names, which default to the operator class name. Read `trace.stage_names` for the stages that a specific run recorded.

The inprocess PDF path opens nested spans under the following name prefixes.

| Span name prefix | Work covered |
| --- | --- |
| `pdf_split.` | PDF page splitting. |
| `pdf_extract.` | Document load, scan detection, text extraction, page rendering, and embedded image extraction. |
| `page_elements.` | Image decode, remote invoke, response parsing, and local preprocess, inference, and postprocess. |
| `ocr.` | Crop preparation, remote invoke, local inference, and response parsing. |
| `table_structure.` | Crop preparation, remote invoke, and local inference. |
| `embed.` | Modality grouping and local inference. |

An operator without nested spans still receives a stage-level span from the executor, so no stage is missing from `stage_table()`.

### Identify what makes a page slow { #find-slow-pages }

`hotspots()` and `slowest_pages()` return plain Python lists, which is convenient for logging and assertions. `hotspots()` yields `(span_name, self_s, pct)` tuples, and `slowest_pages()` yields `((source, page_number), seconds)` tuples.

```python
for name, seconds, percent in trace.hotspots(limit=5):
    print(f"{name}: {seconds:.3f}s ({percent:.1f}%)")

for (source, page_number), seconds in trace.slowest_pages(limit=5):
    print(f"{source} page {page_number}: {seconds:.3f}s")
```

`feature_correlation()` correlates numeric page characteristics against per-page time and returns a pandas Series ordered by absolute strength. A positive coefficient means that pages scoring higher on that characteristic tend to be slower.

```python
print(trace.feature_correlation())

# Explain one stage instead of the whole run.
slowest_stage = trace.stage_table().iloc[0]["stage"]
print(trace.feature_correlation(metric=f"{slowest_stage}_s"))
```

The `metric` argument accepts any column of `page_table()`, and it defaults to `total_s`. The method returns an empty Series when the run covered fewer pages than `min_pages`, which defaults to `3`. It also ignores `page_number`, the metric column, every other column that ends in `_s`, and any characteristic that holds the same value on every page.

!!! warning "Correlation is not causation"

    Coefficients describe association, not cause. A small number of pages
    produces noisy results, so raise `min_pages` on larger corpora and
    confirm a finding with a targeted run before you change configuration.

The trace records a page characteristic only when the underlying column is present at the point in the run where the executor inspects the frame.

| Characteristic | Meaning |
| --- | --- |
| `source_bytes` | Size in bytes of the input document payload. |
| `text_chars`, `text_words` | Character and word counts of the extracted text. |
| `has_text`, `needs_ocr_for_text` | Extraction metadata flags for the page. |
| `dpi` | Render resolution recorded during extraction. |
| `has_error` | Whether the row carried an error payload. |
| `image_height`, `image_width`, `image_megapixels` | Dimensions of the rendered page raster. |
| `image_b64_chars` | Length of the base64-encoded page image. |
| `num_detections` | Number of page element detections found on the page. |
| `detected_<label>` | Page element detections for one label, for example `detected_table`. |
| `ocr_detections` | Number of OCR detections found on the page. |
| `num_tables`, `num_charts`, `num_infographics`, `num_images` | Counts of extracted objects on the page. |
| `rows` | Number of DataFrame rows that the page occupied at the last observed stage. |
| `last_stage` | Name of the last stage that observed the page. |

### Save a trace from the CLI { #save-a-trace-from-the-cli }

`retriever ingest local --save-traces` collects the same payload and writes one JSONL file per ingest job. The command prints the path when the run finishes.

```console
$ retriever ingest local ./data --save-traces
Ingested 2 file(s) → 87 row(s) in LanceDB lancedb/nemo-retriever.
Saved ingest traces to .ingest_traces/ingest-trace-20260910T131328Z-57799.jsonl
  Load with: pandas.read_json(".ingest_traces/ingest-trace-20260910T131328Z-57799.jsonl", lines=True)
```

Traces are written to `./.ingest_traces` unless you pass `--trace-dir`, which names a directory and implies `--save-traces`. The flags require `run_mode="inprocess"`, so `retriever ingest batch` rejects them.

Each line of the file is one span charged to one page. That long form loads directly into pandas and answers both profiling questions from a single table.

```python
import pandas as pd

df = pd.read_json(".ingest_traces/ingest-trace-20260910T131328Z-57799.jsonl", lines=True)

# Slowest operations, then slowest pages.
print(df.groupby("name").seconds.sum().sort_values(ascending=False))
print(df.groupby(["source", "page_number"]).seconds.sum().nlargest(10))
```

Every row carries the span columns `name`, `stage`, `depth`, `seconds`, `span_self_s`, `span_total_s`, `span_pages`, `error`, and `attributes`, plus `run_id` and `n_documents` for the job. Each row also repeats the characteristics captured for its page, so you can correlate them without a join.

```python
# One row per page, then correlate characteristics with time.
pages = df[df.page_number > 0].drop_duplicates(["source", "page_number"])
totals = df.groupby(["source", "page_number"]).seconds.sum().rename("total_s")
print(pages.set_index(["source", "page_number"]).join(totals).corr(numeric_only=True)["total_s"])
```

The Python API writes the same format through `trace.save_jsonl(path)`, and `save_ingest_trace()` in `nemo_retriever.common.tracing` applies the CLI's directory and file-naming policy.

### Export a trace to JSON { #export-a-pipeline-trace }

Traces serialize to JSON so you can archive a run or compare two configurations offline.

```python
trace.save("traces/inprocess-baseline.json")

payload = trace.to_dict()
print(payload["run_mode"], payload["total_s"], payload["stages"])
```

`to_dict()` returns the run mode, start time, total seconds, stage names, notes, every span, and one entry per page with attributed seconds, per-stage seconds, and captured characteristics. `to_json(indent=...)` returns the same payload as a string and coerces values that JSON cannot represent to their string form. `save()` writes that JSON to a path and creates parent directories that do not exist.

## Shared preflight for custom Ray Data graphs

`GraphIngestor` reserves source capacity automatically. For custom graphs, declare source capacity before calling `preflight_executors(...)`. Set `source_cpu_reservation=1` on each `RayDataExecutor` that will receive a filesystem path or glob. `source_cpu_reservation` must be a finite, non-negative CPU value. An executor that only receives an existing Ray dataset can omit the reservation.

```python
file_executor = RayDataExecutor(graph, source_cpu_reservation=1)
inline_executor = RayDataExecutor(graph)
preflight_executors([file_executor, inline_executor], cluster_resources)
```

The shared preflight records these reservations. NeMo Retriever Library rejects a later filesystem input when its executor lacks the required reservation. It rejects the input before it starts Ray work. Construct a new executor with `source_cpu_reservation=1`, and include it in a new shared preflight instead.
