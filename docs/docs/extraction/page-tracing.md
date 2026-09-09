# Page Tracing

Page tracing records how long [NeMo Retriever Library](overview.md) spends on every page of every document that it ingests. Use it to find pipeline bottlenecks, compare detail levels and worker counts, and identify which operator, model, or endpoint dominates a run.

Page tracing is a performance-analysis tool. It does not change extraction output, and it does not replace the row-level error payloads described in [Troubleshoot NeMo Retriever Library](troubleshoot.md).


## How page tracing works { #how-page-tracing-works }

Each page accumulates spans as it moves through the pipeline. A span is one timed operation applied to a page or to a batch of pages. At `full` detail the spans also cover the individual network calls, model invocations, GPU work, and heavy dependency calls that run inside each operator.

When a document finishes, NeMo Retriever Library aggregates the spans for that document into one JSON artifact that retains the full per-page detail. The library writes one file per document. Refer to [Trace file schema](#trace-file-schema).

Page tracing works in all three run modes: `inprocess`, `batch`, and `service`. In service mode, the worker that runs the pipeline records the trace and the gateway returns it to the client.


## Choose a detail level { #choose-a-detail-level }

The detail level controls how much the pipeline records and how much overhead tracing adds. The following table lists the supported values.

| Value | What it records | Overhead |
|-------|-----------------|----------|
| `off` | Nothing. Tracing is disabled. | None. |
| `operator` | One span per operator per page. This is the default in the `inprocess` and `batch` run modes. | Negligible. |
| `full` | Everything in `operator`, plus curated spans inside each operator. | Small. Two timestamps per traced call, and larger trace files. |

In service mode the default is `off` instead, because the trace has to be transmitted rather than kept in the local process. Refer to [Read page traces from the service API](#read-page-traces-from-the-service-api).

At `full` detail, NeMo Retriever Library adds spans for the following work.

| Span name | Category | What it measures |
|-----------|----------|------------------|
| `nim.infer.{model}` | `network` | One inference call through a NIM client, over gRPC or HTTP. |
| `nim.http.post` | `network` | One HTTP request to a NIM endpoint, including the retry attempt and HTTP status. |
| `gpu.{model}` | `gpu` | One forward pass of a local GPU model. |
| `pdfium.render_page` | `cpu` | Rasterizing one PDF page at the configured DPI. |
| `pdfium.extract_text` | `cpu` | Extracting the text layer of one PDF page. |
| `pdfium.extract_images` | `cpu` | Extracting embedded image objects from one PDF page. |
| `pdfium.pages_to_numpy` | `cpu` | Converting rendered pages into arrays. |
| `libreoffice.convert_to_pdf` | `io` | Converting an Office document to PDF through LibreOffice. |
| `image.encode.{format}` | `cpu` | Encoding image crops, for example `image.encode.png`. |
| `image.decode.base64` | `cpu` | Decoding base64 image payloads. |

Every span records the wall time between entering and leaving the operation, and nothing more. Tracing never synchronizes CUDA and never drives a profiler, so turning it on does not change how the pipeline executes. Read `gpu.{model}` spans with that in mind: they measure how long the forward call occupied the calling thread, which for an asynchronous launch is not the same as device execution time. In practice these calls transfer results back to the host before returning, so the two are usually close, but the span cannot promise it. Reach for [Nsight Systems](https://developer.nvidia.com/nsight-systems) when you need true kernel-level attribution. The library already emits NVTX ranges around local GPU inference for exactly that purpose.

You can set the detail level in the following ways, listed from highest precedence to lowest:

- The `--page-trace-detail` option on the ingest commands.
- The `page_trace_detail` parameter on `IngestExecuteParams` and `.ingest()`.
- The `NEMO_RETRIEVER_PAGE_TRACE_DETAIL` environment variable.

When you set none of them, the effective detail level is `operator` in the `inprocess` and `batch` run modes, and `off` in service mode.

Requesting trace output overrides `off`. If the resolved detail level is `off` but you called `.save_page_traces()` or passed `return_page_traces=True`, the run falls back to `operator` detail so that the request produces something to read.


## Enable page tracing from the CLI { #enable-page-tracing-from-the-cli }

The `retriever ingest`, `retriever ingest batch`, and `retriever ingest service` commands accept the following options.

| Option | Description |
|--------|-------------|
| `--page-trace-dir DIR` | Write one trace JSON file per document into `DIR`. The directory is created if it does not exist. |
| `--page-trace-detail LEVEL` | Set the detail level. Valid values are `off`, `operator`, and `full`. The default is `operator`. This option also reads `NEMO_RETRIEVER_PAGE_TRACE_DETAIL`. |

The following command runs a batch ingest at full detail and writes traces to `traces/`. Replace `/path/to/your/pdfs` with a directory of PDF files that you supply.

```bash
retriever ingest batch /path/to/your/pdfs \
  --page-trace-dir traces/ \
  --page-trace-detail full
```


## Enable page tracing in Python { #enable-page-tracing-in-python }

Every ingestor that `create_ingestor()` returns provides the `.save_page_traces()` fluent method. It persists one JSON file per document to a directory that you choose, in every run mode.

The following example runs an in-process ingest at `full` detail and writes the trace files to `traces/`:

```python
from nemo_retriever import create_ingestor

ingestor = (
    create_ingestor(run_mode="inprocess")
    .files("data/report.pdf")
    .extract()
    .save_page_traces(output_directory="traces/")
)
results = ingestor.ingest(page_trace_detail="full")
```

`.save_page_traces()` accepts a `compression` argument. The default is `None`, which writes plain JSON. Pass `compression="gzip"` to write gzip-compressed files instead. Any other value raises a `ValueError`.

To receive the traces in memory rather than reading them back from disk, pass `return_page_traces=True` to `.ingest()`. Each trace is a plain dictionary matching [Trace file schema](#trace-file-schema).

```python
results, page_traces = ingestor.ingest(return_page_traces=True)
```

You can combine `return_page_traces=True` with `.save_page_traces(...)` to both write files and inspect traces in the same process.

The traces from the most recent `.ingest()` call are also available on the `page_traces` property, in every run mode. Reading the property does not change the shape of the `.ingest()` return value, which is convenient when you want the results unpacked normally:

```python
results = ingestor.ingest(page_trace_detail="full")
for trace in ingestor.page_traces:
    print(trace["document"]["source_path"], trace["document_summary"]["total_ms"])
```

The property returns a copy of the list, and each `.ingest()` call resets it. It holds whatever traces the run produced, so in service mode it is empty unless the run opted into tracing by calling `.save_page_traces()`, passing `return_page_traces=True`, or setting `page_trace_detail` explicitly.

The following table lists the page-tracing parameters on `IngestExecuteParams` and `.ingest()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page_trace_detail` | `"off"`, `"operator"`, or `"full"` | Unset, which resolves to `"operator"` | The amount of per-page timing detail that the pipeline records. |
| `return_page_traces` | `bool` | `False` | Append the collected page traces to the return value. |

### Understand the return-tuple ordering { #understand-the-return-tuple-ordering }

`.ingest()` returns the results object on its own when you request no extras. Each extra you request is appended in the fixed order `failures`, `traces`, `page_traces`, regardless of the order in which you pass the flags. The following table lists the resulting shapes.

| Flags passed | Return value |
|--------------|--------------|
| None | `results` |
| `return_page_traces=True` | `(results, page_traces)` |
| `return_failures=True` | `(results, failures)` |
| `return_failures=True`, `return_page_traces=True` | `(results, failures, page_traces)` |
| `return_failures=True`, `return_traces=True`, `return_page_traces=True` | `(results, failures, traces, page_traces)` |

!!! note "return_page_traces is not return_traces"

    `return_traces` is a separate, pre-existing service-mode parameter. It
    returns the raw server-sent events protocol events observed during the
    request, which help you debug client and gateway behavior. It records no
    page timings. Use `return_page_traces` for per-page pipeline timings.


## Read page traces from the service API { #read-page-traces-from-the-service-api }

In service mode the client sends the detail level to the gateway on the request pipeline spec as `page_trace_detail`, so the level applies to the worker that actually runs the pipeline. The worker records the trace, and the status endpoints return it.

Tracing is opt-in in service mode, unlike the local run modes where it defaults to `operator`. A service trace has to travel back on every status response for its document, which for a long document is a substantial payload, so the client asks for one only when you called `.save_page_traces()` or passed `return_page_traces=True`. The wire default is `off`. Passing `page_trace_detail` explicitly also opts in.

Document and job status responses carry an optional `page_trace` object. The field is additive, so existing clients are unaffected. Its value is `null` unless the worker recorded a trace for that document. When present, the object matches [Trace file schema](#trace-file-schema).

The following endpoints include the field:

- `GET /v1/ingest/status/{item_id}`, `GET /v1/ingest/page/status/{page_id}`, and `GET /v1/ingest/document/status/{document_id}`, on `JobStatusResponse`.
- `GET /v1/ingest/job/{job_id}/document/{document_id}`, on `DocumentStatusResponse`.

The `ServiceIngestor` client reads this field for you. Use `.save_page_traces()` or `return_page_traces=True` as described in [Enable page tracing in Python](#enable-page-tracing-in-python) rather than polling the endpoints yourself.


## Analyze traces with the retriever trace commands { #analyze-traces-with-the-retriever-trace-commands }

The `retriever trace` command group reads trace files. Every subcommand takes a `FILES...` argument that accepts trace files, directories that contain `*.trace.json` or `*.trace.json.gz` files, and globs.

### Summarize a run { #summarize-a-run }

`summary` is the default subcommand, so passing paths directly to `retriever trace` runs it. The summary reports the document and page counts, total traced time, time per page, the library version, the run mode, a ranking by operator, the share of time by category, the models and versions that ran, and the slowest pages.

```bash
retriever trace traces/
```

The following table lists the `summary` options.

| Option | Default | Description |
|--------|---------|-------------|
| `--top`, `-n` | `10` | The number of rows to show in each ranked table. |
| `--json` | Off | Emit the rollup as JSON instead of Rich tables. |

The following command widens every ranked table to 20 rows:

```bash
retriever trace traces/ --top 20
```

When the category table is empty, the run recorded only operator spans. Re-run the ingest with `--page-trace-detail full` to get the network, GPU, CPU, and I/O breakdown.

### Inspect a single page { #inspect-a-single-page }

The `page` subcommand prints a nested span waterfall for one page, with the duration, the amortized time, the batch size, and the model, endpoint, protocol, and HTTP status attributes of each span. Use it after the summary identifies a slow page.

```bash
retriever trace page traces/ --page 7 --document report.pdf-1a2b3c4d
```

The following table lists the `page` options.

| Option | Default | Description |
|--------|---------|-------------|
| `--page`, `-p` | Required | The 1-based page number to inspect. |
| `--document`, `-d` | Unset | The document ID or source path to inspect. Required when the paths hold more than one trace. |
| `--json` | Off | Emit the page detail as JSON instead of Rich tables. |

### Export spans for external analysis { #export-spans-for-external-analysis }

The `export` subcommand flattens the spans from one or more trace files into a single table. It writes the same rows that [`spans_dataframe()`](#load-traces-with-pandas) returns.

```bash
retriever trace export traces/ --output spans.parquet
```

The following table lists the `export` options.

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | Required | The destination file. Parent directories are created as needed. |
| `--format` | Inferred from the extension of `--output` | The output format. Valid values are `parquet`, `csv`, and `jsonl`. |

The command infers `parquet` from a `.parquet` or `.pq` extension, `csv` from `.csv`, and `jsonl` from `.jsonl`, `.ndjson`, or `.json`. Pass `--format` when the extension is something else.

Parquet export requires `pyarrow`. If it is not installed, install it or export with `--format csv`. Span attributes do not survive a CSV round-trip, so prefer `parquet` or `jsonl` when you need the `attrs` column.


## Interpret span timings { #interpret-span-timings }

Two distinctions determine whether your analysis is correct. Read this section before you aggregate any trace data.

### Sum amortized_ms, not duration_ms { #sum-amortized-ms-not-duration-ms }

Operators process pages in batches, so a single operator span can cover several pages at once. The trace reports that span once per page it covered, with two durations:

- `duration_ms` is the exact measured wall time of the batched operation. The same value repeats on every page that the span covered.
- `amortized_ms` divides `duration_ms` by `page_fanout`, the number of pages that the span covered.

Sum `amortized_ms`, never `duration_ms`. Amortized values are additive, so summing them across pages reproduces the exact measured duration, and page rollups reconcile with document totals. Summing `duration_ms` multiplies each batch cost by its page count and inflates the result. Use `duration_ms` only when you want the measured cost of one specific batched call.

Every number that `retriever trace` reports is already derived from `amortized_ms`.

### duration_ms includes children, self_ms excludes them { #duration-ms-includes-children-self-ms-excludes-them }

Spans nest. A `network` or `gpu` span sits inside the `operator` span that made the call. `duration_ms` includes the time spent in child spans, and `self_ms` excludes it.

As a result, `document_summary.by_category.operator_ms` is the total pipeline stage time, and `network_ms`, `gpu_ms`, `cpu_ms`, and `io_ms` are the instrumented breakdown inside that time. They are not additional time on top of it. Do not add the category values together and expect the document total.

A large operator `self_ms` means that time is going somewhere that is not yet instrumented. Re-run at `full` detail to narrow it down.

### Span categories { #span-categories }

Every span carries a `category` field. The following table lists the categories.

| Category | Meaning |
|----------|---------|
| `operator` | A pipeline stage applied to a page, such as extraction or OCR. |
| `network` | A remote call, such as a NIM gRPC or HTTP request. |
| `gpu` | GPU work, such as a local model forward pass. |
| `cpu` | Local compute, such as PDF rendering or image encoding. |
| `io` | File, storage, and subprocess access. |


## Load traces with pandas { #load-traces-with-pandas }

The `nemo_retriever.common.tracing` module reads saved trace files and flattens them into pandas `DataFrame` objects. Use `load_traces()` to read a file, a directory, a glob, or a sequence of any of those. Then use `spans_dataframe()` to build a span table, or `page_summaries_dataframe()` to build a table of per-page rollups.

The following example ranks operators by total amortized time across a whole run:

```python
from nemo_retriever.common.tracing import load_traces, spans_dataframe

traces = load_traces("traces/")
spans = spans_dataframe(traces)
spans.groupby("operator").amortized_ms.sum().sort_values(ascending=False)
```

`spans_dataframe()` returns one row per span per page. The columns are the span fields listed in [Span fields](#span-fields), plus `run_id`, `run_mode`, and `library_version` carried down from the trace file. Group by `page_number` to compare pages, by `model_key` to attribute time to a model, or by `category` to separate network time from GPU time. Sum `amortized_ms` in every rollup. Refer to [Sum amortized_ms, not duration_ms](#sum-amortized-ms-not-duration-ms).

`page_summaries_dataframe()` returns one row per page, with the `document_id`, `source_path`, `page_number`, `source_id`, `total_ms`, `wall_ms`, and `span_count` columns. It also adds one `category.{name}_ms` column per span category and one `operator.{name}` column per operator, so you can compare pages without touching the span table.

```python
from nemo_retriever.common.tracing import load_traces, page_summaries_dataframe

pages = page_summaries_dataframe(load_traces("traces/"))
pages.nlargest(10, "total_ms")
```

`load_traces()` raises `TraceFileError` when a path matches nothing or when a file is not a trace artifact.


## Trace file schema { #trace-file-schema }

NeMo Retriever Library writes one file per document, named `{document_id}.trace.json`. With gzip compression, the name is `{document_id}.trace.json.gz`. The current `schema_version` is `1.0`.

Unless the run supplies a document ID, the library derives one from the source path as a filesystem-safe slug of the file name followed by a hyphen and the first eight hexadecimal characters of the SHA-1 digest of the full path. For example, `report.pdf` becomes something like `report.pdf-1a2b3c4d`. The suffix keeps files from colliding when two directories hold the same file name.

The following example shows the structure of a trace file with the span and summary arrays abbreviated:

```json
{
  "schema_version": "1.0",
  "nemo_retriever": { "version": "...", "git_sha": "...", "build_date": "...", "full_version": "..." },
  "run": { "run_id": "...", "run_mode": "batch", "started_at": null, "completed_at": "...", "otel_trace_id": null },
  "document": { "document_id": "report.pdf-1a2b3c4d", "source_path": "data/report.pdf",
                "source_type": "pdf", "page_count": 12, "status": "completed",
                "error_span_count": 0, "wall_ms": 8421.3,
                "started_at": "...", "completed_at": "..." },
  "pipeline": { "operators": ["extract", "embed"], "params": { "page_trace_detail": "full" } },
  "models": [ { "model_key": "ocr", "name": "nvidia/nemoretriever-ocr-v1", "version": "1.3.0",
                "backend": "nim-grpc", "endpoint": "..." } ],
  "document_summary": { "total_ms": 8421.3, "wall_ms": 9102.7, "span_count": 143,
                        "by_operator": [ { "operator": "PDFExtractionActor", "calls": 12,
                                           "total_ms": 5210.4, "self_ms": 812.0,
                                           "pct_of_total": 61.87 } ],
                        "by_category": { "operator_ms": 8421.3, "network_ms": 4903.1,
                                         "gpu_ms": 1220.6, "cpu_ms": 640.2, "io_ms": 88.4 },
                        "by_model": { "ocr": 4903.1 } },
  "page_summaries": [ { "page_number": 1, "source_id": "data/report.pdf_1", "total_ms": 705.2,
                        "wall_ms": 733.9, "span_count": 11,
                        "by_operator": { "PDFExtractionActor": 434.2 },
                        "by_category": { "operator_ms": 705.2, "network_ms": 408.6,
                                         "gpu_ms": 101.7, "cpu_ms": 53.4, "io_ms": 7.4 } } ],
  "spans": [ { "document_id": "report.pdf-1a2b3c4d", "source_path": "data/report.pdf",
               "page_number": 1, "source_id": "data/report.pdf_1",
               "span_id": "9f3c1a", "parent_span_id": null, "name": "PDFExtractionActor",
               "category": "operator", "operator": "PDFExtractionActor", "model_key": null,
               "start_ms": 1757345512345.6, "end_ms": 1757345512466.0,
               "duration_ms": 120.4, "self_ms": 12.0,
               "amortized_ms": 15.05, "amortized_self_ms": 1.5, "page_fanout": 8,
               "batch_size": 8, "status": "ok", "error": null, "worker": "host:1234",
               "attrs": {} } ]
}
```

### Top-level objects { #top-level-objects }

The following table describes each top-level key.

| Key | Description |
|-----|-------------|
| `schema_version` | The trace format version. The current value is `1.0`. |
| `nemo_retriever` | The library `version`, `git_sha`, `build_date`, and `full_version` that produced the trace. |
| `run` | The `run_id`, the `run_mode` that produced the trace (`inprocess`, `batch`, or `service`), the `started_at` and `completed_at` timestamps, and `otel_trace_id`. `otel_trace_id` is reserved for OpenTelemetry correlation and is currently always `null`. |
| `document` | Document identity and outcome. Includes `document_id`, `source_path`, `source_type` (the lowercase file extension, or `unknown`), `page_count`, `status` (`completed` or `error`), `error_span_count`, `wall_ms`, `started_at`, and `completed_at`. |
| `pipeline` | The ordered `operators` list of configured stage names, such as `extract` and `embed`, and the `params` that configured tracing. These are stage names, not the runtime operator names in `spans[].operator`. |
| `models` | One entry per model that the run used, with `model_key`, `name`, `version`, `backend`, and `endpoint`. Values are `null` when the client did not report them. |
| `document_summary` | Document rollups: `total_ms`, `wall_ms`, `span_count`, `by_operator`, `by_category`, and `by_model`. |
| `page_summaries` | One rollup per page, with `page_number`, `source_id`, `total_ms`, `wall_ms`, `span_count`, `by_operator`, and `by_category`. |
| `spans` | Every span the run recorded, repeated once per page that the span covered, sorted by page number and then start time. |

`document_summary.by_operator` is a list of objects sorted by descending `total_ms`, where `pct_of_total` is the operator share of `total_ms`. Inside `page_summaries`, `by_operator` and `by_category` are mappings from name to milliseconds, sorted by descending value. Category keys carry an `_ms` suffix, for example `network_ms`.

### Span fields { #span-fields }

The following table describes each span field.

| Field | Description |
|-------|-------------|
| `document_id`, `source_path` | The document that the span belongs to. |
| `page_number`, `source_id` | The page that the span is charged to. |
| `span_id`, `parent_span_id` | Span identity and nesting. A top-level span has a `parent_span_id` of `null`. |
| `name` | The span label, such as the operator class name or the instrumented call. |
| `category` | One of `operator`, `network`, `gpu`, `cpu`, or `io`. Refer to [Span categories](#span-categories). |
| `operator` | The operator that the span ran under. |
| `model_key` | The key of the model that the span invoked, or `null` when no model is involved. Keys match the entries in `models`. |
| `start_ms`, `end_ms` | The wall-clock span boundaries, as Unix epoch milliseconds. |
| `duration_ms` | The measured wall time of the span, including child spans. |
| `self_ms` | The span time excluding child spans. |
| `amortized_ms` | `duration_ms` divided by `page_fanout`. Sum this field in every rollup. |
| `amortized_self_ms` | `self_ms` divided by `page_fanout`. |
| `page_fanout` | The number of pages that the span was charged across. |
| `batch_size` | The number of pages or items that the underlying operation was handed. |
| `status`, `error` | The span outcome, either `ok` or `error`, and the error detail when the span failed. |
| `worker` | The `host:pid` label of the process that ran the span. |
| `attrs` | Span-specific attributes, such as `endpoint`, `protocol`, `http_status`, `dpi`, or `device`. Spans that roll up a repeated call also carry `calls` and `rolled_up`. |

Spans that a batch shares are charged to every page in that batch, so `page_fanout` is greater than 1. Spans recorded before the document was split into pages have no page of their own, so they are charged across every page that the document produced.

Hot helpers that run many times per page, such as image encoding, are rolled up into one span per operator invocation. Those spans report the summed duration and a `calls` attribute rather than one record per call, which keeps trace files small enough to stay useful.


## Limitations { #limitations }

Consider the following limitations when you interpret a trace:

- Spans record entry and exit wall time only. GPU work is timed around a launch that may be asynchronous, so a `gpu.{model}` span is not a device-execution measurement. Use [Nsight Systems](https://developer.nvidia.com/nsight-systems) against the NVTX ranges for kernel-level attribution.
- Pages that page-range filtering excludes, through the `start_page` or `end_page` split parameters, never enter the pipeline. Those pages are absent from the trace rather than reported as zero cost.
- Page-level `wall_ms` covers the span time range observed for that page. It can overlap with other pages when stages run concurrently. Use `total_ms` for additive comparisons.
- Tracing failures are non-fatal. If aggregation or the trace write fails, the library logs a warning and returns the ingest results without traces.


## Related Topics { #related-topics }

- [Performance Guide](performance_guide.md)
- [Troubleshoot NeMo Retriever Library](troubleshoot.md)
- [Page tracing parameters](nemo-retriever-api-reference.md#page-tracing-parameters)
- [Configure Ray Logging](ray-logging.md)
- [Environment Variables](environment-config.md)
- [Python API guide](nemo-retriever-api-reference.md)
- [CLI ingest options](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docs/cli/README.md)
