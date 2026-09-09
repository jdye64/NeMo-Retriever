# Page Tracing

Page tracing records how long [NeMo Retriever Library](overview.md) spends on every page of every document that it ingests. Use it to find pipeline bottlenecks, compare detail levels and worker counts, and identify which operator, model, or endpoint dominates a run.

Page tracing is a performance-analysis tool. It does not change extraction output, and it does not replace the row-level error payloads described in [Troubleshoot NeMo Retriever Library](troubleshoot.md).


## How page tracing works { #how-page-tracing-works }

Each page accumulates spans as it moves through the pipeline. A span is one timed operation applied to a page or to a batch of pages. NeMo Retriever Library records one span per operator per page, so a trace tells you which pipeline stage a page spent its time in, and which model versions ran.

Pages also accumulate work counters, which record what each page contributed to a stage, such as the number of regions it sent to OCR. Counters are recorded per page whatever the batch size, so they stay meaningful when a run batches many pages into one model call and the timings become batch averages. Refer to [Find the pages that cost the most](#find-the-pages-that-cost-the-most).

When a document finishes, NeMo Retriever Library aggregates the spans for that document into one JSON artifact that retains the full per-page detail. The library writes one file per document. Refer to [Trace file schema](#trace-file-schema).

Page tracing works in all three run modes: `inprocess`, `batch`, and `service`. In service mode, the worker that runs the pipeline records the trace and the gateway returns it to the client.


## Choose a detail level { #choose-a-detail-level }

The detail level controls how much the pipeline records and how much overhead tracing adds. The following table lists the supported values.

| Value | What it records | Overhead |
|-------|-----------------|----------|
| `off` | Nothing. Tracing is disabled. | None. |
| `operator` | One span per operator per page. This is the default in the `inprocess` and `batch` run modes. | Negligible. |

In service mode the default is `off` instead, because the trace has to be transmitted rather than kept in the local process. Refer to [Read page traces from the service API](#read-page-traces-from-the-service-api).

Page tracing attributes time to pipeline stages, not to individual calls inside them. It tells you that a page spent 4 seconds in the OCR stage and which OCR model version ran, but not how that 4 seconds divided between the network round trip and the GPU forward pass. Use [Nsight Systems](https://developer.nvidia.com/nsight-systems) against the NVTX ranges that the library already emits when you need that level of attribution.

Every span records the wall time between entering and leaving the operator, and nothing more. Tracing never synchronizes CUDA and never drives a profiler, so turning it on does not change how the pipeline executes.

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
| `--page-trace-detail LEVEL` | Set the detail level. Valid values are `off` and `operator`. The default is `operator`. This option also reads `NEMO_RETRIEVER_PAGE_TRACE_DETAIL`. |

The following command runs a batch ingest and writes traces to `traces/`. Replace `/path/to/your/pdfs` with a directory of PDF files that you supply.

```bash
retriever ingest batch /path/to/your/pdfs --page-trace-dir traces/
```


## Enable page tracing in Python { #enable-page-tracing-in-python }

Every ingestor that `create_ingestor()` returns provides the `.save_page_traces()` fluent method. It persists one JSON file per document to a directory that you choose, in every run mode.

The following example runs an in-process ingest and writes the trace files to `traces/`:

```python
from nemo_retriever import create_ingestor

ingestor = (
    create_ingestor(run_mode="inprocess")
    .files("data/report.pdf")
    .extract()
    .save_page_traces(output_directory="traces/")
)
results = ingestor.ingest()
```

`.save_page_traces()` accepts a `compression` argument. The default is `None`, which writes plain JSON. Pass `compression="gzip"` to write gzip-compressed files instead. Any other value raises a `ValueError`.

To receive the traces in memory rather than reading them back from disk, pass `return_page_traces=True` to `.ingest()`. Each trace is a plain dictionary matching [Trace file schema](#trace-file-schema).

```python
results, page_traces = ingestor.ingest(return_page_traces=True)
```

You can combine `return_page_traces=True` with `.save_page_traces(...)` to both write files and inspect traces in the same process.

The traces from the most recent `.ingest()` call are also available on the `page_traces` property, in every run mode. Reading the property does not change the shape of the `.ingest()` return value, which is convenient when you want the results unpacked normally:

```python
results = ingestor.ingest()
for trace in ingestor.page_traces:
    print(trace["document"]["source_path"], trace["document_summary"]["total_ms"])
```

The property returns a copy of the list, and each `.ingest()` call resets it. It holds whatever traces the run produced, so in service mode it is empty unless the run opted into tracing by calling `.save_page_traces()`, passing `return_page_traces=True`, or setting `page_trace_detail` explicitly.

The following table lists the page-tracing parameters on `IngestExecuteParams` and `.ingest()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page_trace_detail` | `"off"` or `"operator"` | Unset, which resolves to `"operator"` | Whether the pipeline records per-page timings. |
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

`summary` is the default subcommand, so passing paths directly to `retriever trace` runs it. The summary reports the document and page counts, total traced time, time per page, the library version, the run mode, a ranking by operator, the models and versions that ran, the slowest pages, and the heaviest pages by work volume.

Each row of the slowest-pages table shows whether that page's timings were measured or amortized. When any of them are amortized, the summary explains that those timings are batch averages and points you to the work-volume ranking instead. Refer to [Find the pages that cost the most](#find-the-pages-that-cost-the-most).

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

### Inspect a single page { #inspect-a-single-page }

The `page` subcommand prints the span waterfall for one page, with the duration, the amortized time, and the batch size of each span. It also lists the work counters recorded for the page and states whether the page's timings were measured on it or amortized across a batch. Use it after the summary identifies a slow or heavy page.

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

Spans can nest when one stage runs another inside it. `duration_ms` includes the time spent in child spans, and `self_ms` excludes it. Rollups such as `by_operator` count only the operator's own span, so a nested child is never charged to its operator twice.

Every span carries a `kind` field that distinguishes the two. The value is `operator` for the span that represents a pipeline stage, and `child` for a span opened inside one.


## Find the pages that cost the most { #find-the-pages-that-cost-the-most }

Per-page timings are only as sharp as the batches a run used, so read this section before you conclude that one page is slower than another.

### Per-page timings depend on the run mode { #per-page-timings-depend-on-the-run-mode }

Amortizing divides a batch cost evenly, which assumes every page in the batch cost the same. When a span covered one page that assumption is exact. When a span covered 200 pages, all 200 report the identical amortized value and the timings cannot tell you which page was expensive.

How pages are batched depends on the run mode.

| Run mode | Pages per operator call | Per-page timings |
|----------|-------------------------|------------------|
| `batch` | One, because the Ray Data executor defaults to `batch_size=1`. | Measured for each page. |
| `inprocess` | Every page of the document, in a single call per operator. | Amortized, and identical for all pages of a document. |
| `service` | Every page of the document, because the worker runs the pipeline in process. | Amortized, and identical for all pages of a document. |

Each page summary reports which case it is in, so you never have to infer it:

- `timing_source` is `measured` when the page's timings came from spans that covered only that page, and `amortized` when at least one span was shared with other pages.
- `max_page_fanout` is the largest number of pages that any of the page's operator spans covered. A value of `1` means every timing was measured on that page alone.

To get measured per-page timings, ingest with `run_mode="batch"`. Raising `batch_size` improves GPU throughput but coarsens per-page attribution, so leave it at `1` while you are hunting a slow page.

### Work counters identify costly pages in any run mode { #work-counters-identify-costly-pages-in-any-run-mode }

The heavy stages batch their inference across pages, so no timing can isolate one page's share of a batched model call. Work counters solve this from the other direction: they count what each page contributed, individually, whatever the batch size. A page that yields 34 OCR crops costs far more than one that yields 2, and that stays true no matter how the batch was cut.

Each page summary carries a `work` object whose keys are namespaced by the operator that recorded them. The library records the following counters.

| Counter | Recorded by | What it measures |
|---------|-------------|------------------|
| `rendered_megapixels` | The PDF extraction stage | The rasterized area of the page. This sets the floor on the cost of every later image stage. |
| `detections` | The page element detection stage | The number of detected elements. This drives how much cropping, table, and OCR work the page causes downstream. |
| `table_crops` | The table structure stage | The number of table regions cropped from the page. |
| `crops` | The OCR stage | The number of regions sent to OCR. |
| `parse_crops` | The Nemotron Parse stage | The number of regions sent to Parse. |
| `text_chars` | The text embedding stage | The number of characters embedded for the page. |

`document_summary.work_totals` sums each counter across the document.

### Rank pages by work volume { #rank-pages-by-work-volume }

`retriever trace` prints a **Heaviest pages by work volume** table alongside the slowest-pages table, and warns you when the timings it reported are batch averages.

```console
$ retriever trace traces/
```

The table ranks pages by a `work_index`, which is a page's heaviest counter expressed as a multiple of the per-page average for that counter. A value of `1.0` is a typical page, and `2.8` means the page carries almost three times the average load in whichever counter is most skewed for it. The index compares each counter only against itself, because the counters use incomparable units and no sum across them would be meaningful.

Once a page stands out, inspect it directly. The page view lists the counters recorded for that page and labels whether its timings were measured or amortized.

```console
$ retriever trace page traces/ --page 47 --document report.pdf-1a2b3c4d
```


## Load traces with pandas { #load-traces-with-pandas }

The `nemo_retriever.common.tracing` module reads saved trace files and flattens them into pandas `DataFrame` objects. Use `load_traces()` to read a file, a directory, a glob, or a sequence of any of those. Then use `spans_dataframe()` to build a span table, or `page_summaries_dataframe()` to build a table of per-page rollups.

The following example ranks operators by total amortized time across a whole run:

```python
from nemo_retriever.common.tracing import load_traces, spans_dataframe

traces = load_traces("traces/")
spans = spans_dataframe(traces)
spans.groupby("operator").amortized_ms.sum().sort_values(ascending=False)
```

`spans_dataframe()` returns one row per span per page. The columns are the span fields listed in [Span fields](#span-fields), plus `run_id`, `run_mode`, and `library_version` carried down from the trace file. Group by `page_number` to compare pages, or by `operator` to rank pipeline stages. Sum `amortized_ms` in every rollup. Refer to [Sum amortized_ms, not duration_ms](#sum-amortized-ms-not-duration-ms).

`page_summaries_dataframe()` returns one row per page, with the `document_id`, `source_path`, `page_number`, `source_id`, `total_ms`, `wall_ms`, `span_count`, `max_page_fanout`, and `timing_source` columns. It also adds one `operator.{name}` column per operator and one `work.{operator}.{counter}` column per work counter, so you can compare pages without touching the span table.

```python
from nemo_retriever.common.tracing import load_traces, page_summaries_dataframe

pages = page_summaries_dataframe(load_traces("traces/"))
pages.nlargest(10, "total_ms")
```

Ranking by `total_ms` only separates pages when their timings were measured. Check `timing_source` first, and rank by a work counter when the timings are batch averages. Refer to [Find the pages that cost the most](#find-the-pages-that-cost-the-most).

```python
pages = page_summaries_dataframe(load_traces("traces/"))

if (pages["timing_source"] == "amortized").any():
    # Timings are batch averages here, so rank by what each page contributed.
    print(pages.nlargest(10, "work.OCRActor.crops"))
else:
    print(pages.nlargest(10, "total_ms"))
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
  "pipeline": { "operators": ["extract", "embed"], "params": { "page_trace_detail": "operator" } },
  "models": [ { "model_key": "ocr", "name": "nvidia/nemoretriever-ocr-v1", "version": "1.3.0",
                "backend": "nim-grpc", "endpoint": "..." } ],
  "document_summary": { "total_ms": 8421.3, "wall_ms": 9102.7, "span_count": 143,
                        "by_operator": [ { "operator": "PDFExtractionActor", "calls": 12,
                                           "total_ms": 5210.4, "self_ms": 812.0,
                                           "pct_of_total": 61.87 } ],
                        "by_model": { "ocr": 4903.1 },
                        "work_totals": { "TextEmbedActor.text_chars": 20401.0,
                                         "OCRActor.crops": 84.0 } },
  "page_summaries": [ { "page_number": 1, "source_id": "data/report.pdf_1", "total_ms": 705.2,
                        "wall_ms": 733.9, "span_count": 11,
                        "max_page_fanout": 8, "timing_source": "amortized",
                        "by_operator": { "PDFExtractionActor": 434.2 },
                        "work": { "TextEmbedActor.text_chars": 1840.0,
                                  "OCRActor.crops": 7.0 } } ],
  "spans": [ { "document_id": "report.pdf-1a2b3c4d", "source_path": "data/report.pdf",
               "page_number": 1, "source_id": "data/report.pdf_1",
               "span_id": "9f3c1a", "parent_span_id": null, "name": "PDFExtractionActor",
               "kind": "operator", "operator": "PDFExtractionActor", "model_key": null,
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
| `document_summary` | Document rollups: `total_ms`, `wall_ms`, `span_count`, `by_operator`, `by_model`, and `work_totals`. |
| `page_summaries` | One rollup per page, with `page_number`, `source_id`, `total_ms`, `wall_ms`, `span_count`, `max_page_fanout`, `timing_source`, `by_operator`, and `work`. Refer to [Find the pages that cost the most](#find-the-pages-that-cost-the-most). |
| `spans` | Every span the run recorded, repeated once per page that the span covered, sorted by page number and then start time. |

`document_summary.by_operator` is a list of objects sorted by descending `total_ms`, where `pct_of_total` is the operator share of `total_ms`. Inside `page_summaries`, `by_operator` is a mapping from operator name to milliseconds, sorted by descending value.

### Span fields { #span-fields }

The following table describes each span field.

| Field | Description |
|-------|-------------|
| `document_id`, `source_path` | The document that the span belongs to. |
| `page_number`, `source_id` | The page that the span is charged to. |
| `span_id`, `parent_span_id` | Span identity and nesting. A top-level span has a `parent_span_id` of `null`. |
| `name` | The span label, normally the operator class name. |
| `kind` | Either `operator` for a pipeline stage span, or `child` for a span opened inside one. |
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
| `attrs` | Span-specific attributes, when the span recorded any. |

Spans that a batch shares are charged to every page in that batch, so `page_fanout` is greater than 1. Spans recorded before the document was split into pages have no page of their own, so they are charged across every page that the document produced.


## Limitations { #limitations }

Consider the following limitations when you interpret a trace:

- Tracing attributes time to pipeline stages, not to the individual network, GPU, or dependency calls inside them. Use [Nsight Systems](https://developer.nvidia.com/nsight-systems) against the NVTX ranges for finer attribution.
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
