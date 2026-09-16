# Workflow: Agentic retrieval

Use this workflow after you have ingested documents into a LanceDB table. Agentic retrieval does not ingest files. It queries the same table, embedding model, and storage flags as one-pass `retriever query`.

**Agentic retrieval** runs a large language model (LLM) Reason and Act (ReAct) loop: the agent issues several retrieval sub-queries, fuses candidates with reciprocal rank fusion, and selects a final document ranking. **One-pass retrieval** sends a single dense or hybrid query and returns text-enriched chunk hits. For the concept distinction, refer to [Agentic retrieval (concept)](agentic-retrieval-concept.md).

## Query with the CLI { #query-with-the-cli }

`retriever query --agentic` is the one-shot CLI path. It searches the LanceDB table built by `retriever ingest`. Reuse the same `--lancedb-uri`, `--table-name`, and embedding model that you used at ingest. When `--embed-model-name` is omitted, agentic retrieval uses the selected table's model.

### Local in-process vLLM { #local-in-process-vllm }

The CLI and NRB agentic benchmark paths default to an in-process local vLLM agent
LLM. If you omit `--agentic-llm-model` and `--agentic-invoke-url`, the library
loads `nemotron-8b` (`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`) on the local CUDA
host. This requires a Linux CUDA GPU and the `[local]` extra.

GPU placement follows process-level vLLM behavior. Set `CUDA_VISIBLE_DEVICES` before you start the command.

```bash
CUDA_VISIBLE_DEVICES=0 retriever query "find documents about parser behavior" --agentic
```

The larger `super-49b` profile is also supported. Pass `--agentic-local-tensor-parallel-size 2` with two visible GPUs for that profile.

```bash
CUDA_VISIBLE_DEVICES=0,1 retriever query "find documents about parser behavior" \
  --agentic \
  --agentic-llm-model super-49b \
  --agentic-local-tensor-parallel-size 2
```

Custom in-process LLMs are not supported. The agent loop depends on OpenAI-style tool-call messages. Use an OpenAI-compatible endpoint for custom models.

### Remote OpenAI-compatible NIM or hosted endpoint { #remote-openai-compatible-endpoint }

Providing `--agentic-invoke-url` routes the agent to that remote chat-completions endpoint. `--agentic-llm-model` is required on the remote path and is sent as the remote model ID. The LLM client defaults to `callable`, which calls the endpoint over the shared chat-completions HTTP client and needs no extra LLM SDK.

Self-hosted NIM or a local OpenAI-compatible server:

```bash
retriever query "find documents about parser behavior" \
  --agentic \
  --agentic-llm-model nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --agentic-invoke-url http://localhost:9000/v1/chat/completions
```

NVIDIA-hosted Build endpoint (requires `NVIDIA_API_KEY`; `NGC_API_KEY` is the fallback):

```bash
retriever query "find documents about parser behavior" \
  --agentic \
  --agentic-llm-model nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --agentic-invoke-url https://integrate.api.nvidia.com/v1/chat/completions
```

`--agentic-local-tensor-parallel-size` is ignored when `--agentic-invoke-url` is set. For hosted model IDs, refer to [Default NVCF endpoints](prerequisites-support-matrix.md#default-nvcf-endpoints). For key setup, refer to [Authentication and API keys](api-keys.md).

This self-hosted NIM configuration gap does not apply to NVIDIA-hosted Build endpoints. A self-hosted Super-49B NIM rejects tool-call requests until you add the passthrough arguments. Refer to [Self-hosted Super-49B](#self-hosted-super-49b).

### CLI options { #cli-options }

The following options apply only with `--agentic`. For the full flag list, refer to [Agentic retrieval](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docs/cli/README.md#agentic-retrieval) in the CLI reference.

| Option | Default | Notes |
|---|---|---|
| `--agentic-llm-model` | `nemotron-8b` when no invoke URL is set | Local profile alias (`nemotron-8b` or `super-49b`) or remote model ID when `--agentic-invoke-url` is set. |
| `--agentic-invoke-url` | unset (local vLLM) | OpenAI-compatible `/v1/chat/completions` endpoint. Required together with `--agentic-llm-model` for remote runs. |
| `--agentic-local-tensor-parallel-size` | `1` | vLLM `tensor_parallel_size` for the in-process agent LLM. Set to `2` for local `super-49b`. Ignored when `--agentic-invoke-url` is set. |
| `--agentic-react-max-steps` | `50` | Maximum ReAct loop iterations. |
| `--agentic-reasoning-effort` | `high` | Forwarded on OpenAI-compatible agent LLM calls. Ignored by the local adapter. |
| `--include-usage` | off | Print an object with `hits` and provider-reported LLM `usage` instead of the default hits list. |

Embedding credentials use `NVIDIA_API_KEY` or `NGC_API_KEY` when you call a remote embedding endpoint. The CLI also reuses `--embed-invoke-url`, `--top-k`, `--lancedb-uri`, and `--table-name` from standard retrieval.

## Self-hosted Super-49B { #self-hosted-super-49b }

Use this path when the agent LLM is a self-hosted Super-49B NIM rather than local in-process vLLM or an NVIDIA-hosted Build endpoint.

A Super-49B NIM used only for one-shot `Retriever.answer()` text generation does not require tool calling. Agentic retrieval is a separate CLI path and stays off unless you pass `--agentic`.

The agentic ReAct loop sends OpenAI-style tool-call messages with `tool_choice=auto`. A self-hosted vLLM-backed Super-49B NIM rejects those requests with HTTP 400 unless you also pass `--enable-auto-tool-choice` and `--tool-call-parser llama3_json` in `NIM_PASSTHROUGH_ARGS`.

You can reuse the same Super-49B NIM for agentic retrieval after you add those arguments. One-shot answer generation continues to work.

Confirm the passthrough arguments on the running NIM include `--enable-auto-tool-choice` and `--tool-call-parser llama3_json`. Then run the remote command in [Remote OpenAI-compatible NIM or hosted endpoint](#remote-openai-compatible-endpoint). Point `--agentic-invoke-url` at the NIM chat-completions URL and set `--agentic-llm-model` to `nvidia/llama-3.3-nemotron-super-49b-v1.5`. Reuse the same embedding invoke URL and model name that you used at ingest.

## Result contract { #result-contract }

Every agentic Retriever run writes a lightweight Agent Trajectory Interchange
Format (ATIF) JSON trajectory under `./agentic-traces` by default. The
trajectory bounds observation content to keep the file lightweight. These
traces are not printed in CLI output. If a trace cannot be persisted,
retrieval continues and emits a warning.

One-pass retrieval returns text-enriched chunk hits. Agentic retrieval ranks documents. Each selected document is rehydrated from the retrieval hop that returned it. CLI output then uses a different JSON shape.

CLI `retriever query` without `--agentic` projects each hit to five fields: `modality`, `page_number`, `score`, `source`, and `text`. CLI `retriever query --agentic` does not use that projection. It prints the internal hit dictionary plus these ranking annotations:

- `doc_id` — the document identifier the agent selected.
- `rank` — the position in the final ranking.
- `result_source` — `final_results`, `rrf`, or `selection_agent`, depending on which stage produced the ranked ID.

`modality` and `score` exist only on the dense CLI path. Agentic CLI objects can include internal fields such as `content_type`, `_distance`, `metadata`, `path`, `pdf_basename`, `pdf_page`, and `source_id` when the retrieval hop returned them.

When the agent names a document that no retrieval hop returned, the CLI object contains only `doc_id`, `rank`, and `result_source`. Classic hit keys are absent, not present with null values.

CLI `retriever query --agentic` keeps its default output as a JSON list of those
hits. Add `--include-usage` to return a JSON object that contains `hits` and
provider-reported LLM `usage`:

```bash
retriever query "find documents about parser behavior" \
  --agentic \
  --include-usage
```

```json
{
  "hits": [
    {
      "doc_id": "parser-guide",
      "rank": 1,
      "result_source": "final_results"
    }
  ],
  "usage": {
    "input_tokens": 1250,
    "cache_tokens": 400,
    "output_tokens": 184,
    "total_tokens": 1434
  }
}
```

The `usage` object reports the exact token counts returned by the LLM provider.
It contains `input_tokens`, observed cache-read `cache_tokens`, `output_tokens`,
and `total_tokens`. `cache_tokens` is `null` when no stage reports cache usage.
When available, `stages` preserves the provider-reported breakdown for the
ReAct and final-selection calls, including cache-creation counters. When a
provider reports uncached, cache-creation, and cache-read input separately,
`input_tokens` includes all three counters. Cache tokens are therefore not
added again when calculating `total_tokens`. If the provider does not report
usage, the response sets `usage` to `null`.
`--include-usage` applies only to agentic queries. Classic `retriever query`
output is unchanged.

## Failure and retry behavior { #failure-and-retry-behavior }

Operational failures from the agent LLM or retrieval tool, including embedding, vector database, and reranker endpoint failures, terminate the query with an error instead of returning a successful empty result.

An HTTP `400` from the chat-completions NIM with `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set` means the self-hosted endpoint is not tool-call ready. Refer to [Self-hosted Super-49B](#self-hosted-super-49b). The CLI then exits with `Agentic retrieval failed (llm_call_failed)`.

## Limitations and resource requirements { #limitations-and-resource-requirements }

- Local in-process agent LLMs are limited to the tested `nemotron-8b` and `super-49b` profiles. Custom in-process models require an OpenAI-compatible endpoint instead.
- Local CLI and harness runs need a CUDA GPU host and the `[local]` extra. `super-49b` needs two visible GPUs and `--agentic-local-tensor-parallel-size 2`.
- A self-hosted Super-49B NIM requires the tool-call passthrough arguments before agentic retrieval works. Enabling one-shot answer generation does not configure `--agentic`.
- Agentic ranking is document-level. Rehydrated hits include chunk `text` when a retrieval hop returned the document. Otherwise load the source document by `doc_id`.
- On the CLI, `--rerank` applies to each agent retrieve hop.

## Related Topics { #related-topics }

- [Agentic retrieval (concept)](agentic-retrieval-concept.md)
- [Semantic retrieval](vdbs.md#semantic-retrieval)
- [Metadata and filtering](vdbs.md#metadata-and-filtering)
- [Evaluate on your data](evaluate-on-your-data.md)
- [Authentication and API keys](api-keys.md)
- [CLI reference: Agentic retrieval](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/docs/cli/README.md#agentic-retrieval)
- [Release notes](releasenotes.md#retrieval-and-rag)
