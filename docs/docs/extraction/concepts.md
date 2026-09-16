# Concepts

These terms appear throughout NeMo Retriever Library documentation.

## Job { #job }

An **ingestion job** is a unit of work you run on input content (documents, audio, video, and other supported types). Submit jobs through any of these supported entry points:

- **Python API** — `Ingestor` task chains such as `.extract(...)`. Library and batch modes run ingest in-process. Refer to the [Python API guide](nemo-retriever-api-reference.md).
- **`retriever ingest` CLI** — `retriever ingest` and `retriever ingest batch`. Refer to the [CLI reference](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/docs/cli).

Default tasks target strong recall; customize behavior with task keyword arguments (including chunking and splitting on `.extract()`) or custom UDF-style operations. For UDFs and other extension paths, refer to [Customize & extend](customize-extend.md). Results are structured metadata and annotations (Ray Dataset, pandas `DataFrame`, or similar).

## Collection { #collection }

A **collection** is a named LanceDB table that holds ingested document embeddings. Python and CLI ingest write to a table such as `nemo-retriever`. Query that table with `retriever query` or `Retriever`. Callers choose the table name; they do not use a separate HTTP collection catalog. Refer to [Vector databases](vdbs.md).

## Pipeline and tasks { #pipeline-and-tasks }

NeMo Retriever Library does **not** run one static pipeline on every document. You configure **tasks** such as parsing, chunking, embedding, storage, and filtering per job. For UDFs, custom graph stages, and other extension paths, refer to [Customize & extend](customize-extend.md).

## Extraction metadata { #extraction-metadata }

Output is a **Ray Dataset** (Ray Data) or **pandas** `DataFrame` listing extracted objects (text regions, tables, images, and so on), processing notes, and timing or trace data. Field-level detail is in the [metadata reference](content-metadata.md).

## Embeddings and retrieval { #embeddings-and-retrieval }

Optionally, the library can compute **embeddings** for extracted content and store vectors in [LanceDB](https://lancedb.com/) for downstream semantic search in your application. For upload and retrieval APIs, refer to [Vector databases](vdbs.md). For multimodal (VLM) embedding options, refer to [Multimodal embeddings (VLM)](embedding.md). For iterative, tool-driven retrieval over that index, refer to [Agentic retrieval (concept)](agentic-retrieval-concept.md) and [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).

## Chunking { #chunking }

Chunking is built into the `.extract()` task and depends on **content type**:

- **PDF, DOCX, and PPTX** — Text is grouped using built-in **page** boundaries (one chunk per page where the format has pages).
- **Plain text (`.txt`) and HTML** — Formats without natural page breaks are split into segments of **1024 tokens** by default, using the revision-pinned [Llama Nemotron Embed VL 1B v2 tokenizer](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) so chunk boundaries stay aligned with the default embedding model. Refer to [Token-based splitting](#token-based-splitting) and [Environment variables](environment-config.md) for overrides and other runtimes.
- **Audio and video** — Media is split into **segments** for decoding and ASR using ffmpeg-based rules (configurable **size**, **time**, or **frame** split modes in the media chunking stage). With the Parakeet ASR path, you can optionally emit **sentence-like segments** by passing `asr_params=ASRParams(segment_audio=True)` to `.extract_audio(...)`. Refer to [Speech and audio extraction](audio-video.md#speech-and-audio-extraction) for the import and a runnable example.

For PDF parallelism before Ray processing (large files), refer to [PDF pre-splitting for parallel ingest](nemo-retriever-api-reference.md#pdf-pre-splitting-for-parallel-ingest).

### Token-based splitting { #token-based-splitting }

Token-based splitting uses the revision-pinned tokenizer for the default embedding model (`nvidia/llama-nemotron-embed-vl-1b-v2`) with configurable `max_tokens` and `overlap_tokens`. For graph ingest (`create_ingestor(run_mode="inprocess")` or `create_ingestor(run_mode="batch")`), set those values on `.extract(split_config={"text": {"max_tokens": ..., "overlap_tokens": ...}})`, or omit `split_config` to use default text segmentation for unstructured text. The base library install includes the tokenizer Python dependencies; pre-populate the Hugging Face cache before offline use. For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md).

## Deployment modes { #deployment-modes }

- **Library mode** — Run the Python package and CLI in your environment; refer to [Deployment options](deployment-options.md).
- **Notebooks** — [Jupyter examples](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/README.md) for experimentation and RAG demos.

For a concise comparison, refer to [Deployment options](deployment-options.md).
