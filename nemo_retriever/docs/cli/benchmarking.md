# Benchmarking with the Retriever CLI

End-to-end experiments are maintained in the [NeMo Retriever Benchmark (NRB)
repository](https://gitlab-master.nvidia.com/charlesb/nemo-retriever-benchmark/).
The product CLI retains internal stage micro-benchmarks for focused development
measurements. For product workflows on your own inputs, use
`retriever ingest` and `retriever query`.

## Stage Micro-Benchmarks

`retriever benchmark` measures individual actors rather than an end-to-end
Retriever result. It remains callable for development compatibility but is
hidden from root help.

```bash
retriever benchmark --help
retriever benchmark split --help
retriever benchmark extract --help
retriever benchmark audio-extract --help
retriever benchmark page-elements --help
retriever benchmark ocr --help
retriever benchmark all --help
retriever benchmark fused-compare --help
```

The following example extracts from a directory of PDF files that you supply.

```bash
retriever benchmark extract /path/to/your/pdfs \
  --pdf-extract-batch-size 8 \
  --pdf-extract-actors 4
```

Stage commands report rows per second, or chunk rows per second for audio. They
do not produce the NRB artifact contract and should not be used as retrieval
quality evidence.

## Compare Fused Against Staged Extraction

`retriever benchmark fused-compare` runs the same graph over the same PDFs
twice, once with `--method fused` and once with a staged method, and reports the
two runtimes side by side. Unlike the stage commands, this one measures the full
pipeline including embedding.

```bash
retriever benchmark fused-compare run \
  --pdf-dir /path/to/your/pdfs \
  --iterations 3 \
  --output-json fused-compare.json
```

The command requires a local GPU and the optional `nemo_retriever_fused`
package, which the "Run fused GPU-resident PDF extraction" section of the
NeMo Retriever API reference describes.

Three behaviors matter when you read the output.

Weights are warmed before the timed passes, and the load time is reported on
its own line. Every local GPU actor resolves its model through the warmup
registry, and nothing else caches models between graph builds, so an unwarmed
comparison measures checkpoint loading instead of compute.

The first pass after warmup is reported separately from the timed passes,
because it still pays CUDA kernel autotuning and allocator growth.

The report compares output as well as runtime. It prints `NOT COMPARABLE` and
lists the differences when the two paths disagree on row count, element mix,
embedding count, or embedding dimension, because a path that emits less is not
faster.

Two flags change what is being measured. `--embed-granularity page` is the
default and lets the fused stage absorb embedding, which is the configuration
the fused path is designed for; `element` granularity keeps the separate embed
actor on both paths. `--run-mode batch` runs through Ray, where driver-side
warmup does not reach the actor processes, so every timed pass reloads weights
and the result reflects cold start rather than steady state. Use the default
`inprocess` mode to compare compute.
