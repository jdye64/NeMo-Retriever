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
retriever benchmark pdf-engine --help
```

The following example extracts from a directory of PDF files that you supply.

```bash
retriever benchmark extract /path/to/your/pdfs \
  --pdf-extract-batch-size 8 \
  --pdf-extract-actors 4
```

```bash
retriever benchmark pdf-engine --help
```

`retriever benchmark pdf-engine` compares the CPU PDFium engine and the GPU
engine on a directory of PDF files that you supply. The command reports load,
page-info, and raster times plus encoded image bytes so you can share the
comparison with your team.

```bash
retriever benchmark pdf-engine run \
  --input-dir /path/to/your/pdfs \
  --backends cpu,gpu \
  --dpi 200 \
  --render-mode fit_to_model \
  --output-json /tmp/pdf-engine-bench.json
```

The CPU engine uses PDFium for parse, raster, and JPEG encode, matching the
current PDF extraction path. The GPU engine still uses PDFium for document
load, page metadata, and the CPU bitmap, then converts color and encodes JPEG
on the GPU when CUDA is available. It can keep CHW tensors on device for a
later page-elements model invoke.

Stage commands report rows per second, or chunk rows per second for audio. They
do not produce the NRB artifact contract and should not be used as retrieval
quality evidence.
