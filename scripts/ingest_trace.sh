#!/usr/bin/env bash
# Ingest a corpus locally and capture both layers of tracing the pipeline offers.
#
#   1. Per-page operator spans, written by --page-trace-dir and analyzed with
#      `retriever trace`. These say which pages and which operators were slow.
#   2. Per-batch GPU stage timings from inside the fused model, written when
#      NV_INGEST_FUSED_TRACE_PATH is set. Page traces see the fused stage as a
#      single operator, so this is the only view of the decode / page-elements /
#      table-structure / OCR / embed split within it.
#
# No profiler is attached; both layers come out of the run itself.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

dataset=${1:-/home/local/jdyer/datasets/bo20}
out_dir=${INGEST_OUT_DIR:-$repo_root/.ingest-traces}
run_id=${INGEST_RUN_ID:-$(basename "$dataset")-$(date +%Y%m%d-%H%M%S)}

# fused runs page elements, table structure, OCR, and embedding as one
# GPU-resident model. Any other method uses the staged actors instead.
method=${INGEST_METHOD:-fused}

mirror=${NEMO_RETRIEVER_FUSED_MIRROR:-$repo_root/uber_model/huggingface}

# page lets the fused stage absorb embedding, so one model does everything.
# element instead keeps the dedicated embed stage, which embeds per detected
# element rather than per page.
granularity=${INGEST_EMBED_GRANULARITY:-page}

# Local ingest-time embedder for the element path, where a separate embed stage
# still runs. Ignored at page granularity because the fused model embeds.
embed_backend=${INGEST_EMBED_BACKEND:-hf}

python_bin=$repo_root/.venv/bin/python
retriever_bin=$repo_root/.venv/bin/retriever

log_file=$out_dir/$run_id.log
page_trace_dir=$out_dir/$run_id.pagetraces
fused_jsonl=$out_dir/$run_id.fused.jsonl
fused_json=$out_dir/$run_id.fused.json
lancedb_uri=$out_dir/$run_id.lancedb

die() { echo "error: $*" >&2; exit 1; }

[[ -d $dataset ]] || die "dataset directory not found: $dataset"
[[ -x $python_bin ]] || die "venv interpreter not found at $python_bin; create the venv first"
[[ -x $retriever_bin ]] || die "retriever CLI not found at $retriever_bin; install the library into the venv"

if [[ $method == fused ]]; then
  # The fused model is an optional package, so a missing install is the most
  # likely reason method=fused fails.
  fused_hint="nemo_retriever_fused is not installed; run: uv pip install -e ./uber_model/nemo-retriever"
  "$python_bin" - <<'PY' || die "$fused_hint"
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("nemo_retriever_fused") else 1)
PY

  [[ -d $mirror ]] || die "model mirror not found at $mirror; run: python uber_model/download_models.py"

  # Weights load from the local mirror instead of resolving through the Hub.
  export NEMO_RETRIEVER_FUSED_MIRROR="$mirror"

  # The fused stage appends one JSON record per batch here. It appends, so a
  # stale file from a reused run id would otherwise be counted twice.
  export NV_INGEST_FUSED_TRACE_PATH="$fused_jsonl"
fi

mkdir -p "$out_dir" "$page_trace_dir"
rm -f "$fused_jsonl"

# Ingest falls back to NVIDIA's hosted embedding endpoint when either key is
# present. Clearing them keeps every model in this run local and on device.
unset NVIDIA_API_KEY NGC_API_KEY

# method=fused rejects per-stage NIM URLs, so page elements, table structure,
# and OCR are necessarily the local GPU-resident model. No --embed-invoke-url is
# passed either, so embedding stays local too.
#
# --table-output-format markdown turns on local table-structure extraction, so
# the table stage is exercised rather than skipped.
ingest_args=(
  ingest local
  --method "$method"
  --embed-granularity "$granularity"
  --table-output-format markdown
  --lancedb-uri "$lancedb_uri"
  --page-trace-dir "$page_trace_dir"
  --page-trace-detail operator
  --no-quiet
)

if [[ $granularity == page && $method == fused ]]; then
  # The fused model embeds the page raster, so the embedding is an image one.
  ingest_args+=(--embed-modality image)
  embed_summary="local, inside the fused model"
else
  # Pin the separate embed stage to a local backend rather than letting it
  # resolve an endpoint.
  ingest_args+=(--local-ingest-embed-backend "$embed_backend")
  embed_summary="local, separate stage ($embed_backend)"
fi

ingest_args+=("$dataset")

echo "dataset:      $dataset ($(find "$dataset" -maxdepth 1 -type f | wc -l) files)"
echo "method:       $method"
echo "granularity:  $granularity"
echo "embedding:    $embed_summary"
if [[ $method == fused ]]; then
  echo "mirror:       $mirror"
fi
echo "index:        $lancedb_uri"
echo "log:          $log_file"
echo "page traces:  $page_trace_dir"
if [[ $method == fused ]]; then
  echo "fused trace:  $fused_json"
fi
echo

# Copied into the fused summary document by the aggregation step below.
export FUSED_TRACE_RUN_ID="$run_id"
export FUSED_TRACE_DATASET="$dataset"
export FUSED_TRACE_METHOD="$method"
export FUSED_TRACE_GRANULARITY="$granularity"
export FUSED_TRACE_EMBEDDING="$embed_summary"

SECONDS=0
set +e
"$retriever_bin" "${ingest_args[@]}" 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
set -e
wall_clock=$SECONDS

rows=$("$python_bin" - "$lancedb_uri" <<'PY' 2>/dev/null || echo "unknown"
import sys
import lancedb

db = lancedb.connect(sys.argv[1])
print(sum(db.open_table(name).count_rows() for name in db.table_names()))
PY
)

echo
echo "LanceDB rows: $rows"

# Roll the per-batch records the fused stage wrote into one summary document.
# Run metadata and the row count are only known out here, so fold them in now.
if [[ -s $fused_jsonl ]]; then
  "$python_bin" - "$fused_jsonl" "$fused_json" "$rows" "$wall_clock" <<'PY'
import json
import os
import sys

jsonl_path, json_path, rows, wall_clock = sys.argv[1:5]

batches = []
with open(jsonl_path, encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            batches.append(json.loads(line))

completed = [b for b in batches if b.get("event") == "fused.extract"]
pages = sum(int(b.get("pages", 0)) for b in completed)
seconds = sum(float(b.get("duration_s", 0.0)) for b in completed)

stages: dict[str, float] = {}
for batch in completed:
    for name, value in (batch.get("stages_ms") or {}).items():
        stages[name] = stages.get(name, 0.0) + float(value)
total_stage_ms = sum(stages.values())

trace = {
    "run_id": os.environ.get("FUSED_TRACE_RUN_ID", ""),
    "dataset": os.environ.get("FUSED_TRACE_DATASET", ""),
    "method": os.environ.get("FUSED_TRACE_METHOD", ""),
    "granularity": os.environ.get("FUSED_TRACE_GRANULARITY", ""),
    "embedding": os.environ.get("FUSED_TRACE_EMBEDDING", ""),
    "wall_clock_seconds": round(float(wall_clock), 3),
    "lancedb_rows": int(rows) if rows.isdigit() else None,
    "totals": {
        "batches": len(completed),
        "failed_batches": sum(1 for b in batches if b.get("event") == "fused.extract_failed"),
        "pages": pages,
        "fused_seconds": round(seconds, 3),
        "ms_per_page": round(1000.0 * seconds / pages, 2) if pages else None,
        "pages_per_second": round(pages / seconds, 2) if seconds else None,
        "stages_ms": {k: round(v, 1) for k, v in stages.items()},
        "stage_share_pct": (
            {k: round(100.0 * v / total_stage_ms, 1) for k, v in stages.items()} if total_stage_ms else {}
        ),
    },
    "batches": batches,
}

with open(json_path, "w", encoding="utf-8") as handle:
    json.dump(trace, handle, indent=2)
    handle.write("\n")

totals = trace["totals"]
print()
print(
    f"Fused: {totals['pages']} pages in {totals['fused_seconds']}s "
    f"({totals['ms_per_page']} ms/page) over {totals['batches']} batch(es)"
)
if totals["failed_batches"]:
    print(f"Failed batches: {totals['failed_batches']}")
print()
print("GPU stage time inside the fused model:")
for name, value in sorted(stages.items(), key=lambda kv: -kv[1]):
    print(f"  {name:<16} {value:>9.1f} ms  {totals['stage_share_pct'][name]:>5.1f}%")
PY
elif [[ $method == fused ]]; then
  echo
  echo "Fused stage: nothing recorded — the fused stage did not run"
fi

page_trace_count=$(find "$page_trace_dir" -name "*.trace.json*" | wc -l)

echo
echo "Page traces:  $page_trace_count document(s) in $page_trace_dir"
echo
if [[ $page_trace_count -eq 0 ]]; then
  echo "No page traces were written, so there is nothing for retriever trace to read."
else
  cat <<EOF
Analyze per-page operator spans:

  retriever trace $page_trace_dir

  retriever trace $page_trace_dir --json            # same rollup, machine-readable
  retriever trace $page_trace_dir --page 1          # span waterfall for one page
  retriever trace $page_trace_dir -o $out_dir/$run_id.spans.parquet
EOF
fi

if [[ -s $fused_json ]]; then
  cat <<EOF

Analyze the fused GPU stage breakdown:

  jq '.totals' $fused_json

  jq -r '.totals.stages_ms | to_entries | sort_by(-.value) | .[] | "\(.key)\t\(.value)ms"' $fused_json
EOF
fi

exit "$status"
