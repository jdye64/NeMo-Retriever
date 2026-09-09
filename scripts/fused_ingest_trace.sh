#!/usr/bin/env bash
# Ingest a corpus through the fused GPU-resident extraction stage using only
# local models, then report the per-stage device time the fused model recorded.
#
# No profiler is attached. The fused pipeline times each stage with CUDA events,
# and NV_INGEST_FUSED_TRACE_PATH makes it append one JSON record per batch, so
# the stage costs come out of the run itself. This script rolls those records
# into a single summary document next to the log.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

dataset=${1:-/home/local/jdyer/datasets/bo20}
mirror=${NEMO_RETRIEVER_FUSED_MIRROR:-$repo_root/uber_model/huggingface}
out_dir=${FUSED_OUT_DIR:-$repo_root/.fused-traces}
run_id=${FUSED_RUN_ID:-$(basename "$dataset")-$(date +%Y%m%d-%H%M%S)}

# page lets the fused stage absorb embedding, so one GPU-resident model does
# page elements, table structure, OCR, and embedding. element instead keeps the
# dedicated embed stage, which embeds per detected element rather than per page.
granularity=${FUSED_EMBED_GRANULARITY:-page}

# Local ingest-time embedder for the element path, where a separate embed stage
# still runs. Ignored at page granularity because the fused model embeds.
embed_backend=${FUSED_EMBED_BACKEND:-hf}

python_bin=$repo_root/.venv/bin/python
retriever_bin=$repo_root/.venv/bin/retriever
log_file=$out_dir/$run_id.log
trace_jsonl=$out_dir/$run_id.trace.jsonl
trace_json=$out_dir/$run_id.trace.json
lancedb_uri=$out_dir/$run_id.lancedb

die() { echo "error: $*" >&2; exit 1; }

[[ -d $dataset ]] || die "dataset directory not found: $dataset"
[[ -x $python_bin ]] || die "venv interpreter not found at $python_bin; create the venv first"
[[ -x $retriever_bin ]] || die "retriever CLI not found at $retriever_bin; install the library into the venv"

# The fused model is an optional package, so a missing install is the most
# likely reason method=fused fails.
fused_install_hint="nemo_retriever_fused is not installed; run: uv pip install -e ./uber_model/nemo-retriever"
"$python_bin" - <<'PY' || die "$fused_install_hint"
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("nemo_retriever_fused") else 1)
PY

[[ -d $mirror ]] || die "model mirror not found at $mirror; run: python uber_model/download_models.py"

mkdir -p "$out_dir"

# Weights load from the local mirror instead of resolving through the Hub.
export NEMO_RETRIEVER_FUSED_MIRROR="$mirror"

# Ingest falls back to NVIDIA's hosted embedding endpoint when either key is
# present. Clearing them keeps every model in this run local and on device.
unset NVIDIA_API_KEY NGC_API_KEY

# method=fused rejects per-stage NIM URLs, so page elements, table structure,
# and OCR are necessarily the local GPU-resident model. No --embed-invoke-url is
# passed either, so embedding stays local too.
#
# --table-output-format markdown turns on local table-structure extraction, so
# the table stage inside the fused model is exercised rather than skipped.
ingest_args=(
  ingest local
  --method fused
  --embed-granularity "$granularity"
  --table-output-format markdown
  --lancedb-uri "$lancedb_uri"
  --no-quiet
)

if [[ $granularity == page ]]; then
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
echo "mirror:       $mirror"
echo "granularity:  $granularity"
echo "embedding:    $embed_summary"
echo "index:        $lancedb_uri"
echo "log:          $log_file"
echo "trace:        $trace_json"
echo "batches:      $trace_jsonl"
echo

# Copied into the summary document by the aggregation step below.
export FUSED_TRACE_RUN_ID="$run_id"
export FUSED_TRACE_DATASET="$dataset"
export FUSED_TRACE_GRANULARITY="$granularity"
export FUSED_TRACE_EMBEDDING="$embed_summary"

# The fused stage writes one JSON record per batch to this path. It appends, so
# a stale file from an earlier run with the same id would be counted twice.
export NV_INGEST_FUSED_TRACE_PATH="$trace_jsonl"
rm -f "$trace_jsonl"

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

# Roll the per-batch records the library wrote into one summary document. Run
# metadata and the row count are only known out here, so they are folded in now.
"$python_bin" - "$trace_jsonl" "$trace_json" "$rows" "$wall_clock" <<'PY'
import json
import os
import sys

jsonl_path, json_path, rows, wall_clock = sys.argv[1:5]

batches = []
if os.path.exists(jsonl_path):
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
print(f"LanceDB rows: {rows}")
print()
if not completed:
    print("Fused stage: nothing recorded — the fused stage did not run")
else:
    print(
        f"Fused: {totals['pages']} pages in {totals['fused_seconds']}s "
        f"({totals['ms_per_page']} ms/page) over {totals['batches']} batch(es)"
    )
    if totals["failed_batches"]:
        print(f"Failed batches: {totals['failed_batches']}")
    print()
    print("Per-stage device time:")
    for name, value in sorted(stages.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<16} {value:>9.1f} ms  {totals['stage_share_pct'][name]:>5.1f}%")
PY

cat <<EOF

Analyze the JSON trace:

  jq '.totals' $trace_json

Stage cost, slowest first:

  jq -r '.totals.stages_ms | to_entries | sort_by(-.value) | .[] | "\(.key)\t\(.value)ms"' $trace_json

Per-batch variation across the corpus:

  jq -r '.batches[] | "\(.pages) pages\t\(.ms_per_page) ms/page"' $trace_json

Raw per-batch records, as the library wrote them:

  jq -s '.' $trace_jsonl
EOF

exit "$status"
