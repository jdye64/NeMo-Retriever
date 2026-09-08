# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the HuggingFace snapshots backing the fused NeMo Retriever model.

Revisions are read from ``nemo_retriever.models.hf_model_registry`` so that this
mirror always matches the pins the production pipeline resolves at runtime. The
snapshots land under ``uber_model/huggingface/<org>--<name>/`` and are excluded
from version control; re-run this script to recreate them.

Usage::

    python uber_model/download_models.py                 # all stages
    python uber_model/download_models.py page_elements   # a subset
    python uber_model/download_models.py --source-only   # code/config, no weights
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import logging
import os
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MIRROR_ROOT = Path(__file__).resolve().parent / "huggingface"
REGISTRY_PATH = REPO_ROOT / "nemo_retriever" / "src" / "nemo_retriever" / "models" / "hf_model_registry.py"

logger = logging.getLogger("download_models")


def _load_revision_registry() -> dict[str, str]:
    """Return ``HF_MODEL_REVISIONS`` from the in-repo registry.

    The module is loaded directly from its file rather than imported as
    ``nemo_retriever.models.hf_model_registry`` so that this script does not
    execute the package ``__init__``. That import pulls in the whole library,
    which needs Python 3.12 and the full runtime dependency set; reading the
    pins needs neither.
    """
    spec = importlib.util.spec_from_file_location("_nrl_hf_model_registry", REGISTRY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the revision registry from {REGISTRY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.HF_MODEL_REVISIONS)


# Stage name -> HuggingFace repo id. The revision for each repo comes from the
# in-repo registry rather than being duplicated here.
STAGE_REPOS: dict[str, str] = {
    "page_elements": "nvidia/nemotron-page-elements-v3",
    "table_structure": "nvidia/nemotron-table-structure-v1",
    "ocr": "nvidia/nemotron-ocr-v2",
    "ocr_v1": "nvidia/nemotron-ocr-v1",
    "embed": "nvidia/llama-nemotron-embed-vl-1b-v2",
    "rerank": "nvidia/llama-nemotron-rerank-vl-1b-v2",
}

# Patterns that describe modeling code and configuration but no tensor data.
SOURCE_ONLY_PATTERNS = [
    "*.py",
    "*.json",
    "*.txt",
    "*.md",
    "*.yaml",
    "*.yml",
]

# Tensor formats we never need: the pinned snapshots all ship safetensors, and
# pulling the duplicate .bin/.pt copies doubles the download for no benefit.
IGNORE_PATTERNS = [
    "*.msgpack",
    "*.h5",
    "*.tflite",
    "*.onnx_data",
]


def local_dir_for(repo_id: str) -> Path:
    """Return the mirror directory for *repo_id* (``org/name`` -> ``org--name``)."""
    return MIRROR_ROOT / repo_id.replace("/", "--")


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: dict[str, str] | None = None


def revision_for(repo_id: str) -> str:
    """Return the pinned revision for *repo_id*, loading the registry once."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = _load_revision_registry()
    try:
        return _REGISTRY[repo_id]
    except KeyError as exc:
        raise ValueError(f"{repo_id} has no pinned revision in {REGISTRY_PATH}") from exc


def download_one(stage: str, repo_id: str, *, source_only: bool) -> tuple[str, str, str]:
    """Snapshot *repo_id* at its pinned revision, returning ``(stage, repo, path)``."""
    from huggingface_hub import snapshot_download

    revision = revision_for(repo_id)
    local_dir = local_dir_for(repo_id)
    logger.info("downloading %s (%s) revision=%s", stage, repo_id, revision)

    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=SOURCE_ONLY_PATTERNS if source_only else None,
        ignore_patterns=None if source_only else IGNORE_PATTERNS,
        max_workers=8,
    )
    (local_dir / "REVISION").write_text(f"{repo_id}\n{revision}\n")
    logger.info("finished %s -> %s", stage, path)
    return stage, repo_id, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stages",
        nargs="*",
        metavar="STAGE",
        help=f"Stages to download ({', '.join(sorted(STAGE_REPOS))}). Defaults to every stage.",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Download modeling code and configuration without tensor files.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=3,
        help="Number of repositories to download concurrently.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    stages = args.stages or sorted(STAGE_REPOS)
    unknown = [stage for stage in stages if stage not in STAGE_REPOS]
    if unknown:
        parser.error(f"unknown stage(s) {unknown}; choose from {sorted(STAGE_REPOS)}")

    # huggingface_hub's tqdm bars are not thread safe across concurrent
    # snapshot_download calls and raise "type object 'tqdm' has no attribute
    # '_lock'". Disable them whenever more than one repo downloads at a time.
    if args.jobs > 1 and len(stages) > 1:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    MIRROR_ROOT.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, Exception]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(download_one, stage, STAGE_REPOS[stage], source_only=args.source_only): stage
            for stage in stages
        }
        for future in concurrent.futures.as_completed(futures):
            stage = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - report every stage before exiting
                logger.error("failed %s: %s", stage, exc)
                failures.append((stage, exc))

    if failures:
        logger.error("%d of %d downloads failed", len(failures), len(stages))
        return 1

    logger.info("all %d downloads complete under %s", len(stages), MIRROR_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
