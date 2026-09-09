# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detail-level gating and the process-local model registry for page tracing.

Tracing runs inside Ray Data actors, so the detail level has to reach worker
processes. Two mechanisms cover the deployment shapes NRL supports:

* ``NEMO_RETRIEVER_PAGE_TRACE_DETAIL`` in the environment. Locally started Ray
  clusters inherit the driver environment, so setting it on the driver is
  enough for ``run_mode='batch'`` against an implicit cluster.
* Explicit propagation through the operator adapter that
  :class:`~nemo_retriever.graph.executor.RayDataExecutor` constructs on each
  worker. This covers pre-existing and remote clusters, whose workers do not
  inherit the driver environment.
"""

from __future__ import annotations

import os
import socket
import threading
from typing import Any, Final, Literal, get_args

TraceDetail = Literal["off", "operator", "full"]

TRACE_DETAIL_ENV_VAR: Final = "NEMO_RETRIEVER_PAGE_TRACE_DETAIL"
DEFAULT_TRACE_DETAIL: Final[TraceDetail] = "operator"
VALID_TRACE_DETAILS: Final[tuple[str, ...]] = get_args(TraceDetail)

# Column carrying serialized page trace payloads through the operator graph.
TRACE_COLUMN: Final = "_nrl_trace"

TRACE_SCHEMA_VERSION: Final = "1.0"

_lock = threading.Lock()
_detail_override: TraceDetail | None = None
_models: dict[str, dict[str, Any]] = {}
_worker_label: str | None = None


def normalize_detail(value: Any) -> TraceDetail:
    """Coerce *value* to a valid detail level, falling back to the default."""
    if value is None:
        return DEFAULT_TRACE_DETAIL
    candidate = str(value).strip().lower()
    if candidate in VALID_TRACE_DETAILS:
        return candidate  # type: ignore[return-value]
    # Accept the common boolean spellings so environment overrides are forgiving.
    if candidate in {"0", "false", "no", "none", "disabled"}:
        return "off"
    if candidate in {"1", "true", "yes", "on"}:
        return "operator"
    if candidate in {"all", "verbose", "detailed"}:
        return "full"
    return DEFAULT_TRACE_DETAIL


def set_detail(value: Any, *, export_to_env: bool = False) -> TraceDetail:
    """Set the process-local detail level and return the normalized value."""
    global _detail_override

    detail = normalize_detail(value)
    with _lock:
        _detail_override = detail
    if export_to_env:
        os.environ[TRACE_DETAIL_ENV_VAR] = detail
    return detail


def get_detail() -> TraceDetail:
    """Return the active detail level for this process."""
    with _lock:
        if _detail_override is not None:
            return _detail_override
    return normalize_detail(os.environ.get(TRACE_DETAIL_ENV_VAR))


def reset_detail() -> None:
    """Drop the process-local override so the environment governs again."""
    global _detail_override

    with _lock:
        _detail_override = None


def tracing_enabled() -> bool:
    """Return whether any tracing should be recorded."""
    return get_detail() != "off"


def full_detail_enabled() -> bool:
    """Return whether curated hot-spot spans should be recorded."""
    return get_detail() == "full"


def worker_label() -> str:
    """Return a short ``host:pid`` label identifying this worker process."""
    global _worker_label

    if _worker_label is None:
        try:
            host = socket.gethostname().split(".", 1)[0]
        except OSError:
            host = "unknown"
        _worker_label = f"{host}:{os.getpid()}"
    return _worker_label


def record_model(
    model_key: str,
    *,
    name: str | None = None,
    version: str | None = None,
    backend: str | None = None,
    endpoint: str | None = None,
) -> None:
    """Register the model backing *model_key* so traces can report its version.

    Called from model clients and operator setup. Registrations accumulate in
    the worker process and are attached to trace payloads as rows flow through,
    which is how worker-side model versions reach the aggregating driver.
    """
    if not model_key:
        return

    descriptor = {
        "model_key": str(model_key),
        "name": str(name) if name else None,
        "version": str(version) if version else None,
        "backend": str(backend) if backend else None,
        "endpoint": str(endpoint) if endpoint else None,
    }
    with _lock:
        existing = _models.get(model_key)
        if existing is None:
            _models[model_key] = descriptor
            return
        # Later registrations refine earlier ones; a probe may learn the served
        # version after the configured name was recorded at construction time.
        for field, value in descriptor.items():
            if value is not None:
                existing[field] = value


def registered_models() -> list[dict[str, Any]]:
    """Return the model descriptors registered in this process."""
    with _lock:
        return [dict(descriptor) for descriptor in _models.values()]


def clear_models() -> None:
    """Drop registered model descriptors. Intended for tests."""
    with _lock:
        _models.clear()


def model_identity(descriptor: dict[str, Any]) -> tuple[Any, ...]:
    """Return the dedupe key for a model descriptor."""
    return (
        descriptor.get("model_key"),
        descriptor.get("name"),
        descriptor.get("version"),
        descriptor.get("backend"),
        descriptor.get("endpoint"),
    )
