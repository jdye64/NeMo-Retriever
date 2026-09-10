# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLModel (Pydantic) tables for service persistence (not wired).

These classes recommend a relational shape for the gateway's in-memory
stores. They are not attached to an engine, session, or ``JobTracker`` /
``SidecarStore`` / ``WorkBroker`` / worker result store.
"""

from nemo_retriever.service.orm.base import Base
from nemo_retriever.service.orm.models import (
    DocumentRecord,
    DocumentStatus,
    IngestOperation,
    JobAggregate,
    JobAggregateStatus,
    JobIdempotency,
    SidecarEntry,
    WorkerResult,
    WorkLease,
    WorkRecord,
)

__all__ = [
    "Base",
    "DocumentRecord",
    "DocumentStatus",
    "IngestOperation",
    "JobAggregate",
    "JobAggregateStatus",
    "JobIdempotency",
    "SidecarEntry",
    "WorkLease",
    "WorkRecord",
    "WorkerResult",
]
