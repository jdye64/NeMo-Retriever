# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recommended SQLModel (Pydantic + SQLAlchemy) tables for gateway stores.

Field names and types follow the current in-memory representations:

* :class:`~nemo_retriever.service.services.job_tracker.JobAggregate` /
  :class:`~nemo_retriever.service.services.job_tracker.DocumentRecord`
* :class:`~nemo_retriever.service.services.sidecar_store.SidecarEntry`
* :class:`~nemo_retriever.service.services.work_queue.WorkRecord` /
  :class:`~nemo_retriever.service.services.work_queue.WorkLease`
* the worker result dict in ``worker_result_store`` (``document_id`` →
  ``(stored_at, result_data)``)

Process-local monotonic clocks (``_started_mono``, ``_job_started_mono``,
in-memory result ``stored_at``) are omitted; they are not meaningful
across processes.

This module does not open a database or replace the in-memory stores.
"""

from enum import Enum
from typing import Any, Optional

from sqlalchemy import Enum as SAEnum
from sqlalchemy import Column, ForeignKey, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.types import JSON
from sqlmodel import Field, Relationship

from nemo_retriever.service.orm.base import Base


class DocumentStatus(str, Enum):
    """Mirrors ``job_tracker.DocumentStatus``."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobAggregateStatus(str, Enum):
    """Mirrors ``job_tracker.JobAggregateStatus``."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


class IngestOperation(str, Enum):
    """Mirrors ``common.schemas.collections.IngestOperation``."""

    APPEND = "append"
    REPLACE = "replace"


def _str_enum(enum_cls: type[Enum], *, length: int = 32) -> SAEnum:
    """Persist str-enums as VARCHAR so SQLite and Postgres share one mapping."""
    return SAEnum(
        enum_cls,
        values_callable=lambda members: [member.value for member in members],
        native_enum=False,
        length=length,
        validate_strings=True,
    )


class JobAggregate(Base, table=True):
    """Persistent form of :class:`nemo_retriever.service.services.job_tracker.JobAggregate`.

    Arrival order of documents is ``documents`` ordered by ``arrival_index``,
    matching in-memory ``document_ids``.
    """

    __tablename__ = "jobs"

    job_id: str = Field(primary_key=True, max_length=128)
    expected_documents: int
    counts: dict[str, int] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    status: JobAggregateStatus = Field(
        default=JobAggregateStatus.PENDING,
        sa_column=Column(_str_enum(JobAggregateStatus), nullable=False),
    )
    created_at: str = Field(default="", max_length=64)
    started_at: Optional[str] = Field(default=None, max_length=64)
    finalized_at: Optional[str] = Field(default=None, max_length=64)
    elapsed_s: Optional[float] = Field(default=None)
    label: Optional[str] = Field(default=None, max_length=512)
    # In-memory field is ``metadata``; that name is reserved on SQLAlchemy models.
    job_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    trace_id: Optional[str] = Field(default=None, max_length=64)
    trace_context: dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    retain_results: bool = Field(default=False)
    collection_name: Optional[str] = Field(default=None, max_length=128)
    scope: str = Field(default="default", max_length=128)
    operation: IngestOperation = Field(
        default=IngestOperation.APPEND,
        sa_column=Column(_str_enum(IngestOperation), nullable=False),
    )
    target_document_id: Optional[str] = Field(default=None, max_length=128)
    idempotency_key: Optional[str] = Field(default=None, max_length=256)
    idempotency_fingerprint: Optional[str] = Field(default=None, max_length=128)
    document_manifest: list[dict[str, str]] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    progress_published: int = Field(default=0)

    documents: list["DocumentRecord"] = Relationship(
        back_populates="job",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "order_by": "DocumentRecord.arrival_index",
        },
    )
    idempotency_row: Optional["JobIdempotency"] = Relationship(
        back_populates="job",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "uselist": False},
    )


class DocumentRecord(Base, table=True):
    """Persistent form of :class:`nemo_retriever.service.services.job_tracker.DocumentRecord`."""

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("job_id", "manifest_entry_id", name="uq_documents_job_manifest_entry"),)

    id: str = Field(primary_key=True, max_length=128)
    stable_document_id: str = Field(max_length=128)
    job_id: str = Field(sa_column=Column(String(128), ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False))
    arrival_index: int = Field(default=0)
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING,
        sa_column=Column(_str_enum(DocumentStatus), nullable=False),
    )
    submitted_at: str = Field(default="", max_length=64)
    started_at: Optional[str] = Field(default=None, max_length=64)
    completed_at: Optional[str] = Field(default=None, max_length=64)
    elapsed_s: Optional[float] = Field(default=None)
    result_rows: Optional[int] = Field(default=None)
    result_data: Optional[list[dict[str, Any]]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    filename: Optional[str] = Field(default=None, max_length=1024)
    collection_name: Optional[str] = Field(default=None, max_length=128)
    content_sha256: Optional[str] = Field(default=None, max_length=64)
    manifest_entry_id: Optional[str] = Field(default=None, max_length=128)

    job: Optional[JobAggregate] = Relationship(back_populates="documents")


class JobIdempotency(Base, table=True):
    """Persistent form of ``JobTracker._idempotency`` keyed by ``(scope, key)``."""

    __tablename__ = "job_idempotency"

    scope: str = Field(primary_key=True, max_length=128)
    idempotency_key: str = Field(primary_key=True, max_length=256)
    job_id: str = Field(sa_column=Column(String(128), ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False))
    fingerprint: str = Field(max_length=128)

    job: Optional[JobAggregate] = Relationship(back_populates="idempotency_row")


class SidecarEntry(Base, table=True):
    """Persistent form of :class:`nemo_retriever.service.services.sidecar_store.SidecarEntry`."""

    __tablename__ = "sidecars"

    sidecar_id: str = Field(primary_key=True, max_length=64)
    filename: str = Field(max_length=1024)
    content_type: str = Field(max_length=256)
    payload: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    created_at: float
    expires_at: float
    owner_token: Optional[str] = Field(default=None, max_length=512)
    consume_on_read: bool = Field(default=True)
    sidecar_metadata: dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))


class WorkRecord(Base, table=True):
    """Persistent form of :class:`nemo_retriever.service.services.work_queue.WorkRecord`.

    ``pool`` stores ``PoolType`` values (``realtime`` / ``batch``) as strings
    so this module does not import the pipeline pool.
    Paths are stored as strings rather than :class:`~pathlib.Path`.
    """

    __tablename__ = "work_records"

    work_id: str = Field(primary_key=True, max_length=128)
    job_id: str = Field(index=True, max_length=128)
    pool: str = Field(max_length=32)
    filename: Optional[str] = Field(default=None, max_length=1024)
    spool_path: str = Field(max_length=4096)
    payload_size: int
    payload_sha256: str = Field(max_length=64)
    retain_results: bool = Field(default=False)
    pipeline_spec: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    trace_context: dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    enqueued_at: float
    sidecar_path: Optional[str] = Field(default=None, max_length=4096)
    sidecar_size: int = Field(default=0)
    sidecar_sha256: Optional[str] = Field(default=None, max_length=64)
    sidecar_filename: Optional[str] = Field(default=None, max_length=1024)
    sidecar_content_type: Optional[str] = Field(default=None, max_length=256)
    delivery_attempt: int = Field(default=0)
    generation: int = Field(default=0)
    activated_generation: Optional[int] = Field(default=None)
    extra: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))

    lease: Optional["WorkLease"] = Relationship(
        back_populates="work",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "uselist": False},
    )


class WorkLease(Base, table=True):
    """Persistent form of :class:`nemo_retriever.service.services.work_queue.WorkLease`."""

    __tablename__ = "work_leases"

    work_id: str = Field(
        sa_column=Column(
            String(128),
            ForeignKey("work_records.work_id", ondelete="CASCADE"),
            primary_key=True,
        )
    )
    lease_id: str = Field(max_length=128)
    generation: int
    worker_uid: str = Field(max_length=256)
    worker_ip: str = Field(max_length=64)
    expires_at: float

    work: Optional[WorkRecord] = Relationship(back_populates="lease")


class WorkerResult(Base, table=True):
    """Persistent form of the in-memory worker result map.

    In memory this is ``dict[str, tuple[float, list[dict]]]``. Shared-disk
    generations are a separate filesystem protocol and are not modeled here.
    """

    __tablename__ = "worker_results"

    document_id: str = Field(primary_key=True, max_length=128)
    stored_at: float
    result_data: list[dict[str, Any]] = Field(sa_column=Column(JSON, nullable=False))
