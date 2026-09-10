# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic SQLModel registry for the service ORM recommendation."""

from __future__ import annotations

from sqlmodel import SQLModel


class Base(SQLModel):
    """Shared Pydantic / SQLAlchemy registry.

    Concrete tables inherit this class with ``table=True``. No bind, engine,
    or session is created here. Callers that later wire persistence should
    construct those separately.
    """
