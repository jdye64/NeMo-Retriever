# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage adapters for the fused pipeline.

Each module here wraps one of the models the production pipeline calls
remotely, replacing that model's host-side entry point with a device-resident
one while leaving its weights and inference graph untouched.
"""

from __future__ import annotations

__all__ = ["detectors", "embed", "ocr"]
