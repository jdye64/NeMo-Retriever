# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Removed Kubernetes/Helm harness keys must fail closed."""

from __future__ import annotations

import pytest

from nemo_retriever.harness.config import _parse_cli_overrides


def test_helm_cli_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="Helm harness options are no longer supported"):
        _parse_cli_overrides(["helm_chart=nemo-retriever"])


def test_kubectl_cli_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="kubectl harness options are no longer supported"):
        _parse_cli_overrides(["kubectl_bin=/usr/bin/kubectl"])
