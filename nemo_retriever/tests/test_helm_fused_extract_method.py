# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chart coverage for ``serviceConfig.localModels.extract.method``.

The fused extraction method is only reachable in a Helm deployment if the
chart both exposes the value and renders it into ``local_models.extract`` in
the service ConfigMap. These tests pin both halves, plus the default, so a
deployment cannot silently fall back to the staged pipeline.

The integration tests shell out to ``helm template`` when ``helm`` is on
``$PATH``; otherwise they skip cleanly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from unittest import SkipTest, TestCase, main


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_required_file(path: Path) -> str:
    if not path.is_file():
        raise SkipTest(f"Required file not present in this test environment: {path}")
    return path.read_text(encoding="utf-8")


def _helm_template(extra_args: Sequence[str] = ()) -> subprocess.CompletedProcess[str]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    chart_path = _repo_root() / "nemo_retriever/helm"
    if not chart_path.is_dir():
        raise SkipTest(f"Chart directory missing: {chart_path}")

    cmd: list[str] = [
        helm,
        "template",
        "retriever",
        str(chart_path),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
    ]
    cmd += list(extra_args)
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _assert_helm_ok(self: TestCase, proc: subprocess.CompletedProcess[str]) -> None:
    self.assertEqual(
        proc.returncode,
        0,
        f"`helm template` failed unexpectedly:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
    )


class HelmFusedExtractMethodTests(TestCase):
    def test_values_expose_extract_method(self) -> None:
        values = _read_required_file(_repo_root() / "nemo_retriever/helm/values.yaml")
        self.assertIn("method: pdfium", values)

    def test_configmap_renders_extract_method(self) -> None:
        body = _read_required_file(_repo_root() / "nemo_retriever/helm/templates/configmap.yaml")
        self.assertIn("serviceConfig.localModels.extract.method", body)
        self.assertIn("method:", body)

    def test_helm_template_defaults_to_pdfium(self) -> None:
        # The vectordb sub-stack demands a resolvable query embedding backend,
        # which is unrelated to extraction; disable it to isolate this render.
        proc = _helm_template(extra_args=("--set", "serviceConfig.vectordb.enabled=false"))
        _assert_helm_ok(self, proc)
        self.assertIn('method: "pdfium"', proc.stdout)
        self.assertNotIn('method: "fused"', proc.stdout)

    def test_helm_template_renders_fused_when_selected(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "serviceConfig.localModels.enabled=true",
                "--set",
                "serviceConfig.localModels.extract.method=fused",
            )
        )
        _assert_helm_ok(self, proc)
        self.assertIn('method: "fused"', proc.stdout)


if __name__ == "__main__":
    main()
