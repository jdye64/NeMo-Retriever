# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from typing import get_args

import nemo_retriever.common.params as params_module
from nemo_retriever.models.model import ModelRunMode
from nemo_retriever.common.params import EmbedParams
from nemo_retriever.common.params import ExtractParams
from nemo_retriever.common.params import IngestorRunMode
from nemo_retriever.graph.ingestor_runtime import batch_tuning_to_node_overrides, build_graph


def _graph_node_names(graph: object) -> list[str]:
    names: list[str] = []

    def visit(node: object) -> None:
        names.append(getattr(node.operator, "name", node.name))
        for child in node.children:
            visit(child)

    for root in graph.roots:
        visit(root)
    return names


def test_run_mode_type_aliases_are_domain_specific() -> None:
    assert set(get_args(IngestorRunMode)) == {"inprocess", "batch", "service"}
    assert set(get_args(ModelRunMode)) == {"local", "NIM", "build-endpoint"}


def test_generic_run_mode_aliases_are_not_exported() -> None:
    params_models = import_module("nemo_retriever.common.params.models")
    model_module = import_module("nemo_retriever.models.model")

    assert not hasattr(params_module, "RunMode")
    assert "RunMode" not in params_module.__all__
    assert not hasattr(params_models, "RunMode")
    assert not hasattr(model_module, "RunMode")


def test_fused_tuning_surface_is_exported_and_wired() -> None:
    """The fused tuning surface must stay backed by a real dispatch.

    PR #2115 removed an earlier ``FusedTuningParams`` because it was typing
    with no implementation behind it. This asserts the replacement is wired all
    the way to a graph node, so the surface cannot go stale the same way again.
    """
    params_models = import_module("nemo_retriever.common.params.models")

    assert hasattr(params_module, "FusedTuningParams")
    assert "FusedTuningParams" in params_module.__all__
    assert hasattr(params_models, "FusedTuningParams")

    # Tuning belongs to the extract stage that owns the fused actor, not embed.
    assert "fused_tuning" in ExtractParams.model_fields
    assert "fused_tuning" not in EmbedParams.model_fields

    assert "fused" in get_args(ExtractParams.model_fields["method"].annotation)

    # The dispatch that #2115 found missing: a fused method must actually put a
    # fused node in the graph.
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(method="fused"),
        stage_order=("extract",),
    )
    assert "FusedExtractionActor" in _graph_node_names(graph)


def test_fused_tuning_params_reach_node_overrides() -> None:
    """Every ``FusedTuningParams`` field must land on the fused node."""
    extract_params = ExtractParams(
        method="fused",
        fused_tuning={
            "fused_workers": 3,
            "fused_batch_size": 16,
            "fused_cpus_per_actor": 2,
            "fused_gpus_per_actor": 0.5,
        },
    )

    overrides = batch_tuning_to_node_overrides(extract_params, None)

    assert overrides["FusedExtractionActor"] == {
        "batch_size": 16,
        "concurrency": 3,
        "num_cpus": 2,
        "num_gpus": 0.5,
    }
