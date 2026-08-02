"""Exact ancestor loading and active-clone assembly for S4-R7."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

import torch

from models.static_rgb_act import StaticRGBMoEACTConfig
from models.wam_multimodal import (
    AgentFactorizedFlowWAM,
    CrossAgentWorldConditionedFlow,
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
    ProtectedTeamFuturePredictor,
    ProtectedTeamFuturePredictorConfig,
    S4WorldEvidenceProvider,
    ScaleAlignedActiveWorldFlow,
    UtilityCalibratedWorldFlow,
    WorldEvidenceRouterConfig,
    WorldToFlowAdapterConfig,
)
from scripts.train_s2_r4_future_predictor import (
    CHECKPOINT_FORMAT as R4_CHECKPOINT_FORMAT,
    FLOW_FORMAT,
)
from scripts.train_s2_r5_protected_team import (
    CHECKPOINT_FORMAT as R5_CHECKPOINT_FORMAT,
)
from scripts.train_s3_r6_world_action_flow import (
    CHECKPOINT_FORMAT as R6_CHECKPOINT_FORMAT,
)
from scripts.train_static_rgb_act_moe import _mapping
from train.s2_future_prediction import file_sha256, state_dict_sha256


S4_TASKS = (
    "lift_barrier",
    "long_pipeline_delivery",
    "take_photo",
    "three_robots_stack_cube",
    "camera_alignment",
)


def build_s4_r7_model(
    raw: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[
    UtilityCalibratedWorldFlow,
    CrossAgentWorldConditionedFlow,
    dict[str, Any],
]:
    """Load immutable ancestors once and return a separate active GPU clone."""

    parent = _mapping(raw, "parent")
    legacy_path = _root_path(parent["legacy_r6l_policy"])
    flow_path = _root_path(parent["active_flow_checkpoint"])
    local_path = _root_path(parent["local_future_checkpoint"])
    team_path = _root_path(parent["team_future_checkpoint"])
    pca_path = _root_path(_mapping(raw, "artifacts")["pca_statistics"])
    paths = {
        "legacy_r6l_policy": legacy_path,
        "active_flow_checkpoint": flow_path,
        "local_future_checkpoint": local_path,
        "team_future_checkpoint": team_path,
        "pca_artifact": pca_path,
    }
    expected_keys = {
        "legacy_r6l_policy": "expected_legacy_r6l_policy_sha256",
        "active_flow_checkpoint": "expected_active_flow_sha256",
        "local_future_checkpoint": "expected_local_future_sha256",
        "team_future_checkpoint": "expected_team_future_sha256",
        "pca_artifact": "expected_pca_sha256",
    }
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    for name, key in expected_keys.items():
        expected = str(parent.get(key, ""))
        if not expected or hashes[name] != expected:
            raise ValueError(
                f"S4-R7 {name} hash differs from the preregistered ancestor"
            )

    legacy_payload = _load_mapping(legacy_path)
    legacy_method = _mapping(legacy_payload, "method")
    if (
        legacy_payload.get("format_version") != R6_CHECKPOINT_FORMAT
        or legacy_method.get("round_id") != "s3-r6"
        or legacy_method.get("micro_round") != "R6L"
        or legacy_method.get("candidate_id") != "P1"
        or legacy_method.get("model_kind") != "s3_r6l_protected_local_gated"
        or legacy_method.get("future_scope") != "local"
        or legacy_method.get("injection") is not True
    ):
        raise ValueError("S4-R7 requires the accepted S3 R6L-P1 policy")
    structural = _mapping(legacy_payload, "structural_invariants")
    if not all(
        structural.get(key) is True
        for key in (
            "protected_own_elementwise_exact",
            "protected_parent_model_hashes_unchanged",
            "parent_files_unchanged",
            "parents_excluded_from_optimizer",
        )
    ):
        raise ValueError("R6L-P1 checkpoint lacks its accepted structural audit")
    legacy_parent = _mapping(legacy_payload, "parent_identity")
    if (
        legacy_parent.get("flow_checkpoint_sha256") != hashes["active_flow_checkpoint"]
        or legacy_parent.get("protected_own_checkpoint_sha256")
        != hashes["local_future_checkpoint"]
        or legacy_parent.get("protected_team_checkpoint_sha256")
        != hashes["team_future_checkpoint"]
    ):
        raise ValueError("R6L-P1 does not identify the selected S4 ancestors")

    flow_payload = _load_mapping(flow_path)
    flow_method = _mapping(flow_payload, "method")
    task_runtime = flow_payload.get("task_runtime")
    tasks = tuple(
        str(row.get("task_id"))
        for row in task_runtime
        if isinstance(row, Mapping)
    ) if isinstance(task_runtime, list) else ()
    if (
        flow_payload.get("format_version") != FLOW_FORMAT
        or flow_method.get("action_generator") != "rectified_flow_cold"
        or flow_method.get("round_id") != "s3-r6"
        or flow_method.get("micro_round") != "R6L"
        or flow_method.get("candidate_id") != "P1"
        or flow_method.get("training_scope")
        != "five_task_from_scratch_per_candidate"
        or tasks != S4_TASKS
    ):
        raise ValueError("S4-R7 Flow is not the accepted R6L-P1 five-task parent")
    flow_config = StaticRGBMoEACTConfig.from_dict(
        dict(_mapping(flow_payload, "model_config"))
    )
    base_flow = AgentFactorizedFlowWAM(flow_config)
    base_flow.load_state_dict(flow_payload["model"], strict=True)

    local_payload = _load_mapping(local_path)
    local_method = _mapping(local_payload, "method")
    if (
        local_payload.get("format_version") != R4_CHECKPOINT_FORMAT
        or local_method.get("candidate_id") != "P0"
        or local_method.get("model_kind") != "s2_r4_local_action_conditioned"
        or local_method.get("team_shared") is not False
    ):
        raise ValueError("S4-R7 local future parent is not accepted R4-P0")
    local_config = LocalFuturePredictorConfig.from_dict(
        dict(_mapping(local_payload, "model_config"))
    )
    configured_local = LocalFuturePredictorConfig.from_dict(
        dict(_mapping(raw, "world_model"))
    )
    if local_config != configured_local:
        raise ValueError("S4-R7 world config differs from the local ancestor")
    local_predictor = LocalActionConditionedFuturePredictor(local_config)
    local_predictor.load_state_dict(local_payload["model"], strict=True)

    team_payload = _load_mapping(team_path)
    team_method = _mapping(team_payload, "method")
    if (
        team_payload.get("format_version") != R5_CHECKPOINT_FORMAT
        or team_method.get("candidate_id") != "P0"
        or team_method.get("model_kind") != "s2_r5_protected_shared_team"
        or team_method.get("team_mixer") != "shared"
        or _mapping(team_payload, "protected_parent").get("checkpoint_sha256")
        != hashes["local_future_checkpoint"]
    ):
        raise ValueError("S4-R7 team future parent is not accepted R5-P0")
    team_config = ProtectedTeamFuturePredictorConfig.from_dict(
        dict(_mapping(team_payload, "team_model_config"))
    )
    if team_config != ProtectedTeamFuturePredictorConfig.from_dict(
        dict(_mapping(raw, "team_model"))
    ):
        raise ValueError("S4-R7 team config differs from R5-P0")
    team_predictor = ProtectedTeamFuturePredictor(local_config, team_config)
    team_predictor.load_protected_own(local_payload["model"])
    team_predictor.load_team_state_dict(_mapping(team_payload, "team_model"))

    adapter_config = WorldToFlowAdapterConfig.from_dict(
        dict(_mapping(raw, "legacy_adapter"))
    )
    if dict(_mapping(legacy_payload, "adapter_config")) != adapter_config.to_dict():
        raise ValueError("S4-R7 legacy adapter config differs from R6L-P1")
    legacy_reference = CrossAgentWorldConditionedFlow(
        base_flow,
        local_predictor,
        adapter_config,
        future_scope="local",
        injection=True,
    )
    legacy_reference.load_adapter_state_dict(_mapping(legacy_payload, "adapter"))
    legacy_reference.eval()
    legacy_model_hash = state_dict_sha256(legacy_reference)

    active_parent = ScaleAlignedActiveWorldFlow.from_legacy_reference(
        legacy_reference, team_predictor
    )
    router_config = _router_config(raw, local_config, flow_config.action_dim)
    seed = int(_mapping(raw, "training").get("model_seed", 70707))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = UtilityCalibratedWorldFlow(
            active_parent,
            S4WorldEvidenceProvider(router_config),
            router_config,
        )
    model = model.to(device)
    identity = {
        **{f"{name}_path": str(path) for name, path in paths.items()},
        **{f"{name}_sha256": digest for name, digest in hashes.items()},
        "legacy_reference_model_sha256": legacy_model_hash,
        "initial_active_flow_model_sha256": state_dict_sha256(
            model.active_parent.base_flow
        ),
        "initial_active_local_model_sha256": state_dict_sha256(
            model.active_parent.future_predictor.local_predictor
        ),
        "initial_active_team_model_sha256": state_dict_sha256(
            model.active_parent.future_predictor
        ),
        "flow_task_vocabulary": list(S4_TASKS),
    }
    # Keep the immutable audit instance on CPU until an explicit exact test.
    legacy_reference.cpu()
    del flow_payload, local_payload, team_payload, legacy_payload
    return model, legacy_reference, identity


def _router_config(
    raw: Mapping[str, Any],
    local: LocalFuturePredictorConfig,
    action_dim: int,
) -> WorldEvidenceRouterConfig:
    model = _mapping(raw, "model")
    return WorldEvidenceRouterConfig(
        max_agents=local.max_agents,
        future_horizons=tuple(int(value) for value in model["evidence_horizons"]),
        visual_grid_tokens=local.visual_grid_tokens,
        state_dim=local.state_dim,
        visual_dim=local.visual_latent_dim,
        d_model=int(model["flow_query_dim"]),
        evidence_rank=int(model["evidence_rank"]),
        action_dim=action_dim,
        new_gate_max=float(model["new_gate_max"]),
    )


def _load_mapping(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a checkpoint mapping")
    return value


def _root_path(value: object) -> Path:
    root = Path(__file__).resolve().parents[1]
    return (root / str(value)).expanduser().resolve(strict=True)


__all__ = ["S4_TASKS", "build_s4_r7_model"]
