"""Shared parent loading and identity checks for S3-R6 train/inference."""

from __future__ import annotations

from collections.abc import Mapping
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
    WorldToFlowAdapterConfig,
)
from scripts.train_s2_r4_future_predictor import (
    CHECKPOINT_FORMAT as R4_CHECKPOINT_FORMAT,
    FLOW_FORMAT,
)
from scripts.train_s2_r5_protected_team import (
    CHECKPOINT_FORMAT as R5_CHECKPOINT_FORMAT,
)
from scripts.train_static_rgb_act_moe import _mapping
from train.s2_future_prediction import file_sha256, state_dict_sha256


def build_s3_r6_model(
    raw: Mapping[str, Any],
    *,
    device: torch.device,
    future_scope: str,
    injection: bool,
) -> tuple[CrossAgentWorldConditionedFlow, dict[str, Any]]:
    parent = _mapping(raw, "parent")
    flow_path = _root_path(parent["flow_checkpoint"])
    protected_path = _root_path(parent["protected_own_checkpoint"])
    team_path = _root_path(parent["protected_team_checkpoint"])
    flow_payload = torch.load(flow_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(flow_payload, Mapping)
        or flow_payload.get("format_version") != FLOW_FORMAT
        or _mapping(flow_payload, "method").get("action_generator")
        != "rectified_flow_cold"
    ):
        raise ValueError("S3-R6 requires the promoted S1-R1 cold Flow checkpoint")
    flow_config = StaticRGBMoEACTConfig.from_dict(
        _mapping(flow_payload, "model_config")
    )
    base_flow = AgentFactorizedFlowWAM(flow_config)
    base_flow.load_state_dict(flow_payload["model"], strict=True)
    flow_hash = file_sha256(flow_path)
    del flow_payload

    protected = torch.load(protected_path, map_location="cpu", weights_only=False)
    protected_method = _mapping(protected, "method")
    if (
        protected.get("format_version") != R4_CHECKPOINT_FORMAT
        or protected_method.get("candidate_id") != "P0"
        or protected_method.get("model_kind")
        != "s2_r4_local_action_conditioned"
        or protected_method.get("team_shared") is not False
    ):
        raise ValueError("S3-R6 requires the accepted protected R4-P0 checkpoint")
    protected_hash = file_sha256(protected_path)
    expected_protected = str(parent.get("expected_protected_own_sha256", ""))
    if expected_protected and protected_hash != expected_protected:
        raise ValueError("protected-own checkpoint hash differs from S3 contract")
    local_config = LocalFuturePredictorConfig.from_dict(
        dict(_mapping(protected, "model_config"))
    )
    configured_local = LocalFuturePredictorConfig.from_dict(
        dict(_mapping(raw, "world_model"))
    )
    if local_config != configured_local:
        raise ValueError("S3 local predictor config differs from protected parent")

    team = torch.load(team_path, map_location="cpu", weights_only=False)
    team_method = _mapping(team, "method")
    if (
        team.get("format_version") != R5_CHECKPOINT_FORMAT
        or team_method.get("candidate_id") != "P0"
        or team_method.get("model_kind") != "s2_r5_protected_shared_team"
        or team_method.get("team_mixer") != "shared"
    ):
        raise ValueError("S3-R6 team parent must be the accepted R5-P0 winner")
    team_hash = file_sha256(team_path)
    expected_team = str(parent.get("expected_protected_team_sha256", ""))
    if expected_team and team_hash != expected_team:
        raise ValueError("R5-P0 checkpoint hash differs from S3 contract")
    if _mapping(team, "protected_parent").get("checkpoint_sha256") != protected_hash:
        raise ValueError("R5-P0 was not trained above this protected-own parent")

    if future_scope == "local":
        future_predictor = LocalActionConditionedFuturePredictor(local_config)
        future_predictor.load_state_dict(protected["model"], strict=True)
    elif future_scope == "team_shared":
        team_config = ProtectedTeamFuturePredictorConfig.from_dict(
            dict(_mapping(team, "team_model_config"))
        )
        if team_config != ProtectedTeamFuturePredictorConfig.from_dict(
            dict(_mapping(raw, "team_model"))
        ):
            raise ValueError("S3 team predictor config differs from R5-P0")
        future_predictor = ProtectedTeamFuturePredictor(local_config, team_config)
        future_predictor.load_protected_own(protected["model"])
        future_predictor.load_team_state_dict(_mapping(team, "team_model"))
    else:
        raise ValueError("future_scope must be local or team_shared")
    protected_model_hash = state_dict_sha256(
        future_predictor.protected_own
        if isinstance(future_predictor, ProtectedTeamFuturePredictor)
        else future_predictor
    )
    adapter_config = WorldToFlowAdapterConfig.from_dict(
        _mapping(raw, "adapter")
    )
    if (
        adapter_config.flow_dim != flow_config.d_model
        or adapter_config.action_dim != flow_config.action_dim
        or adapter_config.state_dim != local_config.state_dim
        or adapter_config.visual_dim != local_config.visual_latent_dim
    ):
        raise ValueError("adapter dimensions disagree with frozen parents")
    adapter_seed = int(_mapping(raw, "training").get("adapter_seed", 60606))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(adapter_seed)
        model = CrossAgentWorldConditionedFlow(
            base_flow,
            future_predictor,
            adapter_config,
            future_scope=future_scope,
            injection=injection,
        ).to(device)
    identity = {
        "flow_checkpoint": str(flow_path),
        "flow_checkpoint_sha256": flow_hash,
        "protected_own_checkpoint": str(protected_path),
        "protected_own_checkpoint_sha256": protected_hash,
        "protected_own_model_sha256": protected_model_hash,
        "protected_team_checkpoint": str(team_path),
        "protected_team_checkpoint_sha256": team_hash,
    }
    del protected, team
    return model, identity


def _root_path(value: object) -> Path:
    root = Path(__file__).resolve().parents[1]
    return (root / str(value)).resolve(strict=True)


__all__ = ["build_s3_r6_model"]
