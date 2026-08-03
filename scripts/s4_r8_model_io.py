"""R8 model assembly on top of the exact common S4 active clone."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from models.wam_multimodal import (
    CrossAgentWorldConditionedFlow,
    HorizonCausalActiveTeamFutureProvider,
    UtilityCalibratedWorldFlow,
)
from scripts.s4_r7_model_io import S4_TASKS, build_s4_r7_model
from scripts.train_static_rgb_act_moe import _mapping
from train.s2_future_prediction import state_dict_sha256
from train.s4_model_registry import validate_s4_r8_candidate


def build_s4_r8_model(
    raw: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[
    UtilityCalibratedWorldFlow,
    CrossAgentWorldConditionedFlow,
    dict[str, Any],
]:
    """Build an R8 candidate without consuming any R7 candidate checkpoint."""

    candidate_id, model_kind, aggregator = validate_s4_r8_candidate(raw)
    model_config = _mapping(raw, "model")
    rank = int(model_config.get("action_prefix_rank", 32))
    if rank != 32:
        raise ValueError("S4-R8 causal action prefix rank is fixed at 32")
    model, legacy_reference, identity = build_s4_r7_model(raw, device=device)
    common_provider = model.active_parent.future_predictor
    seed = int(_mapping(raw, "training").get("model_seed", 70707))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 8008)
        replacement = HorizonCausalActiveTeamFutureProvider.from_active_provider(
            common_provider,
            action_prefix_aggregator=aggregator,
            action_prefix_rank=rank,
        ).to(device)
    model.active_parent.future_predictor = replacement
    identity.update(
        {
            "round_id": "s4-r8",
            "candidate_id": candidate_id,
            "model_kind": model_kind,
            "action_prefix_aggregator": aggregator,
            "action_prefix_rank": rank,
            "r7_candidate_checkpoint_consumed": False,
            "r7_method_axis_fixed": "utility_coupling_weight=0",
            "initial_active_local_model_sha256": state_dict_sha256(
                replacement.local_predictor
            ),
            "initial_active_team_model_sha256": state_dict_sha256(replacement),
            "action_prefix_aggregator_audit": (
                replacement.action_prefix_aggregator.audit()
            ),
        }
    )
    return model, legacy_reference, identity


__all__ = ["S4_TASKS", "build_s4_r8_model"]
