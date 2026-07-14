"""Load one independent Research-v2 local runtime from an attested bundle."""

from __future__ import annotations

from dataclasses import replace
import json
import hashlib
from pathlib import Path

import torch

from data.local_observation import LocalObservationSpec
from models.plan_tokenizer import PlanCodeSupport
from models.research_v2 import (
    BeliefEncoderV2,
    BeliefEncoderV2Config,
    BlockTransitionWorldModelV2,
    IntentionPosteriorV2,
    PlanDistributionV2Config,
    PlanProposalV2,
    PlanTokenizerV2,
    PlanTokenizerV2Config,
    WorldModelV2Config,
)
from models.research_v2_decision import CalibrationV2, RiskV2Config
from policies.research_v2 import LocalPlannerV2, MessageCodecV2, PlannerV2Config
from policies.research_v2_runtime import LocalRuntimeV2
from train.research_v2_checkpoint import load_research_v2_checkpoint, sha256_file


def load_independent_local_runtime_v2(
    bundle_manifest: str | Path,
    *,
    agent_id: int,
    device: str | torch.device = "cpu",
    payload_dim: int = 16,
    planner_config: PlannerV2Config | None = None,
    risk_config: RiskV2Config | None = None,
) -> LocalRuntimeV2:
    manifest_path = Path(bundle_manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("bundle_contract") != "fe_pc_wam/research_v2_runtime_bundle":
        raise ValueError("not a Research-v2 runtime bundle")
    if payload.get("privileged_runtime_inputs") != []:
        raise ValueError("runtime bundle declares privileged inputs")
    claimed_bundle_hash = payload.get("bundle_sha256")
    unsigned = dict(payload)
    unsigned.pop("bundle_sha256", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != claimed_bundle_hash:
        raise ValueError("runtime bundle manifest hash mismatch")
    artifacts = payload["artifacts"]
    for artifact in artifacts.values():
        artifact["resolved_path"] = str(_resolve_bundle_path(manifest_path, artifact["path"]))
        if sha256_file(artifact["resolved_path"]) != artifact["sha256"]:
            raise ValueError("runtime artifact hash mismatch")
    target = torch.device(device)
    plan_state = load_research_v2_checkpoint(artifacts["plan"]["resolved_path"], expected_stage="plan", map_location=target)
    belief_state = load_research_v2_checkpoint(artifacts["belief"]["resolved_path"], expected_stage="belief", map_location=target)
    proposal_state = load_research_v2_checkpoint(artifacts["proposal"]["resolved_path"], expected_stage="proposal", map_location=target)
    intention_state = load_research_v2_checkpoint(artifacts["intention"]["resolved_path"], expected_stage="intention", map_location=target)
    calibration_state = load_research_v2_checkpoint(
        artifacts["calibration"]["resolved_path"],
        expected_stage="calibration",
        map_location=target,
    )
    calibration = CalibrationV2.from_mapping(calibration_state["extra"])
    tokenizer = PlanTokenizerV2(PlanTokenizerV2Config(**plan_state["model_config"])).to(target)
    tokenizer.load_state_dict(plan_state["model_state_dict"])
    belief = BeliefEncoderV2(BeliefEncoderV2Config(**belief_state["model_config"])).to(target)
    belief_extra = belief_state.get("extra", {})
    belief_deployment_key = belief_extra.get("deployment_state_dict_key")
    if belief_deployment_key is None:
        # Compatibility for early Stage-A smoke artifacts, before the
        # deployment-weight choice became an explicit checkpoint contract.
        belief_weights = belief_state["model_state_dict"]
    else:
        if belief_deployment_key not in belief_extra:
            raise ValueError(
                "belief checkpoint declares missing deployment weights "
                f"{belief_deployment_key!r}"
            )
        belief_weights = belief_extra[belief_deployment_key]
    belief.load_state_dict(belief_weights)
    proposal = PlanProposalV2(PlanDistributionV2Config(**proposal_state["model_config"])).to(target)
    proposal.load_state_dict(proposal_state["model_state_dict"])
    intention = IntentionPosteriorV2(PlanDistributionV2Config(**intention_state["model_config"])).to(target)
    intention.load_state_dict(intention_state["model_state_dict"])
    worlds = []
    world_hashes: list[str] = []
    world_states: list[dict] = []
    for member in payload["world_ensemble"]:
        member_path = _resolve_bundle_path(manifest_path, member["path"])
        if sha256_file(member_path) != member["sha256"]:
            raise ValueError("world ensemble hash mismatch")
        state = load_research_v2_checkpoint(member_path, expected_stage="world_block", map_location=target)
        world_states.append(state)
        model = BlockTransitionWorldModelV2(WorldModelV2Config(**state["model_config"])).to(target)
        model.load_state_dict(state["model_state_dict"])
        worlds.append(model)
        world_hashes.append(str(member["sha256"]))
    if not worlds:
        raise ValueError("runtime bundle contains no world-model members")
    if world_hashes[0] != artifacts["world_block"]["sha256"]:
        raise ValueError("runtime primary world is not ensemble member zero")

    def require_upstream(state, artifact_name, upstream_name, expected_hash):
        reference = state.get("upstream", {}).get(upstream_name)
        if reference is None or reference.get("sha256") != expected_hash:
            raise ValueError(
                f"{artifact_name} has stale/missing {upstream_name} lineage"
            )

    plan_hash = artifacts["plan"]["sha256"]
    belief_hash = artifacts["belief"]["sha256"]
    primary_world_hash = artifacts["world_block"]["sha256"]
    intention_hash = artifacts["intention"]["sha256"]
    require_upstream(belief_state, "belief", "plan", plan_hash)
    for index, state in enumerate(world_states):
        require_upstream(state, f"world member {index}", "plan", plan_hash)
        require_upstream(state, f"world member {index}", "belief", belief_hash)
    for name, state in (("proposal", proposal_state), ("intention", intention_state)):
        require_upstream(state, name, "plan", plan_hash)
        require_upstream(state, name, "belief", belief_hash)
        require_upstream(state, name, "world_block", primary_world_hash)
    require_upstream(calibration_state, "calibration", "plan", plan_hash)
    require_upstream(calibration_state, "calibration", "belief", belief_hash)
    require_upstream(
        calibration_state, "calibration", "world_block", primary_world_hash
    )
    require_upstream(
        calibration_state, "calibration", "intention", intention_hash
    )
    if calibration_state["extra"]["world_ensemble_sha256"] != world_hashes:
        raise ValueError("calibration was fit on a different world ensemble")
    for index, expected_hash in enumerate(world_hashes):
        require_upstream(
            calibration_state,
            "calibration",
            f"world_block_member_{index:02d}",
            expected_hash,
        )
    computed_epistemic_available = len(set(world_hashes)) >= 2
    declared_epistemic_available = payload.get("epistemic_uncertainty_available")
    if (
        declared_epistemic_available is not None
        and bool(declared_epistemic_available) != computed_epistemic_available
    ):
        raise ValueError("runtime bundle epistemic-availability declaration is inconsistent")
    for model in (tokenizer, belief, proposal, intention, *worlds):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    support = PlanCodeSupport.from_dict(plan_state["extra"]["plan_support"])
    active = torch.zeros(support.codebook_size, dtype=torch.bool)
    active[support.active_codes] = True
    resolved_planner_config = planner_config or PlannerV2Config()
    if bool(plan_state.get("training_config", {}).get("smoke", False)):
        active_count = int(active.sum().item())
        resolved_planner_config = replace(
            resolved_planner_config,
            num_candidates=min(resolved_planner_config.num_candidates, active_count),
            num_hypotheses=min(resolved_planner_config.num_hypotheses, active_count),
        )
    normalization = plan_state["extra"]["normalization"]
    planner = LocalPlannerV2(
        agent_id,
        tokenizer=tokenizer,
        proposal=proposal,
        intention=intention,
        world_ensemble=worlds,
        active_code_mask=active,
        residual_prior_by_code=support.residual_mean,
        action_mean=torch.as_tensor(normalization["action_mean"], device=target),
        action_std=torch.as_tensor(normalization["action_std"], device=target),
        artifact_hash=payload["bundle_sha256"],
        codec=MessageCodecV2(residual_dim=tokenizer.cfg.latent_dim, payload_dim=payload_dim),
        config=resolved_planner_config,
        risk_config=risk_config,
        calibration=calibration,
        epistemic_available=computed_epistemic_available,
    )
    spec = LocalObservationSpec(**plan_state["extra"]["local_observation_spec"])
    return LocalRuntimeV2(agent_id, spec=spec, belief=belief, planner=planner)


def _resolve_bundle_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()
