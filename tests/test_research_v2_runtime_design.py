from __future__ import annotations

from dataclasses import asdict
import json

import pytest
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
from models.research_v2_decision import (
    CalibrationV2,
    RiskV2Config,
    calibrated_posterior_probabilities,
    candidate_hypothesis_risk,
    counterfactual_vpi,
)
from policies.research_v2 import PlannerV2Config
from policies.research_v2_loader import load_independent_local_runtime_v2
from train.research_v2_checkpoint import (
    checkpoint_reference,
    load_research_v2_checkpoint,
    make_research_v2_checkpoint,
    save_research_v2_checkpoint,
    sha256_file,
    write_runtime_bundle_manifest,
)


def test_planner_rejects_unimplemented_multistep_message_ttl() -> None:
    assert PlannerV2Config().plan_valid_steps == 0
    with pytest.raises(ValueError, match="current-step"):
        PlannerV2Config(plan_valid_steps=1)


def test_risk_applies_calibration_projects_crossing_and_marks_single_member() -> None:
    quantiles = torch.tensor([[[[[3.0, 1.0, 4.0]]]]])
    constraints = torch.tensor([[[[2.0]]]])
    actions = torch.zeros(1, 1, 4, 4)
    calibration = CalibrationV2(
        quantile_scale=2.0,
        quantile_bias=1.0,
        constraint_temperature=2.0,
        constraint_logit_bias=-0.5,
    )
    risk = candidate_hypothesis_risk(
        ensemble_return_quantiles=quantiles,
        ensemble_constraint_logits=constraints,
        ego_actions=actions,
        hypothesis_residual_variance=torch.tensor([[2.0]]),
        config=RiskV2Config(lambda_residual_uncertainty=0.25),
        calibration=calibration,
    )
    torch.testing.assert_close(risk["q10"], torch.tensor([[[7.0]]]))
    torch.testing.assert_close(risk["q50"], torch.tensor([[[7.0]]]))
    torch.testing.assert_close(risk["tail"], torch.zeros(1, 1, 1))
    torch.testing.assert_close(
        risk["constraint_probability"], torch.tensor(0.5).sigmoid().reshape(1, 1, 1)
    )
    assert risk["quantile_crossing_rate_before_projection"].item() > 0
    assert risk["epistemic_available"].item() is False
    assert risk["epistemic_weight_applied"].item() == 0.0
    torch.testing.assert_close(risk["residual_uncertainty"], torch.full((1, 1, 1), 2.0))


def test_posterior_temperature_and_vpi_keep_unmodeled_tail_mass() -> None:
    logits = torch.tensor([[0.0, 2.0, -2.0]])
    cold = calibrated_posterior_probabilities(
        logits,
        active_code_mask=torch.ones(3, dtype=torch.bool),
        calibration=CalibrationV2(posterior_temperature=0.5),
    )
    warm = calibrated_posterior_probabilities(
        logits,
        active_code_mask=torch.ones(3, dtype=torch.bool),
        calibration=CalibrationV2(posterior_temperature=2.0),
    )
    assert cold[0, 1] > warm[0, 1]

    G = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]])
    result = counterfactual_vpi(
        G,
        torch.tensor([[0.3, 0.3]]),
        tail_weight=torch.tensor([0.4]),
        tail_risk=torch.tensor([[10.0, 5.0]]),
    )
    # Candidate 1: .3*2 + .3*0 + .4*5 = 2.6.  Renormalizing top-M
    # would instead (incorrectly) choose based only on a cost of 1.0.
    assert result["G_no"].item() == pytest.approx(2.6)
    assert result["normalized_tail_weight"].item() == pytest.approx(0.4)
    assert result["VPI"].item() >= 0.0


def _save_runtime_bundle(tmp_path, *, deployment_ema_value: float | None = None) -> str:
    tokenizer_cfg = PlanTokenizerV2Config(
        horizon=4, codebook_size=8, residual_dim=16, hidden_dim=32
    )
    belief_cfg = BeliefEncoderV2Config(
        history=3,
        local_dim=21,
        model_dim=16,
        num_heads=4,
        temporal_layers=1,
        role_layers=1,
        ffn_dim=32,
        dropout=0.0,
    )
    distribution_cfg = PlanDistributionV2Config(
        belief_dim=16,
        codebook_size=8,
        residual_dim=16,
        model_dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
    )
    world_cfg = WorldModelV2Config(
        horizon=4,
        block_length=2,
        belief_dim=16,
        model_dim=32,
        context_layers=1,
        transition_layers=1,
        heads=4,
        ffn_dim=64,
        dropout=0.0,
    )
    modules = {
        "plan": (PlanTokenizerV2(tokenizer_cfg), tokenizer_cfg),
        "belief": (BeliefEncoderV2(belief_cfg), belief_cfg),
        "world_block": (BlockTransitionWorldModelV2(world_cfg), world_cfg),
        "proposal": (PlanProposalV2(distribution_cfg), distribution_cfg),
        "intention": (IntentionPosteriorV2(distribution_cfg), distribution_cfg),
    }
    support = PlanCodeSupport(
        codebook_size=8,
        min_count=1,
        counts=torch.ones(8, dtype=torch.long),
        probabilities=torch.full((8,), 1.0 / 8.0),
        residual_mean=torch.zeros(8, 16),
        residual_std=torch.ones(8, 16),
    )
    paths = {}
    for stage, (module, cfg) in modules.items():
        extra = {}
        if stage == "plan":
            extra = {
                "plan_support": support.to_dict(),
                "normalization": {
                    "action_mean": [0.0] * 4,
                    "action_std": [1.0] * 4,
                },
                "local_observation_spec": asdict(LocalObservationSpec()),
            }
        elif stage == "belief" and deployment_ema_value is not None:
            ema_state = {
                name: (
                    torch.full_like(value, deployment_ema_value)
                    if value.is_floating_point()
                    else value.clone()
                )
                for name, value in module.state_dict().items()
            }
            extra = {
                "deployment_state_dict_key": "ema_model_state_dict",
                "ema_model_state_dict": ema_state,
            }
        upstream = {}
        if stage == "belief":
            upstream = {
                "plan": checkpoint_reference(
                    paths["plan"], load_research_v2_checkpoint(paths["plan"])
                )
            }
        elif stage == "world_block":
            upstream = {
                name: checkpoint_reference(
                    paths[name], load_research_v2_checkpoint(paths[name])
                )
                for name in ("plan", "belief")
            }
        elif stage in {"proposal", "intention"}:
            upstream = {
                name: checkpoint_reference(
                    paths[name], load_research_v2_checkpoint(paths[name])
                )
                for name in ("plan", "belief")
            }
            upstream["world_block"] = checkpoint_reference(
                paths["world_block"],
                load_research_v2_checkpoint(paths["world_block"]),
            )
        checkpoint = make_research_v2_checkpoint(
            stage=stage,
            model_class=type(module).__name__,
            model_config=cfg,
            model_state_dict=module.state_dict(),
            training_config={"smoke": False},
            dataset_manifest_sha256="dataset-hash",
            forward_inputs=tuple(getattr(module, "INPUT_NAMES", ("belief",))),
            metrics={},
            upstream=upstream,
            extra=extra,
        )
        paths[stage] = save_research_v2_checkpoint(tmp_path / stage / "best.pt", checkpoint)

    calibration = make_research_v2_checkpoint(
        stage="calibration",
        model_class="FrozenCalibrationV2",
        model_config={"version": 1},
        model_state_dict={},
        training_config={"smoke": False},
        dataset_manifest_sha256="dataset-hash",
        forward_inputs=("return_quantiles", "constraint_logits", "posterior_logits"),
        metrics={},
        upstream={
            "plan": checkpoint_reference(
                paths["plan"], load_research_v2_checkpoint(paths["plan"])
            ),
            "belief": checkpoint_reference(
                paths["belief"], load_research_v2_checkpoint(paths["belief"])
            ),
            "world_block": checkpoint_reference(
                paths["world_block"],
                load_research_v2_checkpoint(paths["world_block"]),
            ),
            "world_block_member_00": checkpoint_reference(
                paths["world_block"],
                load_research_v2_checkpoint(paths["world_block"]),
            ),
            "intention": checkpoint_reference(
                paths["intention"],
                load_research_v2_checkpoint(paths["intention"]),
            ),
        },
        extra={
            "quantile_scale": 1.5,
            "quantile_bias": 0.25,
            "constraint_temperature": 2.0,
            "constraint_logit_bias": -0.1,
            "posterior_temperature": 0.75,
            "posterior_variance_scale": 1.25,
            "communication_price_frozen": True,
            "communication_price": 3.5,
            "world_ensemble_size": 1,
            "world_ensemble_sha256": [sha256_file(paths["world_block"])],
        },
    )
    paths["calibration"] = save_research_v2_checkpoint(
        tmp_path / "calibration" / "best.pt", calibration
    )
    return str(
        write_runtime_bundle_manifest(
            tmp_path / "bundle",
            paths,
            ensemble_members=[paths["world_block"]],
            parameter_counts={},
        )
    )


def test_loader_applies_calibration_and_planner_marginalizes_all_candidates(tmp_path) -> None:
    bundle = _save_runtime_bundle(tmp_path)
    manifest = json.loads(open(bundle, encoding="utf-8").read())
    assert manifest["world_ensemble_size"] == 1
    assert manifest["epistemic_uncertainty_available"] is False

    runtime = load_independent_local_runtime_v2(
        bundle,
        agent_id=0,
        planner_config=PlannerV2Config(num_candidates=2, num_hypotheses=2),
    )
    planner = runtime.planner
    assert planner.calibration.quantile_scale == 1.5
    assert planner.calibration.posterior_temperature == 0.75
    assert planner.config.communication_cost == 3.5
    assert planner.epistemic_available is False

    intention_batch_sizes = []

    def record_batch(_module, args):
        intention_batch_sizes.append(args[0].shape[0])

    hook = planner.intention.register_forward_pre_hook(record_batch)
    try:
        planner.prepare(torch.randn(4, 16))
    finally:
        hook.remove()
    assert intention_batch_sizes == [2]
    assert planner._pending is not None
    diagnostics = planner._pending.posterior_diagnostics
    assert diagnostics["intention_conditioning"] == "proposal_marginalized_over_all_candidates"
    assert diagnostics["posterior_tail_probability"] > 0.0
    assert bool((planner._pending.risk["residual_uncertainty"] > 0).all())
    planner.finalize()


def test_runtime_loads_declared_ema_belief_deployment_weights(tmp_path) -> None:
    bundle = _save_runtime_bundle(tmp_path, deployment_ema_value=0.125)
    runtime = load_independent_local_runtime_v2(bundle, agent_id=0)
    first_parameter = next(runtime.belief.parameters())
    torch.testing.assert_close(first_parameter, torch.full_like(first_parameter, 0.125))
