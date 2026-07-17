from __future__ import annotations

import copy
import inspect

import numpy as np
import torch

from eval.joint_wam_training import (
    evaluate_joint_wam_offline,
    joint_wam_offline_acceptance_report,
)
from models.wam import (
    NormalizationStats,
    RWMARConfig,
    RWMARWorldModel,
    StatefulActionFlow,
    StatefulActionFlowConfig,
    WorldModelSequenceInputs,
)
from scripts._train_joint_coupling import (
    _formal_acceptance_checks,
    build_parser as build_training_parser,
)
from train.joint_wam import (
    JointWAMTrainConfig,
    build_deployed_action_chunk,
    differentiable_flow_generate,
    generated_action_consistency_loss,
    rollout_frozen_prior_chunk,
    train_joint_wam_stage,
)


def test_differentiable_solver_matches_runtime_and_reaches_flow() -> None:
    world = _member()
    flow = _flow(world.planning_feature_dim)
    batch = _batch(3)
    _, _, features = world.encode_planning_history(_history(batch))
    runtime = flow.generate(features, solver_steps=4, solver="euler")
    generated = differentiable_flow_generate(
        flow, features, solver_steps=4, solver="euler"
    )
    torch.testing.assert_close(generated, runtime)
    generated.square().mean().backward()
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for name, parameter in flow.named_parameters()
        if not name.startswith("anchor_prior.")
    )
    assert all(parameter.grad is None for parameter in flow.anchor_prior.parameters())


def test_generated_consistency_has_no_demo_target_surface_and_detaches_teacher() -> None:
    joint = _member()
    teacher = copy.deepcopy(joint)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    flow = _flow(joint.planning_feature_dim)
    batch = _batch(2)
    history = _history(batch)
    hidden, state, features = joint.encode_planning_history(history)
    generated = differentiable_flow_generate(flow, features)
    deployed = build_deployed_action_chunk(
        torch.zeros_like(generated), generated, residual_scale=0.1
    )
    with torch.no_grad():
        joint.heads.delta_mean.bias.add_(0.01)
    result = generated_action_consistency_loss(
        joint, teacher, history, hidden, state, deployed
    )
    assert "target_states" not in inspect.signature(
        generated_action_consistency_loss
    ).parameters
    assert result.target_source == (
        "frozen_world_model_same_generated_actions"
    )
    assert result.demo_state_is_ground_truth is False
    assert result.teacher_targets_detached
    result.total.backward()
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for name, parameter in flow.named_parameters()
        if not name.startswith("anchor_prior.")
    )


def test_anchor_rollout_applies_fixed_actions_before_every_imagined_step() -> None:
    world = _member()
    flow = _flow(world.planning_feature_dim)
    batch = _batch(2)
    hidden, state, _ = world.encode_planning_history(_history(batch))
    observed: list[torch.Tensor] = []
    original_imagine_step = world.imagine_step

    def record_imagine_step(
        recurrent: torch.Tensor,
        current: torch.Tensor,
        action: torch.Tensor,
        *,
        sample_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        observed.append(action.detach().clone())
        return original_imagine_step(
            recurrent, current, action, sample_state=sample_state
        )

    world.imagine_step = record_imagine_step  # type: ignore[method-assign]
    chunk = rollout_frozen_prior_chunk(
        world,
        flow,
        hidden,
        state,
        steps=4,
        fixed_actions={3: 1.0, 7: 1.0},
    )

    assert len(observed) == 4
    assert torch.equal(chunk[..., 3], torch.ones_like(chunk[..., 3]))
    assert torch.equal(chunk[..., 7], torch.ones_like(chunk[..., 7]))
    assert all(torch.equal(action[:, 3], torch.ones_like(action[:, 3])) for action in observed)
    assert all(torch.equal(action[:, 7], torch.ones_like(action[:, 7])) for action in observed)


def test_staged_training_unfreezes_only_declared_scope_and_records_coupling() -> None:
    joint = _member()
    teacher = copy.deepcopy(joint)
    flow = _flow(joint.planning_feature_dim)
    batch = _batch(4)
    joint_initial = _state(joint)
    teacher_initial = _state(teacher)
    anchor_initial = _state(flow.anchor_prior)
    flow_initial = _state(flow)

    _, steps = train_joint_wam_stage(
        flow,
        joint,
        teacher,
        [batch],
        device=torch.device("cpu"),
        config=_train_config("flow_only", max_steps=1, member_lr=0.0),
        seed=1,
    )
    assert steps == 1
    assert _delta(joint_initial, joint.state_dict()) == 0.0
    assert _delta(flow_initial, flow.state_dict()) > 0.0

    before_heads = _state(joint)
    _, steps = train_joint_wam_stage(
        flow,
        joint,
        teacher,
        [batch],
        device=torch.device("cpu"),
        config=_train_config("world_heads", max_steps=1),
        seed=2,
    )
    assert steps == 1
    assert _prefix_delta(before_heads, joint.state_dict(), ("decoder.", "heads.")) > 0
    assert _prefix_delta(
        before_heads,
        joint.state_dict(),
        ("features.", "transition_encoder.", "belief_gru."),
    ) == 0.0

    before_full = _state(joint)
    history, steps = train_joint_wam_stage(
        flow,
        joint,
        teacher,
        [batch, batch],
        device=torch.device("cpu"),
        config=_train_config("full_joint", max_steps=2),
        seed=3,
    )
    assert steps == 2
    assert _prefix_delta(
        before_full,
        joint.state_dict(),
        ("features.", "transition_encoder.", "belief_gru."),
    ) > 0.0
    assert max(item["action_to_backbone_gradient_norm"] for item in history) > 0.0
    assert max(item["world_to_backbone_gradient_norm"] for item in history) > 0.0
    assert max(item["consistency_to_flow_gradient_norm"] for item in history) > 0.0
    assert max(item["consistency_to_backbone_gradient_norm"] for item in history) > 0.0
    assert _delta(teacher_initial, teacher.state_dict()) == 0.0
    assert _delta(anchor_initial, flow.anchor_prior.state_dict()) == 0.0


def test_offline_evaluation_accepts_strict_joint_evidence() -> None:
    joint = _member()
    teacher = copy.deepcopy(joint)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    flow = _flow(joint.planning_feature_dim)
    with torch.no_grad():
        joint.heads.delta_mean.bias.add_(1e-4)
    metrics = evaluate_joint_wam_offline(
        joint,
        teacher,
        flow,
        [_batch(2)],
        device=torch.device("cpu"),
        fixed_actions={3: 1.0, 7: 1.0},
    )
    assert metrics["all_offline_values_finite"]
    assert metrics["all_generated_actions_bounded"]
    assert metrics["generated_action_demo_state_is_ground_truth"] is False
    assert metrics["generated_action_world_target_source"] == (
        "frozen_world_model_same_generated_actions"
    )
    report = joint_wam_offline_acceptance_report(
        metrics,
        member_0_parameter_delta=1e-4,
        shared_history_parameter_delta=1e-6,
        world_parameter_delta=1e-4,
        action_flow_parameter_delta=1e-4,
        anchor_prior_parameter_delta=0.0,
        frozen_teacher_parameter_delta=0.0,
        source_checkpoints_immutable=True,
        checkpoint_reload_max_abs_diff=0.0,
        branch_gradient_maxima={
            "action_to_flow_gradient_norm": 1e-3,
            "action_to_backbone_gradient_norm": 1e-3,
            "world_to_backbone_gradient_norm": 1e-3,
            "consistency_to_flow_gradient_norm": 1e-4,
            "consistency_to_backbone_gradient_norm": 1e-4,
        },
        maximum_expert_world_nrmse=10.0,
        maximum_generated_teacher_state_nrmse=10.0,
    )
    failed_checks = [name for name, passed in report["checks"].items() if not passed]
    assert report["passed"], failed_checks

    fuzzy_gradient_report = joint_wam_offline_acceptance_report(
        metrics,
        member_0_parameter_delta=1e-4,
        shared_history_parameter_delta=1e-6,
        world_parameter_delta=1e-4,
        action_flow_parameter_delta=1e-4,
        anchor_prior_parameter_delta=0.0,
        frozen_teacher_parameter_delta=0.0,
        source_checkpoints_immutable=True,
        checkpoint_reload_max_abs_diff=0.0,
        branch_gradient_maxima={
            "action_flow_loss": 1.0,
            "world_learning_rate": 1.0,
            "consistency_loss": 1.0,
        },
    )
    assert not fuzzy_gradient_report["passed"]
    assert not fuzzy_gradient_report["checks"][
        "action_to_backbone_gradient_nonzero"
    ]

    non_finite_auxiliary = copy.deepcopy(metrics)
    non_finite_auxiliary["all_offline_values_finite"] = False
    non_finite_report = joint_wam_offline_acceptance_report(
        non_finite_auxiliary,
        member_0_parameter_delta=1e-4,
        shared_history_parameter_delta=1e-6,
        world_parameter_delta=1e-4,
        action_flow_parameter_delta=1e-4,
        anchor_prior_parameter_delta=0.0,
        frozen_teacher_parameter_delta=0.0,
        source_checkpoints_immutable=True,
        checkpoint_reload_max_abs_diff=0.0,
        branch_gradient_maxima=report["branch_gradient_maxima"],
    )
    assert not non_finite_report["passed"]
    assert not non_finite_report["checks"]["all_offline_values_finite"]

    relabelled = copy.deepcopy(metrics)
    relabelled["teacher_consistency"]["cold_generated"][
        "target_source"
    ] = "dataset_demo_future"
    relabelled_report = joint_wam_offline_acceptance_report(
        relabelled,
        member_0_parameter_delta=1e-4,
        shared_history_parameter_delta=1e-6,
        world_parameter_delta=1e-4,
        action_flow_parameter_delta=1e-4,
        anchor_prior_parameter_delta=0.0,
        frozen_teacher_parameter_delta=0.0,
        source_checkpoints_immutable=True,
        checkpoint_reload_max_abs_diff=0.0,
        branch_gradient_maxima=report["branch_gradient_maxima"],
    )
    assert not relabelled_report["passed"]
    assert not relabelled_report["checks"]["teacher_consistency_target_contract"]


def test_partial_training_guards_fail_closed() -> None:
    parser = build_training_parser()
    formal = parser.parse_args([])
    assert all(
        _formal_acceptance_checks(
            formal,
            completed_steps=704,
            configured_steps=704,
            smoke_subset=False,
        ).values()
    )
    for arguments in (
        ["--max-steps", "1"],
        ["--max-eval-batches", "1"],
        ["--max-episodes-per-split", "1"],
        ["--batch-size", "1"],
    ):
        diagnostic = parser.parse_args(arguments)
        checks = _formal_acceptance_checks(
            diagnostic,
            completed_steps=704,
            configured_steps=704,
            smoke_subset=diagnostic.max_episodes_per_split > 0,
        )
        assert not all(checks.values())
        assert checks["no_debug_limits"] is False


def _train_config(
    scope: str, *, max_steps: int, member_lr: float = 1e-4
) -> JointWAMTrainConfig:
    return JointWAMTrainConfig(
        scope=scope,
        epochs=1,
        flow_learning_rate=1e-3,
        member_learning_rate=member_lr,
        use_amp=False,
        max_steps=max_steps,
        world_horizon=8,
        generated_action_ratio_start=1.0,
        generated_action_ratio_end=1.0,
        gradient_audit_interval=1,
    )


def _history(batch: dict[str, torch.Tensor]) -> WorldModelSequenceInputs:
    return WorldModelSequenceInputs(
        states=batch["states"],
        past_actions=batch["past_actions"],
        valid_mask=batch["valid_mask"],
    )


def _batch(batch_size: int) -> dict[str, torch.Tensor]:
    states = torch.randn(batch_size, 3, 22) * 0.01
    actions = torch.randn(batch_size, 8, 8).clamp(-0.5, 0.5)
    actions[..., 3] = 1.0
    actions[..., 7] = 1.0
    target_states = states[:, -1:, :].expand(-1, 8, -1).clone()
    target_states = target_states + torch.randn_like(target_states) * 0.01
    zeros = torch.zeros(batch_size, 8, 1)
    return {
        "states": states,
        "past_actions": torch.zeros(batch_size, 2, 8),
        "valid_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "candidate_actions": actions,
        "action_quality_weights": torch.ones(batch_size, 1),
        "target_states": target_states,
        "forecast_mask": torch.ones(batch_size, 8, dtype=torch.bool),
        "rewards": zeros.clone(),
        "dones": zeros.clone(),
        "successes": zeros.clone(),
        "failures": zeros.clone(),
        "response_progress": zeros.clone(),
        "coordination_error": zeros.clone(),
        "executed_actions": actions.clone(),
    }


def _member() -> RWMARWorldModel:
    torch.manual_seed(5)
    return RWMARWorldModel(
        RWMARConfig(
            history_horizon=3,
            train_forecast_horizon=8,
            planning_horizon=8,
            encoder_hidden_dim=16,
            gru_hidden_dim=16,
            gru_layers=1,
        ),
        _stats(),
    )


def _flow(feature_dim: int) -> StatefulActionFlow:
    torch.manual_seed(7)
    return StatefulActionFlow(
        StatefulActionFlowConfig(
            feature_dim=feature_dim,
            action_dim=8,
            horizon=8,
            hidden_dim=32,
            hidden_layers=1,
            time_embedding_dim=8,
        ),
        _stats(),
    )


def _stats() -> NormalizationStats:
    return NormalizationStats(
        state_mean=np.zeros(22, dtype=np.float32),
        state_std=np.ones(22, dtype=np.float32),
        action_mean=np.zeros(8, dtype=np.float32),
        action_std=np.ones(8, dtype=np.float32),
        delta_mean=np.zeros(22, dtype=np.float32),
        delta_std=np.ones(22, dtype=np.float32),
        reward_mean=np.zeros(1, dtype=np.float32),
        reward_std=np.ones(1, dtype=np.float32),
    )


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _delta(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> float:
    return max(
        (
            float((before[name] - value.detach()).abs().max())
            if torch.is_floating_point(value)
            else (0.0 if torch.equal(before[name], value.detach()) else float("inf"))
        )
        for name, value in after.items()
    )


def _prefix_delta(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> float:
    names = [name for name in before if name.startswith(prefixes)]
    return _delta(
        {name: before[name] for name in names},
        {name: after[name] for name in names},
    )
