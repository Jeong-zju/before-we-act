"""Joint world/action coupling for the final Joint WAM.

The Joint WAM trainer deliberately keeps three roles separate:

* ``joint_model`` is the initialization world-model copy that may be unfrozen;
* ``frozen_teacher`` is an independent, immutable initialization world-model copy;
* ``flow`` is initialized from the accepted action-flow warm-up artifact and keeps its embedded
  action-prior anchor frozen.

Generated-action consistency never treats the demonstration next state as the
outcome of a different generated action.  Its targets are detached predictions
from ``frozen_teacher`` evaluated on exactly the deployed action chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.wam import (
    RWMARWorldModel,
    StatefulActionFlow,
    WorldModelSequenceInputs,
)
from models.wam.rollout import wrap_to_pi
from train.action_flow import synthetic_shifted_warm_start
from train.rwm_ar_losses import RWMLossWeights, compute_rwm_loss

JointWAMScope = Literal["flow_only", "world_heads", "full_joint"]
ProgressCallback = Callable[[Mapping[str, Any]], None]
GENERATED_ACTION_TARGET_SOURCE = "frozen_world_model_same_generated_actions"


@dataclass(frozen=True)
class JointWAMTrainConfig:
    """One Joint WAM optimization stage.

    The action flow is trainable in every scope.  ``scope`` controls which
    parameters of the Joint world-model copy are additionally trainable:

    * ``flow_only``: no Joint world-model parameters;
    * ``world_heads``: the action-conditioned decoder and prediction heads;
    * ``full_joint``: history encoder, recurrent belief, decoder, and heads.

    Calling :func:`train_joint_wam_stage` once per scope makes the curriculum
    explicit in logs and prevents a silent all-at-once unfreeze.
    """

    scope: JointWAMScope = "flow_only"
    name: str = ""
    epochs: int = 1
    flow_learning_rate: float = 2e-5
    world_model_learning_rate: float = 2e-6
    weight_decay: float = 1e-5
    flow_gradient_clip_norm: float = 10.0
    world_model_gradient_clip_norm: float = 1.0
    use_amp: bool = True
    max_steps: int = -1

    # Action Flow-Matching objective.
    warm_start_probability: float = 0.5
    warm_start_noise_std: float = 0.01
    cold_noise_std: float = 1.0
    cold_zero_probability: float = 0.5
    execution_steps: int = 2
    action_endpoint_weight: float = 1.0
    action_smoothness_weight: float = 0.01

    # Expert-action world loss.  It always consumes the complete batch,
    # including failed/zero-action-quality trajectories.
    world_horizon: int = 8
    world_horizon_decay: float = 0.95
    world_loss_weights: RWMLossWeights = field(default_factory=RWMLossWeights)

    # Differentiable generated/deployed action path.
    solver_steps: int = 4
    solver: str = "euler"
    normalized_action_clip: float = 10.0
    anchor_residual_scale: float = 0.10
    generated_warm_start_probability: float = 0.5
    generated_action_ratio_start: float = 0.0
    generated_action_ratio_end: float = 0.5
    fixed_actions: tuple[tuple[int, float], ...] = ((3, 1.0), (7, 1.0))

    # Top-level and within-consistency weights.
    action_loss_weight: float = 1.0
    world_loss_weight: float = 1.0
    generated_consistency_weight: float = 0.1
    generated_state_weight: float = 1.0
    generated_risk_weight: float = 0.1
    generated_progress_weight: float = 0.1

    # Six branch-gradient probes are intentionally periodic because each audit
    # uses autograd.grad while preserving the graph for the real backward pass.
    gradient_audit_interval: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if self.scope not in {"flow_only", "world_heads", "full_joint"}:
            raise ValueError("scope must be flow_only, world_heads, or full_joint")
        for name in ("epochs", "execution_steps", "world_horizon", "solver_steps"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")
        if self.gradient_audit_interval <= 0:
            raise ValueError("gradient_audit_interval must be positive")
        for name in (
            "flow_learning_rate",
            "flow_gradient_clip_norm",
            "world_model_gradient_clip_norm",
            "normalized_action_clip",
            "anchor_residual_scale",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.world_model_learning_rate < 0.0:
            raise ValueError("world_model_learning_rate must be non-negative")
        if self.scope != "flow_only" and self.world_model_learning_rate == 0.0:
            raise ValueError(
                "trainable world-model scopes require a positive world-model LR"
            )
        if self.world_model_learning_rate >= self.flow_learning_rate:
            raise ValueError(
                "world_model_learning_rate must be smaller than flow_learning_rate"
            )
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        for name in (
            "warm_start_probability",
            "cold_zero_probability",
            "generated_warm_start_probability",
            "generated_action_ratio_start",
            "generated_action_ratio_end",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.generated_action_ratio_end < self.generated_action_ratio_start:
            raise ValueError("generated-action ratio must be non-decreasing")
        if self.execution_steps <= 0:
            raise ValueError("execution_steps must be positive")
        if self.solver not in {"euler", "heun"}:
            raise ValueError("solver must be euler or heun")
        if not 0.0 < self.anchor_residual_scale <= 1.0:
            raise ValueError("anchor_residual_scale must be in (0,1]")
        for name in (
            "warm_start_noise_std",
            "cold_noise_std",
            "action_endpoint_weight",
            "action_smoothness_weight",
            "action_loss_weight",
            "world_loss_weight",
            "generated_consistency_weight",
            "generated_state_weight",
            "generated_risk_weight",
            "generated_progress_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 < self.world_horizon_decay <= 1.0:
            raise ValueError("world_horizon_decay must be in (0,1]")
        fixed_indices: set[int] = set()
        normalized_fixed: list[tuple[int, float]] = []
        for raw_index, raw_value in self.fixed_actions:
            index, value = int(raw_index), float(raw_value)
            if index < 0 or index in fixed_indices:
                raise ValueError("fixed action indices must be unique and non-negative")
            if not -1.0 <= value <= 1.0:
                raise ValueError("fixed action values must be in [-1,1]")
            fixed_indices.add(index)
            normalized_fixed.append((index, value))
        object.__setattr__(self, "fixed_actions", tuple(normalized_fixed))


# A stage is the unit configured above; keep the explicit alias convenient for
# orchestration code that represents a multi-stage curriculum.
JointWAMStageConfig = JointWAMTrainConfig


@dataclass(frozen=True)
class GeneratedActionConsistency:
    """Loss and auditable frozen-teacher targets for one generated sub-batch."""

    total: Tensor
    state: Tensor
    risk: Tensor
    progress: Tensor
    deployed_actions: Tensor
    teacher_candidate_actions: Tensor
    teacher_next_state: Tensor
    teacher_failure_probability: Tensor
    teacher_response_progress: Tensor
    target_source: str = GENERATED_ACTION_TARGET_SOURCE
    demo_state_is_ground_truth: bool = False

    @property
    def teacher_targets_detached(self) -> bool:
        return all(
            not value.requires_grad
            for value in (
                self.teacher_candidate_actions,
                self.teacher_next_state,
                self.teacher_failure_probability,
                self.teacher_response_progress,
            )
        )


def differentiable_flow_generate(
    flow: StatefulActionFlow,
    features: Tensor,
    *,
    initial_actions: Tensor | None = None,
    warm_start_mask: Tensor | None = None,
    solver_steps: int = 4,
    solver: str = "euler",
    normalized_clip: float = 10.0,
) -> Tensor:
    """Integrate the action-flow warm-up velocity field without disabling autograd.

    This mirrors :meth:`StatefulActionFlow.generate` numerically, but uses
    out-of-place clamps and has no ``no_grad`` decorator.  Consequently a world
    consistency loss on its result can update both the flow and its shared
    Joint world-model conditioning features.
    """

    if solver_steps <= 0:
        raise ValueError("solver_steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be euler or heun")
    if normalized_clip <= 0.0:
        raise ValueError("normalized_clip must be positive")
    if features.ndim != 2 or features.shape[-1] != flow.config.feature_dim:
        raise ValueError("features must have shape [B,feature_dim]")
    batch_size = features.shape[0]
    shape = (batch_size, flow.config.horizon, flow.config.action_dim)
    cold = torch.zeros(shape, device=features.device, dtype=features.dtype)
    if initial_actions is None:
        if warm_start_mask is not None and bool(warm_start_mask.bool().any()):
            raise ValueError("warm_start_mask requires initial_actions")
        initial = cold
        warm = torch.zeros(
            (batch_size, 1), device=features.device, dtype=features.dtype
        )
    else:
        if tuple(initial_actions.shape) != shape:
            raise ValueError(f"initial_actions must have shape {shape}")
        if (
            initial_actions.device != features.device
            or initial_actions.dtype != features.dtype
        ):
            raise TypeError("initial_actions must match feature device and dtype")
        normalized_warm = flow.normalize_actions(initial_actions).clamp(
            -normalized_clip, normalized_clip
        )
        if warm_start_mask is None:
            mask = torch.ones((batch_size, 1), device=features.device, dtype=torch.bool)
        else:
            mask = warm_start_mask.to(device=features.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask[:, None]
            if tuple(mask.shape) != (batch_size, 1):
                raise ValueError("warm_start_mask must have shape [B] or [B,1]")
        initial = torch.where(mask[:, :, None], normalized_warm, cold)
        warm = mask.to(dtype=features.dtype)
    current = initial.clone()
    dt = 1.0 / solver_steps
    for step in range(solver_steps):
        tau = torch.full(
            (batch_size,),
            step * dt,
            device=features.device,
            dtype=features.dtype,
        )
        velocity = flow(current, tau, features, initial, warm)
        if solver == "euler":
            current = current + dt * velocity
        else:
            proposal = current + dt * velocity
            next_tau = torch.full_like(tau, (step + 1) * dt)
            correction = flow(proposal, next_tau, features, initial, warm)
            current = current + 0.5 * dt * (velocity + correction)
        current = current.clamp(-normalized_clip, normalized_clip)
    return flow.denormalize_actions(current).clamp(-1.0, 1.0)


def rollout_frozen_prior_chunk(
    world_model: RWMARWorldModel,
    flow: StatefulActionFlow,
    hidden: Tensor,
    current_state: Tensor,
    *,
    steps: int,
    fixed_actions: Mapping[int, float] | Sequence[tuple[int, float]] = (),
) -> Tensor:
    """Roll the embedded frozen prior while preserving input gradients.

    The prior weights remain frozen, but this helper intentionally has no
    ``no_grad`` decorator.  For the Joint copy, action-loss gradients can thus
    reach shared history/belief features.  Call it inside ``torch.no_grad()``
    when constructing immutable teacher chunks.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    entries = tuple(
        fixed_actions.items() if isinstance(fixed_actions, Mapping) else fixed_actions
    )
    for raw_index, raw_value in entries:
        index, value = int(raw_index), float(raw_value)
        if not 0 <= index < flow.config.action_dim:
            raise ValueError("fixed action index is out of range")
        if not -1.0 <= value <= 1.0:
            raise ValueError("fixed action values must be in [-1,1]")
    actions: list[Tensor] = []
    recurrent = hidden
    state = current_state
    for _ in range(steps):
        features = world_model.planning_features(recurrent, state)
        action = flow.anchor_action(features)
        action = _apply_fixed_actions(action, entries)
        actions.append(action)
        recurrent, state, _ = world_model.imagine_step(
            recurrent, state, action, sample_state=False
        )
    return torch.stack(actions, dim=1)


def build_deployed_action_chunk(
    anchor_actions: Tensor,
    generated_actions: Tensor,
    *,
    residual_scale: float,
    fixed_actions: Mapping[int, float] | Sequence[tuple[int, float]] = (),
) -> Tensor:
    """Apply the exact anchored-residual and fixed-action deployment contract."""

    if anchor_actions.shape != generated_actions.shape or anchor_actions.ndim != 3:
        raise ValueError("anchor and generated actions must share shape [B,H,A]")
    if not 0.0 < residual_scale <= 1.0:
        raise ValueError("residual_scale must be in (0,1]")
    result = (
        anchor_actions + residual_scale * (generated_actions - anchor_actions)
    ).clamp(-1.0, 1.0)
    entries = (
        fixed_actions.items() if isinstance(fixed_actions, Mapping) else fixed_actions
    )
    for raw_index, raw_value in entries:
        index, value = int(raw_index), float(raw_value)
        if not 0 <= index < result.shape[-1]:
            raise ValueError("fixed action index is out of range")
        if not -1.0 <= value <= 1.0:
            raise ValueError("fixed action values must be in [-1,1]")
        replacement = torch.full_like(result[..., index], value)
        result = _replace_column(result, index, replacement)
    return result


def generated_action_consistency_loss(
    joint_model: RWMARWorldModel,
    frozen_teacher: RWMARWorldModel,
    history: WorldModelSequenceInputs,
    joint_hidden: Tensor,
    joint_current_state: Tensor,
    deployed_actions: Tensor,
    *,
    state_weight: float = 1.0,
    risk_weight: float = 0.1,
    progress_weight: float = 0.1,
) -> GeneratedActionConsistency:
    """Match Joint predictions to detached frozen-teacher counterfactuals.

    ``deployed_actions`` is the sole counterfactual action input.  The teacher
    receives an exact detached copy, and this function accepts no demonstration
    target-state argument, making accidental demo-state relabeling impossible.
    """

    if joint_model.config != frozen_teacher.config:
        raise ValueError("Joint model and frozen teacher configurations differ")
    for name, value in (
        ("state_weight", state_weight),
        ("risk_weight", risk_weight),
        ("progress_weight", progress_weight),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    student = joint_model.predict_from_encoded_history(
        joint_hidden,
        joint_current_state,
        deployed_actions,
        sample_state=False,
    )
    teacher_actions = deployed_actions.detach()
    # The immutable teacher is a target generator, not an optimized branch.
    # Explicitly disable the caller's autocast context so its GRU hidden state,
    # current state, and candidate actions remain the same FP32 dtype required
    # by ``predict_from_encoded_history``.
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=history.states.device.type,
            enabled=False,
        ),
    ):
        teacher_hidden, teacher_state = frozen_teacher.encode_history(history)
        teacher = frozen_teacher.predict_from_encoded_history(
            teacher_hidden.float(),
            teacher_state.float(),
            teacher_actions.float(),
            sample_state=False,
        )
        teacher_next_state = teacher.next_state_mean.detach()
        teacher_failure = teacher.failure_logit.sigmoid().detach()
        teacher_progress = teacher.response_progress.detach()

    difference = student.next_state_mean - teacher_next_state
    for yaw_index in joint_model.config.yaw_indices:
        difference = _replace_column(
            difference, yaw_index, wrap_to_pi(difference[..., yaw_index])
        )
    continuous = joint_model.continuous_state_mask
    state_std = joint_model.features.state_std[continuous].to(difference)
    state_loss = (difference[..., continuous] / state_std).square().mean()
    risk_loss = F.smooth_l1_loss(
        student.failure_logit.float().sigmoid(), teacher_failure.float()
    )
    progress_loss = F.smooth_l1_loss(
        student.response_progress.float(), teacher_progress.float()
    )
    total = (
        state_weight * state_loss
        + risk_weight * risk_loss
        + progress_weight * progress_loss
    )
    return GeneratedActionConsistency(
        total=total,
        state=state_loss,
        risk=risk_loss,
        progress=progress_loss,
        deployed_actions=deployed_actions,
        teacher_candidate_actions=teacher_actions,
        teacher_next_state=teacher_next_state,
        teacher_failure_probability=teacher_failure,
        teacher_response_progress=teacher_progress,
    )


def configure_joint_wam_scope(
    flow: StatefulActionFlow,
    joint_model: RWMARWorldModel,
    frozen_teacher: RWMARWorldModel,
    scope: JointWAMScope,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Freeze immutable assets and return trainable flow/world-model parameters."""

    if scope not in {"flow_only", "world_heads", "full_joint"}:
        raise ValueError("unsupported Joint WAM training scope")
    if joint_model is frozen_teacher:
        raise ValueError("Joint model and frozen teacher must be independent objects")
    _assert_parameter_storage_disjoint(joint_model, frozen_teacher)
    _freeze_module(frozen_teacher)

    flow.train()
    for name, parameter in flow.named_parameters():
        parameter.requires_grad_(not name.startswith("anchor_prior."))
    flow.freeze_anchor()

    for parameter in joint_model.parameters():
        parameter.requires_grad_(False)
    if scope == "flow_only":
        # Parameters stay frozen, but generated-world consistency must still
        # differentiate through the action-conditioned recurrent rollout back
        # to the flow inputs.  cuDNN refuses RNN input-gradient backward when
        # the GRU forward ran in eval mode.  The accepted world-model config has
        # dropout=0, so train mode changes no stochastic semantics here.
        joint_model.train()
    else:
        joint_model.train()
        modules = (
            (joint_model.decoder, joint_model.heads)
            if scope == "world_heads"
            else (joint_model,)
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    flow_parameters = [
        parameter for parameter in flow.parameters() if parameter.requires_grad
    ]
    world_model_parameters = [
        parameter for parameter in joint_model.parameters() if parameter.requires_grad
    ]
    if not flow_parameters:
        raise RuntimeError(
            "Joint WAM scope exposes no trainable action-flow parameters"
        )
    if scope != "flow_only" and not world_model_parameters:
        raise RuntimeError(
            "Joint WAM scope exposes no trainable Joint world-model parameters"
        )
    if any(parameter.requires_grad for parameter in frozen_teacher.parameters()):
        raise RuntimeError("frozen initialization teacher became trainable")
    if any(parameter.requires_grad for parameter in flow.anchor_prior.parameters()):
        raise RuntimeError("embedded action-prior anchor became trainable")
    return flow_parameters, world_model_parameters


def train_joint_wam_stage(
    flow: StatefulActionFlow,
    joint_model: RWMARWorldModel,
    frozen_teacher: RWMARWorldModel,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    config: JointWAMTrainConfig,
    seed: int,
    positive_weights: Mapping[str, Tensor | float] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Train one explicit Joint WAM scope and return per-step auditable metrics."""

    _validate_model_contract(flow, joint_model, frozen_teacher, config)
    flow.to(device)
    joint_model.to(device)
    frozen_teacher.to(device)
    flow_parameters, world_model_parameters = configure_joint_wam_scope(
        flow, joint_model, frozen_teacher, config.scope
    )
    parameter_groups: list[dict[str, Any]] = [
        {"params": flow_parameters, "lr": config.flow_learning_rate}
    ]
    if world_model_parameters:
        parameter_groups.append(
            {"params": world_model_parameters, "lr": config.world_model_learning_rate}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.use_amp and device.type == "cuda"
    amp_dtype = _preferred_amp_dtype(device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    label_weights = _positive_weights(positive_weights, device)
    estimated_steps = _estimated_stage_steps(loader, config)
    completed_steps = 0
    history: list[dict[str, Any]] = []

    for epoch in range(config.epochs):
        for raw_batch in loader:
            if config.max_steps > 0 and completed_steps >= config.max_steps:
                return history, completed_steps
            batch = _batch_to_device(raw_batch, device)
            _validate_training_batch(batch, flow, config)
            optimizer.zero_grad(set_to_none=True)

            history_inputs = _history(batch)
            with torch.no_grad():
                teacher_hidden, teacher_state, _ = (
                    frozen_teacher.encode_planning_history(history_inputs)
                )
                supervised_actions = rollout_frozen_prior_chunk(
                    frozen_teacher,
                    flow,
                    teacher_hidden,
                    teacher_state,
                    steps=flow.config.horizon,
                    fixed_actions=config.fixed_actions,
                ).detach()

            ratio = _generated_action_ratio(config, completed_steps, estimated_steps)
            # Keep the recurrent history contract in FP32.  Under autocast a
            # GRU returns a BF16 hidden state while ``current_state`` remains
            # the original FP32 tensor; the public encoded-history rollout
            # intentionally rejects that mixed pair.  Encoding outside
            # autocast preserves both the strict dtype contract and the full
            # action/world gradient path into shared history parameters.
            joint_hidden, joint_state, joint_features = (
                joint_model.encode_planning_history(history_inputs)
            )
            joint_features = joint_features.to(dtype=batch["candidate_actions"].dtype)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                action_parts = _action_flow_loss(
                    flow,
                    joint_features,
                    supervised_actions.to(dtype=joint_features.dtype),
                    batch["action_quality_weights"].reshape(-1),
                    config=config,
                    generator=generator,
                )

                expert_actions = batch["candidate_actions"][:, : config.world_horizon]
                expert_world = joint_model.predict_from_encoded_history(
                    joint_hidden,
                    joint_state,
                    expert_actions.to(dtype=joint_state.dtype),
                    sample_state=False,
                )
                world_loss, world_parts = compute_rwm_loss(
                    expert_world,
                    batch,
                    delta_std=joint_model.delta_std,
                    yaw_indices=joint_model.config.yaw_indices,
                    closed_indices=joint_model.config.gripper_closed_indices,
                    positive_weights=label_weights,
                    horizon_decay=config.world_horizon_decay,
                    weights=config.world_loss_weights,
                )

                consistency, generated_samples = _generated_consistency_subbatch(
                    flow,
                    joint_model,
                    frozen_teacher,
                    history_inputs,
                    joint_hidden,
                    joint_state,
                    joint_features,
                    supervised_actions,
                    ratio=ratio,
                    config=config,
                    generator=generator,
                )
                consistency_loss = (
                    consistency.total
                    if consistency is not None
                    else _connected_zero(flow_parameters, world_model_parameters)
                )
                total_loss = (
                    config.action_loss_weight * action_parts["total"]
                    + config.world_loss_weight * world_loss
                    + config.generated_consistency_weight * consistency_loss
                )

            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("non-finite Joint WAM loss")
            audit = _empty_branch_audit()
            if completed_steps % config.gradient_audit_interval == 0:
                audit = audit_joint_wam_branch_gradients(
                    action_parts["total"],
                    world_loss,
                    consistency_loss,
                    flow_parameters=flow_parameters,
                    world_model_parameters=world_model_parameters,
                )
                if not all(math.isfinite(value) for value in audit.values()):
                    raise FloatingPointError(
                        "non-finite Joint WAM branch-gradient audit"
                    )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            flow_gradient_norm = torch.nn.utils.clip_grad_norm_(
                flow_parameters, config.flow_gradient_clip_norm
            )
            world_model_gradient_norm: Tensor | float = 0.0
            if world_model_parameters:
                world_model_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    world_model_parameters, config.world_model_gradient_clip_norm
                )
            if not math.isfinite(float(flow_gradient_norm)) or not math.isfinite(
                float(world_model_gradient_norm)
            ):
                raise FloatingPointError("non-finite Joint WAM gradient norm")
            scaler.step(optimizer)
            scaler.update()
            completed_steps += 1

            consistency_state = (
                float(consistency.state.detach().cpu()) if consistency else 0.0
            )
            consistency_risk = (
                float(consistency.risk.detach().cpu()) if consistency else 0.0
            )
            consistency_progress = (
                float(consistency.progress.detach().cpu()) if consistency else 0.0
            )
            item: dict[str, Any] = {
                "scope": config.scope,
                "flow_action_target_source": "frozen_action_prior_rollout",
                "stage_name": config.name or config.scope,
                "epoch": epoch + 1,
                "step": completed_steps,
                "loss": float(total_loss.detach().cpu()),
                "action_loss": float(action_parts["total"].detach().cpu()),
                "action_flow_loss": float(action_parts["flow"].detach().cpu()),
                "action_endpoint_loss": float(action_parts["endpoint"].detach().cpu()),
                "action_smoothness_loss": float(
                    action_parts["smoothness"].detach().cpu()
                ),
                "action_supervised_samples": int(
                    action_parts["samples"].detach().cpu()
                ),
                "world_loss": float(world_loss.detach().cpu()),
                "world_state_mean_mse": float(world_parts["state_mean_mse"].cpu()),
                "world_state_nll": float(world_parts["state_nll"].cpu()),
                "world_expert_samples": int(batch["states"].shape[0]),
                "generated_consistency_loss": float(consistency_loss.detach().cpu()),
                "generated_state_consistency": consistency_state,
                "generated_risk_consistency": consistency_risk,
                "generated_progress_consistency": consistency_progress,
                "generated_action_ratio": ratio,
                "generated_action_samples": generated_samples,
                "generated_action_target_source": GENERATED_ACTION_TARGET_SOURCE,
                "generated_action_demo_state_is_ground_truth": False,
                "generated_teacher_targets_detached": (
                    True
                    if consistency is None
                    else consistency.teacher_targets_detached
                ),
                "flow_gradient_norm": float(flow_gradient_norm),
                "world_model_gradient_norm": float(world_model_gradient_norm),
                "total_gradient_norm": math.hypot(
                    float(flow_gradient_norm), float(world_model_gradient_norm)
                ),
                "flow_learning_rate": config.flow_learning_rate,
                "world_model_learning_rate": (
                    config.world_model_learning_rate if world_model_parameters else 0.0
                ),
                "gradient_audit_performed": bool(
                    completed_steps - 1
                    == ((completed_steps - 1) // config.gradient_audit_interval)
                    * config.gradient_audit_interval
                ),
                **audit,
            }
            if progress is not None:
                progress(item)
            history.append(item)
    return history, completed_steps


def train_joint_wam(
    flow: StatefulActionFlow,
    joint_model: RWMARWorldModel,
    frozen_teacher: RWMARWorldModel,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    stages: Sequence[JointWAMTrainConfig],
    seed: int,
    positive_weights: Mapping[str, Tensor | float] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Run an explicit sequence of Joint WAM scopes without carrying optimizer state."""

    if not stages:
        raise ValueError("Joint WAM requires at least one training stage")
    history: list[dict[str, Any]] = []
    total_steps = 0
    for stage_index, stage in enumerate(stages):
        stage_history, stage_steps = train_joint_wam_stage(
            flow,
            joint_model,
            frozen_teacher,
            loader,
            device=device,
            config=stage,
            seed=seed + stage_index,
            positive_weights=positive_weights,
            progress=progress,
        )
        for item in stage_history:
            item["stage_index"] = stage_index
            item["global_step"] = total_steps + int(item["step"])
        history.extend(stage_history)
        total_steps += stage_steps
    return history, total_steps


def audit_joint_wam_branch_gradients(
    action_loss: Tensor,
    world_loss: Tensor,
    consistency_loss: Tensor,
    *,
    flow_parameters: Sequence[nn.Parameter],
    world_model_parameters: Sequence[nn.Parameter],
) -> dict[str, float]:
    """Measure the six objective-to-branch gradient paths without populating grads."""

    return {
        "action_to_flow_gradient_norm": _branch_gradient_norm(
            action_loss, flow_parameters
        ),
        "action_to_backbone_gradient_norm": _branch_gradient_norm(
            action_loss, world_model_parameters
        ),
        "world_to_flow_gradient_norm": _branch_gradient_norm(
            world_loss, flow_parameters
        ),
        "world_to_backbone_gradient_norm": _branch_gradient_norm(
            world_loss, world_model_parameters
        ),
        "consistency_to_flow_gradient_norm": _branch_gradient_norm(
            consistency_loss, flow_parameters
        ),
        "consistency_to_backbone_gradient_norm": _branch_gradient_norm(
            consistency_loss, world_model_parameters
        ),
    }


def _action_flow_loss(
    flow: StatefulActionFlow,
    features: Tensor,
    target_actions: Tensor,
    quality: Tensor,
    *,
    config: JointWAMTrainConfig,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    target = flow.normalize_actions(target_actions)
    noise = torch.randn(
        target.shape,
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )
    warm_actions = synthetic_shifted_warm_start(target_actions, config.execution_steps)
    warm_initial = flow.normalize_actions(warm_actions)
    warm_initial = warm_initial + config.warm_start_noise_std * noise
    cold_initial = config.cold_noise_std * noise
    cold_zero = (
        torch.rand((target.shape[0], 1, 1), device=target.device, generator=generator)
        < config.cold_zero_probability
    )
    cold_initial = torch.where(cold_zero, torch.zeros_like(cold_initial), cold_initial)
    warm = (
        torch.rand((target.shape[0], 1, 1), device=target.device, generator=generator)
        < config.warm_start_probability
    )
    initial = torch.where(warm, warm_initial, cold_initial)
    tau = torch.rand(
        (target.shape[0],),
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )
    path = (1.0 - tau[:, None, None]) * initial + tau[:, None, None] * target
    velocity = target - initial
    predicted = flow(
        path,
        tau,
        features,
        initial,
        warm[:, 0].to(dtype=target.dtype),
    )
    flow_per_sample = (predicted - velocity).square().mean(dim=(1, 2))
    endpoint_normalized = path + (1.0 - tau[:, None, None]) * predicted
    endpoint_actions = flow.denormalize_actions(endpoint_normalized)
    endpoint_per_sample = (endpoint_actions - target_actions).square().mean(dim=(1, 2))
    smoothness_per_sample = (
        (
            (endpoint_actions[:, 1:] - endpoint_actions[:, :-1])
            - (target_actions[:, 1:] - target_actions[:, :-1])
        )
        .square()
        .mean(dim=(1, 2))
    )
    weights = quality.to(device=target.device, dtype=target.dtype).clamp_min(0.0)
    selected = weights > 0.0
    denominator = weights.sum()

    def weighted(value: Tensor) -> Tensor:
        if bool(selected.any()):
            return (value * weights).sum() / denominator
        return value.sum() * 0.0

    flow_loss = weighted(flow_per_sample)
    endpoint_loss = weighted(endpoint_per_sample)
    smoothness_loss = weighted(smoothness_per_sample)
    action_total = (
        flow_loss
        + config.action_endpoint_weight * endpoint_loss
        + config.action_smoothness_weight * smoothness_loss
    )
    return {
        "total": action_total,
        "flow": flow_loss,
        "endpoint": endpoint_loss,
        "smoothness": smoothness_loss,
        "samples": selected.sum(),
    }


def _generated_consistency_subbatch(
    flow: StatefulActionFlow,
    joint_model: RWMARWorldModel,
    frozen_teacher: RWMARWorldModel,
    history: WorldModelSequenceInputs,
    joint_hidden: Tensor,
    joint_state: Tensor,
    joint_features: Tensor,
    reference_actions: Tensor,
    *,
    ratio: float,
    config: JointWAMTrainConfig,
    generator: torch.Generator,
) -> tuple[GeneratedActionConsistency | None, int]:
    batch_size = joint_features.shape[0]
    selected = (
        torch.rand((batch_size,), device=joint_features.device, generator=generator)
        < ratio
    )
    count = int(selected.sum().detach().cpu())
    if count == 0:
        return None, 0
    selected_hidden = joint_hidden[:, selected]
    selected_state = joint_state[selected]
    selected_features = joint_features[selected]
    selected_reference_actions = reference_actions[selected].to(
        dtype=selected_features.dtype
    )
    anchor_actions = rollout_frozen_prior_chunk(
        joint_model,
        flow,
        selected_hidden,
        selected_state,
        steps=flow.config.horizon,
        fixed_actions=config.fixed_actions,
    )
    warm_seed = synthetic_shifted_warm_start(
        selected_reference_actions, config.execution_steps
    )
    warm_mask = (
        torch.rand((count,), device=selected_features.device, generator=generator)
        < config.generated_warm_start_probability
    )
    raw_generated = differentiable_flow_generate(
        flow,
        selected_features,
        initial_actions=warm_seed,
        warm_start_mask=warm_mask,
        solver_steps=config.solver_steps,
        solver=config.solver,
        normalized_clip=config.normalized_action_clip,
    )
    deployed = build_deployed_action_chunk(
        anchor_actions,
        raw_generated,
        residual_scale=config.anchor_residual_scale,
        fixed_actions=config.fixed_actions,
    )
    selected_history = _select_history(history, selected)
    return (
        generated_action_consistency_loss(
            joint_model,
            frozen_teacher,
            selected_history,
            selected_hidden,
            selected_state,
            deployed,
            state_weight=config.generated_state_weight,
            risk_weight=config.generated_risk_weight,
            progress_weight=config.generated_progress_weight,
        ),
        count,
    )


def _validate_model_contract(
    flow: StatefulActionFlow,
    joint_model: RWMARWorldModel,
    frozen_teacher: RWMARWorldModel,
    config: JointWAMTrainConfig,
) -> None:
    if joint_model.config != frozen_teacher.config:
        raise ValueError("Joint and frozen teacher model configurations differ")
    if flow.config.feature_dim != joint_model.planning_feature_dim:
        raise ValueError("flow feature dimension does not match Joint world model")
    if flow.config.action_dim != joint_model.config.action_dim:
        raise ValueError("flow action dimension does not match Joint world model")
    if config.world_horizon > flow.config.horizon:
        raise ValueError("world_horizon cannot exceed the action-chunk horizon")
    if config.execution_steps >= flow.config.horizon:
        raise ValueError("execution_steps must be smaller than the action horizon")
    if any(index >= flow.config.action_dim for index, _ in config.fixed_actions):
        raise ValueError("fixed action index exceeds the action dimension")
    if flow.anchor_prior.config.feature_dim != joint_model.planning_feature_dim:
        raise ValueError("embedded action anchor does not match Joint world model")


def _validate_training_batch(
    batch: Mapping[str, Tensor],
    flow: StatefulActionFlow,
    config: JointWAMTrainConfig,
) -> None:
    required = {
        "states",
        "past_actions",
        "valid_mask",
        "candidate_actions",
        "target_states",
        "forecast_mask",
        "rewards",
        "dones",
        "successes",
        "failures",
        "response_progress",
        "coordination_error",
        "executed_actions",
        "action_quality_weights",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise KeyError(f"Joint WAM batch is missing {missing}")
    actions = batch["candidate_actions"]
    if actions.ndim != 3 or actions.shape[1] < flow.config.horizon:
        raise ValueError(
            "candidate_actions does not contain a complete Joint WAM chunk"
        )
    if actions.shape[-1] != flow.config.action_dim:
        raise ValueError("candidate_actions has the wrong action dimension")
    if batch["target_states"].shape[1] < config.world_horizon:
        raise ValueError("target_states is shorter than world_horizon")
    quality = batch["action_quality_weights"]
    if quality.numel() != actions.shape[0]:
        raise ValueError("action_quality_weights must contain one value per sample")
    if not bool(torch.isfinite(quality).all()) or bool((quality < 0.0).any()):
        raise ValueError("action_quality_weights must be finite and non-negative")


def _positive_weights(
    values: Mapping[str, Tensor | float] | None,
    device: torch.device,
) -> dict[str, Tensor]:
    source = values or {"done": 1.0, "success": 1.0, "failure": 1.0}
    missing = {"done", "success", "failure"} - set(source)
    if missing:
        raise KeyError(f"positive weights are missing {sorted(missing)}")
    result: dict[str, Tensor] = {}
    for name in ("done", "success", "failure"):
        raw = source[name]
        value = (
            raw.detach().to(device=device, dtype=torch.float32)
            if isinstance(raw, Tensor)
            else torch.tensor(float(raw), device=device)
        )
        if value.numel() != 1 or not bool(torch.isfinite(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} positive weight must be finite and positive")
        result[name] = value.reshape(())
    return result


def _history(batch: Mapping[str, Tensor]) -> WorldModelSequenceInputs:
    return WorldModelSequenceInputs(
        states=batch["states"],
        past_actions=batch["past_actions"],
        valid_mask=batch["valid_mask"],
    )


def _select_history(
    history: WorldModelSequenceInputs, selected: Tensor
) -> WorldModelSequenceInputs:
    return WorldModelSequenceInputs(
        states=history.states[selected],
        past_actions=history.past_actions[selected],
        valid_mask=history.valid_mask[selected],
    )


def _batch_to_device(
    batch: Mapping[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if isinstance(value, Tensor)
    }


def _apply_fixed_actions(
    actions: Tensor, fixed_actions: Sequence[tuple[int, float]]
) -> Tensor:
    result = actions
    for index, value in fixed_actions:
        result = _replace_column(
            result, index, torch.full_like(result[..., index], value)
        )
    return result


def _replace_column(value: Tensor, index: int, column: Tensor) -> Tensor:
    parts = list(value.split(1, dim=-1))
    parts[index] = column.unsqueeze(-1)
    return torch.cat(parts, dim=-1)


def _generated_action_ratio(
    config: JointWAMTrainConfig, completed_steps: int, estimated_steps: int
) -> float:
    if estimated_steps <= 1:
        fraction = 1.0
    else:
        fraction = min(max(completed_steps / (estimated_steps - 1), 0.0), 1.0)
    return float(
        config.generated_action_ratio_start
        + fraction
        * (config.generated_action_ratio_end - config.generated_action_ratio_start)
    )


def _estimated_stage_steps(
    loader: Iterable[Mapping[str, Tensor]], config: JointWAMTrainConfig
) -> int:
    try:
        value = len(loader) * config.epochs  # type: ignore[arg-type]
    except TypeError:
        value = config.max_steps if config.max_steps > 0 else 1
    if config.max_steps > 0:
        value = min(value, config.max_steps)
    return max(int(value), 1)


def _connected_zero(
    flow_parameters: Sequence[nn.Parameter],
    world_model_parameters: Sequence[nn.Parameter],
) -> Tensor:
    zero = flow_parameters[0].sum() * 0.0
    if world_model_parameters:
        zero = zero + world_model_parameters[0].sum() * 0.0
    return zero


def _branch_gradient_norm(loss: Tensor, parameters: Sequence[nn.Parameter]) -> float:
    if not parameters or not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=True,
        allow_unused=True,
    )
    squared = torch.zeros((), device=loss.device, dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    return float(squared.sqrt().cpu())


def _empty_branch_audit() -> dict[str, float]:
    return {
        "action_to_flow_gradient_norm": 0.0,
        "action_to_backbone_gradient_norm": 0.0,
        "world_to_flow_gradient_norm": 0.0,
        "world_to_backbone_gradient_norm": 0.0,
        "consistency_to_flow_gradient_norm": 0.0,
        "consistency_to_backbone_gradient_norm": 0.0,
    }


def _assert_parameter_storage_disjoint(first: nn.Module, second: nn.Module) -> None:
    first_parameters = dict(first.named_parameters())
    second_parameters = dict(second.named_parameters())
    if first_parameters.keys() != second_parameters.keys():
        raise ValueError("Joint model and teacher parameter structures differ")
    shared = [
        name
        for name in first_parameters
        if first_parameters[name].data_ptr() == second_parameters[name].data_ptr()
    ]
    if shared:
        raise ValueError(
            f"Joint model and frozen teacher share parameter storage: {shared[:3]}"
        )


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _preferred_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


__all__ = [
    "GENERATED_ACTION_TARGET_SOURCE",
    "GeneratedActionConsistency",
    "JointWAMScope",
    "JointWAMStageConfig",
    "JointWAMTrainConfig",
    "audit_joint_wam_branch_gradients",
    "build_deployed_action_chunk",
    "configure_joint_wam_scope",
    "differentiable_flow_generate",
    "generated_action_consistency_loss",
    "rollout_frozen_prior_chunk",
    "train_joint_wam",
    "train_joint_wam_stage",
]
