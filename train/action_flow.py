"""Frozen-backbone Flow Matching warm-up for Joint WAM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import torch
from torch import Tensor

from models.wam import (
    ActionPrior,
    RWMARWorldModel,
    StatefulActionFlow,
    WorldModelSequenceInputs,
)

ProgressCallback = Callable[[Mapping[str, float | int]], None]


@dataclass(frozen=True)
class ActionFlowTrainConfig:
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 10.0
    warm_start_probability: float = 0.5
    warm_start_noise_std: float = 0.05
    cold_noise_std: float = 1.0
    cold_zero_probability: float = 0.5
    endpoint_weight: float = 1.0
    smoothness_weight: float = 0.01
    execution_steps: int = 2
    action_prior_distillation_steps: int = 0
    action_prior_distillation_blend: float = 0.0
    use_amp: bool = True
    max_steps: int = -1

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0.0:
            raise ValueError("epochs and learning_rate must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid weight decay or gradient clipping")
        for name in ("warm_start_probability", "cold_zero_probability"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        for name in (
            "warm_start_noise_std",
            "cold_noise_std",
            "endpoint_weight",
            "smoothness_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.execution_steps <= 0:
            raise ValueError("execution_steps must be positive")
        if self.action_prior_distillation_steps < 0:
            raise ValueError("action_prior_distillation_steps cannot be negative")
        if not 0.0 <= self.action_prior_distillation_blend <= 1.0:
            raise ValueError("action_prior_distillation_blend must be in [0,1]")
        if (
            self.action_prior_distillation_steps == 0
            and self.action_prior_distillation_blend != 0.0
        ):
            raise ValueError("distillation blend requires positive teacher steps")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")


@dataclass(frozen=True)
class ActionFlowOnPolicyTrainConfig:
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 10.0
    cold_replay_probability: float = 0.1
    warm_start_noise_std: float = 0.01
    endpoint_weight: float = 1.0
    smoothness_weight: float = 0.01
    solver_steps: int = 4
    solver_endpoint_weight: float = 10.0
    offline_replay_weight: float = 1.0
    use_amp: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0.0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid weight decay or gradient clipping")
        if not 0.0 <= self.cold_replay_probability <= 1.0:
            raise ValueError("cold_replay_probability must be in [0,1]")
        if self.solver_steps <= 0:
            raise ValueError("solver_steps must be positive")
        for name in (
            "warm_start_noise_std",
            "endpoint_weight",
            "smoothness_weight",
            "solver_endpoint_weight",
            "offline_replay_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


class ActionFlowDistillationBuffer:
    """CPU buffer of direct-flow requests collected without privileged state."""

    def __init__(self) -> None:
        self._features: list[Tensor] = []
        self._targets: list[Tensor] = []
        self._initials: list[Tensor] = []
        self._warm: list[Tensor] = []

    def add(
        self,
        features: Tensor,
        target_actions: Tensor,
        initial_actions: Tensor | None,
    ) -> None:
        if features.ndim != 2 or target_actions.ndim != 3:
            raise ValueError("distillation tensors must include a batch dimension")
        if features.shape[0] != target_actions.shape[0]:
            raise ValueError("distillation batch dimensions must match")
        if initial_actions is not None and initial_actions.shape != target_actions.shape:
            raise ValueError("warm initial actions must match target actions")
        feature_cpu = features.detach().float().cpu()
        target_cpu = target_actions.detach().float().cpu()
        initial_cpu = (
            torch.zeros_like(target_cpu)
            if initial_actions is None
            else initial_actions.detach().float().cpu()
        )
        warm_cpu = torch.full(
            (target_cpu.shape[0], 1),
            initial_actions is not None,
            dtype=torch.bool,
        )
        self._features.append(feature_cpu)
        self._targets.append(target_cpu)
        self._initials.append(initial_cpu)
        self._warm.append(warm_cpu)

    def __len__(self) -> int:
        return sum(item.shape[0] for item in self._features)

    def tensors(self) -> dict[str, Tensor]:
        if not self._features:
            raise RuntimeError("on-policy distillation buffer is empty")
        return {
            "features": torch.cat(self._features),
            "targets": torch.cat(self._targets),
            "initials": torch.cat(self._initials),
            "warm": torch.cat(self._warm),
        }


def synthetic_shifted_warm_start(actions: Tensor, executed_steps: int) -> Tensor:
    """Construct the prior chunk that would remain at the current decision.

    For a target ``a[t:t+H]``, a chunk generated ``executed_steps`` earlier
    contains the first ``H-executed_steps`` current target actions.  Its unknown
    tail is filled by repeating the last known action, matching the runtime
    contract without crossing the episode boundary.
    """

    if actions.ndim != 3:
        raise ValueError("actions must have shape [B,H,A]")
    steps = int(executed_steps)
    if steps <= 0 or steps >= actions.shape[1]:
        raise ValueError("executed_steps must be in [1,H)")
    known = actions[:, : actions.shape[1] - steps]
    tail = known[:, -1:].expand(-1, steps, -1)
    return torch.cat((known, tail), dim=1)


def make_flow_matching_batch(
    flow: StatefulActionFlow,
    target_actions: Tensor,
    *,
    config: ActionFlowTrainConfig,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    """Sample cold and warm-start rectified-flow paths for one batch."""

    expected = (
        target_actions.shape[0],
        flow.config.horizon,
        flow.config.action_dim,
    )
    if tuple(target_actions.shape) != expected:
        raise ValueError(f"target_actions must have shape {expected}")
    target = flow.normalize_actions(target_actions)
    noise = torch.randn(
        target.shape,
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )
    warm_actions = synthetic_shifted_warm_start(
        target_actions, config.execution_steps
    )
    warm_initial = flow.normalize_actions(warm_actions)
    warm_initial = warm_initial + config.warm_start_noise_std * noise
    cold_initial = config.cold_noise_std * noise
    cold_zero = torch.rand(
        (target.shape[0], 1, 1),
        device=target.device,
        generator=generator,
    ) < config.cold_zero_probability
    cold_initial = torch.where(cold_zero, torch.zeros_like(cold_initial), cold_initial)
    warm = torch.rand(
        (target.shape[0], 1, 1),
        device=target.device,
        generator=generator,
    ) < config.warm_start_probability
    initial = torch.where(warm, warm_initial, cold_initial)
    tau = torch.rand(
        (target.shape[0],),
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )
    path = (1.0 - tau[:, None, None]) * initial + tau[:, None, None] * target
    return {
        "target": target,
        "initial": initial,
        "path": path,
        "time": tau,
        "warm": warm[:, 0].to(dtype=target.dtype),
        "velocity": target - initial,
    }


def train_action_flow(
    flow: StatefulActionFlow,
    world_model: RWMARWorldModel,
    loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    config: ActionFlowTrainConfig,
    seed: int,
    action_prior: ActionPrior | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[dict[str, float]], int]:
    """Optimize only the action flow while keeping member 0 bitwise frozen."""

    _freeze_world_model(world_model.to(device))
    if config.action_prior_distillation_steps > flow.config.horizon:
        raise ValueError("distillation steps cannot exceed the action horizon")
    if config.action_prior_distillation_steps > 0 and action_prior is None:
        raise ValueError("action-prior distillation requires a teacher")
    if action_prior is not None:
        action_prior.to(device).eval()
        for parameter in action_prior.parameters():
            parameter.requires_grad_(False)
    flow.to(device).train()
    flow.freeze_anchor()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in flow.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.use_amp and device.type == "cuda"
    amp_dtype = _preferred_amp_dtype(device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    history: list[dict[str, float]] = []
    completed_steps = 0
    for epoch in range(config.epochs):
        for raw_batch in loader:
            if config.max_steps > 0 and completed_steps >= config.max_steps:
                return history, completed_steps
            batch = _batch_to_device(raw_batch, device)
            actions = batch["candidate_actions"][:, : flow.config.horizon]
            quality = batch["action_quality_weights"].reshape(-1)
            if not bool((quality > 0.0).all()):
                raise ValueError("action-flow training loader must contain only eligible chunks")
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                hidden, current_state, features = world_model.encode_planning_history(
                    _history(batch)
                )
            features = features.to(dtype=actions.dtype)
            training_actions = actions
            teacher_actions: Tensor | None = None
            if config.action_prior_distillation_steps > 0:
                assert action_prior is not None
                teacher_actions = action_prior_teacher_chunk(
                    world_model,
                    action_prior,
                    hidden,
                    current_state,
                    steps=config.action_prior_distillation_steps,
                ).to(dtype=actions.dtype)
                training_actions = actions.clone()
                count = config.action_prior_distillation_steps
                blend = config.action_prior_distillation_blend
                training_actions[:, :count] = (
                    (1.0 - blend) * actions[:, :count]
                    + blend * teacher_actions
                )
            sampled = make_flow_matching_batch(
                flow, training_actions, config=config, generator=generator
            )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                predicted_velocity = flow(
                    sampled["path"],
                    sampled["time"],
                    features,
                    sampled["initial"],
                    sampled["warm"],
                )
                flow_per_sample = (
                    predicted_velocity - sampled["velocity"]
                ).square().mean(dim=(1, 2))
                endpoint_normalized = sampled["path"] + (
                    1.0 - sampled["time"][:, None, None]
                ) * predicted_velocity
                endpoint = flow.denormalize_actions(endpoint_normalized)
                endpoint_per_sample = (endpoint - training_actions).square().mean(
                    dim=(1, 2)
                )
                smooth_per_sample = (
                    (endpoint[:, 1:] - endpoint[:, :-1])
                    - (training_actions[:, 1:] - training_actions[:, :-1])
                ).square().mean(dim=(1, 2))
                per_sample = (
                    flow_per_sample
                    + config.endpoint_weight * endpoint_per_sample
                    + config.smoothness_weight * smooth_per_sample
                )
                loss = (per_sample * quality).sum() / quality.sum()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite action-flow loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                flow.parameters(), config.gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            completed_steps += 1
            item = {
                "loss": float(loss.detach().cpu()),
                "flow": float(
                    (flow_per_sample * quality).sum().detach().cpu()
                    / quality.sum().detach().cpu()
                ),
                "endpoint": float(
                    (endpoint_per_sample * quality).sum().detach().cpu()
                    / quality.sum().detach().cpu()
                ),
                "smoothness": float(
                    (smooth_per_sample * quality).sum().detach().cpu()
                    / quality.sum().detach().cpu()
                ),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "warm_fraction": float(sampled["warm"].mean().detach().cpu()),
                "teacher_data_rmse": (
                    float(
                        (
                            teacher_actions
                            - actions[:, : config.action_prior_distillation_steps]
                        )
                        .square()
                        .mean()
                        .sqrt()
                        .detach()
                        .cpu()
                    )
                    if teacher_actions is not None
                    else 0.0
                ),
            }
            history.append(item)
            if progress is not None:
                progress(
                    {
                        "epoch": epoch + 1,
                        "epochs": config.epochs,
                        "step": completed_steps,
                        **item,
                    }
                )
    return history, completed_steps


def fine_tune_action_flow_on_policy(
    flow: StatefulActionFlow,
    buffer: ActionFlowDistillationBuffer,
    *,
    device: torch.device,
    config: ActionFlowOnPolicyTrainConfig,
    seed: int,
    replay_loader: Iterable[Mapping[str, Tensor]] | None = None,
    world_model: RWMARWorldModel | None = None,
    action_prior: ActionPrior | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[dict[str, float]], int]:
    """DAgger-style flow matching on the flow's own non-privileged histories."""

    data = {name: value.to(device) for name, value in buffer.tensors().items()}
    sample_count = int(data["features"].shape[0])
    flow.to(device).train()
    for name, parameter in flow.named_parameters():
        parameter.requires_grad_(not name.startswith("anchor_prior."))
    flow.freeze_anchor()
    replay_enabled = config.offline_replay_weight > 0.0
    if replay_enabled and (
        replay_loader is None or world_model is None or action_prior is None
    ):
        raise ValueError("offline replay requires a loader, world model and action prior")
    replay_iterator = iter(replay_loader) if replay_loader is not None else None
    replay_sampling = ActionFlowTrainConfig(epochs=1)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in flow.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.use_amp and device.type == "cuda"
    amp_dtype = _preferred_amp_dtype(device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    history: list[dict[str, float]] = []
    completed_steps = 0
    for epoch in range(config.epochs):
        order = torch.randperm(sample_count, device=device, generator=generator)
        for offset in range(0, sample_count, config.batch_size):
            indices = order[offset : offset + config.batch_size]
            features = data["features"][indices]
            targets = data["targets"][indices]
            normalized_target = flow.normalize_actions(targets)
            warm = data["warm"][indices]
            cold_replay = torch.rand(
                warm.shape, device=device, generator=generator
            ) < config.cold_replay_probability
            warm = warm & ~cold_replay
            normalized_initial = flow.normalize_actions(data["initials"][indices])
            noise = torch.randn(
                normalized_initial.shape,
                device=device,
                dtype=normalized_initial.dtype,
                generator=generator,
            )
            normalized_initial = normalized_initial + (
                config.warm_start_noise_std * noise
            )
            initial = torch.where(
                warm[:, :, None], normalized_initial, torch.zeros_like(normalized_initial)
            )
            tau = torch.rand(
                (indices.shape[0],),
                device=device,
                dtype=features.dtype,
                generator=generator,
            )
            path = (
                (1.0 - tau[:, None, None]) * initial
                + tau[:, None, None] * normalized_target
            )
            target_velocity = normalized_target - initial
            optimizer.zero_grad(set_to_none=True)
            replay_loss = torch.zeros((), device=device, dtype=features.dtype)
            replay_sampled: dict[str, Tensor] | None = None
            replay_features: Tensor | None = None
            if replay_enabled:
                assert replay_loader is not None
                assert world_model is not None
                assert action_prior is not None
                assert replay_iterator is not None
                try:
                    raw_replay = next(replay_iterator)
                except StopIteration:
                    replay_iterator = iter(replay_loader)
                    raw_replay = next(replay_iterator)
                replay_batch = _batch_to_device(raw_replay, device)
                with torch.no_grad():
                    replay_hidden, replay_state, replay_features = (
                        world_model.encode_planning_history(_history(replay_batch))
                    )
                    replay_targets = action_prior_teacher_chunk(
                        world_model,
                        action_prior,
                        replay_hidden,
                        replay_state,
                        steps=flow.config.horizon,
                    ).to(dtype=replay_features.dtype)
                replay_sampled = make_flow_matching_batch(
                    flow,
                    replay_targets,
                    config=replay_sampling,
                    generator=generator,
                )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                predicted_velocity = flow(
                    path,
                    tau,
                    features,
                    initial,
                    warm.to(dtype=features.dtype),
                )
                flow_loss = (predicted_velocity - target_velocity).square().mean()
                endpoint_normalized = path + (
                    1.0 - tau[:, None, None]
                ) * predicted_velocity
                endpoint = flow.denormalize_actions(endpoint_normalized)
                endpoint_loss = (endpoint - targets).square().mean()
                smoothness_loss = (
                    (endpoint[:, 1:] - endpoint[:, :-1])
                    - (targets[:, 1:] - targets[:, :-1])
                ).square().mean()
                solver_current = initial
                solver_dt = 1.0 / config.solver_steps
                warm_float = warm.to(dtype=features.dtype)
                for solver_step in range(config.solver_steps):
                    solver_time = torch.full(
                        (features.shape[0],),
                        solver_step * solver_dt,
                        device=device,
                        dtype=features.dtype,
                    )
                    solver_current = solver_current + solver_dt * flow(
                        solver_current,
                        solver_time,
                        features,
                        initial,
                        warm_float,
                    )
                solver_actions = flow.denormalize_actions(solver_current)
                solver_endpoint_loss = (solver_actions - targets).square().mean()
                if replay_sampled is not None and replay_features is not None:
                    replay_velocity = flow(
                        replay_sampled["path"],
                        replay_sampled["time"],
                        replay_features,
                        replay_sampled["initial"],
                        replay_sampled["warm"],
                    )
                    replay_flow_loss = (
                        replay_velocity - replay_sampled["velocity"]
                    ).square().mean()
                    replay_endpoint_normalized = replay_sampled["path"] + (
                        1.0 - replay_sampled["time"][:, None, None]
                    ) * replay_velocity
                    replay_endpoint = flow.denormalize_actions(
                        replay_endpoint_normalized
                    )
                    replay_loss = replay_flow_loss + (
                        replay_endpoint
                        - flow.denormalize_actions(replay_sampled["target"])
                    ).square().mean()
                loss = (
                    flow_loss
                    + config.endpoint_weight * endpoint_loss
                    + config.smoothness_weight * smoothness_loss
                    + config.solver_endpoint_weight * solver_endpoint_loss
                    + config.offline_replay_weight * replay_loss
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite on-policy action-flow loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                flow.parameters(), config.gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            completed_steps += 1
            item = {
                "loss": float(loss.detach().cpu()),
                "flow": float(flow_loss.detach().cpu()),
                "endpoint": float(endpoint_loss.detach().cpu()),
                "smoothness": float(smoothness_loss.detach().cpu()),
                "solver_endpoint": float(solver_endpoint_loss.detach().cpu()),
                "offline_replay": float(replay_loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "warm_fraction": float(warm.float().mean().detach().cpu()),
                "epoch": float(epoch + 1),
            }
            history.append(item)
            if progress is not None:
                progress({"step": completed_steps, **item})
    return history, completed_steps


@torch.no_grad()
def action_prior_teacher_chunk(
    world_model: RWMARWorldModel,
    action_prior: ActionPrior,
    hidden: Tensor,
    current_state: Tensor,
    *,
    steps: int,
) -> Tensor:
    """Roll the frozen one-step prior only for offline action-flow teacher targets."""

    if steps <= 0:
        raise ValueError("teacher steps must be positive")
    actions: list[Tensor] = []
    recurrent = hidden
    state = current_state
    for _ in range(steps):
        features = world_model.planning_features(recurrent, state)
        action = action_prior.deterministic_action(features)
        actions.append(action)
        recurrent, state, _ = world_model.imagine_step(
            recurrent, state, action, sample_state=False
        )
    return torch.stack(actions, dim=1)


def _history(batch: Mapping[str, Tensor]) -> WorldModelSequenceInputs:
    return WorldModelSequenceInputs(
        states=batch["states"],
        past_actions=batch["past_actions"],
        valid_mask=batch["valid_mask"],
    )


def _batch_to_device(
    batch: Mapping[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
    }


def _freeze_world_model(model: RWMARWorldModel) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _preferred_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


__all__ = [
    "ActionFlowDistillationBuffer",
    "ActionFlowOnPolicyTrainConfig",
    "ActionFlowTrainConfig",
    "action_prior_teacher_chunk",
    "fine_tune_action_flow_on_policy",
    "make_flow_matching_batch",
    "synthetic_shifted_warm_start",
    "train_action_flow",
]
