"""Stage-wise scratch curriculum for task-side M1 modules."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import torch
from torch import Tensor, nn

from models.wam import StatefulActionFlow, WorldModelSequenceInputs
from models.wam_multimodal import LatentWAM
from train.joint_wam import differentiable_flow_generate
from train.m1_training import (
    M1FlowObjectiveConfig,
    M1LossWeights,
    m1_batch_loss,
    m1_batch_required_keys,
)


ScratchStageObjective = Literal[
    "dynamics_warmup",
    "action_flow_warmup",
    "multimodal_fusion",
    "future_joint",
]


@dataclass(frozen=True)
class ScratchM1StageConfig:
    """One stage in a single-initialization scratch run."""

    name: str
    objective: ScratchStageObjective
    steps: int
    world_learning_rate: float = 0.0
    action_flow_learning_rate: float = 0.0
    multimodal_learning_rate: float = 0.0
    future_learning_rate: float = 0.0
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    losses: M1LossWeights = M1LossWeights()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scratch stage name cannot be empty")
        if self.objective not in {
            "dynamics_warmup",
            "action_flow_warmup",
            "multimodal_fusion",
            "future_joint",
        }:
            raise ValueError(f"unknown scratch stage objective {self.objective!r}")
        if int(self.steps) <= 0:
            raise ValueError("scratch stage steps must be positive")
        learning_rates = (
            self.world_learning_rate,
            self.action_flow_learning_rate,
            self.multimodal_learning_rate,
            self.future_learning_rate,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in learning_rates):
            raise ValueError("scratch learning rates must be finite and non-negative")
        if not any(value > 0.0 for value in learning_rates):
            raise ValueError("scratch stage must enable at least one optimizer group")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid scratch optimizer controls")
        loss_values = asdict(self.losses)
        if any(not math.isfinite(float(value)) for value in loss_values.values()):
            raise ValueError("scratch loss weights must be finite")
        if self.objective == "dynamics_warmup" and (
            self.world_learning_rate <= 0.0
            or any(value > 0.0 for value in learning_rates[1:])
        ):
            raise ValueError("dynamics warmup must train only the world model")
        if (
            self.objective == "dynamics_warmup"
            and self.losses.future_state <= 0.0
        ):
            raise ValueError("dynamics warmup requires a positive state loss")
        if self.objective == "action_flow_warmup" and (
            self.action_flow_learning_rate <= 0.0
            or self.multimodal_learning_rate > 0.0
            or self.future_learning_rate > 0.0
        ):
            raise ValueError("action-flow warmup cannot train multimodal/future modules")
        action_losses = (
            self.losses.flow_matching,
            self.losses.action_endpoint,
            self.losses.action_smoothness,
        )
        if self.objective in {"action_flow_warmup", "multimodal_fusion"} and not any(
            value > 0.0 for value in action_losses
        ):
            raise ValueError(f"{self.objective} requires a positive action loss")
        if self.objective == "multimodal_fusion" and (
            self.action_flow_learning_rate <= 0.0
            or self.multimodal_learning_rate <= 0.0
            or self.future_learning_rate > 0.0
        ):
            raise ValueError("multimodal fusion must train flow+fusion, not future head")
        if self.objective == "future_joint" and self.future_learning_rate <= 0.0:
            raise ValueError("future joint stage must train the future head")
        if (
            self.objective == "future_joint"
            and self.losses.future_visual_latent <= 0.0
        ):
            raise ValueError("future joint stage requires a positive future loss")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["losses"] = asdict(self.losses)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScratchM1StageConfig":
        values = dict(payload)
        raw_losses = values.get("losses", {})
        if not isinstance(raw_losses, Mapping):
            raise ValueError("scratch stage losses must be a mapping")
        values["losses"] = M1LossWeights(**dict(raw_losses))
        return cls(**values)


@dataclass(frozen=True)
class ScratchM1BatchLoss:
    total: Tensor
    dynamics: Tensor
    flow_matching: Tensor
    action_endpoint: Tensor
    action_smoothness: Tensor
    future_visual_latent: Tensor
    future_state: Tensor


ScratchProgressCallback = Callable[[int, Mapping[str, float]], None]


def build_scratch_optimizer(
    model: LatentWAM,
    flow: StatefulActionFlow,
    *,
    weight_decay: float = 1e-5,
) -> torch.optim.AdamW:
    """Build one optimizer whose state survives every scratch stage."""

    if flow.has_anchor:
        raise ValueError("scratch optimizer rejects an action flow with an anchor")
    if weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative")
    role_parameters: dict[str, list[nn.Parameter]] = {
        "world": [],
        "action_flow": list(flow.parameters()),
        "multimodal": [],
        "future": [],
    }
    for name, parameter in model.named_parameters():
        if name.startswith("vision_encoder."):
            continue
        if name.startswith("world_model."):
            role_parameters["world"].append(parameter)
        elif name.startswith("future_head."):
            role_parameters["future"].append(parameter)
        elif name.startswith(
            ("resampler.", "fusion.", "task_embedding.", "action_capacity_mlp.")
        ):
            role_parameters["multimodal"].append(parameter)
        else:
            raise RuntimeError(f"unclassified scratch M1 parameter: {name}")
    if not role_parameters["world"] or not role_parameters["action_flow"]:
        raise ValueError("scratch optimizer requires world and action-flow parameters")
    groups = [
        {"params": parameters, "lr": 0.0, "role": role}
        for role, parameters in role_parameters.items()
        if parameters
    ]
    return torch.optim.AdamW(groups, lr=0.0, weight_decay=weight_decay)


def configure_scratch_stage(
    model: LatentWAM,
    flow: StatefulActionFlow,
    optimizer: torch.optim.Optimizer,
    stage: ScratchM1StageConfig,
) -> dict[str, Any]:
    """Apply stage learning rates/freezes while retaining optimizer moments."""

    if flow.has_anchor:
        raise ValueError("scratch stage rejects an action flow with an anchor")
    rates = {
        "world": float(stage.world_learning_rate),
        "action_flow": float(stage.action_flow_learning_rate),
        "multimodal": float(stage.multimodal_learning_rate),
        "future": float(stage.future_learning_rate),
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    active_counts: dict[str, int] = {}
    observed_roles: set[str] = set()
    for group in optimizer.param_groups:
        role = str(group.get("role", ""))
        if role not in rates or role in observed_roles:
            raise ValueError(f"invalid scratch optimizer role {role!r}")
        observed_roles.add(role)
        group["lr"] = rates[role]
        active = rates[role] > 0.0
        parameters = list(group["params"])
        for parameter in parameters:
            parameter.requires_grad_(active)
        active_counts[role] = sum(parameter.numel() for parameter in parameters) if active else 0
    missing_active_roles = sorted(
        role for role, rate in rates.items() if rate > 0.0 and role not in observed_roles
    )
    if missing_active_roles:
        raise ValueError(
            "scratch stage requested unavailable optimizer roles: "
            f"{missing_active_roles}"
        )
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
        if any(parameter.requires_grad for parameter in model.vision_encoder.parameters()):
            raise RuntimeError("scratch stage attempted to unfreeze the vision encoder")
    if not any(active_counts.values()):
        raise ValueError("scratch stage selected no trainable parameters")
    return {
        "name": stage.name,
        "objective": stage.objective,
        "learning_rates": rates,
        "trainable_parameters_by_role": active_counts,
        "optimizer_state_policy": "preserve_across_stages",
        "vision_encoder_frozen": True,
        "action_anchor_mode": "none",
    }


def scratch_m1_batch_loss(
    model: LatentWAM,
    flow: StatefulActionFlow,
    batch: Mapping[str, Tensor],
    stage: ScratchM1StageConfig,
    *,
    flow_objective: M1FlowObjectiveConfig,
    generator: torch.Generator | None = None,
) -> ScratchM1BatchLoss:
    """Dispatch to the objective appropriate for one scratch stage."""

    if flow_objective.policy_fixed_action_dims:
        raise ValueError("scratch M1 must supervise all action dimensions")
    zero = next(model.parameters()).new_zeros(())
    if stage.objective == "dynamics_warmup":
        dynamics = _dynamics_loss(model, batch)
        return ScratchM1BatchLoss(
            total=stage.losses.future_state * dynamics,
            dynamics=dynamics,
            flow_matching=zero,
            action_endpoint=zero,
            action_smoothness=zero,
            future_visual_latent=zero,
            future_state=dynamics,
        )
    if stage.objective == "action_flow_warmup":
        features = _state_planning_features(model, batch)
        flow_losses = _cold_action_flow_loss(
            flow,
            features.detach() if stage.world_learning_rate == 0.0 else features,
            batch["action_targets"],
            weights=stage.losses,
            flow_objective=flow_objective,
            generator=generator,
        )
        dynamics = (
            _dynamics_loss(model, batch)
            if stage.world_learning_rate > 0.0 and stage.losses.future_state > 0.0
            else zero
        )
        total = flow_losses["total"] + stage.losses.future_state * dynamics
        return ScratchM1BatchLoss(
            total=total,
            dynamics=dynamics,
            flow_matching=flow_losses["flow_matching"],
            action_endpoint=flow_losses["action_endpoint"],
            action_smoothness=flow_losses["action_smoothness"],
            future_visual_latent=zero,
            future_state=dynamics,
        )

    joint = m1_batch_loss(
        model,
        flow,
        batch,
        weights=stage.losses,
        flow_objective=flow_objective,
        generator=generator,
    )
    return ScratchM1BatchLoss(
        total=joint.total,
        dynamics=joint.future_state,
        flow_matching=joint.flow_matching,
        action_endpoint=joint.action_endpoint,
        action_smoothness=joint.action_smoothness,
        future_visual_latent=joint.future_visual_latent,
        future_state=joint.future_state,
    )


def scratch_stage_required_keys(
    model: LatentWAM,
    stage: ScratchM1StageConfig,
) -> frozenset[str]:
    """Return only the sample tensors consumed by one scratch stage."""

    state_history = {"states", "state_valid_mask", "past_actions"}
    if stage.objective == "dynamics_warmup":
        return frozenset(
            {*state_history, "action_targets", "future_states"}
        )
    if stage.objective == "action_flow_warmup":
        required = {*state_history, "action_targets"}
        if stage.world_learning_rate > 0.0 and stage.losses.future_state > 0.0:
            required.add("future_states")
        return frozenset(required)
    return m1_batch_required_keys(model, stage.losses)


def train_scratch_m1_stage(
    model: LatentWAM,
    flow: StatefulActionFlow,
    batches: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    stage: ScratchM1StageConfig,
    *,
    device: torch.device,
    flow_objective: M1FlowObjectiveConfig,
    seed: int,
    precision: str = "fp32",
    progress: ScratchProgressCallback | None = None,
) -> dict[str, Any]:
    """Run a finite stage without reinitializing model or optimizer state."""

    evidence = configure_scratch_stage(model, flow, optimizer, stage)
    model.train()
    flow.train()
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
    if precision not in {"fp32", "bf16"}:
        raise ValueError("scratch precision must be 'fp32' or 'bf16'")
    if precision == "bf16" and device.type != "cuda":
        raise ValueError("scratch bf16 precision requires CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("scratch bf16 precision requires native CUDA BF16 support")
    amp_enabled = precision == "bf16"
    generator = torch.Generator(device=device).manual_seed(int(seed))
    iterator = _DevicePrefetcher(iter(batches), device)
    history: list[dict[str, float]] = []
    for completed in range(1, stage.steps + 1):
        batch = iterator.next()
        if batch is None:
            iterator = _DevicePrefetcher(iter(batches), device)
            batch = iterator.next()
            if batch is None:
                raise ValueError("scratch training iterable yielded no batches")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            loss = scratch_m1_batch_loss(
                model,
                flow,
                batch,
                stage,
                flow_objective=flow_objective,
                generator=generator,
            )
        if not bool(torch.isfinite(loss.total)):
            raise FloatingPointError(f"scratch stage {stage.name} produced non-finite loss")
        loss.total.backward()
        trainable = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
        if not gradients:
            raise FloatingPointError("scratch stage gradients are missing")
        try:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                stage.gradient_clip_norm,
                error_if_nonfinite=True,
            )
        except RuntimeError as exc:
            raise FloatingPointError(
                "scratch stage gradients are non-finite"
            ) from exc
        optimizer.step()
        values = torch.stack(
            (
                loss.total.detach(),
                loss.dynamics.detach(),
                loss.flow_matching.detach(),
                loss.action_endpoint.detach(),
                loss.future_visual_latent.detach(),
                gradient_norm.detach(),
            )
        ).float().cpu().tolist()
        metrics = dict(
            zip(
                (
                    "total",
                    "dynamics",
                    "flow_matching",
                    "action_endpoint",
                    "future_visual_latent",
                    "gradient_norm",
                ),
                map(float, values),
                strict=True,
            )
        )
        history.append(metrics)
        if progress is not None:
            progress(completed, metrics)
    if model.vision_encoder is not None and any(
        parameter.grad is not None for parameter in model.vision_encoder.parameters()
    ):
        raise RuntimeError("frozen vision encoder received gradients")
    return {**evidence, "steps": stage.steps, "history": history}


class _DevicePrefetcher:
    """Overlap pinned-memory H2D copies with the previous CUDA training step."""

    def __init__(
        self,
        source: Iterator[Mapping[str, Tensor]],
        device: torch.device,
    ) -> None:
        self.source = source
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._next: dict[str, Any] | None = None
        self._preload()

    def _preload(self) -> None:
        try:
            raw = next(self.source)
        except StopIteration:
            self._next = None
            return
        if self.stream is None:
            self._next = _batch_to_device(raw, self.device)
            return
        with torch.cuda.stream(self.stream):
            self._next = _batch_to_device(raw, self.device)

    def next(self) -> dict[str, Any] | None:
        batch = self._next
        if batch is None:
            return None
        if self.stream is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_stream(self.stream)
            for value in batch.values():
                if isinstance(value, Tensor):
                    value.record_stream(current)
        self._preload()
        return batch


def _batch_to_device(
    raw: Mapping[str, Tensor],
    device: torch.device,
) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, Tensor)
        else value
        for name, value in raw.items()
    }


def validate_scratch_stage_order(stages: Sequence[ScratchM1StageConfig]) -> None:
    expected = (
        "dynamics_warmup",
        "action_flow_warmup",
        "multimodal_fusion",
        "future_joint",
    )
    observed = tuple(stage.objective for stage in stages)
    if observed != expected:
        raise ValueError(f"scratch stage order must be {expected}, got {observed}")


def _state_planning_features(
    model: LatentWAM, batch: Mapping[str, Tensor]
) -> Tensor:
    required = {"states", "past_actions", "state_valid_mask"}
    missing = sorted(required - set(batch))
    if missing:
        raise KeyError(f"state scratch batch is missing {missing}")
    history = WorldModelSequenceInputs(
        states=batch["states"],
        past_actions=batch["past_actions"],
        valid_mask=batch["state_valid_mask"].bool(),
    )
    return model.world_model.encode_planning_history(history)[2]


def _dynamics_loss(model: LatentWAM, batch: Mapping[str, Tensor]) -> Tensor:
    required = {"states", "past_actions", "state_valid_mask", "action_targets", "future_states"}
    missing = sorted(required - set(batch))
    if missing:
        raise KeyError(f"dynamics scratch batch is missing {missing}")
    history = WorldModelSequenceInputs(
        states=batch["states"],
        past_actions=batch["past_actions"],
        valid_mask=batch["state_valid_mask"].bool(),
    )
    predictions = model.world_model.predict(history, batch["action_targets"])
    target = batch["future_states"]
    if predictions.next_state_mean.shape != target.shape:
        raise ValueError("scratch dynamics prediction/target shapes differ")
    scale = model.world_model.delta_std.to(predictions.next_state_mean)
    return (((predictions.next_state_mean - target) / scale) ** 2).mean()


def _cold_action_flow_loss(
    flow: StatefulActionFlow,
    features: Tensor,
    actions: Tensor,
    *,
    weights: M1LossWeights,
    flow_objective: M1FlowObjectiveConfig,
    generator: torch.Generator | None,
) -> dict[str, Tensor]:
    if actions.ndim != 3 or actions.shape[1:] != (
        flow.config.horizon,
        flow.config.action_dim,
    ):
        raise ValueError("scratch action targets must have shape [B,H,A]")
    normalized = flow.normalize_actions(actions)
    initial = torch.zeros_like(normalized)
    if generator is None:
        tau = torch.rand(actions.shape[0], device=actions.device, dtype=actions.dtype)
    else:
        tau = torch.rand(
            (actions.shape[0],),
            device=actions.device,
            dtype=actions.dtype,
            generator=generator,
        )
    interpolated = tau[:, None, None] * normalized
    warm = torch.zeros((actions.shape[0], 1), device=actions.device, dtype=actions.dtype)
    velocity = flow(interpolated, tau, features, initial, warm)
    flow_matching = (velocity - normalized).square().mean()
    endpoint = differentiable_flow_generate(
        flow,
        features,
        solver_steps=flow_objective.solver_steps,
        solver=flow_objective.solver,
        normalized_clip=flow_objective.normalized_action_clip,
    )
    action_endpoint = (endpoint - actions).square().mean()
    action_smoothness = (
        ((endpoint[:, 1:] - endpoint[:, :-1]) - (actions[:, 1:] - actions[:, :-1]))
        .square()
        .mean()
    )
    total = (
        weights.flow_matching * flow_matching
        + weights.action_endpoint * action_endpoint
        + weights.action_smoothness * action_smoothness
    )
    return {
        "total": total,
        "flow_matching": flow_matching,
        "action_endpoint": action_endpoint,
        "action_smoothness": action_smoothness,
    }


__all__ = [
    "ScratchM1BatchLoss",
    "ScratchM1StageConfig",
    "build_scratch_optimizer",
    "configure_scratch_stage",
    "scratch_m1_batch_loss",
    "scratch_stage_required_keys",
    "train_scratch_m1_stage",
    "validate_scratch_stage_order",
]
