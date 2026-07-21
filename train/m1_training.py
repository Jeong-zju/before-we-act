"""Stage-wise training utilities for the Phase M1 latent WAM.

The module keeps the causal boundaries explicit: future RGB is used only as a
detached frozen-teacher target, while the deployed action path receives current
state/RGB history, task identity, and past executed actions only.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.wam import StatefulActionFlow
from models.wam_multimodal import LatentWAM
from train.action_flow import synthetic_shifted_warm_start
from train.joint_wam import differentiable_flow_generate


M1_STATE_PAIR_MIN_STEP0_ACTION_DELTA = 1e-3


@dataclass(frozen=True)
class M1LossWeights:
    flow_matching: float = 0.5
    action_endpoint: float = 12.0
    action_smoothness: float = 0.05
    future_visual_latent: float = 1.0
    future_state: float = 0.1

    def __post_init__(self) -> None:
        if any(float(value) < 0.0 for value in asdict(self).values()):
            raise ValueError("M1 loss weights must be non-negative")


def m1_batch_required_keys(
    model: LatentWAM,
    weights: M1LossWeights,
) -> frozenset[str]:
    """Return the exact ordinary-batch tensors consumed by one objective."""

    required = {"task_index", "action_targets", "future_horizons"}
    if model.config.use_state:
        required.update({"states", "state_valid_mask", "past_actions"})
        if weights.future_state > 0.0:
            required.add("future_states")
    if model.config.use_vision:
        required.add("images")
    if model.future_head is not None and weights.future_visual_latent > 0.0:
        required.update({"future_images", "future_image_novelty_mask"})
    return frozenset(required)


def action_chunk_required_keys(model: LatentWAM) -> frozenset[str]:
    """Return the deployable inputs and target needed by cold BC validation."""

    required = {"task_index", "action_targets"}
    if model.config.use_state:
        required.update({"states", "state_valid_mask", "past_actions"})
    if model.config.use_vision:
        required.add("images")
    return frozenset(required)


@dataclass(frozen=True)
class M1CausalPairWeights:
    """Weights for observation-matched, RGB-contrasting BC anchors.

    The pair objective is intentionally defined on the same cold four-step
    action solver used at reset.  Pair identity and physical seed are sampler
    metadata only; neither is accepted by :func:`m1_causal_pair_loss`.
    """

    factual_endpoint: float = 6.0
    action_delta: float = 6.0
    delta_direction: float = 0.25
    executed_prefix_weight: float = 4.0

    def __post_init__(self) -> None:
        objectives = (
            float(self.factual_endpoint),
            float(self.action_delta),
            float(self.delta_direction),
        )
        if any(value < 0.0 or not math.isfinite(value) for value in objectives):
            raise ValueError("M1 causal-pair weights must be finite and non-negative")
        if not any(value > 0.0 for value in objectives):
            raise ValueError("at least one M1 causal-pair objective must be enabled")
        prefix = float(self.executed_prefix_weight)
        if not math.isfinite(prefix) or prefix < 1.0:
            raise ValueError("causal-pair executed_prefix_weight must be at least one")


@dataclass(frozen=True)
class M1StateCausalPairWeights:
    """Weights for proprio-identifiable, observation-matched BC anchors.

    State pairs hold task, RGB and past executed actions fixed.  Their target
    delta is intentionally normalized relative to its own (small) natural
    correction magnitude so ordinary large-action examples cannot drown out
    the only offline contrast that identifies current proprioception.
    """

    factual_endpoint: float = 2.0
    action_delta: float = 1.0
    delta_direction: float = 0.1
    target_rms_floor: float = 0.01

    def __post_init__(self) -> None:
        objectives = (
            float(self.factual_endpoint),
            float(self.action_delta),
            float(self.delta_direction),
        )
        if any(value < 0.0 or not math.isfinite(value) for value in objectives):
            raise ValueError(
                "M1 state causal-pair weights must be finite and non-negative"
            )
        if not any(value > 0.0 for value in objectives):
            raise ValueError(
                "at least one M1 state causal-pair objective must be enabled"
            )
        floor = float(self.target_rms_floor)
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError("state causal-pair target_rms_floor must be positive")


@dataclass(frozen=True)
class M1FlowObjectiveConfig:
    """Locked flow objective matching the Phase M1 runtime sampler.

    The policy deterministically starts cold plans at the normalized action
    mean, then replans after two executed actions from a shifted previous
    chunk.  Endpoint supervision must therefore traverse the same four-step
    Euler solver instead of a one-velocity proxy.
    """

    execution_steps: int = 2
    solver_steps: int = 4
    solver: str = "euler"
    normalized_action_clip: float = 10.0
    warm_start_probability: float = 0.5
    warm_start_noise_std: float = 0.01
    policy_fixed_action_dims: tuple[int, ...] = (3, 7)
    executed_prefix_weight: float = 2.0

    def __post_init__(self) -> None:
        if int(self.execution_steps) <= 0:
            raise ValueError("M1 execution_steps must be positive")
        if int(self.solver_steps) <= 0:
            raise ValueError("M1 solver_steps must be positive")
        if self.solver not in {"euler", "heun"}:
            raise ValueError("M1 solver must be euler or heun")
        if float(self.normalized_action_clip) <= 0.0:
            raise ValueError("M1 normalized_action_clip must be positive")
        if not 0.0 <= float(self.warm_start_probability) <= 1.0:
            raise ValueError("M1 warm_start_probability must be in [0,1]")
        if float(self.warm_start_noise_std) < 0.0:
            raise ValueError("M1 warm_start_noise_std cannot be negative")
        fixed = tuple(int(value) for value in self.policy_fixed_action_dims)
        if any(value < 0 for value in fixed) or len(set(fixed)) != len(fixed):
            raise ValueError(
                "policy_fixed_action_dims must contain unique non-negative indices"
            )
        if float(self.executed_prefix_weight) < 1.0:
            raise ValueError("executed_prefix_weight must be at least one")
        object.__setattr__(self, "policy_fixed_action_dims", fixed)


@dataclass(frozen=True)
class M1StageConfig:
    name: str
    steps: int
    learning_rate: float
    world_learning_rate: float = 0.0
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    train_visual_adapter: bool = True
    train_fusion: bool = True
    train_future_head: bool = False
    train_action_flow: bool = False
    train_world_model: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("M1 stage name cannot be empty")
        if int(self.steps) <= 0:
            raise ValueError("M1 stage steps must be positive")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("M1 learning rate must be positive")
        if float(self.world_learning_rate) < 0.0:
            raise ValueError("M1 world learning rate cannot be negative")
        if self.train_world_model and not (
            self.learning_rate / 20.0
            <= self.world_learning_rate
            <= self.learning_rate / 10.0
        ):
            raise ValueError("world-model LR must be 10x to 20x lower than adapter LR")
        if self.weight_decay < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid optimizer controls")


@dataclass(frozen=True)
class M1BatchLoss:
    total: Tensor
    flow_matching: Tensor
    action_endpoint: Tensor
    action_smoothness: Tensor
    future_visual_latent: Tensor
    future_state: Tensor
    endpoint_actions: Tensor
    flow_initial_normalized: Tensor
    warm_start_actions: Tensor
    warm_start_mask: Tensor
    supervised_action_mask: Tensor
    teacher_future_latents: Tensor | None
    future_target_detached: bool


@dataclass(frozen=True)
class M1CausalPairLoss:
    """Differentiable causal-anchor evidence returned by one pair batch."""

    total: Tensor
    factual_endpoint: Tensor
    action_delta: Tensor
    delta_direction: Tensor
    delta_cosine: Tensor
    delta_cosine_valid_fraction: Tensor
    predicted_actions: Tensor
    target_actions: Tensor
    predicted_delta_rms: Tensor
    target_delta_rms: Tensor
    supervised_action_mask: Tensor


@dataclass(frozen=True)
class M1StateCausalPairLoss:
    """Differentiable proprioceptive contrast on the deployed execute-2 path."""

    total: Tensor
    factual_endpoint: Tensor
    action_delta: Tensor
    delta_direction: Tensor
    delta_cosine: Tensor
    delta_cosine_valid_fraction: Tensor
    predicted_actions: Tensor
    target_actions: Tensor
    predicted_delta_rms: Tensor
    target_delta_rms: Tensor
    step0_predicted_delta_rms: Tensor
    step0_target_delta_rms: Tensor
    supervised_action_mask: Tensor


def seed_everything(seed: int) -> None:
    """Seed every local RNG used by the deterministic CPU/GPU trainer."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def m1_batch_loss(
    model: LatentWAM,
    flow: StatefulActionFlow,
    batch: Mapping[str, Tensor],
    *,
    weights: M1LossWeights = M1LossWeights(),
    flow_objective: M1FlowObjectiveConfig = M1FlowObjectiveConfig(),
    generator: torch.Generator | None = None,
) -> M1BatchLoss:
    """Compute action-flow, future-latent, and low-LR world objectives.

    ``future_images`` never enter :meth:`LatentWAM.encode`; they are passed
    directly through the frozen teacher under ``no_grad`` and immediately
    detached.  This makes accidental future-frame conditioning observable in
    both code review and unit tests.
    """

    required = m1_batch_required_keys(model, weights)
    missing = sorted(required - set(batch))
    if missing:
        raise KeyError(f"M1 training batch is missing {missing}")

    # Pass absent modalities as ``None`` even when a generic dataloader happens
    # to include them.  This makes the ablations structural: state-only cannot
    # accidentally read RGB and vision-only cannot accidentally read proprio.
    states = batch["states"] if model.config.use_state else None
    past_actions = batch["past_actions"] if model.config.use_state else None
    valid_mask = batch["state_valid_mask"].bool() if model.config.use_state else None
    images = batch["images"] if model.config.use_vision else None
    task_index = batch["task_index"].long()
    actions = batch["action_targets"]
    future_images = batch.get("future_images")
    future_states = batch.get("future_states")
    novelty = batch.get("future_image_novelty_mask")

    horizons = batch["future_horizons"]
    expected_horizons = torch.tensor(
        model.future_horizons, device=horizons.device, dtype=horizons.dtype
    )
    if horizons.ndim == 1:
        horizon_match = torch.equal(horizons, expected_horizons)
    elif horizons.ndim == 2:
        horizon_match = bool(
            horizons.shape[1] == expected_horizons.numel()
            and torch.all(horizons == expected_horizons.unsqueeze(0))
        )
    else:
        horizon_match = False
    if not horizon_match:
        raise ValueError(
            "future_horizons must exactly match the model target order "
            f"{model.future_horizons}"
        )

    output = model(
        states,
        past_actions,
        valid_mask,
        images,
        task_index,
        candidate_actions=actions,
        compute_world_predictions=weights.future_state > 0.0,
        compute_future_visual_latents=weights.future_visual_latent > 0.0,
    )
    features = output.encoding.planning_features
    if features.shape != (actions.shape[0], flow.config.feature_dim):
        raise ValueError("latent planning feature/action-flow contract mismatch")

    normalized_target = flow.normalize_actions(actions)
    if flow.config.horizon != actions.shape[1]:
        raise ValueError("M1 action target/flow horizon mismatch")
    if not 0 < flow_objective.execution_steps < flow.config.horizon:
        raise ValueError("M1 execution_steps must be in [1,flow horizon)")
    if generator is None:
        warm_noise = torch.randn_like(normalized_target)
        warm_start_mask = (
            torch.rand(actions.shape[0], device=actions.device)
            < flow_objective.warm_start_probability
        )
        tau = torch.rand(actions.shape[0], device=actions.device, dtype=actions.dtype)
    else:
        warm_noise = torch.randn(
            normalized_target.shape,
            generator=generator,
            device=actions.device,
            dtype=actions.dtype,
        )
        warm_start_mask = (
            torch.rand(
                (actions.shape[0],),
                generator=generator,
                device=actions.device,
            )
            < flow_objective.warm_start_probability
        )
        tau = torch.rand(
            (actions.shape[0],),
            generator=generator,
            device=actions.device,
            dtype=actions.dtype,
        )

    # Runtime cold starts are exactly zero in normalized space.  Warm starts
    # shift an execute-2 plan and repeat its last action.  A small normalized
    # perturbation prevents the objective from learning to copy a perfect
    # synthetic prior while preserving the runtime's bounded action contract.
    cold_initial = torch.zeros_like(normalized_target)
    shifted = synthetic_shifted_warm_start(actions, flow_objective.execution_steps)
    warm_normalized = flow.normalize_actions(shifted)
    warm_normalized = warm_normalized + (
        flow_objective.warm_start_noise_std * warm_noise
    )
    warm_start_actions = flow.denormalize_actions(warm_normalized).clamp(-1.0, 1.0)
    warm_normalized = flow.normalize_actions(warm_start_actions).clamp(
        -flow_objective.normalized_action_clip,
        flow_objective.normalized_action_clip,
    )
    initial = torch.where(warm_start_mask[:, None, None], warm_normalized, cold_initial)
    interpolated = (1.0 - tau[:, None, None]) * initial + tau[
        :, None, None
    ] * normalized_target
    target_velocity = normalized_target - initial
    warm = warm_start_mask[:, None].to(dtype=actions.dtype)
    velocity = flow(interpolated, tau, features, initial, warm)
    supervised_action_mask = _supervised_action_mask(
        actions.shape[-1],
        flow_objective.policy_fixed_action_dims,
        device=actions.device,
    )
    flow_matching = _weighted_action_mse(
        velocity,
        target_velocity,
        supervised_action_mask,
    )

    # This is the deployed deterministic solver, kept differentiable so the
    # endpoint loss reaches both the action flow and its multimodal features.
    endpoint_actions = differentiable_flow_generate(
        flow,
        features,
        initial_actions=warm_start_actions,
        warm_start_mask=warm_start_mask,
        solver_steps=flow_objective.solver_steps,
        solver=flow_objective.solver,
        normalized_clip=flow_objective.normalized_action_clip,
    )
    action_endpoint = _weighted_action_mse(
        endpoint_actions,
        actions,
        supervised_action_mask,
        executed_steps=flow_objective.execution_steps,
        executed_prefix_weight=flow_objective.executed_prefix_weight,
    )
    if actions.shape[1] > 1:
        predicted_delta = endpoint_actions[:, 1:] - endpoint_actions[:, :-1]
        target_delta = actions[:, 1:] - actions[:, :-1]
        action_smoothness = _weighted_action_mse(
            predicted_delta,
            target_delta,
            supervised_action_mask,
            executed_steps=max(flow_objective.execution_steps - 1, 0),
            executed_prefix_weight=flow_objective.executed_prefix_weight,
        )
    else:
        action_smoothness = endpoint_actions.new_zeros(())

    teacher_future: Tensor | None = None
    future_latent = endpoint_actions.new_zeros(())
    if output.future_visual_latents is not None and weights.future_visual_latent > 0.0:
        if future_images is None or novelty is None:
            raise KeyError("future-head training requires future RGB and novelty mask")
        if model.vision_encoder is None:
            raise RuntimeError("future-head training requires a frozen visual teacher")
        if future_images.ndim != 6:
            raise ValueError("future_images must have shape [B,H,C,3,H,W]")
        batch_size, horizon_count, camera_count, channels, height, width = (
            future_images.shape
        )
        flattened = future_images.reshape(
            batch_size * horizon_count * camera_count, channels, height, width
        )
        with torch.no_grad():
            encoded = model.vision_encoder(flattened)
            pooled = encoded.pooled_latent.reshape(
                batch_size, horizon_count, camera_count, -1
            ).mean(dim=2)
            teacher_future = pooled.detach()
        prediction = output.future_visual_latents
        if prediction.shape != teacher_future.shape:
            raise ValueError("future visual prediction/teacher shape mismatch")
        novelty = novelty.bool()
        if novelty.ndim == 3:
            novelty = novelty.any(dim=-1)
        if novelty.shape != prediction.shape[:2]:
            raise ValueError("future-image novelty mask has the wrong shape")
        per_item = 1.0 - F.cosine_similarity(prediction, teacher_future, dim=-1)
        valid = novelty.to(per_item)
        future_latent = (per_item * valid).sum() / valid.sum().clamp_min(1.0)

    future_state = endpoint_actions.new_zeros(())
    if output.world_predictions is not None and weights.future_state > 0.0:
        if future_states is None:
            raise KeyError("state-enabled world loss requires future_states")
        predicted_state = output.world_predictions.next_state_mean
        if predicted_state.shape != future_states.shape:
            raise ValueError("world prediction/future state shape mismatch")
        state_scale = model.world_model.features.state_std.to(predicted_state)
        future_state = (((predicted_state - future_states) / state_scale) ** 2).mean()

    total = (
        weights.flow_matching * flow_matching
        + weights.action_endpoint * action_endpoint
        + weights.action_smoothness * action_smoothness
        + weights.future_visual_latent * future_latent
        + weights.future_state * future_state
    )
    return M1BatchLoss(
        total=total,
        flow_matching=flow_matching,
        action_endpoint=action_endpoint,
        action_smoothness=action_smoothness,
        future_visual_latent=future_latent,
        future_state=future_state,
        endpoint_actions=endpoint_actions,
        flow_initial_normalized=initial,
        warm_start_actions=warm_start_actions,
        warm_start_mask=warm_start_mask,
        supervised_action_mask=supervised_action_mask,
        teacher_future_latents=teacher_future,
        future_target_detached=(
            teacher_future is None or not teacher_future.requires_grad
        ),
    )


def m1_causal_pair_loss(
    model: LatentWAM,
    flow: StatefulActionFlow,
    pair_batch: Mapping[str, Tensor],
    *,
    weights: M1CausalPairWeights = M1CausalPairWeights(),
    flow_objective: M1FlowObjectiveConfig = M1FlowObjectiveConfig(),
) -> M1CausalPairLoss:
    """Supervise RGB causality using matched offline demonstration pairs.

    Inputs have shape ``[P,2,...]``.  The two members must have bitwise-equal
    deployable state/action history, differing RGB, and differing H=8 BC action
    targets on policy-controlled dimensions.  The model sees only deployable
    tensors; lineage strings returned by the dataset are deliberately ignored.
    """

    if not model.config.use_vision:
        raise ValueError("causal-pair supervision requires a visual model")
    if (
        flow_objective.execution_steps != 2
        or flow_objective.solver_steps != 4
        or flow_objective.solver != "euler"
        or flow_objective.policy_fixed_action_dims != (3, 7)
        or weights.executed_prefix_weight != 4.0
    ):
        raise ValueError(
            "M1 causal pairs require execute-2, four-step Euler, fixed dims "
            "(3,7), and temporal prefix weight four"
        )

    required = {
        "task_index",
        "action_targets",
        "states",
        "state_valid_mask",
        "past_actions",
        "images",
        "image_valid_mask",
    }
    missing = sorted(required - set(pair_batch))
    if missing:
        raise KeyError(f"M1 causal-pair batch is missing {missing}")

    actions = pair_batch["action_targets"].detach()
    if actions.ndim != 4 or actions.shape[1] != 2:
        raise ValueError("causal-pair action_targets must have shape [P,2,H,A]")
    pair_count, pair_members, horizon, action_dim = actions.shape
    if pair_count <= 0 or pair_members != 2:
        raise ValueError("causal-pair batch must contain non-empty pairs")
    if horizon != flow.config.horizon or action_dim != flow.config.action_dim:
        raise ValueError("causal-pair target/action-flow contract mismatch")

    task_index = pair_batch["task_index"]
    if task_index.shape != (pair_count, 2):
        raise ValueError("causal-pair task_index must have shape [P,2]")
    if not torch.equal(task_index[:, 0], task_index[:, 1]):
        raise ValueError("causal-pair members must share one task")

    audit_states = pair_batch["states"]
    audit_past_actions = pair_batch["past_actions"]
    audit_state_valid_mask = pair_batch["state_valid_mask"]
    if audit_state_valid_mask.dtype != torch.bool:
        raise TypeError("causal-pair state_valid_mask must be boolean")
    for name, value in (
        ("states", audit_states),
        ("past_actions", audit_past_actions),
        ("state_valid_mask", audit_state_valid_mask),
    ):
        if value.ndim < 3 or value.shape[:2] != (pair_count, 2):
            raise ValueError(f"causal-pair {name} must start with [P,2]")
        if not torch.equal(value[:, 0], value[:, 1]):
            raise ValueError(f"causal-pair {name} must be bitwise equal across members")
    states = audit_states if model.config.use_state else None
    past_actions = audit_past_actions if model.config.use_state else None
    state_valid_mask = audit_state_valid_mask if model.config.use_state else None

    images = pair_batch["images"]
    image_valid_mask = pair_batch["image_valid_mask"]
    if image_valid_mask.dtype != torch.bool:
        raise TypeError("causal-pair image_valid_mask must be boolean")
    if images.ndim != 7 or images.shape[:2] != (pair_count, 2):
        raise ValueError("causal-pair images must have shape [P,2,T,Cam,3,H,W]")
    if image_valid_mask.ndim != 4 or image_valid_mask.shape[:2] != (
        pair_count,
        2,
    ):
        raise ValueError("causal-pair image_valid_mask must have shape [P,2,T,Cam]")
    if not torch.equal(image_valid_mask[:, 0], image_valid_mask[:, 1]):
        raise ValueError("causal-pair members must share the RGB validity mask")
    visible_difference = (images[:, 0] != images[:, 1]) & image_valid_mask[
        :, 0, :, :, None, None, None
    ]
    if not torch.all(visible_difference.flatten(1).any(dim=1)):
        raise ValueError("each causal pair must contain differing valid RGB")

    supervised = _supervised_action_mask(
        action_dim,
        flow_objective.policy_fixed_action_dims,
        device=actions.device,
    )
    target_delta = actions[:, 1] - actions[:, 0]
    if not torch.all(
        target_delta[:, : flow_objective.execution_steps, supervised]
        .flatten(1)
        .abs()
        .amax(dim=1)
        > 0
    ):
        raise ValueError(
            "each causal pair must have an execute-2 controlled action difference"
        )

    flattened_size = pair_count * 2
    encoding = model.encode(
        None if states is None else states.reshape(flattened_size, *states.shape[2:]),
        (
            None
            if past_actions is None
            else past_actions.reshape(flattened_size, *past_actions.shape[2:])
        ),
        (
            None
            if state_valid_mask is None
            else state_valid_mask.reshape(flattened_size, *state_valid_mask.shape[2:])
        ),
        None if images is None else images.reshape(flattened_size, *images.shape[2:]),
        task_index.long().reshape(flattened_size),
        image_valid_mask=(
            None
            if image_valid_mask is None
            else image_valid_mask.reshape(flattened_size, *image_valid_mask.shape[2:])
        ),
    )
    predicted = differentiable_flow_generate(
        flow,
        encoding.planning_features,
        solver_steps=flow_objective.solver_steps,
        solver=flow_objective.solver,
        normalized_clip=flow_objective.normalized_action_clip,
    ).reshape(pair_count, 2, horizon, action_dim)

    # Pair supervision is restricted to the two actions the deployed policy
    # actually executes before replanning.  Ordinary BC remains responsible
    # for H=8; allowing pair deltas to shape the unexecuted suffix creates an
    # unnecessary open-loop shortcut and measurably harms held-out BC error.
    execute = slice(0, flow_objective.execution_steps)
    predicted_prefix = predicted[:, :, execute]
    action_prefix = actions[:, :, execute]
    factual = _weighted_action_mse(
        predicted_prefix.reshape(
            flattened_size, flow_objective.execution_steps, action_dim
        ),
        action_prefix.reshape(
            flattened_size, flow_objective.execution_steps, action_dim
        ),
        supervised,
        executed_steps=flow_objective.execution_steps,
        executed_prefix_weight=weights.executed_prefix_weight,
    )
    predicted_delta = predicted_prefix[:, 1] - predicted_prefix[:, 0]
    target_delta = action_prefix[:, 1] - action_prefix[:, 0]
    delta_mse = _weighted_action_mse(
        predicted_delta,
        target_delta,
        supervised,
        executed_steps=flow_objective.execution_steps,
        executed_prefix_weight=weights.executed_prefix_weight,
    )
    temporal = actions.new_full(
        (flow_objective.execution_steps,), weights.executed_prefix_weight
    )
    coordinate_weights = temporal[:, None] * supervised.to(dtype=actions.dtype)[None]
    root_weights = coordinate_weights.sqrt()[None]
    weighted_prediction = (predicted_delta * root_weights).flatten(1)
    weighted_target = (target_delta * root_weights).flatten(1)
    prediction_norm = weighted_prediction.norm(dim=-1)
    target_norm = weighted_target.norm(dim=-1)
    direction_valid = (prediction_norm.detach() > 1e-4) & (target_norm > 1e-8)
    stable_cosine = (weighted_prediction * weighted_target).sum(dim=-1) / (
        prediction_norm.clamp_min(1e-4) * target_norm.clamp_min(1e-8)
    )
    cosine = torch.where(
        direction_valid,
        stable_cosine.clamp(-1.0, 1.0),
        torch.zeros_like(stable_cosine),
    )
    direction = (
        (1.0 - cosine) * direction_valid.to(dtype=cosine.dtype)
    ).sum() / direction_valid.sum().clamp_min(1)
    valid_cosine = (
        cosine * direction_valid.to(dtype=cosine.dtype)
    ).sum() / direction_valid.sum().clamp_min(1)
    total = (
        weights.factual_endpoint * factual
        + weights.action_delta * delta_mse
        + weights.delta_direction * direction
    )
    active_coordinates = supervised.sum() * flow_objective.execution_steps * pair_count
    predicted_rms = torch.sqrt(
        predicted_delta[..., supervised].square().sum()
        / active_coordinates.clamp_min(1)
    )
    target_rms = torch.sqrt(
        target_delta[..., supervised].square().sum() / active_coordinates.clamp_min(1)
    )
    return M1CausalPairLoss(
        total=total,
        factual_endpoint=factual,
        action_delta=delta_mse,
        delta_direction=direction,
        delta_cosine=valid_cosine,
        delta_cosine_valid_fraction=direction_valid.to(dtype=cosine.dtype).mean(),
        predicted_actions=predicted,
        target_actions=actions,
        predicted_delta_rms=predicted_rms,
        target_delta_rms=target_rms,
        supervised_action_mask=supervised,
    )


def m1_state_causal_pair_loss(
    model: LatentWAM,
    flow: StatefulActionFlow,
    pair_batch: Mapping[str, Tensor],
    *,
    weights: M1StateCausalPairWeights = M1StateCausalPairWeights(),
    flow_objective: M1FlowObjectiveConfig = M1FlowObjectiveConfig(),
) -> M1StateCausalPairLoss:
    """Supervise current-state causality with RGB/action-history matched pairs.

    Each branch contains a real adjacent demonstration window.  Task, RGB,
    validity masks, and the complete presented past-action history must be
    bitwise equal; proprioceptive state is the only deployable model input that
    may differ.  Targets and opaque pair identities are never model inputs.
    """

    if not model.config.use_state:
        raise ValueError("state causal-pair supervision requires a state model")
    if (
        flow_objective.execution_steps != 2
        or flow_objective.solver_steps != 4
        or flow_objective.solver != "euler"
        or flow_objective.policy_fixed_action_dims != (3, 7)
    ):
        raise ValueError(
            "M1 state causal pairs require execute-2, four-step Euler, "
            "and fixed dims (3,7)"
        )
    required = {
        "task_index",
        "action_targets",
        "states",
        "state_valid_mask",
        "past_actions",
        "past_action_valid_mask",
        "images",
        "image_valid_mask",
    }
    missing = sorted(required - set(pair_batch))
    if missing:
        raise KeyError(f"M1 state causal-pair batch is missing {missing}")

    actions = pair_batch["action_targets"].detach()
    if actions.ndim != 4 or actions.shape[1] != 2:
        raise ValueError("state causal-pair action_targets must have shape [P,2,H,A]")
    pair_count, pair_members, horizon, action_dim = actions.shape
    if pair_count <= 0 or pair_members != 2:
        raise ValueError("state causal-pair batch must contain non-empty pairs")
    if horizon != flow.config.horizon or action_dim != flow.config.action_dim:
        raise ValueError("state causal-pair target/action-flow contract mismatch")

    task_index = pair_batch["task_index"]
    if task_index.shape != (pair_count, 2):
        raise ValueError("state causal-pair task_index must have shape [P,2]")
    if not torch.equal(task_index[:, 0], task_index[:, 1]):
        raise ValueError("state causal-pair members must share one task")

    states = pair_batch["states"]
    state_valid = pair_batch["state_valid_mask"]
    past_actions = pair_batch["past_actions"]
    past_valid = pair_batch["past_action_valid_mask"]
    if states.ndim != 4 or states.shape[:2] != (pair_count, 2):
        raise ValueError("state causal-pair states must have shape [P,2,T,S]")
    if past_actions.ndim != 4 or past_actions.shape[:2] != (pair_count, 2):
        raise ValueError("state causal-pair past_actions must have shape [P,2,T-1,A]")
    if states.shape[2] != past_actions.shape[2] + 1:
        raise ValueError("state causal-pair state/action history lengths mismatch")
    for name, mask, expected, valid_suffix in (
        ("state_valid_mask", state_valid, states.shape[:3], 4),
        ("past_action_valid_mask", past_valid, past_actions.shape[:3], 3),
    ):
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(expected):
            raise TypeError(f"state causal-pair {name} must be a boolean history mask")
        if not torch.equal(mask[:, 0], mask[:, 1]):
            raise ValueError(f"state causal-pair {name} must match across members")
        expected_mask = torch.zeros_like(mask)
        expected_mask[..., -valid_suffix:] = True
        if not torch.equal(mask, expected_mask):
            raise ValueError(
                f"state causal-pair {name} must have exactly {valid_suffix} "
                "right-aligned valid steps"
            )
    if not torch.equal(past_actions[:, 0], past_actions[:, 1]):
        raise ValueError(
            "state causal-pair past_actions must be bitwise equal across members"
        )
    state_difference = (states[:, 0, -4:] != states[:, 1, -4:]).flatten(1).any(dim=1)
    if not bool(state_difference.all()):
        raise ValueError("each state causal pair must contain differing states")
    if not bool((states[:, 0, -1] != states[:, 1, -1]).flatten(1).any(dim=1).all()):
        raise ValueError("each state causal pair must contain differing current states")

    images = pair_batch["images"]
    image_valid = pair_batch["image_valid_mask"]
    if images.ndim != 7 or images.shape[:2] != (pair_count, 2):
        raise ValueError("state causal-pair images must have shape [P,2,T,Cam,3,H,W]")
    if image_valid.dtype != torch.bool or image_valid.ndim != 4:
        raise TypeError("state causal-pair image_valid_mask must be boolean")
    if image_valid.shape[:2] != (pair_count, 2):
        raise ValueError("state causal-pair image_valid_mask must start with [P,2]")
    if not torch.equal(image_valid[:, 0], image_valid[:, 1]):
        raise ValueError("state causal-pair RGB validity masks must match")
    if not torch.equal(images[:, 0], images[:, 1]):
        raise ValueError("state causal-pair RGB must be bitwise equal")

    supervised = _supervised_action_mask(
        action_dim,
        flow_objective.policy_fixed_action_dims,
        device=actions.device,
    )
    step0_target_delta = actions[:, 1, 0] - actions[:, 0, 0]
    lateral = (0, 4)
    non_lateral_controlled = (1, 2, 5, 6)
    if not torch.all(
        step0_target_delta[:, lateral].abs().amax(dim=1)
        >= M1_STATE_PAIR_MIN_STEP0_ACTION_DELTA
    ):
        raise ValueError(
            "each state causal pair must have a non-trivial lateral step-0 "
            "action difference"
        )
    if bool((step0_target_delta[:, non_lateral_controlled].abs() > 1e-7).any()):
        raise ValueError("state causal-pair step-0 delta must be lateral only")
    if not torch.allclose(
        step0_target_delta[:, 0],
        step0_target_delta[:, 4],
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("state causal-pair robot lateral deltas must agree")
    current_position_delta = 0.5 * (
        states[:, 1, -1, 0]
        + states[:, 1, -1, 11]
        - states[:, 0, -1, 0]
        - states[:, 0, -1, 11]
    )
    mean_lateral_delta = step0_target_delta[:, lateral].mean(dim=1)
    if not bool((mean_lateral_delta * current_position_delta < 0.0).all()):
        raise ValueError(
            "state causal-pair target must be negative feedback on current position"
        )

    flattened_size = pair_count * 2
    encoding = model.encode(
        states.reshape(flattened_size, *states.shape[2:]),
        past_actions.reshape(flattened_size, *past_actions.shape[2:]),
        state_valid.reshape(flattened_size, *state_valid.shape[2:]),
        (
            images.reshape(flattened_size, *images.shape[2:])
            if model.config.use_vision
            else None
        ),
        task_index.long().reshape(flattened_size),
        image_valid_mask=(
            image_valid.reshape(flattened_size, *image_valid.shape[2:])
            if model.config.use_vision
            else None
        ),
    )
    predicted = differentiable_flow_generate(
        flow,
        encoding.planning_features,
        solver_steps=flow_objective.solver_steps,
        solver=flow_objective.solver,
        normalized_clip=flow_objective.normalized_action_clip,
    ).reshape(pair_count, 2, horizon, action_dim)

    execute = slice(0, flow_objective.execution_steps)
    predicted_prefix = flow.normalize_actions(predicted[:, :, execute])[..., supervised]
    target_prefix = flow.normalize_actions(actions[:, :, execute])[..., supervised]
    factual = F.mse_loss(predicted_prefix, target_prefix)
    predicted_delta = predicted_prefix[:, 1] - predicted_prefix[:, 0]
    target_delta = target_prefix[:, 1] - target_prefix[:, 0]
    target_rms_per_pair = target_delta.square().mean(dim=(1, 2)).sqrt()
    relative_scale = target_rms_per_pair.detach().clamp_min(weights.target_rms_floor)
    relative_error = (predicted_delta - target_delta) / relative_scale[:, None, None]
    delta_huber = F.smooth_l1_loss(
        relative_error, torch.zeros_like(relative_error), beta=1.0
    )

    flat_prediction = predicted_delta.flatten(1)
    flat_target = target_delta.flatten(1)
    prediction_norm = flat_prediction.norm(dim=-1)
    target_norm = flat_target.norm(dim=-1)
    direction_valid = target_rms_per_pair >= weights.target_rms_floor
    cosine = (flat_prediction * flat_target).sum(dim=-1) / (
        prediction_norm.clamp_min(1e-3) * target_norm.clamp_min(1e-8)
    )
    cosine = cosine.clamp(-1.0, 1.0)
    direction = (
        (1.0 - cosine) * direction_valid.to(dtype=cosine.dtype)
    ).sum() / direction_valid.sum().clamp_min(1)
    valid_cosine = (
        cosine * direction_valid.to(dtype=cosine.dtype)
    ).sum() / direction_valid.sum().clamp_min(1)
    total = (
        weights.factual_endpoint * factual
        + weights.action_delta * delta_huber
        + weights.delta_direction * direction
    )
    predicted_rms = predicted_delta.square().mean().sqrt()
    target_rms = target_delta.square().mean().sqrt()
    return M1StateCausalPairLoss(
        total=total,
        factual_endpoint=factual,
        action_delta=delta_huber,
        delta_direction=direction,
        delta_cosine=valid_cosine,
        delta_cosine_valid_fraction=direction_valid.to(dtype=cosine.dtype).mean(),
        predicted_actions=predicted,
        target_actions=actions,
        predicted_delta_rms=predicted_rms,
        target_delta_rms=target_rms,
        step0_predicted_delta_rms=predicted_delta[:, 0].square().mean().sqrt(),
        step0_target_delta_rms=target_delta[:, 0].square().mean().sqrt(),
        supervised_action_mask=supervised,
    )


def _supervised_action_mask(
    action_dim: int,
    fixed_dims: tuple[int, ...],
    *,
    device: torch.device,
) -> Tensor:
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if any(value >= action_dim for value in fixed_dims):
        raise ValueError("policy_fixed_action_dims exceed the action dimension")
    mask = torch.ones(action_dim, dtype=torch.bool, device=device)
    if fixed_dims:
        mask[list(fixed_dims)] = False
    if not bool(mask.any()):
        raise ValueError("at least one action dimension must remain supervised")
    return mask


def _weighted_action_mse(
    prediction: Tensor,
    target: Tensor,
    supervised_action_mask: Tensor,
    *,
    executed_steps: int = 0,
    executed_prefix_weight: float = 1.0,
) -> Tensor:
    """MSE over deployable dimensions, emphasizing the executed prefix."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("action loss tensors must share shape [B,H,A]")
    if supervised_action_mask.shape != (prediction.shape[-1],):
        raise ValueError("supervised action mask has the wrong shape")
    if not 0 <= int(executed_steps) <= prediction.shape[1]:
        raise ValueError("executed_steps exceeds the action-loss horizon")
    temporal = prediction.new_ones(prediction.shape[1])
    if executed_steps:
        temporal[: int(executed_steps)] = float(executed_prefix_weight)
    dimensions = supervised_action_mask.to(dtype=prediction.dtype)
    weights = temporal[:, None] * dimensions[None, :]
    denominator = prediction.shape[0] * weights.sum()
    return ((prediction - target).square() * weights[None]).sum() / denominator


def configure_stage(
    model: LatentWAM,
    flow: StatefulActionFlow,
    stage: M1StageConfig,
) -> torch.optim.Optimizer:
    """Apply the M1 freeze curriculum and return an auditable optimizer."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    flow.freeze_anchor()

    known_prefixes = (
        "world_model.",
        "vision_encoder.",
        "resampler.",
        "fusion.",
        "task_embedding.",
        "future_head.",
        "action_capacity_mlp.",
    )
    for name, parameter in model.named_parameters():
        if name.startswith("vision_encoder."):
            continue
        if name.startswith("world_model."):
            parameter.requires_grad_(stage.train_world_model)
        elif name.startswith("resampler."):
            parameter.requires_grad_(stage.train_visual_adapter)
        elif name.startswith("future_head."):
            parameter.requires_grad_(stage.train_future_head)
        elif name.startswith(("fusion.", "task_embedding.", "action_capacity_mlp.")):
            parameter.requires_grad_(stage.train_fusion)
        elif not name.startswith(known_prefixes):
            raise RuntimeError(f"unclassified M1 parameter in freeze plan: {name}")
    if stage.train_action_flow:
        for name, parameter in flow.named_parameters():
            if not name.startswith("anchor_prior."):
                parameter.requires_grad_(True)
    flow.freeze_anchor()

    regular: list[nn.Parameter] = []
    world: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (world if name.startswith("world_model.") else regular).append(parameter)
    regular.extend(
        parameter for parameter in flow.parameters() if parameter.requires_grad
    )
    groups: list[dict[str, Any]] = []
    if regular:
        groups.append(
            {"params": regular, "lr": stage.learning_rate, "role": "adapter_action"}
        )
    if world:
        groups.append(
            {
                "params": world,
                "lr": stage.world_learning_rate,
                "role": "legacy_world_low_lr",
            }
        )
    if not groups:
        raise ValueError("M1 stage selected no trainable parameters")
    optimizer = torch.optim.AdamW(groups, weight_decay=stage.weight_decay)
    # Frozen ImageNet features and frozen prior anchor are phase invariants.
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
        if any(
            parameter.requires_grad for parameter in model.vision_encoder.parameters()
        ):
            raise RuntimeError("M1 vision backbone must remain frozen")
    if any(parameter.requires_grad for parameter in flow.anchor_prior.parameters()):
        raise RuntimeError("M1 embedded action prior anchor must remain frozen")
    return optimizer


def train_m1_stage(
    model: LatentWAM,
    flow: StatefulActionFlow,
    batches: Iterable[Mapping[str, Tensor]],
    stage: M1StageConfig,
    *,
    device: torch.device,
    weights: M1LossWeights = M1LossWeights(),
    flow_objective: M1FlowObjectiveConfig = M1FlowObjectiveConfig(),
    causal_pair_batches: Iterable[Mapping[str, Tensor]] | None = None,
    causal_pair_weights: M1CausalPairWeights | None = None,
    state_causal_pair_batches: Iterable[Mapping[str, Tensor]] | None = None,
    state_causal_pair_weights: M1StateCausalPairWeights | None = None,
    seed: int = 0,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    """Train one finite curriculum stage and return compact evidence."""

    optimizer = configure_stage(model, flow, stage)
    model.train()
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
    if not stage.train_world_model:
        model.world_model.eval()
    flow.train()
    flow.anchor_prior.eval()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    iterator = iter(batches)
    if (causal_pair_batches is None) != (causal_pair_weights is None):
        raise ValueError(
            "causal_pair_batches and causal_pair_weights must be enabled together"
        )
    pair_iterator = None if causal_pair_batches is None else iter(causal_pair_batches)
    if (state_causal_pair_batches is None) != (state_causal_pair_weights is None):
        raise ValueError(
            "state_causal_pair_batches and state_causal_pair_weights must be "
            "enabled together"
        )
    state_pair_iterator = (
        None if state_causal_pair_batches is None else iter(state_causal_pair_batches)
    )
    values: list[dict[str, float]] = []
    required_batch_keys = m1_batch_required_keys(model, weights)
    if progress_callback is not None:
        progress_callback(0, stage.steps, math.nan)
    for step in range(stage.steps):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            try:
                raw_batch = next(iterator)
            except StopIteration as exc:
                raise ValueError("M1 training iterable yielded no batches") from exc
        batch = {
            name: value.to(device, non_blocking=True)
            if isinstance(value, Tensor)
            else value
            for name, value in raw_batch.items()
            if name in required_batch_keys
        }
        optimizer.zero_grad(set_to_none=True)
        loss = m1_batch_loss(
            model,
            flow,
            batch,
            weights=weights,
            flow_objective=flow_objective,
            generator=generator,
        )
        pair_loss: M1CausalPairLoss | None = None
        if causal_pair_batches is not None:
            assert causal_pair_weights is not None
            assert pair_iterator is not None
            try:
                raw_pair_batch = next(pair_iterator)
            except StopIteration:
                pair_iterator = iter(causal_pair_batches)
                try:
                    raw_pair_batch = next(pair_iterator)
                except StopIteration as exc:
                    raise ValueError(
                        "M1 causal-pair iterable yielded no batches"
                    ) from exc
            pair_batch = {
                name: value.to(device, non_blocking=True)
                if isinstance(value, Tensor)
                else value
                for name, value in raw_pair_batch.items()
                if name != "audit_sample_ids"
            }
            # RGB-pair supervision shapes the visual adapter/fusion path.  The
            # ordinary BC objective remains solely responsible for updating
            # the state-sensitive world features and non-anchor action flow;
            # otherwise the pair shortcut can rewire the flow into a visual-
            # only open-loop controller and erase proprioceptive dependence.
            with _causal_pair_visual_gradient_scope(model, flow):
                pair_loss = m1_causal_pair_loss(
                    model,
                    flow,
                    pair_batch,
                    weights=causal_pair_weights,
                    flow_objective=flow_objective,
                )
        state_pair_loss: M1StateCausalPairLoss | None = None
        if state_causal_pair_batches is not None:
            assert state_causal_pair_weights is not None
            assert state_pair_iterator is not None
            try:
                raw_state_pair_batch = next(state_pair_iterator)
            except StopIteration:
                state_pair_iterator = iter(state_causal_pair_batches)
                try:
                    raw_state_pair_batch = next(state_pair_iterator)
                except StopIteration as exc:
                    raise ValueError(
                        "M1 state causal-pair iterable yielded no batches"
                    ) from exc
            state_pair_batch = {
                name: value.to(device, non_blocking=True)
                if isinstance(value, Tensor)
                else value
                for name, value in raw_state_pair_batch.items()
                if name != "audit_sample_ids"
            }
            # State pairs identify the state path by holding RGB and past
            # actions fixed.  Representation deltas are already supplied by
            # the accepted world/fusion path, so this contrast calibrates only
            # the non-anchor flow gain and contributes no direct gradient to
            # either modality representation.
            with _state_causal_pair_gradient_scope(model):
                state_pair_loss = m1_state_causal_pair_loss(
                    model,
                    flow,
                    state_pair_batch,
                    weights=state_causal_pair_weights,
                    flow_objective=flow_objective,
                )
        total_loss = (
            loss.total
            + (loss.total.new_zeros(()) if pair_loss is None else pair_loss.total)
            + (
                loss.total.new_zeros(())
                if state_pair_loss is None
                else state_pair_loss.total
            )
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite M1 loss at {stage.name}/{step}")
        total_loss.backward()
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, stage.gradient_clip_norm
        )
        optimizer.step()
        values.append(
            {
                "total": float(loss.total.detach().cpu()),
                "optimized_total": float(total_loss.detach().cpu()),
                "flow_matching": float(loss.flow_matching.detach().cpu()),
                "action_endpoint": float(loss.action_endpoint.detach().cpu()),
                "future_visual_latent": float(loss.future_visual_latent.detach().cpu()),
                "future_state": float(loss.future_state.detach().cpu()),
                "warm_fraction": float(
                    loss.warm_start_mask.float().mean().detach().cpu()
                ),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "causal_pair_total": float(
                    0.0 if pair_loss is None else pair_loss.total.detach().cpu()
                ),
                "causal_pair_factual_endpoint": float(
                    0.0
                    if pair_loss is None
                    else pair_loss.factual_endpoint.detach().cpu()
                ),
                "causal_pair_action_delta": float(
                    0.0 if pair_loss is None else pair_loss.action_delta.detach().cpu()
                ),
                "causal_pair_delta_cosine": float(
                    0.0 if pair_loss is None else pair_loss.delta_cosine.detach().cpu()
                ),
                "causal_pair_delta_cosine_valid_fraction": float(
                    0.0
                    if pair_loss is None
                    else pair_loss.delta_cosine_valid_fraction.detach().cpu()
                ),
                "causal_pair_predicted_delta_rms": float(
                    0.0
                    if pair_loss is None
                    else pair_loss.predicted_delta_rms.detach().cpu()
                ),
                "causal_pair_target_delta_rms": float(
                    0.0
                    if pair_loss is None
                    else pair_loss.target_delta_rms.detach().cpu()
                ),
                "state_causal_pair_total": float(
                    0.0
                    if state_pair_loss is None
                    else state_pair_loss.total.detach().cpu()
                ),
                "state_causal_pair_factual_endpoint": float(
                    0.0
                    if state_pair_loss is None
                    else state_pair_loss.factual_endpoint.detach().cpu()
                ),
                "state_causal_pair_action_delta": float(
                    0.0
                    if state_pair_loss is None
                    else state_pair_loss.action_delta.detach().cpu()
                ),
                "state_causal_pair_delta_cosine": float(
                    0.0
                    if state_pair_loss is None
                    else state_pair_loss.delta_cosine.detach().cpu()
                ),
                "state_causal_pair_delta_cosine_valid_fraction": float(
                    0.0
                    if state_pair_loss is None
                    else state_pair_loss.delta_cosine_valid_fraction.detach().cpu()
                ),
                "state_causal_pair_predicted_delta_rms": float(
                    0.0
                    if state_pair_loss is None
                    else state_pair_loss.predicted_delta_rms.detach().cpu()
                ),
                "state_causal_pair_target_delta_rms": float(
                    0.0
                    if state_pair_loss is None
                    else state_pair_loss.target_delta_rms.detach().cpu()
                ),
                "state_causal_pair_step0_predicted_delta_rms": float(
                    0.0
                    if state_pair_loss is None
                    else state_pair_loss.step0_predicted_delta_rms.detach().cpu()
                ),
                "state_causal_pair_step0_target_delta_rms": float(
                    0.0
                    if state_pair_loss is None
                    else state_pair_loss.step0_target_delta_rms.detach().cpu()
                ),
            }
        )
        if progress_callback is not None:
            progress_callback(
                step + 1,
                stage.steps,
                values[-1]["optimized_total"],
            )
    flow.freeze_anchor()
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
    summary = {
        name: float(np.mean([item[name] for item in values])) for name in values[0]
    }
    summary.update(
        {
            "name": stage.name,
            "steps": stage.steps,
            "first_total": values[0]["total"],
            "last_total": values[-1]["total"],
            "first_optimized_total": values[0]["optimized_total"],
            "last_optimized_total": values[-1]["optimized_total"],
            "causal_pairs_enabled": causal_pair_batches is not None,
            "causal_pair_gradient_scope": (
                None if causal_pair_batches is None else "visual_adapter_fusion_only"
            ),
            "causal_pair_weights": (
                None if causal_pair_weights is None else asdict(causal_pair_weights)
            ),
            "state_causal_pairs_enabled": state_causal_pair_batches is not None,
            "state_causal_pair_gradient_scope": (
                None
                if state_causal_pair_batches is None
                else "non_anchor_flow_only_model_frozen"
            ),
            "state_causal_pair_weights": (
                None
                if state_causal_pair_weights is None
                else asdict(state_causal_pair_weights)
            ),
            "optimizer_groups": [
                {
                    "role": group.get("role"),
                    "lr": float(group["lr"]),
                    "parameters": sum(p.numel() for p in group["params"]),
                }
                for group in optimizer.param_groups
            ],
            "frozen_backbone": not any(
                parameter.requires_grad
                for parameter in (
                    ()
                    if model.vision_encoder is None
                    else model.vision_encoder.parameters()
                )
            ),
            "frozen_prior_anchor": not any(
                parameter.requires_grad for parameter in flow.anchor_prior.parameters()
            ),
            "flow_objective": asdict(flow_objective),
        }
    )
    return summary


@contextmanager
def _causal_pair_visual_gradient_scope(
    model: LatentWAM, flow: StatefulActionFlow
) -> Iterator[None]:
    """Keep pair gradients out of state/world and action-flow parameters."""

    parameters = tuple(model.world_model.parameters()) + tuple(flow.parameters())
    original = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, original, strict=True):
            parameter.requires_grad_(requires_grad)


@contextmanager
def _state_causal_pair_gradient_scope(model: LatentWAM) -> Iterator[None]:
    """Route state-identification gradients only into the non-anchor flow."""

    parameters = tuple(model.parameters())
    original = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, original, strict=True):
            parameter.requires_grad_(requires_grad)


@torch.inference_mode()
def causal_pair_action_metrics(
    model: LatentWAM,
    flow: StatefulActionFlow,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    flow_objective: M1FlowObjectiveConfig = M1FlowObjectiveConfig(),
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate cold-solver RGB counterfactual sensitivity by task index."""

    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError("max_batches must be positive when provided")
    model.eval()
    flow.eval()
    rows: list[dict[str, float | int]] = []
    observed_batches = 0
    locked_weights = M1CausalPairWeights()
    for batch_index, raw_batch in enumerate(batches):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = {
            name: value.to(device, non_blocking=True)
            if isinstance(value, Tensor)
            else value
            for name, value in raw_batch.items()
        }
        result = m1_causal_pair_loss(
            model,
            flow,
            batch,
            weights=locked_weights,
            flow_objective=flow_objective,
        )
        predicted = result.predicted_actions
        target = result.target_actions
        predicted = predicted[:, :, : flow_objective.execution_steps]
        target = target[:, :, : flow_objective.execution_steps]
        predicted_delta = predicted[:, 1] - predicted[:, 0]
        target_delta = target[:, 1] - target[:, 0]
        horizon = predicted.shape[2]
        temporal = predicted.new_full((horizon,), locked_weights.executed_prefix_weight)
        coordinate = (
            temporal[:, None]
            * result.supervised_action_mask.to(dtype=predicted.dtype)[None]
        )
        coordinate_sum = coordinate.sum()
        root_coordinate = coordinate.sqrt()
        task_indices = batch["task_index"][:, 0]
        for pair_index in range(predicted.shape[0]):
            factual = (
                (predicted[pair_index] - target[pair_index]).square() * coordinate[None]
            ).sum() / (2.0 * coordinate_sum)
            delta_mse = (
                (predicted_delta[pair_index] - target_delta[pair_index]).square()
                * coordinate
            ).sum() / coordinate_sum
            weighted_prediction = (
                predicted_delta[pair_index] * root_coordinate
            ).flatten()
            weighted_target = (target_delta[pair_index] * root_coordinate).flatten()
            prediction_norm = weighted_prediction.norm()
            target_norm = weighted_target.norm()
            cosine = (
                (weighted_prediction * weighted_target).sum()
                / (prediction_norm.clamp_min(1e-4) * target_norm.clamp_min(1e-8))
                if prediction_norm > 1e-4
                else predicted.new_zeros(())
            )
            active_prediction = predicted_delta[pair_index][
                ..., result.supervised_action_mask
            ]
            active_target = target_delta[pair_index][..., result.supervised_action_mask]
            prefix_prediction = active_prediction[: flow_objective.execution_steps]
            prefix_target = active_target[: flow_objective.execution_steps]
            nonzero_target = prefix_target.abs() > 1e-6
            sign_matches = (
                (torch.sign(prefix_prediction) == torch.sign(prefix_target))
                & nonzero_target
            ).sum()
            rows.append(
                {
                    "task_index": int(task_indices[pair_index].cpu()),
                    "factual_endpoint_mse": float(factual.cpu()),
                    "action_delta_mse": float(delta_mse.cpu()),
                    "delta_cosine": float(cosine.clamp(-1.0, 1.0).cpu()),
                    "predicted_delta_rms": float(
                        active_prediction.square().mean().sqrt().cpu()
                    ),
                    "target_delta_rms": float(
                        active_target.square().mean().sqrt().cpu()
                    ),
                    "prefix_sign_matches": int(sign_matches.cpu()),
                    "prefix_sign_targets": int(nonzero_target.sum().cpu()),
                }
            )
        observed_batches += 1
    if not rows:
        raise ValueError("causal-pair metric loader yielded no pairs")

    def aggregate(selected: list[dict[str, float | int]]) -> dict[str, Any]:
        prediction_rms = math.sqrt(
            float(np.mean([float(row["predicted_delta_rms"]) ** 2 for row in selected]))
        )
        target_rms = math.sqrt(
            float(np.mean([float(row["target_delta_rms"]) ** 2 for row in selected]))
        )
        sign_targets = sum(int(row["prefix_sign_targets"]) for row in selected)
        sign_matches = sum(int(row["prefix_sign_matches"]) for row in selected)
        return {
            "pairs": len(selected),
            "factual_endpoint_rmse": math.sqrt(
                float(np.mean([float(row["factual_endpoint_mse"]) for row in selected]))
            ),
            "action_delta_rmse": math.sqrt(
                float(np.mean([float(row["action_delta_mse"]) for row in selected]))
            ),
            "delta_cosine": float(
                np.mean([float(row["delta_cosine"]) for row in selected])
            ),
            "predicted_delta_rms": prediction_rms,
            "target_delta_rms": target_rms,
            "delta_norm_ratio": prediction_rms / max(target_rms, 1e-12),
            "executed_prefix_sign_agreement": (
                sign_matches / sign_targets if sign_targets else None
            ),
            "executed_prefix_nonzero_targets": sign_targets,
        }

    task_ids = sorted({int(row["task_index"]) for row in rows})
    return {
        "cold_start": True,
        "solver_steps": flow_objective.solver_steps,
        "solver": flow_objective.solver,
        "batches": observed_batches,
        **aggregate(rows),
        "by_task_index": {
            str(task_index): aggregate(
                [row for row in rows if int(row["task_index"]) == task_index]
            )
            for task_index in task_ids
        },
    }


@torch.inference_mode()
def state_causal_pair_action_metrics(
    model: LatentWAM,
    flow: StatefulActionFlow,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    flow_objective: M1FlowObjectiveConfig = M1FlowObjectiveConfig(),
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Recompute held-out execute-2 sensitivity on state-identifiable pairs."""

    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError("max_batches must be positive when provided")
    model.eval()
    flow.eval()
    rows: list[dict[str, float | int]] = []
    observed_batches = 0
    locked_weights = M1StateCausalPairWeights()
    for batch_index, raw_batch in enumerate(batches):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = {
            name: value.to(device, non_blocking=True)
            if isinstance(value, Tensor)
            else value
            for name, value in raw_batch.items()
            if name != "audit_sample_ids"
        }
        result = m1_state_causal_pair_loss(
            model,
            flow,
            batch,
            weights=locked_weights,
            flow_objective=flow_objective,
        )
        execute = flow_objective.execution_steps
        supervised = result.supervised_action_mask
        predicted = flow.normalize_actions(result.predicted_actions[:, :, :execute])[
            ..., supervised
        ]
        target = flow.normalize_actions(result.target_actions[:, :, :execute])[
            ..., supervised
        ]
        predicted_delta = predicted[:, 1] - predicted[:, 0]
        target_delta = target[:, 1] - target[:, 0]
        task_indices = batch["task_index"][:, 0]
        for pair_index in range(predicted.shape[0]):
            pair_prediction = predicted_delta[pair_index]
            pair_target = target_delta[pair_index]
            prediction_norm = pair_prediction.flatten().norm()
            target_norm = pair_target.flatten().norm()
            target_rms = pair_target.square().mean().sqrt()
            relative_scale = target_rms.clamp_min(locked_weights.target_rms_floor)
            relative_error = (pair_prediction - pair_target) / relative_scale
            cosine = (pair_prediction * pair_target).sum() / (
                prediction_norm.clamp_min(1e-3) * target_norm.clamp_min(1e-8)
            )
            nonzero_target = (
                pair_target[0].abs() >= M1_STATE_PAIR_MIN_STEP0_ACTION_DELTA
            )
            sign_matches = (
                (torch.sign(pair_prediction[0]) == torch.sign(pair_target[0]))
                & nonzero_target
            ).sum()
            rows.append(
                {
                    "task_index": int(task_indices[pair_index].cpu()),
                    "factual_endpoint_mse": float(
                        (predicted[pair_index] - target[pair_index])
                        .square()
                        .mean()
                        .cpu()
                    ),
                    "action_delta_mse": float(
                        (pair_prediction - pair_target).square().mean().cpu()
                    ),
                    "relative_delta_huber": float(
                        F.smooth_l1_loss(
                            relative_error,
                            torch.zeros_like(relative_error),
                            beta=1.0,
                        ).cpu()
                    ),
                    "step0_action_delta_mse": float(
                        (pair_prediction[0] - pair_target[0]).square().mean().cpu()
                    ),
                    "delta_cosine": float(cosine.clamp(-1.0, 1.0).cpu()),
                    "predicted_delta_rms": float(
                        pair_prediction.square().mean().sqrt().cpu()
                    ),
                    "target_delta_rms": float(target_rms.cpu()),
                    "step0_predicted_delta_rms": float(
                        pair_prediction[0].square().mean().sqrt().cpu()
                    ),
                    "step0_target_delta_rms": float(
                        pair_target[0].square().mean().sqrt().cpu()
                    ),
                    "step0_sign_matches": int(sign_matches.cpu()),
                    "step0_sign_targets": int(nonzero_target.sum().cpu()),
                }
            )
        observed_batches += 1
    if not rows:
        raise ValueError("state causal-pair metric loader yielded no pairs")

    def aggregate(selected: list[dict[str, float | int]]) -> dict[str, Any]:
        def root_mean_square(name: str) -> float:
            return math.sqrt(
                float(np.mean([float(row[name]) ** 2 for row in selected]))
            )

        predicted_rms = root_mean_square("predicted_delta_rms")
        target_rms = root_mean_square("target_delta_rms")
        step0_prediction_rms = root_mean_square("step0_predicted_delta_rms")
        step0_target_rms = root_mean_square("step0_target_delta_rms")
        sign_targets = sum(int(row["step0_sign_targets"]) for row in selected)
        sign_matches = sum(int(row["step0_sign_matches"]) for row in selected)
        return {
            "pairs": len(selected),
            "factual_endpoint_rmse": math.sqrt(
                float(np.mean([float(row["factual_endpoint_mse"]) for row in selected]))
            ),
            "action_delta_rmse": math.sqrt(
                float(np.mean([float(row["action_delta_mse"]) for row in selected]))
            ),
            "relative_delta_huber": float(
                np.mean([float(row["relative_delta_huber"]) for row in selected])
            ),
            "step0_action_delta_rmse": math.sqrt(
                float(
                    np.mean([float(row["step0_action_delta_mse"]) for row in selected])
                )
            ),
            "delta_cosine": float(
                np.mean([float(row["delta_cosine"]) for row in selected])
            ),
            "predicted_delta_rms": predicted_rms,
            "target_delta_rms": target_rms,
            "delta_norm_ratio": predicted_rms / max(target_rms, 1e-12),
            "step0_predicted_delta_rms": step0_prediction_rms,
            "step0_target_delta_rms": step0_target_rms,
            "step0_delta_norm_ratio": (
                step0_prediction_rms / max(step0_target_rms, 1e-12)
            ),
            "step0_sign_agreement": (
                sign_matches / sign_targets if sign_targets else None
            ),
            "step0_nonzero_targets": sign_targets,
        }

    task_ids = sorted({int(row["task_index"]) for row in rows})
    return {
        "cold_start": True,
        "solver_steps": flow_objective.solver_steps,
        "solver": flow_objective.solver,
        "execution_steps": flow_objective.execution_steps,
        "batches": observed_batches,
        **aggregate(rows),
        "by_task_index": {
            str(task_index): aggregate(
                [row for row in rows if int(row["task_index"]) == task_index]
            )
            for task_index in task_ids
        },
    }


@torch.inference_mode()
def action_chunk_rmse(
    model: LatentWAM,
    flow: StatefulActionFlow,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    solver_steps: int = 4,
    max_batches: int = 8,
    policy_fixed_action_dims: tuple[int, ...] = (3, 7),
) -> float:
    """Evaluate deployed cold-start integration on policy-controlled axes."""

    model.eval()
    flow.eval()
    squared = 0.0
    count = 0
    for batch_index, raw in enumerate(batches):
        if batch_index >= max_batches:
            break
        required = action_chunk_required_keys(model)
        batch = {
            name: value.to(device, non_blocking=True)
            if isinstance(value, Tensor)
            else value
            for name, value in raw.items()
            if name in required
        }
        encoding = model.encode(
            batch["states"] if model.config.use_state else None,
            batch["past_actions"] if model.config.use_state else None,
            batch["state_valid_mask"].bool() if model.config.use_state else None,
            batch["images"] if model.config.use_vision else None,
            batch["task_index"].long(),
        )
        predicted = flow.generate(
            encoding.planning_features,
            solver_steps=solver_steps,
            solver="euler",
        )
        target = batch["action_targets"]
        supervised = _supervised_action_mask(
            target.shape[-1],
            tuple(int(value) for value in policy_fixed_action_dims),
            device=target.device,
        )
        error = (predicted - target)[..., supervised]
        squared += float(error.square().sum().cpu())
        count += int(error.numel())
    if count == 0:
        raise ValueError("validation loader yielded no batches")
    return math.sqrt(squared / count)


__all__ = [
    "M1BatchLoss",
    "M1CausalPairLoss",
    "M1CausalPairWeights",
    "M1FlowObjectiveConfig",
    "M1LossWeights",
    "M1StageConfig",
    "M1StateCausalPairLoss",
    "M1StateCausalPairWeights",
    "M1_STATE_PAIR_MIN_STEP0_ACTION_DELTA",
    "action_chunk_required_keys",
    "action_chunk_rmse",
    "causal_pair_action_metrics",
    "configure_stage",
    "m1_batch_loss",
    "m1_batch_required_keys",
    "m1_causal_pair_loss",
    "m1_state_causal_pair_loss",
    "seed_everything",
    "state_causal_pair_action_metrics",
    "train_m1_stage",
]
