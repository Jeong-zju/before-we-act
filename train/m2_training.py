"""Losses and compute pipeline for RoboFactory Phase M2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.wam_multimodal import BlockCausalWAM, VisionEncoderOutput


@dataclass(frozen=True)
class M2LossWeights:
    flow_matching: float = 1.0
    action_endpoint: float = 2.0
    action_smoothness: float = 0.05
    future_state: float = 1.0
    future_visual_latent: float = 1.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not 0.0 <= float(value) < float("inf"):
                raise ValueError(f"M2 loss weight {name} must be finite and non-negative")


@dataclass(frozen=True)
class M2BatchLoss:
    total: Tensor
    flow_matching: Tensor
    action_endpoint: Tensor
    action_endpoint_executed_prefix: Tensor
    action_endpoint_incomplete_horizon: Tensor
    action_smoothness: Tensor
    future_state: Tensor
    future_visual_latent: Tensor
    incomplete_horizon_fraction: Tensor
    active_agent_fraction: Tensor
    past_action_history_dropped_fraction: Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            name: float(value.detach().float().cpu())
            for name, value in self.__dict__.items()
        }


class RGBStatisticsVisionEncoder(nn.Module):
    """Deterministic CPU smoke encoder; production runs reject this backend."""

    family = "rgb_statistics_smoke"

    def __init__(self, output_dim: int = 64, grid: int = 4) -> None:
        super().__init__()
        if output_dim <= 0 or grid <= 0:
            raise ValueError("smoke visual dimensions must be positive")
        self.output_dim = int(output_dim)
        self.grid = int(grid)
        input_dim = 3 * grid * grid
        row = torch.arange(input_dim, dtype=torch.float32)[:, None] + 1.0
        column = torch.arange(output_dim, dtype=torch.float32)[None, :] + 1.0
        projection = torch.sin(row * column * 0.017) + torch.cos(row * column * 0.013)
        projection = F.normalize(projection, dim=0)
        self.register_buffer("projection", projection, persistent=True)
        spatial_row = torch.arange(3, dtype=torch.float32)[:, None] + 1.0
        spatial_projection = (
            torch.sin(spatial_row * column * 0.019)
            + torch.cos(spatial_row * column * 0.011)
        )
        self.register_buffer(
            "spatial_projection",
            F.normalize(spatial_projection, dim=0),
            persistent=True,
        )

    def forward(self, images: Tensor) -> VisionEncoderOutput:
        if images.ndim < 4 or images.shape[-3] != 3:
            raise ValueError("images must have shape [...,3,H,W]")
        leading = images.shape[:-3]
        flattened = images.reshape(-1, *images.shape[-3:]).float().div(255.0)
        pooled = F.adaptive_avg_pool2d(flattened, (self.grid, self.grid)).flatten(1)
        features = F.layer_norm(pooled @ self.projection, (self.output_dim,))
        features = features.reshape(*leading, self.output_dim)
        return VisionEncoderOutput(
            spatial_tokens=features.unsqueeze(-2),
            pooled_latent=features,
        )

    def forward_pooled(self, images: Tensor) -> Tensor:
        return self(images).pooled_latent

    def forward_spatial_grid(
        self,
        images: Tensor,
        *,
        grid_height: int,
        grid_width: int,
    ) -> VisionEncoderOutput:
        if images.ndim < 4 or images.shape[-3] != 3:
            raise ValueError("images must have shape [...,3,H,W]")
        if grid_height <= 0 or grid_width <= 0:
            raise ValueError("spatial grid dimensions must be positive")
        leading = images.shape[:-3]
        flattened = images.reshape(-1, *images.shape[-3:]).float().div(255.0)
        cells = F.adaptive_avg_pool2d(
            flattened,
            (grid_height, grid_width),
        ).permute(0, 2, 3, 1)
        spatial = F.layer_norm(
            cells.reshape(flattened.shape[0], -1, 3) @ self.spatial_projection,
            (self.output_dim,),
        )
        pooled = self.forward_pooled(images)
        return VisionEncoderOutput(
            spatial_tokens=spatial.reshape(
                *leading,
                grid_height * grid_width,
                self.output_dim,
            ),
            pooled_latent=pooled,
        )


def encode_pooled_vision(encoder: nn.Module, images: Tensor) -> Tensor:
    """Use an encoder's memory-bounded pooled path when it provides one."""

    method = getattr(encoder, "forward_pooled", None)
    if callable(method):
        pooled = method(images)
    else:
        pooled = encoder(images).pooled_latent
    if not isinstance(pooled, Tensor) or pooled.shape[:-1] != images.shape[:-3]:
        raise ValueError("vision encoder pooled output has the wrong leading shape")
    return pooled


def encode_m2_vision(
    encoder: nn.Module,
    batch: Mapping[str, Tensor],
    *,
    spatial_grid_height: int = 1,
    spatial_grid_width: int = 1,
) -> tuple[Tensor, Tensor]:
    """Encode current and future RGB in one frozen, GPU-saturating call."""

    current = batch["images"]
    future = batch["future_images"]
    if current.ndim != 6 or future.ndim != 6:
        raise ValueError("M2 RGB tensors must be [B,T,Cam,3,H,W]")
    if current.shape[0] != future.shape[0] or current.shape[2:] != future.shape[2:]:
        raise ValueError("current/future RGB batch contracts differ")
    current_valid = batch["image_valid_mask"].bool()
    future_valid = batch["future_visual_valid_mask"].bool()
    if current_valid.shape != current.shape[:3]:
        raise ValueError("current RGB validity must be [B,T,Cam]")
    if future_valid.shape != future.shape[:3]:
        raise ValueError("future RGB validity must be [B,F,Cam]")
    combined = torch.cat((current, future), dim=1)
    valid = torch.cat((current_valid, future_valid), dim=1)
    flattened = combined.flatten(0, 2)
    flat_valid = valid.flatten()
    if not bool(flat_valid.any()):
        raise ValueError("M2 RGB batch contains no valid camera frame")
    grid_tokens = int(spatial_grid_height) * int(spatial_grid_width)
    if spatial_grid_height <= 0 or spatial_grid_width <= 0:
        raise ValueError("M2 spatial grid dimensions must be positive")
    if grid_tokens == 1:
        with torch.no_grad():
            selected_pooled = encode_pooled_vision(encoder, flattened[flat_valid])
        selected_spatial = selected_pooled.unsqueeze(-2)
    else:
        method = getattr(encoder, "forward_spatial_grid", None)
        if not callable(method):
            raise TypeError(
                "M2 spatial visual tokens require encoder.forward_spatial_grid"
            )
        with torch.no_grad():
            output = method(
                flattened[flat_valid],
                grid_height=spatial_grid_height,
                grid_width=spatial_grid_width,
            )
        if not isinstance(output, VisionEncoderOutput):
            raise TypeError("M2 spatial encoder returned the wrong output type")
        selected_spatial = output.spatial_tokens
        selected_pooled = output.pooled_latent
        if selected_spatial.shape[-2:] != (
            grid_tokens,
            selected_pooled.shape[-1],
        ):
            raise ValueError("M2 spatial encoder returned the wrong grid shape")
    pooled = selected_pooled.new_zeros(
        (*combined.shape[:3], int(selected_pooled.shape[-1]))
    )
    pooled.view(-1, pooled.shape[-1])[flat_valid] = selected_pooled.reshape(
        -1,
        selected_pooled.shape[-1],
    )
    spatial = selected_spatial.new_zeros(
        (
            *combined.shape[:3],
            grid_tokens,
            int(selected_spatial.shape[-1]),
        )
    )
    spatial.view(-1, grid_tokens, spatial.shape[-1])[flat_valid] = (
        selected_spatial.reshape(-1, grid_tokens, selected_spatial.shape[-1])
    )
    current_count = current.shape[1]
    current_features = spatial[:, :current_count].detach()
    if grid_tokens == 1:
        current_features = current_features.squeeze(-2)
    return current_features, pooled[:, current_count:].detach()


def m2_model_context(
    batch: Mapping[str, Tensor],
    visual_features: Tensor,
) -> dict[str, Tensor]:
    image_valid = batch["image_valid_mask"].bool()
    if image_valid.ndim != 3:
        raise ValueError("M2 image validity mask must be [B,T,Cam]")
    return {
        "states": batch["states"],
        "state_valid_mask": batch["state_valid_mask"].bool(),
        "state_dimension_mask": batch["state_dimension_mask"].bool(),
        "past_actions": batch["past_actions"],
        "past_action_valid_mask": batch["past_action_valid_mask"].bool(),
        "action_dimension_mask": batch["action_dimension_mask"].bool(),
        "visual_features": visual_features,
        "image_valid_mask": image_valid,
        "camera_agent_index": batch["camera_agent_index"].long(),
        "action_horizon_mask": batch["action_horizon_mask"].bool(),
        "task_text_tokens": batch["task_text_tokens"].long(),
        "task_index": batch["task_index"].long(),
        "embodiment_index": batch["embodiment_index"].long(),
    }


def m2_batch_loss(
    model: nn.Module,
    batch: Mapping[str, Tensor],
    *,
    visual_features: Tensor,
    future_visual_targets: Tensor,
    weights: M2LossWeights,
    warm_start_probability: float,
    warm_start_noise_std: float,
    execution_steps: int,
    executed_prefix_weight: float,
    active_agent_loss_weight: float = 1.0,
    active_agent_delta_threshold: float = 0.005,
    past_action_history_dropout_probability: float = 0.0,
    state_history_noise_std: float = 0.0,
    past_action_history_noise_std: float = 0.0,
    solver_steps: int,
    solver: str,
    normalized_action_clip: float,
    generator: torch.Generator | None = None,
) -> M2BatchLoss:
    if not 0.0 <= warm_start_probability <= 1.0 or warm_start_noise_std < 0.0:
        raise ValueError("invalid M2 warm-start training controls")
    if solver_steps <= 0 or normalized_action_clip <= 0.0:
        raise ValueError("invalid M2 deployed solver controls")
    if executed_prefix_weight < 1.0:
        raise ValueError("M2 executed-prefix weight must be at least one")
    if active_agent_loss_weight < 1.0 or active_agent_delta_threshold < 0.0:
        raise ValueError("invalid M2 active-agent loss controls")
    if not 0.0 <= past_action_history_dropout_probability <= 1.0:
        raise ValueError("invalid M2 past-action history dropout probability")
    if state_history_noise_std < 0.0 or past_action_history_noise_std < 0.0:
        raise ValueError("invalid M2 history noise controls")
    if solver not in {"euler", "heun"}:
        raise ValueError("M2 deployed solver must be euler or heun")
    core = model.module if hasattr(model, "module") else model
    if not isinstance(core, BlockCausalWAM):
        raise TypeError("M2 loss requires BlockCausalWAM or its DDP wrapper")
    targets = batch["action_targets"]
    target_valid = batch["action_target_valid_mask"].bool()
    if target_valid.shape != targets.shape[:2]:
        raise ValueError("M2 action target validity mask has the wrong shape")
    task_horizon_valid = batch["action_horizon_mask"].bool()
    if (
        task_horizon_valid.shape != target_valid.shape
        or bool((target_valid & ~task_horizon_valid).any())
    ):
        raise ValueError("M2 target validity exceeds the task action horizon")
    action_mask = (
        batch["action_dimension_mask"].bool()[:, None, :]
        & target_valid[:, :, None]
    )
    if targets.shape[1:] != (
        core.config.action_horizon,
        core.config.max_action_dim,
    ):
        raise ValueError("M2 action targets disagree with the model")
    if not 0 < execution_steps < targets.shape[1]:
        raise ValueError("M2 execution_steps must lie inside the action horizon")
    initial = torch.zeros_like(targets)
    batch_size = targets.shape[0]
    random_kwargs: dict[str, Any] = {
        "device": targets.device,
        "generator": generator,
    }
    warm = bool(torch.rand((), **random_kwargs) < warm_start_probability)
    if warm:
        shifted = shift_action_chunk(
            targets,
            execution_steps=execution_steps,
            action_horizon_mask=task_horizon_valid,
        )
        if warm_start_noise_std:
            shifted = shifted + torch.randn(
                shifted.shape,
                dtype=shifted.dtype,
                **random_kwargs,
            ) * warm_start_noise_std
        initial.copy_(shifted)
    tau = torch.rand((batch_size,), dtype=targets.dtype, **random_kwargs)
    interpolated = initial + tau[:, None, None] * (targets - initial)
    context = m2_model_context(batch, visual_features)
    history_drop = torch.rand(
        (batch_size,),
        dtype=targets.dtype,
        **random_kwargs,
    ).lt(past_action_history_dropout_probability)
    if bool(history_drop.any()):
        context["past_actions"] = context["past_actions"].masked_fill(
            history_drop[:, None, None],
            0.0,
        )
        context["past_action_valid_mask"] = (
            context["past_action_valid_mask"]
            & ~history_drop[:, None]
        )
    if state_history_noise_std:
        state_noise = torch.randn(
            context["states"].shape,
            dtype=context["states"].dtype,
            **random_kwargs,
        ) * state_history_noise_std
        state_noise_mask = (
            context["state_valid_mask"][:, :, None]
            & context["state_dimension_mask"][:, None, :]
        )
        context["states"] = context["states"] + state_noise.masked_fill(
            ~state_noise_mask,
            0.0,
        )
    if past_action_history_noise_std:
        action_noise = torch.randn(
            context["past_actions"].shape,
            dtype=context["past_actions"].dtype,
            **random_kwargs,
        ) * past_action_history_noise_std
        action_noise_mask = (
            context["past_action_valid_mask"][:, :, None]
            & context["action_dimension_mask"][:, None, :]
        )
        context["past_actions"] = context["past_actions"] + action_noise.masked_fill(
            ~action_noise_mask,
            0.0,
        )
    past_action_history_dropped_fraction = history_drop.float().mean()
    agent_weights, active_agent_fraction = _active_agent_weights(
        targets,
        target_valid=target_valid,
        action_dimension_mask=batch["action_dimension_mask"].bool(),
        past_actions=batch["past_actions"],
        past_action_valid_mask=batch["past_action_valid_mask"].bool(),
        max_agents=core.config.max_agents,
        active_weight=active_agent_loss_weight,
        delta_threshold=active_agent_delta_threshold,
    )
    per_agent_action_dim = core.config.max_action_dim // core.config.max_agents
    action_agent_weights = agent_weights.repeat_interleave(
        per_agent_action_dim,
        dim=-1,
    )
    needs_future = bool(weights.future_state or weights.future_visual_latent)
    output = model(
        **context,
        action_inputs=interpolated,
        flow_time=tau,
        initial_actions=initial if warm else None,
        future_action_condition=targets,
        include_future=needs_future,
    )
    target_velocity = targets - initial
    incomplete_horizon = ~target_valid.all(dim=1)
    incomplete_horizon_fraction = incomplete_horizon.float().mean()
    action_weights = targets.new_ones((targets.shape[1],))
    action_weights[:execution_steps] = executed_prefix_weight
    flow_matching = _weighted_masked_mean(
        (output.action_velocity - target_velocity).square(),
        action_mask,
        action_weights[None, :, None] * action_agent_weights,
    )
    needs_endpoint = bool(weights.action_endpoint or weights.action_smoothness)
    if needs_endpoint:
        # This endpoint is the deployed computation, not a teacher-forced local
        # extrapolation from x_tau. It removes the x_tau/tau shortcut and forces
        # generation of a valid chunk from the actual cold/warm start.
        endpoint = core.integrate_actions(
            context,
            initial_actions=initial if warm else None,
            solver_steps=solver_steps,
            solver=solver,
            normalized_clip=normalized_action_clip,
            forward_model=model,
        )
        action_endpoint = _weighted_masked_mean(
            (endpoint - targets).square(),
            action_mask,
            action_weights[None, :, None] * action_agent_weights,
        )
        executed_mask = action_mask.clone()
        executed_mask[:, execution_steps:] = False
        action_endpoint_executed_prefix = _masked_mean(
            (endpoint - targets).square() * action_agent_weights,
            executed_mask,
        )
        incomplete_mask = action_mask & incomplete_horizon[:, None, None]
        action_endpoint_incomplete_horizon = _masked_mean(
            (endpoint - targets).square() * action_agent_weights,
            incomplete_mask,
        )
        smooth_valid = target_valid[:, 1:] & target_valid[:, :-1]
        smooth_mask = (
            batch["action_dimension_mask"].bool()[:, None, :]
            & smooth_valid[:, :, None]
        )
        smooth_weights = targets.new_ones((targets.shape[1] - 1,))
        smooth_weights[: max(execution_steps - 1, 0)] = executed_prefix_weight
        action_smoothness = _weighted_masked_mean(
            (
                (endpoint[:, 1:] - endpoint[:, :-1])
                - (targets[:, 1:] - targets[:, :-1])
            ).square(),
            smooth_mask,
            smooth_weights[None, :, None]
            * torch.maximum(agent_weights[:, 1:], agent_weights[:, :-1])
            .repeat_interleave(per_agent_action_dim, dim=-1),
        )
    else:
        action_endpoint = targets.new_zeros(())
        action_endpoint_executed_prefix = targets.new_zeros(())
        action_endpoint_incomplete_horizon = targets.new_zeros(())
        action_smoothness = targets.new_zeros(())
    if needs_future:
        if output.future_states is None or output.future_visual_latents is None:
            raise RuntimeError("M2 joint training unexpectedly skipped the future branch")
        future_state_valid = batch["future_state_valid_mask"].bool()
        if future_state_valid.shape != output.future_states.shape[:2]:
            raise ValueError("M2 future-state validity mask has the wrong shape")
        state_mask = (
            batch["state_dimension_mask"].bool()[:, None, :]
            & future_state_valid[:, :, None]
        )
        if core.config.max_state_dim % core.config.max_agents:
            raise ValueError("M2 state width must contain complete agent slots")
        per_agent_state_dim = core.config.max_state_dim // core.config.max_agents
        state_agent_weights = agent_weights.repeat_interleave(
            per_agent_state_dim,
            dim=-1,
        )
        future_state = _masked_mean(
            (output.future_states - batch["future_states"]).square()
            * state_agent_weights,
            state_mask,
        )
        predicted_visual = F.normalize(output.future_visual_latents.float(), dim=-1)
        target_visual = F.normalize(future_visual_targets.float(), dim=-1)
        if predicted_visual.shape != target_visual.shape:
            raise ValueError(
                "M2 predicted/target future visual tensors must be [B,F,Cam,D]"
            )
        visual_error = (predicted_visual - target_visual).square().mean(dim=-1)
        novelty = batch["future_image_novelty_mask"].bool()
        visual_valid = batch["future_visual_valid_mask"].bool()
        if visual_valid.shape != novelty.shape:
            raise ValueError("M2 future-visual validity mask has the wrong shape")
        # Static frames remain weak supervision; novel frames carry full weight.
        visual_weights = torch.where(novelty, 1.0, 0.25).to(
            visual_error
        )
        future_visual = _weighted_masked_mean(
            visual_error,
            visual_valid,
            visual_weights,
        )
    else:
        future_state = targets.new_zeros(())
        future_visual = targets.new_zeros(())
    total = (
        weights.flow_matching * flow_matching
        + weights.action_endpoint * action_endpoint
        + weights.action_smoothness * action_smoothness
        + weights.future_state * future_state
        + weights.future_visual_latent * future_visual
    )
    return M2BatchLoss(
        total=total,
        flow_matching=flow_matching,
        action_endpoint=action_endpoint,
        action_endpoint_executed_prefix=action_endpoint_executed_prefix,
        action_endpoint_incomplete_horizon=action_endpoint_incomplete_horizon,
        action_smoothness=action_smoothness,
        future_state=future_state,
        future_visual_latent=future_visual,
        incomplete_horizon_fraction=incomplete_horizon_fraction,
        active_agent_fraction=active_agent_fraction,
        past_action_history_dropped_fraction=(
            past_action_history_dropped_fraction
        ),
    )


def _active_agent_weights(
    targets: Tensor,
    *,
    target_valid: Tensor,
    action_dimension_mask: Tensor,
    past_actions: Tensor,
    past_action_valid_mask: Tensor,
    max_agents: int,
    active_weight: float,
    delta_threshold: float,
) -> tuple[Tensor, Tensor]:
    """Return per-step agent weights with a constant mean over valid agents.

    RoboFactory Panda actions use an 8D slot per agent.  LongPipelineDelivery
    normally moves one of four agents at a time, so a flat scalar mean lets
    three idle agents dominate the training signal.  Activity weighting keeps
    every sample/step at unit mean weight while reallocating scalar weight from
    idle to moving agent slots.
    """

    if targets.ndim != 3 or target_valid.shape != targets.shape[:2]:
        raise ValueError("M2 targets/validity have incompatible shapes")
    if max_agents <= 0 or targets.shape[-1] % max_agents:
        raise ValueError("M2 action width must contain complete agent slots")
    per_agent_dim = targets.shape[-1] // max_agents
    if action_dimension_mask.shape != (targets.shape[0], targets.shape[-1]):
        raise ValueError("M2 action dimension mask has the wrong shape")
    if past_actions.shape != (
        targets.shape[0],
        past_action_valid_mask.shape[1],
        targets.shape[-1],
    ):
        raise ValueError("M2 past actions have the wrong shape")
    active, step_valid = _agent_activity_mask(
        targets,
        target_valid=target_valid,
        action_dimension_mask=action_dimension_mask,
        past_actions=past_actions,
        past_action_valid_mask=past_action_valid_mask,
        max_agents=max_agents,
        delta_threshold=delta_threshold,
    )
    raw = torch.where(
        active,
        targets.new_full((), float(active_weight)),
        targets.new_ones(()),
    )
    raw = raw * step_valid.to(raw)
    valid_count = step_valid.sum(dim=-1, keepdim=True).clamp_min(1)
    scale = valid_count.to(raw) / raw.sum(dim=-1, keepdim=True).clamp_min(1.0)
    normalized = raw * scale
    active_count = active.sum()
    valid_agent_steps = step_valid.sum()
    fraction = (
        active_count.to(targets) / valid_agent_steps.to(targets).clamp_min(1.0)
    )
    return normalized, fraction


def _agent_activity_mask(
    targets: Tensor,
    *,
    target_valid: Tensor,
    action_dimension_mask: Tensor,
    past_actions: Tensor,
    past_action_valid_mask: Tensor,
    max_agents: int,
    delta_threshold: float,
) -> tuple[Tensor, Tensor]:
    if targets.ndim != 3 or target_valid.shape != targets.shape[:2]:
        raise ValueError("M2 targets/validity have incompatible shapes")
    if max_agents <= 0 or targets.shape[-1] % max_agents:
        raise ValueError("M2 action width must contain complete agent slots")
    per_agent_dim = targets.shape[-1] // max_agents
    reshaped = targets.reshape(
        targets.shape[0],
        targets.shape[1],
        max_agents,
        per_agent_dim,
    )
    delta = torch.zeros_like(reshaped)
    if targets.shape[1] > 1:
        delta[:, 1:] = reshaped[:, 1:] - reshaped[:, :-1]
    has_past = past_action_valid_mask.any(dim=1)
    if bool(has_past.any()):
        history_indices = torch.arange(
            past_action_valid_mask.shape[1],
            device=targets.device,
        )[None].expand_as(past_action_valid_mask)
        last_indices = history_indices.masked_fill(
            ~past_action_valid_mask,
            -1,
        ).amax(dim=1).clamp_min(0)
        previous = past_actions[
            torch.arange(targets.shape[0], device=targets.device),
            last_indices,
        ].reshape(targets.shape[0], max_agents, per_agent_dim)
        delta[:, 0] = torch.where(
            has_past[:, None, None],
            reshaped[:, 0] - previous,
            delta[:, 0],
        )
    magnitude = delta.abs().amax(dim=-1)
    # Mark both sides of a transition so the first step of an action chunk does
    # not become idle merely because no past action exists at episode reset.
    adjacent = magnitude.clone()
    if targets.shape[1] > 1:
        adjacent[:, :-1] = torch.maximum(adjacent[:, :-1], magnitude[:, 1:])
    agent_valid = action_dimension_mask.reshape(
        targets.shape[0],
        max_agents,
        per_agent_dim,
    ).any(dim=-1)
    step_valid = target_valid[:, :, None] & agent_valid[:, None, :]
    return adjacent.gt(delta_threshold) & step_valid, step_valid


def shift_action_chunk(
    actions: Tensor,
    *,
    execution_steps: int,
    action_horizon_mask: Tensor | None = None,
) -> Tensor:
    if actions.ndim != 3 or not 0 < execution_steps < actions.shape[1]:
        raise ValueError("execution_steps must lie inside the action chunk")
    batch_size, horizon, action_dim = actions.shape
    if action_horizon_mask is None:
        valid = torch.ones(
            (batch_size, horizon),
            dtype=torch.bool,
            device=actions.device,
        )
    else:
        valid = action_horizon_mask
        if valid.dtype != torch.bool or valid.shape != (batch_size, horizon):
            raise ValueError(
                "action_horizon_mask must be boolean with shape [B,H]"
            )
        if valid.device != actions.device:
            raise TypeError("actions and action_horizon_mask must share a device")
    lengths = valid.sum(dim=1)
    expected = torch.arange(horizon, device=actions.device)[None] < lengths[:, None]
    if not torch.equal(valid, expected):
        raise ValueError("action_horizon_mask must be a contiguous valid prefix")
    if bool(lengths.le(execution_steps).any()):
        raise ValueError("every task horizon must exceed execution_steps")
    positions = torch.arange(horizon, device=actions.device)[None]
    sources = torch.minimum(
        positions + execution_steps,
        lengths[:, None] - 1,
    )
    shifted = actions.gather(
        1,
        sources[:, :, None].expand(batch_size, horizon, action_dim),
    )
    return shifted.masked_fill(~valid[:, :, None], 0.0)


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    """Mean valid elements per sample, then mean non-empty samples.

    This hierarchical reduction prevents embodiments with more state/action
    dimensions, or samples with longer valid horizons, from receiving a larger
    optimization weight merely because they contain more scalar targets.
    """

    expanded = mask.expand_as(value)
    axes = tuple(range(1, value.ndim))
    denominator = expanded.sum(dim=axes)
    numerator = (value * expanded.to(value)).sum(dim=axes)
    present = denominator.gt(0)
    if not bool(present.any()):
        return value.sum() * 0.0
    return (numerator[present] / denominator[present].to(value)).mean()


def _weighted_masked_mean(value: Tensor, mask: Tensor, weight: Tensor) -> Tensor:
    expanded_mask = mask.expand_as(value)
    expanded_weight = weight.expand_as(value) * expanded_mask.to(value)
    axes = tuple(range(1, value.ndim))
    denominator = expanded_weight.sum(dim=axes)
    numerator = (value * expanded_weight).sum(dim=axes)
    present = denominator.gt(0)
    if not bool(present.any()):
        return value.sum() * 0.0
    return (numerator[present] / denominator[present]).mean()


class DevicePrefetcher:
    """Overlap pinned-memory H2D transfer with GPU model execution."""

    def __init__(self, source: Iterator[Mapping[str, Tensor]], device: torch.device) -> None:
        self.source = source
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.next_batch: dict[str, Tensor] | None = None
        self._preload()

    def _preload(self) -> None:
        try:
            raw = next(self.source)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = _batch_to_device(raw, self.device)
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = _batch_to_device(raw, self.device)

    def next(self) -> dict[str, Tensor] | None:
        batch = self.next_batch
        if batch is None:
            return None
        if self.stream is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_stream(self.stream)
            for value in batch.values():
                value.record_stream(current)
        self._preload()
        return batch


def _batch_to_device(
    batch: Mapping[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
    }


__all__ = [
    "DevicePrefetcher",
    "M2BatchLoss",
    "M2LossWeights",
    "RGBStatisticsVisionEncoder",
    "encode_m2_vision",
    "encode_pooled_vision",
    "m2_batch_loss",
    "m2_model_context",
    "shift_action_chunk",
]
