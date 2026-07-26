"""Block-causal multimodal Transformer used by RoboFactory Phase M2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BlockCausalWAMConfig:
    max_state_dim: int
    max_action_dim: int
    num_tasks: int
    max_agents: int = 4
    max_cameras: int = 1
    history_steps: int = 16
    visual_history_steps: int = 4
    visual_grid_height: int = 1
    visual_grid_width: int = 1
    action_horizon: int = 16
    future_visual_horizons: tuple[int, ...] = (1, 4, 8, 16)
    visual_feature_dim: int = 1024
    text_vocab_size: int = 257
    max_text_tokens: int = 96
    d_model: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    ffn_dim: int = 4096
    text_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            "max_state_dim",
            "max_action_dim",
            "num_tasks",
            "max_agents",
            "max_cameras",
            "history_steps",
            "visual_history_steps",
            "visual_grid_height",
            "visual_grid_width",
            "action_horizon",
            "visual_feature_dim",
            "text_vocab_size",
            "max_text_tokens",
            "d_model",
            "num_layers",
            "num_heads",
            "ffn_dim",
            "text_layers",
        )
        for name in integer_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.visual_history_steps > self.history_steps:
            raise ValueError("visual history cannot exceed state history")
        horizons = tuple(int(value) for value in self.future_visual_horizons)
        if not horizons or tuple(sorted(set(horizons))) != horizons:
            raise ValueError("future visual horizons must be increasing and unique")
        if horizons[0] <= 0 or horizons[-1] > self.action_horizon:
            raise ValueError("future visual horizons must lie inside action_horizon")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        object.__setattr__(self, "future_visual_horizons", horizons)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["future_visual_horizons"] = list(self.future_visual_horizons)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BlockCausalWAMConfig":
        values = dict(payload)
        if "future_visual_horizons" in values:
            values["future_visual_horizons"] = tuple(values["future_visual_horizons"])
        return cls(**values)


@dataclass(frozen=True)
class BlockCausalWAMOutput:
    action_velocity: Tensor
    future_states: Tensor | None
    future_visual_latents: Tensor | None


def build_block_causal_attention_mask(config: BlockCausalWAMConfig) -> Tensor:
    """Return a boolean mask where True entries are forbidden attention edges.

    History token queries see the current and earlier time blocks only.  Action
    queries see all history plus their causal action prefix.  Future queries see
    history, all generated action tokens, and their causal future prefix.  Thus
    an action output cannot depend on a future-query embedding or future target.
    """

    grid_tokens = config.visual_grid_height * config.visual_grid_width
    if grid_tokens == 1:
        modalities = 3 + config.max_cameras
        history_times = torch.arange(config.history_steps).repeat_interleave(
            modalities
        )
    else:
        base_times = torch.arange(config.history_steps).repeat_interleave(3)
        visual_times = torch.arange(
            config.history_steps - config.visual_history_steps,
            config.history_steps,
        ).repeat_interleave(config.max_cameras * grid_tokens)
        history_times = torch.cat((base_times, visual_times))
    history_tokens = int(history_times.numel())
    action_tokens = config.action_horizon
    future_tokens = config.action_horizon
    total = history_tokens + action_tokens + future_tokens
    allowed = torch.zeros((total, total), dtype=torch.bool)
    allowed[:history_tokens, :history_tokens] = (
        history_times[None, :] <= history_times[:, None]
    )
    action_start = history_tokens
    future_start = history_tokens + action_tokens
    for offset in range(action_tokens):
        query = action_start + offset
        allowed[query, :history_tokens] = True
        allowed[query, action_start : query + 1] = True
    for offset in range(future_tokens):
        query = future_start + offset
        allowed[query, :future_start] = True
        allowed[query, future_start : query + 1] = True
    return ~allowed


class UTF8TaskEncoder(nn.Module):
    def __init__(self, config: BlockCausalWAMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.text_vocab_size,
            config.d_model,
            padding_idx=0,
        )
        self.position = nn.Embedding(config.max_text_tokens, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.text_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise TypeError("task_text_tokens must be int64 [B,L]")
        if tokens.shape[1] > self.config.max_text_tokens or tokens.shape[1] <= 0:
            raise ValueError("task text token length exceeds the model contract")
        if tokens.numel() and (
            int(tokens.min()) < 0 or int(tokens.max()) >= self.config.text_vocab_size
        ):
            raise ValueError("task text token id is outside the vocabulary")
        valid = tokens.ne(0)
        if not torch.all(valid.any(dim=1)):
            raise ValueError("every sample requires non-empty task text")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)[None]
        hidden = self.encoder(hidden, src_key_padding_mask=~valid)
        weights = valid.to(dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output_norm(pooled)


class BlockCausalWAM(nn.Module):
    """Joint action-flow and action-conditioned latent-world model."""

    MODALITIES = ("task", "state", "past_action", "vision")

    def __init__(self, config: BlockCausalWAMConfig) -> None:
        super().__init__()
        self.config = config
        width = config.d_model
        self.task_text_encoder = UTF8TaskEncoder(config)
        self.task_embedding = nn.Embedding(config.num_tasks + 1, width)
        self.embodiment_embedding = nn.Embedding(config.max_agents, width)
        self.state_adapter = nn.Sequential(
            nn.LayerNorm(config.max_state_dim),
            nn.Linear(config.max_state_dim, width),
        )
        self.action_adapter = nn.Sequential(
            nn.LayerNorm(config.max_action_dim),
            nn.Linear(config.max_action_dim, width),
        )
        self.visual_adapter = nn.Sequential(
            nn.LayerNorm(config.visual_feature_dim),
            nn.Linear(config.visual_feature_dim, width),
        )
        self.history_position = nn.Embedding(config.history_steps, width)
        self.modality_embedding = nn.Embedding(len(self.MODALITIES), width)
        self.camera_embedding = nn.Embedding(config.max_cameras, width)
        # Indices 0..max_agents-1 denote agent-mounted cameras.  The final
        # identity is reserved for the global/workspace camera (and harmless
        # masked padding slots).
        self.camera_agent_embedding = nn.Embedding(config.max_agents + 1, width)
        self.visual_spatial_embedding = (
            nn.Embedding(
                config.visual_grid_height * config.visual_grid_width,
                width,
            )
            if config.visual_grid_height * config.visual_grid_width > 1
            else None
        )
        self.action_position = nn.Embedding(config.action_horizon, width)
        self.future_position = nn.Embedding(config.action_horizon, width)
        self.action_type_embedding = nn.Parameter(torch.empty(width))
        self.future_type_embedding = nn.Parameter(torch.empty(width))
        self.warm_start_embedding = nn.Embedding(2, width)
        self.flow_time_mlp = nn.Sequential(
            nn.Linear(64, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.future_action_adapter = nn.Sequential(
            nn.LayerNorm(config.max_action_dim),
            nn.Linear(config.max_action_dim, width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        self.action_velocity_head = nn.Linear(width, config.max_action_dim)
        self.future_state_head = nn.Linear(width, config.max_state_dim)
        self.future_visual_head = nn.Linear(
            width,
            config.max_cameras * config.visual_feature_dim,
        )
        self.register_buffer(
            "block_causal_mask",
            build_block_causal_attention_mask(config),
            persistent=True,
        )
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), 32))
        self.register_buffer("flow_time_frequencies", frequencies, persistent=True)
        nn.init.normal_(self.action_type_embedding, std=width**-0.5)
        nn.init.normal_(self.future_type_embedding, std=width**-0.5)

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        *,
        states: Tensor,
        state_valid_mask: Tensor,
        state_dimension_mask: Tensor,
        past_actions: Tensor,
        past_action_valid_mask: Tensor,
        action_dimension_mask: Tensor,
        visual_features: Tensor,
        image_valid_mask: Tensor,
        camera_agent_index: Tensor,
        action_horizon_mask: Tensor,
        task_text_tokens: Tensor,
        task_index: Tensor,
        embodiment_index: Tensor,
        action_inputs: Tensor,
        flow_time: Tensor,
        initial_actions: Tensor | None = None,
        future_action_condition: Tensor | None = None,
        include_future: bool = True,
    ) -> BlockCausalWAMOutput:
        batch_size = self._validate_inputs(
            states=states,
            state_valid_mask=state_valid_mask,
            state_dimension_mask=state_dimension_mask,
            past_actions=past_actions,
            past_action_valid_mask=past_action_valid_mask,
            action_dimension_mask=action_dimension_mask,
            visual_features=visual_features,
            image_valid_mask=image_valid_mask,
            camera_agent_index=camera_agent_index,
            action_horizon_mask=action_horizon_mask,
            task_text_tokens=task_text_tokens,
            task_index=task_index,
            embodiment_index=embodiment_index,
            action_inputs=action_inputs,
            flow_time=flow_time,
        )
        if initial_actions is None:
            initial_actions = torch.zeros_like(action_inputs)
            warm_index = torch.zeros(batch_size, dtype=torch.long, device=states.device)
        else:
            if initial_actions.shape != action_inputs.shape:
                raise ValueError("initial_actions and action_inputs must share shape")
            warm_index = torch.ones(batch_size, dtype=torch.long, device=states.device)
        history, history_valid = self._history_tokens(
            states=states,
            state_valid_mask=state_valid_mask,
            past_actions=past_actions,
            past_action_valid_mask=past_action_valid_mask,
            visual_features=visual_features,
            image_valid_mask=image_valid_mask,
            camera_agent_index=camera_agent_index,
            task_text_tokens=task_text_tokens,
            task_index=task_index,
            embodiment_index=embodiment_index,
        )
        action_tokens = self._action_tokens(
            action_inputs,
            initial_actions=initial_actions,
            flow_time=flow_time,
            warm_index=warm_index,
        )
        tokens = [history, action_tokens]
        valid = [
            history_valid,
            action_horizon_mask,
        ]
        if include_future:
            if future_action_condition is None:
                future_action_condition = action_inputs
            if future_action_condition.shape != action_inputs.shape:
                raise ValueError("future action condition has the wrong shape")
            tokens.append(self._future_tokens(future_action_condition))
            valid.append(action_horizon_mask)
        sequence = torch.cat(tokens, dim=1)
        key_valid = torch.cat(valid, dim=1)
        mask = self.block_causal_mask[: sequence.shape[1], : sequence.shape[1]]
        hidden = self.trunk(
            sequence,
            mask=mask,
            src_key_padding_mask=~key_valid,
            is_causal=False,
        )
        history_tokens = self.history_token_count
        action_hidden = hidden[
            :, history_tokens : history_tokens + self.config.action_horizon
        ]
        velocity = self.action_velocity_head(action_hidden)
        action_output_mask = (
            action_dimension_mask[:, None, :]
            & action_horizon_mask[:, :, None]
        )
        velocity = velocity.masked_fill(~action_output_mask, 0.0)
        future_states = None
        future_visual = None
        if include_future:
            future_hidden = hidden[:, -self.config.action_horizon :]
            future_states = self.future_state_head(future_hidden)
            future_state_mask = (
                state_dimension_mask[:, None, :]
                & action_horizon_mask[:, :, None]
            )
            future_states = future_states.masked_fill(~future_state_mask, 0.0)
            indices = torch.tensor(
                [value - 1 for value in self.config.future_visual_horizons],
                dtype=torch.long,
                device=hidden.device,
            )
            future_visual = self.future_visual_head(
                future_hidden.index_select(1, indices)
            ).reshape(
                batch_size,
                len(self.config.future_visual_horizons),
                self.config.max_cameras,
                self.config.visual_feature_dim,
            )
            camera_valid = image_valid_mask.any(dim=1)
            future_visual = future_visual.masked_fill(
                ~camera_valid[:, None, :, None],
                0.0,
            )
        return BlockCausalWAMOutput(
            action_velocity=velocity,
            future_states=future_states,
            future_visual_latents=future_visual,
        )

    @property
    def history_token_count(self) -> int:
        grid_tokens = self.config.visual_grid_height * self.config.visual_grid_width
        if grid_tokens == 1:
            return self.config.history_steps * (3 + self.config.max_cameras)
        return (
            self.config.history_steps * 3
            + self.config.visual_history_steps
            * self.config.max_cameras
            * grid_tokens
        )

    def _history_tokens(
        self,
        *,
        states: Tensor,
        state_valid_mask: Tensor,
        past_actions: Tensor,
        past_action_valid_mask: Tensor,
        visual_features: Tensor,
        image_valid_mask: Tensor,
        camera_agent_index: Tensor,
        task_text_tokens: Tensor,
        task_index: Tensor,
        embodiment_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = states.shape[0]
        task_text = self.task_text_encoder(task_text_tokens)
        task_identity = self.task_embedding(task_index)
        embodiment = self.embodiment_embedding(embodiment_index)
        shared = task_identity + embodiment
        state = self.state_adapter(states)
        aligned_actions = states.new_zeros(
            (batch_size, self.config.history_steps, self.config.max_action_dim)
        )
        aligned_actions[:, 1:] = past_actions
        action = self.action_adapter(aligned_actions)
        visual = self.visual_adapter(visual_features)
        grid_tokens = (
            self.config.visual_grid_height * self.config.visual_grid_width
        )
        if grid_tokens > 1:
            task = task_text[:, None, :].expand(
                -1,
                self.config.history_steps,
                -1,
            )
            base = torch.stack((task, state, action), dim=2)
            times = torch.arange(self.config.history_steps, device=states.device)
            base_types = torch.tensor(
                [0, 1, 2],
                dtype=torch.long,
                device=states.device,
            )
            base = (
                base
                + shared[:, None, None, :]
                + self.history_position(times)[None, :, None, :]
                + self.modality_embedding(base_types)[None, None, :, :]
            )
            cameras = torch.arange(
                self.config.max_cameras,
                device=states.device,
            )
            spatial = torch.arange(grid_tokens, device=states.device)
            visual_times = torch.arange(
                self.config.history_steps - self.config.visual_history_steps,
                self.config.history_steps,
                device=states.device,
            )
            assert self.visual_spatial_embedding is not None
            visual = (
                visual
                + shared[:, None, None, None, :]
                + self.history_position(visual_times)[None, :, None, None, :]
                + self.modality_embedding.weight[3][None, None, None, None, :]
                + self.camera_embedding(cameras)[None, None, :, None, :]
                + self.camera_agent_embedding(camera_agent_index)[
                    :, None, :, None, :
                ]
                + self.visual_spatial_embedding(spatial)[None, None, None, :, :]
            )
            action_valid = torch.zeros_like(state_valid_mask)
            action_valid[:, 1:] = past_action_valid_mask
            task_valid = torch.ones_like(state_valid_mask)
            base_valid = torch.stack(
                (task_valid, state_valid_mask, action_valid),
                dim=2,
            )
            visual_valid = (
                image_valid_mask
                & state_valid_mask[
                    :, -self.config.visual_history_steps :, None
                ]
            )
            visual_valid = visual_valid[:, :, :, None].expand(
                -1,
                -1,
                -1,
                grid_tokens,
            )
            return (
                torch.cat(
                    (base.flatten(1, 2), visual.flatten(1, 3)),
                    dim=1,
                ),
                torch.cat(
                    (base_valid.flatten(1, 2), visual_valid.flatten(1, 3)),
                    dim=1,
                ),
            )
        aligned_visual = states.new_zeros(
            (
                batch_size,
                self.config.history_steps,
                self.config.max_cameras,
                self.config.d_model,
            )
        )
        aligned_visual[:, -self.config.visual_history_steps :] = visual
        task = task_text[:, None, :].expand(-1, self.config.history_steps, -1)
        base_modalities = torch.stack((task, state, action), dim=2)
        cameras = torch.arange(self.config.max_cameras, device=states.device)
        visual = (
            aligned_visual
            + self.camera_embedding(cameras)[None, None, :, :]
            + self.camera_agent_embedding(camera_agent_index)[:, None, :, :]
        )
        modalities = torch.cat((base_modalities, visual), dim=2)
        times = torch.arange(self.config.history_steps, device=states.device)
        types = torch.tensor(
            [0, 1, 2] + [3] * self.config.max_cameras,
            dtype=torch.long,
            device=states.device,
        )
        modalities = (
            modalities
            + shared[:, None, None, :]
            + self.history_position(times)[None, :, None, :]
            + self.modality_embedding(types)[None, None, :, :]
        )
        action_valid = torch.zeros_like(state_valid_mask)
        action_valid[:, 1:] = past_action_valid_mask
        visual_valid = torch.zeros(
            (
                batch_size,
                self.config.history_steps,
                self.config.max_cameras,
            ),
            dtype=torch.bool,
            device=states.device,
        )
        visual_valid[:, -self.config.visual_history_steps :] = image_valid_mask
        visual_valid &= state_valid_mask[:, :, None]
        # Task text is a real conditioning token even before the first observed
        # state.  Keeping it valid also prevents an all-masked attention row for
        # left-padded reset histories, which would otherwise produce NaNs.
        task_valid = torch.ones_like(state_valid_mask)
        valid = torch.cat(
            (
                torch.stack((task_valid, state_valid_mask, action_valid), dim=2),
                visual_valid,
            ),
            dim=2,
        )
        return modalities.flatten(1, 2), valid.flatten(1, 2)

    def _action_tokens(
        self,
        actions: Tensor,
        *,
        initial_actions: Tensor,
        flow_time: Tensor,
        warm_index: Tensor,
    ) -> Tensor:
        positions = torch.arange(self.config.action_horizon, device=actions.device)
        phase = flow_time[:, None] * self.flow_time_frequencies.to(flow_time)[None]
        time_features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        return (
            self.action_adapter(actions)
            + self.action_adapter(initial_actions)
            + self.flow_time_mlp(time_features)[:, None, :]
            + self.warm_start_embedding(warm_index)[:, None, :]
            + self.action_position(positions)[None]
            + self.action_type_embedding
        )

    def _future_tokens(self, actions: Tensor) -> Tensor:
        count = torch.arange(
            1,
            self.config.action_horizon + 1,
            device=actions.device,
            dtype=actions.dtype,
        ).view(1, -1, 1)
        prefix = actions.cumsum(dim=1) / count
        positions = torch.arange(self.config.action_horizon, device=actions.device)
        return (
            self.future_action_adapter(prefix)
            + self.future_position(positions)[None]
            + self.future_type_embedding
        )

    def integrate_actions(
        self,
        context: Mapping[str, Tensor],
        *,
        initial_actions: Tensor | None = None,
        solver_steps: int = 4,
        solver: str = "euler",
        normalized_clip: float = 10.0,
        forward_model: nn.Module | None = None,
    ) -> Tensor:
        """Integrate the deployed action field while preserving autograd.

        M2 represents action chunks in per-task z-score space.  This method is
        deliberately shared by training and inference so the endpoint loss is
        taken on the exact computation that deployment executes.
        """

        if solver_steps <= 0 or normalized_clip <= 0.0:
            raise ValueError("solver_steps and normalized_clip must be positive")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be euler or heun")
        velocity_model = self if forward_model is None else forward_model
        batch_size = context["states"].shape[0]
        shape = (
            batch_size,
            self.config.action_horizon,
            self.config.max_action_dim,
        )
        initial = (
            torch.zeros(shape, device=context["states"].device, dtype=context["states"].dtype)
            if initial_actions is None
            else initial_actions.clone()
        )
        if initial.shape != shape:
            raise ValueError(f"initial_actions must have shape {shape}")
        current = initial.clone()
        dt = 1.0 / solver_steps
        for step in range(solver_steps):
            tau = torch.full(
                (batch_size,),
                step * dt,
                device=current.device,
                dtype=current.dtype,
            )
            velocity = velocity_model(
                **context,
                action_inputs=current,
                flow_time=tau,
                initial_actions=initial if initial_actions is not None else None,
                include_future=False,
            ).action_velocity
            if solver == "euler":
                current = current + dt * velocity
            else:
                proposal = current + dt * velocity
                next_tau = torch.full_like(tau, (step + 1) * dt)
                correction = velocity_model(
                    **context,
                    action_inputs=proposal,
                    flow_time=next_tau,
                    initial_actions=initial if initial_actions is not None else None,
                    include_future=False,
                ).action_velocity
                current = current + 0.5 * dt * (velocity + correction)
            # Out-of-place clamp is required when the same solver participates
            # in the differentiable endpoint objective.
            current = current.clamp(-normalized_clip, normalized_clip)
        mask = (
            context["action_dimension_mask"][:, None, :]
            & context["action_horizon_mask"][:, :, None]
        )
        return current.masked_fill(~mask, 0.0)

    @torch.no_grad()
    def generate_actions(
        self,
        context: Mapping[str, Tensor],
        *,
        initial_actions: Tensor | None = None,
        solver_steps: int = 4,
        solver: str = "euler",
        normalized_clip: float = 10.0,
    ) -> Tensor:
        return self.integrate_actions(
            context,
            initial_actions=initial_actions,
            solver_steps=solver_steps,
            solver=solver,
            normalized_clip=normalized_clip,
        )

    def _validate_inputs(self, **values: Tensor) -> int:
        states = values["states"]
        if states.ndim != 3 or states.shape[1:] != (
            self.config.history_steps,
            self.config.max_state_dim,
        ):
            raise ValueError("states have the wrong M2 shape")
        batch_size = states.shape[0]
        expected = {
            "state_valid_mask": (batch_size, self.config.history_steps),
            "state_dimension_mask": (batch_size, self.config.max_state_dim),
            "past_actions": (
                batch_size,
                self.config.history_steps - 1,
                self.config.max_action_dim,
            ),
            "past_action_valid_mask": (batch_size, self.config.history_steps - 1),
            "action_dimension_mask": (batch_size, self.config.max_action_dim),
            "image_valid_mask": (
                batch_size,
                self.config.visual_history_steps,
                self.config.max_cameras,
            ),
            "camera_agent_index": (
                batch_size,
                self.config.max_cameras,
            ),
            "action_horizon_mask": (
                batch_size,
                self.config.action_horizon,
            ),
            "task_index": (batch_size,),
            "embodiment_index": (batch_size,),
            "action_inputs": (
                batch_size,
                self.config.action_horizon,
                self.config.max_action_dim,
            ),
            "flow_time": (batch_size,),
        }
        for name, shape in expected.items():
            if tuple(values[name].shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        visual = values["visual_features"]
        grid_tokens = (
            self.config.visual_grid_height * self.config.visual_grid_width
        )
        expected_visual = (
            batch_size,
            self.config.visual_history_steps,
            self.config.max_cameras,
            self.config.visual_feature_dim,
        )
        if grid_tokens > 1:
            expected_visual = (
                batch_size,
                self.config.visual_history_steps,
                self.config.max_cameras,
                grid_tokens,
                self.config.visual_feature_dim,
            )
        if tuple(visual.shape) != expected_visual:
            raise ValueError("visual_features have the wrong M2 shape")
        boolean_names = (
            "state_valid_mask",
            "state_dimension_mask",
            "past_action_valid_mask",
            "action_dimension_mask",
            "image_valid_mask",
            "action_horizon_mask",
        )
        if any(values[name].dtype != torch.bool for name in boolean_names):
            raise TypeError("M2 validity/dimension masks must be boolean")
        if (
            values["task_index"].dtype != torch.long
            or values["embodiment_index"].dtype != torch.long
            or values["camera_agent_index"].dtype != torch.long
        ):
            raise TypeError("task/embodiment/camera-agent indices must be int64")
        if not bool(values["image_valid_mask"].any(dim=(1, 2)).all()):
            raise ValueError("every M2 sample requires at least one valid camera frame")
        if not bool(values["action_horizon_mask"].any(dim=1).all()):
            raise ValueError("every M2 sample requires a positive task horizon")
        if int(values["task_index"].min()) < 0 or int(values["task_index"].max()) >= self.config.num_tasks:
            raise ValueError("task_index is outside the configured training vocabulary")
        if int(values["embodiment_index"].min()) < 0 or int(values["embodiment_index"].max()) >= self.config.max_agents:
            raise ValueError("embodiment_index is outside the configured slots")
        if (
            int(values["camera_agent_index"].min()) < 0
            or int(values["camera_agent_index"].max()) > self.config.max_agents
        ):
            raise ValueError("camera_agent_index is outside the configured identities")
        devices = {value.device for value in values.values()}
        if len(devices) != 1:
            raise TypeError("all M2 inputs must share a device")
        return batch_size


__all__ = [
    "BlockCausalWAM",
    "BlockCausalWAMConfig",
    "BlockCausalWAMOutput",
    "UTF8TaskEncoder",
    "build_block_causal_attention_mask",
]
