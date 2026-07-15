"""Single-member recurrent world-action model with outer autoregression."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn

from models.wam.api import (
    WorldModelRolloutInputs,
    WorldModelRolloutOutput,
    WorldModelSequenceInputs,
)
from models.wam.config import RWMARConfig
from models.wam.heads import RWMHeadOutput, RWMHeads
from models.wam.normalizer import NormalizationStats
from models.wam.rollout import symexp, wrap_to_pi
from models.wam.state_features import StateFeatureEncoder


@dataclass(frozen=True)
class RWMARRolloutPredictions:
    """Training-oriented single-model rollout with shape ``[B,H,*]``."""

    state_delta_mean: Tensor
    state_delta_log_std: Tensor
    next_state_mean: Tensor
    normalized_delta_mean: Tensor
    normalized_delta_log_std: Tensor
    gripper_closed_logit: Tensor
    reward: Tensor
    reward_symlog: Tensor
    done_logit: Tensor
    success_logit: Tensor
    failure_logit: Tensor
    response_progress: Tensor
    coordination_error: Tensor
    executed_action: Tensor


class RWMARWorldModel(nn.Module):
    """History-conditioned probabilistic RWM-AR used in Phase 1.

    Inner autoregression compresses valid ``(state, preceding action)`` history
    into a GRU belief.  Outer autoregression predicts a future state and feeds
    that prediction back through the same recurrent transition before decoding
    the next candidate action.
    """

    def __init__(self, config: RWMARConfig, stats: NormalizationStats) -> None:
        super().__init__()
        if not config.predict_delta:
            raise ValueError("Phase 1 RWM-AR requires predict_delta=true")
        if stats.state_mean.shape != (config.state_dim,):
            raise ValueError("normalization state dimension does not match config")
        if stats.action_mean.shape != (config.action_dim,):
            raise ValueError("normalization action dimension does not match config")
        self.config = config
        self.features = StateFeatureEncoder(
            stats,
            yaw_indices=config.yaw_indices,
        )
        transition_dim = self.features.feature_dim + config.action_dim
        self.transition_encoder = nn.Sequential(
            nn.Linear(transition_dim, config.encoder_hidden_dim),
            nn.LayerNorm(config.encoder_hidden_dim),
            nn.SiLU(),
        )
        self.belief_gru = nn.GRU(
            input_size=config.encoder_hidden_dim,
            hidden_size=config.gru_hidden_dim,
            num_layers=config.gru_layers,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
            batch_first=True,
        )
        decoder_input_dim = (
            config.gru_hidden_dim + self.features.feature_dim + config.action_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, config.encoder_hidden_dim),
            nn.LayerNorm(config.encoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.encoder_hidden_dim, config.encoder_hidden_dim),
            nn.SiLU(),
        )
        self.heads = RWMHeads(
            config.encoder_hidden_dim,
            config.state_dim,
            config.action_dim,
            closed_count=len(config.gripper_closed_indices),
            min_log_std=config.min_log_std,
            max_log_std=config.max_log_std,
        )
        self.register_buffer(
            "delta_mean", torch.as_tensor(stats.delta_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "delta_std", torch.as_tensor(stats.delta_std, dtype=torch.float32)
        )
        continuous_mask = torch.ones(config.state_dim, dtype=torch.bool)
        continuous_mask[list(config.gripper_closed_indices)] = False
        self.register_buffer("continuous_state_mask", continuous_mask)

    def encode_history(self, inputs: WorldModelSequenceInputs) -> tuple[Tensor, Tensor]:
        self._validate_history(inputs)
        states = inputs.states
        batch_size, history_horizon, _ = states.shape
        # There is no action before the first valid state in a truncated
        # history.  Filling that sentinel with the training mean makes its
        # normalized feature exactly zero.  Raw zero is unsafe when a command
        # dimension is constant (std floor): it can become ~1e6 and overflow
        # the first FP16 linear layer under AMP.
        preceding_actions = (
            self.features.action_mean.to(dtype=states.dtype)
            .view(1, 1, self.config.action_dim)
            .expand(batch_size, history_horizon, self.config.action_dim)
            .clone()
        )
        if history_horizon > 1:
            preceding_actions[:, 1:] = inputs.past_actions
        encoded = self._encode_transition(states, preceding_actions)
        lengths = inputs.valid_mask.sum(dim=1)
        positions = torch.arange(history_horizon, device=states.device).unsqueeze(0)
        source = positions + (history_horizon - lengths).unsqueeze(1)
        source = source.clamp_max(history_horizon - 1)
        compact = encoded.gather(
            1,
            source.unsqueeze(-1).expand(batch_size, history_horizon, encoded.shape[-1]),
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            compact,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.belief_gru(packed)
        # Valid history is required to be a contiguous suffix.  The current
        # state is consequently always the final tensor position.
        current_state = states[:, -1]
        return hidden, current_state

    def predict(
        self,
        history: WorldModelSequenceInputs,
        candidate_actions: Tensor,
        *,
        sample_state: bool = False,
    ) -> RWMARRolloutPredictions:
        self._validate_candidates(history, candidate_actions)
        return self._predict_rollout(
            history,
            candidate_actions,
            sample_state=sample_state,
            teacher_states=None,
        )

    def predict_teacher_forced(
        self,
        history: WorldModelSequenceInputs,
        candidate_actions: Tensor,
        teacher_states: Tensor,
    ) -> RWMARRolloutPredictions:
        """Decode each step from the true preceding state for Gate C ablation.

        This method is deliberately separate from :meth:`predict`; deployment
        and open-loop evaluation therefore cannot accidentally enable teacher
        forcing.
        """

        self._validate_candidates(history, candidate_actions)
        if teacher_states.shape != (
            candidate_actions.shape[0],
            candidate_actions.shape[1],
            self.config.state_dim,
        ):
            raise ValueError("teacher_states must have shape [B,H,state_dim]")
        if (
            teacher_states.device != history.states.device
            or teacher_states.dtype != history.states.dtype
        ):
            raise TypeError("teacher_states must match history dtype and device")
        return self._predict_rollout(
            history,
            candidate_actions,
            sample_state=False,
            teacher_states=teacher_states,
        )

    def _predict_rollout(
        self,
        history: WorldModelSequenceInputs,
        candidate_actions: Tensor,
        *,
        sample_state: bool,
        teacher_states: Tensor | None,
    ) -> RWMARRolloutPredictions:
        hidden, current_state = self.encode_history(history)
        collected: dict[str, list[Tensor]] = {
            name: [] for name in RWMARRolloutPredictions.__dataclass_fields__
        }
        for step in range(candidate_actions.shape[1]):
            action = candidate_actions[:, step]
            head = self._decode(hidden[-1], current_state, action)
            normalized_delta = head.normalized_delta_mean
            if sample_state:
                normalized_delta = (
                    normalized_delta
                    + torch.randn_like(normalized_delta)
                    * head.normalized_delta_log_std.exp()
                )
            raw_delta = normalized_delta * self.delta_std + self.delta_mean
            raw_log_std = head.normalized_delta_log_std + self.delta_std.log()
            next_state = current_state + raw_delta
            for yaw_index in self.config.yaw_indices:
                next_state = self._replace_column(
                    next_state,
                    yaw_index,
                    wrap_to_pi(next_state[..., yaw_index]),
                )
            closed_probability = head.gripper_closed_logit.sigmoid()
            for offset, state_index in enumerate(self.config.gripper_closed_indices):
                closed_value = closed_probability[..., offset]
                if sample_state:
                    closed_value = torch.bernoulli(closed_value)
                next_state = self._replace_column(next_state, state_index, closed_value)
                raw_delta = self._replace_column(
                    raw_delta,
                    state_index,
                    closed_value - current_state[..., state_index],
                )
            values = {
                "state_delta_mean": raw_delta,
                "state_delta_log_std": raw_log_std,
                "next_state_mean": next_state,
                "normalized_delta_mean": head.normalized_delta_mean,
                "normalized_delta_log_std": head.normalized_delta_log_std,
                "gripper_closed_logit": head.gripper_closed_logit,
                # Task rewards are bounded; clamping only protects inverse
                # symlog diagnostics from overflow under mixed precision.
                "reward": symexp(head.reward_symlog.float().clamp(-20.0, 20.0)),
                "reward_symlog": head.reward_symlog,
                "done_logit": head.done_logit,
                "success_logit": head.success_logit,
                "failure_logit": head.failure_logit,
                "response_progress": head.response_progress,
                "coordination_error": head.coordination_error,
                "executed_action": head.executed_action,
            }
            for name, value in values.items():
                collected[name].append(value)
            feedback_state = (
                teacher_states[:, step] if teacher_states is not None else next_state
            )
            transition = self._encode_transition(
                feedback_state.unsqueeze(1), action.unsqueeze(1)
            )
            _, hidden = self.belief_gru(transition, hidden)
            current_state = feedback_state
        return RWMARRolloutPredictions(
            **{name: torch.stack(values, dim=1) for name, values in collected.items()}
        )

    def forward(self, inputs: WorldModelRolloutInputs) -> WorldModelRolloutOutput:
        history = inputs.history
        actions = inputs.candidate_actions
        particles = int(inputs.num_particles)
        if particles == 1:
            predictions = self.predict(history, actions, sample_state=False)
            return self._public_output(predictions, particle_dim=True)
        repeated_history = WorldModelSequenceInputs(
            states=history.states.repeat_interleave(particles, dim=0),
            past_actions=history.past_actions.repeat_interleave(particles, dim=0),
            valid_mask=history.valid_mask.repeat_interleave(particles, dim=0),
        )
        repeated_actions = actions.repeat_interleave(particles, dim=0)
        predictions = self.predict(
            repeated_history, repeated_actions, sample_state=True
        )
        batch_size = actions.shape[0]

        def particles_first(value: Tensor) -> Tensor:
            shaped = value.reshape(batch_size, particles, *value.shape[1:])
            return shaped.transpose(0, 1)

        return self._public_output(
            RWMARRolloutPredictions(
                **{
                    name: particles_first(getattr(predictions, name))
                    for name in predictions.__dataclass_fields__
                }
            ),
            particle_dim=False,
        )

    def _decode(
        self, belief: Tensor, current_state: Tensor, action: Tensor
    ) -> RWMHeadOutput:
        decoded = self.decoder(
            torch.cat(
                (
                    belief,
                    self.features.encode_state(current_state),
                    self.features.normalize_action(action),
                ),
                dim=-1,
            )
        )
        return self.heads(decoded)

    def _encode_transition(self, state: Tensor, preceding_action: Tensor) -> Tensor:
        return self.transition_encoder(
            torch.cat(
                (
                    self.features.encode_state(state),
                    self.features.normalize_action(preceding_action),
                ),
                dim=-1,
            )
        )

    def _public_output(
        self,
        predictions: RWMARRolloutPredictions,
        *,
        particle_dim: bool,
    ) -> WorldModelRolloutOutput:
        def maybe_unsqueeze(value: Tensor) -> Tensor:
            return value.unsqueeze(0) if particle_dim else value

        return WorldModelRolloutOutput(
            state_distribution={
                "mean": maybe_unsqueeze(predictions.next_state_mean),
                "delta_mean": maybe_unsqueeze(predictions.state_delta_mean),
                "delta_log_std": maybe_unsqueeze(predictions.state_delta_log_std),
            },
            rewards=maybe_unsqueeze(predictions.reward),
            termination={
                "done_logit": maybe_unsqueeze(predictions.done_logit),
                "success_logit": maybe_unsqueeze(predictions.success_logit),
                "failure_logit": maybe_unsqueeze(predictions.failure_logit),
            },
            uncertainty={
                "aleatoric_std": maybe_unsqueeze(predictions.state_delta_log_std.exp())
            },
            diagnostics={
                "response_progress": maybe_unsqueeze(predictions.response_progress),
                "coordination_error": maybe_unsqueeze(predictions.coordination_error),
                "executed_action": maybe_unsqueeze(predictions.executed_action),
            },
        )

    def _validate_history(self, inputs: WorldModelSequenceInputs) -> None:
        states = inputs.states
        actions = inputs.past_actions
        mask = inputs.valid_mask
        if states.ndim != 3 or states.shape[-1] != self.config.state_dim:
            raise ValueError("states must have shape [B,T,state_dim]")
        expected_action_shape = (
            states.shape[0],
            max(states.shape[1] - 1, 0),
            self.config.action_dim,
        )
        if tuple(actions.shape) != expected_action_shape:
            raise ValueError(
                f"past_actions must have shape {expected_action_shape}, "
                f"got {tuple(actions.shape)}"
            )
        if tuple(mask.shape) != tuple(states.shape[:2]) or mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with shape [B,T]")
        if not bool(mask[:, -1].all()):
            raise ValueError("every history must end in a valid current state")
        if bool((mask[:, 1:].logical_not() & mask[:, :-1]).any()):
            raise ValueError("valid history must be a contiguous suffix")
        if states.device != actions.device or states.device != mask.device:
            raise TypeError("history tensors must share a device")
        if states.dtype != actions.dtype or not torch.is_floating_point(states):
            raise TypeError("states/actions must share a floating dtype")

    def _validate_candidates(
        self, history: WorldModelSequenceInputs, candidate_actions: Tensor
    ) -> None:
        if (
            candidate_actions.ndim != 3
            or candidate_actions.shape[0] != history.states.shape[0]
            or candidate_actions.shape[-1] != self.config.action_dim
            or candidate_actions.shape[1] <= 0
        ):
            raise ValueError("candidate_actions must have shape [B,H,action_dim]")
        if (
            candidate_actions.device != history.states.device
            or candidate_actions.dtype != history.states.dtype
        ):
            raise TypeError("candidate_actions must match history dtype and device")

    @staticmethod
    def _replace_column(value: Tensor, index: int, column: Tensor) -> Tensor:
        parts = list(value.split(1, dim=-1))
        parts[index] = column.unsqueeze(-1)
        return torch.cat(parts, dim=-1)


__all__ = ["RWMARRolloutPredictions", "RWMARWorldModel"]
