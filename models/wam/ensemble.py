"""Independent bootstrap ensemble and planner-facing risk surface for RWM-U."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from models.wam.api import (
    WorldModelRolloutInputs,
    WorldModelRolloutOutput,
    WorldModelSequenceInputs,
)
from models.wam.config import RWMARConfig, RWMUEnsembleConfig, RWMURiskConfig
from models.wam.normalizer import NormalizationStats
from models.wam.recurrent_dynamics import RWMARWorldModel
from models.wam.rollout import wrap_to_pi


@dataclass(frozen=True)
class RWMUEnsemblePredictions:
    """Deterministic member predictions with leading shape ``[E,B,H,*]``."""

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


class RWMUEnsemble(nn.Module):
    """world-model ensemble ensemble of complete, independently initialized RWM-AR members."""

    def __init__(
        self,
        members: Sequence[RWMARWorldModel],
        config: RWMUEnsembleConfig,
        stats: NormalizationStats,
        *,
        risk_config: RWMURiskConfig | None = None,
    ) -> None:
        super().__init__()
        if len(members) != config.ensemble_size:
            raise ValueError(
                f"expected {config.ensemble_size} members, got {len(members)}"
            )
        if not members:
            raise ValueError("members cannot be empty")
        member_config = members[0].config
        if any(member.config != member_config for member in members[1:]):
            raise ValueError("all ensemble members must have identical configs")
        if stats.state_std.shape != (member_config.state_dim,):
            raise ValueError("normalization state dimension does not match members")
        if stats.action_std.shape != (member_config.action_dim,):
            raise ValueError("normalization action dimension does not match members")
        self.members = nn.ModuleList(members)
        self.config = config
        self.risk_config = risk_config or RWMURiskConfig()
        self.register_buffer(
            "state_std", torch.as_tensor(stats.state_std, dtype=torch.float32)
        )
        self.register_buffer(
            "action_mean", torch.as_tensor(stats.action_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "action_std", torch.as_tensor(stats.action_std, dtype=torch.float32)
        )

    @classmethod
    def create(
        cls,
        member_config: RWMARConfig,
        ensemble_config: RWMUEnsembleConfig,
        stats: NormalizationStats,
        *,
        risk_config: RWMURiskConfig | None = None,
    ) -> "RWMUEnsemble":
        return cls(
            [
                RWMARWorldModel(member_config, stats)
                for _ in range(ensemble_config.ensemble_size)
            ],
            ensemble_config,
            stats,
            risk_config=risk_config,
        )

    @property
    def member_config(self) -> RWMARConfig:
        return self.members[0].config

    def predict(
        self, history: WorldModelSequenceInputs, candidate_actions: Tensor
    ) -> RWMUEnsemblePredictions:
        """Run every member without aleatoric sampling for uncertainty analysis."""

        predictions = [
            member.predict(history, candidate_actions, sample_state=False)
            for member in self.members
        ]
        return RWMUEnsemblePredictions(
            **{
                name: torch.stack(
                    [getattr(prediction, name) for prediction in predictions], dim=0
                )
                for name in RWMUEnsemblePredictions.__dataclass_fields__
            }
        )

    def state_moments(
        self, predictions: RWMUEnsemblePredictions
    ) -> Mapping[str, Tensor]:
        """Decompose predictive variance into epistemic and aleatoric terms."""

        member_states = predictions.next_state_mean
        mean = member_states.mean(dim=0)
        for yaw_index in self.member_config.yaw_indices:
            circular = torch.atan2(
                torch.sin(member_states[..., yaw_index]).mean(dim=0),
                torch.cos(member_states[..., yaw_index]).mean(dim=0),
            )
            mean = _replace_column(mean, yaw_index, circular)
        deviations = member_states - mean.unsqueeze(0)
        for yaw_index in self.member_config.yaw_indices:
            deviations = _replace_column(
                deviations,
                yaw_index,
                wrap_to_pi(deviations[..., yaw_index]),
            )
        epistemic_variance = deviations.square().mean(dim=0)
        aleatoric_variance = (
            2.0 * predictions.state_delta_log_std.float()
        ).exp().mean(dim=0)
        total_variance = epistemic_variance + aleatoric_variance
        return {
            "mean": mean,
            "epistemic_variance": epistemic_variance,
            "aleatoric_variance": aleatoric_variance,
            "total_variance": total_variance,
            "epistemic_std": epistemic_variance.sqrt(),
            "aleatoric_std": aleatoric_variance.sqrt(),
            "total_std": total_variance.sqrt(),
        }

    def risk_scores(
        self,
        predictions: RWMUEnsemblePredictions,
        candidate_actions: Tensor,
    ) -> Mapping[str, Tensor]:
        """Return non-negative per-step risk terms with shape ``[B,H]``."""

        moments = self.state_moments(predictions)
        state_std = self.state_std.to(
            device=candidate_actions.device, dtype=candidate_actions.dtype
        )
        continuous = self.members[0].continuous_state_mask
        epistemic = torch.sqrt(
            (
                moments["epistemic_variance"][..., continuous]
                / state_std[continuous].square()
            ).mean(dim=-1)
        )
        aleatoric = torch.sqrt(
            (
                moments["aleatoric_variance"][..., continuous]
                / state_std[continuous].square()
            ).mean(dim=-1)
        )
        failure = predictions.failure_logit.sigmoid().mean(dim=0).squeeze(-1)
        action_std = self.action_std.to(
            device=candidate_actions.device, dtype=candidate_actions.dtype
        )
        action_mean = self.action_mean.to(
            device=candidate_actions.device, dtype=candidate_actions.dtype
        )
        normalized_action = (candidate_actions - action_mean) / action_std
        action_ood = torch.relu(
            normalized_action.abs() - self.risk_config.action_ood_threshold
        ).mean(dim=-1)
        total = (
            self.risk_config.epistemic_weight * epistemic
            + self.risk_config.aleatoric_weight * aleatoric
            + self.risk_config.failure_weight * failure
            + self.risk_config.action_ood_weight * action_ood
        )
        return {
            "total": total,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "failure_probability": failure,
            "action_ood": action_ood,
        }

    def forward(self, inputs: WorldModelRolloutInputs) -> WorldModelRolloutOutput:
        """Roll out ``num_particles`` paths per member with fixed member identity."""

        deterministic = self.predict(inputs.history, inputs.candidate_actions)
        moments = self.state_moments(deterministic)
        risk = self.risk_scores(deterministic, inputs.candidate_actions)
        particles = int(inputs.num_particles)
        device = inputs.candidate_actions.device
        if particles == 1:
            state_distribution = {
                "mean": deterministic.next_state_mean,
                "delta_mean": deterministic.state_delta_mean,
                "delta_log_std": deterministic.state_delta_log_std,
            }
            rewards = deterministic.reward
            termination = {
                "done_logit": deterministic.done_logit,
                "success_logit": deterministic.success_logit,
                "failure_logit": deterministic.failure_logit,
            }
            rollout_diagnostics = {
                "response_progress": deterministic.response_progress,
                "coordination_error": deterministic.coordination_error,
                "executed_action": deterministic.executed_action,
            }
        else:
            member_outputs = [member(inputs) for member in self.members]

            def concatenate_mapping(name: str, key: str) -> Tensor:
                mappings = [getattr(output, name) for output in member_outputs]
                return torch.cat([mapping[key] for mapping in mappings], dim=0)

            state_distribution = {
                key: concatenate_mapping("state_distribution", key)
                for key in ("mean", "delta_mean", "delta_log_std")
            }
            rewards = torch.cat(
                [output.rewards for output in member_outputs], dim=0
            )
            termination = {
                key: concatenate_mapping("termination", key)
                for key in ("done_logit", "success_logit", "failure_logit")
            }
            rollout_diagnostics = {
                "response_progress": concatenate_mapping(
                    "diagnostics", "response_progress"
                ),
                "coordination_error": concatenate_mapping(
                    "diagnostics", "coordination_error"
                ),
                "executed_action": concatenate_mapping(
                    "diagnostics", "executed_action"
                ),
            }
        member_index = torch.arange(
            self.config.ensemble_size, device=device, dtype=torch.int64
        ).repeat_interleave(particles)
        particle_index = torch.arange(
            particles, device=device, dtype=torch.int64
        ).repeat(self.config.ensemble_size)
        return WorldModelRolloutOutput(
            state_distribution=state_distribution,
            rewards=rewards,
            termination=termination,
            uncertainty={
                "epistemic_std": moments["epistemic_std"],
                "aleatoric_std": moments["aleatoric_std"],
                "total_std": moments["total_std"],
            },
            diagnostics={
                **rollout_diagnostics,
                "member_index": member_index,
                "particle_index": particle_index,
                "risk_total": risk["total"],
                "risk_epistemic": risk["epistemic"],
                "risk_aleatoric": risk["aleatoric"],
                "risk_failure_probability": risk["failure_probability"],
                "risk_action_ood": risk["action_ood"],
            },
        )


def _replace_column(value: Tensor, index: int, column: Tensor) -> Tensor:
    parts = list(value.split(1, dim=-1))
    parts[index] = column.unsqueeze(-1)
    return torch.cat(parts, dim=-1)


__all__ = ["RWMUEnsemble", "RWMUEnsemblePredictions"]
