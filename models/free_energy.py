from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


def normalize_hypothesis_weights(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return a non-negative posterior normalized over the last dimension.

    Hypothesis weights are beliefs, not arbitrary attention logits.  Failing fast
    on an all-zero row avoids silently turning an invalid belief into an arbitrary
    uniform distribution.
    """
    if weights.ndim < 1:
        raise ValueError("hypothesis weights must have at least one dimension")
    if not torch.is_floating_point(weights):
        weights = weights.float()
    if not torch.isfinite(weights).all():
        raise ValueError("hypothesis weights must be finite")
    if (weights < 0).any():
        raise ValueError("hypothesis weights must be non-negative")

    total = weights.sum(dim=-1, keepdim=True)
    if (total <= eps).any():
        raise ValueError("each hypothesis posterior must have positive mass")
    return weights / total


def multi_hypothesis_expected_free_energy(
    hypothesis_G: torch.Tensor,
    hypothesis_weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Aggregate candidate costs under a discrete teammate-plan posterior.

    Args:
        hypothesis_G: Free energies with shape ``[..., K, M]``, where ``K`` is
            the number of ego candidates and ``M`` the number of teammate-plan
            hypotheses.
        hypothesis_weights: Posterior weights broadcastable to ``[..., M]``.

    The no-message controller must choose one candidate before the latent
    teammate plan is revealed, hence ``G_no = min_k E_m[G(k,m)]``.  With a
    perfect reply it may choose a candidate after the reveal, hence
    ``G_reveal = E_m[min_k G(k,m)]``.  Their difference is the (non-negative,
    up to numerical error) value of perfect information.
    """
    if hypothesis_G.ndim < 2:
        raise ValueError("hypothesis_G must have shape [..., candidates, hypotheses]")
    if not torch.is_floating_point(hypothesis_G):
        hypothesis_G = hypothesis_G.float()
    if not torch.isfinite(hypothesis_G).all():
        raise ValueError("hypothesis_G must be finite")

    num_hypotheses = hypothesis_G.shape[-1]
    if hypothesis_weights.ndim < 1 or hypothesis_weights.shape[-1] != num_hypotheses:
        raise ValueError(
            "hypothesis_weights last dimension must match hypothesis_G hypotheses: "
            f"{tuple(hypothesis_weights.shape)} vs {tuple(hypothesis_G.shape)}"
        )

    target_shape = hypothesis_G.shape[:-2] + (num_hypotheses,)
    try:
        weights = torch.broadcast_to(
            hypothesis_weights.to(device=hypothesis_G.device, dtype=hypothesis_G.dtype),
            target_shape,
        )
    except RuntimeError as exc:
        raise ValueError(
            f"hypothesis_weights shape {tuple(hypothesis_weights.shape)} is not "
            f"broadcastable to {target_shape}"
        ) from exc
    weights = normalize_hypothesis_weights(weights)

    expected_by_candidate = (hypothesis_G * weights.unsqueeze(-2)).sum(dim=-1)
    G_no, no_comm_plan_index = expected_by_candidate.min(dim=-1)

    revealed_best_by_hypothesis, reveal_plan_index = hypothesis_G.min(dim=-2)
    G_reveal = (revealed_best_by_hypothesis * weights).sum(dim=-1)
    raw_vpi = G_no - G_reveal
    vpi = raw_vpi.clamp_min(0.0)

    return {
        "hypothesis_weights": weights,
        "expected_G_by_candidate": expected_by_candidate,
        "G_no": G_no,
        "G_reveal": G_reveal,
        "VPI": vpi,
        "raw_VPI": raw_vpi,
        "no_comm_plan_index": no_comm_plan_index,
        "reveal_plan_index": reveal_plan_index,
        "revealed_best_G_by_hypothesis": revealed_best_by_hypothesis,
    }


@dataclass
class FreeEnergyConfig:
    goal_y: float = 3.05
    force_limit: float = 1.0
    alpha_goal: float = 1.0
    alpha_safety: float = 2.0
    alpha_collab: float = 1.0
    alpha_unc: float = 0.5
    alpha_ctrl: float = 0.05
    contact_weight: float = 1.0
    force_weight: float = 1.0
    terminal_goal_weight: float = 1.0
    mean_goal_weight: float = 0.25
    gripper_sync_weight: float = 0.5
    base_sync_weight: float = 1.0
    smooth_weight: float = 0.5
    use_calibrated_utility: bool = True
    return_scale: float = 100.0
    tail_risk_weight: float = 0.5
    constraint_risk_weight: float = 1.0
    success_risk_weight: float = 0.5
    safety_probability_threshold: float = 0.5
    infeasible_penalty: float = 10.0
    calibration_scale: float = 1.0
    calibration_bias: float = 0.0


class FreeEnergyEvaluator:
    def __init__(self, cfg: FreeEnergyConfig):
        self.cfg = cfg

    @staticmethod
    def contact_probability(rollout: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "pred_contact_prob" in rollout:
            return rollout["pred_contact_prob"]
        if "pred_contact_logits" in rollout:
            return torch.sigmoid(rollout["pred_contact_logits"])
        raise KeyError("rollout must contain either pred_contact_prob or pred_contact_logits")

    def goal_cost(self, rollout: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        progress = rollout["pred_progress"]
        terminal_gap = F.relu(torch.as_tensor(cfg.goal_y, device=progress.device, dtype=progress.dtype) - progress[:, -1])
        mean_gap = F.relu(torch.as_tensor(cfg.goal_y, device=progress.device, dtype=progress.dtype) - progress).mean(dim=1)
        return cfg.terminal_goal_weight * terminal_gap + cfg.mean_goal_weight * mean_gap

    def safety_cost(self, rollout: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        contact = self.contact_probability(rollout).mean(dim=1)
        force = rollout["pred_force"].abs()
        force_violation = F.relu(force - cfg.force_limit).mean(dim=1)
        force_mean = force.mean(dim=1)
        return cfg.contact_weight * contact + cfg.force_weight * (force_violation + 0.1 * force_mean)

    def collaboration_cost(self, rollout: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        actions = rollout["pred_actions"]
        a0 = actions[..., 0:4]
        a1 = actions[..., 4:8]
        base_sync = (a0[..., 0:3] - a1[..., 0:3]).pow(2).mean(dim=(1, 2))
        gripper_sync = (a0[..., 3] - a1[..., 3]).abs().mean(dim=1)
        return cfg.base_sync_weight * base_sync + cfg.gripper_sync_weight * gripper_sync

    def control_cost(self, rollout: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        actions = rollout["pred_actions"]
        mag = actions.pow(2).mean(dim=(1, 2))
        if actions.shape[1] > 1:
            smooth = (actions[:, 1:] - actions[:, :-1]).pow(2).mean(dim=(1, 2))
        else:
            smooth = torch.zeros_like(mag)
        return mag + cfg.smooth_weight * smooth

    def total_score(self, rollout: Dict[str, torch.Tensor], uncertainty: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        B = rollout["pred_actions"].shape[0]
        device = rollout["pred_actions"].device
        dtype = rollout["pred_actions"].dtype

        goal = self.goal_cost(rollout)
        safety = self.safety_cost(rollout)
        collab = self.collaboration_cost(rollout)
        ctrl = self.control_cost(rollout)

        if uncertainty is None:
            unc = torch.zeros(B, device=device, dtype=dtype)
        else:
            unc = uncertainty.to(device=device, dtype=dtype).reshape(B)

        legacy_total = (
            cfg.alpha_goal * goal
            + cfg.alpha_safety * safety
            + cfg.alpha_collab * collab
            + cfg.alpha_unc * unc
            + cfg.alpha_ctrl * ctrl
        )

        utility_available = all(
            name in rollout
            for name in (
                "pred_return_quantiles",
                "pred_success_logits",
                "pred_collision_logits",
                "pred_force_violation_logits",
            )
        )
        if cfg.use_calibrated_utility and utility_available:
            quantiles = rollout["pred_return_quantiles"] / max(cfg.return_scale, 1e-8)
            expected_return = quantiles.mean(dim=-1)
            lower_return = quantiles[..., 0]
            tail_risk = (expected_return - lower_return).clamp_min(0.0)
            success_risk = 1.0 - torch.sigmoid(rollout["pred_success_logits"])
            collision_risk = torch.sigmoid(rollout["pred_collision_logits"])
            force_risk = torch.sigmoid(rollout["pred_force_violation_logits"])
            constraint_risk = torch.maximum(collision_risk, force_risk)
            infeasible = constraint_risk > cfg.safety_probability_threshold
            raw_utility_cost = (
                -expected_return
                + cfg.tail_risk_weight * tail_risk
                + cfg.constraint_risk_weight * constraint_risk
                + cfg.success_risk_weight * success_risk
                + infeasible.to(dtype) * cfg.infeasible_penalty
            )
            total = cfg.calibration_scale * raw_utility_cost + cfg.calibration_bias
        else:
            expected_return = torch.zeros(B, device=device, dtype=dtype)
            tail_risk = torch.zeros_like(expected_return)
            success_risk = torch.zeros_like(expected_return)
            constraint_risk = torch.zeros_like(expected_return)
            total = legacy_total

        return {
            "G": total,
            "G_legacy": legacy_total,
            "L_goal": goal,
            "L_safety": safety,
            "L_collab": collab,
            "U_intent": unc,
            "C_ctrl": ctrl,
            "expected_return": expected_return,
            "tail_risk": tail_risk,
            "success_risk": success_risk,
            "constraint_risk": constraint_risk,
        }

    def total_score_hypotheses(
        self,
        rollout: Dict[str, torch.Tensor],
        hypothesis_weights: torch.Tensor,
        uncertainty: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Score rollouts shaped ``[B, K, M, ...]`` and marginalize beliefs.

        ``K`` indexes ego plan candidates and ``M`` indexes teammate-plan
        hypotheses.  This method deliberately consumes only a posterior over
        hypotheses; it does not consume a privileged/true teammate plan.
        """
        actions = rollout.get("pred_actions")
        if actions is None or actions.ndim < 5:
            raise ValueError("rollout['pred_actions'] must have shape [B, K, M, H, action_dim]")
        B, K, M = actions.shape[:3]

        flat_rollout: Dict[str, torch.Tensor] = {}
        for key, value in rollout.items():
            if not torch.is_tensor(value):
                continue
            if value.ndim >= 3 and tuple(value.shape[:3]) == (B, K, M):
                flat_rollout[key] = value.reshape(B * K * M, *value.shape[3:])

        required = {"pred_actions", "pred_progress", "pred_force"}
        if not required.issubset(flat_rollout):
            missing = sorted(required.difference(flat_rollout))
            raise KeyError(f"hypothesis rollout missing required fields: {missing}")
        if "pred_contact_logits" not in flat_rollout and "pred_contact_prob" not in flat_rollout:
            raise KeyError("hypothesis rollout needs pred_contact_logits or pred_contact_prob")

        flat_uncertainty = None
        if uncertainty is not None:
            expected_shape = (B, K, M)
            try:
                uncertainty = torch.broadcast_to(
                    uncertainty.to(device=actions.device, dtype=actions.dtype),
                    expected_shape,
                )
            except RuntimeError as exc:
                raise ValueError(
                    f"uncertainty shape {tuple(uncertainty.shape)} is not broadcastable to {expected_shape}"
                ) from exc
            flat_uncertainty = uncertainty.reshape(B * K * M)

        flat_scores = self.total_score(flat_rollout, uncertainty=flat_uncertainty)
        score_grid = {key: value.reshape(B, K, M) for key, value in flat_scores.items()}
        aggregate = multi_hypothesis_expected_free_energy(score_grid["G"], hypothesis_weights)
        return {**score_grid, **aggregate}


def make_config_from_args(args) -> FreeEnergyConfig:
    return FreeEnergyConfig(
        goal_y=args.goal_y,
        force_limit=args.force_limit,
        alpha_goal=args.alpha_goal,
        alpha_safety=args.alpha_safety,
        alpha_collab=args.alpha_collab,
        alpha_unc=args.alpha_unc,
        alpha_ctrl=args.alpha_ctrl,
    )
