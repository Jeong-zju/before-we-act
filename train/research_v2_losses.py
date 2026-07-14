"""Losses for the matched-action Research-v2 training DAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from models.research_v2 import BeliefRoleTargetHeads


@dataclass(frozen=True)
class WorldLossV2Config:
    belief: float = 1.0
    progress: float = 1.0
    contact: float = 0.5
    force: float = 0.5
    step_reward: float = 1.0
    return_quantile: float = 1.0
    quantile_crossing: float = 0.1
    success: float = 0.5
    constraint: float = 0.5
    branch_ranking: float = 0.5
    rollout_consistency: float = 0.25


def belief_role_loss(
    heads: BeliefRoleTargetHeads,
    belief: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    predictions = heads(belief)
    total = belief.sum() * 0.0
    metrics: dict[str, torch.Tensor] = {}
    for name, prediction in predictions.items():
        target = targets[name].to(device=prediction.device, dtype=prediction.dtype)
        if target.shape != prediction.shape:
            raise ValueError(f"belief target {name} shape mismatch")
        if name == "maneuver":
            loss = F.cross_entropy(prediction, target.argmax(dim=-1))
        else:
            loss = F.smooth_l1_loss(prediction, target)
        total = total + loss
        metrics[f"loss_{name}"] = loss.detach()
    return {"loss": total, **metrics}


def world_model_loss_v2(
    output: Mapping[str, torch.Tensor],
    *,
    target_belief: torch.Tensor,
    target_progress: torch.Tensor,
    target_contact: torch.Tensor,
    target_force: torch.Tensor,
    target_reward: torch.Tensor,
    target_success: torch.Tensor,
    target_constraint: torch.Tensor,
    valid_mask: torch.Tensor,
    return_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    branch_group_id: torch.Tensor | None = None,
    consistency_output: Mapping[str, torch.Tensor] | None = None,
    config: WorldLossV2Config | None = None,
) -> dict[str, torch.Tensor]:
    cfg = config or WorldLossV2Config()
    mask = valid_mask.to(device=target_belief.device, dtype=target_belief.dtype)
    if mask.shape != target_progress.shape:
        raise ValueError("world valid mask/target horizon mismatch")

    def masked(value: torch.Tensor) -> torch.Tensor:
        expanded = mask
        while expanded.ndim < value.ndim:
            expanded = expanded.unsqueeze(-1)
        return (value * expanded).sum() / expanded.expand_as(value).sum().clamp_min(1.0)

    belief_loss = masked(F.smooth_l1_loss(output["future_belief"], target_belief, reduction="none"))
    progress_loss = masked(F.smooth_l1_loss(output["progress"], target_progress, reduction="none"))
    contact_loss = masked(
        F.binary_cross_entropy_with_logits(output["contact_logits"], target_contact, reduction="none")
    )
    force_loss = masked(F.smooth_l1_loss(output["force"], target_force, reduction="none"))
    reward_loss = masked(F.smooth_l1_loss(output["step_reward"], target_reward, reduction="none"))
    realized_return = (target_reward * mask).sum(dim=-1)
    quantile_predictions = output["return_quantiles"]
    levels = quantile_predictions.new_tensor(return_quantiles).reshape(1, -1)
    error = realized_return.unsqueeze(-1) - quantile_predictions
    quantile_loss = torch.maximum(levels * error, (levels - 1) * error).mean()
    crossing = F.relu(quantile_predictions[:, :-1] - quantile_predictions[:, 1:]).mean()
    success_loss = F.binary_cross_entropy_with_logits(output["success_logits"], target_success)
    constraint_loss = F.binary_cross_entropy_with_logits(
        output["constraint_logits"], target_constraint
    )
    ranking = belief_loss.new_zeros(())
    ranking_terms: list[torch.Tensor] = []
    branch_regrets: list[torch.Tensor] = []
    if branch_group_id is not None:
        # All candidates from one branch group are contiguous after flattening.
        identifiers = branch_group_id.reshape(-1)
        predicted = quantile_predictions[:, 1]
        for identifier in identifiers.unique():
            if int(identifier.item()) < 0:
                continue
            rows = identifiers == identifier
            if int(rows.sum().item()) < 2:
                continue
            target_difference = realized_return[rows][:, None] - realized_return[rows][None, :]
            predicted_difference = predicted[rows][:, None] - predicted[rows][None, :]
            valid_pairs = target_difference.abs() > 1e-4
            if valid_pairs.any():
                ranking_terms.append(F.softplus(
                    -target_difference[valid_pairs].sign() * predicted_difference[valid_pairs]
                ).mean())
            selected = predicted[rows].argmax()
            branch_regrets.append(realized_return[rows].max() - realized_return[rows][selected])
        if ranking_terms:
            ranking = torch.stack(ranking_terms).mean()
    branch_regret = (
        torch.stack(branch_regrets).mean() if branch_regrets else belief_loss.new_zeros(())
    )
    consistency = belief_loss.new_zeros(())
    if consistency_output is not None:
        consistency = masked(
            F.smooth_l1_loss(
                output["future_belief"], consistency_output["future_belief"].detach(), reduction="none"
            )
        )
    total = (
        cfg.belief * belief_loss
        + cfg.progress * progress_loss
        + cfg.contact * contact_loss
        + cfg.force * force_loss
        + cfg.step_reward * reward_loss
        + cfg.return_quantile * quantile_loss
        + cfg.quantile_crossing * crossing
        + cfg.success * success_loss
        + cfg.constraint * constraint_loss
        + cfg.branch_ranking * ranking
        + cfg.rollout_consistency * consistency
    )
    return {
        "loss": total,
        "loss_belief": belief_loss.detach(),
        "loss_progress": progress_loss.detach(),
        "loss_contact": contact_loss.detach(),
        "loss_force": force_loss.detach(),
        "loss_step_reward": reward_loss.detach(),
        "loss_return_quantile": quantile_loss.detach(),
        "loss_quantile_crossing": crossing.detach(),
        "loss_success": success_loss.detach(),
        "loss_constraint": constraint_loss.detach(),
        "loss_branch_ranking": ranking.detach(),
        "branch_regret": branch_regret.detach(),
        "loss_rollout_consistency": consistency.detach(),
    }


def plan_distribution_loss_v2(
    output: Mapping[str, torch.Tensor],
    *,
    target_code: torch.Tensor,
    target_residual: torch.Tensor,
    active_code_mask: torch.Tensor,
    diversity_actions: torch.Tensor | None = None,
    diversity_margin: float = 0.05,
) -> dict[str, torch.Tensor]:
    raw_logits = output["code_logits"]
    active = active_code_mask.to(device=raw_logits.device, dtype=torch.bool).reshape(-1)
    if active.numel() != raw_logits.shape[-1] or not active.any():
        raise ValueError("active_code_mask must select at least one output code")
    target_code = target_code.to(device=raw_logits.device, dtype=torch.long).reshape(-1)
    if target_code.numel() != raw_logits.shape[0]:
        raise ValueError("target_code batch size does not match code_logits")
    if target_code.numel() and (
        int(target_code.min().item()) < 0
        or int(target_code.max().item()) >= raw_logits.shape[-1]
    ):
        raise ValueError("target_code is outside the tokenizer codebook")

    # A support set is estimated from a bounded training scan.  A rare code can
    # therefore still occur in validation (or in a later intention batch).  It
    # must not be trained through logits that are deliberately masked to -inf:
    # cross entropy would otherwise become +inf and poison the whole run.
    valid_target = active[target_code]
    zero = raw_logits.sum() * 0.0
    if valid_target.any():
        valid_codes = target_code[valid_target]
        logits = raw_logits[valid_target].masked_fill(~active.reshape(1, -1), -torch.inf)
        code = F.cross_entropy(logits, valid_codes)
        D = target_residual.shape[-1]
        gather = valid_codes.reshape(-1, 1, 1).expand(-1, 1, D)
        mu = output["residual_mu_by_code"][valid_target].gather(1, gather).squeeze(1)
        logvar = (
            output["residual_logvar_by_code"][valid_target]
            .gather(1, gather)
            .squeeze(1)
        )
        residual_target = target_residual.to(mu.device)[valid_target]
        residual = 0.5 * (
            logvar
            + (residual_target - mu).square() / logvar.exp().clamp_min(1e-5)
        ).mean()
    else:
        code = zero
        residual = output["residual_mu_by_code"].sum() * 0.0
    diversity = code.new_zeros(())
    if diversity_actions is not None and diversity_actions.shape[1] > 1:
        flattened = diversity_actions.flatten(2)
        distance = torch.cdist(flattened, flattened)
        eye = torch.eye(distance.shape[-1], device=distance.device, dtype=torch.bool)
        off_diagonal = (~eye).unsqueeze(0).expand_as(distance)
        diversity = F.relu(diversity_margin - distance.masked_select(off_diagonal)).mean()
    return {
        "loss": code + 0.25 * residual + 0.1 * diversity,
        "loss_code": code.detach(),
        "loss_residual": residual.detach(),
        "loss_diversity": diversity.detach(),
        "active_target_fraction": valid_target.float().mean().detach(),
    }
