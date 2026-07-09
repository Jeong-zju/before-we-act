from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


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

        total = (
            cfg.alpha_goal * goal
            + cfg.alpha_safety * safety
            + cfg.alpha_collab * collab
            + cfg.alpha_unc * unc
            + cfg.alpha_ctrl * ctrl
        )

        return {
            "G": total,
            "L_goal": goal,
            "L_safety": safety,
            "L_collab": collab,
            "U_intent": unc,
            "C_ctrl": ctrl,
        }


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
