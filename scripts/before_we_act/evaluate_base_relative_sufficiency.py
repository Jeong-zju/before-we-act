#!/usr/bin/env python3
"""Train matched-capacity (C,B) and (C,H) residual probes for Step 4."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from itertools import chain

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from before_we_act.action_grounded_belief import (
    load_split,
    split_by_episode_key,
)
from before_we_act.base_relative_belief import BaseRelativeBeliefExperiment
from before_we_act.predictive_team_belief_data import (
    PairedSituationBatchSampler,
    PredictiveTeamBeliefDataset,
)
from before_we_act.temporal_history_data import SIX_TASKS, sha256_file
from before_we_act.train_predictive_team_belief import (
    atomic_json,
    config_from_contract,
    device_batch,
    fixed_loader,
)


class MatchedResidualProbe(nn.Module):
    """One identical-capacity residual head for either B or raw legal H."""

    def __init__(
        self, memory_input_dim: int, d_model: int, action_dim: int, heads: int
    ) -> None:
        super().__init__()
        self.memory_adapter = nn.Sequential(
            nn.LayerNorm(memory_input_dim),
            nn.Linear(memory_input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True, bias=False
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.output = nn.Linear(d_model, action_dim)

    def forward(
        self,
        context: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        if memory_mask.shape != memory.shape[:2] or memory_mask.dtype != torch.bool:
            raise ValueError("probe memory mask must be boolean [batch,token]")
        value = self.memory_norm(self.memory_adapter(memory))
        query = self.query_norm(context)
        attended = self.cross_attention(
            query,
            value,
            value,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )[0]
        fused = self.fusion(torch.cat((query, attended), dim=-1))
        return self.output(fused)


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    rows = (prediction - target).float().square().mean(-1)
    return (rows * mask).sum() / mask.sum().clamp_min(1)


def memory_inputs(batch: dict, belief: torch.Tensor) -> tuple:
    visual = batch["runtime_visual_tokens"].squeeze(-2).mean(2)
    task = batch["task_token"].unsqueeze(1).expand(-1, visual.shape[1], -1)
    history = torch.cat(
        (
            visual,
            batch["history_qpos"],
            batch["history_action"],
            task,
        ),
        dim=-1,
    )
    if belief.shape[:2] != history.shape[:2]:
        raise ValueError("probe requires 16 B tokens and 16 legal-history slots")
    padded_belief = torch.nn.functional.pad(
        belief, (0, history.shape[-1] - belief.shape[-1])
    )
    belief_mask = torch.ones(
        belief.shape[:2], dtype=torch.bool, device=belief.device
    )
    return padded_belief, belief_mask, history, batch["history_mask"]


@torch.no_grad()
def frozen_belief(model, batch):
    return model.belief_core(
        batch["runtime_visual_tokens"],
        batch["runtime_visual_mask"],
        batch["history_qpos"],
        batch["history_action"],
        batch["history_mask"],
        batch["action_history_mask"],
        batch["task_token"],
        batch["episode_reset_mask"],
        future_action=batch["action"],
        future_action_mask=batch["action_mask"],
    ).mu


@torch.no_grad()
def evaluate(probe_b, probe_h, model, loader, device) -> dict:
    probe_b.eval()
    probe_h.eval()
    totals = {"cb": [0.0, 0], "ch": [0.0, 0]}
    by_task = {
        name: {task: [0.0, 0] for task in range(len(SIX_TASKS))}
        for name in totals
    }
    for raw in loader:
        batch = device_batch(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            belief = frozen_belief(model, batch)
            b, b_mask, h, h_mask = memory_inputs(batch, belief)
            predictions = {
                "cb": probe_b(batch["decoded_action_hidden"], b, b_mask),
                "ch": probe_h(batch["decoded_action_hidden"], h, h_mask),
            }
        target = batch["action"] - batch["base_action"]
        row_mask = batch["action_mask"]
        for name, prediction in predictions.items():
            squared = (prediction - target).float().square().mean(-1)
            totals[name][0] += float((squared * row_mask).sum().cpu())
            totals[name][1] += int(row_mask.sum().cpu())
            for task in range(len(SIX_TASKS)):
                active = row_mask & (batch["task_index"] == task).unsqueeze(-1)
                by_task[name][task][0] += float((squared * active).sum().cpu())
                by_task[name][task][1] += int(active.sum().cpu())
    mse = {
        name: value[0] / max(value[1], 1) for name, value in totals.items()
    }
    return {
        "residual_mse": mse,
        "g_suf": (mse["cb"] - mse["ch"]) / max(mse["ch"], 1e-12),
        "per_task": {
            name: {
                SIX_TASKS[task]: value[0] / max(value[1], 1)
                for task, value in tasks.items()
            }
            for name, tasks in by_task.items()
        },
        "target": "G_suf <= 0.05 is soft evidence for approximate sufficiency",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--action-context-cache", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.updates < 1:
        raise ValueError("probe updates must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = config_from_contract(contract)
    model = BaseRelativeBeliefExperiment(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)

    dataset = PredictiveTeamBeliefDataset(args.cache, args.action_context_cache)
    split = split_by_episode_key(load_split(args.scenario_split))
    sampler = PairedSituationBatchSampler(
        dataset.episodes,
        split,
        updates=args.updates,
        data_seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    validation = fixed_loader(dataset, split, "validation")
    memory_dim = config.vision_dim + config.state_dim + config.action_dim + config.d_model
    probe_b = MatchedResidualProbe(
        memory_dim, config.d_model, config.action_dim, config.heads
    ).to(device)
    probe_h = MatchedResidualProbe(
        memory_dim, config.d_model, config.action_dim, config.heads
    ).to(device)
    probe_h.load_state_dict(probe_b.state_dict())
    parameters_b = sum(parameter.numel() for parameter in probe_b.parameters())
    parameters_h = sum(parameter.numel() for parameter in probe_h.parameters())
    if parameters_b != parameters_h:
        raise RuntimeError("control-sufficiency probes are not capacity matched")
    optimizer = torch.optim.AdamW(
        chain(probe_b.parameters(), probe_h.parameters()),
        lr=2e-4,
        weight_decay=1e-4,
    )
    last = {}
    for update, raw in enumerate(loader, start=1):
        batch = device_batch(raw, device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            belief = frozen_belief(model, batch)
            b, b_mask, h, h_mask = memory_inputs(batch, belief)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            residual_b = probe_b(batch["decoded_action_hidden"], b, b_mask)
            residual_h = probe_h(batch["decoded_action_hidden"], h, h_mask)
            target = batch["action"] - batch["base_action"]
            loss_b = masked_mse(residual_b, target, batch["action_mask"])
            loss_h = masked_mse(residual_h, target, batch["action_mask"])
            loss = loss_b + loss_h
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite sufficiency-probe loss at {update}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            chain(probe_b.parameters(), probe_h.parameters()), 1.0
        )
        if not torch.isfinite(gradient):
            raise FloatingPointError(
                f"non-finite sufficiency-probe gradient at {update}"
            )
        optimizer.step()
        last = {
            "update": update,
            "cb": float(loss_b.detach()),
            "ch": float(loss_h.detach()),
            "gradient_norm": float(gradient),
        }
        if update == 1 or update % 500 == 0 or update == args.updates:
            print(json.dumps(last, sort_keys=True), flush=True)

    result = evaluate(probe_b, probe_h, model, validation, device)
    payload = {
        "format_version": "before-we-act.a4-control-sufficiency/1",
        "status": "PASSED_SOFT_TARGET" if result["g_suf"] <= 0.05 else "MISSED_SOFT_TARGET",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "contract_sha256": sha256_file(args.contract),
        "probe_seed": args.seed,
        "probe_updates": args.updates,
        "probe_parameters_each": parameters_b,
        "same_initialization": True,
        "same_batches_optimizer_and_budget": True,
        "last_train": last,
        "validation": result,
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
