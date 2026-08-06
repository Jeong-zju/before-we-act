#!/usr/bin/env python3
"""Fail-closed W11-vs-spatial representation sufficiency probe for R12-R3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from before_we_act.data.action_windows import CachedActionWindows
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config


MODES = ("w11", "spatial", "fused")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ProbeHead(nn.Module):
    """Same-capacity attention readout for each frozen representation."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        width = 96
        if mode in ("spatial", "fused"):
            self.spatial_norm = nn.LayerNorm(768)
            self.spatial_projection = nn.Linear(768, width)
            self.view_embedding = nn.Parameter(torch.randn(5, width) * 0.02)
            self.row_embedding = nn.Parameter(torch.randn(4, width) * 0.02)
            self.column_embedding = nn.Parameter(torch.randn(4, width) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pool = nn.MultiheadAttention(width, 4, batch_first=True)
        self.readout = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, 33),
        )

    def forward(self, batch: dict[str, torch.Tensor]):
        token_groups, mask_groups = [], []
        if self.mode in ("w11", "fused"):
            token_groups.append(batch["belief_tokens"])
            mask_groups.append(batch["belief_mask"].bool())
        if self.mode in ("spatial", "fused"):
            spatial = self.spatial_projection(self.spatial_norm(batch["spatial_tokens"]))
            position = (
                self.view_embedding[:, None]
                + self.row_embedding[:, None]
                .expand(-1, 4, -1)
                .reshape(1, 16, -1)
                + self.column_embedding[None]
                .expand(4, -1, -1)
                .reshape(1, 16, -1)
            )
            spatial = (spatial + position[None]).reshape(len(spatial), 80, 96)
            spatial_mask = batch["spatial_view_mask"].bool()[:, :, None].expand(
                -1, -1, 16
            ).reshape(len(spatial), 80)
            token_groups.append(spatial)
            mask_groups.append(spatial_mask)
        tokens = torch.cat(token_groups, dim=1)
        mask = torch.cat(mask_groups, dim=1)
        pooled, _ = self.pool(
            self.query.expand(len(tokens), -1, -1),
            tokens,
            tokens,
            key_padding_mask=~mask,
            need_weights=False,
        )
        output = self.readout(pooled[:, 0])
        return output[:, :32].reshape(-1, 4, 8), output[:, 32].sigmoid()


@torch.inference_mode()
def belief_tokens(model, dataset, device, batch_size: int):
    values, masks = [], []
    data = dataset.data
    for start in range(0, len(dataset), batch_size):
        stop = min(len(dataset), start + batch_size)
        batch = {
            key: data[key][start:stop].to(device)
            for key in ("visual", "view_mask", "qpos", "actions", "agent_mask")
        }
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            belief = model(batch)["belief"]
        tokens = torch.cat(
            [belief.tokens, belief.agent_tokens, belief.consensus_token[:, None]], dim=1
        )
        mask = torch.cat(
            [
                torch.ones(belief.tokens.shape[:2], device=device, dtype=torch.bool),
                belief.agent_mask,
                torch.ones((len(tokens), 1), device=device, dtype=torch.bool),
            ],
            dim=1,
        )
        values.append(tokens.to(device="cpu", dtype=torch.float16))
        masks.append(mask.cpu())
    return torch.cat(values), torch.cat(masks)


def frozen_split(dataset, belief_values, belief_masks):
    return {
        "belief_tokens": belief_values,
        "belief_mask": belief_masks,
        "spatial_tokens": dataset.spatial_data["spatial_tokens"],
        "spatial_view_mask": dataset.spatial_data["spatial_view_mask"],
        "target_action": dataset.data["joint_actions"][:, 0],
        "agent_mask": dataset.data["agent_mask"],
        "progress": dataset.spatial_data["progress"],
    }


def select(values, indices, device):
    return {key: value.index_select(0, indices).to(device) for key, value in values.items()}


def action_error(prediction, target, agent_mask):
    mask = agent_mask[:, :, None].expand_as(target)
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1)


def train_probe(mode, train, device, updates, batch_size, seed):
    seed_everything(seed)
    model = ProbeHead(mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    model.train()
    last = {}
    for update in range(1, updates + 1):
        indices = torch.randint(len(train["progress"]), (batch_size,), generator=generator)
        batch = select(train, indices, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            action, progress = model(batch)
            action_loss = action_error(action, batch["target_action"], batch["agent_mask"])
            progress_loss = (progress - batch["progress"]).square().mean()
            loss = action_loss + 0.1 * progress_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % 500 == 0:
            last = {
                "update": update,
                "loss": float(loss.detach()),
                "action_mse": float(action_loss.detach()),
                "progress_mse": float(progress_loss.detach()),
            }
            print(json.dumps({"mode": mode, **last}, sort_keys=True), flush=True)
    return model.eval(), last


@torch.inference_mode()
def evaluate_probe(model, values, device, batch_size, spatial_permutation=None):
    action_squared = 0.0
    action_count = 0
    progress_values, progress_targets = [], []
    for start in range(0, len(values["progress"]), batch_size):
        indices = torch.arange(start, min(len(values["progress"]), start + batch_size))
        batch = select(values, indices, device)
        if spatial_permutation is not None:
            shuffled = spatial_permutation.index_select(0, indices)
            batch["spatial_tokens"] = values["spatial_tokens"].index_select(0, shuffled).to(device)
            batch["spatial_view_mask"] = values["spatial_view_mask"].index_select(0, shuffled).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            action, progress = model(batch)
        mask = batch["agent_mask"][:, :, None].expand_as(action)
        action_squared += float(((action - batch["target_action"]).square() * mask).sum())
        action_count += int(mask.sum())
        progress_values.append(progress.float().cpu())
        progress_targets.append(batch["progress"].float().cpu())
    prediction = torch.cat(progress_values)
    target = torch.cat(progress_targets)
    mse = float((prediction - target).square().mean())
    variance = float((target - target.mean()).square().mean())
    return {
        "action_mse": action_squared / max(action_count, 1),
        "progress_mse": mse,
        "progress_r2": 1.0 - mse / max(variance, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-cache", required=True)
    parser.add_argument("--spatial-cache", required=True)
    parser.add_argument("--belief-config", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 1:
        raise ValueError("probe updates and batch size must be positive")
    seed = 20260806
    seed_everything(seed)
    device = torch.device(args.device)
    datasets = {
        split: CachedActionWindows(
            args.action_cache, split, spatial_cache_path=args.spatial_cache
        )
        for split in ("train", "validation")
    }
    belief_config = load_r11_config(args.belief_config)
    saved = torch.load(args.belief_checkpoint, map_location="cpu", weights_only=False)
    belief = PredictiveBeliefModel(belief_config).to(device).eval()
    belief.load_state_dict(saved["model"], strict=True)
    for parameter in belief.parameters():
        parameter.requires_grad_(False)
    frozen = {}
    for split, dataset in datasets.items():
        tokens, masks = belief_tokens(belief, dataset, device, args.batch_size)
        frozen[split] = frozen_split(dataset, tokens, masks)
    metrics, training = {}, {}
    fused_model = None
    for offset, mode in enumerate(MODES):
        model, last = train_probe(
            mode,
            frozen["train"],
            device,
            args.updates,
            args.batch_size,
            seed + offset,
        )
        metrics[mode] = evaluate_probe(model, frozen["validation"], device, args.batch_size)
        training[mode] = last
        if mode == "fused":
            fused_model = model
    permutation = torch.randperm(
        len(frozen["validation"]["progress"]),
        generator=torch.Generator().manual_seed(seed + 999),
    )
    shuffled = evaluate_probe(
        fused_model,
        frozen["validation"],
        device,
        args.batch_size,
        spatial_permutation=permutation,
    )
    improvement = 1.0 - metrics["fused"]["action_mse"] / metrics["w11"]["action_mse"]
    shuffle_degradation = shuffled["action_mse"] / metrics["fused"]["action_mse"] - 1.0
    checks = {
        "fused_action_mse_improves_w11_by_at_least_10pct": improvement >= 0.10,
        "fused_progress_r2_not_worse_than_w11_by_over_0p02": metrics["fused"]["progress_r2"] >= metrics["w11"]["progress_r2"] - 0.02,
        "spatial_shuffle_degrades_fused_action_mse_by_at_least_5pct": shuffle_degradation >= 0.05,
        "all_metrics_finite": all(
            np.isfinite(value)
            for group in (*metrics.values(), shuffled)
            for value in group.values()
        ),
    }
    result = {
        "schema_version": 1,
        "stage": "R12-R3-representation-sufficiency",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": seed,
        "updates_per_probe": args.updates,
        "batch_size": args.batch_size,
        "metrics": metrics,
        "fused_spatial_shuffle": shuffled,
        "fused_action_mse_relative_improvement": improvement,
        "spatial_shuffle_relative_degradation": shuffle_degradation,
        "checks": checks,
        "passed": all(checks.values()),
        "training_terminal": training,
        "claim_boundary": "diagnostic prerequisite only; does not replace the mandatory 5x20 closed-loop Gate20",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result | {"training_terminal": "saved"}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 10)


if __name__ == "__main__":
    main()
