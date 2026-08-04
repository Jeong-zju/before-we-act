"""Train the action-only team-role teacher for the observability audit.

The teacher never sees images, task ids, agent ids, future observations, or
environment state.  It learns whether unordered expert action chunks come from
the same synchronized team instant.  Negatives replace exactly one arm with a
time-shifted chunk from the same episode, preserving the single-arm marginals.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_act import _stats, _trajectories, seed_everything


def episode_split(task: str, path: str, key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}|{task}|{path}|{key}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "little") % 100
    return "train" if bucket < 70 else ("val" if bucket < 85 else "test")


class SynchronizedActionSets(Dataset):
    def __init__(self, trajectories, stats, split, seed, horizon=100, stride=8,
                 negative_mode="time_shift"):
        self.stats = stats
        self.horizon = horizon
        self.seed = seed
        self.negative_mode = negative_mode
        self.episodes = []
        self.by_task = defaultdict(list)
        for row in trajectories:
            path, key, n, present, task = row
            if episode_split(task, path, key, seed) != split or n < 2:
                continue
            item = (path, key, n, tuple(present), task)
            self.by_task[task].append(len(self.episodes))
            self.episodes.append(item)
        self.tasks = sorted(self.by_task)
        self.index = []
        for task in self.tasks:
            for episode_index in self.by_task[task]:
                n = self.episodes[episode_index][2]
                for t in range(0, n, stride):
                    self.index.append((episode_index, t))
        if not self.index:
            raise ValueError(f"no {split} synchronized episodes")

    def __len__(self):
        return len(self.index)

    def _chunk(self, action, t):
        chunk = np.asarray(action[t:t + self.horizon], np.float32)
        valid = len(chunk)
        padded = np.empty((self.horizon, action.shape[1]), np.float32)
        padded[:valid] = chunk
        padded[valid:] = chunk[-1]
        return (padded - self.stats["a_mean"]) / self.stats["a_std"]

    def __getitem__(self, index):
        episode_index, t = self.index[index]
        path, key, n, arms, task = self.episodes[episode_index]
        # Deterministic per-sample negative: same episode, one arm, nontrivial
        # temporal displacement.  This makes audit reruns bit-reproducible.
        rng = random.Random(self.seed * 1000003 + index)
        replaced = rng.randrange(len(arms))
        low = max(1, min(16, n // 8))
        high = max(low, min(96, max(1, n // 2)))
        delta = rng.randint(low, high)
        shifted_t = t + delta if t + delta < n else max(0, t - delta)
        positive, negative = [], []
        with h5py.File(path, "r") as h5:
            trajectory = h5[key]
            for arm_index, arm in enumerate(arms):
                action = trajectory["actions"][f"panda-{arm}"]
                local = self._chunk(action, t)
                positive.append(local)
                negative.append(self._chunk(action, shifted_t) if arm_index == replaced else local)
        action_dim = positive[0].shape[-1]
        pos = np.zeros((4, self.horizon, action_dim), np.float32)
        neg = np.zeros_like(pos)
        mask = np.zeros(4, np.bool_)
        pos[:len(arms)] = positive
        neg[:len(arms)] = negative
        use_duplicate = self.negative_mode == "role_duplicate" or (
            self.negative_mode == "mixed" and rng.random() < .5)
        if use_duplicate:
            neg[replaced] = pos[(replaced + 1) % len(arms)]
        mask[:len(arms)] = True
        return torch.from_numpy(pos), torch.from_numpy(neg), torch.from_numpy(mask)


class TeamRoleTeacher(nn.Module):
    def __init__(self, action_dim, width=192, role_dim=64):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Conv1d(action_dim, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.role = nn.Sequential(
            nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU(),
            nn.Linear(width, role_dim),
        )
        self.q = nn.Linear(role_dim, role_dim, bias=False)
        self.k = nn.Linear(role_dim, role_dim, bias=False)

    def forward(self, actions, mask):
        batch, agents, horizon, action_dim = actions.shape
        encoded = self.action_encoder(
            actions.reshape(batch * agents, horizon, action_dim).transpose(1, 2)
        ).squeeze(-1).reshape(batch, agents, -1)
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        total = (encoded * weights).sum(1, keepdim=True)
        peers = (total - encoded * weights) / (weights.sum(1, keepdim=True) - weights).clamp_min(1.0)
        role = F.normalize(self.role(torch.cat((encoded, peers), -1)), dim=-1)
        q, k = self.q(role), self.k(role)
        pair = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(q.shape[-1])
        valid = mask.unsqueeze(1) & mask.unsqueeze(2)
        valid &= ~torch.eye(agents, dtype=torch.bool, device=mask.device).unsqueeze(0)
        score = (pair * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
        return score, role


def auc(labels, scores):
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    positive = labels.bool()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); positive_scores, negative_scores = [], []
    for pos, neg, mask in loader:
        pos, neg, mask = pos.to(device), neg.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p, _ = model(pos, mask); n, _ = model(neg, mask)
        positive_scores.append(p.float().cpu())
        negative_scores.append(n.float().cpu())
    positive_scores = torch.cat(positive_scores)
    negative_scores = torch.cat(negative_scores)
    scores = torch.cat((positive_scores, negative_scores))
    labels = torch.cat((torch.ones_like(positive_scores), torch.zeros_like(negative_scores)))
    return {"auc": auc(labels, scores), "positive_mean": float(positive_scores.mean()),
            "negative_mean": float(negative_scores.mean()),
            "margin": float(positive_scores.mean() - negative_scores.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--negative-mode", choices=("time_shift", "role_duplicate", "mixed"),
                        default="time_shift")
    parser.add_argument("--checkpoint", default=None,
                        help="Evaluate an existing teacher checkpoint with corrected metrics")
    args = parser.parse_args()
    seed_everything(args.seed)
    paths = sorted({p for pattern in args.data.split(",") for p in glob.glob(pattern)})
    trajectories = _trajectories(paths, (0, 1, 2, 3))
    stats = torch.load(args.normalization, map_location="cpu", weights_only=False)["stats"]
    datasets = {split: SynchronizedActionSets(trajectories, stats, split, args.seed,
                                               negative_mode=args.negative_mode)
                for split in ("train", "val", "test")}
    loaders = {split: DataLoader(data, batch_size=args.batch_size, shuffle=(split == "train"),
                                 num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
               for split, data in datasets.items()}
    sample = datasets["train"][0][0]
    device = torch.device("cuda")
    model = TeamRoleTeacher(sample.shape[-1]).to(device)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        metrics = {split: evaluate(model, loaders[split], device) for split in ("val", "test")}
        (out / "corrected_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps({"complete": True, "corrected_metrics": metrics}), flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    split_manifest = {
        split: [{"path": p, "key": k, "task": task, "arms": list(arms), "length": n}
                for p, k, n, arms, task in data.episodes]
        for split, data in datasets.items()
    }
    (out / "episode_split.json").write_text(json.dumps(split_manifest, indent=2))
    iterator = iter(loaders["train"]); running = 0.0
    for update in range(1, args.updates + 1):
        try: pos, neg, mask = next(iterator)
        except StopIteration:
            iterator = iter(loaders["train"]); pos, neg, mask = next(iterator)
        pos, neg, mask = pos.to(device, non_blocking=True), neg.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p, _ = model(pos, mask); n, _ = model(neg, mask)
            loss = F.softplus(-p).mean() + F.softplus(n).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        running += float(loss.detach())
        if update % 250 == 0 or update == args.updates:
            metrics = evaluate(model, loaders["val"], device)
            row = {"update": update, "train_loss": running / 250, "val": metrics}
            print(json.dumps(row), flush=True); running = 0.0
    metrics = {split: evaluate(model, loaders[split], device) for split in ("val", "test")}
    torch.save({"model": model.state_dict(), "stats": stats, "metrics": metrics,
                "config": vars(args), "split_manifest": split_manifest}, out / "teacher.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"complete": True, "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
