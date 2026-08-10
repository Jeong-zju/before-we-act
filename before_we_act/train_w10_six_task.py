"""Train the frozen W10 no-wrist Stereo-CoRE architecture on six tasks.

The model and optimization objective remain the W10 implementation. The data
adapter expands the corpus to the post-Stack six-task portfolio. ``place_food``
contains only a global fixed camera, so its missing local-camera input is
explicitly filled with that same legal global RGB frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from no_wrist_pair_model import NoWristPAIRRoute
from train_act import seed_everything

SIX_TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)


def load_episodes(manifest_paths: list[Path], split: str = "train") -> list[dict]:
    episodes = []
    seen_tasks = set()
    for manifest_path in manifest_paths:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        task = manifest["task"]["id"]
        if task not in SIX_TASKS:
            raise ValueError(f"unsupported task in {manifest_path}: {task}")
        seen_tasks.add(task)
        arm_count = int(manifest["action"]["dimension"]) // 8
        for row in manifest["episodes"]:
            if row["split"] != split:
                continue
            path = manifest_path.parent / row["hdf5_path"]
            episodes.append(
                {
                    "path": str(path),
                    "task": task,
                    "arms": tuple(range(arm_count)),
                    "length": int(row["steps"]),
                    "seed": int(row["seed"]),
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    if seen_tasks != set(SIX_TASKS):
        raise ValueError(f"expected all six tasks, got {sorted(seen_tasks)}")
    missing = [episode["path"] for episode in episodes if not Path(episode["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} HDF5 files; first={missing[0]}")
    return episodes


def compute_stats(episodes: list[dict]) -> dict[str, np.ndarray]:
    qposes, actions = [], []
    for index, episode in enumerate(episodes, 1):
        with h5py.File(episode["path"], "r") as handle:
            data = handle["data"]
            for arm in episode["arms"]:
                qposes.append(
                    np.asarray(
                        data["observation"]["agents"][f"panda_{arm}"]["qpos"][
                            : episode["length"]
                        ],
                        dtype=np.float32,
                    )
                )
                actions.append(
                    np.asarray(
                        data["action"]["agents"][f"panda_{arm}"]["commanded"][
                            : episode["length"]
                        ],
                        dtype=np.float32,
                    )
                )
        if index % 50 == 0:
            print(json.dumps({"stats_episodes": index, "total": len(episodes)}), flush=True)
    qpos = np.concatenate(qposes)
    action = np.concatenate(actions)
    return {
        "q_mean": qpos.mean(0).astype(np.float32),
        "q_std": qpos.std(0).clip(1e-4).astype(np.float32),
        "a_mean": action.mean(0).astype(np.float32),
        "a_std": action.std(0).clip(1e-4).astype(np.float32),
    }


class NoWristFrameDataset(Dataset):
    def __init__(self, episodes: list[dict], horizon: int, stats: dict[str, np.ndarray]):
        self.episodes = episodes
        self.horizon = horizon
        self.stats = stats

    def __len__(self) -> int:
        return sum(e["length"] * len(e["arms"]) for e in self.episodes)

    def __getitem__(self, request):
        episode_index, arm, time_index = request
        episode = self.episodes[episode_index]
        with h5py.File(episode["path"], "r") as handle:
            data = handle["data"]
            images = data["observation"]["images"]
            global_rgb = np.asarray(images["global"][time_index])
            local_key = f"agent_{arm}"
            local_rgb = np.asarray(
                images[local_key if local_key in images else "global"][time_index]
            )
            qpos = np.asarray(
                data["observation"]["agents"][f"panda_{arm}"]["qpos"][time_index],
                dtype=np.float32,
            )
            action_data = data["action"]["agents"][f"panda_{arm}"]["commanded"]
            end = min(time_index + self.horizon, episode["length"])
            future = np.asarray(action_data[time_index:end], dtype=np.float32)
        if global_rgb.shape != (480, 640, 3) or local_rgb.shape != (480, 640, 3):
            raise ValueError(
                f"strict 640x480 RGB required in {episode['path']}: "
                f"global={global_rgb.shape}, local={local_rgb.shape}"
            )
        valid = len(future)
        padded = np.empty((self.horizon, future.shape[1]), dtype=np.float32)
        padded[:valid] = future
        padded[valid:] = future[-1]
        mask = np.zeros(self.horizon, dtype=np.bool_)
        mask[:valid] = True
        return (
            torch.from_numpy(global_rgb).permute(2, 0, 1).contiguous(),
            torch.from_numpy(local_rgb).permute(2, 0, 1).contiguous(),
            torch.from_numpy((qpos - self.stats["q_mean"]) / self.stats["q_std"]),
            torch.from_numpy((padded - self.stats["a_mean"]) / self.stats["a_std"]),
            torch.from_numpy(mask),
        )


class ExactSixTaskBatchSampler(Sampler):
    """Eight local samples per task, deterministic and exactly resumable."""

    def __init__(self, episodes: list[dict], updates: int, seed: int, start_update: int = 0):
        self.episodes = episodes
        self.updates = updates
        self.seed = seed
        self.start_update = start_update
        self.by_task = defaultdict(list)
        for index, episode in enumerate(episodes):
            self.by_task[episode["task"]].append(index)
        if set(self.by_task) != set(SIX_TASKS):
            raise ValueError(f"expected six task buckets, got {sorted(self.by_task)}")

    def __len__(self) -> int:
        return self.updates - self.start_update

    def __iter__(self):
        for update in range(self.start_update + 1, self.updates + 1):
            rng = random.Random(self.seed + 1_000_003 * update)
            batch = []
            for task in SIX_TASKS:
                candidates = self.by_task[task]
                for _ in range(8):
                    episode_index = candidates[rng.randrange(len(candidates))]
                    episode = self.episodes[episode_index]
                    arm = episode["arms"][rng.randrange(len(episode["arms"]))]
                    time_index = rng.randrange(episode["length"])
                    batch.append((episode_index, arm, time_index))
            rng.shuffle(batch)
            yield batch


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--updates", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--router-lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--capability-weight", type=float, default=0.05)
    parser.add_argument("--counterfactual-every", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--milestones", default="20000,40000,60000,80000,100000,120000")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--resume", default="")
    parser.add_argument("--allow-preflight", action="store_true")
    args = parser.parse_args()

    if args.batch_size != 48:
        raise ValueError("six-task W10 protocol requires exact batch size 48")
    if not args.allow_preflight and args.updates != 120_000:
        raise ValueError("formal protocol requires 120000 optimizer updates")
    if not Path(args.dino_model).is_dir():
        raise FileNotFoundError(args.dino_model)

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifests = [Path(item).resolve() for item in args.manifests]
    episodes = load_episodes(manifests)

    resume_path = Path(args.resume) if args.resume else None
    saved = None
    if resume_path and resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        stats = saved["stats"]
        print(json.dumps({"resuming": str(resume_path), "update": saved["update"]}), flush=True)
    else:
        stats = compute_stats(episodes)

    if saved and int(saved["update"]) >= args.updates:
        complete = {
            "status": "PASSED",
            "stage": "complete",
            "model": "W10 NoWristPAIRRoute",
            "tasks": list(SIX_TASKS),
            "complete": True,
            "update": int(saved["update"]),
            "target_updates": args.updates,
            "completed_at_epoch": time.time(),
            "last_metrics": saved.get("last_metrics", {}),
            "resumed_complete_checkpoint": str(resume_path.resolve()),
        }
        atomic_json_save(complete, output / "status.json")
        print(json.dumps(complete), flush=True)
        return

    atomic_json_save(
        {
            "status": "TRAINING",
            "stage": "formal",
            "model": "W10 NoWristPAIRRoute",
            "tasks": list(SIX_TASKS),
            "update": int(saved["update"]) if saved else 0,
            "target_updates": args.updates,
            "started_at_epoch": time.time(),
        },
        output / "status.json",
    )

    dataset = NoWristFrameDataset(episodes, horizon=100, stats=stats)
    start_update = int(saved["update"]) if saved else 0
    sampler = ExactSixTaskBatchSampler(episodes, args.updates, args.seed, start_update)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    model = NoWristPAIRRoute(
        9,
        8,
        horizon=100,
        d_model=384,
        enc_layers=4,
        dec_layers=7,
        roles=4,
        role_rank=32,
        dino_model=args.dino_model,
    ).to(device)

    router_prefix = (
        "compatibility",
        "role_prototypes",
        "route_state",
        "route_observation",
        "route_mlp",
    )
    router, body = [], []
    for name, parameter in model.named_parameters():
        (router if name.startswith(router_prefix) else body).append(parameter)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": body, "lr": args.lr}, {"params": router, "lr": args.router_lr}],
        weight_decay=1e-4,
    )

    def multiplier(step: int) -> float:
        warmup = min(1.0, (step + 1) / max(args.warmup, 1))
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, (step + 1) / args.updates)))
        return warmup * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if "rng" in saved:
            random.setstate(saved["rng"]["python"])
            np.random.set_state(saved["rng"]["numpy"])
            torch.set_rng_state(saved["rng"]["torch"])
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])

    sources = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in manifests
    }
    config = vars(args) | {
        "policy_variant": "no_wrist_rgb_pair_route",
        "state_dim": 9,
        "action_dim": 8,
        "horizon": 100,
        "d_model": 384,
        "enc_layers": 4,
        "dec_layers": 7,
        "roles": 4,
        "role_rank": 32,
        "camera_width": 640,
        "camera_height": 480,
        "patch_grid": [30, 40],
        "fusion_layers": 2,
        "vision_backbone": "dual_dinov3_vitb16_frozen",
        "policy_input": "current global fixed RGB + matching agent fixed RGB + own qpos",
        "excluded_inputs": "wrist RGB/depth, task ID, agent ID, language, peer state/action",
        "tasks": list(SIX_TASKS),
        "task_sampling": "exactly 8 hierarchical episode/arm/time samples per task per batch (48 total)",
        "local_camera_fallback": {
            "place_food": "reuse current global fixed RGB for the missing agent camera",
        },
        "training_manifests": sources,
        "train_episodes": len(episodes),
        "sample_budget": args.batch_size * args.updates,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))
    atomic_torch_save({"stats": stats}, output / "normalization.pt")

    milestones = {int(item) for item in args.milestones.split(",") if item}
    started = time.time()
    capability_reference = saved.get("capability_reference") if saved else None
    last = saved.get("last_metrics", {}) if saved else {}
    for update, batch in enumerate(loader, start=start_update + 1):
        global_rgb, local_rgb, qpos, actions, mask = [
            item.to(device, non_blocking=True) for item in batch
        ]
        global_rgb = global_rgb.float().div_(255)
        local_rgb = local_rgb.float().div_(255)
        do_counterfactual = update % args.counterfactual_every == 0
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, mu, logvar, _aux, routes, counterfactual, target = model(
                global_rgb,
                local_rgb,
                qpos,
                actions,
                return_routing=True,
                counterfactual=do_counterfactual,
            )
            action = ((prediction - actions).square().mean(-1) * mask).sum() / mask.sum().clamp_min(1)
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
            coupling = routes.sum() * 0.0
            if do_counterfactual:
                errors = (counterfactual - target.unsqueeze(2)).square().mean(-1)
                temperature = errors.detach().std(-1, keepdim=True).clamp_min(1e-3)
                capability_target = (-errors.detach() / temperature).softmax(-1)
                coupling = F.kl_div(
                    routes[:1].clamp_min(1e-8).log(),
                    capability_target,
                    reduction="none",
                ).sum(-1).mean()
            loss = action + args.beta * kl + args.capability_weight * coupling
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        last = {
            "loss": float(loss.detach()),
            "action": float(action.detach()),
            "kl": float(kl.detach()),
            "coupling": float(coupling.detach()),
            "route_entropy": float(
                (-(routes.clamp_min(1e-8).log() * routes).sum(-1).mean()).detach()
            ),
        }
        if update == start_update + 1 or update % args.log_every == 0:
            elapsed = time.time() - started
            completed = update - start_update
            progress = {
                "update": update,
                "target_updates": args.updates,
                **last,
                "updates_per_hour": completed / elapsed * 3600,
                "eta_hours": (args.updates - update) * elapsed / completed / 3600,
                "gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "updated_at_epoch": time.time(),
            }
            print(json.dumps(progress), flush=True)
            with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(progress, sort_keys=True) + "\n")
            atomic_json_save(
                {
                    "status": "TRAINING",
                    "stage": "formal",
                    "model": "W10 NoWristPAIRRoute",
                    "tasks": list(SIX_TASKS),
                    **progress,
                },
                output / "status.json",
            )
        if update % args.save_every == 0 or update in milestones or update == args.updates:
            payload = {
                "model": model.state_dict(),
                "stats": stats,
                "config": config,
                "update": update,
                "capability_reference": capability_reference,
                "last_metrics": last,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                },
            }
            atomic_torch_save(payload, output / "checkpoint_latest.pt")
            if update in milestones or update == args.updates:
                atomic_torch_save(payload, output / f"checkpoint_{update:06d}.pt")
            print(json.dumps({"saved_update": update}), flush=True)
    complete = {
        "status": "PASSED",
        "stage": "complete",
        "model": "W10 NoWristPAIRRoute",
        "tasks": list(SIX_TASKS),
        "complete": True,
        "update": args.updates,
        "target_updates": args.updates,
        "completed_at_epoch": time.time(),
        "last_metrics": last,
    }
    atomic_json_save(complete, output / "status.json")
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
