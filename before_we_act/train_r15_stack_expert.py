"""Low-LR Stack-only fine-tuning on original plus new motion-planner experts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import signal
import time

import torch
from torch.utils.data import DataLoader, Sampler

from before_we_act.action_generator.evolution import (
    TaskConditionedActionGenerator,
    load_r12_evolution_config,
)
from before_we_act.data.full_episode_windows import (
    FULL_EPISODE_PROTOCOL,
    FullEpisodeActionWindows,
)
from before_we_act.data.raw_team_windows import TASKS
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config
from before_we_act.train_action_generator_r4 import (
    atomic_json,
    atomic_torch_save,
    capture_rng_state,
    device_batch,
    now,
    restore_rng_state,
    robustify_source_aware_history,
    seed_everything,
    sha256,
)


TASK = "three_robots_stack_cube"
PROTOCOL = "r15_stack_original_plus_raw_success_expert_50_50_v1"


class OriginalExpertStackSampler(Sampler[list[tuple[int, int]]]):
    """Deterministic batches split between frozen and newly collected experts."""

    def __init__(
        self,
        dataset: FullEpisodeActionWindows,
        *,
        updates: int,
        batch_size: int,
        expert_rows: int,
        seed: int,
        start_update: int = 0,
    ) -> None:
        if not 0 <= start_update < updates or not 0 < expert_rows < batch_size:
            raise ValueError("invalid R15 Stack expert sampler budget")
        stack_index = TASKS.index(TASK)
        original, expert = [], []
        for request in dataset.requests_by_task[stack_index]:
            episode_index, _timestep = request
            target = (
                expert
                if "source_episode_id" in dataset.episodes[episode_index]
                else original
            )
            target.append(request)
        if not original or not expert:
            raise ValueError("R15 fine-tuning requires original and new Stack rows")
        self.original = tuple(original)
        self.expert = tuple(expert)
        self.updates = int(updates)
        self.batch_size = int(batch_size)
        self.expert_rows = int(expert_rows)
        self.seed = int(seed)
        self.start_update = int(start_update)

    def __len__(self) -> int:
        return self.updates - self.start_update

    def __iter__(self):
        original_rows = self.batch_size - self.expert_rows
        for update in range(self.start_update + 1, self.updates + 1):
            rng = random.Random(self.seed + 1_000_003 * update)
            batch = [rng.choice(self.original) for _ in range(original_rows)]
            batch.extend(rng.choice(self.expert) for _ in range(self.expert_rows))
            rng.shuffle(batch)
            yield batch


def learning_rate(update: int, updates: int, base: float, warmup: int) -> float:
    if update <= warmup:
        return base * update / warmup
    progress = (update - warmup) / max(updates - warmup, 1)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--belief-config", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--expert-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--expert-rows", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=2_500)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--resume", default="")
    parser.add_argument("--heartbeat", default="")
    args = parser.parse_args()
    if (
        args.updates < 1
        or args.batch_size < 2
        or not 0 < args.expert_rows < args.batch_size
        or args.learning_rate <= 0
        or not 1 <= args.warmup <= args.updates
        or args.workers < 0
        or args.checkpoint_every < 1
        or args.progress_every < 1
    ):
        raise ValueError("invalid R15 expert fine-tuning options")

    config_path = Path(args.config).resolve(strict=True)
    config = load_r12_evolution_config(config_path)
    if config.candidate_id != "p2":
        raise ValueError("R15 expert fine-tuning is locked to W12 ACT=P2")
    seed = 20260807
    seed_everything(seed)
    device = torch.device(args.device)
    parent_path = Path(args.parent_checkpoint).resolve(strict=True)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if (
        parent.get("round") != "R12-E1"
        or parent.get("candidate_id") != "p2"
        or int(parent.get("update", -1)) != 130_000
        or not parent.get("core_free_runtime")
    ):
        raise ValueError("R15 expert parent is not the frozen completed W12 ACT")

    belief_config = load_r11_config(args.belief_config)
    belief_path = Path(args.belief_checkpoint).resolve(strict=True)
    if sha256(belief_path) != str(config.raw["belief_checkpoint_sha256"]):
        raise ValueError("R15 expert W11 checkpoint hash differs")
    belief_saved = torch.load(belief_path, map_location="cpu", weights_only=False)
    belief = PredictiveBeliefModel(belief_config).to(device)
    belief.load_state_dict(belief_saved["model"], strict=True)
    belief.eval()
    for parameter in belief.parameters():
        parameter.requires_grad_(False)

    index_path = Path(args.expert_index).resolve(strict=True)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    extension = index.get("extension", {})
    if (
        index.get("protocol_variant") != FULL_EPISODE_PROTOCOL
        or extension.get("protocol") != "r15_raw_success_expert_direct_dinov3_v1"
        or int(extension.get("expert_episodes", 0)) < 1
    ):
        raise ValueError("R15 expert full-episode index identity differs")
    stats = {
        key: torch.as_tensor(parent["stats"][key], dtype=torch.float32)
        for key in ("a_mean", "a_std")
    }
    dataset = FullEpisodeActionWindows(
        index["episodes"], stats, split="train", cache_episodes=8
    )
    resume = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    start_update = int(resume.get("fine_tune_update", 0)) if resume else 0
    if start_update >= args.updates:
        raise ValueError("R15 expert resume is already complete")
    sampler = OriginalExpertStackSampler(
        dataset,
        updates=args.updates,
        batch_size=args.batch_size,
        expert_rows=args.expert_rows,
        seed=seed,
        start_update=start_update,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    model = TaskConditionedActionGenerator(config).to(device)
    model.set_training_stage("joint")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=float(config.training["weight_decay"]),
    )
    if resume:
        if resume.get("r15_protocol") != PROTOCOL:
            raise ValueError("R15 expert resume protocol differs")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        restore_rng_state(resume["rng_state"])
    else:
        model.load_state_dict(parent["model"], strict=True)
    model.train()

    output = Path(args.output).resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    heartbeat = Path(args.heartbeat).resolve() if args.heartbeat else None
    identity = {
        "schema_version": 1,
        "round": "R15-Evolution",
        "protocol": PROTOCOL,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": sha256(parent_path),
        "expert_index": str(index_path),
        "expert_index_sha256": sha256(index_path),
        "expert_episodes": int(extension["expert_episodes"]),
        "expert_steps": int(extension["expert_steps"]),
        "updates": args.updates,
        "batch_size": args.batch_size,
        "expert_rows_per_batch": args.expert_rows,
        "original_rows_per_batch": args.batch_size - args.expert_rows,
        "learning_rate": args.learning_rate,
        "seed": seed,
        "created_at": now(),
    }
    atomic_json(output / "training_identity.json", identity)
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    progress_path = output / "progress.jsonl"
    started = time.monotonic()
    last: dict[str, float | int] = {}

    def save(update: int, name: str) -> Path:
        path = checkpoints / name
        atomic_torch_save(
            path,
            {
                "schema_version": 1,
                "round": "R12-E1",
                "candidate_id": "p2",
                "update": int(parent["update"]) + update,
                "fine_tune_update": update,
                "stage": "r15_expert_finetune",
                "r15_protocol": PROTOCOL,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": dict(config.raw),
                "stats": parent["stats"],
                "core_free_runtime": True,
                "parent_checkpoint_sha256": identity["parent_checkpoint_sha256"],
                "expert_index_sha256": identity["expert_index_sha256"],
                "last_metrics": last,
                "rng_state": capture_rng_state(),
            },
        )
        return path

    for update, cpu_batch in enumerate(loader, start=start_update + 1):
        batch = device_batch(cpu_batch, device)
        batch["actions"], history_metrics = robustify_source_aware_history(
            batch, stats, config.training, update, seed
        )
        lr = learning_rate(update, args.updates, args.learning_rate, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            belief_state = belief(batch)["belief"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            losses = model.training_loss(
                belief_state,
                batch["spatial_tokens"],
                batch["spatial_view_mask"],
                batch["task_index"],
                batch["joint_actions"],
                batch["action_step_mask"].bool(),
            )
            total = losses["loss"]
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite R15 expert loss at {update}")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [value for value in model.parameters() if value.requires_grad],
            float(config.training["grad_clip"]),
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise FloatingPointError(f"non-finite R15 expert gradient at {update}")
        optimizer.step()
        last = {
            "fine_tune_update": update,
            "effective_update": int(parent["update"]) + update,
            "loss": float(total.detach()),
            "grad_norm": float(grad_norm),
            "learning_rate": lr,
            **history_metrics,
            **{
                key: float(value.detach())
                for key, value in losses.items()
                if key != "loss" and isinstance(value, torch.Tensor) and value.numel() == 1
            },
        }
        if update == start_update + 1 or update % args.progress_every == 0:
            elapsed = time.monotonic() - started
            completed = update - start_update
            row = {
                **last,
                "target_updates": args.updates,
                "updates_per_hour": completed / max(elapsed, 1e-6) * 3600,
                "eta_hours": (args.updates - update) * elapsed / max(completed, 1) / 3600,
                "gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30,
                "time": time.time(),
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            if heartbeat:
                atomic_json(
                    heartbeat,
                    {
                        "producer": "train_r15_stack_expert",
                        "fine_tune_update": update,
                        "target_updates": args.updates,
                        "updated_at": now(),
                    },
                )
        if update % args.checkpoint_every == 0 or update == args.updates or stopping:
            latest = save(update, "checkpoint_latest.pt")
            if update == args.updates:
                save(update, f"checkpoint_{update:06d}.pt")
            print(json.dumps({"checkpoint": str(latest), "fine_tune_update": update}), flush=True)
        if stopping:
            break


if __name__ == "__main__":
    main()
