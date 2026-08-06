"""Train one R12-E1 task-conditioned, image-primary action specialist."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import DataLoader

from before_we_act.action_generator.evolution import (
    TaskConditionedActionGenerator,
    load_r12_evolution_config,
)
from before_we_act.action_generator.r4_base import load_r3_core_warm_start
from before_we_act.data.full_episode_windows import (
    FULL_EPISODE_PROTOCOL,
    FullEpisodeActionWindows,
    TaskWeightedFullEpisodeSampler,
)
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config
from before_we_act.train_action_generator_r4 import (
    atomic_json,
    atomic_torch_save,
    build_optimizer,
    capture_rng_state,
    device_batch,
    now,
    restore_rng_state,
    robustify_source_aware_history,
    seed_everything,
    sha256,
)


EXPECTED_TRAIN = {
    "lift_barrier": 8255,
    "camera_alignment": 11764,
    "three_robots_stack_cube": 48892,
    "long_pipeline_delivery": 88493,
    "take_photo": 23044,
}
EXPECTED_VALIDATION = {
    "lift_barrier": 1015,
    "camera_alignment": 1457,
    "three_robots_stack_cube": 6138,
    "long_pipeline_delivery": 10981,
    "take_photo": 2884,
}


def learning_rate(training, update: int) -> float:
    bridge = int(training["bridge_updates"])
    if update <= bridge:
        return float(training["learning_rate"])
    joint_update = update - bridge
    warmup = int(training["joint_warmup_steps"])
    decay = int(training["joint_decay_steps"])
    base = float(training["learning_rate"])
    floor = float(training["decay_lr_ratio"])
    if joint_update <= warmup:
        return base * joint_update / warmup
    progress = min(1.0, max(0.0, (joint_update - warmup) / (decay - warmup)))
    return base * (
        floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--belief-config", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--full-index", required=True)
    parser.add_argument("--normalization-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--heartbeat", default="")
    args = parser.parse_args()

    config_path = Path(args.config).resolve(strict=True)
    config = load_r12_evolution_config(config_path)
    target_updates = int(args.updates or config.training["updates"])
    if not 1 <= target_updates <= int(config.training["updates"]):
        raise ValueError("requested R12-E1 updates exceed the full budget")
    if args.workers < 0:
        raise ValueError("R12-E1 workers cannot be negative")
    seed = int(config.training["seed"])
    seed_everything(seed)
    device = torch.device(args.device)

    belief_config = load_r11_config(args.belief_config)
    if belief_config.candidate_id != "p0":
        raise ValueError("R12-E1 requires promoted W11=P0")
    belief_path = Path(args.belief_checkpoint).resolve(strict=True)
    if sha256(belief_path) != str(config.raw["belief_checkpoint_sha256"]):
        raise ValueError("R12-E1 W11 checkpoint hash differs")
    belief_saved = torch.load(belief_path, map_location="cpu", weights_only=False)
    belief = PredictiveBeliefModel(belief_config).to(device)
    belief.load_state_dict(belief_saved["model"], strict=True)
    belief.eval()
    for parameter in belief.parameters():
        parameter.requires_grad_(False)

    index_path = Path(args.full_index).resolve(strict=True)
    index = json.loads(index_path.read_text())
    if (
        index.get("schema_version") != 1
        or index.get("round") != "R12-R4"
        or index.get("protocol_variant") != FULL_EPISODE_PROTOCOL
        or index.get("step_counts", {}).get("train") != EXPECTED_TRAIN
        or index.get("step_counts", {}).get("validation") != EXPECTED_VALIDATION
    ):
        raise ValueError("R12-E1 full-data cache count/identity differs")
    normalization_path = Path(args.normalization_checkpoint).resolve(strict=True)
    normalization = torch.load(
        normalization_path, map_location="cpu", weights_only=False
    )
    stats = {
        key: torch.as_tensor(normalization["stats"][key], dtype=torch.float32)
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
    start_update = int(resume["update"]) if resume else 0
    if start_update >= target_updates:
        raise ValueError("R12-E1 resume update is not below target")
    sampler = TaskWeightedFullEpisodeSampler(
        dataset,
        updates=target_updates,
        rows_per_task=config.training["rows_per_task"],
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
    bridge_updates = int(config.training["bridge_updates"])
    stage = "bridge" if start_update < bridge_updates else "joint"
    optimizer, trainable_names = build_optimizer(model, config.training, stage)
    if resume:
        if (
            resume.get("round") != "R12-E1"
            or resume.get("candidate_id") != config.candidate_id
        ):
            raise ValueError("R12-E1 resume identity differs")
        model.load_state_dict(resume["model"], strict=True)
        boundary = (
            start_update == bridge_updates
            and resume.get("stage") == "bridge"
            and stage == "joint"
        )
        if resume.get("stage") != stage and not boundary:
            raise ValueError("R12-E1 resume stage differs")
        if not boundary:
            optimizer.load_state_dict(resume["optimizer"])
        warm_receipt = resume["warm_start_receipt"]
        restore_rng_state(resume["rng_state"])
    else:
        warm_path = Path(
            str(config.training["warm_start_checkpoint"])
        ).resolve(strict=True)
        if sha256(warm_path) != str(config.training["warm_start_sha256"]):
            raise ValueError("R12-E1 warm-start checkpoint hash differs")
        warm_saved = torch.load(warm_path, map_location="cpu", weights_only=False)
        if (
            warm_saved.get("candidate_id") != config.candidate_id
            or int(warm_saved.get("update", -1))
            != int(config.training["warm_start_update"])
        ):
            raise ValueError("R12-E1 R3 warm-start identity differs")
        warm_receipt = load_r3_core_warm_start(model, warm_saved)

    output = Path(args.output).resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    identity = {
        "schema_version": 1,
        "round": "R12-E1",
        "candidate_id": config.candidate_id,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "belief_checkpoint": str(belief_path),
        "belief_checkpoint_sha256": sha256(belief_path),
        "full_index": str(index_path),
        "full_index_sha256": sha256(index_path),
        "full_step_counts": index["step_counts"],
        "rows_per_task": dict(config.training["rows_per_task"]),
        "normalization_checkpoint": str(normalization_path),
        "normalization_checkpoint_sha256": sha256(normalization_path),
        "primary_input": "native_480x640_fixed_view_RGB",
        "supplemental_inputs": ["W11_TeamBeliefState", "task_id"],
        "deployment": dict(config.deployment),
        "bridge_updates": bridge_updates,
        "joint_updates": int(config.training["updates"]) - bridge_updates,
        "stage_a_trainable_parameters": trainable_names if stage == "bridge" else None,
        "warm_start_receipt": warm_receipt,
        "core_free_runtime": True,
        "created_at": now(),
    }
    atomic_json(output / "training_identity.json", identity)
    heartbeat = Path(args.heartbeat).resolve() if args.heartbeat else None
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    last: dict[str, float | int | str] = {}
    started = time.monotonic()

    def save(update: int, name: str) -> Path:
        path = checkpoints / name
        atomic_torch_save(
            path,
            {
                "schema_version": 1,
                "round": "R12-E1",
                "candidate_id": config.candidate_id,
                "update": update,
                "stage": "bridge" if update <= bridge_updates else "joint",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": dict(config.raw),
                "stats": stats,
                "warm_start_receipt": warm_receipt,
                "full_index_sha256": identity["full_index_sha256"],
                "core_free_runtime": True,
                "last_metrics": last,
                "rng_state": capture_rng_state(),
            },
        )
        return path

    update = start_update
    for update, cpu_batch in enumerate(loader, start=start_update + 1):
        if update == bridge_updates + 1 and stage == "bridge":
            stage = "joint"
            optimizer, trainable_names = build_optimizer(
                model, config.training, stage
            )
        batch = device_batch(cpu_batch, device)
        batch["actions"], history_metrics = robustify_source_aware_history(
            batch, stats, config.training, update, seed
        )
        current_lr = learning_rate(config.training, update)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
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
            raise FloatingPointError(f"non-finite R12-E1 loss at update {update}")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [value for value in model.parameters() if value.requires_grad],
            float(config.training["grad_clip"]),
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise FloatingPointError(f"non-finite R12-E1 grad norm at {update}")
        optimizer.step()
        if hasattr(model.core, "after_optimizer_step"):
            model.core.after_optimizer_step()
        last = {
            "update": update,
            "stage": stage,
            "loss": float(total.detach()),
            "grad_norm": float(grad_norm),
            "learning_rate": current_lr,
            **history_metrics,
            **{
                key: float(value.detach())
                for key, value in losses.items()
                if key != "loss"
                and isinstance(value, torch.Tensor)
                and value.numel() == 1
            },
        }
        if update == start_update + 1 or update % int(
            config.training["progress_every"]
        ) == 0:
            elapsed = time.monotonic() - started
            completed = update - start_update
            row = {
                **last,
                "target_updates": target_updates,
                "updates_per_hour": completed / max(elapsed, 1e-6) * 3600,
                "eta_hours": (target_updates - update)
                * elapsed
                / max(completed, 1)
                / 3600,
                "gpu_memory_gb": (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0
                ),
                "time": time.time(),
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            if heartbeat:
                atomic_json(
                    heartbeat,
                    {
                        "producer": "train_action_generator_evolution",
                        "candidate": config.candidate_id,
                        "pid": __import__("os").getpid(),
                        "update": update,
                        "stage": stage,
                        "updated_at": now(),
                    },
                )
        if (
            update % int(config.training["checkpoint_every"]) == 0
            or update == target_updates
            or stopping
        ):
            latest = save(update, "checkpoint_latest.pt")
            if update in (bridge_updates, target_updates):
                save(update, f"checkpoint_{update:06d}.pt")
            print(json.dumps({"saved": str(latest), "update": update}), flush=True)
        if stopping:
            raise SystemExit(130)
    print(json.dumps({"complete": True, "candidate": config.candidate_id, "update": update}))


if __name__ == "__main__":
    main()
