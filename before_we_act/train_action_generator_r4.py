from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.action_generator.r4_base import (
    R4JointActionGenerator,
    load_r12_r4_config,
    load_r3_core_warm_start,
)
from before_we_act.data.full_episode_windows import (
    ExactFiveTaskFullEpisodeSampler,
    FULL_EPISODE_PROTOCOL,
    FullEpisodeActionWindows,
)
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def robustify_source_aware_history(
    batch, stats, training, update: int, seed: int
) -> tuple[torch.Tensor, dict[str, float]]:
    """Perturb demonstration history while preserving genuine recovery rows."""

    actions = batch["actions"].clone()
    maximum = float(training["history_augmentation_probability"])
    ramp = int(training["history_augmentation_ramp_updates"])
    probability = maximum * min(1.0, update / ramp)
    generator = torch.Generator(device=actions.device).manual_seed(
        int(seed) + 2_000_003 * int(update)
    )
    demonstration = batch["source_index"].eq(0)
    selected = (
        torch.rand(len(actions), generator=generator, device=actions.device)
        < probability
    ) & demonstration
    variants = torch.randint(
        0, 3, (len(actions),), generator=generator, device=actions.device
    )
    zero = selected & variants.eq(0)
    noisy = selected & variants.eq(1)
    lagged = selected & variants.eq(2)
    actions[zero] = 0
    if bool(noisy.any()):
        scale = torch.as_tensor(
            stats["a_std"], device=actions.device, dtype=actions.dtype
        )
        noise = torch.randn(
            actions.shape,
            generator=generator,
            device=actions.device,
            dtype=actions.dtype,
        )
        actions[noisy] += (
            noise[noisy] * scale * float(training["history_noise_scale"])
        )
    if bool(lagged.any()):
        shifted = torch.zeros_like(actions[lagged])
        shifted[:, 1:] = actions[lagged, :-1]
        actions[lagged] = shifted
    actions *= batch["agent_mask"][:, None, :, None].to(actions.dtype)
    recovery = ~demonstration
    return actions, {
        "history_aug_probability": probability,
        "history_aug_fraction": float(selected.float().mean()),
        "history_zero_fraction": float(zero.float().mean()),
        "history_noise_fraction": float(noisy.float().mean()),
        "history_lag_fraction": float(lagged.float().mean()),
        "recovery_fraction": float(recovery.float().mean()),
        "recovery_history_modified_fraction": float((selected & recovery).float().mean()),
    }


def joint_learning_rate(training, joint_update: int) -> float:
    base = float(training["learning_rate"])
    warmup = int(training["joint_warmup_steps"])
    decay = int(training["joint_decay_steps"])
    floor = float(training["decay_lr_ratio"])
    if joint_update <= warmup:
        return base * joint_update / warmup
    progress = min(1.0, max(0.0, (joint_update - warmup) / (decay - warmup)))
    return base * (
        floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def build_optimizer(model, training, stage: str):
    names = model.set_training_stage(stage)
    parameters = [value for value in model.parameters() if value.requires_grad]
    if not parameters or not names:
        raise ValueError(f"R12-R4 {stage} stage has no trainable parameters")
    return (
        torch.optim.AdamW(
            parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        ),
        names,
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
    config = load_r12_r4_config(config_path)
    target_updates = int(args.updates or config.training["updates"])
    if not 1 <= target_updates <= int(config.training["updates"]):
        raise ValueError("requested R12-R4 updates exceed the frozen budget")
    if args.workers < 0:
        raise ValueError("R12-R4 workers cannot be negative")
    seed = int(config.training["seed"])
    seed_everything(seed)
    device = torch.device(args.device)

    belief_config = load_r11_config(args.belief_config)
    if belief_config.candidate_id != "p0":
        raise ValueError("R12-R4 requires promoted W11=P0")
    belief_path = Path(args.belief_checkpoint).resolve(strict=True)
    if sha256(belief_path) != str(config.raw["belief_checkpoint_sha256"]):
        raise ValueError("R12-R4 W11 checkpoint hash differs")
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
        or index.get("step_counts", {}).get("train", {})
        != {
            "lift_barrier": 8255,
            "camera_alignment": 11764,
            "three_robots_stack_cube": 48892,
            "long_pipeline_delivery": 88493,
            "take_photo": 23044,
        }
        or index.get("step_counts", {}).get("validation", {})
        != {
            "lift_barrier": 1015,
            "camera_alignment": 1457,
            "three_robots_stack_cube": 6138,
            "long_pipeline_delivery": 10981,
            "take_photo": 2884,
        }
    ):
        raise ValueError("R12-R4 full-data index count/identity differs")
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
        raise ValueError("R12-R4 resume update is not below target")
    sampler = ExactFiveTaskFullEpisodeSampler(
        dataset,
        updates=target_updates,
        rows_per_task=int(config.training["rows_per_task"]),
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
    model = R4JointActionGenerator(config).to(device)
    bridge_updates = int(config.training["bridge_updates"])
    stage = "bridge" if start_update < bridge_updates else "joint"
    optimizer, trainable_names = build_optimizer(model, config.training, stage)
    warm_receipt = None
    if resume:
        if resume.get("candidate_id") != config.candidate_id:
            raise ValueError("R12-R4 resume candidate differs")
        model.load_state_dict(resume["model"], strict=True)
        boundary_transition = (
            start_update == bridge_updates
            and resume.get("stage") == "bridge"
            and stage == "joint"
        )
        if resume.get("stage") != stage and not boundary_transition:
            raise ValueError("R12-R4 resume stage differs")
        if not boundary_transition:
            optimizer.load_state_dict(resume["optimizer"])
        warm_receipt = resume["warm_start_receipt"]
    else:
        warm_path = Path(
            str(config.training["warm_start_checkpoint"])
        ).resolve(strict=True)
        if sha256(warm_path) != str(config.training["warm_start_sha256"]):
            raise ValueError("R12-R4 R3 warm-start hash differs")
        warm_saved = torch.load(warm_path, map_location="cpu", weights_only=False)
        if (
            warm_saved.get("candidate_id") != config.candidate_id
            or int(warm_saved.get("update", -1))
            != int(config.training["warm_start_update"])
        ):
            raise ValueError("R12-R4 R3 warm-start identity differs")
        warm_receipt = load_r3_core_warm_start(model, warm_saved)

    output = Path(args.output).resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    identity = {
        "schema_version": 1,
        "round": "R12-R4",
        "candidate_id": config.candidate_id,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "belief_checkpoint": str(belief_path),
        "belief_checkpoint_sha256": sha256(belief_path),
        "full_index": str(index_path),
        "full_index_sha256": sha256(index_path),
        "full_step_counts": index["step_counts"],
        "normalization_checkpoint": str(normalization_path),
        "normalization_checkpoint_sha256": sha256(normalization_path),
        "recovery_cache": None,
        "source_aware_history": True,
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
    last = {}
    started = time.monotonic()

    def save(update: int, name: str) -> Path:
        path = checkpoints / name
        atomic_torch_save(
            path,
            {
                "schema_version": 1,
                "round": "R12-R4",
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
        if stage == "bridge":
            current_lr = float(config.training["learning_rate"])
        else:
            current_lr = joint_learning_rate(
                config.training, update - bridge_updates
            )
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
                batch["joint_actions"],
                batch["action_step_mask"].bool(),
            )
            total = losses["loss"]
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite R12-R4 loss at update {update}")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [value for value in model.parameters() if value.requires_grad],
            float(config.training["grad_clip"]),
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise FloatingPointError(f"non-finite R12-R4 grad norm at {update}")
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
                        "producer": "train_action_generator_r4",
                        "candidate": config.candidate_id,
                        "pid": os.getpid(),
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
