from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.action_generator.base import JointActionGenerator, load_r12_config
from before_we_act.data.action_windows import CachedActionWindows, ExactFiveTaskWindowSampler
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def learning_rate_at_update(training, update: int) -> float:
    """Linear warmup followed by cosine decay, frozen before the R12-R2 run."""
    base = float(training["learning_rate"])
    warmup = int(training["warmup_steps"])
    decay = int(training["decay_steps"])
    floor = float(training["decay_lr_ratio"])
    if update <= warmup:
        return base * update / warmup
    progress = min(1.0, max(0.0, (update - warmup) / (decay - warmup)))
    return base * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def robustify_action_history(batch, stats, training, update: int, seed: int):
    """Deterministic scheduled-sampling proxy for closed-loop action-history drift.

    Each selected sample receives exactly one of three perturbations: all-zero
    history, scale-aware Gaussian noise, or a one-slot additional lag.  The
    remaining samples retain expert history.  No future/commanded action enters
    the model input.
    """
    actions = batch["actions"].clone()
    maximum = float(training["history_augmentation_probability"])
    ramp = int(training["history_augmentation_ramp_updates"])
    probability = maximum * min(1.0, update / ramp)
    if probability <= 0:
        return actions, {"history_aug_probability": 0.0, "history_aug_fraction": 0.0}
    generator = torch.Generator(device=actions.device)
    generator.manual_seed(int(seed) + 2_000_003 * int(update))
    selected = torch.rand(len(actions), generator=generator, device=actions.device) < probability
    variants = torch.randint(0, 3, (len(actions),), generator=generator, device=actions.device)
    zero = selected & variants.eq(0)
    noisy = selected & variants.eq(1)
    lagged = selected & variants.eq(2)
    actions[zero] = 0
    if bool(noisy.any()):
        scale = torch.as_tensor(stats["a_std"], device=actions.device, dtype=actions.dtype)
        noise = torch.randn(actions.shape, generator=generator, device=actions.device, dtype=actions.dtype)
        actions[noisy] = actions[noisy] + noise[noisy] * scale * float(training["history_noise_scale"])
    if bool(lagged.any()):
        shifted = torch.zeros_like(actions[lagged])
        shifted[:, 1:] = actions[lagged, :-1]
        actions[lagged] = shifted
    actions = actions * batch["agent_mask"][:, None, :, None].to(actions.dtype)
    return actions, {
        "history_aug_probability": probability,
        "history_aug_fraction": float(selected.float().mean()),
        "history_zero_fraction": float(zero.float().mean()),
        "history_noise_fraction": float(noisy.float().mean()),
        "history_lag_fraction": float(lagged.float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--belief-config", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--spatial-cache", required=True)
    parser.add_argument("--recovery-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--ignore-warm-start", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config_path = Path(args.config).resolve(strict=True)
    config = load_r12_config(config_path)
    target_updates = int(args.updates or config.training["updates"])
    if not 1 <= target_updates <= int(config.training["updates"]):
        raise ValueError("requested updates exceed the frozen R12 budget")
    seed_everything(int(config.training["seed"]))
    device = torch.device(args.device)

    belief_config = load_r11_config(args.belief_config)
    if belief_config.candidate_id != "p0":
        raise ValueError("R12 common parent must be promoted W11=P0")
    belief_path = Path(args.belief_checkpoint).resolve(strict=True)
    if sha256(belief_path) != config.raw["belief_checkpoint_sha256"]:
        raise ValueError("W11 belief checkpoint hash differs")
    belief_payload = torch.load(belief_path, map_location="cpu", weights_only=False)
    belief = PredictiveBeliefModel(belief_config).to(device)
    belief.load_state_dict(belief_payload["model"], strict=True)
    belief.eval()
    for parameter in belief.parameters():
        parameter.requires_grad_(False)

    spatial_cache_path = Path(args.spatial_cache).resolve(strict=True)
    recovery_cache_path = Path(args.recovery_cache).resolve(strict=True)
    dataset = CachedActionWindows(
        args.cache,
        "train",
        spatial_cache_path=spatial_cache_path,
        recovery_cache_path=recovery_cache_path,
    )
    if dataset.metadata["seed"] != int(config.training["seed"]):
        raise ValueError("R12 cache seed differs")
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    warm_path = "" if args.ignore_warm_start else str(config.training["warm_start_checkpoint"])
    warm_start = None
    if not resume and warm_path:
        resolved_warm_path = Path(warm_path).resolve(strict=True)
        if sha256(resolved_warm_path) != str(config.training["warm_start_sha256"]):
            raise ValueError("R12-R2 warm-start checkpoint hash differs")
        warm_start = torch.load(resolved_warm_path, map_location="cpu", weights_only=False)
        if warm_start.get("candidate_id") != config.candidate_id:
            raise ValueError("warm-start candidate differs")
        if int(warm_start.get("update", -1)) != int(config.training["warm_start_update"]):
            raise ValueError("warm-start update differs")
    # R12-R3 starts a fresh optimizer/update schedule.  The R12-R2 checkpoint
    # supplies only the already-trained action core; its historical update is
    # provenance, not the new round's sampler cursor.
    start_update = int(resume["update"]) if resume else 0
    if start_update >= target_updates:
        raise ValueError("resume update is not below target")
    sampler = ExactFiveTaskWindowSampler(
        dataset.task_index,
        target_updates,
        int(config.training["seed"]),
        start_update,
        source_indices=dataset.source_index,
        recovery_probability=float(config.training["recovery_sampling_probability"]),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    if int(config.training["batch_size"]) != 5:
        raise ValueError("R12 exact five-task sampler requires batch_size=5")
    model = JointActionGenerator(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    if resume:
        if resume["candidate_id"] != config.candidate_id:
            raise ValueError("resume candidate differs")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
    elif warm_start:
        incompatible = model.load_state_dict(warm_start["model"], strict=False)
        allowed_missing_prefixes = (
            "spatial_norm.",
            "spatial_projection.",
            "spatial_view_embedding",
            "spatial_row_embedding",
            "spatial_column_embedding",
            "spatial_cross_attention.",
            "spatial_gate",
        )
        if incompatible.unexpected_keys or any(
            not name.startswith(allowed_missing_prefixes)
            for name in incompatible.missing_keys
        ):
            raise ValueError(
                "R12-R2 core warm start differs outside the new spatial adapter: "
                f"{incompatible}"
            )

    output = Path(args.output).resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    identity = {
        "schema_version": 1,
        "round": "R12-R3",
        "candidate_id": config.candidate_id,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "belief_checkpoint": str(belief_path),
        "belief_checkpoint_sha256": sha256(belief_path),
        "action_cache": str(Path(args.cache).resolve()),
        "action_cache_sha256": sha256(Path(args.cache)),
        "spatial_cache": str(spatial_cache_path),
        "spatial_cache_sha256": sha256(spatial_cache_path),
        "spatial_cache_protocol": dataset.spatial_metadata["protocol_variant"],
        "recovery_cache": str(recovery_cache_path),
        "recovery_cache_sha256": sha256(recovery_cache_path),
        "recovery_sampling_probability": float(config.training["recovery_sampling_probability"]),
        "trainable_parameters": [name for name, value in model.named_parameters() if value.requires_grad],
        "core_free_runtime": True,
        "observation_mode": config.observation["mode"],
        "batch_size": 5,
        "seed": int(config.training["seed"]),
        "precision": config.training["precision"],
        "warm_start_checkpoint": str(Path(warm_path).resolve()) if warm_start else None,
        "warm_start_sha256": str(config.training["warm_start_sha256"]) if warm_start else None,
        "warm_start_update": int(config.training["warm_start_update"]) if warm_start else 0,
        "warm_start_mode": "r12r2_core_only_zero_gated_spatial_adapter",
        "history_robustification": "deterministic_clean_zero_noise_lag_mixture",
        "learning_rate_schedule": "linear_warmup_cosine_decay",
    }
    (output / "training_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    last = {}
    started = time.monotonic()

    def save(update: int, name: str):
        path = checkpoints / name
        atomic_torch_save(
            {
                "schema_version": 1,
                "round": "R12-R3",
                "candidate_id": config.candidate_id,
                "update": update,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": dict(config.raw),
                "stats": dict(dataset.stats),
                "belief_checkpoint_sha256": sha256(belief_path),
                "spatial_cache_sha256": identity["spatial_cache_sha256"],
                "recovery_cache_sha256": identity["recovery_cache_sha256"],
                "observation_mode": config.observation["mode"],
                "core_free_runtime": True,
                "last_metrics": last,
                "initialization": {
                    "warm_start_checkpoint": identity["warm_start_checkpoint"],
                    "warm_start_sha256": identity["warm_start_sha256"],
                    "warm_start_update": identity["warm_start_update"],
                },
            },
            path,
        )
        return path

    update = start_update
    for update, cpu_batch in enumerate(loader, start=start_update + 1):
        batch = device_batch(cpu_batch, device)
        batch["actions"], history_metrics = robustify_action_history(
            batch, dataset.stats, config.training, update, int(config.training["seed"])
        )
        current_lr = learning_rate_at_update(config.training, update)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            belief_output = belief(batch)["belief"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            losses = model.training_loss(
                belief_output,
                batch["spatial_tokens"],
                batch["spatial_view_mask"],
                batch["joint_actions"],
                batch["action_step_mask"].bool(),
            )
            total = losses["loss"]
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite R12 loss at update {update}")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.training["grad_clip"])
        )
        optimizer.step()
        if hasattr(model.core, "after_optimizer_step"):
            model.core.after_optimizer_step()
        last = {
            "update": update,
            "loss": float(total.detach()),
            "grad_norm": float(grad_norm),
            "learning_rate": current_lr,
            **history_metrics,
            **{
                key: float(value.detach())
                for key, value in losses.items()
                if key != "loss" and isinstance(value, torch.Tensor) and value.numel() == 1
            },
        }
        if update == start_update + 1 or update % int(config.training["progress_every"]) == 0:
            elapsed = time.monotonic() - started
            completed = update - start_update
            row = {
                **last,
                "target_updates": target_updates,
                "updates_per_hour": completed / max(elapsed, 1e-6) * 3600,
                "eta_hours": (target_updates - update) * elapsed / max(completed, 1) / 3600,
                "gpu_memory_gb": (
                    torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
                ),
                "time": time.time(),
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
        if (
            update % int(config.training["checkpoint_every"]) == 0
            or update == target_updates
            or stopping
        ):
            latest = save(update, "checkpoint_latest.pt")
            if update == target_updates:
                save(update, f"checkpoint_{update:06d}.pt")
            print(json.dumps({"saved": str(latest), "update": update}), flush=True)
        if stopping:
            raise SystemExit(130)
    print(json.dumps({"complete": True, "candidate": config.candidate_id, "update": update}))


if __name__ == "__main__":
    main()
