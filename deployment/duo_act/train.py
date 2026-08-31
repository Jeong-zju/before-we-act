from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import TASKS, DuoACTDataset, TaskEpisodeBatchSampler
from .model import ACT


FROZEN_CONFIG_SCHEMA = "before-we-act.duobench-act-frozen-training/1"
FROZEN_CONFIG_SHA256 = "dd4b18826ca080a497db7c3facfc0dae99342215ea6f6d6f3e90cca3d58fdea7"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested(value, *keys):
    for key in keys:
        value = value[key]
    return value


def _explicit_flags(argv: list[str]) -> set[str]:
    return {
        token.split("=", 1)[0]
        for token in argv
        if token.startswith("--")
    }


def _same_cli_value(left, right) -> bool:
    if isinstance(left, Path) or isinstance(right, Path):
        return str(left) == str(right)
    return left == right


def validate_frozen_config(config: dict) -> None:
    if config.get("schema") != FROZEN_CONFIG_SCHEMA:
        raise ValueError(f"unsupported frozen config schema: {config.get('schema')!r}")
    tasks = _nested(config, "data", "tasks")
    constructor = _nested(config, "model", "constructor")
    training = config["training"]
    if constructor["tasks"] != len(tasks) or len(tasks) != 11:
        raise ValueError("frozen config must describe all eleven DuoBench tasks")
    if constructor["horizon"] <= 0:
        raise ValueError("model horizon must be positive")
    if training["updates"] <= 0 or training["batch_size"] <= 0:
        raise ValueError("updates and batch size must be positive")
    if training["scheduler"]["T_max"] != training["updates"]:
        raise ValueError("CosineAnnealingLR T_max must equal the frozen update count")
    if training["samples_drawn"] != training["updates"] * training["batch_size"]:
        raise ValueError("frozen samples_drawn differs from updates times batch size")
    if config["data"]["indexed_local_arm_samples"] != 2 * config["data"]["causal_state_action_pairs"]:
        raise ValueError("indexed local-arm sample count is inconsistent")
    supported = {
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "task_sampling": "uniform_over_11_tasks",
        "stream_sampling": "uniform_within_selected_task",
        "timestep_sampling": "uniform_within_selected_episode_arm_stream",
        "mixed_precision": "cuda_autocast_bfloat16",
    }
    observed = {
        "optimizer": training["optimizer"]["class"],
        "scheduler": training["scheduler"]["class"],
        "task_sampling": config["sampling"]["task_sampling"],
        "stream_sampling": config["sampling"]["episode_arm_stream_sampling"],
        "timestep_sampling": config["sampling"]["timestep_sampling"],
        "mixed_precision": training["mixed_precision"],
    }
    if observed != supported:
        raise ValueError(f"frozen config requests unsupported policy behavior: {observed}")
    if config["sampling"]["augmentation"] != "none":
        raise ValueError("this ACT policy does not implement training augmentation")
    if config["objective"]["prior_loss_frequency"] <= 0:
        raise ValueError("prior loss frequency must be positive")


def validate_frozen_policy_sources(config: dict) -> None:
    """Bind the config to the exact model and dataset implementations used."""

    source_hashes = {
        "historical_model_source_sha256": Path(__file__).with_name("model.py"),
        "historical_dataset_source_sha256": Path(__file__).with_name("dataset.py"),
    }
    artifacts = config["historical_artifacts"]
    for field, path in source_hashes.items():
        observed = _sha256_file(path)
        expected = artifacts[field]
        if observed != expected:
            raise ValueError(
                f"formal DuoBench ACT {path.name} hash differs from the frozen "
                f"policy: {observed} != {expected}"
            )


def apply_frozen_config(args, argv: list[str]):
    if args.config is None:
        return None, None
    config_path = args.config.resolve()
    config_sha256 = _sha256_file(config_path)
    if config_sha256 != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "formal DuoBench ACT config hash differs from the frozen policy: "
            f"{config_sha256} != {FROZEN_CONFIG_SHA256}"
        )
    config = json.loads(config_path.read_text())
    validate_frozen_config(config)
    validate_frozen_policy_sources(config)
    values = {
        "data": Path(_nested(config, "paths", "data_root")),
        "output": Path(_nested(config, "paths", "output_root")),
        "updates": int(_nested(config, "training", "updates")),
        "batch_size": int(_nested(config, "training", "batch_size")),
        "workers": int(_nested(config, "runtime", "dataloader_workers")),
        "horizon": int(_nested(config, "model", "constructor", "horizon")),
        "action_lag": int(_nested(config, "data", "action_lag_rows")),
        "lr": float(_nested(config, "training", "optimizer", "learning_rate")),
        "beta": float(_nested(config, "objective", "kl_weight")),
        "seed": int(_nested(config, "training", "seed")),
        "save_every": int(_nested(config, "checkpointing", "save_every_updates")),
        "init_checkpoint": (
            None
            if _nested(config, "training", "init_checkpoint") is None
            else Path(_nested(config, "training", "init_checkpoint"))
        ),
        "prior_loss_weight": float(_nested(config, "objective", "prior_mse_weight")),
        "prior_loss_frequency": int(_nested(config, "objective", "prior_loss_frequency")),
    }
    flags = _explicit_flags(argv)
    for name, frozen in values.items():
        flag = "--" + name.replace("_", "-")
        current = getattr(args, name)
        if flag in flags and not _same_cli_value(current, frozen):
            raise ValueError(f"{flag}={current!r} conflicts with frozen value {frozen!r}")
        setattr(args, name, frozen)
    if args.smoke:
        raise ValueError("--smoke cannot alter a frozen formal training contract")
    return config, config_sha256


def validate_manifest_against_config(manifest: dict, config: dict) -> None:
    if manifest.get("dataset_revision") != config["data"]["dataset_revision"]:
        raise ValueError("prepared dataset revision differs from frozen config")
    if list(config["data"]["tasks"]) != list(TASKS):
        raise ValueError("task order differs from the policy's frozen task embedding order")
    if manifest.get("total_episodes") != config["data"]["total_demonstrations"]:
        raise ValueError("prepared demonstration count differs from frozen config")
    if manifest.get("total_frames") != config["data"]["total_frames"]:
        raise ValueError("prepared frame count differs from frozen config")
    alignment = manifest.get("recording_alignment", {})
    if alignment.get("action_lag_rows") != config["data"]["action_lag_rows"]:
        raise ValueError("prepared causal action lag differs from frozen config")
    if manifest.get("action_target_contract", {}).get("sha256") != config["data"]["action_target_contract_sha256"]:
        raise ValueError("prepared action-target contract differs from frozen config")
    frozen_norm = config["data"]["normalization"]
    observed_norm = manifest["normalization"]
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
        if not np.array_equal(
            np.asarray(observed_norm[key], dtype=np.float64),
            np.asarray(frozen_norm[key], dtype=np.float64),
        ):
            raise ValueError(f"prepared normalization field {key} differs from frozen config")


def atomic_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="immutable formal training config")
    parser.add_argument("--print-resolved-config", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--updates", type=int, default=120000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--action-lag", type=int, default=0,
                        help="Rows of post-action recording lag to skip before the target chunk")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Initialize model weights only; optimizer/schedule start fresh")
    parser.add_argument("--prior-loss-weight", type=float, default=0.0,
                        help="Direct z=0 deployment-prior reconstruction weight")
    parser.add_argument("--prior-loss-frequency", type=int, default=1,
                        help="Evaluate prior loss every N updates and scale it to keep its expectation")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    frozen_config, frozen_config_sha256 = apply_frozen_config(args, sys.argv[1:])
    if args.print_resolved_config:
        if frozen_config is None:
            parser.error("--print-resolved-config requires --config")
        print(json.dumps({
            "config": str(args.config.resolve()),
            "config_sha256": frozen_config_sha256,
            "resolved": frozen_config,
        }, indent=2))
        return
    if args.data is None or args.output is None:
        parser.error("--data and --output are required unless supplied by --config")
    if args.prior_loss_weight < 0:
        raise ValueError("prior_loss_weight must be non-negative")
    if args.prior_loss_frequency <= 0:
        raise ValueError("prior_loss_frequency must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runtime = frozen_config["runtime"] if frozen_config else {}
    training_settings = frozen_config["training"] if frozen_config else {}
    torch.backends.cudnn.benchmark = bool(training_settings.get("cudnn_benchmark", True))
    torch.backends.cudnn.deterministic = bool(training_settings.get("cudnn_deterministic", False))
    torch.set_float32_matmul_precision(training_settings.get("matmul_precision", "high"))
    device = torch.device(runtime.get("device", "cuda:0"))
    manifest = json.loads((args.data / "manifest.json").read_text())
    if frozen_config is not None:
        validate_manifest_against_config(manifest, frozen_config)
    dataset = DuoACTDataset(args.data, horizon=args.horizon, action_lag=args.action_lag)
    model_kwargs = frozen_config["model"]["constructor"] if frozen_config else {"horizon": args.horizon}
    model = ACT(**model_kwargs).to(device)
    optimizer_settings = training_settings.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(optimizer_settings.get("betas", (0.9, 0.95))),
        eps=float(optimizer_settings.get("epsilon", 1e-8)),
        weight_decay=float(optimizer_settings.get("weight_decay", 1e-4)),
        amsgrad=bool(optimizer_settings.get("amsgrad", False)),
        maximize=bool(optimizer_settings.get("maximize", False)),
    )
    scheduler_settings = training_settings.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        int(scheduler_settings.get("T_max", args.updates)),
        eta_min=float(scheduler_settings.get("eta_min", 2e-6)),
    )
    start = 0
    latest = args.output / "latest.pt"
    if latest.is_file() and not args.smoke:
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        saved_config_sha = saved.get("frozen_training_config_sha256")
        if frozen_config_sha256 is not None and saved_config_sha not in (None, frozen_config_sha256):
            raise ValueError("resume checkpoint was produced by a different frozen config")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = int(saved["update"])
    elif args.init_checkpoint is not None:
        saved = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        expected = {key: value for key, value in saved["model_config"].items() if key != "vision_backbone"}
        current = {key: value for key, value in model.config.items() if key != "vision_backbone"}
        if expected != current:
            raise ValueError(f"initial checkpoint model config differs: {expected} != {current}")
        model.load_state_dict(saved["model"])
    remaining = max(args.updates - start, 0)
    sampler = TaskEpisodeBatchSampler(dataset, args.batch_size, remaining, args.seed + start)
    pin_memory = bool(runtime.get("pin_memory", True))
    persistent_workers = bool(runtime.get("persistent_workers", args.workers > 0)) and args.workers > 0
    prefetch_factor = int(runtime.get("prefetch_factor", 3)) if args.workers else None
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
    )
    model.train()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    image_scale = 255.0
    grad_clip = float(training_settings.get("gradient_clip_norm", 1.0))
    log_every = int((frozen_config or {}).get("checkpointing", {}).get("log_every_updates", 100))
    for offset, batch in enumerate(loader, 1):
        update = start + offset
        image, qpos, task_id, actions, mask = (item.to(device, non_blocking=True) for item in batch)
        optimizer.zero_grad(set_to_none=True)
        image_float = image.float().div_(image_scale)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, mu, logvar = model(image_float, qpos, task_id, actions)
            mse = ((prediction - actions).square().mean(-1) * mask).sum() / mask.sum().clamp_min(1)
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
            prior_active = bool(
                args.prior_loss_weight and update % args.prior_loss_frequency == 0
            )
            if prior_active:
                prior_prediction, _, _ = model(image_float, qpos, task_id)
                prior_mse = ((prior_prediction - actions).square().mean(-1) * mask).sum() / mask.sum().clamp_min(1)
            else:
                prior_mse = mse.new_zeros(())
            prior_scale = args.prior_loss_frequency if prior_active else 0
            loss = mse + args.beta * kl + args.prior_loss_weight * prior_scale * prior_mse
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at update {update}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        if update == 1 or update % log_every == 0 or update == args.updates:
            row = {
                "update": update, "target_updates": args.updates, "loss": float(loss),
                "mse": float(mse), "kl": float(kl), "grad_norm": float(grad_norm),
                "prior_mse": float(prior_mse),
                "lr": scheduler.get_last_lr()[0], "elapsed_seconds": time.time() - started,
                "samples_seen_this_run": offset * args.batch_size,
            }
            with (args.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if update % args.save_every == 0 or update == args.updates:
            payload = {
                "format": "duobench-act-v1", "update": update, "model_config": model.config,
                "frozen_training_config": frozen_config,
                "frozen_training_config_sha256": frozen_config_sha256,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "normalization": manifest["normalization"],
                "dataset_revision": manifest["dataset_revision"],
                "action_target_contract": manifest.get("action_target_contract"),
                "recording_alignment": manifest.get("recording_alignment"),
                "action_encoding": "absolute_joint7_binary_gripper1",
                "policy_contract": "shared_weights_decentralized_head_rgb_local_wrist_rgb_local_qpos_to_local_action8",
                "training_contract": {
                    "all_11_tasks": True, "all_50_demos_per_task": True, "task_balanced": True,
                    "global_batch_size": args.batch_size, "horizon": args.horizon,
                    "updates": args.updates, "seed": args.seed,
                    "action_lag": args.action_lag,
                    "beta": args.beta, "prior_loss_weight": args.prior_loss_weight,
                    "prior_loss_frequency": args.prior_loss_frequency,
                    "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                },
            }
            atomic_save(payload, latest)
            if update == args.updates:
                atomic_save(payload, args.output / "final.pt")


if __name__ == "__main__":
    main()
