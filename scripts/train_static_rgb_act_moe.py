#!/usr/bin/env python3
"""Train shared-agent DINO+ACT+MoE from existing static RGB M2 manifests."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.static_rgb_act import (  # noqa: E402
    StaticRGBMoEACT,
    StaticRGBMoEACTConfig,
)
from models.wam import AffineActionCodec  # noqa: E402
from models.wam_multimodal import (  # noqa: E402
    FrozenDINOv3Config,
    FrozenDINOv3Encoder,
)
from train.robofactory_multitask_dataset import (  # noqa: E402
    RoboFactoryMultitaskDataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/static_act/lpd_static_dino_act_moe.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--progress-log",
        type=Path,
        help="Optional JSONL progress log for external S0 monitoring.",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


class _TaskBalancedBatchSampler(Sampler[list[int]]):
    """Resume-stable task-balanced batches keyed by optimizer update."""

    def __init__(
        self,
        dataset: RoboFactoryMultitaskDataset,
        *,
        batch_size: int,
        first_update: int,
        final_update: int,
        seed: int,
    ) -> None:
        self.ranges = tuple(
            dataset.task_indices(index)
            for index in range(len(dataset.contracts))
        )
        self.batch_size = int(batch_size)
        self.first_update = int(first_update)
        self.final_update = int(final_update)
        self.seed = int(seed)

    def __len__(self) -> int:
        return max(self.final_update - self.first_update + 1, 0)

    def __iter__(self):
        for update in range(self.first_update, self.final_update + 1):
            generator = torch.Generator().manual_seed(self.seed + update)
            tasks = torch.randint(
                len(self.ranges),
                (self.batch_size,),
                generator=generator,
            )
            batch = []
            for task_index in tasks.tolist():
                indices = self.ranges[task_index]
                offset = int(
                    torch.randint(
                        len(indices),
                        (),
                        generator=generator,
                    )
                )
                batch.append(indices.start + offset)
            yield batch


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    raw = _load_yaml(config_path)
    training = _mapping(raw, "training")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal static RGB ACT training requires one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "expose exactly one GPU (for example CUDA_VISIBLE_DEVICES=0)"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("static RGB ACT training requires native BF16")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    seed = int(training.get("seed", 20260727))
    _seed_everything(seed)

    dataset = _dataset(raw)
    batch_size = int(args.batch_size or training.get("batch_size", 4))
    updates = int(args.updates or training.get("updates", 80000))
    if batch_size <= 0 or updates <= 0:
        raise ValueError("batch size and updates must be positive")
    model_config = StaticRGBMoEACTConfig.from_dict(_mapping(raw, "model"))
    if model_config.horizon != dataset.action_horizon:
        raise ValueError("model and dataset ACT horizons differ")
    vision = _vision(raw).to(device).eval()
    model = StaticRGBMoEACT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates)

    checkpoint_config = _mapping(raw, "checkpoint")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (ROOT / str(checkpoint_config["output"])).resolve()
    )
    resume = (
        args.resume.expanduser().resolve()
        if args.resume is not None
        else (ROOT / str(checkpoint_config["resume"])).resolve()
    )
    progress_log = (
        args.progress_log.expanduser().resolve()
        if args.progress_log is not None
        else (
            (ROOT / str(checkpoint_config["progress_log"])).resolve()
            if checkpoint_config.get("progress_log") is not None
            else None
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resume.parent.mkdir(parents=True, exist_ok=True)
    if progress_log is not None:
        progress_log.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if not args.no_resume and resume.is_file():
        saved = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = int(saved["update"])
        _restore_rng(_mapping(saved, "rng"))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed checkpoint {output}")

    runtime = _task_runtime(dataset)
    save_interval = int(training.get("save_interval", 1000))
    beta = float(training.get("kl_weight", 1e-3))
    router_weight = float(training.get("router_aux_weight", 1e-2))
    workers = int(training.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_sampler=_TaskBalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            first_update=start + 1,
            final_update=updates,
            seed=seed,
        ),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed + 10_000_000),
    )
    iterator = iter(loader)
    model.train()
    for update in range(start + 1, updates + 1):
        batch = _local_batch(next(iterator))
        images = batch["images"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        vision_tokens = _frozen_vision_tokens(vision, images)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, mu, logvar, router_aux = model(
                vision_tokens,
                state,
                actions,
            )
            assert mu is not None and logvar is not None
            step_mse = (prediction - actions).square().mean(dim=-1)
            per_sample = (step_mse * valid).sum(dim=1) / valid.sum(
                dim=1
            ).clamp_min(1)
            mse = per_sample.mean()
            kl = -0.5 * (
                1 + logvar - mu.square() - logvar.exp()
            ).sum(dim=-1).mean()
            loss = mse + beta * kl + router_weight * (router_aux - 1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training.get("gradient_clip_norm", 1.0))
        )
        optimizer.step()
        scheduler.step()
        if update % 100 == 0 or update == 1:
            progress = {
                "update": update,
                "updates": updates,
                "loss": float(loss.detach()),
                "mse": float(mse.detach()),
                "kl": float(kl.detach()),
                "router_aux": float(router_aux.detach()),
            }
            print(json.dumps(progress), flush=True)
            if progress_log is not None:
                _append_jsonl(progress_log, progress)
        if update % save_interval == 0 and update < updates:
            _atomic_torch_save(
                {
                    "update": update,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": _capture_rng(),
                },
                resume,
            )

    payload = {
        "format_version": "wam.robofactory.static_rgb_act_moe.checkpoint/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "update": updates,
        "model_config": model_config.to_dict(),
        "model": model.state_dict(),
        "task_runtime": runtime,
        "vision": dict(_mapping(raw, "vision")),
        "training": dict(training),
        "data": {
            "manifests": [
                {
                    "path": str(contract.manifest_path),
                    "sha256": contract.manifest_sha256,
                    "task_id": contract.task_id,
                }
                for contract in dataset.contracts
            ],
            "camera_protocol": (
                "existing world-fixed per-agent RGB; no wrist camera; no depth"
            ),
        },
        "source": {
            "git_commit": _git_commit(),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
        },
    }
    _atomic_torch_save(payload, output)
    if resume.exists():
        resume.unlink()
    print(json.dumps({"checkpoint": str(output), "sha256": _sha256(output)}))
    return 0


def _dataset(config: Mapping[str, Any]) -> RoboFactoryMultitaskDataset:
    data = _mapping(config, "data")
    manifests = [(ROOT / str(value)).resolve(strict=True) for value in data["manifests"]]
    horizon = int(_mapping(config, "model")["horizon"])
    return RoboFactoryMultitaskDataset(
        manifests,
        split="train",
        state_history=1,
        action_horizon=horizon,
        task_action_horizons={
            "lift_barrier": horizon,
            "long_pipeline_delivery": horizon,
        },
        visual_history=1,
        future_horizons=(1,),
        cameras=("global", "agent_0", "agent_1", "agent_2", "agent_3"),
        max_state_dim=72,
        max_action_dim=32,
        max_agents=4,
        max_text_tokens=16,
        stride=int(data.get("stride", 1)),
        hdf5_cache_size=int(data.get("hdf5_cache_size", 4)),
    )


def _local_batch(batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
    states = batch["states"][:, -1].reshape(-1, 4, 18)
    actions = batch["action_targets"].reshape(
        batch["action_targets"].shape[0], batch["action_targets"].shape[1], 4, 8
    ).permute(0, 2, 1, 3)
    horizon_valid = batch["action_target_valid_mask"][:, None, :].expand(
        -1, 4, -1
    )
    agent_count = batch["embodiment_index"] + 1
    agent_valid = torch.arange(4)[None] < agent_count[:, None]
    # Slot zero is global. Existing agent_i images remain world-fixed and are
    # used exactly as recorded; this branch deliberately does not retrofit them.
    images = batch["images"][:, -1, 1:5]
    image_valid = batch["image_valid_mask"][:, -1, 1:5]
    valid_agents = agent_valid & image_valid
    return {
        "images": images[valid_agents],
        "state": states[valid_agents],
        "actions": actions[valid_agents],
        "valid": horizon_valid[valid_agents].to(torch.float32),
    }


def _vision(config: Mapping[str, Any]) -> FrozenDINOv3Encoder:
    value = _mapping(config, "vision")
    return FrozenDINOv3Encoder(
        FrozenDINOv3Config(
            encoder_name=str(value["encoder_name"]),
            model_id=str(value["model_id"]),
            revision=str(value["revision"]),
            config_path=(ROOT / str(value["config_path"])).resolve(strict=True),
            weights_path=(ROOT / str(value["weights_path"])).resolve(strict=True),
            expected_config_sha256=str(value["expected_config_sha256"]),
            expected_weights_sha256=str(value["expected_weights_sha256"]),
            preprocess_id=str(value["preprocess_id"]),
            input_size=None,
            input_height=int(value["input_height"]),
            input_width=int(value["input_width"]),
            inference_batch_size=int(value.get("inference_batch_size", 2)),
        )
    )


def _frozen_vision_tokens(
    vision: FrozenDINOv3Encoder,
    images: Tensor,
) -> Tensor:
    # The ACT projection is trainable and autograd must save these features
    # while computing parameter gradients.  no_grad() freezes DINO without
    # creating inference tensors, which cannot be saved for backward.
    with torch.no_grad():
        return vision(images).spatial_tokens


def _task_runtime(dataset: RoboFactoryMultitaskDataset) -> list[dict[str, Any]]:
    result = []
    for index, contract in enumerate(dataset.contracts):
        codec = dataset.datasets[index].manifest.action_codec
        if not isinstance(codec, AffineActionCodec):
            raise TypeError("static RGB ACT requires an affine action codec")
        result.append(
            {
                **contract.to_dict(),
                "task_index": index,
                "state_mean": dataset._state_means[index].tolist(),
                "state_std": dataset._state_stds[index].tolist(),
                "action_mean": dataset._action_means[index].tolist(),
                "action_std": dataset._action_stds[index].tolist(),
                "action_codec": codec.config.to_dict(),
            }
        )
    return result


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return result


def _load_yaml(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("config root must be a mapping")
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    torch.cuda.set_rng_state_all(payload["cuda"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
