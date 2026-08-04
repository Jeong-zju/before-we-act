"""Common frozen-parent trainer for all four R10 perception candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .bwa_contracts import CoreContext, CoreDeploymentContext
    from .bwa_data import take_hdf5_rows
    from .bwa_perception import (
        EXPECTED_PARENT_SHA256,
        FORMAL_BATCH,
        FORMAL_SEED,
        build_perception_extension,
        load_r10_config,
    )
    from .no_wrist_pair_model import NoWristPAIRRoute
    from .train_act import seed_everything
    from .train_no_wrist_pair import (
        ExactFiveTaskBatchSampler,
        NoWristFrameDataset,
        atomic_torch_save,
        load_episodes,
    )
except ImportError:
    from bwa_contracts import CoreContext, CoreDeploymentContext
    from bwa_data import take_hdf5_rows
    from bwa_perception import (
        EXPECTED_PARENT_SHA256,
        FORMAL_BATCH,
        FORMAL_SEED,
        build_perception_extension,
        load_r10_config,
    )
    from no_wrist_pair_model import NoWristPAIRRoute
    from train_act import seed_everything
    from train_no_wrist_pair import (
        ExactFiveTaskBatchSampler,
        NoWristFrameDataset,
        atomic_torch_save,
        load_episodes,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BWAFrameDataset(NoWristFrameDataset):
    """Native frame dataset plus legal history and training-only future labels."""

    def __init__(
        self,
        episodes,
        horizon,
        stats,
        *,
        history_steps: int,
        history_stride: int,
        include_history_views: bool,
        future_qpos_horizons: tuple[int, ...],
        future_feature_horizons: tuple[int, ...],
    ):
        super().__init__(episodes, horizon, stats)
        self.history_steps = int(history_steps)
        self.history_stride = int(history_stride)
        self.include_history_views = bool(include_history_views)
        self.future_qpos_horizons = tuple(int(value) for value in future_qpos_horizons)
        self.future_feature_horizons = tuple(int(value) for value in future_feature_horizons)
        if self.history_steps < 0 or self.history_stride < 1:
            raise ValueError("invalid history window")

    @staticmethod
    def _image_tensor(values: list[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.stack(values)).permute(0, 3, 1, 2).contiguous()

    def __getitem__(self, request):
        global_rgb, local_rgb, qpos, actions, action_mask = super().__getitem__(request)
        episode_index, arm, time_index = request
        episode = self.episodes[episode_index]
        result = {
            "global_rgb": global_rgb,
            "local_rgb": local_rgb,
            "qpos": qpos,
            "actions": actions,
            "action_mask": action_mask,
        }
        if not self.history_steps and not self.future_qpos_horizons and not self.future_feature_horizons:
            return result
        with h5py.File(episode["path"], "r") as handle:
            data = handle["data"]
            agent = data["observation"]["agents"][f"panda_{arm}"]
            commanded = data["action"]["agents"][f"panda_{arm}"]["commanded"]
            images = data["observation"]["images"]
            if self.history_steps:
                indices, valid = [], []
                for offset in range(self.history_steps, 0, -1):
                    raw = time_index - offset * self.history_stride
                    valid.append(raw >= 0)
                    indices.append(max(0, raw))
                history_qpos = take_hdf5_rows(agent["qpos"], indices)
                history_action = take_hdf5_rows(commanded, indices)
                result["history_qpos"] = torch.from_numpy(
                    (history_qpos - self.stats["q_mean"]) / self.stats["q_std"]
                )
                result["history_actions"] = torch.from_numpy(
                    (history_action - self.stats["a_mean"]) / self.stats["a_std"]
                )
                result["history_mask"] = torch.as_tensor(valid, dtype=torch.bool)
                if self.include_history_views:
                    result["history_global_rgb"] = self._image_tensor(
                        [np.asarray(images["global"][index]) for index in indices]
                    )
                    result["history_local_rgb"] = self._image_tensor(
                        [np.asarray(images[f"agent_{arm}"][index]) for index in indices]
                    )
            if self.future_qpos_horizons:
                qpos_indices = [
                    min(time_index + value, episode["length"] - 1)
                    for value in self.future_qpos_horizons
                ]
                future_qpos = take_hdf5_rows(agent["qpos"], qpos_indices)
                result["future_qpos"] = torch.from_numpy(
                    (future_qpos - self.stats["q_mean"]) / self.stats["q_std"]
                )
            if self.future_feature_horizons:
                feature_indices = [
                    min(time_index + value, episode["length"] - 1)
                    for value in self.future_feature_horizons
                ]
                result["future_global_rgb"] = self._image_tensor(
                    [np.asarray(images["global"][index]) for index in feature_indices]
                )
                result["future_local_rgb"] = self._image_tensor(
                    [np.asarray(images[f"agent_{arm}"][index]) for index in feature_indices]
                )
        return result


def _encode_pooled_history(model, global_rgb, local_rgb, device, chunk: int = 16):
    batch, steps = global_rgb.shape[:2]
    global_rgb = global_rgb.reshape(batch * steps, *global_rgb.shape[2:])
    local_rgb = local_rgb.reshape(batch * steps, *local_rgb.shape[2:])
    rows = []
    with torch.no_grad():
        for start in range(0, len(global_rgb), chunk):
            stop = min(start + chunk, len(global_rgb))
            global_part = global_rgb[start:stop].float().div(255).to(device)
            local_part = local_rgb[start:stop].float().div(255).to(device)
            local_tokens = model._vision_tokens(local_part) + model.local_view
            global_tokens = model._vision_tokens(global_part) + model.global_view
            rows.append(torch.stack((local_tokens.mean(1), global_tokens.mean(1)), dim=1))
    return torch.cat(rows).reshape(batch, steps, 2, -1)


def _native_context(model, views, state_vec, latent):
    routes = model._pair_route(state_vec, views.parent_fused)
    memory = torch.cat(
        (state_vec.unsqueeze(1), model.z_proj(latent).unsqueeze(1), views.parent_fused),
        dim=1,
    )
    return CoreContext(
        views=views,
        observation=views.parent_fused,
        state_vec=state_vec,
        latent=latent,
        memory=memory,
        query=model.query.expand(state_vec.shape[0], -1, -1),
        dense_routes=model.last_dense_routes,
        sparse_routes=routes,
        provenance={"policy": "B9-CoreNative", "privileged_inputs": False},
    )


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("preflight", "screen", "selection"), required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config_path = Path(args.config).resolve(strict=True)
    config = load_r10_config(config_path)
    parent_path = Path(args.parent_checkpoint).resolve(strict=True)
    parent_sha = file_sha256(parent_path)
    if parent_sha != EXPECTED_PARENT_SHA256:
        raise ValueError(f"parent checkpoint SHA-256 mismatch: {parent_sha}")
    phase_updates = {
        "preflight": int(config["training"].get("preflight_updates", 2)),
        "screen": 10_000,
        "selection": 30_000,
    }
    target_updates = phase_updates[args.phase]
    if args.phase == "selection" and not args.resume:
        raise ValueError("selection requires the screen resume checkpoint")

    seed_everything(FORMAL_SEED)
    device = torch.device("cuda:0")
    saved = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_config = saved["config"]
    model = NoWristPAIRRoute(
        parent_config.get("state_dim", 9),
        parent_config.get("action_dim", 8),
        horizon=parent_config.get("horizon", 100),
        d_model=parent_config.get("d_model", 384),
        enc_layers=parent_config.get("enc_layers", 4),
        dec_layers=parent_config.get("dec_layers", 7),
        roles=parent_config.get("roles", 4),
        role_rank=parent_config.get("role_rank", 32),
        dino_model=parent_config["dino_model"],
    ).to(device)
    result = model.load_state_dict(saved["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(str(result))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    extension = build_perception_extension(config["bridge"]).to(device)
    model.register_perception_extension(extension)
    model.eval()
    extension.train()
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any(not name.startswith("perception_extension.") for name, _ in trainable):
        raise RuntimeError(f"trainable-parameter contract failed: {[name for name, _ in trainable]}")

    stats = saved["stats"]
    manifests = [Path(value).resolve(strict=True) for value in args.manifests]
    episodes = load_episodes(manifests)
    bridge = config["bridge"]
    history_steps = int(bridge.get("history_steps", 0))
    history_stride = int(bridge.get("history_stride", 1))
    future_qpos_horizons = tuple(int(value) for value in extension.future_qpos_horizons)
    future_feature_horizons = tuple(int(value) for value in extension.future_feature_horizons)
    dataset = BWAFrameDataset(
        episodes,
        parent_config.get("horizon", 100),
        stats,
        history_steps=history_steps,
        history_stride=history_stride,
        include_history_views=extension.requires_history_views,
        future_qpos_horizons=future_qpos_horizons,
        future_feature_horizons=future_feature_horizons,
    )
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    start_update = int(resume["update"]) if resume else 0
    if start_update >= target_updates:
        raise ValueError(f"resume update {start_update} is not below target {target_updates}")
    sampler = ExactFiveTaskBatchSampler(episodes, target_updates, FORMAL_SEED, start_update)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    optimizer = torch.optim.AdamW(
        [value for _, value in trainable],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    if resume:
        if resume["candidate_id"] != config["candidate_id"] or resume["parent_sha256"] != parent_sha:
            raise ValueError("resume identity differs")
        extension.load_state_dict(resume["extension"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    progress_path = output / "progress.jsonl"
    immutable = {
        "candidate_id": config["candidate_id"],
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "parent_checkpoint": str(parent_path),
        "parent_sha256": parent_sha,
        "parent_commit": config["parent_commit"],
        "trainable_parameters": [name for name, _ in trainable],
        "manifests": {str(path): file_sha256(path) for path in manifests},
        "episodes": len(episodes),
        "batch_size": FORMAL_BATCH,
        "seed": FORMAL_SEED,
        "precision": "bfloat16",
    }
    _atomic_json(immutable, output / "training_identity.json")

    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    weights = {str(key): float(value) for key, value in config["loss_weights"].items()}
    save_every = int(config["training"]["save_every"])
    log_every = int(config["training"]["log_every"])
    started = time.monotonic()
    last: dict[str, float] = {}

    def save(update: int, name: str) -> Path:
        checkpoint = output / "checkpoints" / name
        atomic_torch_save(
            {
                "schema_version": 1,
                "candidate_id": config["candidate_id"],
                "bridge_kind": bridge["kind"],
                "extension": extension.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "parent_sha256": parent_sha,
                "parent_commit": config["parent_commit"],
                "config": config,
                "last_metrics": last,
            },
            checkpoint,
        )
        return checkpoint

    update = start_update
    for update, batch in enumerate(loader, start=start_update + 1):
        current_global = batch["global_rgb"].float().div_(255).to(device, non_blocking=True)
        current_local = batch["local_rgb"].float().div_(255).to(device, non_blocking=True)
        qpos = batch["qpos"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        action_mask = batch["action_mask"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            views = model.encode_view_tokens(current_global, current_local)
            state_vec = model.state(qpos)
            latent = torch.zeros(
                (len(qpos), model.z_proj.in_features), device=device, dtype=state_vec.dtype
            )
            native_context = _native_context(model, views, state_vec, latent)
            native_prediction = model.decode_with_gates(native_context, native_context.sparse_routes)

        targets: dict[str, torch.Tensor] = {}
        deployment_context = None
        if history_steps:
            history_tokens = None
            if extension.requires_history_views:
                history_tokens = _encode_pooled_history(
                    model, batch["history_global_rgb"], batch["history_local_rgb"], device
                )
            history_qpos = batch["history_qpos"].to(device, non_blocking=True)
            history_actions = batch["history_actions"].to(device, non_blocking=True)
            history_mask = batch["history_mask"].to(device, non_blocking=True)
            deployment_context = CoreDeploymentContext(
                view_token_history=history_tokens,
                qpos_history=history_qpos,
                executed_action_history=history_actions,
                history_mask=history_mask,
            )
            targets.update(
                history_qpos=history_qpos,
                history_actions=history_actions,
                history_mask=history_mask,
            )
        if future_qpos_horizons:
            targets["future_qpos"] = batch["future_qpos"].to(device, non_blocking=True)
        if future_feature_horizons:
            targets["future_view_features"] = _encode_pooled_history(
                model, batch["future_global_rgb"], batch["future_local_rgb"], device
            ).detach()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context = model.encode_context(
                current_global,
                current_local,
                qpos,
                latent=latent,
                deployment_context=deployment_context,
                _views=views,
                _state_vec=state_vec,
            )
            prediction = model.decode_with_gates(context, context.sparse_routes)
            action_loss = (
                (prediction - actions).square().mean(-1) * action_mask
            ).sum() / action_mask.sum().clamp_min(1)
            imitation_loss = (prediction - native_prediction.detach()).square().mean()
            auxiliary_losses = extension.training_losses(context.auxiliary, targets)
            gate_loss = extension.perception_gate.float().square()
            loss_terms = {
                "action": action_loss,
                "parent_imitation": imitation_loss,
                "gate_l2": gate_loss,
                **auxiliary_losses,
            }
            unknown_weights = set(weights) - set(loss_terms)
            if unknown_weights:
                raise ValueError(f"loss_weights reference missing terms: {sorted(unknown_weights)}")
            total = sum(weights.get(name, 0.0) * value for name, value in loss_terms.items())
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite loss at update {update}")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [value for _, value in trainable], float(config["training"]["grad_clip"])
        )
        optimizer.step()
        extension.clamp_gate_()
        last = {
            "update": update,
            "loss": float(total.detach()),
            "grad_norm": float(grad_norm),
            "gate": float(torch.tanh(extension.perception_gate.detach())),
            **{name: float(value.detach()) for name, value in loss_terms.items()},
        }
        if update == start_update + 1 or update % log_every == 0:
            elapsed = time.monotonic() - started
            completed = update - start_update
            row = {
                **last,
                "phase": args.phase,
                "target_updates": target_updates,
                "updates_per_hour": completed / max(elapsed, 1e-6) * 3600,
                "eta_hours": (target_updates - update) * elapsed / max(completed, 1) / 3600,
                "gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
                "time": time.time(),
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
        if update % save_every == 0 or update == target_updates or stopping:
            latest = save(update, "checkpoint_latest.pt")
            if update == target_updates:
                save(update, f"checkpoint_{update:06d}.pt")
            print(json.dumps({"saved": str(latest), "update": update}), flush=True)
        if stopping:
            print(json.dumps({"stopped": True, "update": update}), flush=True)
            raise SystemExit(130)
    print(json.dumps({"complete": True, "phase": args.phase, "update": update}), flush=True)


if __name__ == "__main__":
    main()
