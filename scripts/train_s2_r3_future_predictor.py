#!/usr/bin/env python3
"""Train one S2-R3 off-path local future-prediction candidate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wam_multimodal import (  # noqa: E402
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
)
from scripts.train_static_rgb_act_moe import (  # noqa: E402
    _TaskBalancedBatchSampler,
    _append_jsonl,
    _atomic_torch_save,
    _capture_rng,
    _emit_stage,
    _git_commit,
    _load_yaml,
    _mapping,
    _restore_rng,
    _seed_everything,
    _sha256,
    _vision,
)
from train.s2_future_prediction import (  # noqa: E402
    encode_local_visual_targets,
    file_sha256,
    load_s2_artifact,
    masked_future_prediction_losses,
    normalized_state_delta,
    state_dict_sha256,
)
from train.s2_grouped_trajectory import (  # noqa: E402
    S2GroupedTrajectoryDataset,
    grouped_s2_batch,
)
from train.s2_model_registry import validate_s2_candidate  # noqa: E402


CHECKPOINT_FORMAT = "wam.robofactory.s2_r3.local_future_predictor.checkpoint/1"
RESUME_FORMAT = "wam.robofactory.s2_r3.local_future_predictor.resume/1"
FLOW_FORMAT = "wam.robofactory.agent_factorized_flow.checkpoint/1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--progress-log", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    raw = _load_yaml(config_path)
    candidate_id, model_kind, action_conditioning = validate_s2_candidate(
        _mapping(raw, "round")
    )
    training = _mapping(raw, "training")
    device = torch.device(args.device)
    _emit_stage("cuda_preflight", f"checking {device}")
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("S2-R3 training requires exactly one visible GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("S2-R3 training requires native BF16")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    seed = int(training.get("seed", 303))
    _seed_everything(seed)

    base_flow_path = (
        ROOT / str(_mapping(raw, "parent")["flow_checkpoint"])
    ).resolve(strict=True)
    base_flow = torch.load(base_flow_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(base_flow, Mapping)
        or base_flow.get("format_version") != FLOW_FORMAT
        or _mapping(base_flow, "method").get("action_generator")
        != "rectified_flow_cold"
    ):
        raise ValueError("S2-R3 parent must be the promoted S1-R1 cold Flow")
    base_flow_sha256 = file_sha256(base_flow_path)
    del base_flow

    artifact_path = (
        ROOT / str(_mapping(raw, "artifacts")["pca_statistics"])
    ).resolve(strict=True)
    artifact_sha256 = file_sha256(artifact_path)
    artifact = load_s2_artifact(artifact_path, device=device)
    _emit_stage("dataset_validation", "opening the five-task train split")
    dataset = _dataset(raw, split="train")
    _validate_artifact_dataset(artifact, dataset)
    _emit_stage(
        "dataset_ready",
        f"{len(dataset)} grouped windows across {len(dataset.contracts)} tasks",
    )

    batch_size = int(args.batch_size or training.get("batch_size", 1))
    updates = int(args.updates or training.get("updates", 10000))
    if batch_size <= 0 or updates <= 0:
        raise ValueError("batch size and updates must be positive")
    model_config = LocalFuturePredictorConfig.from_dict(
        dict(_mapping(raw, "model"))
    )
    pca = _mapping(raw, "pca")
    if (
        model_config.visual_grid_tokens
        != int(pca["grid_height"]) * int(pca["grid_width"])
    ):
        raise ValueError("model visual_grid_tokens disagrees with PCA grid")
    model = LocalActionConditionedFuturePredictor(model_config).to(device)
    initial_model_sha256 = state_dict_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates)
    _emit_stage("dinov3_load", "loading frozen verified DINOv3")
    vision = _vision(raw).to(device).eval()
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise RuntimeError("DINOv3 must remain frozen in S2-R3")

    checkpoint = _mapping(raw, "checkpoint")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (ROOT / str(checkpoint["output"])).resolve()
    )
    resume = (
        args.resume.expanduser().resolve()
        if args.resume is not None
        else (ROOT / str(checkpoint["resume"])).resolve()
    )
    progress_log = (
        args.progress_log.expanduser().resolve()
        if args.progress_log is not None
        else (ROOT / str(checkpoint["progress_log"])).resolve()
    )
    for path in (output, resume, progress_log):
        path.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if not args.no_resume and resume.is_file():
        _emit_stage("resume_load", f"loading {resume}")
        saved = torch.load(resume, map_location=device, weights_only=False)
        if saved.get("format_version") != RESUME_FORMAT:
            raise ValueError("resume file is not an S2-R3 predictor")
        if (
            saved.get("candidate_id") != candidate_id
            or saved.get("artifact_sha256") != artifact_sha256
            or saved.get("base_flow_sha256") != base_flow_sha256
            or saved.get("initial_model_sha256") != initial_model_sha256
        ):
            raise ValueError("S2-R3 resume identity differs from this run")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = int(saved["update"])
        _restore_rng(_mapping(saved, "rng"))
    else:
        _emit_stage("resume_load", "no resume checkpoint; starting from update 0")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed checkpoint {output}")

    workers = int(training.get("num_workers", 4))
    _emit_stage(
        "dataloader_start",
        f"starting {workers} workers; waiting for first grouped HDF5 batch",
    )
    loader = DataLoader(
        dataset,
        batch_sampler=_TaskBalancedBatchSampler(
            dataset,  # type: ignore[arg-type]
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
    save_interval = int(training.get("save_interval", 500))
    log_interval = int(training.get("log_interval", 20))
    started = time.perf_counter()
    model.train()
    for update in range(start + 1, updates + 1):
        grouped = grouped_s2_batch(next(iterator))
        if update == start + 1:
            _emit_stage(
                "optimizer_training",
                f"first grouped batch ready; updates {start + 1}..{updates}",
            )
        current_visual, target_visual = encode_local_visual_targets(
            vision,
            grouped,
            artifact,
            device=device,
            grid_height=int(pca["grid_height"]),
            grid_width=int(pca["grid_width"]),
        )
        current_state = grouped["current_state"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        actions = grouped["candidate_actions"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        valid_agents = grouped["valid_agent_mask"].to(
            device=device, dtype=torch.bool
        )
        state_valid = grouped["future_state_valid_mask"].to(
            device=device, dtype=torch.bool
        )
        visual_valid = grouped["future_agent_visual_valid_mask"].to(
            device=device, dtype=torch.bool
        )
        target_state = normalized_state_delta(
            grouped, artifact, device=device
        )
        if action_conditioning:
            action_mask = valid_agents
        else:
            actions = torch.zeros_like(actions)
            action_mask = torch.zeros_like(valid_agents)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted_state, predicted_visual = model(
                current_state,
                current_visual,
                actions,
                valid_agents,
                action_mask,
            )
            losses = masked_future_prediction_losses(
                predicted_state,
                target_state,
                state_valid,
                predicted_visual,
                target_visual,
                visual_valid,
            )
        losses["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(training.get("gradient_clip_norm", 1.0)),
        )
        optimizer.step()
        scheduler.step()
        if update == 1 or update % log_interval == 0 or update == updates:
            elapsed = max(time.perf_counter() - started, 1e-9)
            progress = {
                "event": "optimizer_step",
                "candidate_id": candidate_id,
                "program": "train_s2_r3_future_predictor.py",
                "update": update,
                "updates": updates,
                "loss": float(losses["loss"].detach()),
                "state_loss": float(losses["state"].detach()),
                "visual_loss": float(losses["visual"].detach()),
                "gradient_norm": float(gradient_norm),
                "updates_per_second": (update - start) / elapsed,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            print(json.dumps(progress, sort_keys=True), flush=True)
            _append_jsonl(progress_log, progress)
        if update % save_interval == 0 and update < updates:
            _atomic_torch_save(
                {
                    "format_version": RESUME_FORMAT,
                    "candidate_id": candidate_id,
                    "model_kind": model_kind,
                    "update": update,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": _capture_rng(),
                    "artifact_sha256": artifact_sha256,
                    "base_flow_sha256": base_flow_sha256,
                    "initial_model_sha256": initial_model_sha256,
                },
                resume,
            )

    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "update": updates,
        "method": {
            "round_id": "s2-r3",
            "candidate_id": candidate_id,
            "model_kind": model_kind,
            "future_scope": "local",
            "action_conditioning": action_conditioning,
            "action_generator": "rectified_flow_cold",
            "world_predictor_path": "strictly_off_path",
            "future_target_input": False,
        },
        "model_config": model_config.to_dict(),
        "model": model.state_dict(),
        "initial_model_sha256": initial_model_sha256,
        "training": dict(training),
        "data": {
            "summary": dataset.summary(),
            "manifests": [
                {
                    "task_id": contract.task_id,
                    "path": str(contract.manifest_path),
                    "sha256": contract.manifest_sha256,
                }
                for contract in dataset.contracts
            ],
        },
        "future_artifacts": artifact,
        "future_artifacts_sha256": artifact_sha256,
        "frozen_parent": {
            "flow_checkpoint": str(base_flow_path),
            "flow_checkpoint_sha256": base_flow_sha256,
            "dinov3_weights_sha256": vision.artifact_sha256,
            "dinov3_config_sha256": vision.config_sha256,
        },
        "source": {
            "git_commit": _git_commit(),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
        },
    }
    _atomic_torch_save(payload, output)
    resume.unlink(missing_ok=True)
    dataset.close()
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "sha256": file_sha256(output),
                "candidate_id": candidate_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _dataset(
    config: Mapping[str, object],
    *,
    split: str,
) -> S2GroupedTrajectoryDataset:
    data = _mapping(config, "data")
    manifests = [
        (ROOT / str(value)).resolve(strict=True)
        for value in data["manifests"]  # type: ignore[index]
    ]
    return S2GroupedTrajectoryDataset(
        manifests,
        split=split,
        stride=int(data.get("stride", 1)),
        hdf5_cache_size=int(data.get("hdf5_cache_size", 4)),
    )


def _validate_artifact_dataset(
    artifact: Mapping[str, object],
    dataset: S2GroupedTrajectoryDataset,
) -> None:
    data = artifact.get("data")
    manifests = data.get("manifests") if isinstance(data, Mapping) else None
    if not isinstance(manifests, list):
        raise ValueError("S2 artifact does not declare training manifests")
    expected = {
        contract.task_id: contract.manifest_sha256
        for contract in dataset.contracts
    }
    actual = {
        str(value.get("task_id")): str(value.get("sha256"))
        for value in manifests
        if isinstance(value, Mapping)
    }
    if actual != expected:
        raise ValueError("S2 artifact manifest identities differ from training data")


if __name__ == "__main__":
    raise SystemExit(main())
