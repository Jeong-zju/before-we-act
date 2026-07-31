#!/usr/bin/env python3
"""Train only S2-R5 peer/shared modules above an immutable R4-P0 path."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import io
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
    LocalFuturePredictorConfig,
    ProtectedTeamFuturePredictor,
    ProtectedTeamFuturePredictorConfig,
)
from scripts.train_s2_r4_future_predictor import (  # noqa: E402
    CHECKPOINT_FORMAT as R4_CHECKPOINT_FORMAT,
    FLOW_FORMAT,
    _dataset,
    _validate_artifact_dataset,
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
    encode_shared_visual_targets,
    file_sha256,
    load_s2_artifact,
    masked_future_prediction_losses,
    normalized_state_delta,
    state_dict_sha256,
)
from train.s2_grouped_trajectory import grouped_s2_batch  # noqa: E402
from train.s2_model_registry import validate_s2_r5_candidate  # noqa: E402
from train.s2_r4_future_prediction import (  # noqa: E402
    masked_peer_future_prediction_losses,
    masked_shared_future_prediction_losses,
)


CHECKPOINT_FORMAT = "wam.robofactory.s2_r5.protected_team.checkpoint/1"
RESUME_FORMAT = "wam.robofactory.s2_r5.protected_team.resume/1"


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
    candidate_id, model_kind, team_mixer = validate_s2_r5_candidate(
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
        raise RuntimeError("S2-R5 training requires exactly one visible GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("S2-R5 training requires native BF16")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    seed = int(training.get("seed", 505))
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
        raise ValueError("S2-R5 parent must retain the promoted S1-R1 Flow")
    base_flow_sha256 = file_sha256(base_flow_path)
    del base_flow

    p0_path = (
        ROOT / str(_mapping(raw, "parent")["protected_p0_checkpoint"])
    ).resolve(strict=True)
    p0_file_sha256 = file_sha256(p0_path)
    p0 = torch.load(p0_path, map_location="cpu", weights_only=False)
    p0_method = _mapping(p0, "method")
    if (
        p0.get("format_version") != R4_CHECKPOINT_FORMAT
        or p0_method.get("round_id") != "s2-r4"
        or p0_method.get("candidate_id") != "P0"
        or p0_method.get("model_kind") != "s2_r4_local_action_conditioned"
        or p0_method.get("team_shared") is not False
    ):
        raise ValueError("S2-R5 requires the accepted R4-P0 local checkpoint")
    local_config = LocalFuturePredictorConfig.from_dict(
        dict(_mapping(p0, "model_config"))
    )
    if local_config != LocalFuturePredictorConfig.from_dict(
        dict(_mapping(raw, "model"))
    ):
        raise ValueError("S2-R5 local config differs from protected R4-P0")
    team_config = ProtectedTeamFuturePredictorConfig.from_dict(
        dict(_mapping(raw, "team_model"))
    )
    if team_config.team_mixer != team_mixer:
        raise ValueError("round and team_model mixer disagree")
    model = ProtectedTeamFuturePredictor(local_config, team_config).to(device)
    model.load_protected_own(p0["model"])
    p0_model_sha256 = state_dict_sha256(model.protected_own)
    protected_initial = {
        key: value.detach().cpu().clone()
        for key, value in model.protected_own.state_dict().items()
    }
    del p0
    team_parameters = model.trainable_parameters()
    if not team_parameters:
        raise RuntimeError("S2-R5 team trainable set is empty")
    if any(parameter.requires_grad for parameter in model.protected_own.parameters()):
        raise RuntimeError("protected P0 parameters unexpectedly trainable")
    initial_team_sha256 = _state_mapping_sha256(model.team_state_dict())
    optimizer = torch.optim.AdamW(
        team_parameters,
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )

    artifact_path = (
        ROOT / str(_mapping(raw, "artifacts")["pca_statistics"])
    ).resolve(strict=True)
    artifact_sha256 = file_sha256(artifact_path)
    artifact = load_s2_artifact(artifact_path, device=device)
    _emit_stage("dataset_validation", "opening shared five-task train split")
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates)
    _emit_stage("dinov3_load", "loading frozen verified DINOv3")
    vision = _vision(raw).to(device).eval()
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise RuntimeError("DINOv3 must remain frozen in S2-R5")

    checkpoint = _mapping(raw, "checkpoint")
    output = (
        args.output.expanduser().resolve()
        if args.output
        else (ROOT / str(checkpoint["output"])).resolve()
    )
    resume = (
        args.resume.expanduser().resolve()
        if args.resume
        else (ROOT / str(checkpoint["resume"])).resolve()
    )
    progress_log = (
        args.progress_log.expanduser().resolve()
        if args.progress_log
        else (ROOT / str(checkpoint["progress_log"])).resolve()
    )
    for path in (output, resume, progress_log):
        path.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    identity = {
        "candidate_id": candidate_id,
        "model_kind": model_kind,
        "team_mixer": team_mixer,
        "artifact_sha256": artifact_sha256,
        "base_flow_sha256": base_flow_sha256,
        "protected_p0_file_sha256": p0_file_sha256,
        "protected_p0_model_sha256": p0_model_sha256,
        "initial_team_sha256": initial_team_sha256,
    }
    if not args.no_resume and resume.is_file():
        _emit_stage("resume_load", f"loading {resume}")
        saved = torch.load(resume, map_location=device, weights_only=False)
        if saved.get("format_version") != RESUME_FORMAT:
            raise ValueError("resume is not an S2-R5 protected-team file")
        if _mapping(saved, "identity") != identity:
            raise ValueError("S2-R5 resume identity differs from this run")
        model.load_team_state_dict(_mapping(saved, "team_model"))
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = int(saved["update"])
        _restore_rng(_mapping(saved, "rng"))
    else:
        _emit_stage("resume_load", "no resume checkpoint; starting update 0")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {output}")

    workers = int(training.get("num_workers", 4))
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
    gradient_clip = float(training.get("gradient_clip_norm", 1.0))
    started = time.perf_counter()
    model.train()
    pca = _mapping(raw, "pca")
    for update in range(start + 1, updates + 1):
        grouped = grouped_s2_batch(next(iterator))
        if update == start + 1:
            _emit_stage(
                "optimizer_training",
                f"first batch ready; team-only updates {start + 1}..{updates}",
            )
        current_visual, target_visual = encode_local_visual_targets(
            vision,
            grouped,
            artifact,
            device=device,
            grid_height=int(pca["grid_height"]),
            grid_width=int(pca["grid_width"]),
        )
        current_shared, target_shared, _ = encode_shared_visual_targets(
            vision,
            grouped,
            artifact,
            device=device,
            grid_height=int(pca["grid_height"]),
            grid_width=int(pca["grid_width"]),
        )
        current_state = grouped["current_state"].to(device=device, dtype=torch.float32)
        actions = grouped["candidate_actions"].to(device=device, dtype=torch.float32)
        valid_agents = grouped["valid_agent_mask"].to(device=device, dtype=torch.bool)
        state_valid = grouped["future_state_valid_mask"].to(device=device, dtype=torch.bool)
        visual_valid = grouped["future_agent_visual_valid_mask"].to(
            device=device, dtype=torch.bool
        )
        shared_valid = grouped["future_shared_visual_valid_mask"].to(
            device=device, dtype=torch.bool
        )
        target_state = normalized_state_delta(grouped, artifact, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(
                current_state,
                current_visual,
                current_shared,
                actions,
                valid_agents,
            )
            own = masked_future_prediction_losses(
                prediction.own_state,
                target_state,
                state_valid,
                prediction.own_visual,
                target_visual,
                visual_valid,
            )
            peer = masked_peer_future_prediction_losses(
                prediction.peer_state,
                target_state,
                state_valid,
                prediction.peer_visual,
                target_visual,
                visual_valid,
                valid_agents,
            )
            shared = masked_shared_future_prediction_losses(
                prediction.shared_visual,
                target_shared,
                shared_valid,
                valid_agents,
            )
            loss = peer["loss"] + shared["loss"]
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(team_parameters, gradient_clip))
        optimizer.step()
        scheduler.step()
        if update == 1 or update % log_interval == 0 or update == updates:
            elapsed = max(time.perf_counter() - started, 1e-9)
            progress = {
                "event": "optimizer_step",
                "program": "train_s2_r5_protected_team.py",
                "candidate_id": candidate_id,
                "team_mixer": team_mixer,
                "update": update,
                "updates": updates,
                "loss": float(loss.detach()),
                "own_monitor_loss": float(own["loss"].detach()),
                "peer_loss": float(peer["loss"].detach()),
                "shared_loss": float(shared["loss"].detach()),
                "gradient_norm": gradient_norm,
                "updates_per_second": (update - start) / elapsed,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            print(json.dumps(progress, sort_keys=True), flush=True)
            _append_jsonl(progress_log, progress)
        if update % save_interval == 0 and update < updates:
            _atomic_torch_save(
                {
                    "format_version": RESUME_FORMAT,
                    "identity": identity,
                    "update": update,
                    "team_model": model.team_state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": _capture_rng(),
                },
                resume,
            )

    if file_sha256(p0_path) != p0_file_sha256:
        raise RuntimeError("protected P0 checkpoint changed during S2-R5")
    if any(
        not torch.equal(value.cpu(), protected_initial[key])
        for key, value in model.protected_own.state_dict().items()
    ):
        raise RuntimeError("protected P0 state changed during S2-R5")
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "update": updates,
        "method": {
            "round_id": "s2-r5",
            "candidate_id": candidate_id,
            "model_kind": model_kind,
            "team_mixer": team_mixer,
            "protected_own": True,
            "future_scope": "team_shared",
            "action_conditioning": True,
            "action_generator": "rectified_flow_cold",
            "world_predictor_path": "strictly_off_path",
            "future_target_input": False,
        },
        "model_config": local_config.to_dict(),
        "team_model_config": team_config.to_dict(),
        "team_model": model.team_state_dict(),
        "parameter_audit": model.parameter_audit(),
        "runtime_profile": {
            "measured_training_updates_per_second": updates
            / max(time.perf_counter() - started, 1e-9),
            "batch_size": batch_size,
            "mixer_invocations_per_sample": 2,
        },
        "initial_team_sha256": initial_team_sha256,
        "protected_parent": {
            "checkpoint": str(p0_path),
            "checkpoint_sha256": p0_file_sha256,
            "model_sha256": p0_model_sha256,
            "state_unchanged_after_training": True,
            "optimizer_excluded": True,
        },
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
                "protected_p0_unchanged": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _state_mapping_sha256(value: Mapping[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(
        {
            name: tensor.detach().cpu()
            for name, tensor in sorted(value.items())
        },
        buffer,
    )
    return hashlib.sha256(buffer.getvalue()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
