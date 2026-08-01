#!/usr/bin/env python3
"""Train only the S3-R6 world-to-Flow adapter and bounded velocity gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wam_multimodal import ProtectedTeamFuturePredictor  # noqa: E402
from scripts.s3_r6_model_io import build_s3_r6_model  # noqa: E402
from scripts.train_s2_r4_future_predictor import (  # noqa: E402
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
    _task_runtime,
    _vision,
)
from train.s2_future_prediction import (  # noqa: E402
    encode_current_visual_context,
    file_sha256,
    load_s2_artifact,
    state_dict_sha256,
)
from train.s2_grouped_trajectory import grouped_s2_batch  # noqa: E402
from train.s3_model_registry import validate_s3_r6_candidate  # noqa: E402
from train.world_action_flow_training import (  # noqa: E402
    grouped_flow_matching_batch,
    grouped_masked_flow_mse,
)


CHECKPOINT_FORMAT = "wam.robofactory.s3_r6.world_action_flow.checkpoint/1"
RESUME_FORMAT = "wam.robofactory.s3_r6.world_action_flow.resume/1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--progress-log", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    raw = _load_yaml(config_path)
    micro_round, candidate_id, model_kind, future_scope, injection = (
        validate_s3_r6_candidate(_mapping(raw, "round"))
    )
    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("S3-R6 requires exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("S3-R6 requires native BF16")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    training = _mapping(raw, "training")
    seed = int(training.get("seed", 606))
    _seed_everything(seed)
    _emit_stage("parent_load", "loading immutable Flow/protected-own/R5-P0 parents")
    model, parent_identity = build_s3_r6_model(
        raw,
        device=device,
        future_scope=future_scope,
        injection=injection,
    )
    initial_parent_hashes = _parent_model_hashes(model)

    artifact_path = (ROOT / str(_mapping(raw, "artifacts")["pca_statistics"])).resolve(
        strict=True
    )
    artifact = load_s2_artifact(artifact_path, device=device)
    artifact_sha256 = file_sha256(artifact_path)
    _emit_stage("dataset_validation", "opening shared five-task grouped train split")
    dataset = _dataset(raw, split="train")
    _validate_artifact_dataset(artifact, dataset)
    batch_size = int(training.get("batch_size", 1))
    configured_updates = int(training.get("updates", 10000))
    updates = int(args.updates if args.updates is not None else configured_updates)
    if not injection:
        updates = 0
    if batch_size <= 0 or updates < 0:
        raise ValueError("batch size must be positive and updates non-negative")
    if injection and updates <= 0:
        raise ValueError("gated P1 requires a positive training budget")
    vision = _vision(raw).to(device).eval()
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise RuntimeError("DINOv3 must remain frozen in S3-R6")

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
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed checkpoint {output}")

    initial_adapter_hash = state_dict_sha256(model.adapter)
    start = 0
    optimizer = None
    scheduler = None
    parameters = model.trainable_parameters()
    if injection:
        if not parameters or any(
            parameter.requires_grad
            for module in (model.base_flow, model.future_predictor)
            for parameter in module.parameters()
        ):
            raise RuntimeError("S3 optimizer isolation is invalid")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training.get("learning_rate", 2e-4)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates)
        if not args.no_resume and resume.is_file():
            _emit_stage("resume_load", f"loading {resume}")
            saved = torch.load(resume, map_location=device, weights_only=False)
            if saved.get("format_version") != RESUME_FORMAT:
                raise ValueError("resume is not an S3-R6 adapter file")
            expected = {
                "micro_round": micro_round,
                "candidate_id": candidate_id,
                "model_kind": model_kind,
                "future_scope": future_scope,
                "parent_identity": parent_identity,
                "artifact_sha256": artifact_sha256,
                "initial_adapter_sha256": initial_adapter_hash,
            }
            if _mapping(saved, "identity") != expected:
                raise ValueError("S3-R6 resume identity differs from this candidate")
            model.load_adapter_state_dict(_mapping(saved, "adapter"))
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            start = int(saved["update"])
            _restore_rng(_mapping(saved, "rng"))
        else:
            _emit_stage("resume_load", "no resume checkpoint; starting update 0")

    pca = _mapping(raw, "pca")
    workers = int(training.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_sampler=_TaskBalancedBatchSampler(
            dataset,  # type: ignore[arg-type]
            batch_size=batch_size,
            first_update=max(start + 1, 1),
            final_update=max(updates, 1),
            seed=seed,
        ),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed + 10_000_000),
    )
    iterator = iter(loader)
    _emit_stage("first_batch", "reading one fixed batch for structural invariants")
    fixed_grouped = grouped_s2_batch(next(iterator))
    fixed_inputs = _model_inputs(
        vision,
        fixed_grouped,
        artifact,
        device=device,
        pca=pca,
    )
    own_before = _protected_own_output(model, fixed_inputs)
    gate_zero_difference = _gate_zero_difference(model, fixed_inputs)
    if gate_zero_difference != 0.0:
        raise RuntimeError("gate=0 does not exactly reproduce frozen base Flow")

    started = time.perf_counter()
    save_interval = int(training.get("save_interval", 500))
    log_interval = int(training.get("log_interval", 20))
    gradient_clip = float(training.get("gradient_clip_norm", 1.0))
    model.train(injection)
    for update in range(start + 1, updates + 1):
        grouped = fixed_grouped if update == start + 1 else grouped_s2_batch(next(iterator))
        inputs = (
            fixed_inputs
            if update == start + 1
            else _model_inputs(vision, grouped, artifact, device=device, pca=pca)
        )
        actions = grouped["candidate_actions"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        action_inputs, target_velocity, tau = grouped_flow_matching_batch(actions)
        assert optimizer is not None and scheduler is not None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted, diagnostics = model.velocity(
                inputs["raw_local"],
                inputs["state"],
                inputs["local_visual"],
                inputs["shared_visual"],
                action_inputs,
                tau,
                inputs["valid"],
            )
            loss = grouped_masked_flow_mse(
                predicted,
                target_velocity,
                inputs["valid"],
                grouped["action_valid_mask"].to(device=device, dtype=torch.bool),
            )
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, gradient_clip))
        optimizer.step()
        scheduler.step()
        if update == start + 1:
            _emit_stage("optimizer_training", f"adapter/gate update {update}/{updates}")
        if update == 1 or update % log_interval == 0 or update == updates:
            elapsed = max(time.perf_counter() - started, 1e-9)
            progress = {
                "event": "optimizer_step",
                "program": "train_s3_r6_world_action_flow.py",
                "micro_round": micro_round,
                "candidate_id": candidate_id,
                "update": update,
                "updates": updates,
                "loss": float(loss.detach()),
                "gate": float(model.adapter.bounded_gate().detach()),
                "residual_rms": float(diagnostics["residual_rms"]),
                "gradient_norm": gradient_norm,
                "updates_per_second": (update - start) / elapsed,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            print(json.dumps(progress, sort_keys=True), flush=True)
            _append_jsonl(progress_log, progress)
        if update % save_interval == 0 and update < updates:
            identity = {
                "micro_round": micro_round,
                "candidate_id": candidate_id,
                "model_kind": model_kind,
                "future_scope": future_scope,
                "parent_identity": parent_identity,
                "artifact_sha256": artifact_sha256,
                "initial_adapter_sha256": initial_adapter_hash,
            }
            _atomic_torch_save(
                {
                    "format_version": RESUME_FORMAT,
                    "identity": identity,
                    "update": update,
                    "adapter": model.adapter_state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": _capture_rng(),
                },
                resume,
            )

    own_after = _protected_own_output(model, fixed_inputs)
    own_max_diff = _maximum_difference(own_before, own_after)
    final_parent_hashes = _parent_model_hashes(model)
    current_file_hashes = {
        key: file_sha256(value)
        for key, value in {
            "flow_checkpoint_sha256": parent_identity["flow_checkpoint"],
            "protected_own_checkpoint_sha256": parent_identity[
                "protected_own_checkpoint"
            ],
            "protected_team_checkpoint_sha256": parent_identity[
                "protected_team_checkpoint"
            ],
        }.items()
    }
    file_hashes_unchanged = all(
        current_file_hashes[key] == parent_identity[key]
        for key in current_file_hashes
    )
    structural = {
        "protected_own_elementwise_exact": own_max_diff == 0.0,
        "protected_own_max_abs_diff": own_max_diff,
        "protected_parent_model_hashes_unchanged": (
            final_parent_hashes == initial_parent_hashes
        ),
        "parent_files_unchanged": file_hashes_unchanged,
        "parents_excluded_from_optimizer": True,
        "gate_zero_base_action_max_abs_diff": gate_zero_difference,
        "gate_zero_base_action_elementwise_exact": gate_zero_difference == 0.0,
    }
    if not all(
        structural[key]
        for key in (
            "protected_own_elementwise_exact",
            "protected_parent_model_hashes_unchanged",
            "parent_files_unchanged",
            "parents_excluded_from_optimizer",
        )
    ):
        raise RuntimeError("S3-R6 structural parent invariant failed")
    generation = dict(_mapping(raw, "generation"))
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "update": updates,
        "method": {
            "round_id": "s3-r6",
            "micro_round": micro_round,
            "candidate_id": candidate_id,
            "model_kind": model_kind,
            "future_scope": future_scope,
            "injection": injection,
            "action_generator": "rectified_flow_cold_gated_residual",
            "candidate_action_contract": "clean_endpoint_each_solver_evaluation",
            "flow_training_scope": "five_task_from_scratch_per_candidate",
            "trainable_modules": ["world_to_flow_adapter", "velocity_gate"]
            if injection
            else [],
        },
        "adapter_config": model.adapter.config.to_dict(),
        "adapter": model.adapter_state_dict() if injection else {},
        "gate": {
            "parameterization": "max_gate*tanh(alpha)",
            "alpha": float(model.adapter.gate_alpha.detach()),
            "value": float(model.adapter.bounded_gate().detach()) if injection else 0.0,
            "initialized_at_zero": True,
        },
        "generation": generation,
        "training": {**dict(training), "updates_executed": updates},
        "task_runtime": _task_runtime(dataset.source),
        "parent_identity": parent_identity,
        "structural_invariants": structural,
        "future_artifacts_sha256": artifact_sha256,
        "vision": dict(_mapping(raw, "vision")),
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
                "model_kind": model_kind,
                "updates": updates,
                "gate": payload["gate"]["value"],
                "protected_own_exact": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _model_inputs(
    vision: torch.nn.Module,
    grouped: Mapping[str, Tensor],
    artifact: Mapping[str, object],
    *,
    device: torch.device,
    pca: Mapping[str, object],
) -> dict[str, Tensor]:
    raw_local, local_visual, shared_visual = encode_current_visual_context(
        vision,
        grouped,
        artifact,
        device=device,
        grid_height=int(pca["grid_height"]),
        grid_width=int(pca["grid_width"]),
    )
    return {
        "raw_local": raw_local,
        "local_visual": local_visual,
        "shared_visual": shared_visual,
        "state": grouped["current_state"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "valid": grouped["valid_agent_mask"].to(device=device, dtype=torch.bool),
    }


def _protected_own_output(
    model: torch.nn.Module,
    inputs: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor]:
    predictor = model.future_predictor  # type: ignore[attr-defined]
    actions = torch.zeros(
        inputs["state"].shape[0],
        inputs["state"].shape[1],
        model.base_flow.config.horizon,  # type: ignore[attr-defined]
        model.base_flow.config.action_dim,  # type: ignore[attr-defined]
        device=inputs["state"].device,
    )
    with torch.no_grad():
        if isinstance(predictor, ProtectedTeamFuturePredictor):
            output = predictor(
                inputs["state"],
                inputs["local_visual"],
                inputs["shared_visual"],
                actions,
                inputs["valid"],
            )
            return output.own_state.detach().cpu(), output.own_visual.detach().cpu()
        return tuple(
            value.detach().cpu()
            for value in predictor(
                inputs["state"],
                inputs["local_visual"],
                actions,
                inputs["valid"],
                inputs["valid"],
            )
        )  # type: ignore[return-value]


def _gate_zero_difference(model: torch.nn.Module, inputs: Mapping[str, Tensor]) -> float:
    shape = (
        inputs["state"].shape[0],
        inputs["state"].shape[1],
        model.base_flow.config.horizon,  # type: ignore[attr-defined]
        model.base_flow.config.action_dim,  # type: ignore[attr-defined]
    )
    actions = torch.zeros(shape, device=inputs["state"].device)
    tau = torch.full(
        (shape[0],), 0.5, device=actions.device, dtype=actions.dtype
    )
    with torch.no_grad():
        observed = model.velocity(  # type: ignore[attr-defined]
            inputs["raw_local"],
            inputs["state"],
            inputs["local_visual"],
            inputs["shared_visual"],
            actions,
            tau,
            inputs["valid"],
            force_gate_zero=True,
        )[0]
        flat_time = tau[:, None].expand(-1, shape[1]).reshape(-1)
        expected = model.base_flow(  # type: ignore[attr-defined]
            inputs["raw_local"].flatten(0, 1),
            inputs["state"].flatten(0, 1),
            actions.flatten(0, 1),
            flat_time,
        )[0].reshape(shape)
        expected = expected * inputs["valid"][:, :, None, None].to(expected)
    return float((observed - expected).abs().max())


def _maximum_difference(
    left: tuple[Tensor, Tensor], right: tuple[Tensor, Tensor]
) -> float:
    return max(float((a - b).abs().max()) for a, b in zip(left, right, strict=True))


def _parent_model_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {
        "base_flow": state_dict_sha256(model.base_flow),  # type: ignore[attr-defined]
        "future_predictor": state_dict_sha256(  # type: ignore[attr-defined]
            model.future_predictor
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
