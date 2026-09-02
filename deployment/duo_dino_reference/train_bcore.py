"""Train one formal DuoBench PredictiveTeamBeliefPolicy (B-core) seed.

The action backbone is frozen in the B0-H context cache.  Only the predictive
team-belief core and its direct belief residual are optimized here; export
re-attaches those weights to a real :class:`PredictiveTeamBeliefPolicy` and
physically removes the privileged teacher branch.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.predictive_team_belief_policy import PredictiveTeamBeliefPolicy
from before_we_act.predictive_team_belief_training import (
    TeamBeliefExperiment,
    paired_permutation,
)
from before_we_act.team_belief.losses import (
    TeamBeliefLossWeights,
    compute_team_belief_losses,
)
from before_we_act.temporal_history_data import sha256_file
from .bcore_data import (
    BCORE_DEPLOYMENT_FORMAT,
    BCORE_SEEDS,
    BCORE_TRAINING_FORMAT,
    BCORE_UPDATES,
    DATA_SEED,
    DUO_BELIEF_CONFIG,
    DUO_CARE_MEMORY_SEMANTICS,
    DUO_CARE_MEMORY_TOKENS,
    DUO_CARE_MEMORY_WIDTH,
    DuoPairedSituationBatchSampler,
    DuoTeamBeliefDataset,
    fixed_diagnostic_requests,
    validate_b0h_payload,
)
from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SHA256,
)
from .data import ACTION_HORIZON, ACTION_LAG_ROWS, EFFECTIVE_BATCH, TASKS, load_duo_episodes
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID


EVAL_EVERY = 5_000
DEFAULT_LR = 2.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-4

# These are the MARS B-core weights projected without changing CARE's objective
# family.  The temporal anchor count/frequency is Duo-specific, not a method
# change.
WEIGHTS = TeamBeliefLossWeights(
    action=1.0,
    action_posterior_kl=0.0,
    teacher_alignment=0.1,
    future_latent=0.01,
    teacher_reconstruction=0.01,
    teammate_delta=0.1,
    teammate_action=0.1,
    exchange_consistency=0.05,
    anti_collapse=0.01,
    action_pairing=1.0,
    action_pairing_margin_fraction=0.1,
    action_pairing_margin_cap=0.01,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _device_batch(raw: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in raw.items()
    }


def _shuffle_permutation(task: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Deterministic within-task/phase negative for the pairing objective."""

    result = torch.arange(len(task), device=task.device)
    for task_value in torch.unique(task).tolist():
        phases = torch.unique(phase[task == task_value]).tolist()
        for phase_value in phases:
            rows = torch.nonzero(
                (task == task_value) & (phase == phase_value), as_tuple=False
            ).flatten()
            if len(rows) > 1:
                result[rows] = rows.roll(1)
    return result


def _json_config() -> dict[str, Any]:
    value = asdict(DUO_BELIEF_CONFIG)
    value["future_offsets_steps"] = list(DUO_BELIEF_CONFIG.future_offsets_steps)
    value["future_offsets_seconds"] = list(DUO_BELIEF_CONFIG.future_offsets_seconds)
    return value


@torch.no_grad()
def evaluate_offline(
    model: TeamBeliefExperiment,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate only cached/offline labels; no simulator result enters here."""

    model.eval()
    values: dict[str, list[float]] = {
        "b0h": [],
        "b_core": [],
        "b_shuffle": [],
        "direct_reactive": [],
    }
    auxiliary: dict[str, list[float]] = {}
    residual_target: list[torch.Tensor] = []
    residual_output: list[torch.Tensor] = []
    for raw in loader:
        batch = _device_batch(raw, device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(batch)
            negative = _shuffle_permutation(batch["task_index"], batch["phase_bin"])
            shuffled_residual, _ = model.belief_residual(
                batch["decoded_action_hidden"],
                output.candidate.belief.mu[negative],
                output.candidate.belief.sigma[negative],
                output.candidate.belief.reliability[negative],
            )
            shuffled = batch["base_action"] + shuffled_residual
        predictions = {
            "b0h": batch["base_action"],
            "b_core": output.candidate.prediction,
            "b_shuffle": shuffled,
            "direct_reactive": output.direct_prediction,
        }
        for name, prediction in predictions.items():
            error = (prediction - batch["action"]).float().square().mean(-1)
            row = (error * batch["action_mask"]).sum(-1) / batch[
                "action_mask"
            ].sum(-1).clamp_min(1)
            values[name].extend(row.cpu().tolist())
        residual_target.append(
            (batch["action"] - batch["base_action"])[batch["action_mask"]]
            .float()
            .cpu()
        )
        residual_output.append(
            output.candidate.belief_residual[batch["action_mask"]].float().cpu()
        )
        losses = compute_team_belief_losses(
            output.candidate,
            batch["action"],
            batch["action_mask"],
            batch["teammate_delta"],
            batch["teacher_future_anchor_mask"],
            batch["teammate_action"],
            batch["teammate_action_mask"],
            WEIGHTS,
        )
        for key, value in losses.items():
            auxiliary.setdefault(key, []).append(float(value.detach().cpu()))
    return {
        "macro": {
            key: float(np.mean(rows)) if rows else float("nan")
            for key, rows in values.items()
        },
        "auxiliary": {
            key: float(np.mean(rows)) if rows else float("nan")
            for key, rows in auxiliary.items()
        },
        "residual_target_rms": float(
            torch.cat(residual_target).square().mean().sqrt()
        )
        if residual_target
        else 0.0,
        "residual_output_rms": float(
            torch.cat(residual_output).square().mean().sqrt()
        )
        if residual_output
        else 0.0,
        "rows": len(values["b0h"]),
    }


def export_deployment(
    training_checkpoint: Path,
    b0h_checkpoint: Path,
    output: Path,
    *,
    prepared_data: Path,
    bcore_cache: Path,
    dino_model: str | None = None,
) -> dict[str, Any]:
    """Attach trained B-core tensors to a real deployment policy."""

    training = torch.load(training_checkpoint, map_location="cpu", weights_only=False)
    b0h = torch.load(b0h_checkpoint, map_location="cpu", weights_only=False)
    b0h_config = validate_b0h_payload(b0h)
    model_name = str(dino_model or b0h_config.get("dino_model") or "")
    if not model_name:
        raise ValueError("B-core deployment needs the frozen DINO model path")
    policy = PredictiveTeamBeliefPolicy(
        DUO_BELIEF_CONFIG,
        state_dim=8,
        action_dim=8,
        horizon=ACTION_HORIZON,
        d_model=384,
        enc_layers=int(b0h_config.get("enc_layers", 4)),
        dec_layers=int(b0h_config.get("dec_layers", 7)),
        roles=int(b0h_config.get("roles", 4)),
        role_rank=int(b0h_config.get("role_rank", 32)),
        history_layers=int(b0h_config.get("history_layers", 2)),
        dino_model=model_name,
        image_height=int(b0h_config.get("image_height", 224)),
        image_width=int(b0h_config.get("image_width", 224)),
        strict_dino_contract=True,
        include_teacher=False,
        residual_safety={"enabled": False},
    )
    incompatible = policy.load_state_dict(b0h["model"], strict=False)
    expected_missing = {
        key
        for key in policy.state_dict()
        if key.startswith(("belief_core.", "direct_belief_residual."))
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "B0-H backbone could not be attached to PredictiveTeamBeliefPolicy: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    state = training.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("B-core training checkpoint has no model state")
    core_state = {
        key.removeprefix("belief_core."): value
        for key, value in state.items()
        if str(key).startswith("belief_core.")
        and not str(key).startswith("belief_core.teacher_branch.")
    }
    residual_state = {
        key.removeprefix("belief_residual."): value
        for key, value in state.items()
        if str(key).startswith("belief_residual.")
    }
    if not core_state or not residual_state:
        raise ValueError("B-core checkpoint is missing belief core/residual tensors")
    policy.belief_core.load_state_dict(core_state, strict=True)
    policy.direct_belief_residual.load_state_dict(residual_state, strict=True)
    policy.eval()
    manifest_path = prepared_data / "manifest.json"
    cache_receipt = bcore_cache / "cache_receipt.json"
    payload = {
        "format": BCORE_DEPLOYMENT_FORMAT,
        "format_version": BCORE_DEPLOYMENT_FORMAT,
        "model": dict(policy.deployment_state_dict()),
        "stats": b0h.get("stats", {}),
        "update": int(training.get("update", -1)),
        "policy_family": "PredictiveTeamBeliefPolicy",
        "reference_policy_family": "PredictiveTeamBeliefPolicy",
        "method_family": "CARE",
        "architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
        "benchmark_adapter": "DuoBench",
        "vision": "dinov3_vitb16_frozen",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "action_encoding": "absolute_joint7_binary_gripper1",
        "strictly_decentralized": True,
        "strict_local": True,
        "act_provider_allowed": False,
        "teacher_present": False,
        "strict_dino_contract": True,
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
        "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
        "care_memory_width": DUO_CARE_MEMORY_WIDTH,
        "all_550_demonstrations": True,
        "config": {
            **dict(b0h_config),
            "policy_family": "PredictiveTeamBeliefPolicy",
            "reference_policy_family": "PredictiveTeamBeliefPolicy",
            "method_family": "CARE",
            "architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
            "benchmark_adapter": "DuoBench",
            "vision": "dinov3_vitb16_frozen",
            "vision_backbone": "dinov3_vitb16_frozen",
            "image_preprocess_id": IMAGE_PREPROCESS_ID,
            "dino_normalization_id": DINO_NORMALIZATION_ID,
            "action_encoding": "absolute_joint7_binary_gripper1",
            "action_lag_rows": ACTION_LAG_ROWS,
            "strictly_decentralized": True,
            "strict_local": True,
            "act_provider_allowed": False,
            "teacher_present": False,
            "strict_dino_contract": True,
            "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
            "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
            "memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
            "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
            "care_memory_width": DUO_CARE_MEMORY_WIDTH,
            "all_550_demonstrations": True,
            "n2_config": _json_config(),
            "policy_variant": "predictive_team_belief",
            "source_b0h_checkpoint": str(b0h_checkpoint.resolve()),
            "source_b0h_checkpoint_sha256": sha256_file(b0h_checkpoint),
            "source_bcore_training_checkpoint": str(training_checkpoint.resolve()),
            "source_bcore_training_checkpoint_sha256": sha256_file(training_checkpoint),
            "bcore_cache_receipt_sha256": sha256_file(cache_receipt),
            "prepared_manifest_sha256": sha256_file(manifest_path),
            "residual_safety": {"enabled": False},
        },
        "source_b0h_checkpoint_sha256": sha256_file(b0h_checkpoint),
        "bcore_cache_receipt_sha256": sha256_file(cache_receipt),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "training_checkpoint_sha256": sha256_file(training_checkpoint),
        "created_at_utc": _now(),
    }
    _atomic_save(output, payload)
    return payload


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--bcore-cache", type=Path, required=True)
    parser.add_argument("--b0h-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=BCORE_UPDATES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=EFFECTIVE_BATCH)
    parser.add_argument("--save-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--stage", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.stage == "formal" and args.updates != BCORE_UPDATES:
        raise ValueError("formal Duo B-core training is frozen at 120000 updates")
    if args.stage == "smoke" and not 1 <= args.updates <= 10:
        raise ValueError("B-core smoke training is capped at ten updates")
    if args.seed not in BCORE_SEEDS and args.stage == "formal":
        raise ValueError(f"formal B-core seed must be one of {BCORE_SEEDS}")
    if args.batch_size != EFFECTIVE_BATCH:
        raise ValueError("Duo B-core effective batch is frozen at 48")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    device = torch.device(
        "cuda:" + os.environ.get("LOCAL_RANK", "0")
        if torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    episodes = load_duo_episodes(args.prepared_data, require_formal=True)
    dataset = DuoTeamBeliefDataset(
        args.prepared_data,
        episodes,
        args.visual_cache,
        args.bcore_cache,
        cache_limit=max(8, args.workers * 4),
    )
    sampler = DuoPairedSituationBatchSampler(
        episodes,
        updates=BCORE_UPDATES,
        data_seed=DATA_SEED,
        start_update=0,
    )
    latest = args.output / "checkpoint_latest.pt"
    saved = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
    elif latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
    start = int(saved.get("update", 0)) if saved else 0
    if not 0 <= start < args.updates:
        if start == args.updates:
            # A completed run can be safely re-exported without another update.
            if not (args.output / "deployment_checkpoint.pt").is_file():
                export_deployment(
                    latest,
                    args.b0h_checkpoint,
                    args.output / "deployment_checkpoint.pt",
                    prepared_data=args.prepared_data,
                    bcore_cache=args.bcore_cache,
                    dino_model=args.dino_model,
                )
            return
        raise ValueError("resume update is outside requested budget")
    sampler.start_update = start
    if saved:
        sampler.validate_cursor(saved["sample_cursor"])

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    diagnostic_requests = fixed_diagnostic_requests(episodes)
    diagnostic_batches = [
        diagnostic_requests[index : index + args.batch_size]
        for index in range(0, len(diagnostic_requests), args.batch_size)
    ]
    diagnostic = DataLoader(
        dataset,
        batch_sampler=diagnostic_batches,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = TeamBeliefExperiment(DUO_BELIEF_CONFIG).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 1.0
        if step < 80_000
        else 0.1,
    )
    evaluations: list[dict[str, Any]] = []
    if saved:
        expected_provenance = saved.get("provenance", {})
        current_provenance = {
            "seed": args.seed,
            "b0h_checkpoint_sha256": sha256_file(args.b0h_checkpoint),
            "bcore_cache_receipt_sha256": sha256_file(
                args.bcore_cache / "cache_receipt.json"
            ),
            "prepared_manifest_sha256": sha256_file(
                args.prepared_data / "manifest.json"
            ),
            "policy_training_split": "all_550_demonstrations",
            "config": _json_config(),
        }
        if expected_provenance != current_provenance:
            raise ValueError("Duo B-core resume provenance drift")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        evaluations = list(saved.get("evaluations", []))
    provenance = {
        "seed": args.seed,
        "b0h_checkpoint_sha256": sha256_file(args.b0h_checkpoint),
        "bcore_cache_receipt_sha256": sha256_file(
            args.bcore_cache / "cache_receipt.json"
        ),
        "prepared_manifest_sha256": sha256_file(args.prepared_data / "manifest.json"),
        "policy_training_split": "all_550_demonstrations",
        "config": _json_config(),
    }
    _atomic_json(
        args.output / "config.json",
        {
            "format": BCORE_TRAINING_FORMAT,
            "policy_family": "PredictiveTeamBeliefPolicy",
            "reference_policy_family": "PredictiveTeamBeliefPolicy",
            "method_family": "CARE",
            "benchmark_adapter": "DuoBench",
            "vision": "dinov3_vitb16_frozen",
            "vision_backbone": "dinov3_vitb16_frozen",
            "image_preprocess_id": IMAGE_PREPROCESS_ID,
            "dino_normalization_id": DINO_NORMALIZATION_ID,
            "action_encoding": "absolute_joint7_binary_gripper1",
            "strictly_decentralized": True,
            "strict_local": True,
            "strict_dino_contract": True,
            "act_provider_allowed": False,
            "all_550_demonstrations": True,
            "seed": args.seed,
            "protocol_updates": BCORE_UPDATES,
            "update_target": args.updates,
            "effective_batch": EFFECTIVE_BATCH,
            "tasks": list(TASKS),
            "n2_config": _json_config(),
            "source_b0h_checkpoint_sha256": provenance["b0h_checkpoint_sha256"],
            "bcore_cache_receipt_sha256": provenance["bcore_cache_receipt_sha256"],
        },
    )
    _atomic_json(
        args.output / "status.json",
        {
            "status": "TRAINING",
            "seed": args.seed,
            "update": start,
            "target_updates": args.updates,
            "policy_family": "PredictiveTeamBeliefPolicy",
            "benchmark_adapter": "DuoBench",
            "started_at_utc": _now(),
        },
    )
    started = time.time()
    last_metrics: dict[str, Any] = {}
    for update, raw in enumerate(loader, start=start + 1):
        if update > args.updates:
            break
        step_seed = args.seed + 10_000_019 * update
        random.seed(step_seed)
        np.random.seed(step_seed % (2**32))
        torch.manual_seed(step_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(step_seed)
        batch = _device_batch(raw, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(batch)
            partner = paired_permutation(batch["pair_id"])
            swapped = replace(
                output.candidate,
                belief=replace(
                    output.candidate.belief,
                    mu=output.candidate.belief.mu[partner],
                ),
            )
            negative = _shuffle_permutation(batch["task_index"], batch["phase_bin"])
            cf_residual, _ = model.belief_residual(
                batch["decoded_action_hidden"],
                output.candidate.belief.mu[negative],
                output.candidate.belief.sigma[negative],
                output.candidate.belief.reliability[negative],
            )
            residual_target = batch["action"] - batch["base_action"]
            losses = compute_team_belief_losses(
                output.candidate,
                batch["action"],
                batch["action_mask"],
                batch["teammate_delta"],
                batch["teacher_future_anchor_mask"],
                batch["teammate_action"],
                batch["teammate_action_mask"],
                WEIGHTS,
                swapped_output=swapped,
                counterfactual_prediction=batch["base_action"] + cf_residual,
                counterfactual_residual_target=residual_target[negative],
                counterfactual_action_mask=batch["action_mask"][negative],
            )
            direct = (
                (output.direct_prediction - batch["action"]).float().square().mean(-1)
                * batch["action_mask"]
            ).sum() / batch["action_mask"].sum().clamp_min(1)
            loss = losses["total"] + direct
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Duo B-core loss at update {update}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"non-finite Duo B-core gradient at update {update}")
        optimizer.step()
        scheduler.step()
        last_metrics = {
            key: float(value.detach().cpu()) for key, value in losses.items()
        }
        last_metrics.update(
            {
                "direct_reactive": float(direct.detach().cpu()),
                "combined": float(loss.detach().cpu()),
                "gradient_norm": float(gradient.detach().cpu()),
                "learning_rate": scheduler.get_last_lr()[0],
                "update": update,
                "target_updates": args.updates,
                "elapsed_seconds": time.time() - started,
            }
        )
        if update == start + 1 or update % args.log_every == 0 or update == args.updates:
            print(json.dumps(last_metrics, sort_keys=True), flush=True)
            with (args.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(last_metrics, sort_keys=True) + "\n")
            _atomic_json(args.output / "heartbeat.json", last_metrics)
            _atomic_json(
                args.output / "status.json",
                {
                    "status": "TRAINING",
                    "seed": args.seed,
                    "update": update,
                    "target_updates": args.updates,
                    "policy_family": "PredictiveTeamBeliefPolicy",
                    "benchmark_adapter": "DuoBench",
                    **{
                        key: value
                        for key, value in last_metrics.items()
                        if key in ("combined", "gradient_norm", "learning_rate")
                    },
                },
            )
        should_eval = (
            update % max(1, args.eval_every) == 0 or update == args.updates
        )
        if should_eval:
            metrics = evaluate_offline(model, diagnostic, device)
            evaluations.append({"update": update, "validation": metrics})
            print(json.dumps({"evaluation": evaluations[-1]}, sort_keys=True), flush=True)
        if update % max(1, args.save_every) == 0 or update == args.updates:
            checkpoint = {
                "format": BCORE_TRAINING_FORMAT,
                "format_version": BCORE_TRAINING_FORMAT,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update": update,
                "evaluations": evaluations,
                "last_metrics": last_metrics,
                "sample_cursor": sampler.cursor_receipt(update),
                "provenance": provenance,
                "config": {
                    "policy_family": "PredictiveTeamBeliefPolicy",
                    "reference_policy_family": "PredictiveTeamBeliefPolicy",
                    "method_family": "CARE",
                    "architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
                    "benchmark_adapter": "DuoBench",
                    "vision": "dinov3_vitb16_frozen",
                    "vision_backbone": "dinov3_vitb16_frozen",
                    "image_preprocess_id": IMAGE_PREPROCESS_ID,
                    "dino_normalization_id": DINO_NORMALIZATION_ID,
                    "action_encoding": "absolute_joint7_binary_gripper1",
                    "strictly_decentralized": True,
                    "strict_local": True,
                    "strict_dino_contract": True,
                    "act_provider_allowed": False,
                    "all_550_demonstrations": True,
                    "seed": args.seed,
                    "protocol_updates": BCORE_UPDATES,
                    "n2_config": _json_config(),
                    "source_b0h_checkpoint_sha256": provenance[
                        "b0h_checkpoint_sha256"
                    ],
                    "bcore_cache_receipt_sha256": provenance[
                        "bcore_cache_receipt_sha256"
                    ],
                },
            }
            _atomic_save(args.output / "checkpoint_latest.pt", checkpoint)
            _atomic_save(args.output / f"checkpoint_{update:06d}.pt", checkpoint)
    if args.stage == "smoke":
        _atomic_json(
            args.output / "status.json",
            {
                "status": "PASSED_SMOKE",
                "seed": args.seed,
                "update": args.updates,
                "target_updates": args.updates,
                "policy_family": "PredictiveTeamBeliefPolicy",
                "benchmark_adapter": "DuoBench",
                "completed_at_utc": _now(),
            },
        )
        return
    selected_update = args.updates
    if evaluations:
        selected_update = min(
            evaluations,
            key=lambda row: float(row["validation"]["macro"]["b_core"]),
        )["update"]
    selected_checkpoint = args.output / f"checkpoint_{int(selected_update):06d}.pt"
    if not selected_checkpoint.is_file():
        selected_checkpoint = args.output / "checkpoint_latest.pt"
    export_deployment(
        selected_checkpoint,
        args.b0h_checkpoint,
        args.output / "deployment_checkpoint.pt",
        prepared_data=args.prepared_data,
        bcore_cache=args.bcore_cache,
        dino_model=args.dino_model,
    )
    _atomic_json(
        args.output / "checkpoint_receipt.json",
        {
            "schema": "before-we-act.duobench.dino-bcore-training-checkpoint/1",
            "status": "PASSED",
            "format": BCORE_TRAINING_FORMAT,
            "seed": args.seed,
            "update": args.updates,
            "policy_family": "PredictiveTeamBeliefPolicy",
            "reference_policy_family": "PredictiveTeamBeliefPolicy",
            "method_family": "CARE",
            "architecture": "PredictiveTeamBeliefPolicy_direct_belief_residual",
            "benchmark_adapter": "DuoBench",
            "vision_backbone": "dinov3_vitb16_frozen",
            "image_preprocess_id": IMAGE_PREPROCESS_ID,
            "dino_normalization_id": DINO_NORMALIZATION_ID,
            "action_encoding": "absolute_joint7_binary_gripper1",
            "strictly_decentralized": True,
            "strict_dino_contract": True,
            "strict_local": True,
            "act_provider_allowed": False,
            "all_550_demonstrations": True,
            "source_b0h_checkpoint_sha256": sha256_file(args.b0h_checkpoint),
            "deployment_checkpoint_sha256": sha256_file(
                args.output / "deployment_checkpoint.pt"
            ),
            "created_at_utc": _now(),
        },
    )
    _atomic_json(
        args.output / "status.json",
        {
            "status": "COMPLETED",
            "seed": args.seed,
            "update": args.updates,
            "target_updates": args.updates,
            "selected_update": int(selected_update),
            "selected_checkpoint": str(selected_checkpoint.resolve()),
            "deployment_checkpoint": str(
                (args.output / "deployment_checkpoint.pt").resolve()
            ),
            "deployment_checkpoint_sha256": sha256_file(
                args.output / "deployment_checkpoint.pt"
            ),
            "policy_family": "PredictiveTeamBeliefPolicy",
            "benchmark_adapter": "DuoBench",
            "completed_at_utc": _now(),
        },
    )


if __name__ == "__main__":
    main()


__all__ = ["evaluate_offline", "export_deployment"]
