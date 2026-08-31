"""Train the unchanged CARE distributional belief head on physical BiCoord branches.

The main checkpoint is fit on all branch families. Three cross-fit shadow heads
are produced only by the preregistered deployment run and are reserved for OOF
conformal calibration; they are never deployment candidates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from before_we_act.care_belief import CAREBeliefConfig, CAREBeliefHead, care_training_loss
from .bcore_data import BICOORD_CARE_MEMORY_TOKENS, BICOORD_CARE_MEMORY_WIDTH
from .config import ACTION_DIM, ACTION_HORIZON, EFFECTIVE_BATCH, TASKS
from .data import load_normalization_receipt
from .stage_common import (
    artifact, assert_common_paths, atomic_json, canonical_sha256, common_parser,
    publish_result, require_stage_result, sha256_file,
)

FORMAT = "before-we-act.care-bicoord-training/1"
VARIANTS = ("care", "reactive_only", "replay_only", "capacity")
OOF_FOLDS = (0, 1, 2)
OOF_VARIANT = "care"
OOF_SEED = 20260904
LEARNING_RATE = 3e-4
ETA_MIN = 3e-6


def _load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping) or value.get("format_version") != "before-we-act.care-bicoord-prepared-data/1":
        raise ValueError("wrong BiCoord CARE prepared format")
    required = ("memory", "memory_mask", "candidate_chunks", "targets", "hard_safety", "usable", "task_id", "snapshot_ids", "action_std", "reference_checkpoint_sha256")
    if any(key not in value for key in required):
        raise ValueError("prepared CARE tensor is incomplete")
    n = int(value["memory"].shape[0])
    if n < 1:
        raise ValueError("prepared CARE tensor is empty")
    if tuple(value["memory"].shape[1:]) != (BICOORD_CARE_MEMORY_TOKENS, BICOORD_CARE_MEMORY_WIDTH):
        raise ValueError("prepared CARE memory contract differs")
    action_std = np.asarray(value["action_std"], dtype=np.float32)
    if action_std.shape != (ACTION_DIM,) or not np.isfinite(action_std).all() or np.any(action_std <= 0):
        raise ValueError("prepared CARE action_std is not a positive native 7-D vector")
    receipt = value.get("normalization_receipt")
    receipt_sha = value.get("normalization_receipt_sha256")
    if receipt is None or receipt_sha is None:
        raise ValueError("prepared CARE tensor lacks normalization provenance")
    receipt_path = Path(str(receipt))
    if not receipt_path.is_file() or sha256_file(receipt_path) != str(receipt_sha):
        raise ValueError("prepared CARE normalization receipt changed")
    normalization = load_normalization_receipt(receipt_path, require_formal=True)
    if not np.array_equal(np.asarray(normalization["action_std"], np.float32), action_std):
        raise ValueError("prepared CARE action_std differs from full-corpus normalization")
    return dict(value)


def fold_assignments(prepared: Mapping[str, Any], folds: int = 3) -> torch.Tensor:
    """Assign each family by within-task ordinal modulo three."""
    if folds != 3:
        raise ValueError("CARE OOF protocol is frozen to three folds")
    task_ids = torch.as_tensor(prepared["task_id"], dtype=torch.long).flatten()
    counts = [0] * len(TASKS)
    result: list[int] = []
    for task in task_ids.tolist():
        if task < 0 or task >= len(TASKS):
            raise ValueError("prepared CARE task id is out of range")
        result.append(counts[task] % folds)
        counts[task] += 1
    if any(count == 0 for count in counts):
        raise ValueError("OOF fold assignment requires every BiCoord task")
    return torch.tensor(result, dtype=torch.long)


def fold_assignment_receipt(prepared: Mapping[str, Any]) -> dict[str, Any]:
    folds = fold_assignments(prepared)
    values: dict[str, Any] = {
        "protocol": "within_task_ordinal_modulo_3",
        "folds": 3,
        "task_ids": [int(x) for x in torch.as_tensor(prepared["task_id"]).tolist()],
        "fold_ids": [int(x) for x in folds.tolist()],
    }
    values["sha256"] = canonical_sha256(values)
    return values


def _batch(
    prepared: Mapping[str, Any], update: int, seed: int, device: torch.device,
    family_indices: Sequence[int] | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    task_ids = torch.as_tensor(prepared["task_id"], dtype=torch.long)
    usable = torch.as_tensor(prepared["usable"], dtype=torch.bool)
    pool = list(range(int(task_ids.shape[0]))) if family_indices is None else [int(x) for x in torch.as_tensor(family_indices).flatten().tolist()]
    if not pool:
        raise ValueError("CARE fold has no training families")
    rng = random.Random(int(seed) + int(update) * 1_000_003)
    by_task = {task: [idx for idx in pool if int(task_ids[idx]) == task] for task in range(len(TASKS))}
    if any(not values for values in by_task.values()):
        raise ValueError("CARE balanced sampler requires every task in training pool")
    extras = [((update - 1) * 12 + i) % len(TASKS) for i in range(12)]
    counts = Counter({task: 2 for task in range(len(TASKS))}); counts.update(extras)
    rows: list[tuple[int, int, int]] = []
    for task in range(len(TASKS)):
        for _ in range(counts[task]):
            family = rng.choice(by_task[task])
            horizons = torch.nonzero(usable[family], as_tuple=False).flatten().tolist()
            if not horizons:
                raise ValueError(f"CARE family {family} has no usable horizon")
            rows.append((family, rng.choice(horizons), rng.randrange(2)))
    rng.shuffle(rows)
    if len(rows) != EFFECTIVE_BATCH:
        raise AssertionError("CARE effective batch drift")
    family = torch.tensor([row[0] for row in rows], dtype=torch.long)
    horizon = torch.tensor([row[1] for row in rows], dtype=torch.long)
    repeat = torch.tensor([row[2] for row in rows], dtype=torch.long)
    return {
        "family_index": family,
        "memory": torch.as_tensor(prepared["memory"])[family].float().to(device),
        "memory_mask": torch.as_tensor(prepared["memory_mask"])[family].bool().to(device),
        "candidate_chunks": torch.as_tensor(prepared["candidate_chunks"])[family].float().to(device),
        "target": torch.as_tensor(prepared["targets"])[family, horizon, :, repeat].float().to(device),
        "hard_safety": torch.as_tensor(prepared["hard_safety"])[family, horizon, :, repeat].float().to(device),
        "horizon_index": horizon.to(device),
        "task_id": task_ids[family].to(device),
    }


@torch.no_grad()
def _metrics(model: CAREBeliefHead, prepared: Mapping[str, Any], device: torch.device, family_indices: Sequence[int] | torch.Tensor | None = None) -> dict[str, float]:
    model.eval()
    indices = torch.arange(int(prepared["memory"].shape[0])) if family_indices is None else torch.as_tensor(family_indices, dtype=torch.long)
    losses: list[float] = []; regrets: list[float] = []; correct = 0; rows = 0
    for first in range(0, len(indices), 32):
        family = indices[first:first + 32]; usable = torch.as_tensor(prepared["usable"])[family]
        for horizon in range(4):
            selected = family[usable[:, horizon]]
            if not len(selected):
                continue
            memory = torch.as_tensor(prepared["memory"])[selected].float().to(device)
            mask = torch.as_tensor(prepared["memory_mask"])[selected].bool().to(device)
            candidates = torch.as_tensor(prepared["candidate_chunks"])[selected].float().to(device)
            h = torch.full((len(selected),), horizon, dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(memory, mask, candidates, h)
            target = torch.as_tensor(prepared["targets"])[selected, horizon, :, 0].float().to(device)
            hard = torch.as_tensor(prepared["hard_safety"])[selected, horizon, :, 0].float().to(device)
            loss, _ = care_training_loss(output, target, hard, model.config.variant)
            losses.append(float(loss)); component = 0 if model.config.variant == "replay_only" else 2
            prediction = output.quantiles[:, :, component, 2]; truth = target[:, :, component]
            choice = prediction.argmax(1); best = truth.argmax(1)
            regrets.extend((truth.max(1).values - truth.gather(1, choice[:, None]).squeeze(1)).cpu().tolist())
            correct += int((choice == best).sum()); rows += len(selected)
    model.train()
    return {"loss": float(np.mean(losses)) if losses else float("nan"), "mean_regret": float(np.mean(regrets)) if regrets else float("nan"), "top1_accuracy": correct / max(rows, 1), "rows": rows}


def _config(prepared: Mapping[str, Any], variant: str) -> CAREBeliefConfig:
    action_std = tuple(float(x) for x in torch.as_tensor(prepared["action_std"]).tolist())
    return CAREBeliefConfig(d_model=BICOORD_CARE_MEMORY_WIDTH, action_dim=ACTION_DIM, action_horizon=ACTION_HORIZON, action_tokens=16, action_width=128, heads=8, layers=2, candidates=6, outcome_components=3, variant=variant, action_std=action_std)


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _train_one(*, prepared: Mapping[str, Any], prepared_path: Path, config: CAREBeliefConfig, seed: int, updates: int, device: torch.device, root: Path, role: str, family_indices: Sequence[int] | torch.Tensor | None, fold: int | None, fold_receipt: Mapping[str, Any], auto_resume: bool, evaluate: bool, training_seed: int | None = None) -> tuple[Path, Path, list[dict[str, Any]]]:
    root.mkdir(parents=True, exist_ok=True); latest = root / "checkpoint_latest.pt"; progress = root / "progress.jsonl"
    rng_seed = int(seed if training_seed is None else training_seed)
    random.seed(rng_seed); np.random.seed(rng_seed % 2**32); torch.manual_seed(rng_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(rng_seed)
    model = CAREBeliefHead(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=ETA_MIN)
    start = 0; evaluations: list[dict[str, Any]] = []
    family_list = list(range(int(prepared["memory"].shape[0]))) if family_indices is None else [int(x) for x in torch.as_tensor(family_indices).flatten().tolist()]
    family_hash = canonical_sha256(family_list); prepared_sha = sha256_file(prepared_path)
    if auto_resume and latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        expected = {"format_version": FORMAT, "variant": config.variant, "seed": int(seed), "training_seed": rng_seed, "prepared_sha256": prepared_sha, "role": role, "family_indices_sha256": family_hash}
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f"CARE resume provenance differs at {key}")
        model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        start = int(saved.get("update", 0)); evaluations = list(saved.get("evaluations", []))
    if start > updates:
        raise ValueError("CARE checkpoint update exceeds requested updates")
    started = time.time()
    for update in range(start + 1, updates + 1):
        values = _batch(prepared, update, rng_seed, device, family_indices); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(values["memory"], values["memory_mask"], values["candidate_chunks"], values["horizon_index"])
            loss, pieces = care_training_loss(output, values["target"], values["hard_safety"], config.variant)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite CARE loss at {update}")
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"non-finite CARE gradient at {update}")
        optimizer.step(); scheduler.step()
        if update == 1 or update % 20 == 0 or update == updates:
            row = {"update": update, "updates": updates, "variant": config.variant, "seed": int(seed), "training_seed": rng_seed, "role": role, "loss": float(loss), "gradient_norm": float(gradient), "lr": scheduler.get_last_lr()[0], "elapsed_seconds": time.time() - started, **{key: float(value) for key, value in pieces.items()}}
            with progress.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n"); stream.flush(); os.fsync(stream.fileno())
            print(json.dumps(row, sort_keys=True), flush=True)
        if update == updates or update % min(1000, updates) == 0:
            if evaluate:
                evaluations.append({"update": update, "offline_all_families": _metrics(model, prepared, device)})
            payload = {"format_version": FORMAT, "benchmark_adapter": "BiCoord", "method_family": "CARE", "policy_family": "CAREBeliefHead", "reference_policy": "B-core/TUNE", "reference_checkpoint": prepared.get("reference_checkpoint"), "reference_checkpoint_sha256": prepared.get("reference_checkpoint_sha256"), "variant": config.variant, "seed": int(seed), "training_seed": rng_seed, "update": update, "target_updates": int(updates), "role": role, "oof_shadow": role == "oof_shadow", "oof_shadow_fold": fold, "held_out_fold": fold, "config": config.to_dict(), "model": _cpu_state_dict(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "evaluations": evaluations, "prepared": str(prepared_path.resolve()), "prepared_sha256": prepared_sha, "family_indices_sha256": family_hash, "family_indices": family_list, "all_families_for_training": family_indices is None, "held_out_families": 0 if family_indices is None else int(prepared["memory"].shape[0]) - len(family_list), "fold_assignment": dict(fold_receipt), "learning_rate": LEARNING_RATE, "cosine_eta_min": ETA_MIN, "normalization_receipt": prepared.get("normalization_receipt"), "normalization_receipt_sha256": prepared.get("normalization_receipt_sha256")}
            temporary = latest.with_name(f".{latest.name}.{os.getpid()}.tmp"); torch.save(payload, temporary); os.replace(temporary, latest)
    if not latest.is_file():
        raise RuntimeError(f"CARE {role} produced no checkpoint")
    status = root / "status.json"; final_metrics = evaluations[-1].get("offline_all_families") if evaluations else None
    atomic_json(status, {"schema": "before-we-act.bicoord.care-training-status/1", "status": "PASSED_SMOKE" if updates <= 10 else "COMPLETED", "role": role, "oof_shadow": role == "oof_shadow", "oof_shadow_fold": fold, "variant": config.variant, "seed": int(seed), "training_seed": rng_seed, "update": int(updates), "selected_update": int(updates), "selected_checkpoint": str(latest.resolve()), "selected_checkpoint_sha256": sha256_file(latest), "metrics": final_metrics, "all_families_for_training": family_indices is None, "held_out_families": 0 if family_indices is None else int(prepared["memory"].shape[0]) - len(family_list), "fold": fold, "fold_assignment_sha256": fold_receipt.get("sha256"), "normalization_receipt_sha256": prepared.get("normalization_receipt_sha256")})
    return latest, status, evaluations


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args, need_dataset=True)
    smoke = args.operation == "smoke-train" or bool(getattr(args, "smoke", False))
    if args.operation == "formal-train" and bool(getattr(args, "smoke", False)):
        raise ValueError("formal CARE training cannot be marked smoke")
    require_stage_result(args.run, "branch_signal_gate_smoke" if smoke else "branch_signal_gate", config_sha256=args.config_sha256)
    updates = int(args.updates); variant = str(getattr(args, "variant", "care")); seed = int(getattr(args, "seed", OOF_SEED)); batch = int(getattr(args, "global_batch", EFFECTIVE_BATCH)); oof_fold = getattr(args, "oof_shadow_fold", None)
    if variant not in VARIANTS: raise ValueError(variant)
    if batch != EFFECTIVE_BATCH: raise ValueError("CARE global batch is frozen to 48")
    if smoke and not 1 <= updates <= 10: raise ValueError("CARE smoke is capped at ten updates")
    if not smoke and updates != 4000: raise ValueError("formal CARE belief training is fixed at 4000 updates")
    prepared_path = args.run / "artifacts" / ("prepared_branches_smoke.pt" if smoke else "prepared_branches.pt"); prepared = _load(prepared_path); fold_receipt = fold_assignment_receipt(prepared)
    manifest = prepared.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("prepared CARE tensor lacks its smoke/formal manifest")
    if bool(manifest.get("smoke")) is not smoke:
        raise ValueError("prepared CARE tensor smoke/formal namespace differs from training operation")
    expected_families = len(TASKS) if smoke else len(TASKS) * 30
    if int(torch.as_tensor(prepared["memory"]).shape[0]) != expected_families:
        raise ValueError(
            f"prepared CARE family coverage differs for {'smoke' if smoke else 'formal'}: "
            f"{int(torch.as_tensor(prepared['memory']).shape[0])} != {expected_families}"
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); config = _config(prepared, variant)
    folds = torch.as_tensor(fold_receipt["fold_ids"], dtype=torch.long)
    if oof_fold is not None:
        # Explicit OOF jobs are scheduled independently by the supervisor on
        # three GPUs.  The public seed remains the preregistered 20260904;
        # ``training_seed`` only decorrelates initializations across folds.
        oof_fold = int(oof_fold)
        if smoke or variant != OOF_VARIANT or seed != OOF_SEED or updates != 4000 or oof_fold not in OOF_FOLDS:
            raise ValueError("OOF shadow jobs are fixed to care/seed 20260904, folds 0..2, update 4000")
        train_indices = torch.nonzero(folds != oof_fold, as_tuple=False).flatten()
        held_indices = torch.nonzero(folds == oof_fold, as_tuple=False).flatten()
        root = args.run / "artifacts" / "belief_train" / "oof_shadow" / f"fold_{oof_fold}" / f"seed_{seed}"
        latest, status, evaluations = _train_one(prepared=prepared, prepared_path=prepared_path, config=config, seed=seed, training_seed=seed + 1000 + oof_fold, updates=updates, device=device, root=root, role="oof_shadow", family_indices=train_indices, fold=oof_fold, fold_receipt=fold_receipt, auto_resume=bool(args.auto_resume), evaluate=False)
        train_list = [int(x) for x in train_indices.tolist()]; held_list = [int(x) for x in held_indices.tolist()]
        return publish_result(args, stage="belief_train", include_model_contract=True, artifacts=[artifact(latest, kind="oof_shadow_checkpoint"), artifact(status, kind="oof_shadow_status"), artifact(root / "progress.jsonl", kind="oof_shadow_progress")], variant=variant, seed=seed, training_seed=seed + 1000 + oof_fold, updates=updates, checkpoint=str(latest.resolve()), checkpoint_sha256=sha256_file(latest), reference_checkpoint_sha256=str(prepared["reference_checkpoint_sha256"]), oof_shadow=True, oof_shadow_fold=oof_fold, oof_shadow_train_families=int(len(train_indices)), oof_shadow_train_indices_sha256=canonical_sha256(train_list), oof_shadow_held_out_families=int(len(held_indices)), oof_shadow_held_out_indices=held_list, oof_shadow_held_out_indices_sha256=canonical_sha256(held_list), oof_shadow_total_families=int(len(train_indices) + len(held_indices)), oof_shadow_complete_partition=True, all_families_for_training=False, held_out_families=int(len(held_indices)), learning_rate=LEARNING_RATE, cosine_eta_min=ETA_MIN, normalization_receipt=str(prepared["normalization_receipt"]), normalization_receipt_sha256=str(prepared["normalization_receipt_sha256"]), fold_assignment_sha256=fold_receipt["sha256"], oof_calibration_role="shadow_only")
    root = args.run / "artifacts" / ("belief_smoke_train" if smoke else "belief_train") / variant / f"seed_{seed}"
    latest, status, evaluations = _train_one(prepared=prepared, prepared_path=prepared_path, config=config, seed=seed, updates=updates, device=device, root=root, role="deployment_main", family_indices=None, fold=None, fold_receipt=fold_receipt, auto_resume=bool(args.auto_resume), evaluate=True)
    artifacts = [artifact(latest, kind="belief_checkpoint"), artifact(status, kind="training_status"), artifact(root / "progress.jsonl", kind="training_progress")]
    selected = evaluations[-1]["offline_all_families"] if evaluations else {}
    return publish_result(args, stage="belief_smoke_train" if smoke else "belief_train", include_model_contract=True, artifacts=artifacts, variant=variant, seed=seed, training_seed=seed, updates=updates, checkpoint=str(latest.resolve()), checkpoint_sha256=sha256_file(latest), reference_checkpoint_sha256=str(prepared["reference_checkpoint_sha256"]), offline_metrics=selected, all_families_for_training=True, held_out_families=0, learning_rate=LEARNING_RATE, cosine_eta_min=ETA_MIN, normalization_receipt=str(prepared["normalization_receipt"]), normalization_receipt_sha256=str(prepared["normalization_receipt_sha256"]), fold_assignment_sha256=fold_receipt["sha256"], oof_calibration_role="deployment_main_only", oof_shadow=False, oof_shadow_fold=None)


def main(argv: list[str] | None = None) -> int:
    parser = common_parser(__doc__, ("smoke-train", "formal-train")); parser.add_argument("--updates", type=int, required=True); parser.add_argument("--global-batch", type=int, default=EFFECTIVE_BATCH); parser.add_argument("--variant", choices=VARIANTS, default="care"); parser.add_argument("--seed", type=int, default=OOF_SEED); parser.add_argument("--oof-shadow-fold", type=int, choices=OOF_FOLDS); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args(argv); run(args); return 0


if __name__ == "__main__": raise SystemExit(main())


__all__ = ["ETA_MIN", "LEARNING_RATE", "OOF_FOLDS", "OOF_SEED", "OOF_VARIANT", "VARIANTS", "fold_assignment_receipt", "fold_assignments", "run"]
