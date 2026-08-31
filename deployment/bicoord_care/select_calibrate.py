"""Select the preregistered CARE head and calibrate it from three-fold OOF data.

No closed-loop outcome and no in-sample prediction is used here.  The main
``care/20260904`` checkpoint is fixed before looking at branch outcomes; the
three shadow checkpoints are used only to produce family-level simultaneous
conformal scores.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from before_we_act.care_belief import CAREBeliefConfig, CAREBeliefHead
from .bcore_data import BICOORD_CARE_MEMORY_TOKENS, BICOORD_CARE_MEMORY_WIDTH
from .config import ACTION_DIM, ACTION_HORIZON, TASKS
from .data import load_normalization_receipt
from .stage_common import (
    artifact,
    assert_common_paths,
    atomic_json,
    common_parser,
    publish_result,
    require_stage_result,
    sha256_file,
)
from .train_belief import (
    FORMAT as TRAINING_FORMAT,
    OOF_FOLDS,
    OOF_SEED,
    OOF_VARIANT,
    fold_assignment_receipt,
)


DEPLOYMENT_FORMAT = "before-we-act.bicoord.care-deployment/1"
REGISTERED_CALIBRATION = {
    "selector_delta": 0.0,
    "hard_safety_probability_max": 0.25,
    "nominal_simultaneous_coverage": 0.9,
    "primary_horizon": 16,
}


def _flatten_artifacts(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            yield value
        for child in value.values():
            yield from _flatten_artifacts(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_artifacts(child)


def _verified_artifacts(dep: Mapping[str, Any], kinds: set[str]) -> list[Path]:
    result: list[Path] = []
    for row in _flatten_artifacts(dep):
        if row.get("kind") not in kinds:
            continue
        path = Path(str(row.get("path", "")))
        if path.is_file() and isinstance(row.get("sha256"), str) and sha256_file(path) == row["sha256"]:
            result.append(path.resolve())
    return list(dict.fromkeys(result))


def _prepared(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format_version") != "before-we-act.care-bicoord-prepared-data/1":
        raise ValueError("wrong BiCoord prepared branch format")
    for key in ("memory", "memory_mask", "candidate_chunks", "targets", "hard_safety", "usable", "task_id", "action_std", "normalization_receipt_sha256", "reference_checkpoint_sha256"):
        if key not in payload:
            raise ValueError(f"prepared CARE artifact lacks {key}")
    return payload


def finite_sample_quantile(scores: Sequence[float] | np.ndarray, coverage: float = 0.9) -> float:
    """The finite-sample upper conformal order statistic.

    ``ceil((n+1)*coverage)`` is used rather than NumPy's asymptotic
    interpolation.  This guarantees at least the registered coverage for the
    family-level score under exchangeability.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("conformal scores must be non-empty and finite")
    if not 0.0 < float(coverage) <= 1.0:
        raise ValueError("conformal coverage must be in (0,1]")
    rank = min(int(values.size), max(1, int(math.ceil((values.size + 1) * float(coverage)))))
    return float(np.sort(values, kind="mergesort")[rank - 1])


def _load_model(path: Path, device: torch.device) -> tuple[CAREBeliefHead, Mapping[str, Any]]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, Mapping) or saved.get("format_version") != TRAINING_FORMAT:
        raise ValueError(f"invalid CARE training checkpoint: {path}")
    config = CAREBeliefConfig.from_mapping(saved["config"])
    if config.d_model != BICOORD_CARE_MEMORY_WIDTH or config.action_dim != ACTION_DIM or config.action_horizon != ACTION_HORIZON or config.candidates != 6 or config.outcome_components != 3:
        raise ValueError("CARE checkpoint model contract differs")
    model = CAREBeliefHead(config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, saved


def _designated_main(dep: Mapping[str, Any], prepared_sha: str) -> Path:
    """Find exactly the fixed preregistered deployment-main checkpoint."""
    paths: list[Path] = []
    for worker in dep.get("workers", []):
        if not isinstance(worker, Mapping) or worker.get("variant") != OOF_VARIANT or int(worker.get("seed", -1)) != OOF_SEED:
            continue
        for path in _verified_artifacts(worker, {"belief_checkpoint"}):
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("role") == "deployment_main" and int(saved.get("update", -1)) == 4000 and saved.get("prepared_sha256") == prepared_sha:
                paths.append(path)
    # Direct worker results are useful in unit fixtures where ``workers`` is
    # omitted from the aggregate receipt.
    if not paths:
        for path in _verified_artifacts(dep, {"belief_checkpoint"}):
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("variant") == OOF_VARIANT and int(saved.get("seed", -1)) == OOF_SEED and saved.get("role") == "deployment_main" and int(saved.get("update", -1)) == 4000 and saved.get("prepared_sha256") == prepared_sha:
                paths.append(path)
    paths = list(dict.fromkeys(paths))
    if len(paths) != 1:
        raise RuntimeError(f"preregistered CARE deployment checkpoint is not unique: {paths}")
    return paths[0].resolve()


def _shadow_checkpoints(dep: Mapping[str, Any], prepared_sha: str) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in _verified_artifacts(dep, {"oof_shadow_checkpoint"}):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("role") != "oof_shadow" or saved.get("oof_shadow") is not True or saved.get("variant") != OOF_VARIANT or int(saved.get("seed", -1)) != OOF_SEED or saved.get("prepared_sha256") != prepared_sha:
            continue
        fold = saved.get("oof_shadow_fold")
        if fold is None or int(fold) not in OOF_FOLDS:
            raise ValueError(f"invalid OOF shadow fold in {path}")
        fold = int(fold)
        if int(saved.get("training_seed", -1)) != OOF_SEED + 1000 + fold:
            raise ValueError(f"OOF shadow fold {fold} training seed is not preregistered")
        if fold in paths:
            raise RuntimeError(f"duplicate OOF shadow fold {fold}")
        paths[fold] = path.resolve()
    if set(paths) != set(OOF_FOLDS):
        raise RuntimeError(f"complete three-fold OOF shadows are required, found {sorted(paths)}")
    return paths


@torch.no_grad()
def oof_family_scores(prepared: Mapping[str, Any], shadow_paths: Mapping[int, Path], *, device: torch.device, fold_receipt: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return one simultaneous score per held-out family and its index."""
    n = int(torch.as_tensor(prepared["memory"]).shape[0])
    assigned = np.asarray(fold_receipt["fold_ids"], dtype=np.int64)
    if assigned.shape != (n,):
        raise ValueError("OOF fold assignment length differs from prepared families")
    all_indices: list[int] = []
    all_scores: list[float] = []
    horizon = 1  # CARE_HORIZONS[1] == registered primary horizon 16
    for fold in OOF_FOLDS:
        model, saved = _load_model(shadow_paths[fold], device)
        if int(saved.get("oof_shadow_fold", -1)) != fold or saved.get("fold_assignment", {}).get("sha256") != fold_receipt.get("sha256"):
            raise ValueError(f"OOF shadow fold {fold} provenance differs")
        train_indices = set(int(x) for x in saved.get("family_indices", []))
        held = np.flatnonzero(assigned == fold).astype(np.int64)
        expected_train = set(range(n)).difference(set(int(x) for x in held.tolist()))
        if not len(held) or train_indices != expected_train or set(held).intersection(train_indices):
            raise ValueError(f"OOF shadow fold {fold} leaks held-out families")
        for first in range(0, len(held), 32):
            idx = torch.from_numpy(held[first:first + 32]).long()
            memory = torch.as_tensor(prepared["memory"])[idx].float().to(device)
            mask = torch.as_tensor(prepared["memory_mask"])[idx].bool().to(device)
            candidates = torch.as_tensor(prepared["candidate_chunks"])[idx].float().to(device)
            h = torch.full((len(idx),), horizon, dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(memory, mask, candidates, h)
            lower = output.quantiles[:, 1:, 2, 0].float().cpu().numpy()  # [B,5]
            target = torch.as_tensor(prepared["targets"])[idx, horizon, 1:, :, 2].float().cpu().numpy()  # [B,5,2]
            # A family score is simultaneous over all five interventions and
            # both physical repeats; candidate zero is the exact reference.
            score = (lower[:, :, None] - target).max(axis=(1, 2))
            all_indices.extend(int(x) for x in held[first:first + 32].tolist())
            all_scores.extend(float(x) for x in score.tolist())
    order = np.argsort(np.asarray(all_indices, dtype=np.int64), kind="mergesort")
    indices = np.asarray(all_indices, dtype=np.int64)[order]
    scores = np.asarray(all_scores, dtype=np.float64)[order]
    if len(indices) != n or not np.array_equal(indices, np.arange(n, dtype=np.int64)):
        raise ValueError("OOF family coverage is not exactly one prediction per family")
    return indices, scores


def _reference_checkpoint(args: argparse.Namespace) -> tuple[Path, str]:
    dependency = require_stage_result(args.run, "bcore_select", config_sha256=args.config_sha256)
    candidates = _verified_artifacts(dependency, {"deployment_checkpoint", "checkpoint", "bcore_checkpoint"})
    if not candidates:
        raise RuntimeError("B-core selection receipt has no verified deployment checkpoint")
    # Selection is required to emit exactly one deployment artifact.
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise RuntimeError(f"selected B-core checkpoint is not unique: {candidates}")
    return candidates[0], sha256_file(candidates[0])


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args, need_dataset=True)
    dependency = require_stage_result(args.run, "belief_train", config_sha256=args.config_sha256)
    if getattr(args, "offline_only", True) is not True:
        raise ValueError("CARE selection must be offline-only")
    if str(getattr(args, "closed_loop_selection", "false")).strip().lower() != "false":
        raise ValueError("closed-loop CARE model selection is forbidden")
    prepared_path = args.run / "artifacts" / "prepared_branches.pt"
    prepared = _prepared(prepared_path)
    prepared_sha = sha256_file(prepared_path)
    fold_receipt = fold_assignment_receipt(prepared)
    main_path = _designated_main(dependency, prepared_sha)
    shadow_paths = _shadow_checkpoints(dependency, prepared_sha)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main_model, main_saved = _load_model(main_path, device)
    if main_saved.get("role") != "deployment_main" or main_saved.get("all_families_for_training") is not True:
        raise ValueError("designated CARE deployment checkpoint is not an all-family main fit")
    if main_saved.get("fold_assignment", {}).get("sha256") != fold_receipt["sha256"]:
        raise ValueError("main CARE checkpoint fold provenance differs")
    configured_std = np.asarray(main_model.config.action_std, dtype=np.float32)
    prepared_std = np.asarray(torch.as_tensor(prepared["action_std"]), dtype=np.float32)
    if configured_std.shape != (ACTION_DIM,) or not np.array_equal(configured_std, prepared_std):
        raise ValueError("CARE action encoder scale differs from full-corpus normalization")
    norm_path = Path(str(prepared.get("normalization_receipt", "")))
    norm = load_normalization_receipt(norm_path, require_formal=True)
    if not np.array_equal(np.asarray(norm["action_std"], np.float32), prepared_std):
        raise ValueError("CARE normalization receipt differs from prepared data")
    indices, scores = oof_family_scores(prepared, shadow_paths, device=device, fold_receipt=fold_receipt)
    correction = max(0.0, finite_sample_quantile(scores, REGISTERED_CALIBRATION["nominal_simultaneous_coverage"]))
    calibration = {"lower_correction": correction, **REGISTERED_CALIBRATION}
    reference_path, reference_sha = _reference_checkpoint(args)
    if prepared.get("reference_checkpoint_sha256") != reference_sha or main_saved.get("reference_checkpoint_sha256") != reference_sha:
        raise ValueError("CARE branch/training provenance is not bound to selected B-core")
    for fold, path in shadow_paths.items():
        shadow = torch.load(path, map_location="cpu", weights_only=False)
        if shadow.get("reference_checkpoint_sha256") != reference_sha:
            raise ValueError(f"OOF shadow fold {fold} is not bound to selected B-core")
    out_root = args.run / "artifacts" / "offline_selection_calibration"; out_root.mkdir(parents=True, exist_ok=True)
    score_path = out_root / "oof_family_scores.npz"
    temporary = score_path.with_name(f".{score_path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, family_index=indices, score=scores, fold_id=np.asarray(fold_receipt["fold_ids"], np.int64)); os.replace(temporary, score_path)
    deployment = out_root / "care_deployment.pt"
    model_contract = __import__("deployment.bicoord_care.supervisor", fromlist=["MODEL_CONTRACT"]).MODEL_CONTRACT
    payload = {"format_version": DEPLOYMENT_FORMAT, "benchmark_adapter": "BiCoord", "method_family": "CARE", "policy_family": "CAREBeliefHead", "reference_policy": "B-core/TUNE", "model_contract": model_contract, "config": main_saved["config"], "model": main_saved["model"], "calibration": calibration, "variant": OOF_VARIANT, "seed": OOF_SEED, "selected_checkpoint": str(main_path), "selected_checkpoint_sha256": sha256_file(main_path), "reference_checkpoint": str(reference_path), "reference_checkpoint_sha256": reference_sha, "prepared_data": str(prepared_path.resolve()), "prepared_data_sha256": prepared_sha, "normalization_receipt": str(norm_path.resolve()), "normalization_receipt_sha256": sha256_file(norm_path), "oof_calibration_complete": True, "oof_fold_count": 3, "oof_family_count": int(len(indices)), "oof_fold_assignment": fold_receipt, "oof_shadow_checkpoints": {str(fold): str(path) for fold, path in shadow_paths.items()}, "oof_shadow_checkpoint_sha256": {str(fold): sha256_file(path) for fold, path in shadow_paths.items()}, "oof_scores_artifact": str(score_path.resolve()), "oof_scores_artifact_sha256": sha256_file(score_path), "closed_loop_results_used_for_selection": False, "strictly_decentralized": True, "teacher_present": False}
    temporary = deployment.with_name(f".{deployment.name}.{os.getpid()}.tmp"); torch.save(payload, temporary); os.replace(temporary, deployment)
    report = out_root / "offline_report.json"
    atomic_json(report, {"schema": "before-we-act.bicoord.offline-selection/2", "status": "PASSED", "selection_protocol": "fixed_preregistered_care_seed_20260904_update_4000", "closed_loop_results_used_for_selection": False, "selected_checkpoint": str(main_path), "selected_checkpoint_sha256": sha256_file(main_path), "reference_checkpoint_sha256": reference_sha, "oof_calibration_complete": True, "oof_fold_count": 3, "oof_family_count": int(len(indices)), "oof_fold_assignment_sha256": fold_receipt["sha256"], "oof_shadow_checkpoints": {str(k): str(v) for k, v in shadow_paths.items()}, "oof_scores_artifact": str(score_path.resolve()), "oof_scores_artifact_sha256": sha256_file(score_path), "calibration": calibration, "simultaneous_score_definition": "max over candidates 1..5 and repeats 0..1 at horizon 16", "families": int(len(indices)), "held_out_families": 0})
    return publish_result(args, stage="offline_selection_calibration", include_model_contract=True, artifacts=[artifact(deployment, kind="care_deployment_checkpoint"), artifact(report, kind="offline_selection_report"), artifact(score_path, kind="oof_scores")], closed_loop_results_used_for_selection=False, selected_checkpoint=str(deployment.resolve()), selected_checkpoint_sha256=sha256_file(deployment), selected_main_checkpoint=str(main_path), selected_main_checkpoint_sha256=sha256_file(main_path), reference_checkpoint=str(reference_path), reference_checkpoint_sha256=reference_sha, calibration=calibration, oof_calibration_complete=True, oof_fold_count=3, oof_family_count=int(len(indices)), oof_fold_assignment_sha256=fold_receipt["sha256"], oof_shadow_checkpoints={str(k): str(v) for k, v in shadow_paths.items()}, oof_scores_artifact=str(score_path.resolve()), oof_scores_artifact_sha256=sha256_file(score_path), held_out_families=0)


def main(argv: list[str] | None = None) -> int:
    parser = common_parser(__doc__, ("select-calibrate",)); parser.add_argument("--offline-only", action="store_true", default=True); parser.add_argument("--closed-loop-selection", default="false"); args = parser.parse_args(argv); run(args); return 0


__all__ = ["finite_sample_quantile", "oof_family_scores", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
