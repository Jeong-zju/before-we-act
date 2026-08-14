#!/usr/bin/env python3
"""Run SSC-V7 M3-R4-B observability without opening the sealed R4-C test.

The deployable ARB predictor reads only the already frozen legal observation/history
vector.  Train predictions are episode-out-of-fold; confirmation predictions come
from a predictor whose epoch count and probability shrinkage were selected on train
episodes only.  Every action branch uses the A1 shared HC and the direct residual.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.before_we_act import run_ssc_v7_m3 as m3  # noqa: E402
from scripts.before_we_act import run_ssc_v7_m3_r4 as r4  # noqa: E402
from scripts.before_we_act import run_ssc_v7_m3_r4_successor as successor  # noqa: E402


STAGE_ID = "SSC-V7-M3-R4-B-OBSERVABILITY"
FROZEN_STATUS = "FROZEN_R4_B_BEFORE_PREDICTOR_OR_ACTION_METRICS"
TASKS = tuple(successor.TASKS)
HEAD_COUNT = successor.FEATURE_WIDTH
ACTIVE_HEADS = tuple(list(range(24)) + list(range(27, 32)) + list(range(32, 39)))
CONSTANT_HEAD_VALUES = {
    24: 0.0,
    25: 0.0,
    26: 0.0,
    39: 1.0,
    40: 1.0,
    41: 1.0,
    42: 1.0,
    43: 1.0,
    44: 1.0,
    45: 1.0,
    46: 1.0,
    47: 1.0,
}
CONDITIONS = (
    "arb_hat_direct",
    "row_shuffled_direct",
    "time_only_direct",
    "episode_shuffled_direct",
    "stale_8_direct",
    "stale_16_direct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("fit-predictors", "train-branch", "aggregate")
    )
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed-index", type=int, choices=(0, 1, 2))
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def save_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)


def load_gate(path: Path) -> dict[str, Any]:
    gate = read_json(path)
    if gate.get("stage_id") != STAGE_ID or gate.get("status") != FROZEN_STATUS:
        raise RuntimeError("R4-B gate identity/status is not frozen")
    unsigned = {key: value for key, value in gate.items() if key != "integrity"}
    if hashlib.sha256(canonical_bytes(unsigned)).hexdigest() != str(
        gate["integrity"]["payload_sha256"]
    ):
        raise RuntimeError("R4-B gate payload hash mismatch")
    gate["_runtime_gate_sha256"] = sha256_file(path)
    return gate


def preflight(gate: Mapping[str, Any]) -> None:
    implementation = gate["implementation"]
    checks = {
        "script_hash": sha256_file(REPOSITORY / str(implementation["script"]))
        == str(implementation["script_sha256"]),
        "test_hash": sha256_file(REPOSITORY / str(implementation["test"]))
        == str(implementation["test_sha256"]),
        "implementation_is_ancestor": subprocess.call(
            (
                "git",
                "-C",
                str(REPOSITORY),
                "merge-base",
                "--is-ancestor",
                str(implementation["implementation_commit"]),
                "HEAD",
            )
        )
        == 0,
        "repository_clean": subprocess.check_output(
            ("git", "-C", str(REPOSITORY), "status", "--porcelain"), text=True
        ).strip()
        == "",
    }
    if not all(checks.values()):
        raise RuntimeError(f"R4-B preflight failed: {checks}")
    if tuple(gate["schema"]["active_head_indices"]) != ACTIVE_HEADS:
        raise RuntimeError("active ARB predictor heads differ from frozen gate")
    frozen_constants = {
        int(key): float(value)
        for key, value in gate["schema"]["constant_head_values"].items()
    }
    if frozen_constants != CONSTANT_HEAD_VALUES:
        raise RuntimeError("constant ARB fields differ from frozen gate")


@dataclass
class StageData:
    train: successor.CachedBundle
    confirmation: successor.CachedBundle
    train_y: np.ndarray
    confirmation_y: np.ndarray
    arb_mean: np.ndarray
    arb_std: np.ndarray


def load_stage_data(gate: Mapping[str, Any]) -> StageData:
    source = gate["source"]
    cache_receipt_path = Path(str(source["cache_receipt"]))
    if sha256_file(cache_receipt_path) != str(source["cache_receipt_sha256"]):
        raise RuntimeError("source cache receipt hash mismatch")
    cache_receipt = read_json(cache_receipt_path)
    for path_key, hash_key in (
        ("train_cache", "train_cache_sha256"),
        ("confirmation_cache", "confirmation_cache_sha256"),
        ("normalization", "normalization_sha256"),
    ):
        path = Path(str(cache_receipt[path_key]))
        if sha256_file(path) != str(cache_receipt[hash_key]):
            raise RuntimeError(f"source cache artifact hash mismatch: {path_key}")
    train = successor.load_cached_bundle(Path(str(cache_receipt["train_cache"])))
    confirmation = successor.load_cached_bundle(
        Path(str(cache_receipt["confirmation_cache"]))
    )
    normalization = read_json(Path(str(cache_receipt["normalization"])))
    arb_mean = np.asarray(normalization["arb"]["mean"], dtype=np.float32)
    arb_std = np.asarray(normalization["arb"]["std"], dtype=np.float32)
    train_y = np.clip(train.arb * arb_std + arb_mean, 0.0, 1.0).reshape(
        len(train), -1
    )
    confirmation_y = np.clip(
        confirmation.arb * arb_std + arb_mean, 0.0, 1.0
    ).reshape(len(confirmation), -1)
    for index, value in CONSTANT_HEAD_VALUES.items():
        if not np.allclose(train_y[:, index], value, atol=1e-5):
            raise RuntimeError(f"frozen constant ARB field changed: {index}")
    train_ids = set(train.base.episode_ids.astype(str).tolist())
    confirmation_ids = set(confirmation.base.episode_ids.astype(str).tolist())
    if train_ids & confirmation_ids:
        raise RuntimeError("train/confirmation episode overlap")
    return StageData(
        train=train,
        confirmation=confirmation,
        train_y=train_y.astype(np.float32),
        confirmation_y=confirmation_y.astype(np.float32),
        arb_mean=arb_mean,
        arb_std=arb_std,
    )


def torch_setup(seed: int) -> Any:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def build_predictor(input_width: int, seed: int, hidden_width: int = 384) -> Any:
    torch = torch_setup(seed)
    return torch.nn.Sequential(
        torch.nn.LayerNorm(input_width),
        torch.nn.Linear(input_width, hidden_width),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_width, 192),
        torch.nn.SiLU(),
        torch.nn.Linear(192, HEAD_COUNT),
    )


def predictor_loss(logits: Any, targets: Any) -> Any:
    import torch

    indices = torch.as_tensor(ACTIVE_HEADS, device=logits.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits.index_select(1, indices), targets.index_select(1, indices)
    )


def probability_brier(logits: Any, targets: Any) -> Any:
    import torch

    indices = torch.as_tensor(ACTIVE_HEADS, device=logits.device)
    probabilities = torch.sigmoid(logits.index_select(1, indices))
    selected = targets.index_select(1, indices)
    return (probabilities - selected).square().mean()


def train_predictor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray | None,
    validation_y: np.ndarray | None,
    device: str,
    initialization_seed: int,
    sampler_seed: int,
    config: Mapping[str, Any],
    fixed_epochs: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    torch = torch_setup(initialization_seed)
    model = build_predictor(
        train_x.shape[1], initialization_seed, int(config["hidden_width"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    x_train = torch.from_numpy(train_x)
    y_train = torch.from_numpy(train_y)
    batch_size = int(config["batch_size"])
    maximum = int(config["max_epochs"]) if fixed_epochs is None else int(fixed_epochs)
    patience = int(config["patience"])
    best_brier = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(maximum):
        model.train()
        losses: list[float] = []
        for indices in m3.batches(len(train_x), batch_size, sampler_seed, epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train[indices].to(device))
            loss = predictor_loss(logits, y_train[indices].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if validation_x is None:
            score = float(np.mean(losses))
        else:
            model.eval()
            scores: list[float] = []
            with torch.no_grad():
                for first in range(0, len(validation_x), 2048):
                    logits = model(
                        torch.from_numpy(validation_x[first : first + 2048]).to(device)
                    )
                    targets = torch.from_numpy(
                        validation_y[first : first + 2048]
                    ).to(device)
                    scores.append(float(probability_brier(logits, targets).cpu()))
            score = float(np.mean(scores))
        history.append(
            {
                "epoch": float(epoch),
                "train_bce": float(np.mean(losses)),
                "validation_active_head_brier": score,
            }
        )
        if fixed_epochs is not None:
            continue
        if score < best_brier - 1e-8:
            best_brier = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("ARB predictor failed to produce a checkpoint")
        model.load_state_dict(best_state)
    else:
        best_epoch = maximum - 1
        best_brier = float(history[-1]["validation_active_head_brier"])
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "selected_epochs": best_epoch + 1,
        "best_validation_active_head_brier": best_brier,
        "epochs_run": len(history),
        "fixed_epochs": fixed_epochs,
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def predict_logits(model: Any, values: np.ndarray, device: str) -> np.ndarray:
    import torch

    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for first in range(0, len(values), 2048):
            logits = model(torch.from_numpy(values[first : first + 2048]).to(device))
            output.append(logits.detach().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def fit_shrinkage(
    logits: np.ndarray, targets: np.ndarray, prior: np.ndarray
) -> np.ndarray:
    probabilities = sigmoid(logits)
    alphas = np.zeros(HEAD_COUNT, dtype=np.float32)
    grid = np.linspace(0.0, 1.0, 41, dtype=np.float32)
    for head in ACTIVE_HEADS:
        candidates = prior[head] + grid[:, None] * (
            probabilities[:, head][None] - prior[head]
        )
        scores = np.mean((candidates - targets[:, head][None]) ** 2, axis=1)
        alphas[head] = grid[int(np.argmin(scores))]
    return alphas


def calibrated_probabilities(
    logits: np.ndarray, prior: np.ndarray, alphas: np.ndarray
) -> np.ndarray:
    raw = sigmoid(logits)
    output = prior[None] + alphas[None] * (raw - prior[None])
    output = np.clip(output, 0.0, 1.0).astype(np.float32)
    for index, value in CONSTANT_HEAD_VALUES.items():
        output[:, index] = value
    return output


def predictor_reliability(probabilities: np.ndarray, available: bool = True) -> np.ndarray:
    if not available:
        return np.zeros((len(probabilities), 1), dtype=np.float32)
    selected = probabilities[:, ACTIVE_HEADS]
    uncertainty = 2.0 * np.minimum(selected, 1.0 - selected)
    reliability = 1.0 - np.mean(uncertainty, axis=1, keepdims=True)
    return np.clip(reliability, 0.0, 1.0).astype(np.float32)


def time_only_features(bundle: successor.CachedBundle) -> np.ndarray:
    task_one_hot = np.zeros((len(bundle), len(TASKS)), dtype=np.float32)
    for index, task in enumerate(TASKS):
        task_one_hot[:, index] = bundle.base.tasks == task
    return np.concatenate((bundle.base.time, task_one_hot), axis=1).astype(np.float32)


def predictor_inputs(kind: str, bundle: successor.CachedBundle) -> np.ndarray:
    if kind == "legal":
        return bundle.base.legal.astype(np.float32)
    if kind == "time_only":
        return time_only_features(bundle)
    raise ValueError(kind)


def final_train_masks(data: m3.ProbeData) -> tuple[np.ndarray, np.ndarray]:
    validation_ids: set[str] = set()
    for task in TASKS:
        episode_ids = sorted(set(data.episode_ids[data.tasks == task].astype(str)))
        validation_ids.update(episode_ids[::6])
    validation = np.asarray(
        [str(value) in validation_ids for value in data.episode_ids], dtype=bool
    )
    return ~validation, validation


def brier_summary(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    per_head = np.mean((probabilities - targets) ** 2, axis=0)
    return {
        "active_head_mean": float(np.mean(per_head[list(ACTIVE_HEADS)])),
        "per_head": per_head.astype(float).tolist(),
    }


def fit_crossfitted_predictor(
    kind: str,
    data: StageData,
    gate: Mapping[str, Any],
    device: str,
    output_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    config = gate["predictor_training"]
    seeds = gate["seeds"][f"{kind}_predictor"]
    train_x = predictor_inputs(kind, data.train)
    confirmation_x = predictor_inputs(kind, data.confirmation)
    train_probabilities = np.zeros_like(data.train_y)
    train_oof_logits = np.zeros_like(data.train_y)
    folds: list[dict[str, Any]] = []
    selected_epochs: list[int] = []
    for fold in range(3):
        held, inner_fit, inner_validation = m3.nested_social_masks(data.train.base, fold)
        selector, selector_receipt = train_predictor(
            train_x[inner_fit],
            data.train_y[inner_fit],
            train_x[inner_validation],
            data.train_y[inner_validation],
            device,
            int(seeds["initialization"]) + fold * 1009,
            int(seeds["sampler"]) + fold * 1009,
            config,
        )
        epochs = int(selector_receipt["selected_epochs"])
        selected_epochs.append(epochs)
        validation_logits = predict_logits(selector, train_x[inner_validation], device)
        prior = data.train_y[inner_fit].mean(axis=0)
        alphas = fit_shrinkage(
            validation_logits, data.train_y[inner_validation], prior
        )
        refit, refit_receipt = train_predictor(
            train_x[~held],
            data.train_y[~held],
            None,
            None,
            device,
            int(seeds["initialization"]) + fold * 1009,
            int(seeds["sampler"]) + fold * 1009,
            config,
            fixed_epochs=epochs,
        )
        held_logits = predict_logits(refit, train_x[held], device)
        train_oof_logits[held] = held_logits
        train_probabilities[held] = calibrated_probabilities(
            held_logits, prior, alphas
        )
        folds.append(
            {
                "fold": fold,
                "heldout_episode_count": len(
                    set(data.train.base.episode_ids[held].astype(str))
                ),
                "heldout_row_count": int(held.sum()),
                "episode_overlap_count": 0,
                "selector": selector_receipt,
                "refit": refit_receipt,
                "shrinkage_alpha": alphas.astype(float).tolist(),
            }
        )

    final_fit, final_validation = final_train_masks(data.train.base)
    final_selector, final_selector_receipt = train_predictor(
        train_x[final_fit],
        data.train_y[final_fit],
        train_x[final_validation],
        data.train_y[final_validation],
        device,
        int(seeds["initialization"]) + 4001,
        int(seeds["sampler"]) + 4001,
        config,
    )
    final_epochs = int(final_selector_receipt["selected_epochs"])
    final_refit, final_refit_receipt = train_predictor(
        train_x,
        data.train_y,
        None,
        None,
        device,
        int(seeds["initialization"]) + 5003,
        int(seeds["sampler"]) + 5003,
        config,
        fixed_epochs=final_epochs,
    )
    # Train-only, out-of-fold logits provide the post-hoc probability shrinkage.
    final_prior = data.train_y.mean(axis=0)
    final_alphas = fit_shrinkage(train_oof_logits, data.train_y, final_prior)
    confirmation_logits = predict_logits(final_refit, confirmation_x, device)
    confirmation_probabilities = calibrated_probabilities(
        confirmation_logits, final_prior, final_alphas
    )
    checkpoint = output_root / "predictors" / kind / "final_predictor.pt"
    m3.save_torch_checkpoint(
        checkpoint,
        {
            "stage_id": STAGE_ID,
            "kind": kind,
            "input_width": int(train_x.shape[1]),
            "hidden_width": int(config["hidden_width"]),
            "state_dict": final_refit.state_dict(),
            "constant_head_values": CONSTANT_HEAD_VALUES,
            "active_heads": ACTIVE_HEADS,
        },
    )
    receipt = {
        "kind": kind,
        "input_width": int(train_x.shape[1]),
        "legal_inputs_only": kind == "legal",
        "uses_agent_slot_metadata": False,
        "uses_frame_or_episode_identity": False,
        "folds": folds,
        "selected_epochs_across_outer_folds": selected_epochs,
        "final_selector": final_selector_receipt,
        "final_refit": final_refit_receipt,
        "final_shrinkage_alpha": final_alphas.astype(float).tolist(),
        "train_oof_brier": brier_summary(data.train_y, train_probabilities),
        "confirmation_brier": brier_summary(
            data.confirmation_y, confirmation_probabilities
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    return train_probabilities, confirmation_probabilities, receipt


def calibration_summary(
    train_targets: np.ndarray,
    confirmation_targets: np.ndarray,
    confirmation_probabilities: np.ndarray,
    reliability: np.ndarray,
) -> dict[str, Any]:
    train_prior = train_targets.mean(axis=0)
    predictor_brier = np.mean(
        (confirmation_probabilities - confirmation_targets) ** 2, axis=0
    )
    constant_brier = np.mean(
        (train_prior[None] - confirmation_targets) ** 2, axis=0
    )
    head_rows: list[dict[str, Any]] = []
    for index in ACTIVE_HEADS:
        token = index // successor.TOKEN_WIDTH
        field = index % successor.TOKEN_WIDTH
        head_rows.append(
            {
                "index": index,
                "token": r4.TOKEN_NAMES[token],
                "field": r4.FIELD_NAMES[token][field],
                "train_incidence": float(train_prior[index]),
                "confirmation_incidence": float(confirmation_targets[:, index].mean()),
                "predictor_brier": float(predictor_brier[index]),
                "constant_rate_brier": float(constant_brier[index]),
                "improvement": float(constant_brier[index] - predictor_brier[index]),
                "beats_constant": bool(predictor_brier[index] < constant_brier[index]),
            }
        )
    order = np.argsort(reliability[:, 0])
    bins: list[dict[str, Any]] = []
    selected_targets = confirmation_targets[:, ACTIVE_HEADS]
    selected_probabilities = confirmation_probabilities[:, ACTIVE_HEADS]
    for bin_index, indices in enumerate(np.array_split(order, 4)):
        hard_error = np.mean(
            (selected_probabilities[indices] >= 0.5)
            != (selected_targets[indices] >= 0.5)
        )
        bins.append(
            {
                "bin": bin_index,
                "meaning": "least reliable" if bin_index == 0 else (
                    "most reliable" if bin_index == 3 else "intermediate"
                ),
                "rows": len(indices),
                "mean_reliability": float(reliability[indices].mean()),
                "hard_error_rate": float(hard_error),
                "brier": float(
                    np.mean(
                        (selected_probabilities[indices] - selected_targets[indices])
                        ** 2
                    )
                ),
            }
        )
    reliability_directional = bool(
        bins[0]["hard_error_rate"] >= bins[-1]["hard_error_rate"]
        and bins[0]["brier"] >= bins[-1]["brier"]
    )
    improved = sum(row["beats_constant"] for row in head_rows)
    return {
        "active_head_count": len(ACTIVE_HEADS),
        "heads_beating_constant": improved,
        "all_active_heads_beat_constant": improved == len(ACTIVE_HEADS),
        "mean_predictor_brier": float(np.mean(predictor_brier[list(ACTIVE_HEADS)])),
        "mean_constant_rate_brier": float(
            np.mean(constant_brier[list(ACTIVE_HEADS)])
        ),
        "per_head": head_rows,
        "equal_count_reliability_bins_low_to_high": bins,
        "error_rises_as_reliability_falls": reliability_directional,
    }


def fit_predictors(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    output = args.output_root / "predictors" / "predictor_receipt.json"
    if output.exists():
        raise FileExistsError(f"fresh predictor output required: {output}")
    data = load_stage_data(gate)
    train_candidate, confirmation_candidate, candidate_receipt = (
        fit_crossfitted_predictor(
            "legal", data, gate, args.device, args.output_root
        )
    )
    train_time, confirmation_time, time_receipt = fit_crossfitted_predictor(
        "time_only", data, gate, args.device, args.output_root
    )
    train_reliability = predictor_reliability(train_candidate)
    confirmation_reliability = predictor_reliability(confirmation_candidate)
    time_train_reliability = predictor_reliability(train_time)
    time_confirmation_reliability = predictor_reliability(confirmation_time)
    prediction_root = args.output_root / "prediction_cache"
    train_path = prediction_root / "train.npz"
    confirmation_path = prediction_root / "confirmation.npz"
    save_npz(
        train_path,
        {
            "candidate": train_candidate,
            "candidate_reliability": train_reliability,
            "time_only": train_time,
            "time_only_reliability": time_train_reliability,
        },
    )
    save_npz(
        confirmation_path,
        {
            "candidate": confirmation_candidate,
            "candidate_reliability": confirmation_reliability,
            "time_only": confirmation_time,
            "time_only_reliability": time_confirmation_reliability,
        },
    )
    calibration = calibration_summary(
        data.train_y,
        data.confirmation_y,
        confirmation_candidate,
        confirmation_reliability,
    )
    receipt = {
        "format_version": "ssc-v7.m3_r4_b.predictor_receipt/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "gate_sha256": gate["_runtime_gate_sha256"],
        "candidate": candidate_receipt,
        "time_only_control": time_receipt,
        "calibration": calibration,
        "train_prediction_cache": str(train_path),
        "train_prediction_cache_sha256": sha256_file(train_path),
        "confirmation_prediction_cache": str(confirmation_path),
        "confirmation_prediction_cache_sha256": sha256_file(confirmation_path),
        "train_predictions_are_episode_out_of_fold": True,
        "confirmation_not_used_for_predictor_selection": True,
        "read_only_test_used": False,
        "test_paths_opened": 0,
        "r4_b_started": True,
        "r4_c_started": False,
    }
    write_json(output, receipt)
    print("SSC_V7_M3_R4_B_PREDICTORS_AND_OOF_CACHE_COMPLETE")


def load_predictions(
    root: Path, gate: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    receipt_path = root / "predictors" / "predictor_receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("gate_sha256") != gate["_runtime_gate_sha256"]:
        raise RuntimeError("predictor cache belongs to another R4-B gate")
    result: list[dict[str, np.ndarray]] = []
    for path_key, hash_key in (
        ("train_prediction_cache", "train_prediction_cache_sha256"),
        ("confirmation_prediction_cache", "confirmation_prediction_cache_sha256"),
    ):
        path = Path(str(receipt[path_key]))
        if sha256_file(path) != str(receipt[hash_key]):
            raise RuntimeError("prediction cache hash mismatch")
        with np.load(path, allow_pickle=False) as values:
            result.append({key: values[key].copy() for key in values.files})
    return result[0], result[1], receipt


def stale_values(
    probabilities: np.ndarray,
    reliability: np.ndarray,
    data: m3.ProbeData,
    delay: int,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros_like(probabilities)
    output_reliability = np.zeros_like(reliability)
    prior = probabilities.mean(axis=0)
    output[:] = prior
    for episode_id in sorted(set(data.episode_ids.astype(str))):
        for agent_slot in sorted(
            set(data.agent_slots[data.episode_ids == episode_id].astype(int))
        ):
            indices = np.flatnonzero(
                (data.episode_ids == episode_id) & (data.agent_slots == agent_slot)
            )
            indices = indices[np.argsort(data.frame_indices[indices])]
            frames = data.frame_indices[indices]
            for target_position, target_index in enumerate(indices):
                eligible = np.flatnonzero(frames <= frames[target_position] - delay)
                if len(eligible) == 0:
                    continue
                source_index = indices[int(eligible[-1])]
                output[target_index] = probabilities[source_index]
                output_reliability[target_index] = reliability[source_index]
    for index, value in CONSTANT_HEAD_VALUES.items():
        output[:, index] = value
    return output, output_reliability


def condition_prediction(
    condition: str,
    cached: Mapping[str, np.ndarray],
    bundle: successor.CachedBundle,
    data: StageData,
    gate: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    candidate = cached["candidate"]
    reliability = cached["candidate_reliability"]
    if condition == "arb_hat_direct":
        raw, rel = candidate, reliability
    elif condition == "time_only_direct":
        raw, rel = cached["time_only"], cached["time_only_reliability"]
    elif condition == "row_shuffled_direct":
        seed = int(gate["seeds"]["row_shuffle"])
        raw = successor.label_shuffle(candidate, bundle.base, seed)
        rel = successor.label_shuffle(reliability, bundle.base, seed)
    elif condition == "episode_shuffled_direct":
        seed = int(gate["seeds"]["episode_shuffle"])
        raw = successor.episode_shuffle(candidate, bundle.base, seed)
        rel = successor.episode_shuffle(reliability, bundle.base, seed)
    elif condition in {"stale_8_direct", "stale_16_direct"}:
        delay = 8 if condition == "stale_8_direct" else 16
        raw, rel = stale_values(candidate, reliability, bundle.base, delay)
    else:
        raise ValueError(condition)
    normalized = (raw.reshape(-1, successor.TOKEN_COUNT, successor.TOKEN_WIDTH) - data.arb_mean) / data.arb_std
    return normalized.astype(np.float32), rel.astype(np.float32)


def load_source_hc(gate: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = gate["source"]
    receipt_path = Path(str(source["hc_receipt"]))
    if sha256_file(receipt_path) != str(source["hc_receipt_sha256"]):
        raise RuntimeError("source HC receipt hash mismatch")
    receipt = read_json(receipt_path)
    checkpoint = Path(str(receipt["checkpoint"]))
    if sha256_file(checkpoint) != str(receipt["checkpoint_sha256"]):
        raise RuntimeError("source HC checkpoint hash mismatch")
    metrics = Path(str(receipt["metrics"]))
    if sha256_file(metrics) != str(receipt["metrics_sha256"]):
        raise RuntimeError("source HC metrics hash mismatch")
    return receipt, m3.load_torch_checkpoint(checkpoint, "cpu"), read_json(metrics)


def train_branch(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    if args.condition is None or args.seed_index is None:
        raise ValueError("train-branch requires --condition and --seed-index")
    output = (
        args.output_root
        / "formal"
        / "branches"
        / args.condition
        / f"seed_{args.seed_index}"
    )
    if output.exists():
        raise FileExistsError(f"fresh R4-B branch output required: {output}")
    output.mkdir(parents=True)
    data = load_stage_data(gate)
    train_cache, confirmation_cache, predictor_receipt = load_predictions(
        args.output_root, gate
    )
    hc_receipt, hc_payload, _ = load_source_hc(gate)
    train_features, train_reliability = condition_prediction(
        args.condition, train_cache, data.train, data, gate
    )
    confirmation_features, confirmation_reliability = condition_prediction(
        args.condition, confirmation_cache, data.confirmation, data, gate
    )
    hc_seed = int(gate["source"]["hc_noise_seed"])
    train_x = np.concatenate(
        (
            r4.hc_input(data.train, hc_seed),
            train_features.reshape(len(data.train), -1),
            train_reliability,
        ),
        axis=1,
    ).astype(np.float32)
    confirmation_x = np.concatenate(
        (
            r4.hc_input(data.confirmation, hc_seed),
            confirmation_features.reshape(len(data.confirmation), -1),
            confirmation_reliability,
        ),
        axis=1,
    ).astype(np.float32)
    init_seed = int(gate["seeds"]["residual_initialization"][args.seed_index])
    sampler_seed = int(gate["seeds"]["residual_sampler"][args.seed_index])
    model = successor.DirectResidualFactory.create(hc_payload, init_seed)
    config = gate["action_training"]
    model, training = r4.train_residual(
        model,
        train_x,
        data.train.normalized_target,
        data.train.base.target_mask,
        confirmation_x,
        data.confirmation.base,
        data.confirmation.normalized_target,
        args.device,
        float(config["learning_rate"]),
        sampler_seed,
        int(config["max_epochs"]),
        int(config["patience"]),
    )
    checkpoint = output / "action_residual.pt"
    m3.save_torch_checkpoint(
        checkpoint,
        {
            "stage_id": STAGE_ID,
            "condition": args.condition,
            "seed_index": args.seed_index,
            "state_dict": model.state_dict(),
            "input_width": int(train_x.shape[1]),
        },
    )
    metrics = m3.evaluate_model(
        model,
        confirmation_x,
        data.confirmation.base,
        data.confirmation.normalized_target,
        args.device,
    )
    metrics_path = output / "confirmation_metrics.json"
    write_json(metrics_path, metrics)
    receipt = {
        "format_version": "ssc-v7.m3_r4_b.branch_receipt/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "gate_sha256": gate["_runtime_gate_sha256"],
        "condition": args.condition,
        "seed_index": args.seed_index,
        "initialization_seed": init_seed,
        "sampler_seed": sampler_seed,
        "hc_checkpoint_sha256": hc_receipt["checkpoint_sha256"],
        "predictor_receipt_sha256": sha256_file(
            args.output_root / "predictors" / "predictor_receipt.json"
        ),
        "training": training,
        "strict_gate_eligible": bool(training["converged_by_patience"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "test_paths_opened": 0,
        "r4_b_started": True,
        "r4_c_started": False,
    }
    write_json(output / "branch_receipt.json", receipt)
    print(
        f"SSC_V7_M3_R4_B_BRANCH_COMPLETE {args.condition} seed={args.seed_index}"
    )


def load_branch_results(
    root: Path, gate: Mapping[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    metrics: dict[str, list[dict[str, Any]]] = {}
    receipts: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        metrics[condition] = []
        receipts[condition] = []
        for seed_index in range(3):
            branch = root / "formal" / "branches" / condition / f"seed_{seed_index}"
            receipt = read_json(branch / "branch_receipt.json")
            if receipt.get("gate_sha256") != gate["_runtime_gate_sha256"]:
                raise RuntimeError("R4-B branch gate hash mismatch")
            path = Path(str(receipt["metrics"]))
            if sha256_file(path) != str(receipt["metrics_sha256"]):
                raise RuntimeError("R4-B branch metrics hash mismatch")
            receipts[condition].append(receipt)
            metrics[condition].append(read_json(path))
    return metrics, receipts


def load_oracle_direct_metrics(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(str(gate["source"]["successor_root"]))
    values: list[dict[str, Any]] = []
    for seed_index in range(3):
        receipt_path = (
            root
            / "formal"
            / "branches"
            / "arb_direct"
            / f"seed_{seed_index}"
            / "branch_receipt.json"
        )
        receipt = read_json(receipt_path)
        metrics = Path(str(receipt["metrics"]))
        if sha256_file(metrics) != str(receipt["metrics_sha256"]):
            raise RuntimeError("source oracle-direct metric hash mismatch")
        values.append(read_json(metrics))
    return values


def gate_off_audit(
    root: Path,
    gate: Mapping[str, Any],
    data: StageData,
    confirmation_cache: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    import torch

    _, hc_payload, _ = load_source_hc(gate)
    candidate = confirmation_cache["candidate"]
    normalized = (
        candidate.reshape(-1, successor.TOKEN_COUNT, successor.TOKEN_WIDTH)
        - data.arb_mean
    ) / data.arb_std
    hc_seed = int(gate["source"]["hc_noise_seed"])
    hc_values = r4.hc_input(data.confirmation, hc_seed)
    values = np.concatenate(
        (
            hc_values,
            normalized.reshape(len(data.confirmation), -1),
            np.zeros((len(data.confirmation), 1), dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    maximum = 0.0
    sample = torch.from_numpy(values[:1024])
    baseline = r4.HCWrapper.create(hc_payload).eval()
    with torch.no_grad():
        expected = baseline(torch.from_numpy(hc_values[:1024]))
    for seed_index in range(3):
        branch = root / "formal" / "branches" / "arb_hat_direct" / f"seed_{seed_index}"
        receipt = read_json(branch / "branch_receipt.json")
        payload = m3.load_torch_checkpoint(Path(str(receipt["checkpoint"])), "cpu")
        model = successor.DirectResidualFactory.create(
            hc_payload,
            int(gate["seeds"]["residual_initialization"][seed_index]),
        ).eval()
        model.load_state_dict(payload["state_dict"])
        with torch.no_grad():
            actual = model(sample)
        maximum = max(maximum, float((actual - expected).abs().max()))
    return {
        "rows_checked": min(1024, len(data.confirmation)),
        "seed_count": 3,
        "max_abs_difference_from_frozen_hc": maximum,
        "exact_fallback": maximum == 0.0,
    }


def aggregate(args: argparse.Namespace, gate: Mapping[str, Any]) -> None:
    output = args.output_root / "formal" / "r4_b_observability_receipt.json"
    if output.exists():
        raise FileExistsError(f"fresh R4-B aggregate required: {output}")
    data = load_stage_data(gate)
    _, confirmation_cache, predictor_receipt = load_predictions(args.output_root, gate)
    _, _, hc_metrics = load_source_hc(gate)
    loaded, branch_receipts = load_branch_results(args.output_root, gate)
    medians = {
        condition: r4.median_metrics(values) for condition, values in loaded.items()
    }
    statistics_seed = int(gate["seeds"]["statistics"])
    candidate = m3.summarize_gain(
        hc_metrics, medians["arb_hat_direct"], statistics_seed
    )
    controls = {
        condition: m3.summarize_gain(
            medians[condition], medians["arb_hat_direct"], statistics_seed + offset
        )
        for offset, condition in enumerate(CONDITIONS[1:], start=1)
    }
    per_seed = [
        m3.summarize_gain(
            hc_metrics,
            loaded["arb_hat_direct"][seed_index],
            statistics_seed + 20 + seed_index,
        )
        for seed_index in range(3)
    ]
    oracle_direct_metrics = load_oracle_direct_metrics(gate)
    oracle_direct_median = r4.median_metrics(oracle_direct_metrics)
    oracle_direct = m3.summarize_gain(
        hc_metrics, oracle_direct_median, statistics_seed + 40
    )
    oracle_gain = float(oracle_direct["macro_gain"])
    retention = (
        float(candidate["macro_gain"]) / oracle_gain if oracle_gain > 0.0 else math.nan
    )
    acceptance = gate["acceptance"]
    stable_harms = r4.stable_task_harms(
        candidate, float(acceptance["stable_task_harm_threshold_abs"])
    )
    convergence = all(
        bool(receipt["strict_gate_eligible"])
        for condition in CONDITIONS
        for receipt in branch_receipts[condition]
    )
    calibration = predictor_receipt["calibration"]
    gate_off = gate_off_audit(args.output_root, gate, data, confirmation_cache)
    strict_checks = {
        "all_action_branches_converged": convergence,
        "arb_hat_ci_lower_positive": float(candidate["ci95"][0]) > 0.0,
        "at_least_two_positive_tasks": len(candidate["positive_tasks"])
        >= int(acceptance["positive_tasks_min"]),
        "no_stable_task_harm_at_3pct": not stable_harms,
        "at_least_two_of_three_seeds_positive": sum(
            float(item["macro_gain"]) > 0.0 for item in per_seed
        )
        >= 2,
        "no_seed_stably_harmed_at_3pct": all(
            float(item["ci95"][1])
            > -float(acceptance["stable_task_harm_threshold_abs"])
            for item in per_seed
        ),
        "retains_at_least_half_oracle_direct_gain": retention
        >= float(acceptance["oracle_gain_retention_min"]),
        "all_active_heads_beat_constant_brier": bool(
            calibration["all_active_heads_beat_constant"]
        ),
        "reliability_bins_directional": bool(
            calibration["error_rises_as_reliability_falls"]
        ),
        "gate_off_exactly_returns_hc": bool(gate_off["exact_fallback"]),
        **{
            f"beats_{condition}_ci": float(summary["ci95"][0]) > 0.0
            for condition, summary in controls.items()
        },
    }
    if not convergence:
        formal = "INCONCLUSIVE_M3_R4_B_TRAINING_CAP"
    elif all(strict_checks.values()):
        formal = "PASSED_M3_R4_B_OBSERVABILITY"
    else:
        formal = "FAILED_STRICT_M3_R4_B_OBSERVABILITY_GATE"
    positive_seed_count = sum(float(item["macro_gain"]) > 0.0 for item in per_seed)
    positive_control_count = sum(
        float(item["macro_gain"]) > 0.0 for item in controls.values()
    )
    if (
        float(candidate["macro_gain"]) > 0.0
        and len(candidate["positive_tasks"]) >= 2
        and positive_seed_count >= 2
        and positive_control_count >= 3
    ):
        signal = "PROMISING_DEPLOYABLE_ARB_HAT_SIGNAL"
    elif float(candidate["macro_gain"]) > 0.0:
        signal = "MIXED_BUT_POSITIVE_DEPLOYABLE_ARB_HAT_SIGNAL"
    else:
        signal = "NO_POSITIVE_DEPLOYABLE_ARB_HAT_SIGNAL_IN_THIS_RUN"
    receipt = {
        "format_version": "ssc-v7.m3_r4_b.observability_receipt/1",
        "stage_id": STAGE_ID,
        "completed_at_utc": utc_now(),
        "gate_sha256": gate["_runtime_gate_sha256"],
        "formal_decision_code": formal,
        "exploratory_signal_code": signal,
        "arb_hat_direct_vs_hc": candidate,
        "oracle_direct_vs_hc": oracle_direct,
        "oracle_gain_retention": retention,
        "arb_hat_direct_per_seed_vs_hc": per_seed,
        "arb_hat_vs_controls": controls,
        "calibration": calibration,
        "stable_task_harms": stable_harms,
        "gate_off_audit": gate_off,
        "strict_checks": strict_checks,
        "interpretation_policy": gate["interpretation_policy"],
        "r4_b_started": True,
        "r4_b_completed": True,
        "r4_c_authorized": formal == "PASSED_M3_R4_B_OBSERVABILITY",
        "r4_c_started": False,
        "sealed_test_generated": False,
        "test_paths_opened": 0,
        "m4_authorized": False,
        "b_core_authorized": False,
    }
    write_json(output, receipt)
    print(f"{signal} / {formal}")


def main() -> None:
    args = parse_args()
    gate = load_gate(args.gate)
    preflight(gate)
    if args.command == "fit-predictors":
        fit_predictors(args, gate)
    elif args.command == "train-branch":
        train_branch(args, gate)
    elif args.command == "aggregate":
        aggregate(args, gate)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
