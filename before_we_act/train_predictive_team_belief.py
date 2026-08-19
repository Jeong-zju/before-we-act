"""Train one frozen-contract predictive team-belief seed from cached contexts."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.action_grounded_belief import (
    BELIEF_DATA_SEED,
    fixed_requests,
    load_split,
    split_by_episode_key,
)
from before_we_act.predictive_team_belief_data import PredictiveTeamBeliefDataset, PairedSituationBatchSampler
from before_we_act.predictive_team_belief_policy import PredictiveTeamBeliefPolicy
from before_we_act.predictive_team_belief_training import TeamBeliefExperiment, paired_permutation
from before_we_act.temporal_history_data import SIX_TASKS, sha256_file
from before_we_act.team_belief.predictive_core import TeamBeliefConfig, FUTURE_OFFSETS_SECONDS
from before_we_act.team_belief.losses import (
    TeamBeliefLossWeights,
    compute_team_belief_losses,
)


MAX_UPDATES = 120_000
EVAL_EVERY = 5_000
LR_DROP = 80_000
CONDITIONS = ("b0h", "b_core", "b_shuffle", "direct_reactive", "belief_off")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--action-context-cache", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--evaluate-at-end",
        action="store_true",
        help="run the frozen validation diagnostic at the end of a short pilot",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def device_batch(raw: Mapping[str, object], device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()
    }


def config_from_contract(contract: Mapping) -> TeamBeliefConfig:
    architecture = contract["architecture"]
    return TeamBeliefConfig(
        n_belief_tokens=int(architecture["belief_tokens"]),
        n_evidence_queries=int(architecture["evidence_queries"]),
        event_capacity=int(architecture["event_capacity"]),
        temporal_layers=int(architecture["temporal_layers"]),
        d_model=int(architecture["d_model"]),
        heads=int(architecture["heads"]),
        dropout=float(architecture["dropout"]),
        belief_factors=int(architecture["belief_factors"]),
        belief_classes=int(architecture["belief_classes"]),
        belief_unimix=float(architecture["belief_unimix"]),
        belief_free_nats=float(architecture["belief_free_nats"]),
        belief_representation_scale=float(
            architecture["belief_representation_scale"]
        ),
    )


def loss_weights(contract: Mapping) -> TeamBeliefLossWeights:
    values = contract["objectives"]
    return TeamBeliefLossWeights(
        action=float(values["action"]),
        action_posterior_kl=float(values["action_posterior_kl"]),
        teacher_alignment=float(values["teacher_alignment"]),
        future_latent=float(values["future_latent"]),
        teacher_reconstruction=float(values["teacher_reconstruction"]),
        teammate_delta=float(values["teammate_delta"]),
        teammate_action=float(values["teammate_action"]),
        exchange_consistency=float(values["exchange_consistency"]),
        anti_collapse=float(values["anti_collapse"]),
        action_pairing=float(values.get("action_pairing", 0.0)),
        action_pairing_margin_fraction=float(
            values.get("action_pairing_margin_fraction", 0.0)
        ),
        action_pairing_margin_cap=float(
            values.get("action_pairing_margin_cap", 0.0)
        ),
    )


def masked_action_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    row = (prediction - target).float().square().mean(-1)
    return (row * mask).sum() / mask.sum().clamp_min(1)


def row_action_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    squared = (prediction - target).float().square().mean(-1)
    return (squared * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def shuffle_permutation(task: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    result = torch.arange(len(task), device=task.device)
    for task_value in torch.unique(task).tolist():
        for phase_value in torch.unique(phase[task == task_value]).tolist():
            rows = torch.nonzero(
                (task == task_value) & (phase == phase_value), as_tuple=False
            ).flatten()
            if len(rows) > 1:
                result[rows] = rows.roll(1)
    # A singleton task/phase group has no legal negative. Keep it self-paired;
    # the target-separation mask will exclude it from the ranking loss.
    return result


def fixed_loader(dataset, split, name: str) -> DataLoader:
    requests = fixed_requests(dataset.episodes, split, name)
    batches = [requests[index : index + 96] for index in range(0, len(requests), 96)]
    return DataLoader(dataset, batch_sampler=batches, num_workers=0, pin_memory=True)


@torch.no_grad()
def evaluate(
    model: TeamBeliefExperiment,
    loader: DataLoader,
    device: torch.device,
    weights: TeamBeliefLossWeights,
) -> dict:
    model.eval()
    values = {name: [] for name in CONDITIONS}
    tasks = {name: {task: [] for task in range(6)} for name in CONDITIONS}
    auxiliary: dict[str, list[float]] = {}
    future_numerator = {
        name: np.zeros(4, dtype=np.float64)
        for name in ("model", "persistence", "shuffle")
    }
    future_denominator = np.zeros(4, dtype=np.float64)
    observable_future_numerator = {
        name: np.zeros(4, dtype=np.float64)
        for name in (
            "oracle_action",
            "policy_action",
            "shuffled_action",
            "shuffled_belief",
            "persistence",
        )
    }
    observable_future_denominator = np.zeros(4, dtype=np.float64)
    pairing_rows: dict[str, list[float]] = {
        "correct_mse": [],
        "shuffled_mse": [],
        "belief_pair_mse": [],
        "residual_output_mse": [],
        "residual_target_mse": [],
        "residual_energy": [],
    }
    belief_off_max_abs = 0.0
    gate_values: list[float] = []
    reliability_values: list[float] = []
    sigma_values: list[float] = []
    safety_entropy_rows: list[float] = []
    safety_target_norm_rows: list[float] = []
    safety_residual_norm_rows: list[float] = []
    safety_temporal_delta_rows: list[float] = []
    mu_rows: list[torch.Tensor] = []
    categorical_rows: list[torch.Tensor] = []
    uncertainty_rows: list[
        tuple[float, float, float, float, float, float, float, float]
    ] = []
    seen_uncertainty = 0
    for raw in loader:
        batch = device_batch(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
            permutation = shuffle_permutation(batch["task_index"], batch["phase_bin"])
            shuffled_residual, _ = model.belief_residual(
                batch["decoded_action_hidden"],
                output.candidate.belief.mu[permutation],
                output.candidate.belief.sigma[permutation],
                output.candidate.belief.reliability[permutation],
            )
            shuffled = batch["base_action"] + shuffled_residual
            policy_future = model.belief_core.predict_future(
                output.candidate.belief.mu,
                output.candidate.belief.current_visual_reference,
                output.candidate.belief.current_visual_view_mask,
                output.candidate.prediction,
                batch["action_mask"],
            )
            shuffled_action_future = model.belief_core.predict_future(
                output.candidate.belief.mu,
                output.candidate.belief.current_visual_reference,
                output.candidate.belief.current_visual_view_mask,
                batch["action"][permutation],
                batch["action_mask"][permutation],
            )
            shuffled_belief_future = model.belief_core.predict_future(
                output.candidate.belief.mu[permutation],
                output.candidate.belief.current_visual_reference,
                output.candidate.belief.current_visual_view_mask,
                batch["action"],
                batch["action_mask"],
            )
        predictions = {
            "b0h": batch["base_action"],
            "b_core": output.candidate.prediction,
            "b_shuffle": shuffled,
            "direct_reactive": output.direct_prediction,
            "belief_off": batch["base_action"],
        }
        belief_off_max_abs = max(
            belief_off_max_abs,
            float((predictions["belief_off"] - batch["base_action"]).abs().max().cpu()),
        )
        for name, prediction in predictions.items():
            scores = row_action_mse(prediction, batch["action"], batch["action_mask"])
            values[name].extend(scores.cpu().tolist())
            for task in range(6):
                tasks[name][task].extend(scores[batch["task_index"] == task].cpu().tolist())
        correct_mse = row_action_mse(
            output.candidate.prediction, batch["action"], batch["action_mask"]
        )
        shuffled_mse = row_action_mse(
            shuffled, batch["action"], batch["action_mask"]
        )
        belief_pair_mse = (
            output.candidate.belief.mu
            - output.candidate.belief.mu[permutation]
        ).float().square().mean((1, 2))
        residual_output_mse = row_action_mse(
            output.candidate.belief_residual,
            shuffled_residual,
            batch["action_mask"],
        )
        residual_target = batch["action"] - batch["base_action"]
        active_actions = batch["action_mask"]
        safety_target_norm_rows.extend(
            residual_target.float().norm(dim=-1)[active_actions].cpu().tolist()
        )
        safety_residual_norm_rows.extend(
            output.candidate.belief_residual.float()
            .norm(dim=-1)[active_actions]
            .cpu()
            .tolist()
        )
        safety_entropy_rows.extend(
            output.candidate.belief.sigma.float()
            .mean((1, 2))
            .cpu()
            .tolist()
        )
        previous_valid = batch["history_mask"][:, -2]
        if previous_valid.any():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                current_raw_residual, _ = model.belief_residual._candidate(
                    batch["decoded_action_hidden"],
                    output.candidate.belief.mu,
                )
                previous_raw_residual, _ = model.belief_residual._candidate(
                    batch["decoded_action_hidden"],
                    output.candidate.belief.mu_sequence[:, -2],
                )
            temporal_delta = (
                current_raw_residual.float() - previous_raw_residual.float()
            ).norm(dim=-1).mean(1)
            safety_temporal_delta_rows.extend(
                temporal_delta[previous_valid].cpu().tolist()
            )
        common_action_mask = batch["action_mask"] & batch["action_mask"][permutation]
        residual_target_mse = row_action_mse(
            residual_target,
            residual_target[permutation],
            common_action_mask,
        )
        residual_energy = row_action_mse(
            output.candidate.belief_residual,
            torch.zeros_like(output.candidate.belief_residual),
            batch["action_mask"],
        )
        for name, tensor in (
            ("correct_mse", correct_mse),
            ("shuffled_mse", shuffled_mse),
            ("belief_pair_mse", belief_pair_mse),
            ("residual_output_mse", residual_output_mse),
            ("residual_target_mse", residual_target_mse),
            ("residual_energy", residual_energy),
        ):
            pairing_rows[name].extend(tensor.float().cpu().tolist())
        losses = compute_team_belief_losses(
            output.candidate,
            batch["action"],
            batch["action_mask"],
            batch["teammate_delta"],
            batch["teacher_future_anchor_mask"],
            batch["teammate_action"],
            batch["teammate_action_mask"],
            weights,
        )
        for name, value in losses.items():
            auxiliary.setdefault(name, []).append(float(value))
        target = output.candidate.teacher.future_latent_target
        prediction = output.candidate.belief.future_latent_prediction
        persistence = batch["teacher_current_visual_tokens"].squeeze(2)[:, None].expand_as(target)
        legal_persistence = output.candidate.belief.current_visual_reference[
            :, None
        ].expand_as(target)
        shuffled_target = target[permutation]
        view_mask = output.candidate.teacher.future_view_mask
        anchor_mask = output.candidate.teacher.future_anchor_mask
        for anchor in range(4):
            active = view_mask[:, anchor] & anchor_mask[:, anchor : anchor + 1]
            count = int(active.sum()) * target.shape[-1]
            if not count:
                continue
            future_denominator[anchor] += count
            for name, candidate in (
                ("model", prediction),
                ("persistence", persistence),
                ("shuffle", shuffled_target),
            ):
                error = (candidate[:, anchor][active] - target[:, anchor][active]).float()
                future_numerator[name][anchor] += float(error.square().sum().cpu())
            observable_active = (
                active & output.candidate.belief.current_visual_view_mask
            )
            observable_count = int(observable_active.sum()) * target.shape[-1]
            if observable_count:
                observable_future_denominator[anchor] += observable_count
                for name, candidate in (
                    ("oracle_action", prediction),
                    ("policy_action", policy_future),
                    ("shuffled_action", shuffled_action_future),
                    ("shuffled_belief", shuffled_belief_future),
                    ("persistence", legal_persistence),
                ):
                    error = (
                        candidate[:, anchor][observable_active]
                        - target[:, anchor][observable_active]
                    ).float()
                    observable_future_numerator[name][anchor] += float(
                        error.square().sum().cpu()
                    )
        gate_values.append(float(output.candidate.residual_gate.float().mean().cpu()))
        reliability_values.append(float(output.candidate.belief.reliability.float().mean().cpu()))
        sigma_values.append(float(output.candidate.belief.sigma.float().mean().cpu()))
        mu_rows.append(output.candidate.belief.mu.float().cpu())
        categorical_rows.append(output.candidate.belief.categorical_probs.float().cpu())
        if seen_uncertainty < 192:
            occluded = dict(batch)
            occluded_mask = batch["runtime_visual_mask"].clone()
            occluded_mask[:, :, 1] = False
            occluded["runtime_visual_mask"] = occluded_mask
            with torch.autocast("cuda", dtype=torch.bfloat16):
                perturbed = model.belief_core(
                    occluded["runtime_visual_tokens"],
                    occluded["runtime_visual_mask"],
                    occluded["history_qpos"],
                    occluded["history_action"],
                    occluded["history_mask"],
                    occluded["action_history_mask"],
                    occluded["task_token"],
                    occluded["episode_reset_mask"],
                )
            uncertainty_rows.append(
                (
                    float(output.candidate.belief.sigma.float().mean().cpu()),
                    float(perturbed.sigma.float().mean().cpu()),
                    float(
                        output.candidate.belief.epistemic_uncertainty.float()
                        .mean()
                        .cpu()
                    ),
                    float(perturbed.epistemic_uncertainty.float().mean().cpu()),
                    float(output.candidate.belief.reliability.float().mean().cpu()),
                    float(perturbed.reliability.float().mean().cpu()),
                    float(
                        output.candidate.belief.view_evidence_count.float()
                        .mean()
                        .cpu()
                    ),
                    float(perturbed.view_evidence_count.float().mean().cpu()),
                )
            )
            seen_uncertainty += len(batch["task_index"])
    belief_by_slot = torch.cat(mu_rows, dim=0)
    # A fixed slot identity is architecture, not estimated situation state.
    # Center every slot across validation situations before measuring rank so
    # static agent/free-slot offsets cannot manufacture multidimensionality.
    belief = (belief_by_slot - belief_by_slot.mean(0, keepdim=True)).flatten(0, 1)
    feature_std = belief.std(0, unbiased=False)
    covariance = torch.cov(belief.T)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    effective_rank = float(
        (eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12)).cpu()
    )
    categorical = torch.cat(categorical_rows, dim=0)
    # Estimate information independently per slot, then aggregate factors.
    # Flattening slots first would count a constant slot identity as belief.
    marginal = categorical.mean(0)
    log_classes = np.log(categorical.shape[-1])
    marginal_entropy = -(marginal * marginal.clamp_min(1e-12).log()).sum(-1) / log_classes
    conditional_entropy = -(
        categorical * categorical.clamp_min(1e-12).log()
    ).sum(-1).mean(0) / log_classes
    slot_factor_information = (marginal_entropy - conditional_entropy).clamp_min(0)
    factor_information = slot_factor_information.mean(0)
    future = {
        name: {
            f"{seconds:.1f}s": float(future_numerator[name][index] / max(future_denominator[index], 1))
            for index, seconds in enumerate(FUTURE_OFFSETS_SECONDS)
        }
        for name in future_numerator
    }
    observable_future = {
        name: {
            f"{seconds:.1f}s": float(
                observable_future_numerator[name][index]
                / max(observable_future_denominator[index], 1)
            )
            for index, seconds in enumerate(FUTURE_OFFSETS_SECONDS)
        }
        for name in observable_future_numerator
    }
    pairing_means = {
        name: float(np.mean(rows)) for name, rows in pairing_rows.items()
    }

    def quantiles(rows: list[float]) -> dict[str, float]:
        if not rows:
            return {"p95": 0.0, "p99": 0.0, "maximum": 0.0}
        values = np.asarray(rows, dtype=np.float64)
        return {
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "maximum": float(values.max()),
        }

    return {
        "macro": {name: float(np.mean(rows)) for name, rows in values.items()},
        "per_task": {
            name: {str(task): float(np.mean(rows)) for task, rows in by_task.items()}
            for name, by_task in tasks.items()
        },
        "auxiliary": {name: float(np.mean(rows)) for name, rows in auxiliary.items()},
        # Keep the old all-teacher-view diagnostic under an explicit name. The
        # runtime model never observes teammate-local current visual evidence,
        # so using that third view in its primary persistence comparison is not
        # a valid model-selection metric.
        "future_privileged_view_diagnostic_mse": future,
        "future_mse": {
            "model": observable_future["oracle_action"],
            "persistence": observable_future["persistence"],
            "shuffle": observable_future["shuffled_action"],
        },
        "future_observable_mse": observable_future,
        "future_horizon_gain": {
            f"{seconds:.1f}s": float(value)
            for seconds, value in zip(
                FUTURE_OFFSETS_SECONDS,
                output.candidate.belief.future_horizon_gain.float().cpu().tolist(),
                strict=True,
            )
        },
        "action_pairing": {
            **pairing_means,
            "shuffle_minus_correct_mse": pairing_means["shuffled_mse"]
            - pairing_means["correct_mse"],
            "output_to_target_sensitivity": pairing_means["residual_output_mse"]
            / max(pairing_means["residual_target_mse"], 1e-12),
            "output_to_residual_energy": pairing_means["residual_output_mse"]
            / max(pairing_means["residual_energy"], 1e-12),
        },
        "belief_off_max_abs": belief_off_max_abs,
        "belief": {
            "feature_std_mean": float(feature_std.mean()),
            "feature_std_min": float(feature_std.min()),
            "effective_rank": effective_rank,
            "categorical_conditional_entropy": float(conditional_entropy.mean()),
            "categorical_marginal_entropy": float(marginal_entropy.mean()),
            "categorical_mutual_information": float(factor_information.mean()),
            "active_categorical_factors_mi_gt_0_01": int(
                (factor_information > 0.01).sum()
            ),
            "mutual_information_scope": "within_slot_across_situations",
            "categorical_factors": int(categorical.shape[-2]),
            "gate_mean": float(np.mean(gate_values)),
            "reliability_mean": float(np.mean(reliability_values)),
            "sigma_mean": float(np.mean(sigma_values)),
        },
        "uncertainty_occlusion": {
            "rows": seen_uncertainty,
            # Normalized categorical entropy measures ambiguity.  It is kept
            # as a diagnostic but is not required to rise when a conflicting
            # view is removed.
            "sigma_clean": float(np.mean([row[0] for row in uncertainty_rows])),
            "sigma_occluded": float(np.mean([row[1] for row in uncertainty_rows])),
            "epistemic_uncertainty_clean": float(
                np.mean([row[2] for row in uncertainty_rows])
            ),
            "epistemic_uncertainty_occluded": float(
                np.mean([row[3] for row in uncertainty_rows])
            ),
            "reliability_clean": float(np.mean([row[4] for row in uncertainty_rows])),
            "reliability_occluded": float(
                np.mean([row[5] for row in uncertainty_rows])
            ),
            "view_evidence_count_clean": float(
                np.mean([row[6] for row in uncertainty_rows])
            ),
            "view_evidence_count_occluded": float(
                np.mean([row[7] for row in uncertainty_rows])
            ),
        },
        "deployment_safety_calibration": {
            "scope": "held-out one-step validation; no closed-loop outcomes",
            "belief_entropy": quantiles(safety_entropy_rows),
            "target_residual_l2": quantiles(safety_target_norm_rows),
            "candidate_residual_l2": quantiles(safety_residual_norm_rows),
            "temporal_residual_delta_l2": quantiles(
                safety_temporal_delta_rows
            ),
        },
        "rows": len(values["b0h"]),
    }


def trailing_smooth(values: list[float], width: int = 3) -> list[float]:
    return [float(np.mean(values[max(0, index - width + 1) : index + 1])) for index in range(len(values))]


def training_sufficiency(metrics: list[dict]) -> tuple[str, dict]:
    if not metrics or int(metrics[-1]["update"]) < MAX_UPDATES:
        return "INCONCLUSIVE_TRAINING_NOT_CONVERGED", {"reason": "maximum budget not completed"}
    series = {
        "b_core_action": [float(row["validation"]["macro"]["b_core"]) for row in metrics],
        "future_1.6s": [float(row["validation"]["future_mse"]["model"]["1.6s"]) for row in metrics],
        "teacher_alignment": [float(row["validation"]["auxiliary"]["teacher_alignment"]) for row in metrics],
        "teammate_delta": [float(row["validation"]["auxiliary"]["teammate_delta"]) for row in metrics],
        "teammate_action": [float(row["validation"]["auxiliary"]["teammate_action"]) for row in metrics],
    }
    receipt = {"points": [int(row["update"]) for row in metrics], "series": {}}
    platform = True
    for name, raw in series.items():
        smooth = trailing_smooth(raw)
        recent = smooth[-4:]
        improvements = [
            (first - second) / max(abs(first), 1e-12)
            for first, second in zip(recent, recent[1:])
        ]
        passed = len(recent) == 4 and all(value < 0.01 for value in improvements)
        receipt["series"][name] = {
            "raw": raw,
            "smoothed": smooth,
            "last_four_smoothed": recent,
            "relative_improvements": improvements,
            "all_three_below_one_percent": passed,
        }
        platform &= passed
    primary = series["b_core_action"][-4:]
    overfit = len(primary) == 4 and all(second > first for first, second in zip(primary, primary[1:]))
    receipt["learning_rate_drop_update"] = LR_DROP
    receipt["minimum_updates"] = MAX_UPDATES
    receipt["maximum_updates"] = MAX_UPDATES
    receipt["overfit_last_three_intervals"] = overfit
    if overfit:
        return "SATURATED_BY_OVERFIT", receipt
    return ("PLATFORM_REACHED" if platform else "INCONCLUSIVE_TRAINING_NOT_CONVERGED"), receipt


def export_deployment(
    selected_checkpoint: Path,
    output: Path,
    contract: Mapping,
    config: TeamBeliefConfig,
    *,
    residual_safety: Mapping[str, object] | None = None,
) -> None:
    selected = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    base_path = Path(contract["inputs"]["b0h_checkpoint"])
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    base_config = base["config"]
    policy = PredictiveTeamBeliefPolicy(
        config,
        state_dim=int(base_config.get("state_dim", 9)),
        action_dim=int(base_config.get("action_dim", 8)),
        horizon=int(base_config.get("horizon", 100)),
        d_model=int(base_config.get("d_model", 384)),
        enc_layers=int(base_config.get("enc_layers", 4)),
        dec_layers=int(base_config.get("dec_layers", 7)),
        roles=int(base_config.get("roles", 4)),
        role_rank=int(base_config.get("role_rank", 32)),
        history_layers=int(base_config.get("history_layers", 2)),
        dino_model=str(base_config["dino_model"]),
        include_teacher=False,
    )
    incompatible = policy.load_state_dict(base["model"], strict=False)
    allowed_missing = {
        key for key in policy.state_dict() if key.startswith(("belief_core.", "direct_belief_residual."))
    }
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"B0-H deployment load differs: {incompatible}")
    model_state = selected["model"]
    runtime_core = {
        key.removeprefix("belief_core."): value
        for key, value in model_state.items()
        if key.startswith("belief_core.") and not key.startswith("belief_core.teacher_branch.")
    }
    policy.belief_core.load_state_dict(runtime_core, strict=True)
    residual = {
        key.removeprefix("belief_residual."): value
        for key, value in model_state.items()
        if key.startswith("belief_residual.")
    }
    policy.direct_belief_residual.load_state_dict(residual, strict=True)
    payload = {
        "format_version": "before-we-act.b3-n2-deployment-checkpoint/1",
        "model": policy.deployment_state_dict(),
        "stats": base["stats"],
        "config": {
            **base_config,
            "policy_variant": "b3_n2_predictive_team_belief",
            "n2_config": config.__dict__,
            "teacher_present": False,
            "residual_safety": dict(residual_safety or {"enabled": False}),
            "source_b0h_checkpoint": str(base_path.resolve()),
            "source_b0h_checkpoint_sha256": sha256_file(base_path),
            "source_training_checkpoint": str(selected_checkpoint.resolve()),
            "source_training_checkpoint_sha256": sha256_file(selected_checkpoint),
        },
        "update": int(selected["update"]),
        "provenance": selected["provenance"],
    }
    atomic_save(payload, output)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("format_version") != "before-we-act.b3-n2-contract/2":
        raise RuntimeError("unsupported N2 contract")
    if contract.get("status") != "FROZEN_BEFORE_F0_F1":
        raise RuntimeError("N2 contract is not frozen")
    if args.seed not in contract["training"]["seeds"]:
        raise RuntimeError("N2 seed is outside the frozen contract")
    if not 1 <= args.updates <= MAX_UPDATES:
        raise ValueError("invalid N2 update target")
    if args.updates == MAX_UPDATES and args.save_every != EVAL_EVERY:
        raise ValueError("formal N2 checkpoint interval is frozen at 5000")
    if sha256_file(args.scenario_split) != contract["inputs"]["scenario_split_sha256"]:
        raise RuntimeError("N2 scenario split hash differs")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(min(12, os.cpu_count() or 12))
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = PredictiveTeamBeliefDataset(args.cache, args.action_context_cache)
    split = split_by_episode_key(load_split(args.scenario_split))
    config = config_from_contract(contract)
    weights = loss_weights(contract)
    model = TeamBeliefExperiment(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 if step < LR_DROP else 0.1
    )
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / "checkpoint_latest.pt"
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.is_file() else None
    start = 0
    metrics: list[dict] = []
    provenance = {
        "seed": args.seed,
        "contract_sha256": sha256_file(args.contract),
        "scenario_split_sha256": sha256_file(args.scenario_split),
        "n1_metadata_sha256": sha256_file(args.cache / "metadata.json"),
        "action_context_cache_receipt_sha256": sha256_file(args.action_context_cache / "cache_receipt.json"),
    }
    if saved:
        if saved["provenance"] != provenance:
            raise RuntimeError("N2 resume provenance differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = int(saved["update"])
        metrics = list(saved["evaluations"])
    sampler = PairedSituationBatchSampler(
        dataset.episodes,
        split,
        updates=MAX_UPDATES,
        data_seed=BELIEF_DATA_SEED,
        start_update=start,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    validation = fixed_loader(dataset, split, "validation")
    atomic_json(args.output / "status.json", {
        "status": "TRAINING",
        "seed": args.seed,
        "update": start,
        "target_updates": args.updates,
        "started_at_utc": utc_now(),
    })
    started = time.time()
    last: dict[str, float] = saved.get("last_losses", {}) if saved else {}
    for update, raw in enumerate(loader, start=start + 1):
        if update > args.updates:
            break
        step_seed = args.seed + 10_000_019 * update
        random.seed(step_seed)
        np.random.seed(step_seed % 2**32)
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        batch = device_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
            partner = paired_permutation(batch["pair_id"])
            swapped_belief = replace(
                output.candidate.belief,
                mu=output.candidate.belief.mu[partner],
            )
            swapped_output = replace(output.candidate, belief=swapped_belief)
            negative_permutation = shuffle_permutation(
                batch["task_index"], batch["phase_bin"]
            )
            counterfactual_residual, _ = model.belief_residual(
                batch["decoded_action_hidden"],
                output.candidate.belief.mu[negative_permutation],
                output.candidate.belief.sigma[negative_permutation],
                output.candidate.belief.reliability[negative_permutation],
            )
            counterfactual_prediction = (
                batch["base_action"] + counterfactual_residual
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
                weights,
                swapped_output=swapped_output,
                counterfactual_prediction=counterfactual_prediction,
                counterfactual_residual_target=residual_target[
                    negative_permutation
                ],
                counterfactual_action_mask=batch["action_mask"][
                    negative_permutation
                ],
            )
            direct = masked_action_mse(
                output.direct_prediction, batch["action"], batch["action_mask"]
            )
            loss = losses["total"] + float(contract["objectives"]["direct_reactive_action"]) * direct
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite N2 loss at {update}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite N2 gradient at {update}")
        optimizer.step()
        scheduler.step()
        last = {name: float(value.detach()) for name, value in losses.items()}
        last["direct_reactive_action"] = float(direct.detach())
        last["combined"] = float(loss.detach())
        if update == start + 1 or update % args.log_every == 0 or update == args.updates:
            elapsed = time.time() - started
            completed = update - start
            row = {
                "update": update,
                "target_updates": args.updates,
                **last,
                "grad_norm": float(grad_norm),
                "learning_rate": scheduler.get_last_lr()[0],
                "updates_per_hour": completed / max(elapsed, 1e-6) * 3600,
                "eta_hours": (args.updates - update) * elapsed / max(completed, 1) / 3600,
                "gpu_memory_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                "updated_at_epoch": time.time(),
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            atomic_json(args.output / "heartbeat.json", row)
        should_evaluate = (
            update % EVAL_EVERY == 0 and args.updates == MAX_UPDATES
        ) or (args.evaluate_at_end and update == args.updates)
        if should_evaluate:
            validation_metrics = evaluate(model, validation, device, weights)
            evaluation = {
                "update": update,
                "train": last,
                "validation": validation_metrics,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            metrics.append(evaluation)
            with (args.output / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(evaluation, sort_keys=True) + "\n")
            print(json.dumps({"evaluation": evaluation}, sort_keys=True), flush=True)
        if update == args.updates or update % args.save_every == 0:
            checkpoint = {
                "format_version": "before-we-act.b3-n2-training-checkpoint/1",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update": update,
                "evaluations": metrics,
                "last_losses": last,
                "sample_cursor": sampler.cursor_receipt(update),
                "provenance": provenance,
                "config": config.__dict__,
            }
            atomic_save(checkpoint, latest)
            atomic_save(checkpoint, args.output / f"checkpoint_{update:06d}.pt")
    if args.updates < MAX_UPDATES:
        pilot_status = {
            "status": "PASSED_SMOKE",
            "seed": args.seed,
            "update": args.updates,
            "target_updates": args.updates,
            "completed_at_utc": utc_now(),
        }
        if metrics:
            pilot_status["selected_validation"] = metrics[-1]["validation"]
        atomic_json(args.output / "status.json", pilot_status)
        return
    status, sufficiency = training_sufficiency(metrics)
    selection_updates = set(contract["training"]["selection_window_updates"])
    candidates = [row for row in metrics if int(row["update"]) in selection_updates]
    selected = min(candidates, key=lambda row: row["validation"]["macro"]["b_core"])
    training_sufficiency_path = args.output / "training_sufficiency.json"
    atomic_json(training_sufficiency_path, {
        "format_version": "before-we-act.training-sufficiency/1",
        "status": status,
        "minimum_exposure_met": True,
        "u_b0h": MAX_UPDATES,
        "learning_rate_drop_completed": True,
        "selected_update": int(selected["update"]),
        "all_evaluation_points": [int(row["update"]) for row in metrics],
        "receipt": sufficiency,
    })
    selected_checkpoint = args.output / f"checkpoint_{int(selected['update']):06d}.pt"
    export_deployment(
        selected_checkpoint,
        args.output / "deployment_checkpoint.pt",
        contract,
        config,
    )
    atomic_json(args.output / "status.json", {
        "status": status,
        "seed": args.seed,
        "update": MAX_UPDATES,
        "selected_update": int(selected["update"]),
        "selected_validation": selected["validation"],
        "training_sufficiency_sha256": sha256_file(training_sufficiency_path),
        "deployment_checkpoint_sha256": sha256_file(args.output / "deployment_checkpoint.pt"),
        "elapsed_hours": (time.time() - started) / 3600,
        "completed_at_utc": utc_now(),
    })


if __name__ == "__main__":
    main()
