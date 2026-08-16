#!/usr/bin/env python3
"""Issue the fail-closed decision for the predictive-pairing repair pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    status = json.loads((args.pilot / "status.json").read_text(encoding="utf-8"))
    progress = [
        json.loads(line)
        for line in (args.pilot / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if status.get("status") != "PASSED_SMOKE" or "selected_validation" not in status:
        raise RuntimeError("repair pilot did not finish with its validation diagnostic")
    validation = status["selected_validation"]
    alignment = [float(row["teacher_alignment"]) for row in progress]
    kl_upper_bound = 8.867290
    kl_stable = bool(alignment) and all(
        math.isfinite(value) and value <= kl_upper_bound for value in alignment
    )

    belief = validation["belief"]
    effective_rank = float(belief["effective_rank"])
    active_factors = int(belief["active_categorical_factors_mi_gt_0_01"])
    multidimensional = effective_rank > 4.0 and active_factors >= 4

    pairing = validation["action_pairing"]
    shuffle_gap = float(pairing["shuffle_minus_correct_mse"])
    output_to_residual = float(pairing["output_to_residual_energy"])
    action_binding = (
        math.isfinite(shuffle_gap)
        and shuffle_gap > 0.0
        and math.isfinite(output_to_residual)
        and output_to_residual >= 0.01
    )
    b_core = float(validation["macro"]["b_core"])
    direct = float(validation["macro"]["direct_reactive"])
    action_quality = b_core <= 1.01 * direct

    future = validation["future_observable_mse"]
    horizons = [f"{seconds:.1f}s" for seconds in (0.2, 0.4, 0.8, 1.6)]
    persistence_wins = sum(
        float(future["oracle_action"][horizon])
        < float(future["persistence"][horizon])
        for horizon in horizons
    )
    shuffled_action_worse = sum(
        float(future["shuffled_action"][horizon])
        > float(future["oracle_action"][horizon])
        for horizon in horizons
    )
    future_prediction = persistence_wins >= 3 and shuffled_action_worse >= 3

    uncertainty = validation["uncertainty_occlusion"]
    epistemic_clean = float(uncertainty["epistemic_uncertainty_clean"])
    epistemic_occluded = float(uncertainty["epistemic_uncertainty_occluded"])
    reliability_clean = float(uncertainty["reliability_clean"])
    reliability_occluded = float(uncertainty["reliability_occluded"])
    evidence_clean = float(uncertainty["view_evidence_count_clean"])
    evidence_occluded = float(uncertainty["view_evidence_count_occluded"])
    uncertainty_direction = (
        epistemic_occluded > epistemic_clean
        and reliability_occluded < reliability_clean
        and evidence_occluded < evidence_clean
    )

    if not kl_stable:
        decision = "FAILED_KL_STABILITY"
        summary = "旧的 KL 问题复发，不能继续训练。"
    elif not multidimensional:
        decision = "FAILED_RETAINED_BELIEF_STATE"
        summary = "动作/未来修复破坏了已经得到的多维 belief，不能继续训练。"
    elif not action_binding:
        decision = "FAILED_ACTION_BELIEF_BINDING"
        summary = "正确 belief 和错配 belief 仍不能让真实动作输出拉开差距，不能靠延长训练解决。"
    elif not uncertainty_direction:
        decision = "FAILED_OCCLUSION_UNCERTAINTY"
        summary = "遮掉一个运行视角后，证据不确定性仍没有上升或可靠度没有下降，不能继续训练。"
    elif not future_prediction:
        decision = "FAILED_ACTION_CONDITIONED_FUTURE"
        summary = "未来头还没有公平地击败 persistence，或打乱动作后没有变差，不能靠延长训练解决。"
    elif not action_quality:
        decision = "FAILED_ACTION_QUALITY_GUARD"
        summary = "因果约束虽生效，但动作精度损失超过预注册上限，不能继续训练。"
    else:
        decision = "PASSED_CAUSAL_REPAIR_GATES_FORMAL_TRAINING_REQUIRES_OWNER_DECISION"
        summary = "belief 错配会伤害动作、动作错配会伤害未来预测，而且未来头公平击败 persistence；仍不自动启动长训练。"

    payload = {
        "format_version": "before-we-act.b3-n2-repair-pilot/2",
        "stage": contract["stage_id"],
        "status": decision,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "pilot": {
            "seed": status["seed"],
            "updates": status["update"],
            "logged_points": len(progress),
        },
        "gates": {
            "kl_stable": {
                "passed": kl_stable,
                "maximum_observed": max(alignment, default=float("nan")),
                "theoretical_upper_bound": kl_upper_bound,
                "final": alignment[-1] if alignment else None,
            },
            "retained_multidimensional_team_state": {
                "passed": multidimensional,
                "effective_rank": effective_rank,
                "active_categorical_factors_mi_gt_0_01": active_factors,
                "required_effective_rank_strictly_greater_than": 4.0,
                "required_active_factors": 4,
            },
            "action_belief_binding": {
                "passed": action_binding,
                "b_shuffle_minus_b_core_mse": shuffle_gap,
                "residual_output_mse_over_residual_energy": output_to_residual,
                "required_ratio": 0.01,
            },
            "action_conditioned_future": {
                "passed": future_prediction,
                "oracle_beats_legal_persistence_horizons": persistence_wins,
                "shuffled_action_worse_than_oracle_horizons": shuffled_action_worse,
                "required_each": 3,
                "mse": future,
                "learned_horizon_gain": validation["future_horizon_gain"],
            },
            "occlusion_uncertainty_direction": {
                "passed": uncertainty_direction,
                "categorical_entropy_is_diagnostic_not_gate": {
                    "clean": uncertainty["sigma_clean"],
                    "occluded": uncertainty["sigma_occluded"],
                },
                "epistemic_uncertainty_clean": epistemic_clean,
                "epistemic_uncertainty_occluded": epistemic_occluded,
                "reliability_clean": reliability_clean,
                "reliability_occluded": reliability_occluded,
                "view_evidence_count_clean": evidence_clean,
                "view_evidence_count_occluded": evidence_occluded,
            },
            "action_quality_guard": {
                "passed": action_quality,
                "b_core_mse": b_core,
                "direct_reactive_mse": direct,
                "maximum_relative_degradation": 0.01,
            },
        },
        "action_diagnostics": {
            "b_core_mse": validation["macro"]["b_core"],
            "b_shuffle_mse": validation["macro"]["b_shuffle"],
            "direct_reactive_mse": validation["macro"]["direct_reactive"],
            "b0h_mse": validation["macro"]["b0h"],
            "pairing": pairing,
        },
        "human_summary": summary,
        "formal_training_started": False,
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
