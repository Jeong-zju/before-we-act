#!/usr/bin/env python3
"""Freeze a pilot or formal roadmap Step-4 A4 contract."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from before_we_act.temporal_history_data import sha256_file


SOURCE_FILES = (
    "before_we_act/base_relative_belief.py",
    "before_we_act/base_relative_belief_losses.py",
    "before_we_act/deployment_safety.py",
    "before_we_act/train_base_relative_belief.py",
    "before_we_act/predictive_team_belief_training.py",
    "before_we_act/predictive_team_belief_policy.py",
    "before_we_act/temporal_action_backbone.py",
    "before_we_act/evaluate_predictive_team_belief.py",
    "before_we_act/team_belief/predictive_core.py",
    "before_we_act/team_belief/losses.py",
    "before_we_act/predictive_team_belief_data.py",
    "before_we_act/train_predictive_team_belief.py",
    "scripts/before_we_act/prepare_base_relative_belief.py",
    "scripts/before_we_act/evaluate_base_relative_sufficiency.py",
    "scripts/before_we_act/verify_base_relative_belief.py",
    "scripts/before_we_act/analyze_base_relative_belief.py",
    "scripts/before_we_act/analyze_bottleneck_isolation.py",
    "scripts/before_we_act/prepare_step4r_stability.py",
    "scripts/before_we_act/analyze_step4r_stability.py",
    "scripts/before_we_act/run_base_relative_training.sh",
    "scripts/before_we_act/run_base_relative_repair_isolation.sh",
    "scripts/before_we_act/run_step4r_stability.sh",
    "scripts/before_we_act/run_base_relative_closed_loop.sh",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-n2-contract", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "formal"), required=True)
    parser.add_argument("--beta-b", type=float, required=True)
    parser.add_argument("--prior-fit", type=float, default=0.1)
    parser.add_argument("--bradley-terry", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.005)
    parser.add_argument("--margin-fraction", type=float, default=0.1)
    parser.add_argument("--margin-cap", type=float, default=0.01)
    parser.add_argument("--pilot-seed", type=int, default=20260815)
    parser.add_argument("--pilot-updates", type=int, default=25_000)
    parser.add_argument("--pilot-eval-every", type=int, default=5_000)
    parser.add_argument("--pilot-status", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.beta_b,
        args.prior_fit,
        args.bradley_terry,
        args.margin_fraction,
        args.margin_cap,
    ) < 0.0:
        raise ValueError("Step-4 objective weights must be non-negative")
    if args.temperature <= 0.0:
        raise ValueError("Bradley-Terry temperature must be positive")
    if not 1 <= args.pilot_updates <= 120_000:
        raise ValueError("pilot updates must be in [1, 120000]")
    if (
        args.pilot_eval_every < 1
        or args.pilot_updates % args.pilot_eval_every != 0
    ):
        raise ValueError("pilot evaluation interval must divide pilot updates")

    base = json.loads(args.base_n2_contract.read_text(encoding="utf-8"))
    if base.get("format_version") != "before-we-act.b3-n2-contract/2":
        raise RuntimeError("Step-4 requires the accepted N2 contract")
    root = args.source_root.resolve()
    source_code = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_code[relative] = sha256_file(path)

    pilot_receipt = None
    if args.phase == "formal":
        if args.pilot_status is None or not args.pilot_status.is_file():
            raise RuntimeError("formal contract requires the completed pilot status")
        pilot = json.loads(args.pilot_status.read_text(encoding="utf-8"))
        if pilot.get("status") != "PASSED_LOSS_SCALE_PILOT":
            raise RuntimeError("formal contract requires a passed loss-scale pilot")
        pilot_receipt = {
            "path": str(args.pilot_status.resolve()),
            "sha256": sha256_file(args.pilot_status),
            "update": int(pilot["update"]),
            "last_losses": pilot.get("last_losses", {}),
            "selected_validation": pilot.get("selected_validation"),
        }

    objectives = deepcopy(base["objectives"])
    # The Step-3 hinge is replaced, not added a second time.
    objectives["action_pairing"] = 0.0
    objectives["action_pairing_margin_fraction"] = 0.0
    objectives["action_pairing_margin_cap"] = 0.0
    training = deepcopy(base["training"])
    training["loss_scale_pilot"] = {
        "seed": args.pilot_seed,
        "updates": args.pilot_updates,
        "evaluation_interval": args.pilot_eval_every,
        "closed_loop_forbidden": True,
        "selection_forbidden": True,
        "purpose": "freeze beta_b and loss scale after checking finite gradients, collapse, and magnitude only",
    }
    training["variants"] = ["a4_full", "a4_no_bottleneck"]
    training["pilot_variants"] = ["a4_no_bottleneck", "a4_full"]
    training["formal_independent_initialization"] = True
    training["formal_checkpoint_inheritance"] = "forbidden"
    training["validation5_scope"] = "three A4-full seeds only"
    if args.phase == "pilot":
        training["seeds"] = [args.pilot_seed]

    inputs = deepcopy(base["inputs"])
    inputs["base_n2_contract"] = str(args.base_n2_contract.resolve())
    inputs["base_n2_contract_sha256"] = sha256_file(args.base_n2_contract)
    contract = {
        "format_version": "before-we-act.a4-contract/1",
        "status": "FROZEN_PILOT" if args.phase == "pilot" else "FROZEN_FORMAL",
        "method": "base-relative-control-sufficient-belief",
        "runtime_path": "unchanged B-core: H -> (C,B) -> A0 + rho*g*DeltaA",
        "training_only_components": [
            "privileged posterior q_phi(B|H,Y_priv)",
            "base-conditioned categorical prior p_omega(B|C)",
        ],
        "architecture": deepcopy(base["architecture"])
        | {
            "base_conditioned_prior": "16 learned queries cross-attend frozen B0-H decoded action context C; no H/teacher/future/ID input",
            "deployment_prior_present": False,
            "deployment_residual_safety": "held-out p99 residual trust region plus entropy and temporal-consistency hard fallback; progress watchdog falls back to frozen B0-H",
        },
        "objectives": objectives,
        "base_relative_objectives": {
            "conditional_prior_fit": args.prior_fit,
            "bradley_terry": args.bradley_terry,
            "bradley_terry_temperature": args.temperature,
            "bradley_terry_margin_fraction": args.margin_fraction,
            "bradley_terry_margin_cap": args.margin_cap,
            "bradley_terry_form": "softplus(-gap/tau)-softplus(-margin/tau), clamped at zero; zero gradient after finite margin",
            "conditional_kl_split": "fit p toward stopgrad(q) in both arms; beta_b compresses q toward stopgrad(p) only",
        },
        "variants": {
            "a4_full": {"beta_b": args.beta_b},
            "a4_no_bottleneck": {"beta_b": 0.0},
        },
        "training": training,
        "inputs": inputs,
        "source_code": source_code,
        "pilot_receipt": pilot_receipt,
        "pilot_sensitivity_guard": {
            "output_to_target_sensitivity_max": 15.0,
            "output_to_residual_energy_max": 100.0,
            "shuffle_to_matched_ratio_max": 30.0,
        },
        "acceptance": {
            "belief_off_max_abs": 0.0,
            "control_sufficiency_gap_target_max": 0.05,
            "action_relevance_seeds_positive_min": 2,
            "action_relevance_mean_relative_improvement_min": 0.05,
            "per_seed_action_mse_vs_b_core_max_ratio": 1.05,
            "mean_action_mse_better_than_b0h": True,
            "validation5_mean_min_successes": 24,
            "validation5_single_seed_min_successes": 23,
            "validation20_selected_min_successes": 105,
            "deployment_residual_safety_required": True,
            "curve_stability_relative_range_max": 0.01,
            "conditional_minimality_is_soft": True,
            "nuisance_proxy_is_soft": True,
        },
    }
    atomic_json(args.output, contract)


if __name__ == "__main__":
    main()
