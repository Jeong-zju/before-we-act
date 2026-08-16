#!/usr/bin/env python3
"""Measure runtime-to-teacher belief estimation without the free-nats clamp."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch

from before_we_act.action_grounded_belief import load_split, split_by_episode_key
from before_we_act.predictive_team_belief_data import PredictiveTeamBeliefDataset
from before_we_act.predictive_team_belief_training import TeamBeliefExperiment
from before_we_act.temporal_history_data import sha256_file
from before_we_act.train_predictive_team_belief import (
    config_from_contract,
    device_batch,
    fixed_loader,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@torch.no_grad()
def measure(
    checkpoint_path: Path,
    contract: dict,
    loader,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TeamBeliefExperiment(config_from_contract(contract)).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    divergences: list[torch.Tensor] = []
    agreements: list[torch.Tensor] = []
    for raw in loader:
        output = model(device_batch(raw, device)).candidate
        if output.teacher is None:
            raise RuntimeError("belief estimation audit requires the training teacher")
        teacher = output.teacher
        belief = output.belief
        divergence = (
            teacher.categorical_probs
            * (teacher.categorical_log_probs - belief.categorical_log_probs)
        ).sum(-1)
        agreement = (
            teacher.categorical_probs.argmax(-1)
            == belief.categorical_probs.argmax(-1)
        )
        divergences.append(divergence.float().cpu().flatten())
        agreements.append(agreement.float().cpu().flatten())
    values = torch.cat(divergences)
    matches = torch.cat(agreements)
    return {
        "update": int(checkpoint["update"]),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "raw_kl_mean_nats_per_factor": float(values.mean()),
        "raw_kl_p50": float(values.quantile(0.50)),
        "raw_kl_p90": float(values.quantile(0.90)),
        "raw_kl_p99": float(values.quantile(0.99)),
        "fraction_over_one_nat": float((values > 1.0).float().mean()),
        "categorical_top1_agreement": float(matches.mean()),
        "rows": int(values.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--action-context-cache", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    dataset = PredictiveTeamBeliefDataset(args.cache, args.action_context_cache)
    split = split_by_episode_key(load_split(args.scenario_split))
    loader = fixed_loader(dataset, split, "validation")
    device = torch.device(args.device)
    baseline = measure(args.baseline_checkpoint, contract, loader, device)
    candidate = measure(args.candidate_checkpoint, contract, loader, device)
    kl_reduction = (
        baseline["raw_kl_mean_nats_per_factor"]
        - candidate["raw_kl_mean_nats_per_factor"]
    ) / max(baseline["raw_kl_mean_nats_per_factor"], 1e-12)
    chance = 1.0 / int(contract["architecture"]["belief_classes"])
    passed = (
        kl_reduction > 0.5
        and candidate["categorical_top1_agreement"] > 0.5
        and candidate["fraction_over_one_nat"] < 0.01
    )
    payload = {
        "format_version": "before-we-act.b3-n2-belief-estimation-audit/1",
        "status": "PASSED_BELIEF_ESTIMABLE" if passed else "FAILED_BELIEF_ESTIMABLE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "scope": "frozen validation episodes; raw per-factor categorical KL before free-nats clamp",
        "chance_top1_agreement": chance,
        "baseline": baseline,
        "candidate": candidate,
        "raw_kl_relative_reduction": kl_reduction,
        "gates": {
            "raw_kl_reduction_gt_50_percent": kl_reduction > 0.5,
            "candidate_top1_agreement_gt_50_percent": candidate[
                "categorical_top1_agreement"
            ]
            > 0.5,
            "candidate_fraction_over_one_nat_lt_1_percent": candidate[
                "fraction_over_one_nat"
            ]
            < 0.01,
        },
        "human_summary": (
            "运行分支已经能在未见 episode 上逼近全知教师的离散 belief。"
            if passed
            else "运行分支仍不能可靠逼近全知教师的离散 belief。"
        ),
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
