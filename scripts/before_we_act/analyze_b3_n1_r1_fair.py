#!/usr/bin/env python3
"""Evaluate selected R1-1 checkpoints and classify the fair-probe gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from before_we_act.b3_n1_data import N1RawSignalDataset
from before_we_act.b3_n1_r1 import (
    FrozenR1Backbones,
    R1FairProbeSet,
    R1_SEEDS,
    load_split,
    split_by_episode_key,
)
from before_we_act.step2_temporal_data import SIX_TASKS, sha256_file
from before_we_act.train_b3_n1_r1_fair_probe import evaluate_fair, fixed_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scenario-split", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def paired_group_bootstrap(samples: Sequence[Mapping], *, seed: int) -> dict:
    by_group: dict[str, list[float]] = {}
    for row in samples:
        delta = float(row["scores"]["h"] - row["scores"]["h_b"])
        by_group.setdefault(str(row["scenario_group"]), []).append(delta)
    groups = sorted(by_group)
    values = np.asarray([np.mean(by_group[group]) for group in groups], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "unit": "scenario_group_mean_paired_delta_h_minus_h_b",
        "groups": len(groups),
        "point": float(values.mean()),
        "ci95": [float(low), float(high)],
        "bootstrap_draws": 10_000,
        "seed": seed,
    }


def summarize(metrics: Mapping, *, seed: int) -> dict:
    macro = metrics["macro"]
    h = float(macro["h"])
    hb = float(macro["h_b"])
    per_task = {}
    for task_index, task in enumerate(SIX_TASKS):
        base = float(metrics["per_task"]["h"][str(task_index)])
        belief = float(metrics["per_task"]["h_b"][str(task_index)])
        per_task[task] = {
            "h": base,
            "h_b": belief,
            "absolute_h_minus_h_b": base - belief,
            "relative_improvement": (base - belief) / max(abs(base), 1e-12),
        }
    return {
        "macro": {condition: float(value) for condition, value in macro.items()},
        "absolute_h_minus_h_b": h - hb,
        "relative_improvement": (h - hb) / max(abs(h), 1e-12),
        "per_task": per_task,
        "bootstrap": paired_group_bootstrap(metrics["samples"], seed=seed),
        "rows": int(metrics["rows"]),
    }


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("stage_id") != "B3-N1-R1-ACTION-GROUNDED-BELIEF":
        raise RuntimeError("wrong R1 contract")
    split_payload = load_split(args.scenario_split)
    split = split_by_episode_key(split_payload)
    group_by_key = {
        row["episode_key"]: row["scenario_group"] for row in split_payload["episodes"]
    }
    statuses = {}
    for seed in R1_SEEDS:
        status_path = args.run_root / "r1_1_fair_probe" / f"seed_{seed}" / "status.json"
        statuses[str(seed)] = json.loads(status_path.read_text(encoding="utf-8"))
    sufficient = all(
        row["status"] in {"PLATFORM_REACHED", "SATURATED_BY_OVERFIT"}
        for row in statuses.values()
    )
    if not sufficient:
        payload = {
            "format_version": "before-we-act.b3-n1-r1-fair-conclusion/1",
            "stage": "R1-1-FAIR-PROBE",
            "status": "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
            "completed_at_utc": utc_now(),
            "training_status": statuses,
            "test_opened": False,
            "r1_2_required": False,
            "human_summary": "至少一个公平探针还没训练到平台，当前不能说 belief 有用或没用。",
        }
        atomic_json(args.output, payload)
        print(json.dumps({"status": payload["status"]}), flush=True)
        return

    dataset = N1RawSignalDataset(args.cache)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    evaluation: dict[str, dict] = {}
    for seed in R1_SEEDS:
        selected_update = int(statuses[str(seed)]["selected_update"])
        checkpoint_path = (
            args.run_root
            / "r1_1_fair_probe"
            / f"seed_{seed}"
            / f"checkpoint_{selected_update:06d}.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        probes = R1FairProbeSet().to(device)
        probes.load_state_dict(checkpoint["probes"], strict=True)
        n1_path = Path(
            contract["old_n1_read_only"]["representation_checkpoints"][str(seed)]["path"]
        )
        backbones = FrozenR1Backbones(
            b0h_checkpoint=Path(contract["b0h"]["checkpoint"]),
            n1_checkpoint=n1_path,
            visual_mean=dataset.visual_mean,
            visual_std=dataset.visual_std,
        ).to(device)
        evaluation[str(seed)] = {"selected_update": selected_update}
        for split_name in ("validation", "test"):
            metrics = evaluate_fair(
                backbones,
                probes,
                fixed_loader(dataset, split, split_name),
                dataset,
                group_by_key,
                device,
                include_samples=True,
            )
            evaluation[str(seed)][split_name] = summarize(
                metrics, seed=seed + (0 if split_name == "validation" else 1000)
            )
        del backbones, probes
        torch.cuda.empty_cache()

    every_seed_better = all(
        evaluation[str(seed)][split_name]["absolute_h_minus_h_b"] > 0
        for seed in R1_SEEDS
        for split_name in ("validation", "test")
    )
    task_medians: dict[str, dict] = {}
    positive_tasks = 0
    for task in SIX_TASKS:
        validation_values = [
            evaluation[str(seed)]["validation"]["per_task"][task]["relative_improvement"]
            for seed in R1_SEEDS
        ]
        test_values = [
            evaluation[str(seed)]["test"]["per_task"][task]["relative_improvement"]
            for seed in R1_SEEDS
        ]
        validation_median = float(np.median(validation_values))
        test_median = float(np.median(test_values))
        positive = validation_median > 0 and test_median > 0
        positive_tasks += int(positive)
        task_medians[task] = {
            "validation_relative_median": validation_median,
            "test_relative_median": test_median,
            "positive_both_splits": positive,
        }
    controls_clean = all(
        evaluation[str(seed)][split_name]["macro"]["h_b"]
        < evaluation[str(seed)][split_name]["macro"][control]
        for seed in R1_SEEDS
        for split_name in ("validation", "test")
        for control in ("h_b_shuffle", "h_matched_capacity")
    )
    passed = every_seed_better and positive_tasks >= 4 and controls_clean
    status = "PASSED_R1_1_FAIR_PROBE" if passed else "FAILED_R1_1_FAIR_PROBE"
    payload = {
        "format_version": "before-we-act.b3-n1-r1-fair-conclusion/1",
        "stage": "R1-1-FAIR-PROBE",
        "status": status,
        "completed_at_utc": utc_now(),
        "contract_sha256": sha256_file(args.contract),
        "scenario_split_sha256": sha256_file(args.scenario_split),
        "training_status": statuses,
        "test_opened": True,
        "evaluation": evaluation,
        "gate": {
            "h_b_beats_h_every_seed_on_validation_and_test": every_seed_better,
            "cross_seed_task_median_positive_both_splits": positive_tasks,
            "required_positive_tasks": 4,
            "shuffle_and_matched_capacity_do_not_reproduce": controls_clean,
            "passed": passed,
        },
        "task_medians": task_medians,
        "r1_2_required": not passed,
        "n2_authorized": False,
        "human_summary": (
            "修公平以后，旧 belief 确实在所有 seed、场景组和对照上给 B0-H 增添了动作信息；"
            "但还必须通过同状态分叉、全知教师和合法学生，不能现在进入 N2。"
            if passed
            else "修公平以后，旧 belief 仍没有稳定赢过真正冻结的 B0-H，所以下一步必须先做队友 oracle，判断是数据没答案，还是旧监督没学到答案。"
        ),
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "positive_tasks": positive_tasks,
                "every_seed_better": every_seed_better,
                "controls_clean": controls_clean,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
