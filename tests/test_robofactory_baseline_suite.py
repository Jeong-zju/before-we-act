from __future__ import annotations

import json
from pathlib import Path

from benchmarks.robofactory_baselines import (
    BASELINES,
    SIX_TASKS,
    aggregate_validation20,
    build_contract,
    validate_data_root,
)


def test_contract_covers_exact_requested_baselines_and_tasks(tmp_path: Path) -> None:
    contract = build_contract(data_root=tmp_path / "data", output_root=tmp_path / "out")
    assert tuple(contract["tasks"]) == SIX_TASKS
    assert [row["key"] for row in contract["baselines"]] == [
        "act", "dp", "latent_tom", "gaudp", "maniflow", "rdt_1b", "openvla_oft"
    ]
    assert contract["episodes_per_task"] == 20
    assert contract["validation"] == "closed_loop"


def test_validate_data_root_requires_every_manifest(tmp_path: Path) -> None:
    for task in SIX_TASKS:
        task_root = tmp_path / task
        task_root.mkdir()
        (task_root / "training_manifest.json").write_text(
            json.dumps({"episodes": [{"id": 1}]}), encoding="utf-8"
        )
        (task_root / "normalization.npz").touch()
    report = validate_data_root(tmp_path)
    assert report["valid"] is True
    assert set(report["tasks"]) == set(SIX_TASKS)


def test_aggregate_never_invents_missing_results(tmp_path: Path) -> None:
    result = tmp_path / "act" / SIX_TASKS[0]
    result.mkdir(parents=True)
    (result / "summary.json").write_text(
        json.dumps({"successes": 15, "episodes_completed": 20}), encoding="utf-8"
    )
    report = aggregate_validation20(tmp_path)
    act = report["baselines"]["act"]
    assert act["tasks_completed"] == 1
    assert act["micro_success_rate"] == 0.75
    assert all(
        report["baselines"][spec.key]["micro_success_rate"] is None
        for spec in BASELINES[1:]
    )
