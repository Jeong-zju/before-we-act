import json

import pytest

from scripts.before_we_act.summarize_b3_n2_validation20 import (
    EXPECTED_BASELINES,
    SIX_TASKS,
    load_baseline,
    select_validation20_candidate,
    sha256_file,
    summarize,
)


def test_validation20_seed_selection_never_reads_closed_loop() -> None:
    training = {
        "20260815": {
            "selected_update": 120000,
            "selected_validation": {"macro": {"b_core": 0.3}},
            "deployment_checkpoint": "/run/15.pt",
            "deployment_checkpoint_sha256": "15",
        },
        "20260816": {
            "selected_update": 115000,
            "selected_validation": {"macro": {"b_core": 0.2}},
            "deployment_checkpoint": "/run/16.pt",
            "deployment_checkpoint_sha256": "16",
        },
        "20260817": {
            "selected_update": 110000,
            "selected_validation": {"macro": {"b_core": 0.2}},
            "deployment_checkpoint": "/run/17.pt",
            "deployment_checkpoint_sha256": "17",
        },
    }

    selected = select_validation20_candidate(training)

    assert selected["seed"] == 20260816
    assert selected["selected_update"] == 115000
    assert selected["closed_loop_results_used_for_selection"] is False


def test_validation20_summary_compares_frozen_baselines(tmp_path) -> None:
    seed_root = tmp_path / "seeds"
    validation_root = tmp_path / "validation"
    seed_root.mkdir()
    validation_root.mkdir()
    checkpoint_sha = "candidate"
    n2_successes = {}
    for index, task in enumerate(SIX_TASKS):
        seeds = list(range(index * 100, index * 100 + 20))
        seed_file = seed_root / f"{task}.json"
        seed_file.write_text(json.dumps({"seeds": seeds}), encoding="utf-8")
        successes = 20 if task != "place_food" else 10
        n2_successes[task] = successes
        rows = [
            {
                "seed": seed,
                "success": position < successes,
                "steps": 10,
                "paired_inactivity_steps": 1,
            }
            for position, seed in enumerate(seeds)
        ]
        (validation_root / f"{task}.json").write_text(
            json.dumps(
                {
                    "mode": "n2",
                    "episodes": 20,
                    "successes": successes,
                    "steps": 200,
                    "paired_inactivity_steps": 20,
                    "rows": rows,
                    "checkpoint_sha256": checkpoint_sha,
                    "seed_protocol": {"sha256": sha256_file(seed_file)},
                }
            ),
            encoding="utf-8",
        )
    baseline_tasks = {
        task: {"episodes": 20, "successes": 15, "success_rate": 0.75}
        for task in SIX_TASKS
    }
    w10 = {"checkpoint_sha256": "w10", "successes": 88, "tasks": baseline_tasks}
    b0h = {"checkpoint_sha256": "b0h", "successes": 95, "tasks": baseline_tasks}
    conclusion = {
        "status": "POSITIVE_SIGNAL",
        "validation20_candidate": {
            "seed": 20260816,
            "deployment_checkpoint_sha256": checkpoint_sha,
            "closed_loop_results_used_for_selection": False,
        },
    }

    result = summarize(
        conclusion, validation_root, seed_root, w10=w10, b0h=b0h
    )

    assert result["n2"]["successes"] == sum(n2_successes.values())
    assert result["comparison"]["versus_w10"]["success_delta"] == 22
    assert result["comparison"]["versus_b0h"]["success_delta"] == 15
    assert result["formal_pass"] is False


def test_baseline_loader_verifies_per_task_seed_receipts(tmp_path, monkeypatch) -> None:
    seed_root = tmp_path / "seeds"
    result_root = tmp_path / "baseline"
    seed_root.mkdir()
    result_root.mkdir()
    checkpoint = "/frozen/checkpoint.pt"
    checkpoint_sha = "frozen-sha"
    monkeypatch.setitem(
        EXPECTED_BASELINES,
        "w10",
        {"checkpoint_sha256": checkpoint_sha, "successes": 120},
    )
    tasks = {}
    for index, task in enumerate(SIX_TASKS):
        seeds = list(range(index * 100, index * 100 + 20))
        seed_file = seed_root / f"{task}.json"
        seed_file.write_text(json.dumps({"seeds": seeds}), encoding="utf-8")
        result = {
            "checkpoint": checkpoint,
            "episodes": 20,
            "successes": 20,
            "rows": [{"seed": seed, "success": True} for seed in seeds],
            "seed_protocol": {"sha256": sha256_file(seed_file)},
        }
        (result_root / f"{task}.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        tasks[task] = {"episodes": 20, "successes": 20}
    summary_path = result_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha,
                "episodes": 120,
                "successes": 120,
                "tasks": tasks,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_baseline(summary_path, seed_root, "w10")

    assert set(loaded["validation_receipts"]) == set(SIX_TASKS)


def test_validation20_requires_positive_validation5(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="positive frozen Validation5 or owner exception"):
        summarize(
            {"status": "WEAK_SIGNAL", "validation20_candidate": {}},
            tmp_path,
            tmp_path,
            w10={},
            b0h={},
        )


def test_validation20_accepts_frozen_owner_exception(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.before_we_act.summarize_b3_n2_validation20.load_candidate",
        lambda *_args, **_kwargs: {
            "episodes": 120,
            "successes": 90,
            "tasks": {
                task: {"episodes": 20, "successes": 15}
                for task in SIX_TASKS
            },
        },
    )
    baseline = {
        "checkpoint_sha256": "baseline",
        "successes": 90,
        "tasks": {
            task: {"episodes": 20, "successes": 15}
            for task in SIX_TASKS
        },
    }
    conclusion = {
        "status": "OWNER_AUTHORIZED_CLOSED_LOOP_AFTER_PRIMARY_PLATEAU",
        "validation20_candidate": {
            "deployment_checkpoint_sha256": "candidate",
            "closed_loop_results_used_for_selection": False,
        },
    }

    result = summarize(
        conclusion, tmp_path, tmp_path, w10=baseline, b0h=baseline
    )

    assert result["status"] == "COMPLETED_OWNER_AUTHORIZED_VALIDATION20_DIAGNOSTIC"
    assert result["formal_pass"] is False
