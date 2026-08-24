import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_formal_act_contract_is_fully_pinned() -> None:
    contract = json.loads((ROOT / "configs/robofactory_act_formal_v1.json").read_text())
    assert contract["protocol"] == "robofactory_act_six_task_v1"
    assert len(contract["tasks"]) == 6
    assert contract["data"]["episodes_per_task"] == 120
    assert contract["training"]["batch_size"] == 40
    assert contract["training"]["updates"] == 120000
    assert contract["training"]["episode_block_updates"] == 64
    assert contract["training"]["seed"] == 20260819
    assert contract["closed_loop"]["episodes_per_task"] == 20
    assert contract["closed_loop"]["seed_start"] == 20260820
    assert contract["closed_loop"]["max_steps_profile"] == "care"
    assert contract["closed_loop"]["max_steps_by_task"] == {
        "lift_barrier": 500,
        "camera_alignment": 1500,
        "long_pipeline_delivery": 1500,
        "take_photo": 1500,
        "pass_shoe": 500,
        "place_food": 500,
    }
    assert contract["legacy_uniform_500_result"] == {
        "successes": 49,
        "episodes": 120,
        "macro_success_rate": 0.4083333333333333,
        "status": "superseded_by_care_horizon_validation20",
    }
    assert contract["reported_result"] == {
        "successes": 65,
        "episodes": 120,
        "macro_success_rate": 0.5416666666666666,
        "summary_sha256": "893a6321248118c34337c1b3efe5944288b170a9b9884bb72f89bd190d6e5e7f",
    }


def test_reproduction_launcher_exposes_both_stages() -> None:
    launcher = (ROOT / "scripts/reproduce_act_robofactory.sh").read_text()
    assert "adapt_wam_to_dp_zarr.py" in launcher
    assert "--formal-six-task" in launcher
    assert "evaluate_act_closed_loop.py" in launcher
    assert "--max-steps-profile care" in launcher
    assert "20260819" in launcher
    assert "20260820" in launcher


def test_care_horizon_validation20_runner_is_pinned() -> None:
    runner = (ROOT / "scripts/run_act_care_horizon_validation20.sh").read_text()
    assert "--max-steps-profile care" in runner
    assert "camera_alignment:1:0:5 camera_alignment:1:5:10" in runner
    assert "long_pipeline_delivery:2:0:5 long_pipeline_delivery:2:5:10" in runner
    assert "take_photo:3:0:5 take_photo:3:5:10" in runner
    assert "lift_barrier:1 pass_shoe:2 place_food:3" in runner
    assert "CUDA_VISIBLE_DEVICES=0" not in runner
    assert runner.count("--cpu-threads 10") == 3
    assert "shards do not cover episodes 0..19 exactly once" in runner
    assert "act_care_horizon_validation20_v1" in runner
