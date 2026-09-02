from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from before_we_act.mars_care_recorder import (
    MarsCARERolloutRecorder,
    derive_events,
)


def _observation(value: int = 0) -> dict[str, object]:
    return {
        "agent": {"panda-0": {"qpos": np.zeros(8, dtype=np.float32)}},
        "sensor_data": {
            "head_camera_agent0": {
                "rgb": np.full((1, 16, 24, 3), value, dtype=np.uint8)
            }
        },
    }


def test_event_derivation_marks_proxies_and_canonical_stack_name() -> None:
    before = {
        "progress": 0.1,
        "stage_id": "approach",
        "factorized_predicates": {"hammer_functional_point_distance": 0.2},
    }
    after = {
        "progress": 0.2,
        "stage_id": "struck",
        "factorized_predicates": {"hammer_functional_point_distance": 0.04},
    }
    strike = derive_events("strike_cube_hard", before, after)
    assert strike["strike_proxy"] is True
    assert strike["strike_proxy_is_not_contact"] is True
    assert strike["stage_changed"] is True

    stack = derive_events(
        "four_robots_stack_cube",
        {"factorized_predicates": {"is_cubeA_on_cubeB": False, "grasp_flags": [False]}},
        {"factorized_predicates": {"is_cubeA_on_cubeB": True, "grasp_flags": [True]}},
    )
    assert stack["cubeA_on_cubeB"] is True
    assert stack["grasp_transition"] is True


def test_recorder_flushes_telemetry_video_and_action_hash(tmp_path: Path) -> None:
    root = tmp_path / "episode"
    recorder = MarsCARERolloutRecorder(
        root, task="place_cube_in_cup", seed=7, arms=(0,), fps=10.0
    )
    observation = _observation(17)
    recorder.start(observation, metadata={"selector_mode": "selector_off"})
    action = {"panda-0": np.arange(8, dtype=np.float32) / 10}
    plan = {"panda-0": np.repeat(action["panda-0"][None], 100, axis=0)}
    candidates = [np.repeat(plan["panda-0"][None], 6, axis=0)]
    physical = {
        "progress": 0.2,
        "success": False,
        "stage_id": "approach",
        "collision_or_drop": False,
        "robot_conflict": False,
        "factorized_predicates": {
            "horizontal_distance": 0.2,
            "valid_rotation": True,
        },
    }
    qpos = {"panda-0": np.zeros(8, dtype=np.float32)}
    recorder.record_step(
        step=0,
        observation_before=observation,
        observation_after=observation,
        qpos_before=qpos,
        qpos_after=qpos,
        qpos_normalized=qpos,
        reference_plans=plan,
        base_plans=plan,
        candidates=candidates,
        candidate_legality=[[True] * 6],
        selected=[0],
        masked_lower=[[0.0, float("-inf"), 0.1, 0.2, 0.3, 0.4]],
        best_lower=[0.4],
        reason_names=["reference_below_delta"],
        illegal=[[False, True, False, False, False, False]],
        learned_unsafe=[[False] * 6],
        assembly={"applied_rows": [], "strict_decentralized": True},
        action_before_canonicalize=action,
        action_applied=action,
        action_bounds={
            "panda-0": {
                "low": np.full(8, -1, dtype=np.float32),
                "high": np.full(8, 1, dtype=np.float32),
            }
        },
        diagnostics={"gate": 0.1, "reliability": 0.2, "sigma": 0.3, "events": 0},
        physical=physical,
        info={"success": False},
    )
    summary = recorder.finish(
        success=False,
        final_observation=observation,
        final_info={"success": False},
        final_physical=physical,
    )
    assert summary["steps"] == 1
    assert summary["video_streams"]["panda-0"]["frames"] == 1
    assert (root / "videos/panda-0.mp4").stat().st_size > 0
    assert (root / "arrays/step_000000.npz").is_file()
    rows = [json.loads(line) for line in (root / "telemetry.jsonl").read_text().splitlines()]
    assert [row["type"] for row in rows] == ["episode_start", "step", "episode_end"]
    assert rows[1]["masked_lower"][0][1] is None
    assert rows[1]["privileged_metrics"]["progress"] == 0.2


def test_recorder_abort_retains_flushed_prefix(tmp_path: Path) -> None:
    root = tmp_path / "aborted"
    recorder = MarsCARERolloutRecorder(root, task="strike_cube_hard", seed=9, arms=(0,))
    recorder.start(_observation())
    recorder.abort(error=RuntimeError("simulator failed"))
    rows = [json.loads(line) for line in (root / "telemetry.jsonl").read_text().splitlines()]
    assert rows[-1]["type"] == "episode_abort"
    assert "simulator failed" in rows[-1]["error"]
