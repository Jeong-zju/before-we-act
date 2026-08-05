from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

import torch

from before_we_act.benchmark import TASKS
from before_we_act.contracts import ActionProposalBatch
from before_we_act.data.action_windows import ExactFiveTaskWindowSampler


def test_action_proposal_rejects_absent_agent_motion():
    actions = torch.zeros(2, 1, 4, 100, 8)
    actions[0, 0, 3, 0, 0] = 1
    proposal = ActionProposalBatch(
        actions=actions,
        base_index=0,
        valid_mask=torch.ones(2, 1, dtype=torch.bool),
        agent_mask=torch.tensor([[True, True, True, False], [True, True, True, True]]),
        source=("candidate",),
    )
    try:
        proposal.validate()
    except ValueError as exc:
        assert "absent-agent" in str(exc)
    else:
        raise AssertionError("absent agent action was accepted")


def test_exact_sampler_has_one_window_per_task_and_is_resumable():
    indices = torch.tensor([0, 1, 2, 3, 4] * 4)
    full = list(ExactFiveTaskWindowSampler(indices, updates=4, seed=12))
    resumed = list(ExactFiveTaskWindowSampler(indices, updates=4, seed=12, start_update=2))
    assert resumed == full[2:]
    assert all(sorted(indices[row].tolist()) == [0, 1, 2, 3, 4] for row in full)


def test_gate20_task_order_matches_frozen_contract():
    assert tuple(TASKS) == (
        "lift_barrier",
        "camera_alignment",
        "three_robots_stack_cube",
        "long_pipeline_delivery",
        "take_photo",
    )


def test_r12_runtime_reports_true_gate20_progress(tmp_path: Path):
    from scripts.before_we_act.r12_runtime import gate20_progress

    root = tmp_path / "candidate"
    for index, task in enumerate(TASKS):
        path = root / "validation/gate20" / f"{task}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"episodes": 20, "successes": index, "latency_ms": {"p95": index + 1}}),
            encoding="utf-8",
        )
    result = gate20_progress(root)
    assert result["complete_tasks"] == 5
    assert result["episodes"] == 100
    assert result["successes"] == 10
    assert result["p95"] == 5


def test_r12_atomic_json_allows_concurrent_status_and_heartbeat(tmp_path: Path):
    from scripts.before_we_act.r12_runtime import atomic_json

    path = tmp_path / "heartbeat.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: atomic_json(path, {"index": index}), range(200)))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["index"] in range(200)
    assert not list(tmp_path.glob(".heartbeat.json.*.tmp"))


def test_r12_decision_uses_gpu_hours_before_candidate_id(tmp_path: Path):
    acceptances = []
    statuses = []
    for index in range(4):
        candidate = f"p{index}"
        acceptance = tmp_path / f"{candidate}_acceptance.json"
        status = tmp_path / f"{candidate}_status.json"
        acceptance.write_text(
            json.dumps(
                {
                    "candidate_id": candidate,
                    "qualified": True,
                    "valid_component": True,
                    "commit": f"commit-{candidate}",
                    "checkpoint": f"/{candidate}.pt",
                    "latency_p95_ms_max_task": 10,
                    "gate20": {
                        "candidate_total_successes": 80,
                        "tasks": {
                            task: {"candidate": 16, "paired_wins": 1}
                            for task in TASKS
                        },
                    },
                    "acceptance": [],
                }
            ),
            encoding="utf-8",
        )
        # P1 is the only faster equally qualified candidate.  Candidate ID
        # would otherwise choose P0, so this isolates the GPU-hours ordering.
        terminal_minute = 30 if candidate == "p1" else 40
        status.write_text(
            json.dumps(
                {
                    "candidate": candidate,
                    "created_at": "2026-08-05T00:00:00Z",
                    "updated_at": f"2026-08-05T00:{terminal_minute}:00Z",
                }
            ),
            encoding="utf-8",
        )
        acceptances.extend(["--acceptance", f"{candidate}={acceptance}"])
        statuses.extend(["--status", f"{candidate}={status}"])
    output = tmp_path / "decision.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/before_we_act/decide_r12_winner.py",
            *acceptances,
            *statuses,
            "--baseline-commit",
            "baseline",
            "--baseline-checkpoint-sha256",
            "0" * 64,
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["unique_winner"] == "p1"
    assert decision["candidate_results"]["p1"]["gpu_hours"] == 0.5
