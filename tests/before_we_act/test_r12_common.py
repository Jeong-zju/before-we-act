from __future__ import annotations

import json
from pathlib import Path

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
