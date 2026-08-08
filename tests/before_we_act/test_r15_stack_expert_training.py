from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from before_we_act.data.raw_team_windows import TASKS
from before_we_act.train_r15_stack_expert import (
    PHASE_PROTOCOL,
    OriginalExpertStackSampler,
    PhaseBalancedOriginalExpertStackSampler,
    TASK,
)


def fake_dataset():
    stack = TASKS.index(TASK)
    episodes = [
        {"task": TASK, "steps": 3},
        {"task": TASK, "steps": 2, "source_episode_id": 0},
    ]
    requests = {index: [] for index in range(len(TASKS))}
    requests[stack] = [(0, step) for step in range(3)] + [
        (1, step) for step in range(2)
    ]
    return SimpleNamespace(episodes=episodes, requests_by_task=requests)


def test_original_expert_sampler_is_balanced_and_resume_stable():
    dataset = fake_dataset()
    complete = list(
        OriginalExpertStackSampler(
            dataset, updates=5, batch_size=6, expert_rows=3, seed=17
        )
    )
    resumed = list(
        OriginalExpertStackSampler(
            dataset,
            updates=5,
            batch_size=6,
            expert_rows=3,
            seed=17,
            start_update=3,
        )
    )
    assert resumed == complete[3:]
    for batch in complete:
        assert sum(episode == 1 for episode, _step in batch) == 3
        assert sum(episode == 0 for episode, _step in batch) == 3


def test_expert_runner_uses_manifest_branch_instead_of_main_only():
    runner = (
        Path(__file__).resolve().parents[2]
        / "scripts/before_we_act/run_r15_expert_finetune_candidate.sh"
    ).read_text()
    assert '"${IDENTITY[1]}" == "$CURRENT_BRANCH"' in runner
    assert '"${IDENTITY[1]}" == bwa/r15-closed-loop-evolution' not in runner


def phase_dataset_and_manifest():
    stack = TASKS.index(TASK)
    episodes = [
        {"task": TASK, "steps": 3},
        {
            "task": TASK,
            "steps": 6,
            "episode_index": 150,
            "seed": 5100,
            "source_episode_id": 0,
        },
    ]
    requests = {index: [] for index in range(len(TASKS))}
    requests[stack] = [(0, step) for step in range(3)] + [
        (1, step) for step in range(6)
    ]
    manifest = {
        "protocol": PHASE_PROTOCOL,
        "training_only_privileged_labels": True,
        "episodes": [
            {
                "source_episode_id": 0,
                "episode_index": 150,
                "seed": 5100,
                "steps": 6,
                "phase_boundaries": [0, 2, 4, 6],
            }
        ],
    }
    return SimpleNamespace(episodes=episodes, requests_by_task=requests), manifest


def test_phase_balanced_sampler_preserves_source_ratio_and_three_phases():
    dataset, manifest = phase_dataset_and_manifest()
    sampler = PhaseBalancedOriginalExpertStackSampler(
        dataset,
        manifest,
        updates=4,
        batch_size=12,
        expert_rows=9,
        seed=17,
    )
    batches = list(sampler)
    resumed = list(
        PhaseBalancedOriginalExpertStackSampler(
            dataset,
            manifest,
            updates=4,
            batch_size=12,
            expert_rows=9,
            seed=17,
            start_update=2,
        )
    )
    assert resumed == batches[2:]
    for batch in batches:
        expert_steps = [step for episode, step in batch if episode == 1]
        assert len(expert_steps) == 9
        assert sum(step < 2 for step in expert_steps) == 3
        assert sum(2 <= step < 4 for step in expert_steps) == 3
        assert sum(step >= 4 for step in expert_steps) == 3
        assert sum(episode == 0 for episode, _step in batch) == 3


def test_phase_balanced_sampler_requires_three_way_expert_rows():
    dataset, manifest = phase_dataset_and_manifest()
    with pytest.raises(ValueError, match="phase-balanced"):
        PhaseBalancedOriginalExpertStackSampler(
            dataset,
            manifest,
            updates=1,
            batch_size=12,
            expert_rows=8,
            seed=17,
        )


def test_phase_balanced_launcher_carries_manifest_to_trainer():
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "scripts/before_we_act/launch_r15_expert_finetune_tmux.sh"
    ).read_text()
    runner = (
        root / "scripts/before_we_act/run_r15_expert_finetune_candidate.sh"
    ).read_text()
    assert "phase-balanced-expert" in launcher
    assert "phase-balanced-expert" in runner
    assert "--phase-manifest" in launcher
    assert "--phase-manifest" in runner


def test_phase_balanced_promotion_precedes_e21_search():
    root = Path(__file__).resolve().parents[2]
    handoff = (
        root / "scripts/before_we_act/handoff_r15_phase_balanced_promotion.sh"
    ).read_text()
    assert handoff.index("DISCOVERY_ACCEPTANCE=") < handoff.index(
        "VALIDATION_ACCEPTANCE="
    ) < handoff.index("FORMAL_ACCEPTANCE=")
    assert "r15e45-20260808-phase-balanced-e9-ft5k-discovery20" in handoff
    assert "bwa-r15s-phase-e45" in handoff
    assert "bwa-r15s-phase-e30" not in handoff
    assert "r15e21-20260807-expert20-e6-ft5k-discovery20" in handoff
    assert "launch_r15_formal_stack_tmux.sh" in handoff
    assert "--session bwa-r15s-phase-e31" in handoff
    assert "--session bwa-r15s-phase-e32" in handoff
    assert "wait_for_gpu0 bwa-r15s-p1" not in handoff
