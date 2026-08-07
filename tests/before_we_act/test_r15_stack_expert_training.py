from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from before_we_act.data.raw_team_windows import TASKS
from before_we_act.train_r15_stack_expert import OriginalExpertStackSampler, TASK


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
