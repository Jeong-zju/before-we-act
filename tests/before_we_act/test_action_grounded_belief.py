from __future__ import annotations

from collections import Counter

import torch

from before_we_act.raw_team_signal_data import TeamEpisode
from before_we_act.action_grounded_belief import (
    FrozenBeliefFeatures,
    ActionGroundedBatchSampler,
    ActionGroundedProbeSet,
    PrivilegedOracleProbeSet,
    deterministic_permutations,
    oracle_predictions,
    predictions,
)
from before_we_act.belief_distillation import (
    DirectReactiveControl,
    LegalBeliefStudent,
    PrivilegedBeliefTeacher,
    gaussian_nll,
)


def episodes_and_split():
    episodes = []
    split = {}
    tasks = (
        "lift_barrier",
        "camera_alignment",
        "long_pipeline_delivery",
        "take_photo",
        "pass_shoe",
        "place_food",
    )
    for task_index, task in enumerate(tasks):
        for local_index in range(120):
            key = f"{task}-{local_index}"
            episodes.append(
                TeamEpisode(
                    task=task,
                    task_index=task_index,
                    local_index=local_index,
                    offset=local_index * 40,
                    length=40,
                    split="train",
                    episode_key=key,
                    hdf5_sha256=f"{task_index:02d}{local_index:062d}",
                )
            )
            split[key] = "train" if local_index < 96 else "validation" if local_index < 108 else "test"
    return episodes, split


def test_r1_cursor_is_balanced_and_group_split_restricted():
    episodes, split = episodes_and_split()
    sampler = ActionGroundedBatchSampler(
        episodes, split, updates=10, data_seed=20260815
    )
    first = sampler.requests_for_update(1)
    assert len(first) == 48
    assert Counter(row.task for row in first) == Counter(
        {task: 8 for task in {episode.task for episode in episodes}}
    )
    assert all(split[episodes[row.episode_index].episode_key] == "train" for row in first)
    assert sampler.requests_for_update(1) == first
    assert sampler.cursor_receipt(0)["next_sample_keys"] == [row.sample_key for row in first]


def test_full_token_heads_and_matched_capacity_are_shape_and_parameter_matched():
    torch.manual_seed(7)
    probes = ActionGroundedProbeSet().eval()
    frozen = FrozenBeliefFeatures(
        h=torch.randn(8, 384), belief=torch.randn(8, 16, 384)
    )
    batch = {
        "task_index": torch.arange(8) % 2,
        "phase": torch.linspace(0, 1, 8),
        "phase_bin": torch.arange(8) % 4,
        "episode_label": torch.arange(8),
    }
    output = predictions(probes, frozen, batch)
    assert set(output) == {
        "h",
        "b_only",
        "h_b",
        "h_b_shuffle",
        "h_matched_capacity",
        "h_b_row",
        "h_b_phase",
        "time",
    }
    assert all(tuple(value.shape) == (8, 16, 8) for value in output.values())
    assert probes.parameter_counts()["h_b"] == probes.parameter_counts()["h_matched_capacity"]


def test_deterministic_controls_have_no_fixed_points():
    batch = {
        "task_index": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        "phase": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
        "phase_bin": torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]),
        "episode_label": torch.arange(8),
    }
    first = deterministic_permutations(batch)
    second = deterministic_permutations(batch)
    for name in first:
        assert torch.equal(first[name], second[name])
        assert not torch.any(first[name] == torch.arange(8))


def test_oracle_and_zero_information_capacity_heads_are_parameter_matched():
    probes = PrivilegedOracleProbeSet().eval()
    frozen = FrozenBeliefFeatures(
        h=torch.randn(8, 384), belief=torch.randn(8, 16, 384)
    )
    batch = {
        "task_index": torch.arange(8) % 2,
        "phase": torch.linspace(0, 1, 8),
        "phase_bin": torch.arange(8) % 4,
        "episode_label": torch.arange(8),
        "teammate_qpos": torch.randn(8, 9),
        "previous_teammate_qpos": torch.randn(8, 9),
        "teammate_delta": torch.randn(8, 4, 9),
        "future_mask": torch.ones(8, 4, dtype=torch.bool),
        "oracle_teammate_action": torch.randn(8, 16, 8),
        "oracle_teammate_action_mask": torch.ones(8, 16, dtype=torch.bool),
    }
    output = oracle_predictions(probes, frozen, batch)
    assert all(tuple(value.shape) == (8, 16, 8) for value in output.values())
    counts = probes.parameter_counts()
    assert counts["h_oracle"] == counts["h_matched_capacity"]


def _oracle_batch(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "teammate_qpos": torch.randn(batch_size, 9),
        "previous_teammate_qpos": torch.randn(batch_size, 9),
        "teammate_delta": torch.randn(batch_size, 4, 9),
        "future_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "oracle_teammate_action": torch.randn(batch_size, 16, 8),
        "oracle_teammate_action_mask": torch.ones(
            batch_size, 16, dtype=torch.bool
        ),
    }


def test_teacher_and_legal_student_preserve_full_token_and_action_contracts():
    batch_size = 3
    base = torch.randn(batch_size, 16, 8)
    hidden = torch.randn(batch_size, 384)
    history = torch.randn(batch_size, 16, 384)
    history_mask = torch.ones(batch_size, 16, dtype=torch.bool)
    history_mask[0, :5] = False

    teacher = PrivilegedBeliefTeacher().eval()
    student = LegalBeliefStudent().eval()
    direct = DirectReactiveControl().eval()
    teacher_output = teacher(base, hidden, _oracle_batch(batch_size))
    student_output = student(base, hidden, history, history_mask)

    assert tuple(teacher_output.tokens.shape) == (batch_size, 16, 384)
    assert tuple(student_output.tokens.shape) == (batch_size, 16, 384)
    assert tuple(teacher_output.action.shape) == (batch_size, 16, 8)
    assert tuple(student_output.action.shape) == (batch_size, 16, 8)
    assert tuple(direct(base, hidden, history, history_mask).shape) == (
        batch_size,
        16,
        8,
    )
    assert tuple(teacher_output.teammate_action_mean.shape) == (
        batch_size,
        16,
        8,
    )
    assert not hasattr(teacher_output, "shared_change")
    assert not hasattr(teacher_output, "branch_value")
    assert tuple(student_output.teammate_delta.shape) == (batch_size, 4, 9)
    assert tuple(student_output.future_visual.shape) == (batch_size, 4, 2, 768)


def test_residual_heads_start_at_exact_h_and_belief_off_is_bitwise_h():
    batch_size = 2
    base = torch.randn(batch_size, 16, 8)
    hidden = torch.randn(batch_size, 384)
    history = torch.randn(batch_size, 16, 384)
    mask = torch.ones(batch_size, 16, dtype=torch.bool)
    oracle = _oracle_batch(batch_size)

    teacher = PrivilegedBeliefTeacher().eval()
    student = LegalBeliefStudent().eval()
    direct = DirectReactiveControl().eval()

    assert torch.equal(teacher(base, hidden, oracle).action, base)
    assert torch.equal(student(base, hidden, history, mask).action, base)
    assert torch.equal(direct(base, hidden, history, mask), base)
    assert torch.equal(student.belief_off(base), base)


def test_gaussian_nll_ignores_masked_rows_and_stays_finite():
    mean = torch.zeros(2, 3, 4)
    logvar = torch.zeros_like(mean)
    target = torch.ones_like(mean)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    loss = gaussian_nll(mean, logvar, target, mask)
    assert torch.isfinite(loss)
    assert torch.equal(loss, torch.tensor(0.5))


def test_student_and_direct_have_exactly_matched_action_path_capacity():
    student = LegalBeliefStudent()
    direct = DirectReactiveControl()
    student_action_path = (
        student.queries.numel()
        + sum(parameter.numel() for parameter in student.reader.parameters())
        + sum(parameter.numel() for parameter in student.token_norm.parameters())
        + sum(
            parameter.numel()
            for parameter in student.action_residual.parameters()
        )
    )
    direct_action_path = sum(parameter.numel() for parameter in direct.parameters())
    assert student_action_path == direct_action_path
