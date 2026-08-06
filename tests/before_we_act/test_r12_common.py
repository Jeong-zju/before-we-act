from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

import h5py
import numpy as np
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


def test_exact_sampler_can_force_task_balanced_recovery_rows():
    task_indices = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    sources = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    batches = list(
        ExactFiveTaskWindowSampler(
            task_indices,
            updates=3,
            seed=9,
            source_indices=sources,
            recovery_probability=1.0,
        )
    )
    assert all(sorted(task_indices[row].tolist()) == [0, 1, 2, 3, 4] for row in batches)
    assert all(sources[row].eq(1).all() for row in batches)


def test_causal_history_matches_closed_loop_cold_start_and_lag():
    from scripts.before_we_act.prepare_r12_action_cache import (
        causal_history_indices,
        previous_action_index,
    )

    assert causal_history_indices(0) == [0, 0, 0]
    assert causal_history_indices(1) == [0, 0, 1]
    assert causal_history_indices(2) == [0, 1, 2]
    assert causal_history_indices(7) == [5, 6, 7]
    assert previous_action_index(0) is None
    assert previous_action_index(1) == 0
    assert previous_action_index(7) == 6


def test_causal_cache_selection_adds_every_episode_cold_start():
    from scripts.before_we_act.prepare_r12_action_cache import choose_examples

    manifests = {
        task: {
            "episodes": [
                {
                    "split": "train",
                    "steps": 10,
                    "hdf5_path": f"{task}.hdf5",
                }
            ]
        }
        for task in TASKS
    }
    rows = choose_examples(manifests, "train", per_episode=5, seed=12)
    cold = {(task, current) for task, _episode, current in rows if current < 3}

    assert len(rows) == 5 * len(TASKS) + 3 * len(TASKS)
    assert cold == {(task, current) for task in TASKS for current in (0, 1, 2)}


def test_history_robustification_is_deterministic_causal_and_masked():
    from before_we_act.train_action_generator import robustify_action_history

    batch = {
        "actions": torch.arange(2 * 3 * 4 * 8, dtype=torch.float32).reshape(2, 3, 4, 8),
        "agent_mask": torch.tensor([[True, True, False, False], [True, True, True, False]]),
    }
    training = {
        "history_augmentation_probability": 1.0,
        "history_augmentation_ramp_updates": 1,
        "history_noise_scale": 0.25,
    }
    stats = {"a_std": torch.ones(8)}
    first, first_metrics = robustify_action_history(batch, stats, training, 11, 7)
    second, second_metrics = robustify_action_history(batch, stats, training, 11, 7)
    torch.testing.assert_close(first, second)
    assert first_metrics == second_metrics
    assert first_metrics["history_aug_fraction"] == 1.0
    assert first[0, :, 2:].eq(0).all()
    assert first[1, :, 3:].eq(0).all()


def test_r12_r3_learning_rate_warmup_and_decay():
    from before_we_act.train_action_generator import learning_rate_at_update

    training = {
        "learning_rate": 1e-3,
        "warmup_steps": 100,
        "decay_steps": 1000,
        "decay_lr_ratio": 0.1,
    }
    assert learning_rate_at_update(training, 1) == 1e-5
    assert learning_rate_at_update(training, 100) == 1e-3
    assert abs(learning_rate_at_update(training, 1000) - 1e-4) < 1e-12


def test_r12_r3_spatial_probe_preserves_view_and_grid_tokens():
    from scripts.before_we_act.probe_r12_representation import ProbeHead

    batch = {
        "belief_tokens": torch.randn(2, 21, 96),
        "belief_mask": torch.ones(2, 21, dtype=torch.bool),
        "spatial_tokens": torch.randn(2, 5, 16, 768),
        "spatial_view_mask": torch.tensor(
            [[True, True, False, False, False], [True, True, True, True, False]]
        ),
    }
    for mode in ("w11", "spatial", "fused"):
        action, progress = ProbeHead(mode)(batch)
        assert action.shape == (2, 4, 8)
        assert progress.shape == (2,)
        assert torch.isfinite(action).all()
        assert torch.isfinite(progress).all()


def test_r12_r3_spatial_observation_contract_is_hash_locked():
    from before_we_act.spatial_observation import locked_r12_spatial_observation

    contract = locked_r12_spatial_observation()
    assert contract["spatial_grid"] == [4, 4]
    assert contract["max_views"] == 5
    assert contract["feature_dim"] == 768
    assert len(contract["weights_sha256"]) == 64
    assert contract["fusion"] == "zero_gated_cross_attention_into_w11_tokens"


def test_causal_cache_reads_only_prior_actions_and_matches_cold_start(tmp_path: Path):
    from scripts.before_we_act.prepare_r12_action_cache import read_causal_example

    task = next(iter(TASKS))
    episode_dir = tmp_path / task
    episode_dir.mkdir()
    episode = episode_dir / "episode.hdf5"
    with h5py.File(episode, "w") as handle:
        data = handle.create_group("data")
        observations = data.create_group("observation")
        agent = observations.create_group("agents").create_group("panda-0")
        agent.create_dataset(
            "qpos", data=np.arange(6 * 9, dtype=np.float32).reshape(6, 9)
        )
        images = observations.create_group("images")
        images.create_dataset("global", data=np.zeros((6, 4, 4, 3), dtype=np.uint8))
        images.create_dataset("agent_0", data=np.ones((6, 4, 4, 3), dtype=np.uint8))
        actions = data.create_group("action").create_group("agents").create_group("panda-0")
        actions.create_dataset(
            "executed", data=100 + np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
        )
        actions.create_dataset(
            "commanded", data=200 + np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
        )
    metadata = {"hdf5_path": "episode.hdf5", "steps": 6}
    stats = {"a_mean": np.zeros(8, dtype=np.float32), "a_std": np.ones(8, dtype=np.float32)}

    cold = read_causal_example(tmp_path, (task, metadata, 0), stats, horizon=4)
    later = read_causal_example(tmp_path, (task, metadata, 2), stats, horizon=4)

    assert cold["qpos"][:, 0].eq(cold["qpos"][0, 0]).all()
    assert cold["actions"].eq(0).all()
    torch.testing.assert_close(
        cold["joint_actions"][0, 0], torch.arange(8, dtype=torch.float32) + 200
    )
    torch.testing.assert_close(later["actions"][0, 0], torch.zeros(8))
    torch.testing.assert_close(
        later["actions"][1, 0], torch.arange(8, dtype=torch.float32) + 100
    )
    torch.testing.assert_close(
        later["actions"][2, 0], torch.arange(8, dtype=torch.float32) + 108
    )
    torch.testing.assert_close(
        later["joint_actions"][0, 0], torch.arange(8, dtype=torch.float32) + 216
    )


def test_grouped_cache_reader_is_exactly_equal_to_single_window_reader(tmp_path: Path):
    from scripts.before_we_act.prepare_r12_action_cache import (
        read_causal_episode_group,
        read_causal_example,
    )

    task = next(iter(TASKS))
    episode_dir = tmp_path / task
    episode_dir.mkdir()
    episode = episode_dir / "episode.hdf5"
    with h5py.File(episode, "w") as handle:
        data = handle.create_group("data")
        observations = data.create_group("observation")
        agent = observations.create_group("agents").create_group("panda-0")
        agent.create_dataset("qpos", data=np.arange(8 * 9, dtype=np.float32).reshape(8, 9))
        images = observations.create_group("images")
        pixels = np.arange(8 * 4 * 4 * 3, dtype=np.uint8).reshape(8, 4, 4, 3)
        images.create_dataset("global", data=pixels)
        images.create_dataset("agent_0", data=pixels[:, ::-1])
        actions = data.create_group("action").create_group("agents").create_group("panda-0")
        actions.create_dataset("executed", data=np.arange(8 * 8, dtype=np.float32).reshape(8, 8))
        actions.create_dataset("commanded", data=100 + np.arange(8 * 8, dtype=np.float32).reshape(8, 8))
    metadata = {"hdf5_path": "episode.hdf5", "steps": 8}
    stats = {"a_mean": np.arange(8, dtype=np.float32), "a_std": np.arange(8, dtype=np.float32) + 1}
    grouped = read_causal_episode_group(tmp_path, task, metadata, [0, 2, 5], stats, horizon=4)
    for current in (0, 2, 5):
        single = read_causal_example(tmp_path, (task, metadata, current), stats, horizon=4)
        assert grouped[current].keys() == single.keys()
        for key in single:
            torch.testing.assert_close(grouped[current][key], single[key])


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
