from __future__ import annotations

import json
from pathlib import Path
import subprocess

import torch

from models.wam_multimodal import (
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
)
from scripts.accept_s2_r3 import REQUIRED_TASKS, build_acceptance
from scripts.s2_r3_runtime import (
    initialize_run,
    render_monitor,
    update_status,
)
from scripts.verify_s2_r3_dataset_local import quick_validate_dataset
from train.s2_future_prediction import masked_future_prediction_losses
from train.s2_grouped_trajectory import grouped_s2_batch
from train.s2_model_registry import S2_R3_MODEL_KINDS


ROOT = Path(__file__).resolve().parents[1]


def _raw_grouped_batch() -> dict[str, torch.Tensor]:
    batch_size = 2
    future_states = torch.zeros(batch_size, 100, 72)
    for step in range(100):
        future_states[:, step] = float(step + 1)
    return {
        "dataset_index": torch.tensor([3, 7]),
        "task_index": torch.tensor([0, 1]),
        "episode_index": torch.tensor([11, 12]),
        "episode_seed": torch.tensor([101, 102]),
        "decision_t": torch.tensor([5, 9]),
        "states": torch.zeros(batch_size, 1, 72),
        "action_targets": torch.randn(batch_size, 100, 32),
        "embodiment_index": torch.tensor([1, 2]),
        "images": torch.zeros(batch_size, 1, 5, 3, 4, 6, dtype=torch.uint8),
        "image_valid_mask": torch.tensor(
            [
                [[True, True, True, False, False]],
                [[True, True, True, True, False]],
            ]
        ),
        "future_states": future_states,
        "future_state_valid_mask": torch.tensor(
            [
                [True] * 100,
                [True] * 50 + [False] * 50,
            ]
        ),
        "future_images": torch.zeros(
            batch_size, 4, 5, 3, 4, 6, dtype=torch.uint8
        ),
        "future_visual_valid_mask": torch.tensor(
            [
                [[True] * 5] * 4,
                [[True] * 5, [True] * 5, [True] * 5, [False] * 5],
            ]
        ),
        "future_horizons": torch.tensor([[1, 25, 50, 100]] * batch_size),
    }


def test_grouped_adapter_preserves_agent_global_and_future_masks():
    grouped = grouped_s2_batch(_raw_grouped_batch())

    assert grouped["current_state"].shape == (2, 4, 18)
    assert grouped["candidate_actions"].shape == (2, 4, 100, 8)
    assert grouped["agent_observations"].shape == (2, 4, 3, 4, 6)
    assert grouped["shared_observation"].shape == (2, 3, 4, 6)
    assert grouped["future_agent_observations"].shape == (2, 4, 4, 3, 4, 6)
    assert grouped["valid_agent_mask"].tolist() == [
        [True, True, False, False],
        [True, True, True, False],
    ]
    assert grouped["future_state_delta"][0, 0, :, 0].tolist() == [
        1.0,
        25.0,
        50.0,
        100.0,
    ]
    assert grouped["future_state_valid_mask"][1, 0].tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert not bool(grouped["future_state_valid_mask"][:, 3].any())


def test_local_predictor_has_matched_parameters_and_w0_masks_actions():
    config = LocalFuturePredictorConfig(
        action_horizon=6,
        future_horizons=(1, 2),
        visual_grid_tokens=2,
        visual_latent_dim=8,
        d_model=16,
        ffn_dim=32,
        layers=1,
        heads=4,
        dropout=0.0,
    )
    model = LocalActionConditionedFuturePredictor(config).eval()
    state = torch.randn(2, 4, 18)
    visual = torch.randn(2, 4, 2, 8)
    first = torch.randn(2, 4, 6, 8)
    second = torch.randn(2, 4, 6, 8)
    valid = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    off = torch.zeros_like(valid)
    on = valid

    w0_first = model(state, visual, first, valid, off)
    w0_second = model(state, visual, second, valid, off)
    w1_first = model(state, visual, first, valid, on)
    w1_second = model(state, visual, second, valid, on)

    torch.testing.assert_close(w0_first[0], w0_second[0], rtol=0, atol=0)
    torch.testing.assert_close(w0_first[1], w0_second[1], rtol=0, atol=0)
    assert not torch.equal(w1_first[0], w1_second[0])
    assert not torch.equal(w1_first[1], w1_second[1])
    assert w0_first[0][:, 3].count_nonzero().item() == 0
    assert w0_first[1][:, 3].count_nonzero().item() == 0


def test_invalid_agent_slots_contribute_zero_to_s2_loss():
    target_state = torch.zeros(1, 4, 2, 18)
    predicted_state = torch.zeros_like(target_state)
    target_visual = torch.ones(1, 4, 2, 2, 8)
    predicted_visual = target_visual.clone()
    valid = torch.tensor([[[True, True], [True, True], [False, False], [False, False]]])
    baseline = masked_future_prediction_losses(
        predicted_state,
        target_state,
        valid,
        predicted_visual,
        target_visual,
        valid,
    )
    predicted_state[:, 2:] = 1e6
    predicted_visual[:, 2:] = -1e6
    perturbed = masked_future_prediction_losses(
        predicted_state,
        target_state,
        valid,
        predicted_visual,
        target_visual,
        valid,
    )

    torch.testing.assert_close(baseline["loss"], perturbed["loss"])


def _evaluation(candidate_id: str, conditioned: bool) -> dict:
    per_task = {}
    for index, task in enumerate(sorted(REQUIRED_TASKS)):
        control_loss = 1.0 + index * 0.1
        normal = control_loss if not conditioned else control_loss - 0.01
        per_task[task] = {
            "normal_composite_future_loss": normal,
            "shuffled_composite_future_loss": normal + (0.0 if not conditioned else 0.2),
            "shuffle_delta": 0.0 if not conditioned else 0.2,
            "shuffle_delta_bootstrap_95": {
                "lower": 0.0 if not conditioned else 0.1,
                "upper": 0.0 if not conditioned else 0.3,
            },
        }
    return {
        "format_version": "wam.robofactory.s2_r3.action_shuffle_evaluation/1",
        "candidate_id": candidate_id,
        "model_kind": (
            "s2_r3_local_action_conditioned"
            if conditioned
            else "s2_r3_local_action_independent"
        ),
        "action_conditioning": conditioned,
        "comparison_contract": {"same": True},
        "per_task": per_task,
        "action_equivalence": {"passed": True},
        "frozen_parent": {"passed": True},
        "checkpoint": f"{candidate_id}.pt",
    }


def test_s2_acceptance_applies_special_five_task_rule():
    accepted = build_acceptance(_evaluation("W0", False), _evaluation("W1", True))
    failed_input = _evaluation("W1", True)
    failed_input["per_task"]["take_photo"][
        "shuffle_delta_bootstrap_95"
    ]["lower"] = -0.01
    rejected = build_acceptance(_evaluation("W0", False), failed_input)

    assert accepted["passed"] is True
    assert accepted["decision"] == "pass_enter_r4"
    assert rejected["passed"] is False
    assert (
        rejected["checks"][
            "w1_action_shuffle_mean_and_ci_lower_positive_on_every_task"
        ]
        is False
    )


def test_model_allowlist_contains_both_s2_candidates():
    assert S2_R3_MODEL_KINDS == {
        "s2_r3_local_action_independent": False,
        "s2_r3_local_action_conditioned": True,
    }


def test_s2_acceptance_rejects_model_outside_allowlist():
    candidate = _evaluation("W1", True)
    candidate["model_kind"] = "unregistered_future_predictor"

    try:
        build_acceptance(_evaluation("W0", False), candidate)
    except ValueError as error:
        assert "allowlist" in str(error)
    else:
        raise AssertionError("acceptance must fail closed for unknown model kinds")


def test_s2_runtime_monitor_reports_program_heartbeat_and_special_gate(
    tmp_path: Path,
):
    run_root = tmp_path / "run"
    initialize_run(
        run_root,
        run_id="fixture",
        session="permanent",
        base_repo=ROOT,
        worktrees=[f"W0={tmp_path / 'w0'}", f"W1={tmp_path / 'w1'}"],
        window_prefix="fixture",
        monitor_window="fixture-monitor",
    )
    update_status(
        run_root,
        candidate="W1",
        phase="training",
        program="train_s2_r3_future_predictor.py",
        detail="five-task joint training",
        gpu_index=1,
        total_updates=10000,
        exit_code=None,
    )
    progress = run_root / "candidates/w1/train/progress.jsonl"
    progress.parent.mkdir(parents=True)
    progress.write_text(
        json.dumps({"update": 500, "updates": 10000, "loss": 0.25}) + "\n"
    )

    rendered = render_monitor(run_root)

    assert "WAM S2-R3 monitor" in rendered
    assert "train_s2_r3_future_predictor.py" in rendered
    assert "500/10000" in rendered
    assert "heartbeat=missing" in rendered
    assert "S2-R3 special acceptance: pending" in rendered
    assert "Permanent tmux stays alive" in rendered


def test_s2_shell_scripts_are_syntax_valid_and_hf_download_is_hardened():
    for name in (
        "prepare_s2_r3_shared.sh",
        "run_s2_r3_candidate.sh",
        "launch_s2_r3_2gpu_tmux.sh",
        "stop_s2_r3_2gpu_tmux.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts" / name)],
            check=True,
            cwd=ROOT,
        )
    prepare = (ROOT / "scripts/prepare_s2_r3_shared.sh").read_text()
    assert "HF_HUB_DISABLE_XET=1" in prepare
    assert 'HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"' in prepare
    assert 'HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"' in prepare
    assert "--max-workers 1" in prepare
    assert "--local-dir" in prepare
    assert "verify_s2_r3_dataset_local.py" in prepare
    assert "snapshot_download" not in prepare
    assert "hf_hub_download" not in prepare
    dino = (ROOT / "scripts/prepare_dinov3_encoder.py").read_text()
    assert "snapshot_download" not in dino
    assert '"--max-workers",' in dino
    assert '"HF_HUB_DISABLE_XET": "1"' in dino


def test_s2_quick_local_dataset_check_rejects_partial_download(tmp_path: Path):
    root = tmp_path / "task"
    (root / "hdf5").mkdir(parents=True)
    (root / "hdf5/episode_000000.hdf5").write_bytes(b"hdf5")
    (root / "normalization.npz").write_bytes(b"normalization")
    (root / "conversion_manifest.json").write_text("{}")
    manifest = {
        "format_version": "wam.multimodal.trajectory.training_manifest/1",
        "dataset_protocol": "generic_multimodal_trajectory",
        "episodes": [
            {
                "task_id": "lift_barrier",
                "hdf5_path": "hdf5/episode_000000.hdf5",
            }
        ],
        "normalization": {"path": "normalization.npz"},
        "source": {"conversion_manifest_path": "conversion_manifest.json"},
    }
    manifest_path = root / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    complete = quick_validate_dataset(
        manifest_path,
        expected_task="lift_barrier",
        expected_episodes=1,
    )
    assert complete["complete"] is True
    assert complete["episodes"] == 1

    manifest["episodes"][0]["hdf5_path"] = "hdf5/episode_000001.hdf5"
    partial_path = root / "partial_manifest.json"
    partial_path.write_text(json.dumps(manifest))
    try:
        quick_validate_dataset(
            partial_path,
            expected_task="lift_barrier",
            expected_episodes=1,
        )
    except ValueError as error:
        assert "missing locally" in str(error)
    else:
        raise AssertionError("partial local dataset must not be accepted")


def test_s2_launcher_dry_run_describes_two_gpus_and_permanent_tmux(tmp_path: Path):
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/launch_s2_r3_2gpu_tmux.sh"),
            "--run-id",
            "fixture",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "S2_R3_RUN_ROOT": str(tmp_path / "run"),
        },
    )

    assert "W0 GPU0" in result.stdout
    assert "W1 GPU1" in result.stdout
    assert "s2/r3-w0-action-independent" in result.stdout
    assert "s2/r3-w1-action-conditioned" in result.stdout
    assert "never kills/exits it" in result.stdout
    assert "600s default read timeout" in result.stdout
    assert "program, status, heartbeat" in result.stdout
    assert not (tmp_path / "run").exists()
