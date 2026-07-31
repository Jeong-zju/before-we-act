from __future__ import annotations

import json
from pathlib import Path
import subprocess

import torch
import yaml

from models.wam_multimodal import (
    LocalFuturePredictorConfig,
    TeamSharedFuturePredictor,
    TeamSharedFuturePredictorConfig,
)
from scripts.accept_s2_r4 import REQUIRED_TASKS, build_acceptance
from scripts.s2_r4_runtime import initialize_run, render_monitor, update_status
from train.s2_model_registry import (
    S2_R4_MODEL_KINDS,
    validate_s2_r4_candidate,
)
from train.s2_r4_future_prediction import (
    masked_peer_future_prediction_losses,
    masked_shared_future_prediction_losses,
    peer_actions_shuffled_by_focal,
)


ROOT = Path(__file__).resolve().parents[1]


def _evaluation(candidate: str, *, team_shared: bool) -> dict:
    per_task = {}
    for index, task in enumerate(sorted(REQUIRED_TASKS)):
        p0_loss = 1.0 + index * 0.1
        if team_shared:
            per_task[task] = {
                "own": {
                    "normal_composite_future_loss": p0_loss - 0.01,
                },
                "peer_shared": {
                    "normal_composite_future_loss": 0.8,
                    "persistence_composite_future_loss": 1.0,
                    "shuffled_composite_future_loss": 1.1,
                    "shuffle_delta": 0.3,
                    "shuffle_delta_bootstrap_95": {
                        "lower": 0.2,
                        "upper": 0.4,
                    },
                },
            }
        else:
            per_task[task] = {
                "normal_composite_future_loss": p0_loss,
            }
    return {
        "format_version": "wam.robofactory.s2_r4.future_scope_evaluation/1",
        "candidate_id": candidate,
        "model_kind": (
            "s2_r4_team_shared_action_conditioned"
            if team_shared
            else "s2_r4_local_action_conditioned"
        ),
        "team_shared": team_shared,
        "action_conditioning": True,
        "comparison_contract": {"same": True},
        "per_task": per_task,
        "action_equivalence": {"passed": True},
        "frozen_parent": {"passed": True},
        "checkpoint": f"{candidate}.pt",
    }


def test_s2_r4_model_allowlist_is_fail_closed():
    assert S2_R4_MODEL_KINDS == {
        "s2_r4_local_action_conditioned": False,
        "s2_r4_team_shared_action_conditioned": True,
    }
    assert validate_s2_r4_candidate(_evaluation("P0", team_shared=False)) == (
        "P0",
        "s2_r4_local_action_conditioned",
        False,
    )
    invalid = _evaluation("P1", team_shared=True)
    invalid["model_kind"] = "unregistered_team_predictor"
    try:
        validate_s2_r4_candidate(invalid)
    except ValueError as error:
        assert "allowlist" in str(error)
    else:
        raise AssertionError("unknown R4 model kinds must fail closed")


def test_s2_r4_acceptance_applies_team_shared_special_rule():
    accepted = build_acceptance(
        _evaluation("P0", team_shared=False),
        _evaluation("P1", team_shared=True),
    )
    failed_input = _evaluation("P1", team_shared=True)
    failed_input["per_task"]["take_photo"]["peer_shared"][
        "shuffle_delta_bootstrap_95"
    ]["lower"] = 0.0
    rejected = build_acceptance(
        _evaluation("P0", team_shared=False),
        failed_input,
    )

    assert accepted["passed"] is True
    assert accepted["decision"] == "pass_enter_s3"
    assert rejected["passed"] is False
    assert (
        rejected["checks"][
            "p1_peer_action_shuffle_mean_and_ci_lower_positive_on_every_task"
        ]
        is False
    )


def _small_team_model() -> TeamSharedFuturePredictor:
    return TeamSharedFuturePredictor(
        LocalFuturePredictorConfig(
            action_horizon=6,
            future_horizons=(1, 2),
            visual_grid_tokens=2,
            visual_latent_dim=8,
            d_model=16,
            ffn_dim=32,
            layers=1,
            heads=4,
            dropout=0.0,
        ),
        TeamSharedFuturePredictorConfig(
            layers=1,
            heads=4,
            ffn_dim=32,
            dropout=0.0,
            own_residual_max=0.1,
        ),
    ).eval()


def test_team_shared_predictor_keeps_local_path_and_exposes_pairwise_slots():
    torch.manual_seed(1)
    model = _small_team_model()
    state = torch.randn(2, 4, 18)
    visual = torch.randn(2, 4, 2, 8)
    shared = torch.randn(2, 2, 8)
    actions = torch.randn(2, 4, 6, 8)
    valid = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    normal = model(state, visual, shared, actions, valid)
    shuffled = model(
        state,
        visual,
        shared,
        actions,
        valid,
        actions_by_focal=peer_actions_shuffled_by_focal(actions, valid),
    )

    assert normal.own_state.shape == (2, 4, 2, 18)
    assert normal.own_visual.shape == (2, 4, 2, 2, 8)
    assert normal.peer_state.shape == (2, 4, 4, 2, 18)
    assert normal.peer_visual.shape == (2, 4, 4, 2, 2, 8)
    assert normal.shared_visual.shape == (2, 4, 2, 2, 8)
    torch.testing.assert_close(
        normal.own_state,
        shuffled.own_state,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(normal.peer_state, shuffled.peer_state)
    assert normal.peer_state[:, 3].count_nonzero().item() == 0
    assert normal.shared_visual[:, 3].count_nonzero().item() == 0


def test_peer_action_shuffle_preserves_each_focal_diagonal():
    actions = torch.arange(2 * 3 * 2 * 1, dtype=torch.float32).reshape(
        2, 3, 2, 1
    )
    valid = torch.ones(2, 3, dtype=torch.bool)
    shuffled = peer_actions_shuffled_by_focal(actions, valid)

    for focal in range(3):
        torch.testing.assert_close(
            shuffled[:, focal, focal],
            actions[:, focal],
        )
        peer = (focal + 1) % 3
        torch.testing.assert_close(
            shuffled[:, focal, peer],
            actions.roll(1, dims=0)[:, peer],
        )


def test_peer_and_shared_losses_mask_self_and_invalid_slots():
    valid = torch.tensor([[True, True, False, False]])
    state_target = torch.zeros(1, 4, 2, 18)
    state_valid = valid[:, :, None].expand(-1, -1, 2)
    visual_target = torch.ones(1, 4, 2, 2, 8)
    visual_valid = state_valid
    predicted_state = state_target[:, None].expand(-1, 4, -1, -1, -1).clone()
    predicted_visual = visual_target[:, None].expand(
        -1, 4, -1, -1, -1, -1
    ).clone()
    peer = masked_peer_future_prediction_losses(
        predicted_state,
        state_target,
        state_valid,
        predicted_visual,
        visual_target,
        visual_valid,
        valid,
    )
    shared = masked_shared_future_prediction_losses(
        torch.ones(1, 4, 2, 2, 8),
        torch.ones(1, 2, 2, 8),
        torch.ones(1, 2, dtype=torch.bool),
        valid,
    )

    torch.testing.assert_close(peer["loss"], torch.tensor(0.0))
    torch.testing.assert_close(shared["loss"], torch.tensor(0.0))


def test_s2_r4_pair_validator_allows_only_team_scope(tmp_path: Path):
    p1 = yaml.safe_load(
        (ROOT / "configs/wam_flow/s2_r4_future.yaml").read_text()
    )
    p0 = json.loads(json.dumps(p1))
    p0["name"] = "wam.robofactory/s2-r4-p0-local-future"
    p0["round"].update(
        {
            "candidate_id": "P0",
            "model_kind": "s2_r4_local_action_conditioned",
            "team_shared": False,
        }
    )
    p0.pop("team_model")
    for key, value in p0["checkpoint"].items():
        p0["checkpoint"][key] = str(value).replace(
            "p1_team_shared", "p0_local"
        )
    p0_path = tmp_path / "p0.yaml"
    p1_path = tmp_path / "p1.yaml"
    p0_path.write_text(yaml.safe_dump(p0, sort_keys=False))
    p1_path.write_text(yaml.safe_dump(p1, sort_keys=False))

    subprocess.run(
        [
            str(ROOT / "scripts/validate_s2_r4_branch_pair.py"),
            "--p0-config",
            str(p0_path),
            "--p1-config",
            str(p1_path),
        ],
        check=True,
        cwd=ROOT,
    )


def test_s2_r4_monitor_reports_program_heartbeat_and_special_gate(
    tmp_path: Path,
):
    run_root = tmp_path / "run"
    initialize_run(
        run_root,
        run_id="fixture",
        session="permanent",
        base_repo=ROOT,
        worktrees=[f"P0={tmp_path / 'p0'}", f"P1={tmp_path / 'p1'}"],
        window_prefix="fixture",
        monitor_window="fixture-monitor",
    )
    update_status(
        run_root,
        candidate="P1",
        phase="training",
        program="train_s2_r4_future_predictor.py",
        detail="team/shared five-task training",
        gpu_index=1,
        total_updates=10000,
        exit_code=None,
    )
    progress = run_root / "candidates/p1/train/progress.jsonl"
    progress.parent.mkdir(parents=True)
    progress.write_text(
        json.dumps({"update": 500, "updates": 10000, "loss": 0.25}) + "\n"
    )
    (run_root / "r3_recovery_progress.jsonl").write_text(
        json.dumps(
            {
                "update": 1200,
                "updates": 10000,
                "loss": 0.125,
                "created_at": "2999-01-01T00:00:00+00:00",
            }
        )
        + "\n"
    )

    rendered = render_monitor(run_root)

    assert "WAM S2-R4 monitor" in rendered
    assert "train_s2_r4_future_predictor.py" in rendered
    assert "500/10000" in rendered
    assert "S2-R3 W1 recovery 1200/10000" in rendered
    assert "heartbeat=missing" in rendered
    assert "S2-R4 special acceptance: pending" in rendered
    assert "Permanent tmux stays alive" in rendered


def test_s2_r4_monitor_renders_every_special_gate(tmp_path: Path):
    run_root = tmp_path / "run"
    initialize_run(
        run_root,
        run_id="fixture",
        session="permanent",
        base_repo=ROOT,
        worktrees=[f"P0={tmp_path / 'p0'}", f"P1={tmp_path / 'p1'}"],
        window_prefix="fixture",
        monitor_window="fixture-monitor",
    )
    acceptance = build_acceptance(
        _evaluation("P0", team_shared=False),
        _evaluation("P1", team_shared=True),
    )
    (run_root / "acceptance.json").write_text(json.dumps(acceptance))

    rendered = render_monitor(run_root)

    assert "S2-R4 special acceptance: PASS -> enter S3" in rendered
    assert "p1_own_target_no_worse_on_every_task=PASS" in rendered
    assert "p1_peer_shared_beats_persistence_on_every_task=PASS" in rendered
    assert (
        "p1_peer_action_shuffle_mean_and_ci_lower_positive_on_every_task=PASS"
        in rendered
    )
    assert "peer/shared" in rendered
    assert "persistence" in rendered


def test_s2_r4_shell_scripts_are_valid_and_reuse_s0_download_path():
    for name in (
        "prepare_s2_r4_shared.sh",
        "run_s2_r4_candidate.sh",
        "launch_s2_r4_2gpu_tmux.sh",
        "stop_s2_r4_2gpu_tmux.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts" / name)],
            check=True,
            cwd=ROOT,
        )
    prepare = (ROOT / "scripts/prepare_s2_r4_shared.sh").read_text()
    assert "prepare_s2_r3_shared.sh" in prepare
    assert "train_s2_r3_future_predictor.py" in prepare
    assert "verify_s2_r3_w1_checkpoint.py" in prepare
    inherited = (ROOT / "scripts/prepare_s2_r3_shared.sh").read_text()
    assert "S0 mode Xet=on workers=8" in inherited
    assert 'HF_HUB_DISABLE_XET=1' in inherited
    assert "--max-workers 1" in inherited


def test_s2_r4_launcher_dry_run_describes_two_5090_slots_and_monitor(
    tmp_path: Path,
):
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/launch_s2_r4_2gpu_tmux.sh"),
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
            "S2_R4_RUN_ROOT": str(tmp_path / "run"),
        },
    )

    assert "P0 GPU0" in result.stdout
    assert "P1 GPU1" in result.stdout
    assert "s2/r4-p0-local" in result.stdout
    assert "s2/r4-p1-team-shared" in result.stdout
    assert "never kills/exits it" in result.stdout
    assert "S0 mode (Xet enabled, default 8 workers" in result.stdout
    assert "current program, status, heartbeat" in result.stdout
    assert not (tmp_path / "run").exists()
