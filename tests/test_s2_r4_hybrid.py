from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import torch

from models.wam_multimodal import (
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
    ProtectedHybridFuturePredictor,
    TeamSharedFuturePredictor,
    TeamSharedFuturePredictorConfig,
    exact_own_difference,
)
from scripts.evaluate_s2_r4_hybrid_checkpoint import (
    REQUIRED_TASKS,
    build_diagnostic,
)
from scripts.s2_r4_hybrid_runtime import initialize_run, render_monitor
from train.s2_model_registry import (
    S2_R4_HYBRID_MODEL_KINDS,
    require_trainable_s2_r4_model_kind,
    validate_s2_r4_hybrid_diagnostic,
)


ROOT = Path(__file__).resolve().parents[1]


def _configs() -> tuple[LocalFuturePredictorConfig, TeamSharedFuturePredictorConfig]:
    return (
        LocalFuturePredictorConfig(
            max_agents=2,
            state_dim=3,
            action_dim=2,
            action_horizon=4,
            future_horizons=(1, 4),
            visual_grid_tokens=1,
            visual_latent_dim=4,
            d_model=8,
            ffn_dim=16,
            layers=1,
            heads=2,
            dropout=0.0,
        ),
        TeamSharedFuturePredictorConfig(
            layers=1,
            heads=2,
            ffn_dim=16,
            dropout=0.0,
            own_residual_max=0.1,
        ),
    )


def test_hybrid_kind_is_evaluate_only_and_fail_closed() -> None:
    assert S2_R4_HYBRID_MODEL_KINDS == {
        "s2_r4_protected_hybrid_diagnostic"
    }
    value = {
        "model_kind": "s2_r4_protected_hybrid_diagnostic",
        "mode": "evaluate_only",
        "training_allowed": False,
    }
    assert validate_s2_r4_hybrid_diagnostic(value) == value["model_kind"]
    with pytest.raises(ValueError, match="evaluate-only"):
        require_trainable_s2_r4_model_kind(value["model_kind"])
    with pytest.raises(ValueError, match="allowlist"):
        validate_s2_r4_hybrid_diagnostic(
            {**value, "model_kind": "unregistered_hybrid"}
        )


def test_hybrid_replaces_p1_local_and_returns_bit_exact_p0_own() -> None:
    local_config, team_config = _configs()
    torch.manual_seed(11)
    p0 = LocalActionConditionedFuturePredictor(local_config)
    p0.eval()
    torch.manual_seed(22)
    p1 = TeamSharedFuturePredictor(local_config, team_config)
    with torch.no_grad():
        p1.own_residual_gate.fill_(4.0)
    p1_team_before = {
        key: value.clone()
        for key, value in p1.state_dict().items()
        if key.startswith(("team_encoder.", "peer_", "shared_"))
    }
    hybrid = ProtectedHybridFuturePredictor(local_config, team_config)
    hybrid.load_sources(
        own_state_dict=p0.state_dict(), team_state_dict=p1.state_dict()
    )
    assert not any(parameter.requires_grad for parameter in hybrid.parameters())
    with pytest.raises(RuntimeError, match="evaluate-only"):
        hybrid.train()
    for key, expected in p0.state_dict().items():
        assert torch.equal(
            hybrid.team_source.local_predictor.state_dict()[key], expected
        )
    for key, expected in p1_team_before.items():
        assert torch.equal(hybrid.team_source.state_dict()[key], expected)

    current_state = torch.randn(2, 2, 3)
    current_visual = torch.randn(2, 2, 1, 4)
    current_shared = torch.randn(2, 1, 4)
    actions = torch.randn(2, 2, 4, 2)
    valid = torch.ones(2, 2, dtype=torch.bool)
    with torch.inference_mode():
        reference = p0(
            current_state, current_visual, actions, valid, valid
        )
        observed = hybrid(
            current_state,
            current_visual,
            current_shared,
            actions,
            valid,
        )
    difference = exact_own_difference(reference, observed)
    assert difference["max_abs_diff"] == 0.0
    assert difference["state_elementwise_exact"] is True
    assert difference["visual_elementwise_exact"] is True


def _task_metrics(*, exact: bool = True, team: bool = True) -> dict:
    return {
        "protected_own": {
            "state_elementwise_exact": exact,
            "visual_elementwise_exact": exact,
            "loss_elementwise_exact": exact,
            "max_abs_diff": 0.0 if exact else 0.1,
            "loss_difference": 0.0 if exact else 0.1,
        },
        "peer_shared": {
            "normal_composite_future_loss": 1.0 if team else 2.0,
            "persistence_composite_future_loss": 1.5,
            "shuffle_delta": 0.2 if team else -0.1,
            "shuffle_delta_bootstrap_95": {
                "lower": 0.1 if team else -0.2,
                "upper": 0.3,
            },
        },
    }


def test_hybrid_special_acceptance_distinguishes_r5_and_wiring_failures() -> None:
    tasks = {task: _task_metrics() for task in REQUIRED_TASKS}
    passed = build_diagnostic(
        tasks, action_equivalence={"passed": True}, sources_stable=True
    )
    assert passed["passed"] is True
    assert passed["conclusion"] == "pass_existing_team_compatible"

    team_failed = dict(tasks)
    team_failed["lift_barrier"] = _task_metrics(team=False)
    failed = build_diagnostic(
        team_failed, action_equivalence={"passed": True}, sources_stable=True
    )
    assert failed["passed"] is False
    assert failed["next_action"] == "enter_s2_r5_retrain_team_from_protected_p0"

    own_failed = dict(tasks)
    own_failed["lift_barrier"] = _task_metrics(exact=False)
    stopped = build_diagnostic(
        own_failed, action_equivalence={"passed": True}, sources_stable=True
    )
    assert stopped["next_action"] == "stop_before_r5_fix_hybrid"


def test_hybrid_monitor_shows_program_heartbeat_progress_and_special_gate(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    own = tmp_path / "p0.pt"
    team = tmp_path / "p1.pt"
    own.touch()
    team.touch()
    run_root = tmp_path / "run"
    initialize_run(
        run_root,
        run_id="hybrid-test",
        session="permanent",
        window_prefix="hybrid-test",
        monitor_window="hybrid-test-monitor",
        repo=repo,
        own_source=own,
        team_source=team,
    )
    (run_root / "evaluation_progress.jsonl").write_text(
        json.dumps(
            {
                "program": "evaluate_s2_r4_hybrid_checkpoint.py",
                "task_id": "lift_barrier",
                "batch": 2,
                "batches": 5,
                "completed_batches": 7,
                "total_batches": 25,
                "completed_fraction": 0.28,
                "windows": 8,
                "own_max_abs_diff": 0.0,
                "peer_shared_loss": 1.0,
                "persistence_loss": 1.5,
                "peer_shuffle_delta": 0.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rendered = render_monitor(run_root)
    assert "program=s2_r4_hybrid_runtime.py" in rendered
    assert "heartbeat=STALE" in rendered
    assert "progress=7/25 (28.0%)" in rendered
    assert "task=lift_barrier" in rendered
    assert "own max_abs_diff=0.000000" in rendered
    assert "peer-shuffle delta=0.200000" in rendered


def test_hybrid_shell_scripts_preserve_tmux_and_forbid_strict_mode() -> None:
    scripts = [
        "prepare_s2_r4_hybrid.sh",
        "run_s2_r4_hybrid_evaluation.sh",
        "launch_s2_r4_hybrid_tmux.sh",
        "stop_s2_r4_hybrid_tmux.sh",
    ]
    for name in scripts:
        path = ROOT / "scripts" / name
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" not in text
        assert "set -Eeuo pipefail" not in text
    launcher = (ROOT / "scripts/launch_s2_r4_hybrid_tmux.sh").read_text()
    stopper = (ROOT / "scripts/stop_s2_r4_hybrid_tmux.sh").read_text()
    assert "CUDA_VISIBLE_DEVICES=0" in launcher
    assert "remain-on-exit" in launcher
    assert "kill-session" not in stopper
    assert "shared data/cache/artifacts" in stopper
