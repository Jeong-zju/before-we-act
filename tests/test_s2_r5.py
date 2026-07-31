from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import pytest
import torch
import yaml

from models.wam_multimodal import (
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
    ProtectedTeamFuturePredictor,
    ProtectedTeamFuturePredictorConfig,
)
from scripts.accept_s2_r5 import EVALUATION_FORMAT, REQUIRED_TASKS, build_acceptance
from scripts.s2_r5_runtime import initialize_run, render_monitor, update_status
from train.s2_model_registry import validate_s2_r5_candidate


ROOT = Path(__file__).resolve().parents[1]


def _round(candidate: str) -> dict[str, object]:
    mixer = "shared" if candidate == "P0" else "role_mot"
    kind = (
        "s2_r5_protected_shared_team"
        if candidate == "P0"
        else "s2_r5_protected_role_mot_team"
    )
    return {
        "candidate_id": candidate,
        "model_kind": kind,
        "team_mixer": mixer,
        "protected_own": True,
    }


def test_s2_r5_model_allowlist_is_fail_closed() -> None:
    assert validate_s2_r5_candidate(_round("P0")) == (
        "P0",
        "s2_r5_protected_shared_team",
        "shared",
    )
    assert validate_s2_r5_candidate(_round("P1"))[2] == "role_mot"
    invalid = _round("P0")
    invalid["model_kind"] = "s2_r5_unregistered"
    with pytest.raises(ValueError, match="allowlist"):
        validate_s2_r5_candidate(invalid)


@pytest.mark.parametrize("mixer", ["shared", "role_mot"])
def test_protected_team_model_keeps_own_exact_and_optimizer_excluded(
    mixer: str,
) -> None:
    local_config = LocalFuturePredictorConfig(
        max_agents=3,
        state_dim=5,
        action_dim=2,
        action_horizon=4,
        future_horizons=(1, 4),
        visual_grid_tokens=2,
        visual_latent_dim=6,
        d_model=12,
        ffn_dim=24,
        layers=1,
        heads=3,
        dropout=0.0,
    )
    source = LocalActionConditionedFuturePredictor(local_config).eval()
    model = ProtectedTeamFuturePredictor(
        local_config,
        ProtectedTeamFuturePredictorConfig(
            layers=1,
            heads=3,
            ffn_dim=24,
            dropout=0.0,
            team_mixer=mixer,
        ),
    )
    model.load_protected_own(source.state_dict())
    model.train()
    assert not model.protected_own.training
    assert all(not parameter.requires_grad for parameter in model.protected_own.parameters())
    assert all(
        not name.startswith("protected_own.")
        for name in model.parameter_audit()["trainable_names"]
    )
    assert all(not key.startswith("protected_own.") for key in model.team_state_dict())
    batch = 2
    state = torch.randn(batch, 3, 5)
    visual = torch.randn(batch, 3, 2, 6)
    shared = torch.randn(batch, 2, 6)
    actions = torch.randn(batch, 3, 4, 2)
    valid = torch.ones(batch, 3, dtype=torch.bool)
    reference = source(state, visual, actions, valid, valid)
    prediction = model(state, visual, shared, actions, valid)
    assert torch.equal(reference[0], prediction.own_state)
    assert torch.equal(reference[1], prediction.own_visual)
    loss = prediction.peer_state.square().mean() + prediction.shared_visual.square().mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in model.protected_own.parameters())


def _evaluation(candidate: str, *, macro_offset: float = 0.0) -> dict[str, object]:
    value: dict[str, object] = {
        "format_version": EVALUATION_FORMAT,
        **_round(candidate),
        "checkpoint": f"{candidate}.pt",
        "comparison_contract": {"selection": "fixed"},
        "protected_own_evidence": {
            "elementwise_exact": True,
            "maximum_absolute_difference": 0.0,
            "checkpoint_stable": True,
            "optimizer_excluded": True,
        },
        "action_equivalence": {"passed": True},
        "frozen_parent": {"passed": True},
        "per_task": {},
    }
    for index, task in enumerate(sorted(REQUIRED_TASKS)):
        value["per_task"][task] = {  # type: ignore[index]
            "peer_shared": {
                "normal_composite_future_loss": 1.0 + macro_offset + index / 100,
                "persistence_composite_future_loss": 2.0,
                "shuffle_delta": 0.1,
                "shuffle_delta_bootstrap_95": {"lower": 0.01},
            }
        }
    return value


def test_s2_r5_acceptance_selects_lower_macro_and_tie_prefers_p0() -> None:
    result = build_acceptance(_evaluation("P0"), _evaluation("P1", macro_offset=-0.1))
    assert result["passed"] is True
    assert result["winner"] == "P1"
    tied = build_acceptance(_evaluation("P0"), _evaluation("P1"))
    assert tied["winner"] == "P0"
    failed = _evaluation("P1")
    failed["per_task"]["lift_barrier"]["peer_shared"][  # type: ignore[index]
        "shuffle_delta_bootstrap_95"
    ]["lower"] = -0.001
    partial = build_acceptance(_evaluation("P0"), failed)
    assert partial["winner"] == "P0"
    assert partial["candidates"]["P1"]["passed"] is False


def test_s2_r5_pair_validator_allows_only_mixer_identity(tmp_path: Path) -> None:
    p0 = {
        "name": "s2-r5-p0",
        "round": {"round_id": "s2-r5", **_round("P0")},
        "team_model": {"team_mixer": "shared", "layers": 2},
        "training": {"seed": 505, "updates": 10000},
        "data": {"split": "fixed"},
        "evaluation": {"windows_per_episode": 4},
        "checkpoint": {
            "output": "p0/predictor.pt",
            "resume": "p0/resume.pt",
        },
    }
    p1 = copy.deepcopy(p0)
    p1["name"] = "wam.robofactory/s2-r5-p1-protected-role-mot-team"
    p1["round"].update(
        {
            "candidate_id": "P1",
            "model_kind": "s2_r5_protected_role_mot_team",
            "team_mixer": "role_mot",
        }
    )
    p1["team_model"]["team_mixer"] = "role_mot"
    for key in p1["checkpoint"]:
        p1["checkpoint"][key] = p1["checkpoint"][key].replace("p0", "p1")
    p0_path = tmp_path / "p0.yaml"
    p1_path = tmp_path / "p1.yaml"
    p0_path.write_text(yaml.safe_dump(p0))
    p1_path.write_text(yaml.safe_dump(p1))
    command = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/validate_s2_r5_branch_pair.py"),
        "--p0-config",
        str(p0_path),
        "--p1-config",
        str(p1_path),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    p1["training"]["seed"] += 1
    p1_path.write_text(yaml.safe_dump(p1))
    assert subprocess.run(command, cwd=ROOT).returncode != 0


def test_s2_r5_monitor_reports_heartbeat_program_progress_and_gates(
    tmp_path: Path,
) -> None:
    initialize_run(
        tmp_path,
        run_id="r5-test",
        session="ssh_tmux",
        base_repo=ROOT,
        worktrees=["P0=/tmp/p0", "P1=/tmp/p1"],
        window_prefix="r5-test",
        monitor_window="r5-test-monitor",
    )
    update_status(
        tmp_path,
        candidate="P0",
        phase="training",
        program="train_s2_r5_protected_team.py",
        detail="team only",
        gpu_index=0,
        total_updates=10000,
        exit_code=None,
    )
    rendered = render_monitor(tmp_path)
    assert "S2-R5 monitor" in rendered
    assert "train_s2_r5_protected_team.py" in rendered
    assert "HEARTBEAT" in rendered
    assert "special acceptance: pending" in rendered
    update_status(
        tmp_path,
        candidate="P0",
        phase="complete",
        program="run_s2_r5_candidate.sh",
        detail="done",
        gpu_index=0,
        total_updates=10000,
        exit_code=0,
    )
    assert "finished" in render_monitor(tmp_path)


def test_s2_r5_shell_contracts_and_s0_download_delegation() -> None:
    scripts = [
        "prepare_s2_r5_shared.sh",
        "run_s2_r5_candidate.sh",
        "launch_s2_r5_2gpu_tmux.sh",
        "stop_s2_r5_2gpu_tmux.sh",
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], check=True)
    launcher = (ROOT / "scripts/launch_s2_r5_2gpu_tmux.sh").read_text()
    prepare = (ROOT / "scripts/prepare_s2_r5_shared.sh").read_text()
    stopper = (ROOT / "scripts/stop_s2_r5_2gpu_tmux.sh").read_text()
    assert "remain-on-exit on" in launcher
    assert "s0_hf_token_fifo.sh" in launcher
    assert "prepare_s2_r4_shared.sh" in prepare
    assert "S2_R5_USE_S0_PREP" in prepare
    assert "tmux kill-session" not in stopper
    assert "datasets" in stopper and "Preserved" in stopper
