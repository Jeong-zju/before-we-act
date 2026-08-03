from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.accept_s4_r7 import TASKS
from scripts.evaluate_s4_r7_causal import (
    EVALUATION_ORDER,
    EXPOSURE_FORMAT,
    FORCED_FORMAT,
    GATE_FORMAT,
    GRADIENT_FORMAT,
    RESUME_FORMAT,
    UTILITY_FORMAT,
    _candidate_condition,
    _condition_macro,
    _reuse_offline_audit,
    _spearman,
    _validate_gate_summary,
    _validate_training_audits,
)


def test_spearman_handles_order_reversal_and_ties() -> None:
    assert _spearman(np.asarray([1.0, 2.0, 3.0]), np.asarray([2.0, 4.0, 8.0])) == pytest.approx(1.0)
    assert _spearman(np.asarray([1.0, 2.0, 3.0]), np.asarray([8.0, 4.0, 2.0])) == pytest.approx(-1.0)
    assert _spearman(np.asarray([1.0, 1.0, 1.0]), np.asarray([3.0, 2.0, 1.0])) == 0.0


def test_gate20_identity_binds_condition_checkpoint_and_exact_seeds(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "policy.pt"
    config.write_text("name: fixture\n", encoding="utf-8")
    checkpoint.write_bytes(b"fixture")
    episodes = [
        {"seed": seed, "success": seed % 2 == 0}
        for seed in range(900, 920)
    ]
    value = {
        "format_version": GATE_FORMAT,
        "mode": "gate",
        "task_order": list(TASKS),
        "seed_protocol": {"seed_start": 900, "episodes_per_task": 20},
        "candidate": {
            "policy_kind": "s4_flow",
            "config": str(config),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": "a" * 64,
            "client": {
                "policy": {
                    "world_intervention": "normal",
                    "model_kind": "s4_r7_token_preserving",
                    "action_source": "s4_r7_token_preserving_world_flow",
                }
            },
        },
        **{
            task: {
                "episodes": episodes,
                "success_rate": 0.5,
                "successes": 10,
            }
            for task in TASKS
        },
    }
    observed = _validate_gate_summary(
        value,
        condition="normal",
        config_path=config.resolve(),
        checkpoint_path=checkpoint.resolve(),
        checkpoint_sha256="a" * 64,
        candidate_id="P0",
        model_kind="s4_r7_token_preserving",
    )
    assert observed is value
    condition = _candidate_condition(value)
    assert _condition_macro(condition) == 0.5

    value["candidate"]["client"]["policy"]["world_intervention"] = "shuffle_all"
    with pytest.raises(ValueError, match="identity differs"):
        _validate_gate_summary(
            value,
            condition="normal",
            config_path=config.resolve(),
            checkpoint_path=checkpoint.resolve(),
            checkpoint_sha256="a" * 64,
            candidate_id="P0",
            model_kind="s4_r7_token_preserving",
        )


def test_offline_audit_reuse_is_checkpoint_hash_bound(tmp_path: Path) -> None:
    forced = tmp_path / "forced.npz"
    utility = tmp_path / "utility.json"
    np.savez_compressed(
        forced,
        format_version=np.asarray(FORCED_FORMAT),
        checkpoint_sha256=np.asarray("b" * 64),
        candidate_id=np.asarray("P1"),
    )
    import hashlib

    digest = hashlib.sha256(forced.read_bytes()).hexdigest()
    utility.write_text(
        json.dumps(
            {
                "format_version": UTILITY_FORMAT,
                "candidate_id": "P1",
                "checkpoint_sha256": "b" * 64,
                "heldout_flow_error": 0.2,
                "forced_evidence_errors": {"sha256": digest},
            }
        ),
        encoding="utf-8",
    )
    assert (
        _reuse_offline_audit(
            forced,
            utility,
            checkpoint_sha256="b" * 64,
            candidate_id="P1",
        )
        is not None
    )
    assert (
        _reuse_offline_audit(
            forced,
            utility,
            checkpoint_sha256="c" * 64,
            candidate_id="P1",
        )
        is None
    )


def test_training_audits_require_formal_budget_and_recoverable_resume(
    tmp_path: Path,
) -> None:
    resume = tmp_path / "resume.pt"
    torch.save(
        {
            "format_version": RESUME_FORMAT,
            "update": 29_000,
            "identity": {"candidate_id": "P0"},
        },
        resume,
    )
    modules = {
        name: {"team_windows_seen": 1, "valid_agent_windows_seen": 3}
        for name in (
            "flow",
            "future_body",
            "future_heads",
            "legacy_adapter",
            "evidence",
            "router",
        )
    }
    gradient = {
        "format_version": GRADIENT_FORMAT,
        "candidate_id": "P0",
        "passed": True,
    }
    exposure = {
        "format_version": EXPOSURE_FORMAT,
        "candidate_id": "P0",
        "passed": True,
        "formal_budget_complete": True,
        "team_windows_seen": 360_000,
        "agent_windows_seen_by_module": modules,
    }
    _validate_training_audits(
        gradient,
        exposure,
        {"agent_windows_seen_by_module": modules},
        resume_path=resume,
        candidate_id="P0",
        total_updates=30_000,
        effective_team_batch=12,
    )
    exposure["formal_budget_complete"] = False
    with pytest.raises(ValueError, match="exposure audit"):
        _validate_training_audits(
            gradient,
            exposure,
            {"agent_windows_seen_by_module": modules},
            resume_path=resume,
            candidate_id="P0",
            total_updates=30_000,
            effective_team_batch=12,
        )


def test_normal_precedes_every_shuffle_for_donor_bank_creation() -> None:
    assert EVALUATION_ORDER[0] == "normal"
    assert EVALUATION_ORDER[:4] == (
        "normal",
        "legacy_reference",
        "world_evidence_gate_zero",
        "shuffle_all",
    )
    assert set(EVALUATION_ORDER) == {
        "normal",
        "legacy_reference",
        "world_evidence_gate_zero",
        "all_world_gates_zero",
        "shuffle_all",
        "shuffle_own",
        "shuffle_peer",
        "shuffle_shared",
    }
