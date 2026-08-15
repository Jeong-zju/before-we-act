from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from before_we_act.b3_n1_data import ACTION_PROBE_HORIZON, FUTURE_OFFSETS
from before_we_act.b3_n1_r1 import R1_SEEDS
from before_we_act.b3_n1_r1_teacher_student import (
    DirectReactiveControl,
    LegalBeliefStudent,
    PrivilegedBeliefTeacher,
)


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = ROOT / "scripts" / "before_we_act" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_fair() -> dict:
    return {
        "status": "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
        "test_opened": False,
        "training_status": {
            str(seed): {
                "selected_validation": {
                    "macro": {
                        "h": 1.0,
                        "h_b": 0.6,
                        "h_b_shuffle": 0.9,
                        "h_matched_capacity": 0.95,
                    }
                }
            }
            for seed in R1_SEEDS
        },
    }


def test_owner_unlock_preserves_inconclusive_status() -> None:
    module = load_script("prepare_b3_n1_r1_teacher.py")
    receipt = module.r1_1_strong_validation_trend(fake_fair())
    assert receipt["formal_status_preserved"] == "INCONCLUSIVE_TRAINING_NOT_CONVERGED"
    assert receipt["test_opened"] is False
    assert all(
        abs(row["relative_improvement_vs_h"] - 0.4) < 1e-9
        for row in receipt["per_seed"].values()
    )


def test_teacher_has_no_counterfactual_pilot_outputs() -> None:
    batch = 2
    d_model = 32
    model = PrivilegedBeliefTeacher(d_model=d_model)
    base = torch.randn(batch, ACTION_PROBE_HORIZON, 8)
    h = torch.randn(batch, d_model)
    oracle = {
        "teammate_qpos": torch.randn(batch, 9),
        "previous_teammate_qpos": torch.randn(batch, 9),
        "teammate_delta": torch.randn(batch, len(FUTURE_OFFSETS), 9),
        "future_mask": torch.ones(batch, len(FUTURE_OFFSETS), dtype=torch.bool),
        "oracle_teammate_action": torch.randn(batch, ACTION_PROBE_HORIZON, 8),
        "oracle_teammate_action_mask": torch.ones(
            batch, ACTION_PROBE_HORIZON, dtype=torch.bool
        ),
    }
    output = model(base, h, oracle)
    assert not hasattr(output, "branch_value")
    assert not hasattr(output, "shared_change")
    assert torch.equal(output.action, base)


def test_student_and_direct_share_action_path_capacity() -> None:
    d_model = 32
    student = LegalBeliefStudent(d_model=d_model)
    direct = DirectReactiveControl(d_model=d_model)
    student_action_path = (
        student.queries.numel()
        + sum(value.numel() for value in student.reader.parameters())
        + sum(value.numel() for value in student.token_norm.parameters())
        + sum(value.numel() for value in student.action_residual.parameters())
    )
    assert student_action_path == sum(value.numel() for value in direct.parameters())


def test_pipeline_does_not_feed_pilot_into_teacher_or_student() -> None:
    pipeline = (
        ROOT
        / "scripts"
        / "before_we_act"
        / "run_b3_n1_r1_owner_revision_pipeline.sh"
    ).read_text(encoding="utf-8")
    teacher_command = pipeline.split("-m before_we_act.train_b3_n1_r1_teacher", 1)[1].split(
        "pids+=(", 1
    )[0]
    assert "--pilot" not in teacher_command
    teacher_source = (
        ROOT / "before_we_act" / "train_b3_n1_r1_teacher.py"
    ).read_text(encoding="utf-8")
    assert "PilotBranchBatcher" not in teacher_source
    assert "branch_value_target" not in teacher_source


def test_student_continuation_preserves_teacher_inconclusive_and_sealed_test() -> None:
    module = load_script("prepare_b3_n1_r1_student_continuation.py")
    training_status = {}
    for seed in R1_SEEDS:
        training_status[str(seed)] = {
            "status": "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
            "selected_update": 80_000,
            "selected_validation": {
                "macro": {
                    "h": 1.0,
                    "h_teacher": 0.6,
                    "h_teacher_shuffle": 1.1,
                    "h_matched_capacity": 0.9,
                },
                "per_task": {
                    "h": {str(index): 1.0 for index in range(6)},
                    "h_teacher": {str(index): 0.6 for index in range(6)},
                },
            },
        }
    gate = module.validation_gate({"training_status": training_status})
    assert gate["passed_for_exploratory_student_validation"] is True
    assert gate["positive_tasks"] == 6
    assert all(
        row["training_status"] == "INCONCLUSIVE_TRAINING_NOT_CONVERGED"
        for row in gate["per_seed"].values()
    )


def test_student_analyzer_keeps_validation_only_contract_sealed() -> None:
    source = (
        ROOT / "scripts" / "before_we_act" / "analyze_b3_n1_r1_student.py"
    ).read_text(encoding="utf-8")
    assert 'split_names = ("validation", "test") if sealed_test_allowed' in source
    assert '"test_opened": sealed_test_allowed' in source


def test_student_validation_diagnostic_separates_signal_from_attribution() -> None:
    module = load_script("summarize_b3_n1_r1_student_validation.py")
    statuses = {}
    for seed_index, seed in enumerate(R1_SEEDS):
        student = 0.8
        direct = 0.7 if seed_index == 1 else 0.9
        statuses[str(seed)] = {
            "status": "INCONCLUSIVE_TRAINING_NOT_CONVERGED",
            "selected_update": 80_000,
            "selected_validation": {
                "belief_off_max_abs": 0.0,
                "macro": {
                    "h": 1.0,
                    "belief_off": 1.0,
                    "h_student": student,
                    "h_student_shuffle": 1.1,
                    "direct_reactive": direct,
                },
                "per_task": {
                    "h": {str(index): 1.0 for index in range(6)},
                    "h_student": {str(index): student for index in range(6)},
                },
            },
        }
    trend = module.summarize(statuses)
    assert trend["student_beats_h_all_seeds"] is True
    assert trend["student_beats_shuffle_all_seeds"] is True
    assert trend["positive_tasks"] == 6
    assert trend["student_beats_direct_seed_count"] == 2
    assert trend["student_beats_direct_all_seeds"] is False
    assert trend["all_training_platform_reached"] is False


def test_preserved_real_fair_receipt_has_strong_three_seed_trend() -> None:
    path = (
        ROOT
        / "docs"
        / "experiments"
        / "n1_r1"
        / "20260815"
        / "r1_1_fair_probe"
        / "conclusion.json"
    )
    if not path.is_file():
        return
    module = load_script("prepare_b3_n1_r1_teacher.py")
    receipt = module.r1_1_strong_validation_trend(json.loads(path.read_text()))
    improvements = [
        row["relative_improvement_vs_h"] for row in receipt["per_seed"].values()
    ]
    assert min(improvements) > 0.38
    assert max(improvements) < 0.42
