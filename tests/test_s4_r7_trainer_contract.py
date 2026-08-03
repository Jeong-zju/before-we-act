from __future__ import annotations

import argparse
import json
import math

import torch

from scripts.train_s4_r7_world_utility import (
    RESUME_FORMAT,
    _name_hash,
    _resume_payload,
    _set_learning_rates,
    _write_terminal_oom_preflight,
)
from train.s4_hierarchical_team_sampler import S4ExposureCounter


def test_flow_schedule_is_exactly_frozen_then_has_its_own_warmup() -> None:
    flow = torch.nn.Parameter(torch.ones(()), requires_grad=False)
    router = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [
            {"name": "flow", "params": [flow], "lr": 0.0, "base_lr": 2e-5},
            {
                "name": "router",
                "params": [router],
                "lr": 0.0,
                "base_lr": 3e-4,
            },
        ]
    )

    _set_learning_rates(
        optimizer,
        update=6_399,
        total_updates=30_000,
        flow_unfreeze_update=6_400,
        warmup_updates=500,
        flow_warmup_updates=500,
    )
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert optimizer.param_groups[1]["lr"] > 0.0

    _set_learning_rates(
        optimizer,
        update=6_400,
        total_updates=30_000,
        flow_unfreeze_update=6_400,
        warmup_updates=500,
        flow_warmup_updates=500,
    )
    assert math.isclose(optimizer.param_groups[0]["lr"], 2e-5 / 500)


def test_resume_binds_gradient_and_per_module_exposure_state() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    exposure = S4ExposureCounter(team_windows_seen=120, valid_agent_windows_seen=384)
    audit = {
        "format_version": "wam.robofactory.s4_r7.gradient_audit/1",
        "candidate_id": "P1",
        "wuc_only": {"passed": True},
    }
    by_module = {
        "flow": {"team_windows_seen": 0, "valid_agent_windows_seen": 0},
        "router": {"team_windows_seen": 120, "valid_agent_windows_seen": 384},
    }

    payload = _resume_payload(
        {"seed": 707},
        10,
        model,
        optimizer,
        exposure,
        bytes.fromhex("11" * 32),
        gradient_audit=audit,
        normal_categories_seen={"flow": False, "router": True},
        flow_frozen_gradient_exact_zero=True,
        flow_frozen_observed=True,
        flow_unfrozen_gradient_nonzero=False,
        flow_unfrozen_observed=False,
        module_exposure=by_module,
    )

    assert payload["format_version"] == RESUME_FORMAT
    assert payload["sampler_resume_key"] == [707, 11, 0, 0]
    assert payload["training_audit_state"]["gradient_audit"] == audit
    assert payload["module_exposure_state"] == by_module
    assert payload["dataset_chain_sha256"] == "11" * 32


def test_trainable_name_hash_is_order_independent() -> None:
    assert _name_hash(["b.weight", "a.weight"]) == _name_hash(
        ["a.weight", "b.weight"]
    )


def test_early_cuda_oom_writes_minimal_paired_fallback_report(
    tmp_path, monkeypatch
) -> None:
    config = tmp_path / "s4_r7.yaml"
    config.write_text(
        """\
round:
  round_id: s4-r7
  candidate_id: P0
  model_kind: s4_r7_token_preserving
vision:
  inference_batch_size: 16
training:
  budget_mode: fast_selection_30k
  updates: 30000
  flow_unfreeze_update: 6400
  agent_window_budget: 1152000
  micro_team_batch: 4
  gradient_accumulation: 3
  effective_team_batch: 12
  utility_coupling_weight: 0.0
  relation_weight: 0.0
  specialization_weight: 0.0
  anchor_weight: 0.0
""",
        encoding="utf-8",
    )
    report = tmp_path / "preflight.json"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 0)

    class _Properties:
        total_memory = 32 * 1024**3

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: _Properties())
    _write_terminal_oom_preflight(
        argparse.Namespace(
            config=config,
            device="cuda:0",
            preflight_report=report,
        )
    )

    value = json.loads(report.read_text(encoding="utf-8"))
    assert value["completed"] is False
    assert value["oom"] is True
    assert value["peak_memory_bytes"] == 0
    assert value["gpu_total_memory_bytes"] == 32 * 1024**3
    assert value["failure_scope"] == "whole_preflight_gpu_lifecycle"
