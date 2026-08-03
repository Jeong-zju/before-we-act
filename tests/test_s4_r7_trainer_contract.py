from __future__ import annotations

import argparse
import json
import math

import pytest
import torch

from scripts.train_s2_r4_future_predictor import _s4_dataset_reuse_environment
from scripts.train_s4_r7_world_utility import (
    RESUME_FORMAT,
    _name_hash,
    _resume_payload,
    _set_learning_rates,
    _write_terminal_oom_preflight,
)
from train.s4_hierarchical_team_sampler import S4ExposureCounter


@pytest.mark.parametrize(
    ("round_id", "prefix"), (("s4-r7", "S4_R7"), ("s4-r8", "S4_R8"))
)
def test_shared_dataset_reuse_uses_the_current_round_namespace(
    monkeypatch: pytest.MonkeyPatch, round_id: str, prefix: str
) -> None:
    for candidate_prefix in ("S4_R7", "S4_R8"):
        for suffix in (
            "SHARED_HDF5_RECEIPT",
            "SHARED_HDF5_RECEIPT_SHA256",
            "FUTURE_FEATURE_CACHE",
            "FUTURE_FEATURE_CACHE_SHA256",
        ):
            monkeypatch.delenv(f"{candidate_prefix}_{suffix}", raising=False)
    monkeypatch.setenv(f"{prefix}_SHARED_HDF5_RECEIPT", "/verified/receipt.json")
    monkeypatch.setenv(f"{prefix}_SHARED_HDF5_RECEIPT_SHA256", "a" * 64)
    monkeypatch.setenv(f"{prefix}_FUTURE_FEATURE_CACHE", "/verified/cache")
    monkeypatch.setenv(f"{prefix}_FUTURE_FEATURE_CACHE_SHA256", "b" * 64)

    assert _s4_dataset_reuse_environment({"round": {"round_id": round_id}}) == (
        "/verified/receipt.json",
        "a" * 64,
        True,
    )


def test_r8_dataset_reuse_rejects_r7_namespace_instead_of_rescanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for suffix, value in (
        ("SHARED_HDF5_RECEIPT", "/wrong/receipt.json"),
        ("SHARED_HDF5_RECEIPT_SHA256", "a" * 64),
        ("FUTURE_FEATURE_CACHE", "/wrong/cache"),
        ("FUTURE_FEATURE_CACHE_SHA256", "b" * 64),
    ):
        monkeypatch.setenv(f"S4_R7_{suffix}", value)
        monkeypatch.delenv(f"S4_R8_{suffix}", raising=False)

    with pytest.raises(ValueError, match="other S4 round namespace"):
        _s4_dataset_reuse_environment({"round": {"round_id": "s4-r8"}})


def test_r8_dataset_reuse_rejects_an_incomplete_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("S4_R8_SHARED_HDF5_RECEIPT", "/verified/receipt.json")
    monkeypatch.setenv("S4_R8_SHARED_HDF5_RECEIPT_SHA256", "a" * 64)
    monkeypatch.setenv("S4_R8_FUTURE_FEATURE_CACHE", "/verified/cache")
    monkeypatch.delenv("S4_R8_FUTURE_FEATURE_CACHE_SHA256", raising=False)
    for suffix in (
        "SHARED_HDF5_RECEIPT",
        "SHARED_HDF5_RECEIPT_SHA256",
        "FUTURE_FEATURE_CACHE",
        "FUTURE_FEATURE_CACHE_SHA256",
    ):
        monkeypatch.delenv(f"S4_R7_{suffix}", raising=False)

    with pytest.raises(ValueError, match="FUTURE_FEATURE_CACHE_SHA256"):
        _s4_dataset_reuse_environment({"round": {"round_id": "s4-r8"}})


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
data:
  future_feature_cache_mode: shared_float32_projected_next_view
  training_split: all
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
