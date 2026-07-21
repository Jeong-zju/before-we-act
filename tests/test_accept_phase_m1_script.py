from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from eval.m1_future_probe import cluster_aware_probe_comparisons
from scripts.accept_phase_m1 import (
    CANONICAL_CONFIG,
    PROBE_ACTION_SOURCE,
    REQUIRED_PROBE_BASELINES,
    _active_capacity_contract,
    _active_capacity_counts_match,
    _audit_checkpoints,
    _contains_forbidden_diagnostic_marker,
    _evidence_paths,
    _expected_causal_pair_contract,
    _expected_state_causal_pair_contract,
    _exact_seed_matrix,
    _fixed_gate_thresholds,
    _formal_causal_pair_evidence_check,
    _formal_state_pair_exclusion_check,
    _formal_state_causal_pair_evidence_check,
    _formal_future_probe_evidence_check,
    _formal_preflight_state_pair_exclusion,
    _formal_visual_evidence_check,
    _legacy_checkpoint_evidence_matches_training,
    _loaded_checkpoint_contract,
    _module_sha256,
    _primary_checkpoint_evidence_matches_training,
    _sha256,
    _stage_state_causal_pair_evidence,
    _training_contract,
    _visual_protocol_check,
    build_parser,
)
from train.m1_checkpointing import CHECKPOINT_FORMAT_VERSION


def _config() -> dict:
    value = yaml.safe_load(CANONICAL_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_canonical_acceptance_paths_are_formal_and_bound_to_config() -> None:
    args = build_parser().parse_args([])
    paths = _evidence_paths(
        _config(), config_path=CANONICAL_CONFIG.resolve(), args=args
    )
    assert paths["formal_protocol"] is True
    assert paths["output"].name == "phase_m1_acceptance.json"
    assert paths["episode_records"].name == "visual_required_episodes.jsonl"


def test_diagnostic_override_requires_separate_output(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--training-summary", str(tmp_path / "training.json")]
    )
    with pytest.raises(ValueError, match="separate --output"):
        _evidence_paths(_config(), config_path=CANONICAL_CONFIG.resolve(), args=args)

    args = build_parser().parse_args(
        [
            "--training-summary",
            str(tmp_path / "training.json"),
            "--output",
            str(tmp_path / "acceptance.json"),
        ]
    )
    paths = _evidence_paths(
        _config(), config_path=CANONICAL_CONFIG.resolve(), args=args
    )
    assert paths["formal_protocol"] is False
    assert paths["output"] == (tmp_path / "acceptance.json").resolve()


def test_fixed_thresholds_and_three_stage_contract_fail_closed() -> None:
    config = _config()
    assert _fixed_gate_thresholds(config)["passed"]
    contract = _training_contract(config)
    assert contract["stage_order"] == [
        "adapter_fusion",
        "fusion_future_action",
        "legacy_low_lr",
    ]
    assert contract["stage1_old_world_action_frozen"]
    assert contract["legacy_learning_rate_ratio"] == pytest.approx(0.1)

    mutated = deepcopy(config)
    mutated["acceptance"]["minimum_visual_gain"] = 0.0
    assert not _fixed_gate_thresholds(mutated)["passed"]


def test_causal_pair_evidence_is_hash_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    config = _config()
    contract, valid = _expected_causal_pair_contract(config)
    assert valid
    tasks = config["model"]["task_vocabulary"]
    train_pairs = {
        "pairs": 900,
        "pairs_by_task": {task: 300 for task in tasks},
        "anchor_t_by_task": {
            "visual_event_stop": {"14": 300},
            "visual_target_select": {"0": 300},
            "visual_obstacle_avoid": {"0": 300},
        },
        "visual_history_alignment": "deployable_prefix_right_padding",
        "single_effective_frame_pairs": 600,
        "two_effective_frame_pairs": 300,
        "controlled_action_dims": [0, 1, 2, 4, 5, 6],
        "audit_sample_ids_sha256": "a" * 64,
    }
    evidence = {
        "train": {
            "windows": 10_000,
            "observationally_ambiguous_windows": 4200,
            "sampling_eligible_windows": 5800,
            "observationally_ambiguous_sample_ids_sha256": "b" * 64,
        },
        "causal_pairs": {"contract": contract, "train": train_pairs},
    }
    evidence_path = tmp_path / "data_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    training = {"data_evidence_sha256": _sha256(evidence_path)}
    assert _formal_causal_pair_evidence_check(
        config,
        training,
        evidence,
        data_evidence_path=evidence_path,
    )["passed"]

    tampered = deepcopy(evidence)
    tampered["causal_pairs"]["train"]["anchor_t_by_task"]["visual_event_stop"] = {
        "13": 300
    }
    assert not _formal_causal_pair_evidence_check(
        config,
        training,
        tampered,
        data_evidence_path=evidence_path,
    )["passed"]


def test_state_causal_pairs_are_diagnostic_only_and_excluded_from_formal_data(
    tmp_path: Path,
) -> None:
    config = _config()
    contract, valid = _expected_state_causal_pair_contract(config)
    assert valid
    assert contract["diagnostic_only"] is True
    assert contract["formal_enabled"] is False
    assert contract["rejection_reason"] == "gap1_not_runtime_replan_and_cold_only"
    evidence = {
        "formal_state_causal_pairs": {
            "enabled": False,
            "loaded": False,
            "trained": False,
            "rejection_reason": "gap1_not_runtime_replan_and_cold_only",
        }
    }
    assert _formal_state_pair_exclusion_check(config, evidence)["passed"]

    evidence_path = tmp_path / "data_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert _formal_state_causal_pair_evidence_check(
        config,
        {},
        evidence,
        data_evidence_path=evidence_path,
    )["passed"]

    loaded = deepcopy(evidence)
    loaded["state_causal_pairs"] = {"train": {"pairs": 574}}
    assert not _formal_state_pair_exclusion_check(config, loaded)["passed"]

    contaminated = deepcopy(evidence)
    contaminated["lineage"] = {"diagnostic_state_pair_calibration": True}
    assert not _formal_state_pair_exclusion_check(config, contaminated)["passed"]

    formalized = deepcopy(config)
    formalized["training"]["state_causal_pairs"]["formal_enabled"] = True
    assert not _formal_state_pair_exclusion_check(formalized, evidence)["passed"]

def test_state_causal_pair_stage_evidence_requires_explicit_zero_use() -> None:
    metric_names = (
        "state_causal_pair_total",
        "state_causal_pair_factual_endpoint",
        "state_causal_pair_action_delta",
        "state_causal_pair_delta_cosine",
        "state_causal_pair_delta_cosine_valid_fraction",
        "state_causal_pair_predicted_delta_rms",
        "state_causal_pair_target_delta_rms",
        "state_causal_pair_step0_predicted_delta_rms",
        "state_causal_pair_step0_target_delta_rms",
    )
    disabled = {
        "state_causal_pairs_enabled": False,
        "state_causal_pair_gradient_scope": None,
        "state_causal_pair_weights": None,
        **{name: 0.0 for name in metric_names},
    }
    assert _stage_state_causal_pair_evidence(disabled)
    enabled = deepcopy(disabled)
    enabled["state_causal_pairs_enabled"] = True
    assert not _stage_state_causal_pair_evidence(enabled)
    nonzero = deepcopy(disabled)
    nonzero["state_causal_pair_action_delta"] = 1e-6
    assert not _stage_state_causal_pair_evidence(nonzero)


def test_formal_artifact_diagnostic_marker_detection_is_recursive() -> None:
    assert not _contains_forbidden_diagnostic_marker({"stages": [{"total": 1.0}]})
    assert _contains_forbidden_diagnostic_marker(
        {"provenance": {"diagnostic_state_pair_calibration": True}}
    )
    assert _contains_forbidden_diagnostic_marker(
        {"reports": [{"diagnostic_only": False}]}
    )


def test_formal_preflight_requires_state_pair_disabled_in_every_stage() -> None:
    metric_names = (
        "state_causal_pair_total",
        "state_causal_pair_factual_endpoint",
        "state_causal_pair_action_delta",
        "state_causal_pair_delta_cosine",
        "state_causal_pair_delta_cosine_valid_fraction",
        "state_causal_pair_predicted_delta_rms",
        "state_causal_pair_target_delta_rms",
        "state_causal_pair_step0_predicted_delta_rms",
        "state_causal_pair_step0_target_delta_rms",
    )
    disabled_stage = {
        "state_causal_pairs_enabled": False,
        "state_causal_pair_gradient_scope": None,
        "state_causal_pair_weights": None,
        **{name: 0.0 for name in metric_names},
    }
    training = {
        "preflight": {
            "passed": True,
            "overfit_256": {
                "formal_state_causal_pairs_enabled": False,
                "training": deepcopy(disabled_stage),
            },
            "one_percent": {
                "stages": [deepcopy(disabled_stage) for _ in range(3)]
            },
        }
    }
    assert _formal_preflight_state_pair_exclusion(training)
    training["preflight"]["one_percent"]["stages"][1][
        "state_causal_pairs_enabled"
    ] = True
    assert not _formal_preflight_state_pair_exclusion(training)


def test_missing_checkpoint_matrix_cannot_produce_reload_evidence(
    tmp_path: Path,
) -> None:
    config = _config()
    config["training"]["checkpoint_root"] = str(tmp_path / "missing")
    training = {
        "variants": [
            "state_only",
            "vision_only",
            "state_vision_no_future",
            "state_vision_future",
            "parameter_matched_mlp",
        ],
        "train_seeds": list(config["training"]["seeds"]),
        "checkpoint_sha256": {},
        "strict_reload": {},
    }
    audit = _audit_checkpoints(
        config,
        training,
        device=torch.device("cpu"),
        manifest_sha256=str(config["data"]["expected_manifest_sha256"]),
    )
    assert not audit["check"]["passed"]
    assert audit["parameter_counts"]["trainable"] is None
    assert all(
        not row["passed"]
        for values in audit["strict_reload"].values()
        for row in values.values()
    )


def test_checkpoint_evidence_matrices_are_exact_and_probe_is_training_bound() -> None:
    seeds = (101, 202, 303)
    sha = "a" * 64
    hashes = {
        variant: {str(seed): sha for seed in seeds}
        for variant in (
            "state_only",
            "vision_only",
            "state_vision_no_future",
            "state_vision_future",
            "parameter_matched_mlp",
        )
    }
    assert _exact_seed_matrix(hashes, train_seeds=seeds)
    with_extra = deepcopy(hashes)
    with_extra["state_only"]["404"] = sha
    assert not _exact_seed_matrix(with_extra, train_seeds=seeds)

    training = {"checkpoint_sha256": hashes}
    primary = dict(hashes["state_vision_future"])
    reloads = {str(seed): True for seed in seeds}
    assert _primary_checkpoint_evidence_matches_training(
        primary, reloads, training, train_seeds=seeds
    )
    primary["101"] = "b" * 64
    assert not _primary_checkpoint_evidence_matches_training(
        primary, reloads, training, train_seeds=seeds
    )


def test_visual_protocol_is_bound_to_canonical_seeds_rates_and_deadlines() -> None:
    config = _config()
    evaluation = config["evaluation"]
    seeds = list(
        range(
            evaluation["physical_seed_start"],
            evaluation["physical_seed_start"]
            + evaluation["formal_physical_seeds_per_task"],
        )
    )
    visual = {
        "evaluation_seeds": seeds,
        "aggregation": {"passed": True},
        "runtime": {
            "decimated": True,
            "warmup_actions_excluded": True,
            "control_hz": evaluation["control_hz"],
            "visual_hz": evaluation["visual_refresh_hz"],
            "deadline_misses": 0,
            "worker_processes": 3,
            "torch_threads_per_worker": 8,
            "resolved_device": "cpu",
            "sensor_to_action_definition": "policy_act_wall_time_ms",
            "action_age_definition": "frame receipt to action",
            "hardware": {
                "platform": "test-platform",
                "machine": "x86_64",
                "torch_version": "test",
                "logical_cpu_count": 24,
                "cpu_model": "test-cpu",
            },
            "runtime_scope": {
                "model_variant": "state_vision_future",
                "intervention": "clean",
                "all_configured_train_seeds": True,
                "all_visual_required_tasks": True,
            },
            "replan_contract": {
                "execution_steps": 2,
                "warm_start_enabled": False,
                "training_warm_start_probability": 0.0,
                "observation_regrounding": "cold_start_every_execute_2_replan",
                "scope": "m1_latent_flow_visual_required_only",
                "observed_replan_events": 7,
                "observed_cold_replan_events": 7,
                "observed_warm_replan_events": 0,
            },
            "deadline_contract": {
                "direct_without_vision_ms": config["acceptance"][
                    "maximum_sensor_to_action_p95_ms"
                ],
                "decimated_with_vision_ms": config["acceptance"][
                    "maximum_decimated_action_age_ms"
                ],
            },
        },
    }
    records = [
        {
            "model_variant": "state_vision_future",
            "intervention": "clean",
            "steps": 13,
            "replan_events": 7,
            "cold_replan_events": 7,
            "warm_replan_events": 0,
        }
    ]
    assert _visual_protocol_check(visual, config, records=records)["passed"]
    mutated = deepcopy(visual)
    mutated["evaluation_seeds"][0] += 1
    assert not _visual_protocol_check(mutated, config, records=records)["passed"]
    mutated = deepcopy(visual)
    mutated["runtime"]["deadline_misses"] = 1
    assert _visual_protocol_check(mutated, config, records=records)["passed"]
    mutated["runtime"]["deadline_misses"] = None
    assert not _visual_protocol_check(mutated, config, records=records)["passed"]
    warm_records = deepcopy(records)
    warm_records[0]["cold_replan_events"] = 6
    warm_records[0]["warm_replan_events"] = 1
    assert not _visual_protocol_check(
        visual, config, records=warm_records
    )["passed"]
    missing_counter = deepcopy(records)
    del missing_counter[0]["warm_replan_events"]
    assert not _visual_protocol_check(
        visual, config, records=missing_counter
    )["passed"]
    mutated = deepcopy(visual)
    mutated["runtime"]["replan_contract"]["observed_replan_events"] = 8
    assert not _visual_protocol_check(mutated, config, records=records)["passed"]


def test_visual_report_is_bound_to_exact_episode_jsonl(tmp_path: Path) -> None:
    records_path = (tmp_path / "episodes.jsonl").resolve()
    records_path.write_text('{"success":true}\n', encoding="utf-8")
    records_sha256 = _sha256(records_path)
    visual = {
        "format_version": "wam.multimodal.m1.visual_evaluation/1",
        "formal_protocol": True,
        "passed": True,
        "phase": "formal",
        "config_sha256": "a" * 64,
        "training_summary_sha256": "b" * 64,
        "records": 1,
        "expected_records": 1,
        "episode_records": str(records_path),
        "episode_records_sha256": records_sha256,
    }
    assert _formal_visual_evidence_check(
        visual,
        config_sha256="a" * 64,
        training_sha256="b" * 64,
        episode_records_path=records_path,
        episode_records_sha256=records_sha256,
        record_count=1,
    )["passed"]

    stale_summary = deepcopy(visual)
    stale_summary["episode_records_sha256"] = "c" * 64
    result = _formal_visual_evidence_check(
        stale_summary,
        config_sha256="a" * 64,
        training_sha256="b" * 64,
        episode_records_path=records_path,
        episode_records_sha256=records_sha256,
        record_count=1,
    )
    assert not result["passed"]
    assert not result["checks"]["episode_records_sha256_matches"]


def test_legacy_checkpoint_evidence_is_bound_to_primary_training_hashes() -> None:
    seeds = (101, 202, 303)
    hashes = {str(seed): f"{seed:064x}" for seed in seeds}
    training = {"checkpoint_sha256": {"state_vision_future": hashes}}
    evidence = {
        str(seed): {
            "tree_sha256": hashes[str(seed)],
            "train_seed": seed,
            "model_variant": "state_vision_future",
            "strict_reload_passed": True,
            "embedded_legacy_matches_source": True,
            "schema_format_version": CHECKPOINT_FORMAT_VERSION,
        }
        for seed in seeds
    }
    assert _legacy_checkpoint_evidence_matches_training(
        evidence, training, train_seeds=seeds
    )
    evidence["101"]["embedded_legacy_matches_source"] = False
    assert not _legacy_checkpoint_evidence_matches_training(
        evidence, training, train_seeds=seeds
    )


def test_loaded_checkpoint_contract_binds_embedded_config_and_formal_metrics(
    tmp_path: Path,
) -> None:
    config = _config()
    variant = "state_vision_future"
    seed = 101
    path = (tmp_path / variant / f"seed_{seed}").resolve()
    path.mkdir(parents=True)
    tree_sha256 = "c" * 64
    resampler = SimpleNamespace(
        input_dim=config["model"]["vision_patch_dim"],
        width=config["model"]["resampler_width"],
        num_latents=config["model"]["resampler_tokens"],
        num_layers=config["model"]["resampler_layers"],
        num_heads=config["model"]["resampler_heads"],
        mlp_ratio=config["model"]["resampler_mlp_ratio"],
        dropout=config["model"]["dropout"],
        raw_patch_grid=config["model"]["raw_patch_grid"],
        raw_patch_hidden_dim=config["model"]["raw_patch_hidden_dim"],
        raw_shortcut_hidden_dim=config["model"]["raw_shortcut_hidden_dim"],
    )
    vision = SimpleNamespace(
        artifact_sha256=config["initialization"]["expected_vision_weights_sha256"],
        config=SimpleNamespace(
            crop_size=config["model"]["vision_input_size"],
            resize_shorter_side=config["model"]["vision_input_size"],
        ),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            use_state=True,
            use_vision=True,
            capacity_control="future_head",
            task_vocabulary=tuple(config["model"]["task_vocabulary"]),
            future_horizons=tuple(config["model"]["future_visual_horizons"]),
            action_dim=config["data"]["action_dim"],
            visual_skip_initial_scale=config["model"]["visual_skip_initial_scale"],
            resampler=resampler,
        ),
        vision_encoder=vision,
        action_capacity_mlp=None,
    )
    anchor = torch.nn.Linear(2, 2)
    flow = SimpleNamespace(
        anchor_prior=anchor,
        config=SimpleNamespace(horizon=config["action_chunk"]["horizon"]),
    )
    counts = {
        "total_trainable": 20_000_000,
        "total_active": 20_000_000,
        "capacity_padding": 0,
        "future_head": 1_000_000,
        "action_mlp_functional": 0,
    }
    anchor_sha256 = _module_sha256(anchor)
    causal_contract, causal_config_valid = _expected_causal_pair_contract(config)
    assert causal_config_valid
    causal_weights = {
        name: float(config["training"]["causal_pairs"][name])
        for name in (
            "factual_endpoint",
            "action_delta",
            "delta_direction",
            "executed_prefix_weight",
        )
    }
    cold_flow_objective = {
        "execution_steps": 2,
        "solver_steps": 4,
        "solver": "euler",
        "normalized_action_clip": 10.0,
        "warm_start_probability": 0.0,
        "warm_start_noise_std": 0.0,
        "policy_fixed_action_dims": [3, 7],
        "executed_prefix_weight": 2.0,
    }
    stages = [
        {
            "name": config["training"][name]["name"],
            "steps": config["training"][name]["steps"],
            "frozen_backbone": True,
            "frozen_prior_anchor": True,
            "flow_objective": cold_flow_objective,
            "warm_fraction": 0.0,
            "causal_pairs_enabled": True,
            "causal_pair_gradient_scope": "visual_adapter_fusion_only",
            "causal_pair_weights": causal_weights,
            "causal_pair_total": 1.0,
            "causal_pair_factual_endpoint": 0.1,
            "causal_pair_action_delta": 0.1,
            "causal_pair_delta_cosine": 0.8,
            "causal_pair_delta_cosine_valid_fraction": 1.0,
            "causal_pair_predicted_delta_rms": 0.8,
            "causal_pair_target_delta_rms": 1.0,
            "state_causal_pairs_enabled": False,
            "state_causal_pair_gradient_scope": None,
            "state_causal_pair_weights": None,
            "state_causal_pair_total": 0.0,
            "state_causal_pair_factual_endpoint": 0.0,
            "state_causal_pair_action_delta": 0.0,
            "state_causal_pair_delta_cosine": 0.0,
            "state_causal_pair_delta_cosine_valid_fraction": 0.0,
            "state_causal_pair_predicted_delta_rms": 0.0,
            "state_causal_pair_target_delta_rms": 0.0,
            "state_causal_pair_step0_predicted_delta_rms": 0.0,
            "state_causal_pair_step0_target_delta_rms": 0.0,
        }
        for name in ("stage_1", "stage_2", "stage_3")
    ]
    assert not any(row["state_causal_pairs_enabled"] for row in stages)

    def pair_metrics(*, delta_rmse: float, cosine: float, sign: float) -> dict:
        row = {
            "pairs": 10,
            "factual_endpoint_rmse": 0.1,
            "action_delta_rmse": delta_rmse,
            "delta_cosine": cosine,
            "predicted_delta_rms": 1.0,
            "target_delta_rms": 1.0,
            "delta_norm_ratio": 1.0,
            "executed_prefix_sign_agreement": sign,
            "executed_prefix_nonzero_targets": 10,
        }
        return {
            "cold_start": True,
            "solver_steps": 4,
            "solver": "euler",
            "batches": 3,
            **row,
            "by_task_index": {str(index): dict(row) for index in range(3)},
        }

    before_pair_metrics = pair_metrics(delta_rmse=2.0, cosine=0.1, sign=0.5)
    after_pair_metrics = pair_metrics(delta_rmse=1.0, cosine=0.8, sign=0.9)
    causal_summary_sha256 = "d" * 64
    metrics = {
        "format_version": "wam.multimodal.m1.training_run/1",
        "formal_protocol": True,
        "variant": variant,
        "train_seed": seed,
        "preflight_passed": True,
        "parameter_counts": counts,
        "frozen_anchor_sha256_before": anchor_sha256,
        "frozen_anchor_sha256_after": anchor_sha256,
        "stages": stages,
        "causal_pair_contract": causal_contract,
        "causal_pair_summary_sha256": causal_summary_sha256,
        "causal_pair_metrics_before": before_pair_metrics,
        "causal_pair_metrics_after": after_pair_metrics,
        "causal_pair_passed": True,
        "formal_state_causal_pairs_enabled": False,
    }
    run_report = {
        **metrics,
        "checkpoint": str(path),
        "checkpoint_tree_sha256": tree_sha256,
        "strict_reload": {"passed": True, "max_abs_diff": 0.0},
    }
    metadata = {
        "experiment_config": {**config, "latent_wam_config": {}},
        "metrics": metrics,
        "dataset_manifest": {
            "causal_pair_contract": causal_contract,
            "causal_pair_summary_sha256": causal_summary_sha256,
            "causal_pair_summary": {
                "controlled_action_dims": [0, 1, 2, 4, 5, 6],
                "visual_history_alignment": "deployable_prefix_right_padding",
            },
        },
        "provenance": {
            "source_checkpoint_tree_sha256": config["initialization"][
                "expected_legacy_tree_sha256"
            ],
            "visual_source_sha256": config["initialization"][
                "expected_vision_weights_sha256"
            ],
        },
        "schema": {
            "vision_source_sha256": config["initialization"][
                "expected_vision_weights_sha256"
            ]
        },
    }
    assert _loaded_checkpoint_contract(
        config,
        variant=variant,
        train_seed=seed,
        model=model,
        flow=flow,
        metadata=metadata,
        run_report=run_report,
        path=path,
        tree_sha256=tree_sha256,
        actual_counts=counts,
    )["passed"]
    failed_run_report = deepcopy(run_report)
    failed_run_report["diagnostic_only"] = False
    assert not _loaded_checkpoint_contract(
        config,
        variant=variant,
        train_seed=seed,
        model=model,
        flow=flow,
        metadata=metadata,
        run_report=failed_run_report,
        path=path,
        tree_sha256=tree_sha256,
        actual_counts=counts,
    )["passed"]
    lineage = metadata["dataset_manifest"]
    lineage["diagnostic_state_pair_calibration"] = True
    assert not _loaded_checkpoint_contract(
        config,
        variant=variant,
        train_seed=seed,
        model=model,
        flow=flow,
        metadata=metadata,
        run_report=run_report,
        path=path,
        tree_sha256=tree_sha256,
        actual_counts=counts,
    )["passed"]
    del lineage["diagnostic_state_pair_calibration"]
    metadata["provenance"]["diagnostic_only"] = True
    assert not _loaded_checkpoint_contract(
        config,
        variant=variant,
        train_seed=seed,
        model=model,
        flow=flow,
        metadata=metadata,
        run_report=run_report,
        path=path,
        tree_sha256=tree_sha256,
        actual_counts=counts,
    )["passed"]
    del metadata["provenance"]["diagnostic_only"]
    metadata["experiment_config"] = deepcopy(metadata["experiment_config"])
    metadata["experiment_config"]["model"] = deepcopy(config["model"])
    metadata["experiment_config"]["model"]["resampler_tokens"] = 8
    assert not _loaded_checkpoint_contract(
        config,
        variant=variant,
        train_seed=seed,
        model=model,
        flow=flow,
        metadata=metadata,
        run_report=run_report,
        path=path,
        tree_sha256=tree_sha256,
        actual_counts=counts,
    )["passed"]


def _seed_probe_result(*, object_model_error: float = 0.1) -> dict[str, object]:
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 100)
    model = labels.copy()
    baseline = labels.copy()
    for pair in range(100):
        if pair % 10 == 0:
            model[2 * pair : 2 * pair + 2] = 1 - labels[2 * pair : 2 * pair + 2]
        if pair % 10 < 4:
            baseline[2 * pair : 2 * pair + 2] = 1 - labels[2 * pair : 2 * pair + 2]
    return {
        "object_model_errors": [object_model_error] * 200,
        "object_baseline_errors": [0.3] * 200,
        "object_action_baseline_errors": [0.35] * 200,
        "event_model_predictions": model.tolist(),
        "event_baseline_predictions": baseline.tolist(),
        "event_action_baseline_predictions": baseline.tolist(),
        "event_labels": labels.tolist(),
        "object_test_sample_ids_sha256": "a" * 64,
        "event_test_sample_ids_sha256": "b" * 64,
    }


def _future_probe_bundle(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict, Path, str, str]:
    config = _config()
    config["statistics"] = deepcopy(config["statistics"])
    config["statistics"]["bootstrap_samples"] = 300
    seeds = tuple(int(value) for value in config["training"]["seeds"])
    per_seed = {str(seed): _seed_probe_result() for seed in seeds}
    clustered = cluster_aware_probe_comparisons(
        per_seed,
        train_seeds=seeds,
        confidence=float(config["statistics"]["confidence"]),
        bootstrap_samples=int(config["statistics"]["bootstrap_samples"]),
        bootstrap_seed=int(config["statistics"]["bootstrap_seed"]),
    )
    checkpoint_hashes = {
        str(seed): f"{offset:x}" * 64 for offset, seed in enumerate(seeds, start=1)
    }
    training: dict[str, object] = {
        "checkpoint_sha256": {"state_vision_future": checkpoint_hashes}
    }
    training_path = (tmp_path / "training_summary.json").resolve()
    training_path.write_text(
        json.dumps(training, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    training_sha256 = _sha256(training_path)
    config_sha256 = "c" * 64
    future: dict[str, object] = {
        "format_version": "wam.multimodal.m1.future_probe/1",
        "phase": "M1",
        "formal_protocol": True,
        "passed": True,
        "claim_allowed": True,
        "baseline": REQUIRED_PROBE_BASELINES[0],
        "required_baselines": list(REQUIRED_PROBE_BASELINES),
        "probe_action_source": PROBE_ACTION_SOURCE,
        "probe_action_protocol": {
            "mode": "offline_teacher_forced_same_demonstration",
            "action_chunk_horizon": 8,
            "target_consistent_with_demonstration_h8_labels": True,
            "matched_information_baseline_receives_identical_action_chunk": True,
            "deployed_action_flow_used": False,
            "deployed_causal_claim_allowed": False,
        },
        "position_probe_target": {
            "field": "h8_center_xy",
            "semantics": "future_robot_carrier_center_xy",
            "carried_object_position_proxy": True,
            "explicit_visual_target_position": False,
            "unseen_target_generalization_claim_allowed": False,
        },
        "horizon": 8,
        "train_seeds": list(seeds),
        **clustered["aggregate"],
        "comparisons": clustered["comparisons"],
        "cluster_evidence": clustered["cluster_evidence"],
        "per_seed": per_seed,
        "selection": {
            "test": {
                "object_sample_ids_sha256": "a" * 64,
                "event_sample_ids_sha256": "b" * 64,
            }
        },
        "checkpoint_sha256": checkpoint_hashes,
        "primary_checkpoint_tree_sha256": checkpoint_hashes,
        "primary_checkpoint_trees_stable_during_probe": True,
        "strict_reload": {str(seed): True for seed in seeds},
        "artifact_sha256": {
            "config": config_sha256,
            "dataset_manifest": config["data"]["expected_manifest_sha256"],
            "visual_backbone": config["initialization"][
                "expected_vision_weights_sha256"
            ],
            "training_summary": training_sha256,
        },
        "training_summary": str(training_path),
        "training_summary_sha256": training_sha256,
        "training_summary_stable_during_probe": True,
        "no_future_target_leakage": {
            "passed": True,
            "candidate_actions_source": PROBE_ACTION_SOURCE,
            "dataset_action_targets_read": True,
            "dataset_action_targets_forwarded_to_future_head": True,
            "matched_baseline_receives_identical_action_targets": True,
            "target_action_pairing": "same_demonstration_window",
            "deployed_causal_claim_allowed": False,
            "future_images_forwarded": False,
            "future_states_forwarded": False,
            "probe_labels_forwarded": False,
            "test_samples_used_for_fit_or_selection": 0,
            "train_validation_test_sample_ids_disjoint": True,
            "manifest_hdf5_sha256_verified": True,
            "manifest_hdf5_contract_verified": True,
        },
    }
    return (
        future,
        training,
        config,
        training_path,
        training_sha256,
        config_sha256,
    )


def _check_future_bundle(
    future: dict[str, object],
    training: dict[str, object],
    config: dict,
    training_path: Path,
    training_sha256: str,
    config_sha256: str,
) -> dict[str, object]:
    return _formal_future_probe_evidence_check(
        future,
        training,
        config=config,
        config_sha256=config_sha256,
        training_summary_path=training_path,
        training_summary_sha256=training_sha256,
        train_seeds=tuple(int(value) for value in config["training"]["seeds"]),
    )


def test_future_probe_evidence_is_recomputed_and_fully_bound(tmp_path: Path) -> None:
    values = _future_probe_bundle(tmp_path)
    assert _check_future_bundle(*values)["passed"]

    stale_training = deepcopy(values[0])
    stale_training["artifact_sha256"]["training_summary"] = "d" * 64
    assert not _check_future_bundle(stale_training, *values[1:])["passed"]

    missing_baseline = deepcopy(values[0])
    missing_baseline["required_baselines"] = [REQUIRED_PROBE_BASELINES[0]]
    assert not _check_future_bundle(missing_baseline, *values[1:])["passed"]

    deployed_claim = deepcopy(values[0])
    deployed_claim["probe_action_protocol"]["deployed_causal_claim_allowed"] = True
    assert not _check_future_bundle(deployed_claim, *values[1:])["passed"]

    mismatched_sample_ids = deepcopy(values[0])
    mismatched_sample_ids["per_seed"]["303"]["object_test_sample_ids_sha256"] = "e" * 64
    assert not _check_future_bundle(mismatched_sample_ids, *values[1:])["passed"]


def test_future_probe_cannot_hide_failed_seed_behind_passed_booleans(
    tmp_path: Path,
) -> None:
    values = _future_probe_bundle(tmp_path)
    tampered = deepcopy(values[0])
    tampered["per_seed"]["303"]["object_model_errors"] = [0.5] * 200
    assert tampered["passed"] is True
    assert tampered["comparisons"]["robust_improvement"] is True

    result = _check_future_bundle(tampered, *values[1:])
    assert not result["passed"]
    assert not result["checks"]["reported_aggregate_matches_recomputation"]
    assert not result["checks"]["recomputed_every_seed_and_cluster_pass"]


def test_active_capacity_rejects_padding_and_requires_real_match() -> None:
    future_model = SimpleNamespace(action_capacity_mlp=None)
    future_counts = {
        "total_trainable": 100,
        "total_active": 100,
        "capacity_padding": 0,
        "future_head": 20,
        "action_mlp_functional": 0,
    }
    functional_mlp = SimpleNamespace(padding_parameter_count=0)
    mlp_model = SimpleNamespace(action_capacity_mlp=functional_mlp)
    mlp_counts = {
        "total_trainable": 100,
        "total_active": 100,
        "capacity_padding": 0,
        "future_head": 0,
        "action_mlp_functional": 20,
    }
    assert _active_capacity_contract("state_vision_future", future_model, future_counts)
    assert _active_capacity_contract("parameter_matched_mlp", mlp_model, mlp_counts)
    matrices = {
        "state_vision_future": {
            seed: deepcopy(future_counts) for seed in (101, 202, 303)
        },
        "parameter_matched_mlp": {
            seed: deepcopy(mlp_counts) for seed in (101, 202, 303)
        },
    }
    assert _active_capacity_counts_match(matrices, train_seeds=(101, 202, 303))

    padded_module = SimpleNamespace(
        capacity_padding=torch.nn.Parameter(torch.zeros(10)),
        padding_parameter_count=10,
    )
    padded_counts = {
        **mlp_counts,
        "total_trainable": 110,
        "capacity_padding": 10,
    }
    assert not _active_capacity_contract(
        "parameter_matched_mlp",
        SimpleNamespace(action_capacity_mlp=padded_module),
        padded_counts,
    )
    matrices["parameter_matched_mlp"][303]["total_active"] = 99
    assert not _active_capacity_counts_match(matrices, train_seeds=(101, 202, 303))
