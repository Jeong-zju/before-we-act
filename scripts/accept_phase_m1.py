"""Aggregate the canonical Phase M1 evidence and apply every fail-closed gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.visual_required_env import VISUAL_REQUIRED_TASKS  # noqa: E402
from eval.m1_acceptance import (  # noqa: E402
    MAXIMUM_DECIMATED_ACTION_AGE_P95_MS,
    MAXIMUM_DIRECT_P95_MS,
    MAXIMUM_LEGACY_REGRESSION,
    MAXIMUM_TRAINABLE_PARAMETERS,
    MINIMUM_STATE_SHUFFLE_DROP,
    MINIMUM_TRAINABLE_PARAMETERS,
    MINIMUM_VISUAL_INTERVENTION_DROP,
    MINIMUM_VISION_GAIN,
    REQUIRED_VARIANTS,
    m1_acceptance_report,
)
from eval.m1_future_probe import (  # noqa: E402
    PROBE_ACTION_SOURCE,
    cluster_aware_probe_comparisons,
)
from eval.m1_vision_contract import (  # noqa: E402
    allowed_checkpoint_formats,
    validate_loaded_checkpoint_vision,
    validate_source_artifacts,
    validate_training_summary_vision,
    vision_artifact_paths,
)
from train.m1_checkpointing import (  # noqa: E402
    CHECKPOINT_FORMAT_VERSION,
    LEGACY_CHECKPOINT_FORMAT_VERSION,
    checkpoint_tree_sha256,
    load_m1_checkpoint,
)


CANONICAL_CONFIG = ROOT / "configs/wam_multimodal/m1_latent_wam_dinov3.yaml"
FORMAT_VERSION = "wam.multimodal.m1.acceptance_bundle/2"
REQUIRED_PROBE_BASELINES = (
    "current_frame_only",
    "current_frame_plus_same_demonstration_action_chunk",
)
EXPECTED_CAUSAL_PAIR_LITERALS = {
    "enabled": True,
    "contract_version": "wam.multimodal.m1.causal_pairs/1",
    "construction": "first_equal_history_rgb_different_execute2_action_delta_v2",
    "apply_to_vision_variants": True,
    "filter_unobservable_conflicts": True,
    "gradient_scope": "visual_adapter_fusion_only",
}
EXPECTED_CAUSAL_PAIR_WEIGHTS = {
    "factual_endpoint": 6.0,
    "action_delta": 6.0,
    "delta_direction": 0.25,
    "executed_prefix_weight": 4.0,
}
FORBIDDEN_FORMAL_DIAGNOSTIC_MARKERS = frozenset(
    {"diagnostic_state_pair_calibration", "diagnostic_only"}
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CANONICAL_CONFIG)
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--visual-evaluation", type=Path)
    parser.add_argument("--episode-records", type=Path)
    parser.add_argument("--future-probe", type=Path)
    parser.add_argument("--legacy-regression", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve(strict=True)
    config = _load_yaml(config_path)
    paths = _evidence_paths(config, config_path=config_path, args=args)
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(max(1, min(4, args.torch_threads)))
    device = _device(args.device)

    training = _load_json(paths["training_summary"])
    validate_training_summary_vision(training, config, project_root=ROOT)
    training_data_evidence_path = (
        paths["training_summary"].parent / "data_evidence.json"
    )
    training_data_evidence = _load_json(training_data_evidence_path)
    visual = _load_json(paths["visual_evaluation"])
    future = _load_json(paths["future_probe"])
    legacy = _load_json(paths["legacy_regression"])
    upstream = _load_json(paths["upstream_m0"])
    records = _load_jsonl(paths["episode_records"])

    config_sha256 = _sha256(config_path)
    training_sha256 = _sha256(paths["training_summary"])
    episode_records_sha256 = _sha256(paths["episode_records"])
    train_seeds = tuple(int(value) for value in config["training"]["seeds"])
    manifest_path = _root_path(str(config["data"]["manifest"]))
    vision_path, _ = vision_artifact_paths(config, project_root=ROOT)
    vision_artifacts = validate_source_artifacts(config, project_root=ROOT)
    legacy_checkpoint = _root_path(
        str(config["initialization"]["legacy_joint_wam_checkpoint"])
    )
    source_before = checkpoint_tree_sha256(legacy_checkpoint)
    checkpoint_audit = _audit_checkpoints(
        config,
        training,
        device=device,
        manifest_sha256=_sha256(manifest_path),
    )
    source_after = checkpoint_tree_sha256(legacy_checkpoint)

    bundle_checks = {
        "canonical_request": _check(paths["formal_protocol"]),
        "fixed_gate_thresholds": _fixed_gate_thresholds(config),
        "upstream_m0_accepted_and_pinned": _check(
            upstream.get("formal_protocol") is True
            and upstream.get("passed") is True
            and _sha256(paths["upstream_m0"])
            == str(config["acceptance"]["expected_upstream_m0_acceptance_sha256"]),
            observed_sha256=_sha256(paths["upstream_m0"]),
            expected_sha256=str(
                config["acceptance"]["expected_upstream_m0_acceptance_sha256"]
            ),
        ),
        "canonical_data_and_initialization": _check(
            _sha256(manifest_path) == str(config["data"]["expected_manifest_sha256"])
            and vision_artifacts["passed"] is True
            and source_before
            == str(config["initialization"]["expected_legacy_tree_sha256"])
            and source_before == source_after,
            manifest_sha256=_sha256(manifest_path),
            visual_backbone_sha256=_sha256(vision_path),
            visual_artifacts=vision_artifacts,
            legacy_tree_sha256_before=source_before,
            legacy_tree_sha256_after=source_after,
        ),
        "formal_training_evidence": _check(
            training.get("format_version") == "wam.multimodal.m1.training/1"
            and training.get("formal_protocol") is True
            and training.get("passed") is True
            and Path(str(training.get("config", ""))).resolve() == config_path
            and training.get("config_sha256") == config_sha256
            and training.get("manifest_sha256")
            == str(config["data"]["expected_manifest_sha256"])
            and training.get("visual_backbone_sha256")
            == str(config["initialization"]["expected_vision_weights_sha256"])
            and training.get("source_checkpoint_immutable") is True
            and not _contains_forbidden_diagnostic_marker(training),
        ),
        "formal_preflight_excludes_state_causal_pairs": _check(
            _formal_preflight_state_pair_exclusion(training)
        ),
        "formal_preflight_uses_cold_replans": _check(
            _formal_preflight_cold_replan(training, config)
        ),
        "causal_pair_training_protocol": _formal_causal_pair_evidence_check(
            config,
            training,
            training_data_evidence,
            data_evidence_path=training_data_evidence_path,
        ),
        "state_causal_pair_diagnostics_excluded": _formal_state_pair_exclusion_check(
            config, training_data_evidence
        ),
        "formal_visual_evaluation_evidence": _formal_visual_evidence_check(
            visual,
            config_sha256=config_sha256,
            training_sha256=training_sha256,
            episode_records_path=paths["episode_records"],
            episode_records_sha256=episode_records_sha256,
            record_count=len(records),
        ),
        "visual_protocol_bound_to_config": _visual_protocol_check(
            visual, config, records=records
        ),
        "formal_future_probe_evidence": _formal_future_probe_evidence_check(
            future,
            training,
            config=config,
            config_sha256=config_sha256,
            training_summary_path=paths["training_summary"],
            training_summary_sha256=training_sha256,
            train_seeds=train_seeds,
        ),
        "formal_legacy_regression_evidence": _check(
            legacy.get("format_version") == "wam.multimodal.m1.legacy_regression/1"
            and legacy.get("formal_protocol") is True
            and legacy.get("passed") is True
            and legacy.get("config_sha256") == config_sha256
            and legacy.get("training_summary_sha256") == training_sha256
            and legacy.get("source_checkpoint", {}).get("immutable") is True
            and legacy.get("source_checkpoint", {}).get("tree_sha256_before")
            == source_before
            and legacy.get("source_checkpoint", {}).get("tree_sha256_after")
            == source_after
            and _integer_tuple(legacy.get("train_seeds"))
            == tuple(int(value) for value in config["training"]["seeds"])
            and _strict_integer(legacy.get("expected_episodes_per_suite"))
            == int(config["evaluation"]["legacy_regression_episodes_per_suite"])
            and _legacy_checkpoint_evidence_matches_training(
                legacy.get("checkpoint_evidence"),
                training,
                train_seeds=train_seeds,
                config=config,
            ),
        ),
        "all_checkpoint_trees_and_strict_reloads": checkpoint_audit["check"],
    }

    evaluation_seeds = tuple(int(value) for value in visual.get("evaluation_seeds", ()))
    architecture = _architecture(config, checkpoint_audit)
    training_contract = _training_contract(config)
    evidence = {
        "artifact_sha256": {
            "dataset_manifest": _sha256(manifest_path),
            "episode_records": episode_records_sha256,
            "config": config_sha256,
            "visual_backbone": _sha256(vision_path),
        },
        "visual_backbone_weights_sha256": _sha256(vision_path),
        "checkpoint_sha256": checkpoint_audit["checkpoint_sha256"],
        "strict_reload": checkpoint_audit["strict_reload"],
        "source_checkpoint_immutable": bool(
            source_before == source_after
            and source_before
            == str(config["initialization"]["expected_legacy_tree_sha256"])
        ),
    }
    report = m1_acceptance_report(
        records,
        tasks=tuple(str(value) for value in VISUAL_REQUIRED_TASKS),
        evaluation_seeds=evaluation_seeds,
        cue_variants=tuple(
            int(value) for value in config["evaluation"]["cue_variants"]
        ),
        train_seeds=train_seeds,
        architecture=architecture,
        training_contract=training_contract,
        parameter_counts=checkpoint_audit["parameter_counts"],
        legacy_suites=legacy.get("suites", {}),
        future_probe=future,
        runtime=visual.get("runtime", {}),
        evidence=evidence,
        formal_protocol=bool(paths["formal_protocol"]),
        confidence=float(config["statistics"]["confidence"]),
        bootstrap_samples=int(config["statistics"]["bootstrap_samples"]),
        bootstrap_seed=int(config["statistics"]["bootstrap_seed"]),
    )
    core_passed = report["passed"] is True
    bundle_passed = all(value["passed"] for value in bundle_checks.values())
    report.update(
        {
            "bundle_format_version": FORMAT_VERSION,
            "passed": bool(core_passed and bundle_passed),
            "claim_allowed": bool(core_passed and bundle_passed),
            "core_gate_passed": core_passed,
            "bundle_checks_passed": bundle_passed,
            "bundle_checks": bundle_checks,
            "evidence_files": {
                name: {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for name, path in paths.items()
                if name != "output" and isinstance(path, Path) and path.is_file()
            },
            "limitations": list(config.get("limitations", ())),
        }
    )
    _atomic_json(paths["output"], report)
    print(
        f"Phase M1 formal={report['formal_protocol']} passed={report['passed']} "
        f"records={len(records)}",
        flush=True,
    )
    return 0 if report["passed"] else 1


def _evidence_paths(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    training_root = _root_path(str(config["training"]["report_root"]))
    evaluation_root = _root_path(str(config["evaluation"]["output_directory"]))
    canonical = {
        "training_summary": training_root / "training_summary.json",
        "visual_evaluation": evaluation_root / "visual_required_evaluation.json",
        "episode_records": evaluation_root / "visual_required_episodes.jsonl",
        "future_probe": evaluation_root / "future_probe.json",
        "legacy_regression": evaluation_root / "legacy_regression.json",
        "upstream_m0": _root_path(str(config["acceptance"]["upstream_m0_acceptance"])),
        "output": _root_path(str(config["acceptance"]["output"])),
    }
    override_names = (
        "training_summary",
        "visual_evaluation",
        "episode_records",
        "future_probe",
        "legacy_regression",
        "output",
    )
    formal = bool(
        config_path == CANONICAL_CONFIG.resolve()
        and all(getattr(args, name) is None for name in override_names)
    )
    if not formal and args.output is None:
        raise ValueError("diagnostic acceptance overrides require a separate --output")
    paths: dict[str, Any] = dict(canonical)
    for name in override_names:
        value = getattr(args, name)
        if value is not None:
            paths[name] = value.resolve()
    if not formal and paths["output"] == canonical["output"]:
        raise ValueError("diagnostic acceptance cannot overwrite canonical M1 evidence")
    paths["formal_protocol"] = formal
    return paths


def _audit_checkpoints(
    config: Mapping[str, Any],
    training: Mapping[str, Any],
    *,
    device: torch.device,
    manifest_sha256: str,
) -> dict[str, Any]:
    train_seeds = tuple(int(value) for value in config["training"]["seeds"])
    checkpoint_root = _root_path(str(config["training"]["checkpoint_root"]))
    declared_hashes = training.get("checkpoint_sha256")
    declared_reloads = training.get("strict_reload")
    run_reports, reports_exact = _training_run_reports(
        training.get("reports"), train_seeds=train_seeds
    )
    observed_hashes: dict[str, dict[str, str]] = {}
    strict_reload: dict[str, dict[str, dict[str, Any]]] = {}
    details: dict[str, Any] = {}
    passed = bool(
        _exact_seed_matrix(declared_hashes, train_seeds=train_seeds)
        and _exact_seed_matrix(declared_reloads, train_seeds=train_seeds)
        and _string_tuple(training.get("variants")) == REQUIRED_VARIANTS
        and _integer_tuple(training.get("train_seeds")) == train_seeds
        and isinstance(training.get("checkpoint_root"), str)
        and Path(str(training["checkpoint_root"])).resolve() == checkpoint_root
        and reports_exact
    )
    primary_counts: list[dict[str, int]] = []
    variant_counts: dict[str, dict[int, dict[str, int]]] = {
        variant: {} for variant in REQUIRED_VARIANTS
    }
    for variant in REQUIRED_VARIANTS:
        observed_hashes[variant] = {}
        strict_reload[variant] = {}
        for train_seed in train_seeds:
            path = checkpoint_root / variant / f"seed_{train_seed}"
            row: dict[str, Any] = {"path": str(path), "passed": False}
            try:
                tree_sha256 = checkpoint_tree_sha256(path)
                expected_hash = _seed_value(declared_hashes, variant, train_seed)
                declared_reload = _seed_value(declared_reloads, variant, train_seed)
                model, flow, _, _, metadata = load_m1_checkpoint(
                    path,
                    device=device,
                    expected_schema_version=str(config["data"]["schema_version"]),
                )
                schema = metadata["schema"]
                lineage = metadata["dataset_manifest"]
                run_report = run_reports.get((variant, train_seed))
                actual_counts = model.parameter_breakdown(flow)
                exact_declared_reload = bool(
                    isinstance(declared_reload, Mapping)
                    and declared_reload.get("passed") is True
                    and _exact_zero(declared_reload.get("max_abs_diff"))
                )
                checkpoint_contract = _loaded_checkpoint_contract(
                    config,
                    variant=variant,
                    train_seed=train_seed,
                    model=model,
                    flow=flow,
                    metadata=metadata,
                    run_report=run_report,
                    path=path,
                    tree_sha256=tree_sha256,
                    actual_counts=actual_counts,
                )
                row["passed"] = bool(
                    tree_sha256 == expected_hash
                    and exact_declared_reload
                    and schema.get("model_variant") == variant
                    and int(schema.get("train_seed", -1)) == train_seed
                    and lineage.get("manifest_sha256") == manifest_sha256
                    and lineage.get("split") == "train"
                    and model.vision_encoder is not None
                    and not any(
                        parameter.requires_grad
                        for parameter in model.vision_encoder.parameters()
                    )
                    and checkpoint_contract["passed"]
                )
                row["tree_sha256"] = tree_sha256
                row["contract"] = checkpoint_contract
                observed_hashes[variant][str(train_seed)] = tree_sha256
                strict_reload[variant][str(train_seed)] = {
                    "passed": row["passed"],
                    "max_abs_diff": 0.0 if row["passed"] else None,
                }
                variant_counts[variant][train_seed] = actual_counts
                if variant == "state_vision_future":
                    primary_counts.append(actual_counts)
                del model, flow
                gc.collect()
            except (
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as error:
                row["error"] = f"{type(error).__name__}:{error}"
                strict_reload[variant][str(train_seed)] = {
                    "passed": False,
                    "max_abs_diff": None,
                }
            passed = passed and row["passed"]
            details[f"{variant}/seed_{train_seed}"] = row
    counts_equal = bool(
        len(primary_counts) == len(train_seeds)
        and all(value == primary_counts[0] for value in primary_counts)
    )
    matched_capacity = _active_capacity_counts_match(
        variant_counts,
        train_seeds=train_seeds,
    )
    passed = passed and counts_equal and matched_capacity
    parameter_counts = {
        "trainable": (int(primary_counts[0]["total_active"]) if counts_equal else None),
        "frozen_visual_backbone": (
            int(primary_counts[0]["vision_encoder_frozen"]) if counts_equal else None
        ),
        "breakdown": primary_counts[0] if counts_equal else None,
    }
    return {
        "check": _check(
            passed,
            expected_checkpoints=len(REQUIRED_VARIANTS) * len(train_seeds),
            audited_checkpoints=len(details),
            primary_parameter_counts_equal=counts_equal,
            future_and_mlp_parameter_counts_equal=matched_capacity,
            future_and_mlp_active_forward_counts_equal=matched_capacity,
            capacity_padding_rejected=True,
            details=details,
        ),
        "checkpoint_sha256": observed_hashes,
        "strict_reload": strict_reload,
        "parameter_counts": parameter_counts,
    }


def _loaded_checkpoint_contract(
    config: Mapping[str, Any],
    *,
    variant: str,
    train_seed: int,
    model: Any,
    flow: Any,
    metadata: Mapping[str, Any],
    run_report: Mapping[str, Any] | None,
    path: Path,
    tree_sha256: str,
    actual_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Bind a reload to the canonical architecture and formal training run."""

    expected_switches = {
        "state_only": (True, False, "none"),
        "vision_only": (False, True, "none"),
        "state_vision_no_future": (True, True, "none"),
        "state_vision_future": (True, True, "future_head"),
        "parameter_matched_mlp": (True, True, "action_mlp"),
    }
    expected_state, expected_vision, expected_capacity = expected_switches[variant]
    latent = model.config
    resampler = latent.resampler
    model_config = config["model"]
    experiment = metadata.get("experiment_config")
    metrics = metadata.get("metrics")
    lineage = metadata.get("dataset_manifest")
    provenance = metadata.get("provenance")
    schema = metadata.get("schema")
    formal_checkpoint_payload = {
        key: value for key, value in metadata.items() if key != "experiment_config"
    }
    causal_contract, causal_config_matches = _expected_causal_pair_contract(config)
    causal_pairs_expected = expected_vision
    expected_stages = [
        (
            str(config["training"][name]["name"]),
            int(config["training"][name]["steps"]),
        )
        for name in ("stage_1", "stage_2", "stage_3")
    ]
    observed_stages = metrics.get("stages") if isinstance(metrics, Mapping) else None
    if isinstance(schema, Mapping) and schema.get("format_version") is None:
        # Backward-compatible unit surface for the pre-v2 contract helper. Real
        # checkpoint metadata reaches this function only after strict loader
        # validation and always contains a format_version.
        encoder = getattr(model, "vision_encoder", None)
        encoder_config = getattr(encoder, "config", None)
        visual_source_matches = bool(
            encoder is not None
            and getattr(encoder, "artifact_sha256", None)
            == str(config["initialization"]["expected_vision_weights_sha256"])
            and int(getattr(encoder_config, "crop_size", -1))
            == int(model_config["vision_input_size"])
            and int(getattr(encoder_config, "resize_shorter_side", -1))
            == int(model_config["vision_input_size"])
            and schema.get("vision_source_sha256")
            == str(config["initialization"]["expected_vision_weights_sha256"])
        )
    else:
        try:
            validate_loaded_checkpoint_vision(config, model, metadata)
            visual_source_matches = True
        except (KeyError, TypeError, ValueError):
            visual_source_matches = False
    stages_match = bool(
        isinstance(observed_stages, list)
        and len(observed_stages) == len(expected_stages)
        and all(
            isinstance(row, Mapping)
            and row.get("name") == expected_name
            and _strict_integer(row.get("steps")) == expected_steps
            and row.get("frozen_backbone") is True
            and row.get("frozen_prior_anchor") is True
            and _stage_causal_pair_evidence(
                row,
                enabled=causal_pairs_expected,
                expected_weights=EXPECTED_CAUSAL_PAIR_WEIGHTS,
            )
            and _stage_state_causal_pair_evidence(row)
            and _stage_cold_replan_evidence(row, config)
            for row, (expected_name, expected_steps) in zip(
                observed_stages, expected_stages, strict=True
            )
        )
    )
    anchor_sha256 = _module_sha256(flow.anchor_prior)
    metrics_match = bool(
        isinstance(metrics, Mapping)
        and metrics.get("format_version") == "wam.multimodal.m1.training_run/1"
        and metrics.get("formal_protocol") is True
        and metrics.get("variant") == variant
        and _strict_integer(metrics.get("train_seed")) == train_seed
        and metrics.get("preflight_passed") is True
        and metrics.get("causal_pair_passed") is True
        and metrics.get("formal_state_causal_pairs_enabled") is False
        and metrics.get("parameter_counts") == dict(actual_counts)
        and metrics.get("frozen_anchor_sha256_before") == anchor_sha256
        and metrics.get("frozen_anchor_sha256_after") == anchor_sha256
        and metrics.get("causal_pair_contract") == causal_contract
        and _is_sha256(metrics.get("causal_pair_summary_sha256"))
        and _checkpoint_causal_pair_metrics(
            metrics,
            enabled=causal_pairs_expected,
            require_quality=(variant == "state_vision_future"),
        )
        and not _contains_forbidden_diagnostic_marker(metrics)
        and stages_match
    )
    report_match = bool(
        isinstance(run_report, Mapping)
        and run_report.get("format_version") == "wam.multimodal.m1.training_run/1"
        and run_report.get("formal_protocol") is True
        and run_report.get("variant") == variant
        and _strict_integer(run_report.get("train_seed")) == train_seed
        and Path(str(run_report.get("checkpoint", ""))).resolve() == path
        and run_report.get("checkpoint_tree_sha256") == tree_sha256
        and run_report.get("stages") == metrics.get("stages")
        and isinstance(run_report.get("strict_reload"), Mapping)
        and run_report["strict_reload"].get("passed") is True
        and _exact_zero(run_report["strict_reload"].get("max_abs_diff"))
        and run_report.get("causal_pair_contract")
        == metrics.get("causal_pair_contract")
        and run_report.get("causal_pair_summary_sha256")
        == metrics.get("causal_pair_summary_sha256")
        and run_report.get("causal_pair_metrics_before")
        == metrics.get("causal_pair_metrics_before")
        and run_report.get("causal_pair_metrics_after")
        == metrics.get("causal_pair_metrics_after")
        and run_report.get("causal_pair_passed") is True
        and run_report.get("causal_pair_passed") == metrics.get("causal_pair_passed")
        and run_report.get("formal_state_causal_pairs_enabled") is False
        and not _contains_forbidden_diagnostic_marker(run_report)
    )
    checks = {
        "embedded_experiment_config_matches": bool(
            isinstance(experiment, Mapping)
            and all(experiment.get(name) == value for name, value in config.items())
        ),
        "causal_pair_config_matches": causal_config_matches,
        "causal_pair_lineage_matches": bool(
            isinstance(lineage, Mapping)
            and lineage.get("causal_pair_contract") == causal_contract
            and lineage.get("causal_pair_summary_sha256")
            == metrics.get("causal_pair_summary_sha256")
            and isinstance(lineage.get("causal_pair_summary"), Mapping)
            and lineage["causal_pair_summary"].get("controlled_action_dims")
            == [0, 1, 2, 4, 5, 6]
            and lineage["causal_pair_summary"].get("visual_history_alignment")
            == "deployable_prefix_right_padding"
        ),
        "state_causal_pair_diagnostics_absent": bool(
            isinstance(lineage, Mapping)
            and isinstance(provenance, Mapping)
            and not _contains_forbidden_diagnostic_marker(lineage)
            and not _contains_forbidden_diagnostic_marker(provenance)
            and not _contains_forbidden_diagnostic_marker(formal_checkpoint_payload)
        ),
        "variant_switches_match": bool(
            latent.use_state is expected_state
            and latent.use_vision is expected_vision
            and latent.capacity_control == expected_capacity
        ),
        "task_vocabulary_matches": tuple(latent.task_vocabulary)
        == tuple(str(value) for value in model_config["task_vocabulary"]),
        "resampler_matches": bool(
            int(resampler.input_dim) == int(model_config["vision_patch_dim"])
            and int(resampler.width) == int(model_config["resampler_width"])
            and int(resampler.num_latents) == int(model_config["resampler_tokens"])
            and int(resampler.num_layers) == int(model_config["resampler_layers"])
            and int(resampler.num_heads) == int(model_config["resampler_heads"])
            and int(resampler.mlp_ratio) == int(model_config["resampler_mlp_ratio"])
            and float(resampler.dropout) == float(model_config["dropout"])
            and int(resampler.raw_patch_grid) == int(model_config["raw_patch_grid"])
            and int(resampler.raw_patch_hidden_dim)
            == int(model_config["raw_patch_hidden_dim"])
            and int(resampler.raw_shortcut_hidden_dim)
            == int(model_config["raw_shortcut_hidden_dim"])
            and float(latent.visual_skip_initial_scale)
            == float(model_config["visual_skip_initial_scale"])
        ),
        "future_horizons_match": tuple(int(value) for value in latent.future_horizons)
        == tuple(int(value) for value in model_config["future_visual_horizons"]),
        "action_contract_matches": bool(
            int(latent.action_dim) == int(config["data"]["action_dim"])
            and int(flow.config.horizon) == int(config["action_chunk"]["horizon"])
        ),
        "visual_source_matches": visual_source_matches,
        "formal_training_metrics_match": metrics_match,
        "training_summary_run_matches": report_match,
        "active_capacity_contract": _active_capacity_contract(
            variant, model, actual_counts
        ),
        "provenance_matches": bool(
            isinstance(provenance, Mapping)
            and provenance.get("source_checkpoint_tree_sha256")
            == str(config["initialization"]["expected_legacy_tree_sha256"])
            and provenance.get("visual_source_sha256")
            == str(config["initialization"]["expected_vision_weights_sha256"])
        ),
    }
    return _check(all(checks.values()), checks=checks)


def _stage_causal_pair_evidence(
    row: Mapping[str, Any],
    *,
    enabled: bool,
    expected_weights: Mapping[str, float],
) -> bool:
    if row.get("causal_pairs_enabled") is not enabled:
        return False
    expected_scope = "visual_adapter_fusion_only" if enabled else None
    if row.get("causal_pair_gradient_scope") != expected_scope:
        return False
    metric_names = (
        "causal_pair_total",
        "causal_pair_factual_endpoint",
        "causal_pair_action_delta",
        "causal_pair_delta_cosine",
        "causal_pair_delta_cosine_valid_fraction",
        "causal_pair_predicted_delta_rms",
        "causal_pair_target_delta_rms",
    )
    if enabled:
        if row.get("causal_pair_weights") != dict(expected_weights):
            return False
        values = [_finite_scalar(row.get(name)) for name in metric_names]
        return bool(
            all(value is not None for value in values)
            and values[-1] is not None
            and values[-1] > 0.0
            and values[4] is not None
            and 0.0 <= values[4] <= 1.0
        )
    return bool(
        row.get("causal_pair_weights") is None
        and all(_exact_zero(row.get(name)) for name in metric_names)
    )


def _stage_state_causal_pair_evidence(
    row: Mapping[str, Any],
) -> bool:
    """Require explicit zero-use evidence in every formal training stage."""

    if row.get("state_causal_pairs_enabled") is not False:
        return False
    if row.get("state_causal_pair_gradient_scope") is not None:
        return False
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
    return bool(
        row.get("state_causal_pair_weights") is None
        and all(_exact_zero(row.get(name)) for name in metric_names)
    )


def _formal_preflight_state_pair_exclusion(training: Mapping[str, Any]) -> bool:
    preflight = training.get("preflight")
    if not isinstance(preflight, Mapping) or preflight.get("passed") is not True:
        return False
    overfit = preflight.get("overfit_256")
    one_percent = preflight.get("one_percent")
    if not isinstance(overfit, Mapping) or not isinstance(one_percent, Mapping):
        return False
    overfit_stage = overfit.get("training")
    one_percent_stages = one_percent.get("stages")
    return bool(
        overfit.get("formal_state_causal_pairs_enabled") is False
        and isinstance(overfit_stage, Mapping)
        and _stage_state_causal_pair_evidence(overfit_stage)
        and isinstance(one_percent_stages, list)
        and len(one_percent_stages) == 3
        and all(
            isinstance(stage, Mapping) and _stage_state_causal_pair_evidence(stage)
            for stage in one_percent_stages
        )
    )


def _formal_preflight_cold_replan(
    training: Mapping[str, Any], config: Mapping[str, Any]
) -> bool:
    preflight = training.get("preflight")
    if not isinstance(preflight, Mapping) or preflight.get("passed") is not True:
        return False
    overfit = preflight.get("overfit_256")
    one_percent = preflight.get("one_percent")
    if not isinstance(overfit, Mapping) or not isinstance(one_percent, Mapping):
        return False
    overfit_stage = overfit.get("training")
    one_percent_stages = one_percent.get("stages")
    return bool(
        isinstance(overfit_stage, Mapping)
        and _stage_cold_replan_evidence(overfit_stage, config)
        and isinstance(one_percent_stages, list)
        and len(one_percent_stages) == 3
        and all(
            isinstance(stage, Mapping) and _stage_cold_replan_evidence(stage, config)
            for stage in one_percent_stages
        )
    )


def _stage_cold_replan_evidence(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> bool:
    chunk = config.get("action_chunk")
    objective = row.get("flow_objective")
    if not isinstance(chunk, Mapping) or not isinstance(objective, Mapping):
        return False
    expected = {
        "execution_steps": 2,
        "solver_steps": 4,
        "solver": "euler",
        "normalized_action_clip": 10.0,
        "warm_start_probability": 0.0,
        "warm_start_noise_std": 0.0,
        "policy_fixed_action_dims": [3, 7],
        "executed_prefix_weight": 2.0,
    }
    return bool(
        chunk.get("replan_warm_start_enabled") is False
        and _exact_number(chunk.get("warm_start_probability"), 0.0)
        and _exact_number(chunk.get("warm_start_noise_std"), 0.0)
        and dict(objective) == expected
        and _exact_zero(row.get("warm_fraction"))
    )


def _checkpoint_causal_pair_metrics(
    metrics: Mapping[str, Any], *, enabled: bool, require_quality: bool
) -> bool:
    before = metrics.get("causal_pair_metrics_before")
    after = metrics.get("causal_pair_metrics_after")
    if not enabled:
        return before is None and after is None
    if not (_valid_causal_pair_metrics(before) and _valid_causal_pair_metrics(after)):
        return False
    assert isinstance(before, Mapping) and isinstance(after, Mapping)
    if not require_quality:
        return True
    if not (
        float(after["action_delta_rmse"]) < float(before["action_delta_rmse"])
        and float(after["delta_cosine"]) > 0.5
        and float(after["executed_prefix_sign_agreement"]) >= 0.75
        and 0.25 <= float(after["delta_norm_ratio"]) <= 1.75
    ):
        return False
    before_tasks = before["by_task_index"]
    after_tasks = after["by_task_index"]
    assert isinstance(before_tasks, Mapping) and isinstance(after_tasks, Mapping)
    return all(
        float(after_tasks[task]["action_delta_rmse"])
        < float(before_tasks[task]["action_delta_rmse"])
        and float(after_tasks[task]["delta_cosine"]) > 0.5
        and float(after_tasks[task]["executed_prefix_sign_agreement"]) >= 0.75
        and 0.25 <= float(after_tasks[task]["delta_norm_ratio"]) <= 1.75
        for task in ("0", "1", "2")
    )


def _valid_causal_pair_metrics(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    by_task = value.get("by_task_index")
    numeric_names = (
        "factual_endpoint_rmse",
        "action_delta_rmse",
        "delta_cosine",
        "predicted_delta_rms",
        "target_delta_rms",
        "delta_norm_ratio",
        "executed_prefix_sign_agreement",
    )
    if not (
        value.get("cold_start") is True
        and _strict_integer(value.get("solver_steps")) == 4
        and value.get("solver") == "euler"
        and (_strict_integer(value.get("pairs")) or 0) > 0
        and all(_finite_scalar(value.get(name)) is not None for name in numeric_names)
        and isinstance(by_task, Mapping)
        and set(by_task) == {"0", "1", "2"}
    ):
        return False
    return all(
        isinstance(by_task[task], Mapping)
        and (_strict_integer(by_task[task].get("pairs")) or 0) > 0
        and all(
            _finite_scalar(by_task[task].get(name)) is not None
            for name in numeric_names
        )
        for task in ("0", "1", "2")
    )


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _active_capacity_contract(
    variant: str, model: Any, counts: Mapping[str, Any]
) -> bool:
    """Reject dormant capacity and validate the forward-active count schema."""

    total_trainable = _strict_nonnegative_integer(counts.get("total_trainable"))
    total_active = _strict_nonnegative_integer(counts.get("total_active"))
    capacity_padding = _strict_nonnegative_integer(counts.get("capacity_padding"))
    future_head = _strict_nonnegative_integer(counts.get("future_head"))
    action_mlp = _strict_nonnegative_integer(counts.get("action_mlp_functional"))
    if None in (
        total_trainable,
        total_active,
        capacity_padding,
        future_head,
        action_mlp,
    ):
        return False
    if capacity_padding != 0 or total_active > total_trainable:
        return False

    capacity_module = getattr(model, "action_capacity_mlp", None)
    padding_attribute = (
        getattr(capacity_module, "capacity_padding", None)
        if capacity_module is not None
        else None
    )
    if padding_attribute is not None:
        try:
            if int(padding_attribute.numel()) != 0:
                return False
        except (AttributeError, TypeError, ValueError):
            return False
    padding_count_attribute = (
        getattr(capacity_module, "padding_parameter_count", 0)
        if capacity_module is not None
        else 0
    )
    if _strict_nonnegative_integer(padding_count_attribute) != 0:
        return False

    if variant == "state_vision_future":
        return bool(
            capacity_module is None
            and future_head > 0
            and action_mlp == 0
            and total_active == total_trainable
        )
    if variant == "parameter_matched_mlp":
        return bool(
            capacity_module is not None
            and future_head == 0
            and action_mlp > 0
            and total_active == total_trainable
        )
    return bool(capacity_module is None and capacity_padding == 0)


def _active_capacity_counts_match(
    variant_counts: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    train_seeds: Sequence[int],
) -> bool:
    future_counts = variant_counts.get("state_vision_future")
    mlp_counts = variant_counts.get("parameter_matched_mlp")
    if not isinstance(future_counts, Mapping) or not isinstance(mlp_counts, Mapping):
        return False
    for seed in train_seeds:
        future = future_counts.get(int(seed))
        mlp = mlp_counts.get(int(seed))
        if not isinstance(future, Mapping) or not isinstance(mlp, Mapping):
            return False
        future_active = _strict_nonnegative_integer(future.get("total_active"))
        mlp_active = _strict_nonnegative_integer(mlp.get("total_active"))
        future_head = _strict_nonnegative_integer(future.get("future_head"))
        mlp_functional = _strict_nonnegative_integer(mlp.get("action_mlp_functional"))
        future_padding = _strict_nonnegative_integer(future.get("capacity_padding"))
        mlp_padding = _strict_nonnegative_integer(mlp.get("capacity_padding"))
        if (
            future_active is None
            or mlp_active is None
            or future_head is None
            or mlp_functional is None
            or future_padding != 0
            or mlp_padding != 0
            or future_active != mlp_active
            or future_head != mlp_functional
        ):
            return False
    return True


def _training_run_reports(
    value: Any, *, train_seeds: Sequence[int]
) -> tuple[dict[tuple[str, int], Mapping[str, Any]], bool]:
    if not isinstance(value, list):
        return {}, False
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    valid = True
    for row in value:
        if not isinstance(row, Mapping):
            valid = False
            continue
        variant = row.get("variant")
        train_seed = _strict_integer(row.get("train_seed"))
        if variant not in REQUIRED_VARIANTS or train_seed is None:
            valid = False
            continue
        identity = (str(variant), train_seed)
        if identity in result:
            valid = False
        result[identity] = row
    expected = {
        (variant, int(train_seed))
        for variant in REQUIRED_VARIANTS
        for train_seed in train_seeds
    }
    return result, bool(
        valid and set(result) == expected and len(value) == len(expected)
    )


def _exact_seed_matrix(value: Any, *, train_seeds: Sequence[int]) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(REQUIRED_VARIANTS):
        return False
    return all(
        _exact_seed_mapping(value.get(variant), train_seeds)
        for variant in REQUIRED_VARIANTS
    )


def _exact_seed_mapping(value: Any, seeds: Sequence[int]) -> bool:
    if not isinstance(value, Mapping) or len(value) != len(seeds):
        return False
    observed: set[int] = set()
    for key in value:
        normalized = _strict_seed_key(key)
        if normalized is None or normalized in observed:
            return False
        observed.add(normalized)
    return observed == {int(seed) for seed in seeds}


def _strict_seed_key(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value:
        try:
            result = int(value)
        except ValueError:
            return None
        if value == str(result) and result >= 0:
            return result
    return None


def _strict_integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _strict_nonnegative_integer(value: Any) -> int | None:
    result = _strict_integer(value)
    return result if result is not None and result >= 0 else None


def _integer_tuple(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    result = tuple(_strict_integer(item) for item in value)
    return (
        None
        if any(item is None for item in result)
        else tuple(int(item) for item in result)
    )


def _string_tuple(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return tuple(value) if all(isinstance(item, str) for item in value) else None


def _exact_zero(value: Any) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and float(value) == 0.0
    )


def _module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _architecture(
    config: Mapping[str, Any], checkpoint_audit: Mapping[str, Any]
) -> dict[str, Any]:
    model = config["model"]
    chunk = config["action_chunk"]
    audit_passed = checkpoint_audit["check"]["passed"] is True
    return {
        "state_feature_encoder_preserved": audit_passed,
        "world_heads_preserved": audit_passed,
        "stateful_action_flow": audit_passed,
        "prior_anchor_frozen": audit_passed,
        "prior_anchor_immutable": audit_passed,
        "visual_backbone_pretrained": audit_passed
        and bool(model["use_pretrained_frozen_backbone"]),
        "visual_backbone_frozen": audit_passed,
        "resampler_layers": int(model["resampler_layers"]),
        "visual_tokens": int(model["resampler_tokens"]),
        "planning_feature_fuses_state_and_visual": audit_passed,
        "future_visual_latent_head": audit_passed,
        "future_horizons": [int(value) for value in model["future_visual_horizons"]],
        "action_residual": audit_passed,
        "action_flow_experts": int(chunk["expert_count"]),
        "action_chunk_steps": int(chunk["horizon"]),
        "execution_steps": int(chunk["execution_steps"]),
    }


def _training_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config["training"]
    chunk = config["action_chunk"]
    stage1 = training["stage_1"]
    stage2 = training["stage_2"]
    stage3 = training["stage_3"]
    return {
        "stage_order": [
            str(stage1["name"]),
            str(stage2["name"]),
            str(stage3["name"]),
        ],
        "stage1_old_world_action_frozen": not bool(stage1["train_action_flow"])
        and not bool(stage1["train_world_model"]),
        "stage1_visual_backbone_frozen": True,
        "stage2_visual_backbone_frozen": True,
        "legacy_learning_rate_ratio": float(stage3["world_learning_rate"])
        / float(stage3["learning_rate"]),
        "vision_unfrozen_blocks": 0,
        "m1_stable_before_visual_unfreeze": False,
        "replan_warm_start_enabled": bool(chunk["replan_warm_start_enabled"]),
        "training_warm_start_probability": float(chunk["warm_start_probability"]),
        "observation_regrounding": "cold_start_every_execute_2_replan",
        "cold_replan_scope": "m1_latent_flow_visual_required_only",
    }


def _expected_causal_pair_contract(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    raw = config.get("training", {}).get("causal_pairs", {})
    if not isinstance(raw, Mapping):
        return {}, False
    contract = {
        **EXPECTED_CAUSAL_PAIR_LITERALS,
        "pair_batch_size": 8,
        "weights": dict(EXPECTED_CAUSAL_PAIR_WEIGHTS),
        "calibration_steps": 96,
        "calibration_learning_rate": 0.0001,
        "state_only_pair_supervision": "disabled_no_vision",
    }
    raw_expected = {
        **EXPECTED_CAUSAL_PAIR_LITERALS,
        "pair_batch_size": contract["pair_batch_size"],
        **EXPECTED_CAUSAL_PAIR_WEIGHTS,
        "calibration_steps": contract["calibration_steps"],
        "calibration_learning_rate": contract["calibration_learning_rate"],
    }
    observed = {name: raw.get(name) for name in raw_expected}
    return contract, observed == raw_expected


def _expected_state_causal_pair_contract(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Expose the diagnostic selector contract without making it formal evidence."""

    raw = config.get("training", {}).get("state_causal_pairs", {})
    if not isinstance(raw, Mapping):
        return {}, False
    required = {
        "diagnostic_only": True,
        "formal_enabled": False,
        "rejection_reason": "gap1_not_runtime_replan_and_cold_only",
    }
    observed = {name: raw.get(name) for name in required}
    return dict(raw), observed == required


def _contains_forbidden_diagnostic_marker(value: Any) -> bool:
    """Detect diagnostic calibration markers in formal artifact payloads."""

    if isinstance(value, Mapping):
        if any(key in FORBIDDEN_FORMAL_DIAGNOSTIC_MARKERS for key in value):
            return True
        return any(
            _contains_forbidden_diagnostic_marker(item) for item in value.values()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_forbidden_diagnostic_marker(item) for item in value)
    return False


def _formal_state_pair_exclusion_check(
    config: Mapping[str, Any], data_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove the gap-1/cold-only state-pair diagnostic never entered formal M1."""

    _, diagnostic_config_valid = _expected_state_causal_pair_contract(config)
    expected = {
        "enabled": False,
        "loaded": False,
        "trained": False,
        "rejection_reason": "gap1_not_runtime_replan_and_cold_only",
    }
    observed = data_evidence.get("formal_state_causal_pairs")
    checks = {
        "config_marks_selector_diagnostic_only": diagnostic_config_valid,
        "formal_data_did_not_load_or_train_pairs": observed == expected,
        "state_pair_payload_absent": "state_causal_pairs" not in data_evidence,
        "diagnostic_markers_absent_from_formal_data": (
            not _contains_forbidden_diagnostic_marker(data_evidence)
        ),
    }
    return _check(all(checks.values()), checks=checks, observed=observed)


def _formal_state_causal_pair_evidence_check(
    config: Mapping[str, Any],
    training: Mapping[str, Any],
    data_evidence: Mapping[str, Any],
    *,
    data_evidence_path: Path,
) -> dict[str, Any]:
    del training, data_evidence_path
    return _formal_state_pair_exclusion_check(config, data_evidence)


def _state_causal_pair_section_valid(
    config: Mapping[str, Any], state_pairs: Any
) -> bool:
    del config, state_pairs
    return False


def _valid_state_causal_pair_summary(
    value: Any, *, split: str, config: Mapping[str, Any]
) -> bool:
    del value, split, config
    return False


def _formal_causal_pair_evidence_check(
    config: Mapping[str, Any],
    training: Mapping[str, Any],
    data_evidence: Mapping[str, Any],
    *,
    data_evidence_path: Path,
) -> dict[str, Any]:
    expected_contract, config_matches = _expected_causal_pair_contract(config)
    causal = data_evidence.get("causal_pairs")
    train_windows = data_evidence.get("train")
    train_pairs = causal.get("train") if isinstance(causal, Mapping) else None
    task_order = tuple(str(value) for value in config["model"]["task_vocabulary"])
    pairs_by_task = (
        train_pairs.get("pairs_by_task") if isinstance(train_pairs, Mapping) else None
    )
    anchors = (
        train_pairs.get("anchor_t_by_task")
        if isinstance(train_pairs, Mapping)
        else None
    )
    expected_anchors = {
        "visual_event_stop": {"14": 300},
        "visual_target_select": {"0": 300},
        "visual_obstacle_avoid": {"0": 300},
    }
    ambiguous_count = (
        _strict_integer(train_windows.get("observationally_ambiguous_windows"))
        if isinstance(train_windows, Mapping)
        else None
    )
    effective_count = (
        _strict_integer(train_windows.get("sampling_eligible_windows"))
        if isinstance(train_windows, Mapping)
        else None
    )
    window_count = (
        _strict_integer(train_windows.get("windows"))
        if isinstance(train_windows, Mapping)
        else None
    )
    checks = {
        "canonical_config_contract": config_matches,
        "training_binds_data_evidence_sha256": bool(
            training.get("data_evidence_sha256") == _sha256(data_evidence_path)
        ),
        "data_contract_matches": bool(
            isinstance(causal, Mapping) and causal.get("contract") == expected_contract
        ),
        "train_pair_count_and_balance": bool(
            isinstance(train_pairs, Mapping)
            and _strict_integer(train_pairs.get("pairs")) == 900
            and pairs_by_task == {task_id: 300 for task_id in task_order}
        ),
        "causal_boundaries_match_audit": anchors == expected_anchors,
        "reset_frame_alignment_matches_deployment": bool(
            isinstance(train_pairs, Mapping)
            and train_pairs.get("visual_history_alignment")
            == "deployable_prefix_right_padding"
            and _strict_integer(train_pairs.get("single_effective_frame_pairs")) == 600
            and _strict_integer(train_pairs.get("two_effective_frame_pairs")) == 300
        ),
        "controlled_action_dimensions_locked": bool(
            isinstance(train_pairs, Mapping)
            and train_pairs.get("controlled_action_dims") == [0, 1, 2, 4, 5, 6]
        ),
        "opaque_pair_ids_hash_bound": bool(
            isinstance(train_pairs, Mapping)
            and _is_sha256(train_pairs.get("audit_sample_ids_sha256"))
        ),
        "unobservable_conflicts_removed_from_sampler": bool(
            isinstance(train_windows, Mapping)
            and ambiguous_count == 4200
            and effective_count is not None
            and window_count is not None
            and effective_count == window_count - 4200
            and _is_sha256(
                train_windows.get("observationally_ambiguous_sample_ids_sha256")
            )
        ),
    }
    return _check(
        all(checks.values()),
        checks=checks,
        expected_contract=expected_contract,
        data_evidence_path=str(data_evidence_path),
    )


def _fixed_gate_thresholds(config: Mapping[str, Any]) -> dict[str, Any]:
    acceptance = config.get("acceptance", {})
    expected = {
        "primary_visual_value_gate": "paired_success_gain_10pp",
        "minimum_trainable_parameters": MINIMUM_TRAINABLE_PARAMETERS,
        "maximum_trainable_parameters": MAXIMUM_TRAINABLE_PARAMETERS,
        "minimum_visual_gain": MINIMUM_VISION_GAIN,
        "maximum_legacy_regression": MAXIMUM_LEGACY_REGRESSION,
        "minimum_visual_intervention_drop": MINIMUM_VISUAL_INTERVENTION_DROP,
        "minimum_state_shuffle_drop": MINIMUM_STATE_SHUFFLE_DROP,
        "maximum_sensor_to_action_p95_ms": MAXIMUM_DIRECT_P95_MS,
        "maximum_decimated_action_age_ms": MAXIMUM_DECIMATED_ACTION_AGE_P95_MS,
    }
    observed = {name: acceptance.get(name) for name in expected}
    return _check(observed == expected, expected=expected, observed=observed)


def _formal_visual_evidence_check(
    visual: Mapping[str, Any],
    *,
    config_sha256: str,
    training_sha256: str,
    episode_records_path: Path,
    episode_records_sha256: str,
    record_count: int,
) -> dict[str, Any]:
    """Bind the visual summary to the exact JSONL accepted below."""

    declared_path = visual.get("episode_records")
    path_matches = bool(
        isinstance(declared_path, str)
        and Path(declared_path).resolve() == episode_records_path.resolve()
    )
    checks = {
        "format_and_formal_status": bool(
            visual.get("format_version") == "wam.multimodal.m1.visual_evaluation/1"
            and visual.get("formal_protocol") is True
            and visual.get("passed") is True
            and visual.get("phase") == "formal"
        ),
        "config_and_training_bound": bool(
            visual.get("config_sha256") == config_sha256
            and visual.get("training_summary_sha256") == training_sha256
        ),
        "record_counts_match": bool(
            _strict_integer(visual.get("records")) == record_count
            and _strict_integer(visual.get("expected_records")) == record_count
        ),
        "episode_records_path_matches": path_matches,
        "episode_records_sha256_matches": bool(
            _is_sha256(episode_records_sha256)
            and visual.get("episode_records_sha256") == episode_records_sha256
        ),
    }
    return _check(
        all(checks.values()),
        checks=checks,
        observed_records=record_count,
        declared_records=visual.get("records"),
        expected_records=visual.get("expected_records"),
        actual_episode_records_sha256=episode_records_sha256,
        declared_episode_records_sha256=visual.get("episode_records_sha256"),
    )


def _formal_future_probe_evidence_check(
    future: Mapping[str, Any],
    training: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    training_summary_path: Path,
    training_summary_sha256: str,
    train_seeds: Sequence[int],
) -> dict[str, Any]:
    """Recompute the robust probe instead of trusting report booleans."""

    seeds = tuple(int(value) for value in train_seeds)
    artifact_hashes = future.get("artifact_sha256")
    action_protocol = future.get("probe_action_protocol")
    position_target = future.get("position_probe_target")
    leakage = future.get("no_future_target_leakage")
    reported_per_seed = future.get("per_seed")
    reported_comparisons = future.get("comparisons")
    reported_clusters = future.get("cluster_evidence")
    checks: dict[str, bool] = {
        "format_and_formal_status": bool(
            future.get("format_version") == "wam.multimodal.m1.future_probe/1"
            and future.get("phase") == "M1"
            and future.get("formal_protocol") is True
            and future.get("passed") is True
            and future.get("claim_allowed") is True
            and _strict_integer(future.get("horizon")) == 8
        ),
        "exact_configured_train_seeds": bool(
            len(seeds) == 3
            and len(set(seeds)) == 3
            and _integer_tuple(future.get("train_seeds")) == seeds
            and _exact_seed_mapping(reported_per_seed, seeds)
        ),
        "config_manifest_and_training_hashes_bound": bool(
            isinstance(artifact_hashes, Mapping)
            and artifact_hashes.get("config") == config_sha256
            and artifact_hashes.get("dataset_manifest")
            == str(config["data"]["expected_manifest_sha256"])
            and artifact_hashes.get("visual_backbone")
            == str(config["initialization"]["expected_vision_weights_sha256"])
            and artifact_hashes.get("training_summary") == training_summary_sha256
            and future.get("training_summary_sha256") == training_summary_sha256
            and future.get("training_summary_stable_during_probe") is True
            and isinstance(future.get("training_summary"), str)
            and Path(str(future["training_summary"])).resolve()
            == training_summary_path.resolve()
        ),
        "checkpoint_evidence_matches_training": (
            _primary_checkpoint_evidence_matches_training(
                future.get("checkpoint_sha256"),
                future.get("strict_reload"),
                training,
                train_seeds=seeds,
            )
            and _json_equivalent(
                future.get("primary_checkpoint_tree_sha256"),
                future.get("checkpoint_sha256"),
            )
            and future.get("primary_checkpoint_trees_stable_during_probe") is True
        ),
        "exact_required_baselines": bool(
            future.get("baseline") == REQUIRED_PROBE_BASELINES[0]
            and _string_tuple(future.get("required_baselines"))
            == REQUIRED_PROBE_BASELINES
        ),
        "same_demonstration_action_protocol": bool(
            future.get("probe_action_source") == PROBE_ACTION_SOURCE
            and isinstance(action_protocol, Mapping)
            and action_protocol.get("mode")
            == "offline_teacher_forced_same_demonstration"
            and _strict_integer(action_protocol.get("action_chunk_horizon")) == 8
            and action_protocol.get("target_consistent_with_demonstration_h8_labels")
            is True
            and action_protocol.get(
                "matched_information_baseline_receives_identical_action_chunk"
            )
            is True
            and action_protocol.get("deployed_action_flow_used") is False
            and action_protocol.get("deployed_causal_claim_allowed") is False
        ),
        "position_target_semantics_are_not_overclaimed": bool(
            isinstance(position_target, Mapping)
            and position_target.get("field") == "h8_center_xy"
            and position_target.get("semantics") == "future_robot_carrier_center_xy"
            and position_target.get("carried_object_position_proxy") is True
            and position_target.get("explicit_visual_target_position") is False
            and position_target.get("unseen_target_generalization_claim_allowed")
            is False
        ),
        "future_target_contract": bool(
            isinstance(leakage, Mapping)
            and leakage.get("passed") is True
            and leakage.get("candidate_actions_source") == PROBE_ACTION_SOURCE
            and leakage.get("dataset_action_targets_read") is True
            and leakage.get("dataset_action_targets_forwarded_to_future_head") is True
            and leakage.get("matched_baseline_receives_identical_action_targets")
            is True
            and leakage.get("target_action_pairing") == "same_demonstration_window"
            and leakage.get("deployed_causal_claim_allowed") is False
            and leakage.get("future_images_forwarded") is False
            and leakage.get("future_states_forwarded") is False
            and leakage.get("probe_labels_forwarded") is False
            and _strict_integer(leakage.get("test_samples_used_for_fit_or_selection"))
            == 0
            and leakage.get("train_validation_test_sample_ids_disjoint") is True
            and leakage.get("manifest_hdf5_sha256_verified") is True
            and leakage.get("manifest_hdf5_contract_verified") is True
        ),
    }

    recomputed: Mapping[str, Any] | None = None
    recompute_error: str | None = None
    try:
        if not isinstance(reported_per_seed, Mapping):
            raise ValueError("per_seed probe evidence is not a mapping")
        recomputed = cluster_aware_probe_comparisons(
            reported_per_seed,
            train_seeds=seeds,
            confidence=float(config["statistics"]["confidence"]),
            bootstrap_samples=int(config["statistics"]["bootstrap_samples"]),
            bootstrap_seed=int(config["statistics"]["bootstrap_seed"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        recompute_error = f"{type(error).__name__}:{error}"

    recomputed_comparisons = (
        recomputed.get("comparisons") if isinstance(recomputed, Mapping) else None
    )
    recomputed_aggregate = (
        recomputed.get("aggregate") if isinstance(recomputed, Mapping) else None
    )
    recomputed_clusters = (
        recomputed.get("cluster_evidence") if isinstance(recomputed, Mapping) else None
    )
    reported_aggregate = {
        name: future.get(name)
        for name in (
            "object_model_errors",
            "object_baseline_errors",
            "object_action_baseline_errors",
            "event_model_predictions",
            "event_baseline_predictions",
            "event_action_baseline_predictions",
            "event_labels",
        )
    }
    checks.update(
        {
            "per_seed_and_cluster_recomputed": recomputed is not None,
            "reported_aggregate_matches_recomputation": _json_equivalent(
                reported_aggregate, recomputed_aggregate
            ),
            "reported_comparisons_match_recomputation": _json_equivalent(
                reported_comparisons, recomputed_comparisons
            ),
            "reported_cluster_evidence_matches_recomputation": _json_equivalent(
                reported_clusters, recomputed_clusters
            ),
            "recomputed_every_seed_and_cluster_pass": bool(
                isinstance(recomputed_comparisons, Mapping)
                and recomputed_comparisons.get("robust_improvement") is True
                and recomputed_comparisons.get("all_train_seeds_significantly_better")
                is True
                and recomputed_comparisons.get("object_significantly_better") is True
                and recomputed_comparisons.get("event_significantly_better") is True
                and recomputed_comparisons.get("formal_requires_every_train_seed")
                is True
                and recomputed_comparisons.get("formal_requires_both_baselines") is True
            ),
            "sample_id_hashes_are_exact_and_bound": _probe_sample_hashes_match(
                future, seeds=seeds, recomputed_clusters=recomputed_clusters
            ),
        }
    )
    return _check(
        all(checks.values()),
        checks=checks,
        recompute_error=recompute_error,
        recomputed_comparisons=recomputed_comparisons,
    )


def _probe_sample_hashes_match(
    future: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    recomputed_clusters: Any,
) -> bool:
    if not isinstance(recomputed_clusters, Mapping):
        return False
    object_hash = recomputed_clusters.get("object_sample_ids_sha256")
    event_hash = recomputed_clusters.get("event_sample_ids_sha256")
    if not _is_sha256(object_hash) or not _is_sha256(event_hash):
        return False
    per_seed = future.get("per_seed")
    selection = future.get("selection")
    test_selection = selection.get("test") if isinstance(selection, Mapping) else None
    if not isinstance(per_seed, Mapping) or not isinstance(test_selection, Mapping):
        return False
    if (
        test_selection.get("object_sample_ids_sha256") != object_hash
        or test_selection.get("event_sample_ids_sha256") != event_hash
    ):
        return False
    for seed in seeds:
        row = _seed_mapping_value(per_seed, int(seed))
        if not isinstance(row, Mapping):
            return False
        if (
            row.get("object_test_sample_ids_sha256") != object_hash
            or row.get("event_test_sample_ids_sha256") != event_hash
        ):
            return False
    return True


def _visual_protocol_check(
    visual: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    expected_seeds = tuple(
        range(
            int(evaluation["physical_seed_start"]),
            int(evaluation["physical_seed_start"])
            + int(evaluation["formal_physical_seeds_per_task"]),
        )
    )
    runtime = visual.get("runtime")
    deadline = (
        runtime.get("deadline_contract") if isinstance(runtime, Mapping) else None
    )
    hardware = runtime.get("hardware") if isinstance(runtime, Mapping) else None
    runtime_scope = (
        runtime.get("runtime_scope") if isinstance(runtime, Mapping) else None
    )
    replan = runtime.get("replan_contract") if isinstance(runtime, Mapping) else None
    resolved_device = (
        runtime.get("resolved_device") if isinstance(runtime, Mapping) else None
    )
    primary_clean = [
        row
        for row in records
        if isinstance(row, Mapping)
        and row.get("model_variant") == "state_vision_future"
        and row.get("intervention") == "clean"
    ]
    replan_fields = (
        "replan_events",
        "cold_replan_events",
        "warm_replan_events",
    )
    raw_replan_records_valid = bool(
        primary_clean
        and all(
            all(
                name in row and _strict_nonnegative_integer(row.get(name)) is not None
                for name in replan_fields
            )
            and (_strict_nonnegative_integer(row.get("replan_events")) or 0) > 0
            and (_strict_nonnegative_integer(row.get("steps")) or 0) > 0
            and row.get("replan_events") == (int(row["steps"]) + 1) // 2
            and row.get("warm_replan_events") == 0
            and row.get("cold_replan_events") == row.get("replan_events")
            for row in primary_clean
        )
    )
    observed_replans = (
        sum(int(row["replan_events"]) for row in primary_clean)
        if raw_replan_records_valid
        else -1
    )
    observed_cold = (
        sum(int(row["cold_replan_events"]) for row in primary_clean)
        if raw_replan_records_valid
        else -1
    )
    observed_warm = (
        sum(int(row["warm_replan_events"]) for row in primary_clean)
        if raw_replan_records_valid
        else -1
    )
    checks = {
        "canonical_evaluation_seeds": _integer_tuple(visual.get("evaluation_seeds"))
        == expected_seeds,
        "aggregation_passed": bool(
            isinstance(visual.get("aggregation"), Mapping)
            and visual["aggregation"].get("passed") is True
        ),
        "decimated_runtime_declared": bool(
            isinstance(runtime, Mapping)
            and runtime.get("decimated") is True
            and runtime.get("warmup_actions_excluded") is True
        ),
        "control_and_visual_rates_match": bool(
            isinstance(runtime, Mapping)
            and _exact_number(runtime.get("control_hz"), evaluation["control_hz"])
            and _exact_number(runtime.get("visual_hz"), evaluation["visual_refresh_hz"])
        ),
        "deadline_contract_matches": bool(
            isinstance(deadline, Mapping)
            and _exact_number(
                deadline.get("direct_without_vision_ms"),
                config["acceptance"]["maximum_sensor_to_action_p95_ms"],
            )
            and _exact_number(
                deadline.get("decimated_with_vision_ms"),
                config["acceptance"]["maximum_decimated_action_age_ms"],
            )
        ),
        "deadline_misses_reported": bool(
            isinstance(runtime, Mapping)
            and _strict_nonnegative_integer(runtime.get("deadline_misses")) is not None
        ),
        "runtime_concurrency_recorded": bool(
            isinstance(runtime, Mapping)
            and _strict_nonnegative_integer(runtime.get("worker_processes"))
            not in {None, 0}
            and _strict_nonnegative_integer(runtime.get("torch_threads_per_worker"))
            not in {None, 0}
        ),
        "resolved_device_and_hardware_recorded": bool(
            isinstance(resolved_device, str)
            and resolved_device
            and isinstance(hardware, Mapping)
            and isinstance(hardware.get("platform"), str)
            and bool(hardware.get("platform"))
            and isinstance(hardware.get("machine"), str)
            and bool(hardware.get("machine"))
            and isinstance(hardware.get("torch_version"), str)
            and bool(hardware.get("torch_version"))
            and _strict_nonnegative_integer(hardware.get("logical_cpu_count"))
            not in {None, 0}
            and (
                (
                    resolved_device.startswith("cpu")
                    and isinstance(hardware.get("cpu_model"), str)
                    and bool(hardware.get("cpu_model"))
                )
                or (
                    resolved_device.startswith("cuda")
                    and isinstance(hardware.get("accelerator_name"), str)
                    and bool(hardware.get("accelerator_name"))
                )
            )
        ),
        "latency_definitions_recorded": bool(
            isinstance(runtime, Mapping)
            and runtime.get("sensor_to_action_definition") == "policy_act_wall_time_ms"
            and isinstance(runtime.get("action_age_definition"), str)
            and bool(runtime.get("action_age_definition"))
        ),
        "runtime_scope_is_primary_clean": bool(
            isinstance(runtime_scope, Mapping)
            and runtime_scope.get("model_variant") == "state_vision_future"
            and runtime_scope.get("intervention") == "clean"
            and runtime_scope.get("all_configured_train_seeds") is True
            and runtime_scope.get("all_visual_required_tasks") is True
        ),
        "cold_replan_config_locked": bool(
            config["action_chunk"].get("replan_warm_start_enabled") is False
            and _exact_number(config["action_chunk"].get("warm_start_probability"), 0.0)
            and _exact_number(config["action_chunk"].get("warm_start_noise_std"), 0.0)
        ),
        "cold_replans_recomputed_from_episode_records": bool(
            raw_replan_records_valid
            and isinstance(replan, Mapping)
            and _strict_integer(replan.get("execution_steps")) == 2
            and replan.get("warm_start_enabled") is False
            and _exact_number(replan.get("training_warm_start_probability"), 0.0)
            and replan.get("observation_regrounding")
            == "cold_start_every_execute_2_replan"
            and replan.get("scope") == "m1_latent_flow_visual_required_only"
            and _strict_integer(replan.get("observed_replan_events"))
            == observed_replans
            and _strict_integer(replan.get("observed_cold_replan_events"))
            == observed_cold
            and _strict_integer(replan.get("observed_warm_replan_events"))
            == observed_warm
            and observed_replans == observed_cold
            and observed_warm == 0
        ),
    }
    return _check(all(checks.values()), checks=checks)


def _primary_checkpoint_evidence_matches_training(
    checkpoint_hashes: Any,
    strict_reload: Any,
    training: Mapping[str, Any],
    *,
    train_seeds: Sequence[int],
) -> bool:
    expected_hashes = training.get("checkpoint_sha256")
    if isinstance(expected_hashes, Mapping):
        expected_hashes = expected_hashes.get("state_vision_future")
    if not (
        _exact_seed_mapping(checkpoint_hashes, train_seeds)
        and _exact_seed_mapping(strict_reload, train_seeds)
        and _exact_seed_mapping(expected_hashes, train_seeds)
    ):
        return False
    return all(
        _seed_mapping_value(checkpoint_hashes, seed)
        == _seed_mapping_value(expected_hashes, seed)
        and _seed_mapping_value(strict_reload, seed) is True
        for seed in train_seeds
    )


def _legacy_checkpoint_evidence_matches_training(
    checkpoint_evidence: Any,
    training: Mapping[str, Any],
    *,
    train_seeds: Sequence[int],
    config: Mapping[str, Any] | None = None,
) -> bool:
    expected_hashes = training.get("checkpoint_sha256")
    if isinstance(expected_hashes, Mapping):
        expected_hashes = expected_hashes.get("state_vision_future")
    if not (
        _exact_seed_mapping(checkpoint_evidence, train_seeds)
        and _exact_seed_mapping(expected_hashes, train_seeds)
    ):
        return False
    allowed_formats = (
        allowed_checkpoint_formats(config)
        if config is not None
        else frozenset({LEGACY_CHECKPOINT_FORMAT_VERSION, CHECKPOINT_FORMAT_VERSION})
    )
    for train_seed in train_seeds:
        row = _seed_mapping_value(checkpoint_evidence, train_seed)
        if not isinstance(row, Mapping):
            return False
        if not (
            row.get("tree_sha256") == _seed_mapping_value(expected_hashes, train_seed)
            and _strict_integer(row.get("train_seed")) == train_seed
            and row.get("model_variant") == "state_vision_future"
            and row.get("strict_reload_passed") is True
            and row.get("embedded_legacy_matches_source") is True
            and row.get("schema_format_version") in allowed_formats
        ):
            return False
    return True


def _seed_mapping_value(value: Any, seed: int) -> Any:
    if not isinstance(value, Mapping):
        return None
    return value.get(str(seed), value.get(seed))


def _exact_number(first: Any, second: Any) -> bool:
    return bool(
        not isinstance(first, bool)
        and not isinstance(second, bool)
        and isinstance(first, (int, float))
        and isinstance(second, (int, float))
        and float(first) == float(second)
    )


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_equivalent(first: Any, second: Any) -> bool:
    try:
        return json.dumps(
            first,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) == json.dumps(
            second,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return False


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seed_value(value: Any, variant: str, seed: int) -> Any:
    if not isinstance(value, Mapping):
        return None
    seeds = value.get(variant)
    if not isinstance(seeds, Mapping):
        return None
    return seeds.get(str(seed), seeds.get(seed))


def _check(passed: Any, **details: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **details}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise ValueError(f"{path}:{line_number} is blank")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        result.append(value)
    if not result:
        raise ValueError(f"{path} contains no episode records")
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("M1 config must contain a mapping")
    return value


def _root_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
