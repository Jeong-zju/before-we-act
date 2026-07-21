"""Diagnostically calibrate one M1 primary checkpoint on state-only pairs.

This entry point is deliberately separate from the RGB causal-pair calibrator.
It combines ordinary behavior cloning with state-identifying pairs while
freezing the complete latent WAM and the action-flow anchor.  Consequently the
only mutable family is the non-anchor action flow.  Outputs are diagnostic and
are refused under the canonical Phase M1 formal output roots.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.m1_vision_contract import (  # noqa: E402
    training_summary_vision_payload,
    validate_loaded_checkpoint_vision,
)
from scripts.calibrate_m1_causal_pairs import (  # noqa: E402
    _changed_parameters,
    _parameter_hashes,
    _state_pair_criteria,
)
from scripts.train_multimodal_wam import (  # noqa: E402
    _dataset,
    _device,
    _flow_objective,
    _load_yaml,
    _loss_weights,
    _module_sha256,
    _sha256,
    _state_causal_pair_contract,
    _state_causal_pair_dataset,
    _state_causal_pair_loader,
    _state_causal_pair_weights,
    _strict_reload_evidence,
    _write_json,
)
from train.m1_checkpointing import (  # noqa: E402
    checkpoint_tree_sha256,
    load_m1_checkpoint,
    save_m1_checkpoint,
)
from train.m1_manifest_dataset import M1ManifestIndex  # noqa: E402
from train.m1_training import (  # noqa: E402
    M1StageConfig,
    action_chunk_rmse,
    seed_everything,
    state_causal_pair_action_metrics,
    train_m1_stage,
)


REPORT_FORMAT = "wam.multimodal.m1.state_pair_calibration/1"
RUN_FORMAT = "wam.multimodal.m1.state_pair_calibration_run/1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_multimodal/m1_latent_wam_dinov3.yaml",
    )
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=24)
    parser.add_argument("--skip-hdf5-hashes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve(strict=True)
    config = _load_yaml(config_path)
    state_contract = _state_causal_pair_contract(config)
    steps = int(
        state_contract["calibration_steps"] if args.steps is None else args.steps
    )
    learning_rate = float(
        state_contract["calibration_learning_rate"]
        if args.learning_rate is None
        else args.learning_rate
    )
    _validate_controls(steps, learning_rate, args.torch_threads)

    input_checkpoint = args.input_checkpoint.resolve(strict=True)
    output_checkpoint_root = args.output_checkpoint_root.resolve()
    output_root = args.output_root.resolve()
    _validate_diagnostic_output_paths(
        input_checkpoint=input_checkpoint,
        output_checkpoint_root=output_checkpoint_root,
        output_root=output_root,
        config=config,
    )

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(max(1, min(4, args.torch_threads)))
    device = _device(args.device)
    parent_tree_before = checkpoint_tree_sha256(input_checkpoint)
    model, flow, legacy_world, legacy_flow, metadata = load_m1_checkpoint(
        input_checkpoint,
        device=device,
        expected_schema_version=str(config["data"]["schema_version"]),
    )
    if not (
        model.config.use_state
        and model.config.use_vision
        and model.config.capacity_control == "future_head"
    ):
        raise ValueError(
            "state-pair calibration input must be the state_vision_future primary"
        )
    train_seed = int(metadata["schema"].get("train_seed", -1))
    if train_seed < 0:
        raise ValueError("input checkpoint has an invalid train seed")
    seed_everything(train_seed)
    validate_loaded_checkpoint_vision(config, model, metadata)

    output_checkpoint = (
        output_checkpoint_root / "state_vision_future" / f"seed_{train_seed}"
    )
    _refuse_stale_output(output_checkpoint, output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = (ROOT / str(config["data"]["manifest"])).resolve()
    if _sha256(manifest_path) != str(config["data"]["expected_manifest_sha256"]):
        raise ValueError("canonical manifest changed before calibration")
    manifest = M1ManifestIndex.from_path(
        manifest_path,
        verify_hdf5_sha256=not args.skip_hdf5_hashes,
        verify_hdf5_contract=True,
    )
    train_windows = _dataset(manifest, "train", config)
    validation_windows = _dataset(manifest, "validation", config)
    train_state_pairs = _state_causal_pair_dataset(manifest, "train", config)
    validation_state_pairs = _state_causal_pair_dataset(manifest, "validation", config)

    sampler = train_windows.make_weighted_sampler(
        num_samples=steps * int(config["training"]["batch_size"]),
        decision_window_boost=float(config["training"]["decision_window_boost"]),
        seed=train_seed,
    )
    train_loader = DataLoader(
        train_windows,
        batch_size=int(config["training"]["batch_size"]),
        sampler=sampler,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_windows,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    state_pair_loader = _state_causal_pair_loader(
        train_state_pairs, config, seed=train_seed, shuffle=True
    )
    validation_state_pair_loader = _state_causal_pair_loader(
        validation_state_pairs, config, seed=train_seed, shuffle=False
    )
    flow_objective = _flow_objective(config)

    frozen_before = _frozen_hashes(model, flow, legacy_world, legacy_flow)
    parameter_hashes_before = _parameter_hashes(model, flow)
    state_pair_before = state_causal_pair_action_metrics(
        model,
        flow,
        validation_state_pair_loader,
        device=device,
        flow_objective=flow_objective,
    )
    ordinary_rmse_before = action_chunk_rmse(
        model,
        flow,
        validation_loader,
        device=device,
        solver_steps=flow_objective.solver_steps,
        max_batches=4,
        policy_fixed_action_dims=flow_objective.policy_fixed_action_dims,
    )

    stage = _state_only_stage(config, steps=steps, learning_rate=learning_rate)
    training = train_m1_stage(
        model,
        flow,
        train_loader,
        stage,
        device=device,
        weights=_loss_weights(config, future=False, world=False),
        flow_objective=flow_objective,
        causal_pair_batches=None,
        causal_pair_weights=None,
        state_causal_pair_batches=state_pair_loader,
        state_causal_pair_weights=_state_causal_pair_weights(config),
        seed=train_seed,
    )

    state_pair_after = state_causal_pair_action_metrics(
        model,
        flow,
        validation_state_pair_loader,
        device=device,
        flow_objective=flow_objective,
    )
    ordinary_rmse_after = action_chunk_rmse(
        model,
        flow,
        validation_loader,
        device=device,
        solver_steps=flow_objective.solver_steps,
        max_batches=4,
        policy_fixed_action_dims=flow_objective.policy_fixed_action_dims,
    )
    frozen_after = _frozen_hashes(model, flow, legacy_world, legacy_flow)
    changed_parameters = _changed_parameters(
        parameter_hashes_before, _parameter_hashes(model, flow)
    )
    allowed_changes = _allowed_parameter_changes(changed_parameters)
    training_scope = _training_scope_evidence(training)
    parent_tree_after = checkpoint_tree_sha256(input_checkpoint)
    state_pair_criteria = _state_pair_criteria(state_pair_before, state_pair_after)
    diagnostic_criteria_met = bool(
        all(state_pair_criteria.values())
        and ordinary_rmse_after <= 1.25 * ordinary_rmse_before
        and frozen_before == frozen_after
        and allowed_changes["passed"]
        and training_scope["passed"]
        and parent_tree_before == parent_tree_after
    )

    supervision = {
        "ordinary_behavior_cloning": True,
        "state_causal_pairs": True,
        "rgb_causal_pairs": False,
        "model_gradient_scope": "fully_frozen",
        "flow_gradient_scope": "non_anchor_only",
    }
    metrics = {
        "format_version": RUN_FORMAT,
        "formal_protocol": False,
        "diagnostic_only": True,
        "variant": "state_vision_future",
        "train_seed": train_seed,
        "parent_m1_checkpoint": str(input_checkpoint),
        "parent_m1_checkpoint_tree_sha256": parent_tree_before,
        "diagnostic_criteria_met": diagnostic_criteria_met,
        "supervision": supervision,
        "state_causal_pair_contract": state_contract,
        "state_causal_pair_summary_sha256": (train_state_pairs.pair_summary_sha256()),
        "state_causal_pair_metrics_before": state_pair_before,
        "state_causal_pair_metrics_after": state_pair_after,
        "state_pair_criteria": state_pair_criteria,
        "validation_action_chunk_rmse_before": ordinary_rmse_before,
        "validation_action_chunk_rmse_after": ordinary_rmse_after,
        "validation_action_chunk_rmse_max_batches": 4,
        "training": training,
        "training_scope": training_scope,
        "parameter_counts": model.parameter_breakdown(flow),
        "changed_parameters": changed_parameters,
        "allowed_parameter_changes": allowed_changes,
        "frozen_hashes_before": frozen_before,
        "frozen_hashes_after": frozen_after,
    }
    dataset_lineage = train_windows.checkpoint_lineage()
    dataset_lineage.update(
        {
            "diagnostic_state_pair_calibration": True,
            "rgb_causal_pair_supervision": False,
            "state_causal_pair_contract": state_contract,
            "state_causal_pair_summary_sha256": (
                train_state_pairs.pair_summary_sha256()
            ),
            "state_causal_pair_summary": train_state_pairs.pair_summary(),
        }
    )
    legacy_checkpoint = (
        ROOT / str(config["initialization"]["legacy_joint_wam_checkpoint"])
    ).resolve()
    save_m1_checkpoint(
        output_checkpoint,
        model,
        flow,
        legacy_world,
        legacy_flow,
        metadata["normalization"],
        experiment_config=config,
        dataset_manifest=dataset_lineage,
        metrics=metrics,
        provenance={
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "diagnostic_state_pair_calibration": True,
            "rgb_causal_pair_supervision": False,
            "parent_m1_checkpoint": str(input_checkpoint),
            "parent_m1_checkpoint_tree_sha256": parent_tree_before,
            "source_checkpoint": str(legacy_checkpoint),
            "source_checkpoint_tree_sha256": checkpoint_tree_sha256(legacy_checkpoint),
            "visual_source_sha256": model.vision_encoder.artifact_sha256,
        },
        schema_version=str(config["data"]["schema_version"]),
        train_seed=train_seed,
        model_variant="state_vision_future",
    )
    strict_reload = _strict_reload_evidence(
        output_checkpoint,
        model,
        flow,
        legacy_world,
        legacy_flow,
        device=device,
        schema_version=str(config["data"]["schema_version"]),
    )
    output_tree = checkpoint_tree_sha256(output_checkpoint)
    report = {
        **metrics,
        "format_version": REPORT_FORMAT,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "manifest_sha256": manifest.manifest_sha256,
        "parent_checkpoint_immutable": parent_tree_before == parent_tree_after,
        "parent_m1_checkpoint_tree_sha256_after": parent_tree_after,
        "checkpoint": str(output_checkpoint),
        "checkpoint_tree_sha256": output_tree,
        "strict_reload": strict_reload,
        "passed": bool(diagnostic_criteria_met and strict_reload["passed"]),
    }
    _write_json(output_root / "calibration_report.json", report)
    summary = {
        "format_version": "wam.multimodal.m1.training/1",
        "formal_protocol": False,
        "diagnostic_only": True,
        "diagnostic_kind": "state_pair_non_anchor_flow_calibration",
        "rgb_causal_pair_supervision": False,
        "passed": report["passed"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "manifest_sha256": manifest.manifest_sha256,
        "visual_backbone": training_summary_vision_payload(config, project_root=ROOT),
        "visual_backbone_sha256": model.vision_encoder.artifact_sha256,
        "variants": ["state_vision_future"],
        "train_seeds": [train_seed],
        "checkpoint_root": str(output_checkpoint_root),
        "reports": [report],
        "checkpoint_sha256": {"state_vision_future": {str(train_seed): output_tree}},
        "strict_reload": {"state_vision_future": {str(train_seed): strict_reload}},
    }
    _write_json(output_root / "training_summary.json", summary)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


def _validate_controls(steps: int, learning_rate: float, torch_threads: int) -> None:
    if steps <= 0:
        raise ValueError("calibration steps must be positive")
    if not 0.0 < learning_rate <= 1e-3 or not math.isfinite(learning_rate):
        raise ValueError("calibration learning rate is invalid")
    if int(torch_threads) <= 0:
        raise ValueError("torch-threads must be positive")


def _state_only_stage(
    config: Mapping[str, Any], *, steps: int, learning_rate: float
) -> M1StageConfig:
    return M1StageConfig(
        name="diagnostic_state_pair_flow_calibration",
        steps=steps,
        learning_rate=learning_rate,
        world_learning_rate=0.0,
        weight_decay=float(config["training"]["weight_decay"]),
        gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        train_visual_adapter=False,
        train_fusion=False,
        train_future_head=False,
        train_action_flow=True,
        train_world_model=False,
    )


def _frozen_hashes(
    model: torch.nn.Module,
    flow: Any,
    legacy_world: torch.nn.Module,
    legacy_flow: torch.nn.Module,
) -> dict[str, str]:
    return {
        "complete_model": _module_sha256(model),
        "vision_encoder": _module_sha256(model.vision_encoder),
        "world_model": _module_sha256(model.world_model),
        "future_head": _module_sha256(model.future_head),
        "flow_anchor": _module_sha256(flow.anchor_prior),
        "legacy_world": _module_sha256(legacy_world),
        "legacy_flow": _module_sha256(legacy_flow),
    }


def _allowed_parameter_changes(changed: list[str]) -> dict[str, Any]:
    forbidden = [
        name
        for name in changed
        if not name.startswith("flow.") or name.startswith("flow.anchor_prior.")
    ]
    non_anchor_flow_changed = any(
        name.startswith("flow.") and not name.startswith("flow.anchor_prior.")
        for name in changed
    )
    return {
        "passed": not forbidden and non_anchor_flow_changed,
        "forbidden": forbidden,
        "complete_model_changed": any(name.startswith("model.") for name in changed),
        "flow_anchor_changed": any(
            name.startswith("flow.anchor_prior.") for name in changed
        ),
        "non_anchor_flow_changed": non_anchor_flow_changed,
    }


def _training_scope_evidence(training: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "rgb_causal_pairs_enabled": training.get("causal_pairs_enabled"),
        "state_causal_pairs_enabled": training.get("state_causal_pairs_enabled"),
        "state_causal_pair_gradient_scope": training.get(
            "state_causal_pair_gradient_scope"
        ),
    }
    expected = {
        "rgb_causal_pairs_enabled": False,
        "state_causal_pairs_enabled": True,
        "state_causal_pair_gradient_scope": "non_anchor_flow_only_model_frozen",
    }
    groups = training.get("optimizer_groups")
    one_flow_group = bool(
        isinstance(groups, list)
        and len(groups) == 1
        and groups[0].get("role") == "adapter_action"
        and int(groups[0].get("parameters", 0)) > 0
    )
    return {
        "passed": observed == expected and one_flow_group,
        "observed": observed,
        "expected": expected,
        "one_nonempty_optimizer_group": one_flow_group,
    }


def _validate_diagnostic_output_paths(
    *,
    input_checkpoint: Path,
    output_checkpoint_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
) -> None:
    formal_checkpoint_root = (
        ROOT / str(config["training"]["checkpoint_root"])
    ).resolve()
    formal_report_root = (ROOT / str(config["training"]["report_root"])).resolve()
    for candidate in (output_checkpoint_root, output_root):
        if _overlaps(candidate, formal_checkpoint_root) or _overlaps(
            candidate, formal_report_root
        ):
            raise ValueError(
                "diagnostic calibration outputs must be outside formal M1 roots"
            )
        if _overlaps(candidate, input_checkpoint):
            raise ValueError("diagnostic output cannot overlap the parent checkpoint")
    if _overlaps(output_checkpoint_root, output_root):
        raise ValueError("checkpoint and report outputs must use independent roots")


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _refuse_stale_output(output_checkpoint: Path, output_root: Path) -> None:
    if output_checkpoint.exists() and any(output_checkpoint.iterdir()):
        raise FileExistsError(
            f"refusing to mix stale calibration checkpoint files: {output_checkpoint}"
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"refusing to mix stale calibration report files: {output_root}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
