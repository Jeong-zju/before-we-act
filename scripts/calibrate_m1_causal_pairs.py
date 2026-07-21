"""Diagnostically calibrate one M1 checkpoint on leakage-safe modality pairs."""

from __future__ import annotations

import argparse
import hashlib
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
from scripts.train_multimodal_wam import (  # noqa: E402
    _causal_pair_contract,
    _causal_pair_dataset,
    _causal_pair_loader,
    _causal_pair_weights,
    _dataset,
    _device,
    _flow_objective,
    _load_yaml,
    _loss_weights,
    _module_sha256,
    _sha256,
    _strict_reload_evidence,
    _state_causal_pair_contract,
    _state_causal_pair_dataset,
    _state_causal_pair_loader,
    _state_causal_pair_weights,
    _training_lineage,
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
    causal_pair_action_metrics,
    seed_everything,
    state_causal_pair_action_metrics,
    train_m1_stage,
)


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
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=24)
    parser.add_argument("--skip-hdf5-hashes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve(strict=True)
    config = _load_yaml(config_path)
    contract = _causal_pair_contract(config)
    state_contract = _state_causal_pair_contract(config)
    steps = int(args.steps or contract["calibration_steps"])
    learning_rate = float(args.learning_rate or contract["calibration_learning_rate"])
    if steps <= 0:
        raise ValueError("calibration steps must be positive")
    if not 0.0 < learning_rate <= 1e-3 or not math.isfinite(learning_rate):
        raise ValueError("calibration learning rate is invalid")
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(max(1, min(4, args.torch_threads)))
    device = _device(args.device)
    seed_everything(args.seed)

    input_checkpoint = args.input_checkpoint.resolve(strict=True)
    output_checkpoint_root = args.output_checkpoint_root.resolve()
    output_root = args.output_root.resolve()
    output_checkpoint = (
        output_checkpoint_root / "state_vision_future" / f"seed_{args.seed}"
    )
    if output_checkpoint.exists() and any(output_checkpoint.iterdir()):
        raise FileExistsError(
            f"refusing to mix stale calibration checkpoint files: {output_checkpoint}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

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
        raise ValueError("calibration input must be the state_vision_future variant")
    schema = metadata["schema"]
    if int(schema.get("train_seed", -1)) != int(args.seed):
        raise ValueError("calibration seed must match the input checkpoint")
    validate_loaded_checkpoint_vision(config, model, metadata)

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
    train_pairs = _causal_pair_dataset(manifest, "train", config)
    validation_pairs = _causal_pair_dataset(manifest, "validation", config)
    train_state_pairs = _state_causal_pair_dataset(manifest, "train", config)
    validation_state_pairs = _state_causal_pair_dataset(manifest, "validation", config)

    sampler = train_windows.make_weighted_sampler(
        num_samples=steps * int(config["training"]["batch_size"]),
        decision_window_boost=float(config["training"]["decision_window_boost"]),
        seed=args.seed,
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
    pair_loader = _causal_pair_loader(train_pairs, config, seed=args.seed, shuffle=True)
    validation_pair_loader = _causal_pair_loader(
        validation_pairs, config, seed=args.seed, shuffle=False
    )
    state_pair_loader = _state_causal_pair_loader(
        train_state_pairs, config, seed=args.seed, shuffle=True
    )
    validation_state_pair_loader = _state_causal_pair_loader(
        validation_state_pairs, config, seed=args.seed, shuffle=False
    )
    flow_objective = _flow_objective(config)

    frozen_before = _frozen_hashes(model, flow, legacy_world, legacy_flow)
    parameter_hashes_before = _parameter_hashes(model, flow)
    pair_before = causal_pair_action_metrics(
        model,
        flow,
        validation_pair_loader,
        device=device,
        flow_objective=flow_objective,
    )
    state_pair_before = state_causal_pair_action_metrics(
        model,
        flow,
        validation_state_pair_loader,
        device=device,
        flow_objective=flow_objective,
    )
    action_rmse_before = action_chunk_rmse(
        model,
        flow,
        validation_loader,
        device=device,
        solver_steps=flow_objective.solver_steps,
        max_batches=4,
        policy_fixed_action_dims=flow_objective.policy_fixed_action_dims,
    )

    stage = M1StageConfig(
        name="diagnostic_causal_pair_calibration",
        steps=steps,
        learning_rate=learning_rate,
        world_learning_rate=0.0,
        weight_decay=float(config["training"]["weight_decay"]),
        gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        train_visual_adapter=True,
        train_fusion=True,
        train_future_head=False,
        train_action_flow=True,
        train_world_model=False,
    )
    training = train_m1_stage(
        model,
        flow,
        train_loader,
        stage,
        device=device,
        weights=_loss_weights(config, future=False, world=False),
        flow_objective=flow_objective,
        causal_pair_batches=pair_loader,
        causal_pair_weights=_causal_pair_weights(config),
        state_causal_pair_batches=state_pair_loader,
        state_causal_pair_weights=_state_causal_pair_weights(config),
        seed=args.seed,
    )
    pair_after = causal_pair_action_metrics(
        model,
        flow,
        validation_pair_loader,
        device=device,
        flow_objective=flow_objective,
    )
    state_pair_after = state_causal_pair_action_metrics(
        model,
        flow,
        validation_state_pair_loader,
        device=device,
        flow_objective=flow_objective,
    )
    action_rmse_after = action_chunk_rmse(
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
    parent_tree_after = checkpoint_tree_sha256(input_checkpoint)
    pair_criteria = _pair_criteria(pair_before, pair_after)
    state_pair_criteria = _state_pair_criteria(state_pair_before, state_pair_after)
    diagnostic_criteria_met = bool(
        all(pair_criteria.values())
        and all(state_pair_criteria.values())
        and action_rmse_after <= 1.25 * action_rmse_before
        and frozen_before == frozen_after
        and allowed_changes["passed"]
        and parent_tree_before == parent_tree_after
    )

    parameter_counts = model.parameter_breakdown(flow)
    metrics = {
        "format_version": "wam.multimodal.m1.causal_calibration_run/1",
        "formal_protocol": False,
        "variant": "state_vision_future",
        "train_seed": int(args.seed),
        "parent_m1_checkpoint": str(input_checkpoint),
        "parent_m1_checkpoint_tree_sha256": parent_tree_before,
        "diagnostic_criteria_met": diagnostic_criteria_met,
        "causal_pair_contract": contract,
        "causal_pair_summary_sha256": train_pairs.pair_summary_sha256(),
        "causal_pair_metrics_before": pair_before,
        "causal_pair_metrics_after": pair_after,
        "pair_criteria": pair_criteria,
        "state_causal_pair_contract": state_contract,
        "state_causal_pair_summary_sha256": (train_state_pairs.pair_summary_sha256()),
        "state_causal_pair_metrics_before": state_pair_before,
        "state_causal_pair_metrics_after": state_pair_after,
        "state_pair_criteria": state_pair_criteria,
        "validation_action_chunk_rmse_before": action_rmse_before,
        "validation_action_chunk_rmse_after": action_rmse_after,
        "training": training,
        "parameter_counts": parameter_counts,
        "changed_parameters": changed_parameters,
        "allowed_parameter_changes": allowed_changes,
        "frozen_hashes_before": frozen_before,
        "frozen_hashes_after": frozen_after,
    }
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
        dataset_manifest=_training_lineage(
            train_windows, train_pairs, train_state_pairs, config
        ),
        metrics=metrics,
        provenance={
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "diagnostic_calibration": True,
            "parent_m1_checkpoint": str(input_checkpoint),
            "parent_m1_checkpoint_tree_sha256": parent_tree_before,
            "source_checkpoint": str(legacy_checkpoint),
            "source_checkpoint_tree_sha256": checkpoint_tree_sha256(legacy_checkpoint),
            "visual_source_sha256": model.vision_encoder.artifact_sha256,
        },
        schema_version=str(config["data"]["schema_version"]),
        train_seed=int(args.seed),
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
        "format_version": "wam.multimodal.m1.causal_calibration/1",
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
        "passed": report["passed"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "manifest_sha256": manifest.manifest_sha256,
        "visual_backbone": training_summary_vision_payload(config, project_root=ROOT),
        "visual_backbone_sha256": model.vision_encoder.artifact_sha256,
        "variants": ["state_vision_future"],
        "train_seeds": [int(args.seed)],
        "checkpoint_root": str(output_checkpoint_root),
        "reports": [report],
        "checkpoint_sha256": {"state_vision_future": {str(args.seed): output_tree}},
        "strict_reload": {"state_vision_future": {str(args.seed): strict_reload}},
    }
    _write_json(output_root / "training_summary.json", summary)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


def _parameter_hashes(model: torch.nn.Module, flow: torch.nn.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for family, module in (("model", model), ("flow", flow)):
        for name, tensor in module.state_dict().items():
            digest = hashlib.sha256(
                tensor.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            result[f"{family}.{name}"] = digest
    return result


def _changed_parameters(
    before: Mapping[str, str], after: Mapping[str, str]
) -> list[str]:
    if set(before) != set(after):
        raise RuntimeError("calibration parameter keys changed")
    return sorted(name for name in before if before[name] != after[name])


def _allowed_parameter_changes(changed: list[str]) -> dict[str, Any]:
    model_prefixes = (
        "model.resampler.",
        "model.fusion.",
        "model.task_embedding.",
    )
    flow_prefix = "flow."
    forbidden = [
        name
        for name in changed
        if not name.startswith((*model_prefixes, flow_prefix))
        or name.startswith("flow.anchor_prior.")
    ]
    required = {
        "resampler": any(name.startswith("model.resampler.") for name in changed),
        "fusion": any(name.startswith("model.fusion.") for name in changed),
        "non_anchor_flow": any(
            name.startswith("flow.") and not name.startswith("flow.anchor_prior.")
            for name in changed
        ),
    }
    return {
        "passed": not forbidden and all(required.values()),
        "forbidden": forbidden,
        "required_families_changed": required,
        "task_embedding_is_fusion_context": True,
    }


def _frozen_hashes(
    model: Any, flow: Any, legacy_world: Any, legacy_flow: Any
) -> dict[str, str | None]:
    return {
        "vision_encoder": _module_sha256(model.vision_encoder),
        "world_model": _module_sha256(model.world_model),
        "future_head": (
            None if model.future_head is None else _module_sha256(model.future_head)
        ),
        "flow_anchor": _module_sha256(flow.anchor_prior),
        "legacy_world": _module_sha256(legacy_world),
        "legacy_flow": _module_sha256(legacy_flow),
    }


def _pair_criteria(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, bool]:
    per_task = {
        task: bool(
            after["by_task_index"][task]["action_delta_rmse"]
            < before["by_task_index"][task]["action_delta_rmse"]
            and after["by_task_index"][task]["delta_cosine"] > 0.5
            and after["by_task_index"][task]["executed_prefix_sign_agreement"] >= 0.75
            and 0.25 <= after["by_task_index"][task]["delta_norm_ratio"] <= 1.75
        )
        for task in sorted(before["by_task_index"])
    }
    return {
        "overall_action_delta_rmse_improved": bool(
            after["action_delta_rmse"] < before["action_delta_rmse"]
        ),
        "overall_delta_cosine_above_half": bool(after["delta_cosine"] > 0.5),
        "overall_prefix_sign_agreement_75pct": bool(
            after["executed_prefix_sign_agreement"] >= 0.75
        ),
        "overall_delta_norm_ratio_bounded": bool(
            0.25 <= after["delta_norm_ratio"] <= 1.75
        ),
        "every_task_passed": all(per_task.values()),
        **{f"task_{task}_passed": passed for task, passed in per_task.items()},
    }


def _state_pair_criteria(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, bool]:
    per_task = {
        task: bool(
            after["by_task_index"][task]["action_delta_rmse"]
            < before["by_task_index"][task]["action_delta_rmse"]
            and after["by_task_index"][task]["step0_action_delta_rmse"]
            < before["by_task_index"][task]["step0_action_delta_rmse"]
            and after["by_task_index"][task]["delta_cosine"] > 0.5
            and after["by_task_index"][task]["step0_sign_agreement"] >= 0.75
            and 0.25 <= after["by_task_index"][task]["delta_norm_ratio"] <= 1.75
            and 0.25 <= after["by_task_index"][task]["step0_delta_norm_ratio"] <= 1.75
        )
        for task in sorted(before["by_task_index"])
    }
    return {
        "overall_action_delta_rmse_improved": bool(
            after["action_delta_rmse"] < before["action_delta_rmse"]
        ),
        "overall_step0_action_delta_rmse_improved": bool(
            after["step0_action_delta_rmse"] < before["step0_action_delta_rmse"]
        ),
        "overall_delta_cosine_above_half": bool(after["delta_cosine"] > 0.5),
        "overall_step0_sign_agreement_75pct": bool(
            after["step0_sign_agreement"] >= 0.75
        ),
        "overall_step0_delta_norm_ratio_bounded": bool(
            0.25 <= after["step0_delta_norm_ratio"] <= 1.75
        ),
        "overall_delta_norm_ratio_bounded": bool(
            0.25 <= after["delta_norm_ratio"] <= 1.75
        ),
        "every_task_passed": all(per_task.values()),
        **{f"task_{task}_passed": passed for task, passed in per_task.items()},
    }


if __name__ == "__main__":
    raise SystemExit(main())
