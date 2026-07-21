"""Train the five Phase M1 latent-WAM contrasts with three fixed seeds."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wam_multimodal import (  # noqa: E402
    DINOV3_ENCODER_SPECS,
    FrozenDINOv3Config,
    FrozenDINOv3Encoder,
    FrozenResNet18Config,
    FrozenResNet18Encoder,
    LatentWAM,
    LatentWAMConfig,
    PerceiverResamplerConfig,
    canonical_json_sha256,
)
from train.joint_wam_checkpointing import load_joint_wam_checkpoint  # noqa: E402
from train.m1_checkpointing import (  # noqa: E402
    checkpoint_tree_sha256,
    load_m1_checkpoint,
    save_m1_checkpoint,
)
from train.m1_manifest_dataset import (  # noqa: E402
    M1CausalPairDataset,
    M1ManifestIndex,
    M1StateCausalPairDataset,
    M1WindowDataset,
)
from train.progress import TrainingProgress  # noqa: E402
from train.m1_training import (  # noqa: E402
    M1CausalPairWeights,
    M1FlowObjectiveConfig,
    M1LossWeights,
    M1StageConfig,
    M1StateCausalPairWeights,
    action_chunk_required_keys,
    action_chunk_rmse,
    causal_pair_action_metrics,
    m1_batch_required_keys,
    seed_everything,
    train_m1_stage,
)


INTERNAL_VARIANTS = (
    "state_only",
    "vision_only",
    "state_vision_no_future",
    "state_vision_future",
    "state_vision_param_matched_mlp",
)
CANONICAL_VARIANT = {
    "state_only": "state_only",
    "vision_only": "vision_only",
    "state_vision_no_future": "state_vision_no_future",
    "state_vision_future": "state_vision_future",
    "state_vision_param_matched_mlp": "parameter_matched_mlp",
}


@dataclass(frozen=True)
class _VisionArtifacts:
    encoder_name: str
    weights_path: Path
    weights_sha256: str
    config_path: Path | None = None
    config_sha256: str | None = None
    model_id: str | None = None
    revision: str | None = None
    preprocess_id: str | None = None
    input_size: int | None = None


class _RichProgress:
    """Adapt callback-style M1 loops to the repository's shared Rich display."""

    def __init__(self, description: str, *, show_loss_chart: bool) -> None:
        self.description = description
        self.show_loss_chart = show_loss_chart
        self.owner = TrainingProgress(enabled=True, total_stages=1)
        self.phase: Any | None = None
        self.completed = 0
        self.total = 0
        self.last_loss = math.nan
        self.closed = False

    def manifest(self, current: int, total: int) -> None:
        self._advance(current, total, loss=None)

    def training(self, current: int, total: int, loss: float) -> None:
        self._advance(current, total, loss=loss)

    def _advance(self, current: int, total: int, *, loss: float | None) -> None:
        if self.closed:
            return
        if self.phase is None:
            self.owner.__enter__()
            self.total = int(total)
            self.phase = self.owner.add_phase(
                self.description,
                self.total,
                show_loss_chart=self.show_loss_chart,
            )
        if int(total) != self.total or not self.completed <= int(current) <= self.total:
            raise RuntimeError("M1 progress callback received an invalid position")
        if loss is not None and math.isfinite(loss):
            self.last_loss = float(loss)
        for position in range(self.completed + 1, int(current) + 1):
            values: dict[str, Any]
            if self.show_loss_chart:
                values = {"step": position, "loss": self.last_loss}
            else:
                values = {"episode": position, "episodes": self.total}
            self.phase.advance(values)
        self.completed = int(current)
        if self.completed == self.total:
            detail = (
                f"{self.total} steps, loss {self.last_loss:.5f}"
                if self.show_loss_chart
                else f"{self.total} episodes verified"
            )
            self.phase.finish(detail)
            self.owner.__exit__(None, None, None)
            self.closed = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/wam_multimodal/m1_latent_wam_dinov3.yaml",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variants", nargs="+", choices=INTERNAL_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--steps-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-hdf5-hashes", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=24)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume the canonical formal run after fail-closed validation of "
            "existing preflight, reports, and checkpoint trees"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    variants = tuple(args.variants or _sequence(config["training"], "variants"))
    seeds = tuple(args.seeds or _sequence(config["training"], "seeds"))
    formal = _formal_request(args, config, variants=variants, seeds=seeds)
    if args.resume and not formal:
        raise ValueError("--resume is only valid for the canonical formal protocol")
    if args.steps_scale <= 0.0 or args.steps_scale > 1.0:
        raise ValueError("steps-scale must be in (0,1]")
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(max(1, min(4, args.torch_threads)))
    device = _device(args.device)
    checkpoint_root = (
        args.checkpoint_root or ROOT / str(config["training"]["checkpoint_root"])
    ).resolve()
    output_root = (
        args.output_root or ROOT / str(config["training"]["report_root"])
    ).resolve()
    if not formal and (args.checkpoint_root is None or args.output_root is None):
        raise ValueError(
            "diagnostic overrides require separate --checkpoint-root and --output-root"
        )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    source_checkpoint = (
        ROOT / str(config["initialization"]["legacy_joint_wam_checkpoint"])
    ).resolve()
    source_tree_before = checkpoint_tree_sha256(source_checkpoint)
    expected_source_tree = str(config["initialization"]["expected_legacy_tree_sha256"])
    if source_tree_before != expected_source_tree:
        raise ValueError("legacy Joint WAM tree hash differs from the M1 config")
    manifest_path = (ROOT / str(config["data"]["manifest"])).resolve()
    if _sha256(manifest_path) != str(config["data"]["expected_manifest_sha256"]):
        raise ValueError("canonical M0 manifest hash differs from the M1 config")
    vision_artifacts = _vision_artifacts(config)

    index_started = time.perf_counter()
    manifest = M1ManifestIndex.from_path(
        manifest_path,
        verify_hdf5_sha256=not args.skip_hdf5_hashes,
        verify_hdf5_contract=True,
        progress_callback=_RichProgress(
            "verify M1 HDF5/index", show_loss_chart=False
        ).manifest,
    )
    train_dataset = _dataset(manifest, "train", config)
    validation_dataset = _dataset(manifest, "validation", config)
    test_dataset = _dataset(manifest, "test", config)
    train_causal_pairs = _causal_pair_dataset(manifest, "train", config)
    validation_causal_pairs = _causal_pair_dataset(manifest, "validation", config)
    test_causal_pairs = _causal_pair_dataset(manifest, "test", config)
    data_evidence = {
        "index_seconds": time.perf_counter() - index_started,
        "input_pipeline": _input_pipeline_evidence(config, device),
        "train": train_dataset.window_summary(),
        "validation": validation_dataset.window_summary(),
        "test": test_dataset.window_summary(),
        "causal_pairs": {
            "contract": _causal_pair_contract(config),
            "train": train_causal_pairs.pair_summary(),
            "validation": validation_causal_pairs.pair_summary(),
            "test": test_causal_pairs.pair_summary(),
        },
        "formal_state_causal_pairs": {
            "enabled": False,
            "loaded": False,
            "trained": False,
            "rejection_reason": "gap1_not_runtime_replan_and_cold_only",
        },
        "lineage": {
            split: manifest.checkpoint_lineage(split)
            for split in ("train", "validation", "test")
        },
    }
    _write_json(output_root / "data_evidence.json", data_evidence)

    preflight_path = output_root / "preflight.json"
    preflight: dict[str, Any] = {"skipped": bool(args.skip_preflight), "passed": True}
    reused_preflight = bool(args.resume and preflight_path.is_file())
    if reused_preflight:
        preflight = _validated_resume_preflight(preflight_path, config)
        print(
            f"M1 resume reused formal preflight sha256={_sha256(preflight_path)}",
            flush=True,
        )
    elif not args.skip_preflight:
        preflight = _run_preflight(
            config,
            train_dataset,
            validation_dataset,
            train_causal_pairs,
            validation_causal_pairs,
            source_checkpoint=source_checkpoint,
            vision_artifacts=vision_artifacts,
            device=device,
            steps_scale=args.steps_scale,
        )
        _write_json(preflight_path, preflight)
        if not preflight["passed"]:
            raise RuntimeError(f"M1 training preflight failed: {preflight}")
    if args.preflight_only:
        return 0

    reports: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, dict[str, str]] = {
        CANONICAL_VARIANT[value]: {} for value in variants
    }
    strict_reloads: dict[str, dict[str, Any]] = {
        CANONICAL_VARIANT[value]: {} for value in variants
    }
    resumed_reports = 0
    for variant in variants:
        for train_seed in seeds:
            canonical = CANONICAL_VARIANT[variant]
            report_path = output_root / f"{canonical}_seed_{train_seed}.json"
            report = None
            if args.resume:
                report = _validated_resume_report(
                    report_path,
                    checkpoint_root=checkpoint_root,
                    config=config,
                    variant=variant,
                    train_seed=int(train_seed),
                )
            if report is None:
                report = _train_one(
                    config,
                    variant=variant,
                    train_seed=int(train_seed),
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    train_causal_pairs=train_causal_pairs,
                    validation_causal_pairs=validation_causal_pairs,
                    source_checkpoint=source_checkpoint,
                    vision_artifacts=vision_artifacts,
                    checkpoint_root=checkpoint_root,
                    device=device,
                    steps_scale=args.steps_scale,
                    formal=formal,
                    preflight=preflight,
                )
                _write_json(report_path, report)
                print(
                    f"M1 trained {canonical}/seed-{train_seed}: "
                    f"val_rmse={report['validation_action_chunk_rmse']:.5f}",
                    flush=True,
                )
            else:
                resumed_reports += 1
                print(
                    f"M1 resume verified {canonical}/seed-{train_seed}: "
                    f"tree={report['checkpoint_tree_sha256']}",
                    flush=True,
                )
            reports.append(report)
            checkpoint_hashes[canonical][str(train_seed)] = report[
                "checkpoint_tree_sha256"
            ]
            strict_reloads[canonical][str(train_seed)] = report["strict_reload"]

    source_tree_after = checkpoint_tree_sha256(source_checkpoint)
    summary = {
        "format_version": "wam.multimodal.m1.training/1",
        "formal_protocol": formal,
        "passed": bool(
            formal
            and preflight.get("passed") is True
            and source_tree_before == source_tree_after
            and len(reports) == len(INTERNAL_VARIANTS) * 3
            and all(item["strict_reload"]["passed"] for item in reports)
            and all(item["causal_pair_passed"] for item in reports)
        ),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config.resolve()),
        "manifest_sha256": manifest.manifest_sha256,
        "visual_backbone": {
            "encoder_name": vision_artifacts.encoder_name,
            "model_id": vision_artifacts.model_id,
            "revision": vision_artifacts.revision,
            "weights": str(vision_artifacts.weights_path),
            "weights_sha256": vision_artifacts.weights_sha256,
            "config": (
                None
                if vision_artifacts.config_path is None
                else str(vision_artifacts.config_path)
            ),
            "config_sha256": vision_artifacts.config_sha256,
            "preprocess_id": vision_artifacts.preprocess_id,
            "input_size": vision_artifacts.input_size,
        },
        # Kept as a scalar compatibility surface for acceptance tooling.
        "visual_backbone_sha256": vision_artifacts.weights_sha256,
        "input_pipeline": _input_pipeline_evidence(config, device),
        "legacy_checkpoint_tree_before": source_tree_before,
        "legacy_checkpoint_tree_after": source_tree_after,
        "source_checkpoint_immutable": source_tree_before == source_tree_after,
        "variants": [CANONICAL_VARIANT[value] for value in variants],
        "train_seeds": list(seeds),
        "checkpoint_root": str(checkpoint_root),
        "reports": reports,
        "checkpoint_sha256": checkpoint_hashes,
        "strict_reload": strict_reloads,
        "preflight": preflight,
        "resume": {
            "requested": bool(args.resume),
            "preflight_reused": reused_preflight,
            "reports_reused": resumed_reports,
            "reports_trained": len(reports) - resumed_reports,
        },
        "data_evidence_sha256": _sha256(output_root / "data_evidence.json"),
    }
    _write_json(output_root / "training_summary.json", summary)
    if formal and not summary["passed"]:
        raise RuntimeError("formal M1 training evidence is incomplete")
    return 0


def _run_preflight(
    config: Mapping[str, Any],
    train_dataset: M1WindowDataset,
    validation_dataset: M1WindowDataset,
    train_causal_pairs: M1CausalPairDataset,
    validation_causal_pairs: M1CausalPairDataset,
    *,
    source_checkpoint: Path,
    vision_artifacts: _VisionArtifacts,
    device: torch.device,
    steps_scale: float,
) -> dict[str, Any]:
    seed = int(config["training"]["seeds"][0])
    flow_objective = _flow_objective(config)
    overfit_count = min(int(config["training"]["overfit_samples"]), len(train_dataset))
    model, flow, _, _ = _build_model(
        config,
        variant="state_vision_future",
        source_checkpoint=source_checkpoint,
        vision_artifacts=vision_artifacts,
        device=device,
    )
    weights = _loss_weights(config, future=True, world=False)
    overfit_subset = Subset(
        train_dataset.project(m1_batch_required_keys(model, weights)),
        list(range(overfit_count)),
    )
    batch_size = min(int(config["training"]["batch_size"]), overfit_count)
    loader = DataLoader(
        overfit_subset,
        batch_size=batch_size,
        shuffle=True,
        **_data_loader_kwargs(config, device),
    )
    causal_pair_loader = _causal_pair_loader(
        train_causal_pairs,
        config,
        seed=seed,
        shuffle=True,
        device=device,
    )
    validation_pair_loader = _causal_pair_loader(
        validation_causal_pairs,
        config,
        seed=seed,
        shuffle=False,
        device=device,
    )
    validation_loader = DataLoader(
        Subset(
            validation_dataset.project(action_chunk_required_keys(model)),
            list(range(min(256, len(validation_dataset)))),
        ),
        batch_size=batch_size,
        shuffle=False,
        **_data_loader_kwargs(config, device, role="validation"),
    )
    initial_rmse = action_chunk_rmse(
        model,
        flow,
        validation_loader,
        device=device,
        solver_steps=flow_objective.solver_steps,
        max_batches=4,
        policy_fixed_action_dims=flow_objective.policy_fixed_action_dims,
    )
    initial_pair_metrics = causal_pair_action_metrics(
        model,
        flow,
        validation_pair_loader,
        device=device,
        flow_objective=flow_objective,
        max_batches=4,
    )
    overfit_steps = max(
        8, int(round(int(config["training"]["overfit_steps"]) * steps_scale))
    )
    overfit_stage = M1StageConfig(
        name="overfit256_joint_visual_action",
        steps=overfit_steps,
        learning_rate=3e-4,
        train_visual_adapter=True,
        train_fusion=True,
        train_future_head=True,
        train_action_flow=True,
        train_world_model=False,
    )
    overfit_train = train_m1_stage(
        model,
        flow,
        loader,
        overfit_stage,
        device=device,
        weights=weights,
        flow_objective=flow_objective,
        causal_pair_batches=causal_pair_loader,
        causal_pair_weights=_causal_pair_weights(config),
        state_causal_pair_batches=None,
        state_causal_pair_weights=None,
        seed=seed,
        progress_callback=_RichProgress(
            "train M1 preflight overfit-256", show_loss_chart=True
        ).training,
    )
    final_rmse = action_chunk_rmse(
        model,
        flow,
        validation_loader,
        device=device,
        solver_steps=flow_objective.solver_steps,
        max_batches=4,
        policy_fixed_action_dims=flow_objective.policy_fixed_action_dims,
    )
    final_pair_metrics = causal_pair_action_metrics(
        model,
        flow,
        validation_pair_loader,
        device=device,
        flow_objective=flow_objective,
        max_batches=4,
    )
    del loader

    one_percent_count = max(
        overfit_count,
        int(
            math.ceil(
                len(train_dataset) * float(config["training"]["one_percent_fraction"])
            )
        ),
    )
    model_1p, flow_1p, _, _ = _build_model(
        config,
        variant="state_vision_future",
        source_checkpoint=source_checkpoint,
        vision_artifacts=vision_artifacts,
        device=device,
    )
    one_percent_stages = _stage_configs(config, steps_scale=steps_scale)
    one_percent_weights = tuple(
        _loss_weights(
            config,
            future=stage.train_future_head,
            world=stage.train_world_model,
        )
        for stage in one_percent_stages
    )
    one_percent_keys = frozenset().union(
        *(m1_batch_required_keys(model_1p, value) for value in one_percent_weights)
    )
    one_percent_subset = Subset(
        train_dataset.project(one_percent_keys),
        list(range(min(one_percent_count, len(train_dataset)))),
    )
    one_loader = DataLoader(
        one_percent_subset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        **_data_loader_kwargs(config, device),
    )
    stage_reports = []
    for stage, stage_weights in zip(
        one_percent_stages, one_percent_weights, strict=True
    ):
        stage_reports.append(
            train_m1_stage(
                model_1p,
                flow_1p,
                one_loader,
                stage,
                device=device,
                weights=stage_weights,
                flow_objective=flow_objective,
                causal_pair_batches=causal_pair_loader,
                causal_pair_weights=_causal_pair_weights(config),
                state_causal_pair_batches=None,
                state_causal_pair_weights=None,
                seed=seed + len(stage_reports),
                progress_callback=_RichProgress(
                    f"train M1 preflight one-percent/{stage.name}",
                    show_loss_chart=True,
                ).training,
            )
        )
    one_percent_rmse = action_chunk_rmse(
        model_1p,
        flow_1p,
        validation_loader,
        device=device,
        solver_steps=flow_objective.solver_steps,
        max_batches=4,
        policy_fixed_action_dims=flow_objective.policy_fixed_action_dims,
    )
    causal_pair_passed = bool(
        final_pair_metrics["action_delta_rmse"]
        < initial_pair_metrics["action_delta_rmse"]
        and final_pair_metrics["delta_cosine"] > initial_pair_metrics["delta_cosine"]
    )
    overfit_passed = bool(
        math.isfinite(initial_rmse)
        and math.isfinite(final_rmse)
        and final_rmse < initial_rmse
        and overfit_train["last_total"] < overfit_train["first_total"]
        and causal_pair_passed
    )
    one_percent_passed = bool(
        math.isfinite(one_percent_rmse)
        and all(math.isfinite(stage["total"]) for stage in stage_reports)
    )
    return {
        "passed": overfit_passed and one_percent_passed,
        "overfit_256": {
            "passed": overfit_passed,
            "samples": overfit_count,
            "steps": overfit_steps,
            "initial_validation_action_rmse": initial_rmse,
            "final_validation_action_rmse": final_rmse,
            "training": overfit_train,
            "causal_pair_metrics_before": initial_pair_metrics,
            "causal_pair_metrics_after": final_pair_metrics,
            "causal_pair_passed": causal_pair_passed,
            "formal_state_causal_pairs_enabled": False,
        },
        "one_percent": {
            "passed": one_percent_passed,
            "samples": len(one_percent_subset),
            "fraction": len(one_percent_subset) / len(train_dataset),
            "validation_action_rmse": one_percent_rmse,
            "stages": stage_reports,
        },
    }


def _train_one(
    config: Mapping[str, Any],
    *,
    variant: str,
    train_seed: int,
    train_dataset: M1WindowDataset,
    validation_dataset: M1WindowDataset,
    train_causal_pairs: M1CausalPairDataset,
    validation_causal_pairs: M1CausalPairDataset,
    source_checkpoint: Path,
    vision_artifacts: _VisionArtifacts,
    checkpoint_root: Path,
    device: torch.device,
    steps_scale: float,
    formal: bool,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    seed_everything(train_seed)
    started = time.perf_counter()
    flow_objective = _flow_objective(config)
    model, flow, legacy_world, legacy_flow = _build_model(
        config,
        variant=variant,
        source_checkpoint=source_checkpoint,
        vision_artifacts=vision_artifacts,
        device=device,
    )
    initial_anchor_hash = _module_sha256(flow.anchor_prior)
    parameter_counts = model.parameter_breakdown(flow)
    causal_pair_loader = (
        _causal_pair_loader(
            train_causal_pairs,
            config,
            seed=train_seed,
            shuffle=True,
            device=device,
        )
        if model.config.use_vision
        else None
    )
    validation_pair_loader = (
        _causal_pair_loader(
            validation_causal_pairs,
            config,
            seed=train_seed,
            shuffle=False,
            device=device,
        )
        if model.config.use_vision
        else None
    )
    stages = _stage_configs(config, steps_scale=steps_scale)
    stage_weights = tuple(
        _loss_weights(
            config,
            future=(variant == "state_vision_future" and stage.train_future_head),
            world=stage.train_world_model,
        )
        for stage in stages
    )
    total_steps = sum(stage.steps for stage in stages)
    sampler = train_dataset.make_weighted_sampler(
        num_samples=max(
            int(config["training"]["batch_size"]),
            total_steps * int(config["training"]["batch_size"]),
        ),
        decision_window_boost=float(config["training"]["decision_window_boost"]),
        seed=train_seed,
    )
    train_loader = DataLoader(
        train_dataset.project(
            frozenset().union(
                *(m1_batch_required_keys(model, value) for value in stage_weights)
            )
        ),
        batch_size=int(config["training"]["batch_size"]),
        sampler=sampler,
        drop_last=True,
        **_data_loader_kwargs(config, device),
    )
    validation_loader = DataLoader(
        validation_dataset.project(action_chunk_required_keys(model)),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        **_data_loader_kwargs(config, device, role="validation"),
    )
    initial_rmse = action_chunk_rmse(
        model,
        flow,
        validation_loader,
        device=device,
        solver_steps=flow_objective.solver_steps,
        max_batches=4,
        policy_fixed_action_dims=flow_objective.policy_fixed_action_dims,
    )
    initial_pair_metrics = (
        causal_pair_action_metrics(
            model,
            flow,
            validation_pair_loader,
            device=device,
            flow_objective=flow_objective,
        )
        if validation_pair_loader is not None
        else None
    )
    stage_reports: list[dict[str, Any]] = []
    for stage_index, (stage, weights) in enumerate(
        zip(stages, stage_weights, strict=True)
    ):
        print(
            f"M1 training {CANONICAL_VARIANT[variant]}/seed-{train_seed} "
            f"stage={stage.name} steps={stage.steps}",
            flush=True,
        )
        stage_reports.append(
            train_m1_stage(
                model,
                flow,
                train_loader,
                stage,
                device=device,
                weights=weights,
                flow_objective=flow_objective,
                causal_pair_batches=causal_pair_loader,
                causal_pair_weights=(
                    _causal_pair_weights(config)
                    if causal_pair_loader is not None
                    else None
                ),
                state_causal_pair_batches=None,
                state_causal_pair_weights=None,
                seed=train_seed * 10 + stage_index,
                progress_callback=_RichProgress(
                    f"train M1 {CANONICAL_VARIANT[variant]}/seed-{train_seed}/"
                    f"{stage.name}",
                    show_loss_chart=True,
                ).training,
            )
        )
        print(
            f"M1 finished {CANONICAL_VARIANT[variant]}/seed-{train_seed} "
            f"stage={stage.name} loss={stage_reports[-1]['total']:.6f}",
            flush=True,
        )
    validation_rmse = action_chunk_rmse(
        model,
        flow,
        validation_loader,
        device=device,
        solver_steps=flow_objective.solver_steps,
        max_batches=8,
        policy_fixed_action_dims=flow_objective.policy_fixed_action_dims,
    )
    final_pair_metrics = (
        causal_pair_action_metrics(
            model,
            flow,
            validation_pair_loader,
            device=device,
            flow_objective=flow_objective,
        )
        if validation_pair_loader is not None
        else None
    )
    causal_pair_passed = (
        _causal_pair_quality_passed(initial_pair_metrics, final_pair_metrics)
        if model.config.use_vision
        else initial_pair_metrics is None and final_pair_metrics is None
    )
    anchor_hash_after = _module_sha256(flow.anchor_prior)
    if anchor_hash_after != initial_anchor_hash:
        raise RuntimeError("M1 training mutated the frozen action prior anchor")
    canonical = CANONICAL_VARIANT[variant]
    directory = checkpoint_root / canonical / f"seed_{train_seed}"
    metrics = {
        "format_version": "wam.multimodal.m1.training_run/1",
        "formal_protocol": formal,
        "variant": canonical,
        "train_seed": train_seed,
        "initial_validation_action_chunk_rmse": initial_rmse,
        "validation_action_chunk_rmse": validation_rmse,
        "stages": stage_reports,
        "parameter_counts": parameter_counts,
        "frozen_anchor_sha256_before": initial_anchor_hash,
        "frozen_anchor_sha256_after": anchor_hash_after,
        "preflight_passed": preflight.get("passed") is True,
        "causal_pair_contract": _causal_pair_contract(config),
        "causal_pair_summary_sha256": train_causal_pairs.pair_summary_sha256(),
        "causal_pair_metrics_before": initial_pair_metrics,
        "causal_pair_metrics_after": final_pair_metrics,
        "causal_pair_passed": causal_pair_passed,
        "formal_state_causal_pairs_enabled": False,
    }
    save_m1_checkpoint(
        directory,
        model,
        flow,
        legacy_world,
        legacy_flow,
        _normalization(source_checkpoint),
        experiment_config=config,
        dataset_manifest=_training_lineage(
            train_dataset,
            train_causal_pairs,
            config,
        ),
        metrics=metrics,
        provenance={
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_tree_sha256": checkpoint_tree_sha256(source_checkpoint),
            "visual_source": str(vision_artifacts.weights_path),
            "visual_source_sha256": vision_artifacts.weights_sha256,
            "visual_encoder_name": vision_artifacts.encoder_name,
            "visual_model_id": vision_artifacts.model_id,
            "visual_revision": vision_artifacts.revision,
            "visual_config_source": (
                None
                if vision_artifacts.config_path is None
                else str(vision_artifacts.config_path)
            ),
            "visual_config_sha256": vision_artifacts.config_sha256,
            "visual_preprocess_id": vision_artifacts.preprocess_id,
            "visual_encoder_input_size": vision_artifacts.input_size,
        },
        schema_version=str(config["data"]["schema_version"]),
        train_seed=train_seed,
        model_variant=canonical,
    )
    strict = _strict_reload_evidence(
        directory,
        model,
        flow,
        legacy_world,
        legacy_flow,
        device=device,
        schema_version=str(config["data"]["schema_version"]),
    )
    if not strict["passed"]:
        raise RuntimeError(f"M1 strict checkpoint reload failed: {strict}")
    return {
        **metrics,
        "checkpoint": str(directory),
        "checkpoint_tree_sha256": checkpoint_tree_sha256(directory),
        "strict_reload": strict,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _build_model(
    config: Mapping[str, Any],
    *,
    variant: str,
    source_checkpoint: Path,
    vision_artifacts: _VisionArtifacts,
    device: torch.device,
) -> tuple[LatentWAM, Any, Any, Any]:
    world, flow, _ = load_joint_wam_checkpoint(
        source_checkpoint,
        device=device,
        expected_schema_version="wam.proprio/1.0",
    )
    legacy_world = deepcopy(world).eval()
    legacy_flow = deepcopy(flow).eval()
    for module in (legacy_world, legacy_flow):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    use_state = variant != "vision_only"
    use_vision = variant != "state_only"
    capacity = (
        "future_head"
        if variant == "state_vision_future"
        else "action_mlp"
        if variant == "state_vision_param_matched_mlp"
        else "none"
    )
    model_cfg = config["model"]
    resampler = PerceiverResamplerConfig(
        input_dim=int(model_cfg["vision_patch_dim"]),
        width=int(model_cfg["resampler_width"]),
        num_latents=int(model_cfg["resampler_tokens"]),
        num_layers=int(model_cfg["resampler_layers"]),
        num_heads=int(model_cfg["resampler_heads"]),
        mlp_ratio=int(model_cfg["resampler_mlp_ratio"]),
        dropout=float(model_cfg["dropout"]),
        raw_patch_grid=int(model_cfg["raw_patch_grid"]),
        raw_patch_hidden_dim=int(model_cfg["raw_patch_hidden_dim"]),
        raw_shortcut_hidden_dim=int(model_cfg["raw_shortcut_hidden_dim"]),
    )
    latent_config = LatentWAMConfig(
        task_vocabulary=tuple(str(value) for value in model_cfg["task_vocabulary"]),
        use_state=use_state,
        use_vision=use_vision,
        capacity_control=capacity,
        action_dim=int(config["data"]["action_dim"]),
        future_latent_dim=int(model_cfg["future_latent_dim"]),
        visual_skip_initial_scale=float(model_cfg["visual_skip_initial_scale"]),
        resampler=resampler,
    )
    vision = _build_vision_encoder(config, vision_artifacts).to(device)
    model = LatentWAM(latent_config, world, vision).to(device)
    return model, flow, legacy_world, legacy_flow


def _vision_artifacts(config: Mapping[str, Any]) -> _VisionArtifacts:
    initialization = _mapping(config, "initialization")
    encoder_name = str(initialization["vision_backbone"])
    weights_path = _root_or_absolute(initialization["vision_weights"])
    expected = str(initialization["expected_vision_weights_sha256"])
    if _sha256(weights_path) != expected:
        raise ValueError(
            f"frozen {encoder_name} source hash differs from the M1 config"
        )
    config_path: Path | None = None
    config_sha256: str | None = None
    if encoder_name in DINOV3_ENCODER_SPECS:
        config_path = _root_or_absolute(initialization["vision_config"])
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw_config, Mapping):
            raise ValueError("DINOv3 config.json root must be a mapping")
        config_sha256 = canonical_json_sha256(raw_config)
        if config_sha256 != str(initialization["expected_vision_config_sha256"]):
            raise ValueError("DINOv3 config identity differs from the M1 config")
    elif encoder_name != "resnet18_imagenet1k_v1":
        raise ValueError(f"unsupported M1 vision_backbone {encoder_name!r}")
    return _VisionArtifacts(
        encoder_name=encoder_name,
        weights_path=weights_path,
        weights_sha256=expected,
        config_path=config_path,
        config_sha256=config_sha256,
        model_id=(
            str(initialization["vision_model_id"])
            if encoder_name in DINOV3_ENCODER_SPECS
            else "torchvision/resnet18-imagenet1k-v1"
        ),
        revision=(
            str(initialization["vision_revision"])
            if encoder_name in DINOV3_ENCODER_SPECS
            else "f37072fd"
        ),
        preprocess_id=str(initialization["vision_preprocess"]),
        input_size=(
            int(_mapping(config, "model")["vision_encoder_input_size"])
            if encoder_name in DINOV3_ENCODER_SPECS
            else int(_mapping(config, "model")["vision_input_size"])
        ),
    )


def _build_vision_encoder(
    config: Mapping[str, Any], artifacts: _VisionArtifacts
) -> torch.nn.Module:
    initialization = _mapping(config, "initialization")
    model_cfg = _mapping(config, "model")
    if artifacts.encoder_name in DINOV3_ENCODER_SPECS:
        if artifacts.config_path is None:
            raise ValueError("DINOv3 requires a local config.json artifact")
        return FrozenDINOv3Encoder(
            FrozenDINOv3Config(
                encoder_name=artifacts.encoder_name,
                model_id=str(initialization["vision_model_id"]),
                revision=str(initialization["vision_revision"]),
                config_path=artifacts.config_path,
                weights_path=artifacts.weights_path,
                expected_weights_sha256=artifacts.weights_sha256,
                expected_config_sha256=str(
                    initialization["expected_vision_config_sha256"]
                ),
                input_size=int(model_cfg["vision_encoder_input_size"]),
                preprocess_id=str(initialization["vision_preprocess"]),
                inference_batch_size=int(model_cfg["vision_encoder_batch_size"]),
            )
        )
    return FrozenResNet18Encoder(
        FrozenResNet18Config(
            weights_path=artifacts.weights_path,
            expected_sha256=artifacts.weights_sha256,
            resize_shorter_side=int(model_cfg["vision_input_size"]),
            crop_size=int(model_cfg["vision_input_size"]),
        )
    )


def _root_or_absolute(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"M1 config {key!r} must be a mapping")
    return dict(result)


def _stage_configs(
    config: Mapping[str, Any], *, steps_scale: float
) -> tuple[M1StageConfig, ...]:
    result = []
    for name in ("stage_1", "stage_2", "stage_3"):
        value = config["training"][name]
        result.append(
            M1StageConfig(
                name=str(value["name"]),
                steps=max(1, int(round(int(value["steps"]) * steps_scale))),
                learning_rate=float(value["learning_rate"]),
                world_learning_rate=float(value["world_learning_rate"]),
                weight_decay=float(config["training"]["weight_decay"]),
                gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
                train_visual_adapter=bool(value["train_visual_adapter"]),
                train_fusion=bool(value["train_fusion"]),
                train_future_head=bool(value["train_future_head"]),
                train_action_flow=bool(value["train_action_flow"]),
                train_world_model=bool(value["train_world_model"]),
            )
        )
    return tuple(result)


def _loss_weights(
    config: Mapping[str, Any], *, future: bool, world: bool
) -> M1LossWeights:
    values = config["training"]["losses"]
    return M1LossWeights(
        flow_matching=float(values["flow_matching"]),
        action_endpoint=float(values["action_endpoint"]),
        action_smoothness=float(values["action_smoothness"]),
        future_visual_latent=float(values["future_visual_latent"]) if future else 0.0,
        future_state=float(values["future_state"]) if world else 0.0,
    )


def _flow_objective(config: Mapping[str, Any]) -> M1FlowObjectiveConfig:
    chunk = config["action_chunk"]
    return M1FlowObjectiveConfig(
        execution_steps=int(chunk["execution_steps"]),
        solver_steps=int(chunk["solver_steps"]),
        solver=str(chunk["solver"]),
        normalized_action_clip=float(chunk["normalized_action_clip"]),
        warm_start_probability=float(chunk["warm_start_probability"]),
        warm_start_noise_std=float(chunk["warm_start_noise_std"]),
        policy_fixed_action_dims=tuple(
            int(value) for value in chunk["policy_fixed_action_dims"]
        ),
        executed_prefix_weight=float(chunk["executed_prefix_weight"]),
    )


def _dataset(
    manifest: M1ManifestIndex, split: str, config: Mapping[str, Any]
) -> M1WindowDataset:
    data = config["data"]
    return M1WindowDataset(
        manifest,
        split=split,
        state_history=int(data["state_history"]),
        action_chunk=int(data["action_horizon"]),
        cameras=tuple(str(value) for value in data["camera_order"]),
        visual_history=int(data["visual_history_frames"]),
        future_horizons=tuple(int(value) for value in data["future_visual_horizons"]),
    )


def _causal_pair_dataset(
    manifest: M1ManifestIndex, split: str, config: Mapping[str, Any]
) -> M1CausalPairDataset:
    data = config["data"]
    _ = _causal_pair_contract(config)
    return M1CausalPairDataset(
        manifest,
        split=split,
        state_history=int(data["state_history"]),
        action_chunk=int(data["action_horizon"]),
        cameras=tuple(str(value) for value in data["camera_order"]),
        visual_history=int(data["visual_history_frames"]),
    )


def _state_causal_pair_dataset(
    manifest: M1ManifestIndex, split: str, config: Mapping[str, Any]
) -> M1StateCausalPairDataset:
    data = config["data"]
    contract = _state_causal_pair_contract(config)
    return M1StateCausalPairDataset(
        manifest,
        split=split,
        state_history=int(data["state_history"]),
        action_chunk=int(data["action_horizon"]),
        cameras=tuple(str(value) for value in data["camera_order"]),
        visual_history=int(contract["visual_history_frames"]),
        valid_state_steps=int(contract["state_valid_steps"]),
        decision_gap=int(contract["decision_gap"]),
        step0_min_action_delta=float(contract["step0_min_action_delta"]),
        feedback_equality_atol=float(contract["feedback_equality_atol"]),
    )


def _causal_pair_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config["training"].get("causal_pairs")
    if not isinstance(raw, Mapping):
        raise ValueError("M1 training requires a causal_pairs mapping")
    expected_literals = {
        "enabled": True,
        "contract_version": "wam.multimodal.m1.causal_pairs/1",
        "construction": ("first_equal_history_rgb_different_execute2_action_delta_v2"),
        "apply_to_vision_variants": True,
        "filter_unobservable_conflicts": True,
        "gradient_scope": "visual_adapter_fusion_only",
    }
    observed_literals = {name: raw.get(name) for name in expected_literals}
    if observed_literals != expected_literals:
        raise ValueError(
            "M1 causal-pair construction/variant/conflict contract changed"
        )
    pair_batch_size = int(raw.get("pair_batch_size", 0))
    calibration_steps = int(raw.get("calibration_steps", 0))
    calibration_learning_rate = float(raw.get("calibration_learning_rate", 0.0))
    if pair_batch_size <= 0 or calibration_steps <= 0:
        raise ValueError(
            "M1 causal-pair batch size and calibration steps must be positive"
        )
    if not 0.0 < calibration_learning_rate <= 1e-3:
        raise ValueError("M1 causal-pair calibration learning rate is invalid")
    weights = _causal_pair_weights(config)
    if weights != M1CausalPairWeights():
        raise ValueError("canonical M1 causal-pair loss weights changed")
    return {
        **expected_literals,
        "pair_batch_size": pair_batch_size,
        "weights": {
            "factual_endpoint": weights.factual_endpoint,
            "action_delta": weights.action_delta,
            "delta_direction": weights.delta_direction,
            "executed_prefix_weight": weights.executed_prefix_weight,
        },
        "calibration_steps": calibration_steps,
        "calibration_learning_rate": calibration_learning_rate,
        "state_only_pair_supervision": "disabled_no_vision",
    }


def _causal_pair_weights(config: Mapping[str, Any]) -> M1CausalPairWeights:
    raw = config["training"].get("causal_pairs")
    if not isinstance(raw, Mapping):
        raise ValueError("M1 training requires a causal_pairs mapping")
    return M1CausalPairWeights(
        factual_endpoint=float(raw["factual_endpoint"]),
        action_delta=float(raw["action_delta"]),
        delta_direction=float(raw["delta_direction"]),
        executed_prefix_weight=float(raw["executed_prefix_weight"]),
    )


def _state_causal_pair_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config["training"].get("state_causal_pairs")
    if not isinstance(raw, Mapping):
        raise ValueError("M1 training requires a state_causal_pairs mapping")
    expected_literals = {
        "enabled": True,
        "diagnostic_only": True,
        "formal_enabled": False,
        "rejection_reason": "gap1_not_runtime_replan_and_cold_only",
        "contract_version": "wam.multimodal.m1.state_causal_pairs/1",
        "construction": "adjacent_equal_rgb_past_state_feedback_execute2_v1",
        "apply_to_state_variants": True,
        "vision_only_pair_supervision": "disabled_no_state",
        "gradient_scope": "non_anchor_flow_only_model_frozen",
        "decision_gap": 1,
        "state_valid_steps": 4,
        "past_action_valid_steps": 3,
        "visual_history_frames": 2,
        "feedback_action_dims": [0, 4],
        "zero_delta_action_dims": [1, 2, 5, 6],
        "feedback_state_position_dims": [0, 11],
    }
    observed = {name: raw.get(name) for name in expected_literals}
    if observed != expected_literals:
        raise ValueError("M1 state causal-pair selector contract changed")
    step0_min = float(raw.get("step0_min_action_delta", 0.0))
    equality_atol = float(raw.get("feedback_equality_atol", -1.0))
    pair_batch_size = int(raw.get("pair_batch_size", 0))
    calibration_steps = int(raw.get("calibration_steps", 0))
    calibration_learning_rate = float(raw.get("calibration_learning_rate", 0.0))
    if step0_min != 1e-3 or equality_atol != 1e-7:
        raise ValueError("M1 state causal-pair feedback thresholds changed")
    if pair_batch_size <= 0 or calibration_steps <= 0:
        raise ValueError(
            "M1 state causal-pair batch size and calibration steps must be positive"
        )
    if not 0.0 < calibration_learning_rate <= 1e-3:
        raise ValueError("M1 state causal-pair calibration learning rate is invalid")
    weights = _state_causal_pair_weights(config)
    if weights != M1StateCausalPairWeights():
        raise ValueError("canonical M1 state causal-pair loss weights changed")
    return {
        **expected_literals,
        "step0_min_action_delta": step0_min,
        "feedback_equality_atol": equality_atol,
        "pair_batch_size": pair_batch_size,
        "weights": {
            "factual_endpoint": weights.factual_endpoint,
            "action_delta": weights.action_delta,
            "delta_direction": weights.delta_direction,
            "target_rms_floor": weights.target_rms_floor,
        },
        "calibration_steps": calibration_steps,
        "calibration_learning_rate": calibration_learning_rate,
    }


def _state_causal_pair_weights(
    config: Mapping[str, Any],
) -> M1StateCausalPairWeights:
    raw = config["training"].get("state_causal_pairs")
    if not isinstance(raw, Mapping):
        raise ValueError("M1 training requires a state_causal_pairs mapping")
    return M1StateCausalPairWeights(
        factual_endpoint=float(raw["factual_endpoint"]),
        action_delta=float(raw["action_delta"]),
        delta_direction=float(raw["delta_direction"]),
        target_rms_floor=float(raw["target_rms_floor"]),
    )


def _causal_pair_quality_passed(before: Any, after: Any) -> bool:
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return False
    before_tasks = before.get("by_task_index")
    after_tasks = after.get("by_task_index")
    if not isinstance(before_tasks, Mapping) or not isinstance(after_tasks, Mapping):
        return False

    def improved(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
        return bool(
            float(new["action_delta_rmse"]) < float(old["action_delta_rmse"])
            and float(new["delta_cosine"]) > 0.5
            and float(new["executed_prefix_sign_agreement"]) >= 0.75
            and 0.25 <= float(new["delta_norm_ratio"]) <= 1.75
        )

    return bool(
        improved(before, after)
        and set(before_tasks) == {"0", "1", "2"}
        and set(after_tasks) == {"0", "1", "2"}
        and all(
            improved(before_tasks[task], after_tasks[task]) for task in before_tasks
        )
    )


def _state_causal_pair_quality_passed(before: Any, after: Any) -> bool:
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return False
    before_tasks = before.get("by_task_index")
    after_tasks = after.get("by_task_index")
    if not isinstance(before_tasks, Mapping) or not isinstance(after_tasks, Mapping):
        return False

    def improved(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
        return bool(
            float(new["action_delta_rmse"]) < float(old["action_delta_rmse"])
            and float(new["step0_action_delta_rmse"])
            < float(old["step0_action_delta_rmse"])
            and float(new["delta_cosine"]) > 0.5
            and float(new["step0_sign_agreement"]) >= 0.75
            and 0.25 <= float(new["delta_norm_ratio"]) <= 1.75
            and 0.25 <= float(new["step0_delta_norm_ratio"]) <= 1.75
        )

    return bool(
        improved(before, after)
        and set(before_tasks) == {"1", "2"}
        and set(after_tasks) == {"1", "2"}
        and all(
            improved(before_tasks[task], after_tasks[task]) for task in before_tasks
        )
    )


def _data_loader_kwargs(
    config: Mapping[str, Any],
    device: torch.device,
    *,
    role: str = "train",
) -> dict[str, Any]:
    """Return the common worker/pinning policy for every M1 DataLoader."""
    training = config["training"]
    worker_key = {
        "train": "num_workers",
        "validation": "validation_num_workers",
        "pair": "pair_num_workers",
    }.get(role)
    if worker_key is None:
        raise ValueError(f"unknown M1 DataLoader role {role!r}")
    num_workers = int(training.get(worker_key, training["num_workers"]))
    if num_workers < 0:
        raise ValueError("M1 training num_workers must be non-negative")

    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers == 0:
        return kwargs

    prefetch_factor = int(training["prefetch_factor"])
    if prefetch_factor <= 0:
        raise ValueError("M1 training prefetch_factor must be positive")
    persistent_workers = training["persistent_workers"]
    if not isinstance(persistent_workers, bool):
        raise ValueError("M1 training persistent_workers must be a boolean")
    kwargs.update(
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    return kwargs


def _input_pipeline_evidence(
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Record throughput-only execution knobs alongside formal evidence."""

    return {
        "batch_size": int(config["training"]["batch_size"]),
        "vision_encoder_batch_size": int(config["model"]["vision_encoder_batch_size"]),
        "train_loader": _data_loader_kwargs(config, device, role="train"),
        "validation_loader": _data_loader_kwargs(config, device, role="validation"),
        "pair_loader": _data_loader_kwargs(config, device, role="pair"),
        "project_unused_fields": True,
    }


def _causal_pair_loader(
    dataset: M1CausalPairDataset,
    config: Mapping[str, Any],
    *,
    seed: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["causal_pairs"]["pair_batch_size"]),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        drop_last=False,
        **_data_loader_kwargs(config, device, role="pair"),
    )


def _state_causal_pair_loader(
    dataset: M1StateCausalPairDataset,
    config: Mapping[str, Any],
    *,
    seed: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["state_causal_pairs"]["pair_batch_size"]),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        drop_last=False,
        **_data_loader_kwargs(config, device, role="pair"),
    )


def _training_lineage(
    windows: M1WindowDataset,
    causal_pairs: M1CausalPairDataset,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    lineage = windows.checkpoint_lineage()
    lineage.update(
        {
            "causal_pair_contract": _causal_pair_contract(config),
            "causal_pair_summary_sha256": causal_pairs.pair_summary_sha256(),
            "causal_pair_summary": causal_pairs.pair_summary(),
        }
    )
    return lineage


def _normalization(source_checkpoint: Path) -> Any:
    _, _, metadata = load_joint_wam_checkpoint(
        source_checkpoint,
        device="cpu",
        expected_schema_version="wam.proprio/1.0",
    )
    return metadata["normalization"]


def _strict_reload_evidence(
    directory: Path,
    model: torch.nn.Module,
    flow: torch.nn.Module,
    legacy_world: torch.nn.Module,
    legacy_flow: torch.nn.Module,
    *,
    device: torch.device,
    schema_version: str,
) -> dict[str, Any]:
    loaded = load_m1_checkpoint(
        directory, device="cpu", expected_schema_version=schema_version
    )
    comparisons = zip((model, flow, legacy_world, legacy_flow), loaded[:4], strict=True)
    maximum = 0.0
    for module_index, (expected, actual) in enumerate(comparisons):
        expected_state = expected.state_dict()
        actual_state = actual.state_dict()
        if module_index == 0:
            expected_state = {
                name: value
                for name, value in expected_state.items()
                if not name.startswith("vision_encoder.")
            }
            actual_state = {
                name: value
                for name, value in actual_state.items()
                if not name.startswith("vision_encoder.")
            }
        if set(expected_state) != set(actual_state):
            return {"passed": False, "max_abs_diff": None, "reason": "state_keys"}
        for name in expected_state:
            expected_value = expected_state[name].detach().cpu()
            actual_value = actual_state[name].detach().cpu()
            if expected_value.dtype == torch.bool:
                difference = float(
                    torch.logical_xor(expected_value, actual_value).any()
                )
            else:
                difference = float((expected_value - actual_value).abs().max())
            maximum = max(maximum, difference)
    return {"passed": maximum == 0.0, "max_abs_diff": maximum}


def _formal_request(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    *,
    variants: Sequence[str],
    seeds: Sequence[int],
) -> bool:
    return bool(
        args.variants is None
        and args.seeds is None
        and args.steps_scale == 1.0
        and args.checkpoint_root is None
        and args.output_root is None
        and not args.skip_preflight
        and not args.preflight_only
        and not args.skip_hdf5_hashes
        and tuple(variants) == INTERNAL_VARIANTS
        and tuple(int(value) for value in seeds)
        == tuple(int(value) for value in config["training"]["seeds"])
    )


def _validated_resume_preflight(
    path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = _load_json_mapping(path, label="M1 resume preflight")
    overfit = preflight.get("overfit_256")
    one_percent = preflight.get("one_percent")
    if preflight.get("passed") is not True:
        raise RuntimeError("M1 resume preflight is not passed")
    if not isinstance(overfit, Mapping) or overfit.get("passed") is not True:
        raise RuntimeError("M1 resume overfit-256 preflight is not passed")
    if not isinstance(one_percent, Mapping) or one_percent.get("passed") is not True:
        raise RuntimeError("M1 resume one-percent preflight is not passed")
    if overfit.get("causal_pair_passed") is not True:
        raise RuntimeError("M1 resume preflight lacks passing RGB causal evidence")
    if overfit.get("formal_state_causal_pairs_enabled") is not False:
        raise RuntimeError("M1 resume preflight contains formal state causal pairs")
    training = overfit.get("training")
    if not isinstance(training, Mapping):
        raise RuntimeError("M1 resume overfit-256 training evidence is missing")
    _validate_cold_stage(training, config, label="resume preflight overfit-256")
    stages = one_percent.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise RuntimeError("M1 resume one-percent stages are incomplete")
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise RuntimeError("M1 resume one-percent stage is not a mapping")
        _validate_cold_stage(
            stage,
            config,
            label=f"resume preflight one-percent stage {index}",
        )
    return dict(preflight)


def _validated_resume_report(
    report_path: Path,
    *,
    checkpoint_root: Path,
    config: Mapping[str, Any],
    variant: str,
    train_seed: int,
) -> dict[str, Any] | None:
    canonical = CANONICAL_VARIANT[variant]
    directory = (checkpoint_root / canonical / f"seed_{train_seed}").resolve()
    if not report_path.is_file():
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(
                f"M1 resume found checkpoint without a formal report: {directory}"
            )
        return None

    report = _load_json_mapping(
        report_path,
        label=f"M1 resume report {canonical}/seed-{train_seed}",
    )
    expected_scalars = {
        "format_version": "wam.multimodal.m1.training_run/1",
        "formal_protocol": True,
        "variant": canonical,
        "train_seed": train_seed,
        "preflight_passed": True,
        "causal_pair_passed": True,
        "formal_state_causal_pairs_enabled": False,
    }
    for key, expected in expected_scalars.items():
        if report.get(key) != expected:
            raise RuntimeError(
                f"M1 resume report {canonical}/seed-{train_seed} has invalid {key}"
            )
    if not math.isfinite(float(report.get("validation_action_chunk_rmse", math.nan))):
        raise RuntimeError("M1 resume report validation RMSE is not finite")
    if Path(str(report.get("checkpoint", ""))).resolve() != directory:
        raise RuntimeError("M1 resume report checkpoint path is not canonical")
    if not directory.is_dir():
        raise RuntimeError(f"M1 resume checkpoint is missing: {directory}")
    reported_tree = str(report.get("checkpoint_tree_sha256", ""))
    actual_tree = checkpoint_tree_sha256(directory)
    if reported_tree != actual_tree:
        raise RuntimeError(
            f"M1 resume checkpoint tree hash mismatch for {canonical}/seed-{train_seed}"
        )

    strict = report.get("strict_reload")
    if (
        not isinstance(strict, Mapping)
        or strict.get("passed") is not True
        or float(strict.get("max_abs_diff", math.nan)) != 0.0
    ):
        raise RuntimeError("M1 resume report lacks an exact strict reload")
    if report.get("frozen_anchor_sha256_before") != report.get(
        "frozen_anchor_sha256_after"
    ):
        raise RuntimeError("M1 resume report mutated the frozen action anchor")

    stages = report.get("stages")
    expected_stages = [
        config["training"]["stage_1"],
        config["training"]["stage_2"],
        config["training"]["stage_3"],
    ]
    if not isinstance(stages, list) or len(stages) != len(expected_stages):
        raise RuntimeError("M1 resume report training stages are incomplete")
    for index, (stage, expected) in enumerate(
        zip(stages, expected_stages, strict=True)
    ):
        if not isinstance(stage, Mapping):
            raise RuntimeError("M1 resume report stage is not a mapping")
        if stage.get("name") != expected["name"] or int(stage.get("steps", -1)) != int(
            expected["steps"]
        ):
            raise RuntimeError("M1 resume report stage contract differs from config")
        _validate_cold_stage(
            stage,
            config,
            label=f"resume report {canonical}/seed-{train_seed} stage {index}",
        )

    checkpoint_config = _load_yaml(directory / "config.yaml")
    for key, value in config.items():
        if checkpoint_config.get(key) != value:
            raise RuntimeError(
                f"M1 resume checkpoint config differs at top-level key {key!r}"
            )
    if checkpoint_config.get("train_seed") != train_seed:
        raise RuntimeError("M1 resume checkpoint seed differs from its report")
    if checkpoint_config.get("model_variant") != canonical:
        raise RuntimeError("M1 resume checkpoint variant differs from its report")
    return dict(report)


def _validate_cold_stage(
    stage: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if float(stage.get("warm_fraction", math.nan)) != 0.0:
        raise RuntimeError(f"{label} contains warm-start training samples")
    if stage.get("state_causal_pairs_enabled") is not False:
        raise RuntimeError(f"{label} contains formal state causal pairs")
    if stage.get("frozen_backbone") is not True:
        raise RuntimeError(f"{label} did not freeze the visual backbone")
    if stage.get("frozen_prior_anchor") is not True:
        raise RuntimeError(f"{label} did not freeze the action-prior anchor")
    objective = stage.get("flow_objective")
    if not isinstance(objective, Mapping):
        raise RuntimeError(f"{label} has no flow objective evidence")
    action = config["action_chunk"]
    expected = {
        "execution_steps": int(action["execution_steps"]),
        "solver_steps": int(action["solver_steps"]),
        "solver": str(action["solver"]),
        "warm_start_probability": float(action["warm_start_probability"]),
        "warm_start_noise_std": float(action["warm_start_noise_std"]),
        "policy_fixed_action_dims": list(action["policy_fixed_action_dims"]),
        "executed_prefix_weight": float(action["executed_prefix_weight"]),
        "normalized_action_clip": float(action["normalized_action_clip"]),
    }
    for key, value in expected.items():
        if objective.get(key) != value:
            raise RuntimeError(f"{label} flow objective differs at {key!r}")


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("M1 config must contain a mapping")
    return value


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} cannot be read: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _sequence(value: Mapping[str, Any], key: str) -> list[Any]:
    result = value.get(key)
    if not isinstance(result, list) or not result:
        raise ValueError(f"M1 config {key!r} must be a non-empty list")
    return result


def _module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
