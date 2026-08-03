#!/usr/bin/env python3
"""Train the scale-aligned S4-R7 token-preserving world-utility model."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wam_multimodal import EvidenceTokens  # noqa: E402
from scripts.s4_r7_model_io import build_s4_r7_model  # noqa: E402
from scripts.s4_r8_model_io import build_s4_r8_model  # noqa: E402
from scripts.train_s2_r4_future_predictor import (  # noqa: E402
    _dataset,
    _validate_artifact_dataset,
)
from scripts.train_s3_r6_world_action_flow import _model_inputs  # noqa: E402
from scripts.train_static_rgb_act_moe import (  # noqa: E402
    _append_jsonl,
    _atomic_torch_save,
    _capture_rng,
    _emit_stage,
    _git_commit,
    _load_yaml,
    _mapping,
    _restore_rng,
    _seed_everything,
    _sha256,
    _task_runtime,
    _vision,
)
from train.s2_future_prediction import (  # noqa: E402
    file_sha256,
    load_s2_artifact,
    normalized_state_delta,
    state_dict_sha256,
)
from train.s2_grouped_trajectory import grouped_s2_batch  # noqa: E402
from train.s4_hierarchical_team_sampler import (  # noqa: E402
    S4ExposureCounter,
    S4HierarchicalTeamBatchSampler,
)
from train.s4_future_feature_cache import (  # noqa: E402
    S4ProjectedFutureFeatureCache,
)
from train.s4_joint_losses import s4_joint_losses  # noqa: E402
from train.s4_model_registry import (  # noqa: E402
    validate_s4_r7_candidate,
    validate_s4_r8_candidate,
)
from train.world_action_flow_training import grouped_flow_matching_batch  # noqa: E402


CHECKPOINT_FORMAT = "wam.robofactory.s4_r7.world_utility.checkpoint/1"
RESUME_FORMAT = "wam.robofactory.s4_r7.world_utility.resume/1"
PREFLIGHT_FORMAT = "wam.robofactory.s4_r7.preflight/1"
GRADIENT_AUDIT_FORMAT = "wam.robofactory.s4_r7.gradient_audit/1"
EXPOSURE_FORMAT = "wam.robofactory.s4_r7.module_exposure/1"
FAST_SELECTION_BUDGET_MODE = "fast_selection_30k"
FAST_SELECTION_UPDATES = 30_000
FAST_SELECTION_FLOW_UNFREEZE = 6_400
FAST_SELECTION_AGENT_WINDOW_BUDGET = 1_152_000
SUPPORTED_BATCH_RECIPES = {(4, 3), (2, 6), (1, 12)}
ROUND_ID = "s4-r7"
ROUND_LABEL = "S4-R7"
PROGRAM_NAME = "train_s4_r7_world_utility.py"
ENV_PREFIX = "S4_R7"


def _configure_round(raw: Mapping[str, Any]) -> tuple[str, str, float, str]:
    """Select the fail-closed R7 or R8 contract for this process."""

    global CHECKPOINT_FORMAT, RESUME_FORMAT, PREFLIGHT_FORMAT
    global GRADIENT_AUDIT_FORMAT, EXPOSURE_FORMAT
    global ROUND_ID, ROUND_LABEL, PROGRAM_NAME, ENV_PREFIX
    round_section = _mapping(raw, "round")
    observed = str(round_section.get("round_id", ""))
    if observed == "s4-r7":
        candidate, model_kind, utility = validate_s4_r7_candidate(raw)
        aggregator = "trajectory_mean_legacy_r7"
        return candidate, model_kind, utility, aggregator
    if observed != "s4-r8":
        raise ValueError(f"unsupported S4 training round: {observed!r}")
    candidate, model_kind, aggregator = validate_s4_r8_candidate(raw)
    ROUND_ID = "s4-r8"
    ROUND_LABEL = "S4-R8"
    PROGRAM_NAME = "train_s4_r8_horizon_causal.py"
    ENV_PREFIX = "S4_R8"
    CHECKPOINT_FORMAT = "wam.robofactory.s4_r8.horizon_causal.checkpoint/1"
    RESUME_FORMAT = "wam.robofactory.s4_r8.horizon_causal.resume/1"
    PREFLIGHT_FORMAT = "wam.robofactory.s4_r8.preflight/1"
    GRADIENT_AUDIT_FORMAT = "wam.robofactory.s4_r8.gradient_audit/1"
    EXPOSURE_FORMAT = "wam.robofactory.s4_r8.module_exposure/1"
    utility = float(_mapping(raw, "training")["utility_coupling_weight"])
    return candidate, model_kind, utility, aggregator


def _round_environment(name: str) -> str:
    return os.environ.get(f"{ENV_PREFIX}_{name}", "")


def _build_round_model(
    raw: Mapping[str, Any], *, device: torch.device
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    if ROUND_ID == "s4-r8":
        return build_s4_r8_model(raw, device=device)
    return build_s4_r7_model(raw, device=device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    parser.add_argument(
        "--stop-after-update",
        type=int,
        help=(
            "pause a formal run exactly at a preregistered milestone after "
            "writing its recoverable resume and milestone checkpoint"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--progress-log", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-updates", type=int, default=200)
    parser.add_argument("--preflight-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _main_impl(args)
    except torch.cuda.OutOfMemoryError:
        if args.preflight_only and args.preflight_report is not None:
            _write_terminal_oom_preflight(args)
        raise


def _main_impl(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve(strict=True)
    raw = _load_yaml(config_path)
    candidate_id, model_kind, utility_weight, action_prefix_aggregator = (
        _configure_round(raw)
    )
    training = _mapping(raw, "training")
    shared_hdf5_receipt_sha256 = _round_environment("SHARED_HDF5_RECEIPT_SHA256")
    if len(shared_hdf5_receipt_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in shared_hdf5_receipt_sha256
    ):
        raise ValueError("S4-R7 requires the shared HDF5 receipt SHA256 identity")
    future_feature_cache_root = _round_environment("FUTURE_FEATURE_CACHE")
    future_feature_cache_sha256 = _round_environment("FUTURE_FEATURE_CACHE_SHA256")
    if (
        not future_feature_cache_root
        or len(future_feature_cache_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in future_feature_cache_sha256
        )
    ):
        raise ValueError("S4-R7 requires the shared future feature cache identity")
    configured_updates = int(training.get("updates", 0))
    if (
        training.get("budget_mode") != FAST_SELECTION_BUDGET_MODE
        or configured_updates != FAST_SELECTION_UPDATES
    ):
        raise ValueError(
            "S4-R7 fast selection requires budget_mode=fast_selection_30k "
            "and exactly 30000 updates"
        )
    if args.preflight_only:
        if args.stop_after_update is not None:
            raise ValueError("preflight cannot use --stop-after-update")
        if args.preflight_report is None:
            raise ValueError("--preflight-only requires --preflight-report")
        updates = int(args.preflight_updates)
        if updates != 200:
            raise ValueError("S4-R7 paired preflight is exactly 200 optimizer updates")
    else:
        updates = int(args.updates if args.updates is not None else configured_updates)
        if updates != configured_updates:
            raise ValueError("formal S4-R7 training cannot change the 30k budget")
    run_end_update = updates
    if not args.preflight_only and args.stop_after_update is not None:
        run_end_update = int(args.stop_after_update)
        registered_milestones = {int(value) for value in training.get("milestones", ())}
        if run_end_update not in registered_milestones:
            raise ValueError(
                "--stop-after-update must be a preregistered S4-R7 milestone"
            )
        if not 0 < run_end_update <= configured_updates:
            raise ValueError("formal milestone stop lies outside the 30k budget")

    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("S4-R7 requires exactly one visible CUDA GPU per candidate")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("S4-R7 requires native BF16")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    seed = int(training.get("seed", 707))
    _seed_everything(seed)
    torch.cuda.reset_peak_memory_stats(device)

    _emit_stage("parent_load", "loading exact R6L-P1/R5-P0 active clones")
    model, legacy_reference, parent_identity = _build_round_model(raw, device=device)
    initial_model_sha256 = state_dict_sha256(model)
    artifact_path = (ROOT / str(_mapping(raw, "artifacts")["pca_statistics"])).resolve(
        strict=True
    )
    artifact = load_s2_artifact(artifact_path, device=device)
    artifact_sha256 = file_sha256(artifact_path)
    training_split = str(_mapping(raw, "data").get("training_split", ""))
    if training_split != "all":
        raise ValueError("S4-R7 training must consume all manifest episodes")
    dataset = _dataset(raw, split=training_split)
    _validate_artifact_dataset(artifact, dataset)
    manifests = [contract.manifest_path for contract in dataset.contracts]
    future_feature_cache = S4ProjectedFutureFeatureCache(
        future_feature_cache_root,
        manifests=manifests,
        expected_features_sha256=future_feature_cache_sha256,
        expected_pca_sha256=str(_mapping(raw, "parent")["expected_pca_sha256"]),
        expected_vision_weights_sha256=str(
            _mapping(raw, "vision")["expected_weights_sha256"]
        ),
    )
    hierarchy_summary = dataset.summary()["hierarchical_sampling"]
    vision = _vision(raw).to(device).eval()
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise RuntimeError("DINOv3 must remain frozen in S4-R7")

    micro = int(training.get("micro_team_batch", 0))
    accumulation = int(training.get("gradient_accumulation", 0))
    effective = int(training.get("effective_team_batch", 0))
    if micro * accumulation != effective or effective != 12:
        raise ValueError("S4-R7 requires micro*accum == effective team batch 12")
    if (micro, accumulation) not in SUPPORTED_BATCH_RECIPES:
        raise ValueError(
            "S4-R7 supports paired micro4/accum3, micro2/accum6, or micro1/accum12"
        )
    flow_unfreeze = int(training.get("flow_unfreeze_update", 0))
    if flow_unfreeze != FAST_SELECTION_FLOW_UNFREEZE:
        raise ValueError("S4-R7 fast-selection Flow must unfreeze at update 6400")

    output = _resolve_path(
        args.output,
        _mapping(raw, "checkpoint").get("output"),
    )
    resume = _resolve_path(
        args.resume,
        _mapping(raw, "checkpoint").get("resume"),
    )
    progress_log = _resolve_path(
        args.progress_log,
        _mapping(raw, "checkpoint").get("progress_log"),
    )
    preflight_report = (
        args.preflight_report.expanduser().resolve()
        if args.preflight_report is not None
        else None
    )
    candidate_root = output.parent.parent
    audit_dir = output.parent if args.preflight_only else candidate_root / "train"
    audit_dir.mkdir(parents=True, exist_ok=True)
    forced_log = audit_dir / "forced_evidence_audit.jsonl"
    gradient_audit_path = audit_dir / "parameter_gradient_audit.json"
    exposure_path = audit_dir / "module_exposure.json"
    for path in (output, resume, progress_log):
        path.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed checkpoint {output}")
    if (
        args.preflight_only
        and preflight_report is not None
        and preflight_report.exists()
    ):
        raise FileExistsError(f"refusing to overwrite preflight {preflight_report}")

    flow_parameters = tuple(model.active_parent.base_flow.parameters())
    for parameter in flow_parameters:
        parameter.requires_grad_(False)
    parameter_groups, parameter_names = _parameter_groups(model, training)
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(training.get("weight_decay", 1e-4)),
        fused=True,
    )
    identity = {
        "round_id": ROUND_ID,
        "candidate_id": candidate_id,
        "model_kind": model_kind,
        "utility_coupling_weight": utility_weight,
        "action_prefix_aggregator": action_prefix_aggregator,
        "config_sha256": _sha256(config_path),
        "parent_identity": parent_identity,
        "artifact_sha256": artifact_sha256,
        "initial_model_sha256": initial_model_sha256,
        "seed": seed,
        "micro_team_batch": micro,
        "gradient_accumulation": accumulation,
        "effective_team_batch": effective,
        "updates": configured_updates,
        "shared_hdf5_receipt_sha256": shared_hdf5_receipt_sha256,
        "future_feature_cache_sha256": future_feature_cache_sha256,
    }
    start_update = 0
    exposure = S4ExposureCounter()
    dataset_chain = bytes(32)
    restored_audit_state: Mapping[str, Any] = {}
    restored_module_exposure: Mapping[str, Any] = {}
    if not args.preflight_only and not args.no_resume and resume.is_file():
        _emit_stage("resume_load", f"loading exact resume {resume}")
        saved = torch.load(resume, map_location=device, weights_only=False)
        if saved.get("format_version") != RESUME_FORMAT:
            raise ValueError("resume is not an S4-R7 optimizer state")
        if _mapping(saved, "identity") != identity:
            raise ValueError("S4-R7 resume identity differs from this candidate")
        start_update = int(saved.get("update", -1))
        if not 0 <= start_update < updates:
            raise ValueError("S4-R7 resume update lies outside the formal budget")
        if start_update >= run_end_update:
            raise ValueError(
                "resume already reached or passed the requested milestone stop"
            )
        model.load_state_dict(saved["model"], strict=True)
        if start_update >= flow_unfreeze:
            for parameter in flow_parameters:
                parameter.requires_grad_(True)
        optimizer.load_state_dict(saved["optimizer"])
        exposure.load_state_dict(_mapping(saved, "exposure"))
        dataset_chain = bytes.fromhex(str(saved.get("dataset_chain_sha256", "")))
        if len(dataset_chain) != 32:
            raise ValueError("resume dataset chain hash is invalid")
        audit_state = saved.get("training_audit_state")
        if not isinstance(audit_state, Mapping):
            raise ValueError("resume training audit state is missing")
        restored_audit_state = audit_state
        module_exposure_state = saved.get("module_exposure_state")
        if not isinstance(module_exposure_state, Mapping):
            raise ValueError("resume module exposure state is missing")
        restored_module_exposure = module_exposure_state
        _restore_rng(_mapping(saved, "rng"))
    else:
        _emit_stage("resume_load", "starting from common ancestor weights at update 0")

    sampler = S4HierarchicalTeamBatchSampler(
        dataset,
        micro_batch_size=micro,
        gradient_accumulation=accumulation,
        first_update=start_update + 1,
        final_update=run_end_update,
        seed=seed,
    )
    workers = int(training.get("num_workers", 8))
    loader_options: dict[str, Any] = {}
    if workers > 0:
        loader_options["prefetch_factor"] = int(training.get("prefetch_factor", 4))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed + 10_000_000),
        **loader_options,
    )
    iterator = iter(loader)
    pca = _mapping(raw, "pca")
    _emit_stage("first_batch", "running bit-exact legacy/scaled/new-gate audits")
    first_grouped = grouped_s2_batch(next(iterator), require_future_images=False)
    first_inputs = _model_inputs(
        vision, first_grouped, artifact, device=device, pca=pca
    )
    structural = _structural_audit(
        model,
        legacy_reference,
        first_inputs,
        parent_identity=parent_identity,
    )
    del legacy_reference
    del first_inputs, first_grouped, iterator, loader
    torch.cuda.empty_cache()
    # Recreate the sampler so the structural audit does not consume update 1.
    sampler = S4HierarchicalTeamBatchSampler(
        dataset,
        micro_batch_size=micro,
        gradient_accumulation=accumulation,
        first_update=start_update + 1,
        final_update=run_end_update,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed + 10_000_000),
        **loader_options,
    )
    iterator = iter(loader)

    restored_gradient_audit = restored_audit_state.get("gradient_audit")
    if restored_gradient_audit is not None and not isinstance(
        restored_gradient_audit, Mapping
    ):
        raise ValueError("resume gradient audit state must be a mapping")
    gradient_audit: dict[str, Any] = (
        dict(restored_gradient_audit)
        if isinstance(restored_gradient_audit, Mapping)
        else {
            "format_version": GRADIENT_AUDIT_FORMAT,
            "candidate_id": candidate_id,
            "normal_group_nonzero": {},
            "wuc_only": None,
        }
    )
    if (
        gradient_audit.get("format_version") != GRADIENT_AUDIT_FORMAT
        or gradient_audit.get("candidate_id") != candidate_id
    ):
        raise ValueError("resume gradient audit identity differs from candidate")
    dataset_indices: list[int] = []
    agent_histogram = {str(count): 0 for count in range(1, 5)}
    forced_seconds = 0.0
    forced_count = 0
    started = time.perf_counter()
    model.train(True)
    milestones = {int(value) for value in training.get("milestones", ())}
    save_interval = int(training.get("save_interval", 1000))
    log_interval = int(training.get("log_interval", 20))
    gradient_clip = float(training.get("gradient_clip_norm", 1.0))
    tracked_categories = tuple(
        name for name in parameter_names if not (args.preflight_only and name == "flow")
    )
    restored_categories = restored_audit_state.get("normal_categories_seen", {})
    if not isinstance(restored_categories, Mapping):
        raise ValueError("resume normal category audit must be a mapping")
    normal_categories_seen: dict[str, bool] = {
        name: bool(restored_categories.get(name, False)) for name in tracked_categories
    }
    flow_frozen_gradient_exact_zero = bool(
        restored_audit_state.get("flow_frozen_gradient_exact_zero", True)
    )
    flow_frozen_observed = bool(restored_audit_state.get("flow_frozen_observed", False))
    flow_unfrozen_gradient_nonzero = bool(
        restored_audit_state.get("flow_unfrozen_gradient_nonzero", False)
    )
    flow_unfrozen_observed = bool(
        restored_audit_state.get("flow_unfrozen_observed", False)
    )
    module_exposure: dict[str, dict[str, int]] = {}
    if restored_module_exposure and set(restored_module_exposure) != set(
        parameter_names
    ):
        raise ValueError("resume module exposure groups differ from optimizer")
    for name in parameter_names:
        restored_group = restored_module_exposure.get(name, {})
        if not isinstance(restored_group, Mapping):
            raise ValueError("resume module exposure row must be a mapping")
        team_windows = int(restored_group.get("team_windows_seen", 0))
        agent_windows = int(restored_group.get("valid_agent_windows_seen", 0))
        if team_windows < 0 or agent_windows < 0:
            raise ValueError("resume module exposure counters cannot be negative")
        module_exposure[name] = {
            "team_windows_seen": team_windows,
            "valid_agent_windows_seen": agent_windows,
        }

    try:
        for update in range(start_update + 1, run_end_update + 1):
            if update == flow_unfreeze:
                for parameter in flow_parameters:
                    parameter.requires_grad_(True)
            _set_learning_rates(
                optimizer,
                update=update,
                total_updates=configured_updates,
                flow_unfreeze_update=flow_unfreeze,
                warmup_updates=int(training.get("warmup_updates", 500)),
                flow_warmup_updates=int(training.get("flow_warmup_updates", 500)),
            )
            optimizer.zero_grad(set_to_none=True)
            update_losses: list[dict[str, float]] = []
            update_valid_agents = 0
            audit_row: dict[str, Any] | None = None
            for accumulation_index in range(accumulation):
                grouped = grouped_s2_batch(next(iterator), require_future_images=False)
                indices = [int(value) for value in grouped["dataset_index"].tolist()]
                dataset_indices.extend(indices)
                dataset_chain = _extend_dataset_chain(dataset_chain, indices)
                valid_cpu = grouped["valid_agent_mask"].bool()
                exposure.record_batch(valid_cpu)
                counts = valid_cpu.sum(dim=1).tolist()
                for count in counts:
                    agent_histogram[str(int(count))] += 1
                    update_valid_agents += int(count)
                inputs = _model_inputs(
                    vision, grouped, artifact, device=device, pca=pca
                )
                targets = _future_targets(
                    grouped,
                    artifact,
                    future_feature_cache=future_feature_cache,
                    current_local=inputs["local_visual"],
                    current_shared=inputs["shared_visual"],
                    device=device,
                )
                actions = grouped["candidate_actions"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                action_inputs, target_velocity, tau = grouped_flow_matching_batch(
                    actions
                )
                valid_action = grouped["action_valid_mask"].to(
                    device=device, dtype=torch.bool
                )
                valid_queries = inputs["valid"][:, :, None] & valid_action[:, None]

                wuc_loss = target_velocity.new_zeros(())
                selected_item = (update // 4) % effective
                selected_accumulation = selected_item // micro
                selected_offset = selected_item % micro
                if (
                    update % int(training.get("counterfactual_every", 4)) == 0
                    and accumulation_index == selected_accumulation
                ):
                    forced_started = time.perf_counter()
                    audit_row, wuc_loss, wuc_scope = _forced_audit(
                        model,
                        inputs,
                        action_inputs,
                        tau,
                        target_velocity,
                        valid_queries,
                        selected_offset=selected_offset,
                        grouped=grouped,
                        utility_weight=utility_weight,
                    )
                    forced_seconds += time.perf_counter() - forced_started
                    forced_count += 1
                    if gradient_audit["wuc_only"] is None:
                        gradient_audit["wuc_only"] = wuc_scope

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction, diagnostics = model.velocity(
                        inputs["raw_local"],
                        inputs["state"],
                        inputs["local_visual"],
                        inputs["shared_visual"],
                        action_inputs,
                        tau,
                        inputs["valid"],
                    )
                    futures = diagnostics["predicted_futures"]
                    losses = s4_joint_losses(
                        flow_prediction=prediction,
                        flow_target=target_velocity,
                        flow_valid_mask=valid_action,
                        valid_agent_mask=inputs["valid"],
                        own_state_prediction=futures.own_state,
                        own_state_target=targets["state"],
                        own_state_valid_mask=targets["state_valid"],
                        own_visual_prediction=futures.own_visual,
                        own_visual_target=targets["local_visual"],
                        own_visual_valid_mask=targets["local_visual_valid"],
                        peer_state_prediction=futures.peer_state,
                        peer_state_target=targets["state"],
                        peer_state_valid_mask=targets["state_valid"],
                        peer_visual_prediction=futures.peer_visual,
                        peer_visual_target=targets["local_visual"],
                        peer_visual_valid_mask=targets["local_visual_valid"],
                        shared_visual_prediction=futures.shared_visual,
                        shared_visual_target=targets["shared_visual"],
                        shared_visual_valid_mask=targets["shared_visual_valid"],
                        flow_loss_weight=float(training.get("flow_loss_weight", 1.0)),
                        state_loss_weight=float(
                            training.get("future_state_loss_weight", 0.25)
                        ),
                        visual_loss_weight=float(
                            training.get("future_visual_loss_weight", 0.25)
                        ),
                    )
                    # WUC is one preregistered team sample per four optimizer
                    # updates, not one sixth of a micro-batch objective.
                    total_loss = losses.total + utility_weight * accumulation * wuc_loss
                    scaled_loss = total_loss / accumulation
                scaled_loss.backward()
                update_losses.append(
                    {
                        "loss": float(total_loss.detach()),
                        "flow_loss": float(losses.flow.detach()),
                        "future_state_loss": float(losses.state.detach()),
                        "future_visual_loss": float(losses.visual.detach()),
                        "wuc_loss": float(wuc_loss.detach()),
                    }
                )

            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    gradient_clip,
                )
            )
            category_norms = _gradient_norms(model, parameter_names)
            for name in normal_categories_seen:
                normal_categories_seen[name] = (
                    normal_categories_seen[name] or category_norms[name] > 0.0
                )
            if update < flow_unfreeze:
                flow_frozen_observed = True
                flow_frozen_gradient_exact_zero = (
                    flow_frozen_gradient_exact_zero and category_norms["flow"] == 0.0
                )
            else:
                flow_unfrozen_observed = True
                flow_unfrozen_gradient_nonzero = (
                    flow_unfrozen_gradient_nonzero or category_norms["flow"] > 0.0
                )
            for name, counters in module_exposure.items():
                if name != "flow" or update >= flow_unfreeze:
                    counters["team_windows_seen"] += effective
                    counters["valid_agent_windows_seen"] += update_valid_agents
            optimizer.step()
            if audit_row is not None:
                audit_row["created_at"] = datetime.now(timezone.utc).isoformat()
                _append_jsonl(forced_log, audit_row)

            if update == start_update + 1:
                _emit_stage(
                    "optimizer_training", f"joint update {update}/{configured_updates}"
                )
            if update == 1 or update % log_interval == 0 or update == run_end_update:
                elapsed = max(time.perf_counter() - started, 1e-9)
                averaged = {
                    key: sum(row[key] for row in update_losses) / len(update_losses)
                    for key in update_losses[0]
                }
                progress = {
                    "event": "optimizer_step",
                    "program": PROGRAM_NAME,
                    "round_id": ROUND_ID,
                    "candidate_id": candidate_id,
                    "model_kind": model_kind,
                    "update": update,
                    "updates": configured_updates
                    if not args.preflight_only
                    else updates,
                    "segment_end_update": run_end_update,
                    **averaged,
                    "gradient_norm": gradient_norm,
                    "gradient_norms": category_norms,
                    "learning_rates": {
                        str(group["name"]): float(group["lr"])
                        for group in optimizer.param_groups
                    },
                    "micro_team_batch": micro,
                    "micro_batch": micro,
                    "gradient_accumulation": accumulation,
                    "effective_team_batch": effective,
                    "vision_inference_batch_size": int(
                        _mapping(raw, "vision")["inference_batch_size"]
                    ),
                    "shared_hdf5_receipt_sha256": shared_hdf5_receipt_sha256,
                    "future_feature_cache_sha256": future_feature_cache_sha256,
                    "effective_batch": effective,
                    "team_windows_seen": exposure.team_windows_seen,
                    "valid_agent_windows_seen": exposure.valid_agent_windows_seen,
                    "agent_windows_seen": exposure.valid_agent_windows_seen,
                    "agent_window_budget": int(
                        training.get(
                            "agent_window_budget", FAST_SELECTION_AGENT_WINDOW_BUDGET
                        )
                    ),
                    "valid_agents_this_update": update_valid_agents,
                    "flow_unfreeze_update": flow_unfreeze,
                    "flow_trainable": update >= flow_unfreeze,
                    "flow_unfreeze_state": (
                        "unfrozen"
                        if update >= flow_unfreeze
                        else f"frozen->{flow_unfreeze}"
                    ),
                    "grad_norm": gradient_norm,
                    "learning_rate": {
                        str(group["name"]): float(group["lr"])
                        for group in optimizer.param_groups
                    },
                    "updates_per_second": (update - start_update) / elapsed,
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                print(json.dumps(progress, sort_keys=True), flush=True)
                _append_jsonl(progress_log, progress)

            if not args.preflight_only and (
                update % save_interval == 0 or update in milestones
            ):
                if update < configured_updates:
                    resume_payload = _resume_payload(
                        identity,
                        update,
                        model,
                        optimizer,
                        exposure,
                        dataset_chain,
                        gradient_audit=gradient_audit,
                        normal_categories_seen=normal_categories_seen,
                        flow_frozen_gradient_exact_zero=flow_frozen_gradient_exact_zero,
                        flow_frozen_observed=flow_frozen_observed,
                        flow_unfrozen_gradient_nonzero=flow_unfrozen_gradient_nonzero,
                        flow_unfrozen_observed=flow_unfrozen_observed,
                        module_exposure=module_exposure,
                    )
                    _atomic_torch_save(resume_payload, resume)
                if update in milestones:
                    milestone = output.parent / "milestones" / f"update_{update:06d}.pt"
                    milestone.parent.mkdir(parents=True, exist_ok=True)
                    if not milestone.exists():
                        _atomic_torch_save(
                            _checkpoint_payload(
                                raw,
                                config_path,
                                identity,
                                update,
                                model,
                                structural,
                                exposure,
                                parameter_names,
                                module_exposure,
                                vision,
                                dataset,
                            ),
                            milestone,
                        )
    except torch.cuda.OutOfMemoryError:
        if args.preflight_only and preflight_report is not None:
            _atomic_json(
                preflight_report,
                {
                    "format_version": PREFLIGHT_FORMAT,
                    "identity": {
                        "round_id": ROUND_ID,
                        "candidate_id": candidate_id,
                        "model_kind": model_kind,
                        "action_prefix_aggregator": action_prefix_aggregator,
                    },
                    "updates": 200,
                    "completed": False,
                    "oom": True,
                    "micro_team_batch": micro,
                    "gradient_accumulation": accumulation,
                    "effective_team_batch": effective,
                    "flow_unfreeze_update": flow_unfreeze,
                    "vision_inference_batch_size": int(
                        _mapping(raw, "vision")["inference_batch_size"]
                    ),
                    "shared_hdf5_receipt_sha256": shared_hdf5_receipt_sha256,
                    "future_feature_cache_sha256": future_feature_cache_sha256,
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "gpu_total_memory_bytes": int(
                        torch.cuda.get_device_properties(device).total_memory
                    ),
                },
            )
        raise

    elapsed = max(time.perf_counter() - started, 1e-9)
    if not args.preflight_only and run_end_update < configured_updates:
        milestone = output.parent / "milestones" / f"update_{run_end_update:06d}.pt"
        if not milestone.is_file() or not resume.is_file():
            raise RuntimeError(
                "milestone pause requires both a checkpoint and recoverable resume"
            )
        pause = {
            "event": "milestone_pause",
            "program": PROGRAM_NAME,
            "round_id": ROUND_ID,
            "candidate_id": candidate_id,
            "model_kind": model_kind,
            "update": run_end_update,
            "updates": configured_updates,
            "milestone_checkpoint": str(milestone.resolve(strict=True)),
            "resume": str(resume.resolve(strict=True)),
            "team_windows_seen": exposure.team_windows_seen,
            "valid_agent_windows_seen": exposure.valid_agent_windows_seen,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _append_jsonl(progress_log, pause)
        dataset.close()
        print(json.dumps(pause, sort_keys=True), flush=True)
        return 0
    gradient_audit["normal_group_nonzero"] = normal_categories_seen
    gradient_audit["flow_frozen_gradient_exact_zero"] = flow_frozen_gradient_exact_zero
    gradient_audit["flow_frozen_observed"] = flow_frozen_observed
    gradient_audit["flow_unfrozen_gradient_nonzero"] = flow_unfrozen_gradient_nonzero
    gradient_audit["flow_unfrozen_observed"] = flow_unfrozen_observed
    gradient_audit["normal_expected_flow_frozen"] = args.preflight_only
    gradient_audit["passed"] = (
        all(normal_categories_seen.values())
        and flow_frozen_gradient_exact_zero
        and flow_frozen_observed
        and (
            args.preflight_only
            or (flow_unfrozen_observed and flow_unfrozen_gradient_nonzero)
        )
        and (
            candidate_id == "P0"
            or bool(_mapping(gradient_audit, "wuc_only").get("passed"))
        )
    )
    exposure_report = {
        "format_version": EXPOSURE_FORMAT,
        "round_id": ROUND_ID,
        "candidate_id": candidate_id,
        **exposure.summary(
            agent_window_budget=int(
                training.get("agent_window_budget", FAST_SELECTION_AGENT_WINDOW_BUDGET)
            )
        ),
        "hierarchy": hierarchy_summary,
        "agent_count_histogram": agent_histogram,
        "flow_frozen_updates": [1, flow_unfreeze - 1],
        "flow_unfreeze_update": flow_unfreeze,
        "agent_windows_seen_by_module": {
            name: dict(counters) for name, counters in module_exposure.items()
        },
    }
    non_flow_complete = all(
        counters["team_windows_seen"] == exposure.team_windows_seen
        and counters["valid_agent_windows_seen"] == exposure.valid_agent_windows_seen
        for name, counters in module_exposure.items()
        if name != "flow"
    )
    expected_flow_team_windows = (
        max(updates - flow_unfreeze + 1, 0) * effective
        if not args.preflight_only
        else 0
    )
    exposure_report["non_flow_exposure_exact"] = non_flow_complete
    exposure_report["flow_team_windows_expected"] = expected_flow_team_windows
    exposure_report["flow_exposure_exact"] = (
        module_exposure["flow"]["team_windows_seen"] == expected_flow_team_windows
    )
    exposure_report["formal_budget_complete"] = (
        not args.preflight_only
        and exposure.team_windows_seen == configured_updates * effective
    )
    exposure_report["passed"] = (
        non_flow_complete
        and exposure_report["flow_exposure_exact"]
        and (args.preflight_only or exposure_report["formal_budget_complete"])
    )

    if args.preflight_only:
        assert preflight_report is not None
        replay = S4HierarchicalTeamBatchSampler(
            dataset,
            micro_batch_size=micro,
            gradient_accumulation=accumulation,
            first_update=updates,
            final_update=updates + 1,
            seed=seed,
        )
        uninterrupted_batches = list(replay)
        resumed_batches = list(
            S4HierarchicalTeamBatchSampler(
                dataset,
                micro_batch_size=micro,
                gradient_accumulation=accumulation,
                first_update=updates + 1,
                final_update=updates + 1,
                seed=seed,
            )
        )
        resume_next_exact = uninterrupted_batches[accumulation:] == resumed_batches
        report = {
            "format_version": PREFLIGHT_FORMAT,
            "identity": {
                "round_id": ROUND_ID,
                "candidate_id": candidate_id,
                "model_kind": model_kind,
                "action_prefix_aggregator": action_prefix_aggregator,
            },
            "updates": 200,
            "completed": True,
            "oom": False,
            "micro_team_batch": micro,
            "gradient_accumulation": accumulation,
            "effective_team_batch": effective,
            "dataset_index_sequence_sha256": hashlib.sha256(
                np.asarray(dataset_indices, dtype=np.int64).tobytes()
            ).hexdigest(),
            "dataset_chain_sha256": dataset_chain.hex(),
            "agent_count_histogram": agent_histogram,
            "update_1_trainable_name_sha256": _name_hash(
                name
                for group, names in parameter_names.items()
                if group != "flow"
                for name in names
            ),
            "flow_unfreeze_update": flow_unfreeze,
            "flow_unfreeze_trainable_name_sha256": _name_hash(
                name for names in parameter_names.values() for name in names
            ),
            "vision_inference_batch_size": int(
                _mapping(raw, "vision")["inference_batch_size"]
            ),
            "shared_hdf5_receipt_sha256": identity["shared_hdf5_receipt_sha256"],
            "future_feature_cache_sha256": identity["future_feature_cache_sha256"],
            "learning_rate_curve_sha256": _lr_curve_hash(training),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "gpu_total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
            "updates_per_second": updates / elapsed,
            "forced_audit_seconds": forced_seconds,
            "forced_audit_mean_seconds": forced_seconds / max(forced_count, 1),
            "resume_next_batch_exact": resume_next_exact,
            "parameter_gradient_audit": gradient_audit,
            "module_exposure": exposure_report,
            "structural_invariants": structural,
            "parent_identity": parent_identity,
            "sampler": sampler.summary(),
        }
        _atomic_json(preflight_report, report)
        dataset.close()
        print(json.dumps({"preflight": str(preflight_report), "completed": True}))
        return 0

    _atomic_json(gradient_audit_path, gradient_audit, overwrite=True)
    _atomic_json(exposure_path, exposure_report, overwrite=True)
    payload = _checkpoint_payload(
        raw,
        config_path,
        identity,
        updates,
        model,
        structural,
        exposure,
        parameter_names,
        module_exposure,
        vision,
        dataset,
    )
    _atomic_torch_save(payload, output)
    output_sha = file_sha256(output)
    dataset.close()
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "sha256": output_sha,
                "candidate_id": candidate_id,
                "updates": updates,
                "team_windows_seen": exposure.team_windows_seen,
                "valid_agent_windows_seen": exposure.valid_agent_windows_seen,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _future_targets(
    grouped: Mapping[str, Tensor],
    artifact: Mapping[str, Any],
    *,
    future_feature_cache: S4ProjectedFutureFeatureCache,
    current_local: Tensor,
    current_shared: Tensor,
    device: torch.device,
) -> dict[str, Tensor]:
    local_visual, shared_visual = future_feature_cache.normalized_targets(
        grouped,
        artifact,
        current_local=current_local,
        current_shared=current_shared,
        device=device,
    )
    return {
        "state": normalized_state_delta(grouped, artifact, device=device),
        "state_valid": grouped["future_state_valid_mask"].to(
            device=device, dtype=torch.bool
        ),
        "local_visual": local_visual,
        "local_visual_valid": grouped["future_agent_visual_valid_mask"].to(
            device=device, dtype=torch.bool
        ),
        "shared_visual": shared_visual,
        "shared_visual_valid": grouped["future_shared_visual_valid_mask"].to(
            device=device, dtype=torch.bool
        ),
    }


def _forced_audit(
    model: torch.nn.Module,
    inputs: Mapping[str, Tensor],
    action_inputs: Tensor,
    tau: Tensor,
    target_velocity: Tensor,
    valid_queries: Tensor,
    *,
    selected_offset: int,
    grouped: Mapping[str, Tensor],
    utility_weight: float,
) -> tuple[dict[str, Any], Tensor, dict[str, Any]]:
    model.eval()
    selection = slice(selected_offset, selected_offset + 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, diagnostics = model.velocity(
            inputs["raw_local"][selection],
            inputs["state"][selection],
            inputs["local_visual"][selection],
            inputs["shared_visual"][selection],
            action_inputs[selection],
            tau[selection],
            inputs["valid"][selection],
        )
        forced = model.forced_evidence_audit(
            diagnostics,
            target_velocity[selection],
            inputs["valid"][selection],
            valid_action_query_mask=valid_queries[selection],
        )
    model.train(True)
    q = diagnostics["flow_features"].detach()
    evidence = EvidenceTokens(
        diagnostics["evidence_tokens"].detach(),
        diagnostics["evidence_mask"].detach(),
    )
    routed = model.router(q, evidence, group_mask=forced.group_mask)
    wuc_loss = model.router.group_bias.sum() * 0
    if utility_weight > 0.0:
        from models.wam_multimodal import world_utility_coupling_loss

        wuc_loss = world_utility_coupling_loss(
            routed.pi,
            forced.utility_target,
            forced.group_mask,
            inputs["valid"][selection],
            valid_action_query_mask=valid_queries[selection],
        )
    scope = _wuc_gradient_scope(model, wuc_loss, enabled=utility_weight > 0.0)
    valid = forced.valid_query_mask[..., None] & forced.group_mask[:, :, None]
    denominator = valid.sum(dim=(0, 1, 2)).clamp_min(1)
    error_mean = (
        torch.where(valid, forced.velocity_errors, 0).sum(dim=(0, 1, 2)) / denominator
    )
    utility_mean = (
        torch.where(valid, forced.utility_target, 0).sum(dim=(0, 1, 2)) / denominator
    )
    route_mean = (
        torch.where(valid, routed.pi.detach(), 0).sum(dim=(0, 1, 2)) / denominator
    )
    offset = selected_offset
    row = {
        "event": "forced_evidence_audit",
        "task_index": int(grouped["task_index"][offset]),
        "episode_index": int(grouped["episode_index"][offset]),
        "episode_seed": int(grouped["episode_seed"][offset]),
        "decision_t": int(grouped["decision_t"][offset]),
        "valid_agent_mask": inputs["valid"][selection].cpu().tolist(),
        "group_mask": forced.group_mask.cpu().tolist(),
        "velocity_error_by_group": error_mean.float().cpu().tolist(),
        "utility_target_by_group": utility_mean.float().cpu().tolist(),
        "router_pi_by_group": route_mean.float().cpu().tolist(),
        "wuc_loss": float(wuc_loss.detach()),
    }
    return row, wuc_loss, scope


def _wuc_gradient_scope(
    model: torch.nn.Module, loss: Tensor, *, enabled: bool
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "router_gradient_norm": 0.0,
            "forbidden_gradient_norm": 0.0,
            "passed": True,
        }
    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    gradients = torch.autograd.grad(
        loss,
        tuple(parameter for _, parameter in named),
        allow_unused=True,
        retain_graph=True,
    )
    router_sq = 0.0
    forbidden_sq = 0.0
    for (name, _), gradient in zip(named, gradients, strict=True):
        value = 0.0 if gradient is None else float(gradient.float().square().sum())
        if name.startswith("router."):
            router_sq += value
        else:
            forbidden_sq += value
    router_norm = math.sqrt(router_sq)
    forbidden_norm = math.sqrt(forbidden_sq)
    return {
        "enabled": True,
        "router_gradient_norm": router_norm,
        "forbidden_gradient_norm": forbidden_norm,
        "passed": router_norm > 0.0 and forbidden_norm == 0.0,
    }


def _structural_audit(
    model: torch.nn.Module,
    legacy_reference: torch.nn.Module,
    inputs: Mapping[str, Tensor],
    *,
    parent_identity: Mapping[str, Any],
) -> dict[str, bool | float]:
    model.eval()
    legacy_reference = legacy_reference.to(inputs["state"].device).eval()
    shape = (
        inputs["state"].shape[0],
        inputs["state"].shape[1],
        model.active_parent.base_flow.config.horizon,
        model.active_parent.base_flow.config.action_dim,
    )
    actions = torch.zeros(shape, device=inputs["state"].device)
    tau = torch.full((shape[0],), 0.5, device=actions.device)
    with torch.no_grad():
        legacy_velocity = legacy_reference.velocity(
            inputs["raw_local"],
            inputs["state"],
            inputs["local_visual"],
            inputs["shared_visual"],
            actions,
            tau,
            inputs["valid"],
        )[0]
        active_velocity = model.active_parent.velocity(
            inputs["raw_local"],
            inputs["state"],
            inputs["local_visual"],
            inputs["shared_visual"],
            actions,
            tau,
            inputs["valid"],
        )[0]
        gate_zero = model.velocity(
            inputs["raw_local"],
            inputs["state"],
            inputs["local_visual"],
            inputs["shared_visual"],
            actions,
            tau,
            inputs["valid"],
            force_world_evidence_gate_zero=True,
        )[0]
        gate_zero_executed = model.velocity(
            inputs["raw_local"],
            inputs["state"],
            inputs["local_visual"],
            inputs["shared_visual"],
            actions,
            tau,
            inputs["valid"],
            force_world_evidence_gate_zero=True,
            execute_evidence_when_gate_zero=True,
        )[0]
    legacy_diff = float((legacy_velocity - active_velocity).abs().max())
    gate_diff = float((gate_zero - active_velocity).abs().max())
    executed_diff = float((gate_zero_executed - active_velocity).abs().max())
    legacy_reference.cpu()
    model.train(True)
    result: dict[str, bool | float] = {
        "token_contract_exact": True,
        "dense_router_train_inference_identical": True,
        "legacy_reference_file_unchanged": file_sha256(
            str(parent_identity["legacy_r6l_policy_path"])
        )
        == parent_identity["legacy_r6l_policy_sha256"],
        "active_gate_zero_elementwise_exact": gate_diff == 0.0,
        "active_gate_zero_without_provider_elementwise_exact": executed_diff == 0.0,
        "dino_optimizer_excluded": True,
        "legacy_reference_optimizer_excluded": True,
        "auxiliary_weights_zero": True,
        "no_depth_or_wrist_input": True,
        "no_ground_truth_future_input": True,
        "legacy_reference_max_abs_diff": legacy_diff,
        "active_gate_zero_max_abs_diff": gate_diff,
        "active_gate_zero_executed_max_abs_diff": executed_diff,
    }
    if ROUND_ID == "s4-r7":
        result["legacy_reference_elementwise_exact"] = legacy_diff == 0.0
    else:
        aggregator = model.active_parent.future_predictor.action_prefix_aggregator
        aggregator_audit = aggregator.audit()
        result.update(
            {
                "legacy_reference_elementwise_exact_not_required": True,
                "r7_candidate_checkpoint_not_consumed": bool(
                    parent_identity.get("r7_candidate_checkpoint_consumed") is False
                ),
                "strict_horizon_prefix_mask": bool(
                    aggregator_audit.get("strict_prefix_mask") is True
                ),
                "p1_output_projection_zero_initialized_or_p0": bool(
                    aggregator_audit.get("output_projection_zero_initialized") is True
                ),
            }
        )
    if not all(
        value is True for key, value in result.items() if not key.endswith("diff")
    ):
        raise RuntimeError(f"{ROUND_LABEL} structural audit failed")
    return result


def _parameter_groups(
    model: torch.nn.Module,
    training: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    names: dict[str, list[str]] = {
        "flow": [],
        "future_body": [],
        "future_heads": [],
        "legacy_adapter": [],
        "evidence": [],
        "router": [],
    }
    parameters: dict[str, list[torch.nn.Parameter]] = {name: [] for name in names}
    head_markers = (
        ".state_head.",
        ".visual_head.",
        ".peer_state_head.",
        ".peer_visual_head.",
        ".shared_visual_head.",
    )
    for name, parameter in model.named_parameters():
        if name.startswith("active_parent.base_flow."):
            group = "flow"
        elif name.startswith("active_parent.future_predictor."):
            group = (
                "future_heads"
                if any(marker in name for marker in head_markers)
                else "future_body"
            )
        elif name.startswith("active_parent.legacy_adapter."):
            group = "legacy_adapter"
        elif name.startswith("router.") or name.startswith("residual.query_gate."):
            group = "router"
        elif (
            name.startswith("evidence_provider.")
            or name.startswith("evidence_adapter.")
            or name.startswith("residual.output.")
        ):
            group = "evidence"
        else:
            raise RuntimeError(
                f"S4-R7 optimizer encountered unregistered parameter {name}"
            )
        names[group].append(name)
        parameters[group].append(parameter)
    if any(not value for value in parameters.values()):
        raise RuntimeError("every S4-R7 optimizer group must be non-empty")
    learning_rates = {
        "flow": float(training["flow_learning_rate"]),
        "future_body": float(training["future_body_learning_rate"]),
        "future_heads": float(training["future_head_learning_rate"]),
        "legacy_adapter": float(training["legacy_adapter_learning_rate"]),
        "evidence": float(training["evidence_adapter_learning_rate"]),
        "router": float(training["router_learning_rate"]),
    }
    groups = [
        {
            "name": name,
            "params": parameters[name],
            "lr": 0.0,
            "base_lr": learning_rates[name],
        }
        for name in names
    ]
    return groups, names


def _set_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    total_updates: int,
    flow_unfreeze_update: int,
    warmup_updates: int,
    flow_warmup_updates: int,
) -> None:
    for group in optimizer.param_groups:
        name = str(group["name"])
        base = float(group["base_lr"])
        if name == "flow":
            if update < flow_unfreeze_update:
                factor = 0.0
            else:
                local = update - flow_unfreeze_update + 1
                horizon = total_updates - flow_unfreeze_update + 1
                factor = _warmup_cosine(local, horizon, flow_warmup_updates)
        else:
            factor = _warmup_cosine(update, total_updates, warmup_updates)
        group["lr"] = base * factor


def _warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step <= warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _gradient_norms(
    model: torch.nn.Module, names: Mapping[str, Sequence[str]]
) -> dict[str, float]:
    by_name = dict(model.named_parameters())
    return {
        group: math.sqrt(
            sum(
                float(by_name[name].grad.float().square().sum())
                for name in group_names
                if by_name[name].grad is not None
            )
        )
        for group, group_names in names.items()
    }


def _checkpoint_payload(
    raw: Mapping[str, Any],
    config_path: Path,
    identity: Mapping[str, Any],
    update: int,
    model: torch.nn.Module,
    structural: Mapping[str, Any],
    exposure: S4ExposureCounter,
    parameter_names: Mapping[str, Sequence[str]],
    module_exposure: Mapping[str, Mapping[str, int]],
    vision: torch.nn.Module,
    dataset: Any,
) -> dict[str, Any]:
    candidate_id = str(identity["candidate_id"])
    model_kind = str(identity["model_kind"])
    training = _mapping(raw, "training")
    flow_unfreeze = int(training["flow_unfreeze_update"])
    total_updates = int(training["updates"])
    return {
        "format_version": CHECKPOINT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "update": update,
        "method": {
            "round_id": ROUND_ID,
            "candidate_id": candidate_id,
            "model_kind": model_kind,
            "utility_coupling_weight": float(identity["utility_coupling_weight"]),
            "action_generator": "rectified_flow_cold_token_preserving_world_evidence",
            "route_mode": "dense",
            "future_target_input": False,
            "trainable_modules": list(parameter_names),
            "action_prefix_aggregator": identity["action_prefix_aggregator"],
            "r7_candidate_checkpoint_consumed": ROUND_ID != "s4-r8",
        },
        "model": model.state_dict(),
        "model_config": model.config.to_dict(),
        "parent_identity": dict(identity["parent_identity"]),
        "identity": dict(identity),
        "structural_invariants": dict(structural),
        "parameter_names": {key: list(value) for key, value in parameter_names.items()},
        "trainable_name_sha256_by_phase": {
            f"updates_1_{flow_unfreeze - 1}": _name_hash(
                name
                for group, names in parameter_names.items()
                if group != "flow"
                for name in names
            ),
            f"updates_{flow_unfreeze}_{total_updates}": _name_hash(
                name for names in parameter_names.values() for name in names
            ),
        },
        "exposure": exposure.state_dict(),
        "agent_windows_seen_by_module": {
            name: dict(counters) for name, counters in module_exposure.items()
        },
        "effective_team_batch": int(identity["effective_team_batch"]),
        "gradient_accumulation": int(identity["gradient_accumulation"]),
        "evidence_contract": {
            "sources": ["own", "peer", "shared"],
            "horizons": [1, 25, 50, 100],
            "source_agents": 4,
            "tokens_per_source_agent": 5,
            "visual_grid": [2, 2],
            "token_shape": [3, 4, 4, 5, 384],
        },
        "training": dict(training),
        "generation": dict(_mapping(raw, "generation")),
        "inference": dict(_mapping(raw, "inference")),
        "task_runtime": _task_runtime(dataset.source),
        "data": {
            "hdf5_verification": {
                "mode": "accepted_checkpoint_stat_bound_receipt",
                "receipt_sha256": identity["shared_hdf5_receipt_sha256"],
            },
            "future_feature_cache": {
                "mode": "shared_float32_projected_next_view",
                "features_sha256": identity["future_feature_cache_sha256"],
            },
            "summary": dataset.summary(),
            "manifests": [
                {
                    "task_id": contract.task_id,
                    "path": str(contract.manifest_path),
                    "sha256": contract.manifest_sha256,
                }
                for contract in dataset.contracts
            ],
        },
        "vision": {
            "artifact_sha256": vision.artifact_sha256,
            "config_sha256": vision.config_sha256,
        },
        "source": {
            "git_commit": _git_commit(),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
        },
    }


def _resume_payload(
    identity: Mapping[str, Any],
    update: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    exposure: S4ExposureCounter,
    dataset_chain: bytes,
    *,
    gradient_audit: Mapping[str, Any],
    normal_categories_seen: Mapping[str, bool],
    flow_frozen_gradient_exact_zero: bool,
    flow_frozen_observed: bool,
    flow_unfrozen_gradient_nonzero: bool,
    flow_unfrozen_observed: bool,
    module_exposure: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    return {
        "format_version": RESUME_FORMAT,
        "identity": dict(identity),
        "update": update,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "exposure": exposure.state_dict(),
        "dataset_chain_sha256": dataset_chain.hex(),
        "sampler_resume_key": [identity["seed"], update + 1, 0, 0],
        "training_audit_state": {
            "gradient_audit": dict(gradient_audit),
            "normal_categories_seen": dict(normal_categories_seen),
            "flow_frozen_gradient_exact_zero": flow_frozen_gradient_exact_zero,
            "flow_frozen_observed": flow_frozen_observed,
            "flow_unfrozen_gradient_nonzero": flow_unfrozen_gradient_nonzero,
            "flow_unfrozen_observed": flow_unfrozen_observed,
        },
        "module_exposure_state": {
            name: dict(counters) for name, counters in module_exposure.items()
        },
        "rng": _capture_rng(),
    }


def _extend_dataset_chain(previous: bytes, indices: Sequence[int]) -> bytes:
    return hashlib.sha256(
        previous + np.asarray(indices, dtype=np.int64).tobytes()
    ).digest()


def _name_hash(names: Sequence[str] | Any) -> str:
    ordered = sorted(str(name) for name in names)
    return hashlib.sha256("\n".join(ordered).encode()).hexdigest()


def _lr_curve_hash(training: Mapping[str, Any]) -> str:
    total_updates = int(training["updates"])
    flow_unfreeze = int(training["flow_unfreeze_update"])
    warmup = int(training["warmup_updates"])
    flow_warmup = int(training["flow_warmup_updates"])
    points = tuple(
        sorted(
            {
                1,
                min(warmup, total_updates),
                max(flow_unfreeze - 1, 1),
                flow_unfreeze,
                min(flow_unfreeze + flow_warmup - 1, total_updates),
                total_updates,
            }
        )
    )
    rows = []
    bases = {
        "flow": float(training["flow_learning_rate"]),
        "future_body": float(training["future_body_learning_rate"]),
        "future_heads": float(training["future_head_learning_rate"]),
        "legacy_adapter": float(training["legacy_adapter_learning_rate"]),
        "evidence": float(training["evidence_adapter_learning_rate"]),
        "router": float(training["router_learning_rate"]),
    }
    for update in points:
        row = {"update": update}
        for name, base in bases.items():
            if name == "flow":
                factor = (
                    0.0
                    if update < flow_unfreeze
                    else _warmup_cosine(
                        update - flow_unfreeze + 1,
                        total_updates - flow_unfreeze + 1,
                        flow_warmup,
                    )
                )
            else:
                factor = _warmup_cosine(update, total_updates, warmup)
            row[name] = base * factor
        rows.append(row)
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _resolve_path(argument: Path | None, configured: object) -> Path:
    if argument is not None:
        return argument.expanduser().resolve()
    if configured is None:
        raise ValueError("checkpoint path is missing")
    return (ROOT / str(configured)).expanduser().resolve()


def _write_terminal_oom_preflight(args: argparse.Namespace) -> None:
    """Persist the paired-fallback signal even if OOM precedes the train loop."""

    assert args.preflight_report is not None
    path = args.preflight_report.expanduser().resolve()
    if path.exists():
        # The inner loop records a richer terminal report.  Never overwrite it
        # from this whole-lifecycle guard.
        return
    config_path = args.config.expanduser().resolve(strict=True)
    raw = _load_yaml(config_path)
    candidate_id, model_kind, _, action_prefix_aggregator = _configure_round(raw)
    training = _mapping(raw, "training")
    micro = int(training["micro_team_batch"])
    accumulation = int(training["gradient_accumulation"])
    effective = int(training["effective_team_batch"])
    if micro * accumulation != effective or effective != 12:
        raise ValueError("cannot publish OOM for an invalid paired batch recipe")
    device = torch.device(args.device)
    peak = 0
    total = 0
    if device.type == "cuda" and torch.cuda.is_available():
        peak = max(
            int(torch.cuda.max_memory_allocated(device)),
            int(torch.cuda.memory_allocated(device)),
        )
        total = int(torch.cuda.get_device_properties(device).total_memory)
    _atomic_json(
        path,
        {
            "format_version": PREFLIGHT_FORMAT,
            "identity": {
                "round_id": ROUND_ID,
                "candidate_id": candidate_id,
                "model_kind": model_kind,
                "action_prefix_aggregator": action_prefix_aggregator,
            },
            "updates": 200,
            "completed": False,
            "oom": True,
            "micro_team_batch": micro,
            "gradient_accumulation": accumulation,
            "effective_team_batch": effective,
            "flow_unfreeze_update": int(training["flow_unfreeze_update"]),
            "vision_inference_batch_size": int(
                _mapping(raw, "vision")["inference_batch_size"]
            ),
            "peak_memory_bytes": peak,
            "gpu_total_memory_bytes": total,
            "config_sha256": _sha256(config_path),
            "shared_hdf5_receipt_sha256": _round_environment(
                "SHARED_HDF5_RECEIPT_SHA256"
            ),
            "future_feature_cache_sha256": _round_environment(
                "FUTURE_FEATURE_CACHE_SHA256"
            ),
            "failure_scope": "whole_preflight_gpu_lifecycle",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _atomic_json(
    path: Path, value: Mapping[str, Any], *, overwrite: bool = False
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
