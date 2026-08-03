#!/usr/bin/env python3
"""Run the preregistered offline and closed-loop S4-R7 causal audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.accept_s4_r7 import (  # noqa: E402
    CANDIDATE_REPORT_FORMAT,
    CONDITIONS,
    STRUCTURAL_GATES,
    TASKS,
)
from scripts.evaluate_s2_r5_protected_team import (  # noqa: E402
    _dataset as _validation_dataset,
    _validation_batches,
    _validation_indices,
)
from scripts.s4_r7_model_io import build_s4_r7_model  # noqa: E402
from scripts.train_s2_r4_future_predictor import (  # noqa: E402
    _validate_artifact_dataset,
)
from scripts.train_s3_r6_world_action_flow import _model_inputs  # noqa: E402
from scripts.train_s4_r7_world_utility import CHECKPOINT_FORMAT  # noqa: E402
from scripts.train_static_rgb_act_moe import (  # noqa: E402
    _append_jsonl,
    _load_yaml,
    _mapping,
    _seed_everything,
    _sha256,
    _vision,
)
from train.s2_future_prediction import (  # noqa: E402
    file_sha256,
    load_s2_artifact,
)
from train.s2_grouped_trajectory import grouped_s2_batch  # noqa: E402
from train.s4_model_registry import validate_s4_r7_candidate  # noqa: E402
from train.world_action_flow_training import grouped_flow_matching_batch  # noqa: E402


UTILITY_FORMAT = "wam.robofactory.s4_r7.router_utility_spearman/1"
SOURCE_SHUFFLE_FORMAT = "wam.robofactory.s4_r7.source_shuffle_gate20/1"
CAUSAL_GATE_FORMAT = "wam.robofactory.s4_r7.legacy_scaled_zero_shuffle_gate20/1"
ARTIFACT_HASH_FORMAT = "wam.robofactory.s4_r7.artifact_hashes/1"
FORCED_FORMAT = "wam.robofactory.s4_r7.forced_evidence_errors/1"
GATE_FORMAT = "wam.robofactory.lpd_fixed_seed_gate/3"
GRADIENT_FORMAT = "wam.robofactory.s4_r7.gradient_audit/1"
EXPOSURE_FORMAT = "wam.robofactory.s4_r7.module_exposure/1"
RESUME_FORMAT = "wam.robofactory.s4_r7.world_utility.resume/1"
EVALUATION_ORDER = (
    "normal",
    "legacy_reference",
    "world_evidence_gate_zero",
    "shuffle_all",
    "all_world_gates_zero",
    "shuffle_own",
    "shuffle_peer",
    "shuffle_shared",
)
GROUP_NAMES = tuple(
    f"{source}@{horizon}"
    for source in ("own", "peer", "shared")
    for horizon in (1, 25, 50, 100)
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-log", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    progress_log = args.progress_log.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate report {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_log.parent.mkdir(parents=True, exist_ok=True)

    raw = _load_yaml(config_path)
    candidate_id, model_kind, utility_weight = validate_s4_r7_candidate(raw)
    training = _mapping(raw, "training")
    total_updates = int(training["updates"])
    checkpoint_sha256 = file_sha256(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(saved, Mapping) or saved.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not an S4-R7 world-utility policy")
    method = _mapping(saved, "method")
    if (
        saved.get("update") != total_updates
        or method.get("round_id") != "s4-r7"
        or method.get("candidate_id") != candidate_id
        or method.get("model_kind") != model_kind
        or float(method.get("utility_coupling_weight", -1.0)) != utility_weight
    ):
        raise ValueError("checkpoint method/update differs from the candidate config")
    source = _mapping(saved, "source")
    if source.get("config_sha256") != _sha256(config_path):
        raise ValueError("checkpoint/config SHA256 identity differs")

    candidate_root = output.parent.parent
    train_root = candidate_root / "train"
    validation_root = output.parent
    forced_path = validation_root / "forced_evidence_errors.npz"
    utility_path = validation_root / "router_utility_spearman.json"
    source_path = validation_root / "source_shuffle_gate20.json"
    causal_path = validation_root / "legacy_scaled_zero_shuffle_gate20.json"
    artifact_path = validation_root / "artifact_hashes.json"
    evidence_bank = validation_root / "predicted_future_bank"
    gates_root = validation_root / "gate20"
    gradient_path = train_root / "parameter_gradient_audit.json"
    exposure_path = train_root / "module_exposure.json"
    resume_path = checkpoint_path.with_name("resume.pt")

    gradient = _read_json(gradient_path)
    exposure = _read_json(exposure_path)
    _validate_training_audits(
        gradient,
        exposure,
        saved,
        resume_path=resume_path,
        candidate_id=candidate_id,
        total_updates=total_updates,
        effective_team_batch=int(training["effective_team_batch"]),
    )
    structural = dict(_mapping(saved, "structural_invariants"))
    if any(structural.get(name) is not True for name in STRUCTURAL_GATES):
        raise ValueError("checkpoint did not preserve every structural invariant")

    offline = _reuse_offline_audit(
        forced_path,
        utility_path,
        checkpoint_sha256=checkpoint_sha256,
        candidate_id=candidate_id,
    )
    if offline is None:
        _preserve_if_exists(forced_path)
        _preserve_if_exists(utility_path)
        offline = _run_offline_audit(
            raw,
            saved,
            config_path=config_path,
            checkpoint_sha256=checkpoint_sha256,
            candidate_id=candidate_id,
            model_kind=model_kind,
            device=torch.device(args.device),
            forced_path=forced_path,
            utility_path=utility_path,
            progress_log=progress_log,
        )
    structural.update(dict(_mapping(offline, "structural_invariants")))
    if any(structural.get(name) is not True for name in STRUCTURAL_GATES):
        raise RuntimeError("trained checkpoint failed a structural invariant")

    # The offline model/vision objects are gone before a rollout child loads the
    # same checkpoint on this candidate's single visible GPU.
    torch.cuda.empty_cache()
    gate_reports: dict[str, Mapping[str, Any]] = {}
    for condition in EVALUATION_ORDER:
        gate_reports[condition] = _run_or_reuse_gate(
            condition,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            candidate_id=candidate_id,
            model_kind=model_kind,
            output_root=gates_root / condition,
            evidence_bank=evidence_bank,
            progress_log=progress_log,
        )
        if condition == "normal" and not _donor_bank_complete(evidence_bank):
            _preserve_if_exists(gates_root / condition)
            gate_reports[condition] = _run_or_reuse_gate(
                condition,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                candidate_id=candidate_id,
                model_kind=model_kind,
                output_root=gates_root / condition,
                evidence_bank=evidence_bank,
                progress_log=progress_log,
            )
            if not _donor_bank_complete(evidence_bank):
                raise RuntimeError("normal Gate20 did not create both donor episodes")

    conditions = {
        name: _candidate_condition(gate_reports[name]) for name in CONDITIONS
    }
    macros = {name: _condition_macro(row) for name, row in conditions.items()}
    source_report = {
        "format_version": SOURCE_SHUFFLE_FORMAT,
        "round_id": "s4-r7",
        "candidate_id": candidate_id,
        "checkpoint_sha256": checkpoint_sha256,
        "task_order": list(TASKS),
        "protocol": "within_task_different_episode_predicted_future",
        "normal_macro": macros["normal"],
        "conditions": {
            name: conditions[name]
            for name in ("normal", "shuffle_own", "shuffle_peer", "shuffle_shared")
        },
        "source_gaps": {
            source_name: macros["normal"] - macros[f"shuffle_{source_name}"]
            for source_name in ("own", "peer", "shared")
        },
        "created_at": _now(),
    }
    causal_report = {
        "format_version": CAUSAL_GATE_FORMAT,
        "round_id": "s4-r7",
        "candidate_id": candidate_id,
        "checkpoint_sha256": checkpoint_sha256,
        "task_order": list(TASKS),
        "conditions": {
            name: conditions[name]
            for name in (
                "legacy_reference",
                "normal",
                "world_evidence_gate_zero",
                "all_world_gates_zero",
                "shuffle_all",
            )
        },
        "macros": {
            name: macros[name]
            for name in (
                "legacy_reference",
                "normal",
                "world_evidence_gate_zero",
                "all_world_gates_zero",
                "shuffle_all",
            )
        },
        "causal_gates": {
            "normal_not_below_legacy": macros["normal"]
            >= macros["legacy_reference"],
            "normal_strictly_above_world_evidence_gate_zero": macros["normal"]
            > macros["world_evidence_gate_zero"],
            "normal_strictly_above_shuffle_all": macros["normal"]
            > macros["shuffle_all"],
        },
        "all_world_gates_zero_report_only": True,
        "created_at": _now(),
    }
    _write_or_validate_json(source_path, source_report, checkpoint_sha256)
    _write_or_validate_json(causal_path, causal_report, checkpoint_sha256)

    manifest_files = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "resume": resume_path,
        "parameter_gradient_audit": gradient_path,
        "module_exposure": exposure_path,
        "forced_evidence_errors": forced_path,
        "router_utility_spearman": utility_path,
        "source_shuffle_gate20": source_path,
        "legacy_scaled_zero_shuffle_gate20": causal_path,
        **{
            f"gate20_{name}": gates_root / name / "gate_summary.json"
            for name in EVALUATION_ORDER
        },
    }
    artifact_hashes = {
        "format_version": ARTIFACT_HASH_FORMAT,
        "round_id": "s4-r7",
        "candidate_id": candidate_id,
        "checkpoint_sha256": checkpoint_sha256,
        "files": {
            name: {
                "path": str(path.resolve(strict=True)),
                "sha256": file_sha256(path.resolve(strict=True)),
            }
            for name, path in manifest_files.items()
        },
        "ancestors": dict(_mapping(saved, "parent_identity")),
        "dataset_manifests": list(_mapping(saved, "data").get("manifests", [])),
        "source": dict(source),
        "created_at": _now(),
    }
    _write_or_validate_json(artifact_path, artifact_hashes, checkpoint_sha256)

    wuc = _mapping(gradient, "wuc_only")
    utility_calibration = {
        "forced_evidence_audit_present": True,
        "utility_coupling_weight": utility_weight,
        "spearman": float(offline["spearman"]),
        "episode_bootstrap_95_lower": float(
            offline["episode_bootstrap_95_lower"]
        ),
        "episode_bootstrap_95_upper": float(
            offline["episode_bootstrap_95_upper"]
        ),
        "wuc_router_gradient_norm": float(wuc.get("router_gradient_norm", 0.0)),
        "wuc_forbidden_gradient_norm": float(
            wuc.get("forbidden_gradient_norm", 0.0)
        ),
        "wuc_backward_disabled": utility_weight == 0.0
        and wuc.get("enabled") is False,
    }
    report = {
        "format_version": CANDIDATE_REPORT_FORMAT,
        "identity": {
            "round_id": "s4-r7",
            "candidate_id": candidate_id,
            "model_kind": model_kind,
        },
        "created_at": _now(),
        "checkpoint_sha256": checkpoint_sha256,
        "structural_invariants": structural,
        "training_audits": {
            "checkpoint_update_30000": saved.get("update") == 30_000,
            "parameter_gradient_audit_passed": gradient.get("passed") is True,
            "module_exposure_passed": exposure.get("passed") is True,
            "formal_budget_complete": exposure.get("formal_budget_complete")
            is True,
        },
        "reports": {
            "parameter_gradient_audit": _report_reference(gradient_path),
            "module_exposure": _report_reference(exposure_path),
            "forced_evidence_errors": _report_reference(forced_path),
            "router_utility_spearman": _report_reference(utility_path),
            "source_shuffle_gate20": _report_reference(source_path),
            "legacy_scaled_zero_shuffle_gate20": _report_reference(causal_path),
            "artifact_hashes": _report_reference(artifact_path),
        },
        "gate20": conditions,
        "utility_calibration": utility_calibration,
        "heldout_flow_error": float(offline["heldout_flow_error"]),
        "validation_selection_sha256": offline["selection_sha256"],
    }
    _atomic_json(output, report)
    _progress(
        progress_log,
        event="candidate_report_complete",
        condition="special_acceptance_inputs",
        detail=f"four core plus four diagnostic Gate20 conditions complete; report={output}",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _validate_training_audits(
    gradient: Mapping[str, Any],
    exposure: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    resume_path: Path,
    candidate_id: str,
    total_updates: int,
    effective_team_batch: int,
) -> None:
    if (
        gradient.get("format_version") != GRADIENT_FORMAT
        or gradient.get("candidate_id") != candidate_id
        or gradient.get("passed") is not True
    ):
        raise ValueError("formal parameter gradient audit did not pass")
    if (
        exposure.get("format_version") != EXPOSURE_FORMAT
        or exposure.get("candidate_id") != candidate_id
        or exposure.get("passed") is not True
        or exposure.get("formal_budget_complete") is not True
        or exposure.get("team_windows_seen")
        != total_updates * effective_team_batch
    ):
        raise ValueError("formal per-module exposure audit did not pass")
    by_module = exposure.get("agent_windows_seen_by_module")
    if not isinstance(by_module, Mapping) or set(by_module) != {
        "flow",
        "future_body",
        "future_heads",
        "legacy_adapter",
        "evidence",
        "router",
    }:
        raise ValueError("module exposure does not cover the exact optimizer groups")
    if checkpoint.get("agent_windows_seen_by_module") != by_module:
        raise ValueError("checkpoint and exposure report module counters differ")
    resume = torch.load(resume_path.resolve(strict=True), map_location="cpu", weights_only=False)
    if (
        not isinstance(resume, Mapping)
        or resume.get("format_version") != RESUME_FORMAT
        or not 0 <= int(resume.get("update", -1)) < total_updates
        or _mapping(resume, "identity").get("candidate_id") != candidate_id
    ):
        raise ValueError("formal resume artifact is not a valid recoverable state")


def _run_offline_audit(
    raw: Mapping[str, Any],
    saved: Mapping[str, Any],
    *,
    config_path: Path,
    checkpoint_sha256: str,
    candidate_id: str,
    model_kind: str,
    device: torch.device,
    forced_path: Path,
    utility_path: Path,
    progress_log: Path,
) -> dict[str, Any]:
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("S4-R7 causal evaluation requires one visible CUDA GPU")
    _seed_everything(int(_mapping(raw, "evaluation").get("bootstrap_seed", 70707)))
    model, legacy_reference, parent_identity = build_s4_r7_model(raw, device=device)
    if dict(_mapping(saved, "parent_identity")) != parent_identity:
        raise ValueError("evaluation ancestor identity differs from checkpoint")
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    del legacy_reference
    artifact_path = (
        ROOT / str(_mapping(raw, "artifacts")["pca_statistics"])
    ).resolve(strict=True)
    artifact = load_s2_artifact(artifact_path, device=device)
    dataset = _validation_dataset(raw)
    _validate_artifact_dataset(artifact, dataset)
    vision = _vision(raw).to(device).eval()
    evaluation = _mapping(raw, "evaluation")
    selected = _validation_indices(
        dataset,
        windows_per_episode=int(evaluation.get("offline_windows_per_episode", 4)),
    )
    selection_payload = [
        {
            "task_id": task,
            "dataset_index": index,
            "episode_index": dataset.sample_lineage(index).episode_index,
            "decision_t": dataset.sample_lineage(index).decision_t,
        }
        for task, indices in selected.items()
        for index in indices
    ]
    selection_sha256 = hashlib.sha256(
        json.dumps(selection_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    batch_size = int(evaluation.get("offline_batch_size", 2))
    workers = int(evaluation.get("offline_num_workers", 2))
    pca = _mapping(raw, "pca")
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    query_correlations: dict[tuple[str, int], list[float]] = defaultdict(list)
    flow_errors: list[float] = []
    structural = dict(_mapping(saved, "structural_invariants"))
    max_gate_zero_diff = 0.0
    max_executed_zero_diff = 0.0

    with torch.inference_mode():
        for task_index, task in enumerate(TASKS):
            indices = selected[task]
            batches = _validation_batches(indices, batch_size=batch_size)
            loader = DataLoader(
                dataset,
                batch_sampler=batches,
                num_workers=workers,
                pin_memory=True,
            )
            for batch_index, raw_batch in enumerate(loader, start=1):
                grouped = grouped_s2_batch(raw_batch)
                inputs = _model_inputs(
                    vision, grouped, artifact, device=device, pca=pca
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
                    gate_zero = model.velocity(
                        inputs["raw_local"],
                        inputs["state"],
                        inputs["local_visual"],
                        inputs["shared_visual"],
                        action_inputs,
                        tau,
                        inputs["valid"],
                        force_world_evidence_gate_zero=True,
                    )[0]
                    executed_zero = model.velocity(
                        inputs["raw_local"],
                        inputs["state"],
                        inputs["local_visual"],
                        inputs["shared_visual"],
                        action_inputs,
                        tau,
                        inputs["valid"],
                        force_world_evidence_gate_zero=True,
                        execute_evidence_when_gate_zero=True,
                    )[0]
                active_parent = _tensor(diagnostics, "active_parent_velocity")
                max_gate_zero_diff = max(
                    max_gate_zero_diff,
                    float((gate_zero - active_parent).abs().max()),
                )
                max_executed_zero_diff = max(
                    max_executed_zero_diff,
                    float((executed_zero - active_parent).abs().max()),
                )
                forced = model.forced_evidence_audit(
                    diagnostics,
                    target_velocity,
                    inputs["valid"],
                    valid_action_query_mask=valid_queries,
                )
                router_pi = _tensor(diagnostics, "router_pi").float()
                team_flow = _per_team_flow_error(
                    prediction, target_velocity, inputs["valid"], valid_action
                )
                flow_errors.extend(team_flow.cpu().tolist())
                batch = int(inputs["valid"].shape[0])
                task_ids = np.asarray([task] * batch)
                episodes = grouped["episode_index"].cpu().numpy().astype(np.int64)
                decisions = grouped["decision_t"].cpu().numpy().astype(np.int64)
                arrays["task_id"].append(task_ids)
                arrays["task_index"].append(
                    np.full(batch, task_index, dtype=np.int64)
                )
                arrays["episode_index"].append(episodes)
                arrays["decision_t"].append(decisions)
                arrays["valid_agent_mask"].append(inputs["valid"].cpu().numpy())
                arrays["group_mask"].append(forced.group_mask.cpu().numpy())
                arrays["valid_query_mask"].append(
                    forced.valid_query_mask.cpu().numpy()
                )
                errors = forced.velocity_errors.float().cpu().numpy()
                pi = router_pi.cpu().numpy()
                arrays["velocity_errors"].append(errors)
                arrays["negative_velocity_errors"].append(-errors)
                arrays["router_pi"].append(pi)
                group_mask = forced.group_mask.cpu().numpy()
                query_mask = forced.valid_query_mask.cpu().numpy()
                for item in range(batch):
                    key = (task, int(episodes[item]))
                    for agent in range(query_mask.shape[1]):
                        valid_groups = group_mask[item, agent]
                        if int(valid_groups.sum()) < 2:
                            continue
                        for query in range(query_mask.shape[2]):
                            if not query_mask[item, agent, query]:
                                continue
                            query_correlations[key].append(
                                _spearman(
                                    pi[item, agent, query, valid_groups],
                                    -errors[item, agent, query, valid_groups],
                                )
                            )
                _progress(
                    progress_log,
                    event="offline_forced_evidence_batch",
                    condition="offline_forced_evidence",
                    task=task,
                    episode=batch_index,
                    episodes_total=len(batches),
                    detail=f"held-out forced audit batch {batch_index}/{len(batches)}",
                )

    structural["active_gate_zero_elementwise_exact"] = max_gate_zero_diff == 0.0
    structural["active_gate_zero_without_provider_elementwise_exact"] = (
        max_executed_zero_diff == 0.0
    )
    structural["active_gate_zero_max_abs_diff"] = max_gate_zero_diff
    structural["active_gate_zero_executed_max_abs_diff"] = max_executed_zero_diff
    if any(structural.get(name) is not True for name in STRUCTURAL_GATES):
        raise RuntimeError("offline trained-checkpoint structural audit failed")
    concatenated = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
    _atomic_npz(
        forced_path,
        **concatenated,
        format_version=np.asarray(FORCED_FORMAT),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        candidate_id=np.asarray(candidate_id),
        model_kind=np.asarray(model_kind),
        selection_sha256=np.asarray(selection_sha256),
        group_names=np.asarray(GROUP_NAMES),
    )
    episode_rows = [
        {
            "task_id": task,
            "episode_index": episode,
            "valid_action_queries": len(values),
            "mean_query_spearman": float(np.mean(values)),
        }
        for (task, episode), values in sorted(query_correlations.items())
        if values
    ]
    if not episode_rows:
        raise RuntimeError("held-out forced audit produced no valid episode correlation")
    episode_values = np.asarray(
        [row["mean_query_spearman"] for row in episode_rows], dtype=np.float64
    )
    bootstrap_samples = int(evaluation.get("bootstrap_samples", 10_000))
    bootstrap_seed = int(evaluation.get("bootstrap_seed", 70_707))
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(
        0,
        len(episode_values),
        size=(bootstrap_samples, len(episode_values)),
    )
    means = episode_values[draws].mean(axis=1)
    utility = {
        "format_version": UTILITY_FORMAT,
        "round_id": "s4-r7",
        "candidate_id": candidate_id,
        "model_kind": model_kind,
        "checkpoint_sha256": checkpoint_sha256,
        "selection_sha256": selection_sha256,
        "spearman": float(episode_values.mean()),
        "episode_bootstrap_95_lower": float(np.quantile(means, 0.025)),
        "episode_bootstrap_95_upper": float(np.quantile(means, 0.975)),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "heldout_flow_error": float(np.mean(flow_errors)),
        "episodes": len(episode_rows),
        "windows": int(concatenated["episode_index"].shape[0]),
        "episode_rows": episode_rows,
        "structural_invariants": structural,
        "forced_evidence_errors": _report_reference(forced_path),
        "created_at": _now(),
    }
    _atomic_json(utility_path, utility)
    dataset.close()
    del model, vision, artifact, saved
    torch.cuda.empty_cache()
    return utility


def _reuse_offline_audit(
    forced_path: Path,
    utility_path: Path,
    *,
    checkpoint_sha256: str,
    candidate_id: str,
) -> Mapping[str, Any] | None:
    if not forced_path.is_file() or not utility_path.is_file():
        return None
    try:
        utility = _read_json(utility_path)
        if (
            utility.get("format_version") != UTILITY_FORMAT
            or utility.get("checkpoint_sha256") != checkpoint_sha256
            or utility.get("candidate_id") != candidate_id
            or not np.isfinite(float(utility.get("heldout_flow_error", np.nan)))
        ):
            return None
        with np.load(forced_path, allow_pickle=False) as value:
            if (
                str(value["format_version"].item()) != FORCED_FORMAT
                or str(value["checkpoint_sha256"].item()) != checkpoint_sha256
                or str(value["candidate_id"].item()) != candidate_id
            ):
                return None
        reference = _mapping(utility, "forced_evidence_errors")
        if reference.get("sha256") != file_sha256(forced_path):
            return None
        return utility
    except (KeyError, OSError, ValueError):
        return None


def _run_or_reuse_gate(
    condition: str,
    *,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    candidate_id: str,
    model_kind: str,
    output_root: Path,
    evidence_bank: Path,
    progress_log: Path,
) -> Mapping[str, Any]:
    summary_path = output_root / "gate_summary.json"
    if summary_path.is_file():
        try:
            return _validate_gate_summary(
                _read_json(summary_path),
                condition=condition,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                candidate_id=candidate_id,
                model_kind=model_kind,
            )
        except (KeyError, OSError, ValueError):
            _preserve_if_exists(output_root)
    _progress(
        progress_log,
        event="gate20_condition_start",
        condition=condition,
        detail=f"starting five-task Gate20 condition {condition}",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "LPD_POLICY_KIND": "s4_flow",
            "LPD_CONFIG": str(config_path),
            "LPD_CHECKPOINT": str(checkpoint_path),
            "LPD_GATE_MODE": "gate",
            "LPD_EPISODES": "20",
            "LPD_SEED_START": "900",
            "LPD_EXPERIMENT_SLUG": f"s4_r7_{candidate_id.lower()}_{condition}",
            "LPD_RUN_ID": f"s4_r7_{candidate_id.lower()}_{condition}",
            "LPD_OUTPUT_ROOT": str(output_root),
            "S4_R7_INTERVENTION": condition,
            "S4_R7_EVIDENCE_BANK_DIR": str(evidence_bank),
            "S4_R7_ROLLOUT_PROGRESS": str(progress_log),
        }
    )
    result = subprocess.run(
        ["bash", "scripts/run_lpd_single_5090.sh", "gate"],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Gate20 condition {condition} exited {result.returncode}"
        )
    summary = _validate_gate_summary(
        _read_json(summary_path),
        condition=condition,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        candidate_id=candidate_id,
        model_kind=model_kind,
    )
    _progress(
        progress_log,
        event="gate20_condition_complete",
        condition=condition,
        detail=f"completed five-task Gate20 condition {condition}",
    )
    return summary


def _validate_gate_summary(
    value: Mapping[str, Any],
    *,
    condition: str,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    candidate_id: str,
    model_kind: str,
) -> Mapping[str, Any]:
    if (
        value.get("format_version") != GATE_FORMAT
        or value.get("mode") != "gate"
        or tuple(value.get("task_order", ())) != TASKS
    ):
        raise ValueError("Gate20 summary format/task scope differs")
    seed = _mapping(value, "seed_protocol")
    if seed.get("seed_start") != 900 or seed.get("episodes_per_task") != 20:
        raise ValueError("Gate20 summary seed protocol differs")
    candidate = _mapping(value, "candidate")
    client = _mapping(candidate, "client")
    policy = _mapping(client, "policy")
    expected_action_source = (
        "s4_r7_token_preserving_world_flow"
        if candidate_id == "P0"
        else "s4_r7_world_utility_coupled_flow"
    )
    if (
        candidate.get("policy_kind") != "s4_flow"
        or candidate.get("checkpoint_sha256") != checkpoint_sha256
        or Path(str(candidate.get("checkpoint"))).resolve(strict=True)
        != checkpoint_path
        or Path(str(candidate.get("config"))).resolve(strict=True) != config_path
        or policy.get("world_intervention") != condition
        or policy.get("model_kind") != model_kind
        or policy.get("action_source") != expected_action_source
    ):
        raise ValueError("Gate20 client/checkpoint/intervention identity differs")
    for task in TASKS:
        row = _mapping(value, task)
        episodes = row.get("episodes")
        if not isinstance(episodes, list) or [item.get("seed") for item in episodes] != list(
            range(900, 920)
        ):
            raise ValueError(f"Gate20 {condition}/{task} episodes are not paired")
    return value


def _candidate_condition(gate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_order": list(TASKS),
        "tasks": {task: dict(_mapping(gate, task)) for task in TASKS},
    }


def _condition_macro(value: Mapping[str, Any]) -> float:
    tasks = _mapping(value, "tasks")
    return sum(float(_mapping(tasks, task)["success_rate"]) for task in TASKS) / len(
        TASKS
    )


def _donor_bank_complete(root: Path) -> bool:
    return all(
        (root / task / f"episode_{episode:03d}.pt").is_file()
        and (root / task / f"episode_{episode:03d}.pt").stat().st_size > 0
        for task in TASKS
        for episode in (0, 1)
    )


def _per_team_flow_error(
    prediction: Tensor,
    target: Tensor,
    valid_agents: Tensor,
    valid_action: Tensor,
) -> Tensor:
    squared = (prediction.float() - target.float()).square().mean(dim=-1)
    valid = valid_agents[:, :, None] & valid_action[:, None]
    per_agent = torch.where(valid, squared, 0).sum(dim=-1) / valid.sum(
        dim=-1
    ).clamp_min(1)
    return (
        torch.where(valid_agents, per_agent, 0).sum(dim=-1)
        / valid_agents.sum(dim=-1).clamp_min(1)
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 2:
        raise ValueError("Spearman inputs need the same length >=2")
    left_rank = _rankdata(left.astype(np.float64, copy=False))
    right_rank = _rankdata(right.astype(np.float64, copy=False))
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    return 0.0 if denominator == 0.0 else float(left_rank.dot(right_rank) / denominator)


def _rankdata(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(value.size, dtype=np.float64)
    start = 0
    while start < value.size:
        stop = start + 1
        while stop < value.size and value[order[stop]] == value[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _tensor(value: Mapping[str, Any], key: str) -> Tensor:
    result = value.get(key)
    if not isinstance(result, Tensor):
        raise TypeError(f"diagnostics.{key} must be a Tensor")
    return result


def _report_reference(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_or_validate_json(
    path: Path, value: Mapping[str, Any], checkpoint_sha256: str
) -> None:
    if path.exists():
        observed = _read_json(path)
        if observed.get("checkpoint_sha256") == checkpoint_sha256:
            comparable_observed = dict(observed)
            comparable_value = dict(value)
            comparable_observed.pop("created_at", None)
            comparable_value.pop("created_at", None)
            if comparable_observed == comparable_value:
                return
        _preserve_if_exists(path)
    _atomic_json(path, value)


def _preserve_if_exists(path: Path) -> None:
    if not path.exists():
        return
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    target = path.with_name(f"{path.name}.superseded_{suffix}_{os.getpid()}")
    path.rename(target)


def _progress(
    path: Path,
    *,
    event: str,
    condition: str,
    detail: str,
    task: str | None = None,
    episode: int | None = None,
    episodes_total: int | None = None,
) -> None:
    _append_jsonl(
        path,
        {
            "event": event,
            "program": "evaluate_s4_r7_causal.py",
            "condition": condition,
            "task": task,
            "episode": episode,
            "episodes_total": episodes_total,
            "detail": detail,
            "created_at": _now(),
        },
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
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
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
