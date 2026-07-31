#!/usr/bin/env python3
"""Run the zero-training protected-own/team-source S2-R4 diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wam_multimodal import (  # noqa: E402
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
    ProtectedHybridFuturePredictor,
    TeamSharedFuturePredictorConfig,
    exact_own_difference,
)
from scripts.compose_s2_r4_hybrid_checkpoint import (  # noqa: E402
    FORMAT_VERSION as MANIFEST_FORMAT,
)
from scripts.evaluate_s2_r4_future_predictor import (  # noqa: E402
    _action_equivalence_smoke,
    _dataset,
    _task_metrics,
    _validation_batches,
    _validation_indices,
)
from scripts.train_s2_r4_future_predictor import CHECKPOINT_FORMAT  # noqa: E402
from scripts.train_static_rgb_act_moe import (  # noqa: E402
    _append_jsonl,
    _load_yaml,
    _mapping,
    _vision,
)
from train.s2_future_prediction import (  # noqa: E402
    encode_local_visual_targets,
    encode_shared_visual_targets,
    file_sha256,
    load_s2_artifact,
    masked_future_prediction_losses,
    normalized_persistence_state,
    normalized_persistence_visual,
    normalized_state_delta,
)
from train.s2_grouped_trajectory import grouped_s2_batch  # noqa: E402
from train.s2_model_registry import (  # noqa: E402
    validate_s2_r4_hybrid_diagnostic,
)
from train.s2_r4_future_prediction import (  # noqa: E402
    masked_peer_future_prediction_losses,
    masked_shared_future_prediction_losses,
    peer_actions_shuffled_by_focal,
)


FORMAT_VERSION = "wam.robofactory.s2_r4.protected_hybrid_diagnostic/1"
REQUIRED_TASKS = {
    "camera_alignment",
    "lift_barrier",
    "long_pipeline_delivery",
    "take_photo",
    "three_robots_stack_cube",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hybrid-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-log", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    manifest_path = args.hybrid_manifest.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    progress_log = args.progress_log.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite hybrid diagnostic {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_log.parent.mkdir(parents=True, exist_ok=True)

    raw = _load_yaml(config_path)
    model_kind = validate_s2_r4_hybrid_diagnostic(_mapping(raw, "round"))
    manifest = _json_mapping(manifest_path)
    if (
        manifest.get("format_version") != MANIFEST_FORMAT
        or manifest.get("model_kind") != model_kind
        or manifest.get("training_performed") is not False
        or manifest.get("optimizer_created") is not False
        or manifest.get("statistics_fitted") is not False
    ):
        raise ValueError("invalid protected hybrid composition manifest")
    sources = _mapping(manifest, "sources")
    own_identity = _mapping(sources, "protected_own")
    team_identity = _mapping(sources, "team")
    own_path = Path(str(own_identity["path"])).resolve(strict=True)
    team_path = Path(str(team_identity["path"])).resolve(strict=True)
    source_hashes_before = {
        "protected_own": file_sha256(own_path),
        "team": file_sha256(team_path),
    }
    if (
        source_hashes_before["protected_own"] != own_identity.get("sha256")
        or source_hashes_before["team"] != team_identity.get("sha256")
    ):
        raise ValueError("hybrid source hash changed after composition")

    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("S2-R4 hybrid evaluation requires one visible GPU")
    own_saved = _source_checkpoint(own_path, candidate="P0")
    team_saved = _source_checkpoint(team_path, candidate="P1")
    local_config = LocalFuturePredictorConfig.from_dict(
        dict(_mapping(own_saved, "model_config"))
    )
    if team_saved.get("model_config") != own_saved.get("model_config"):
        raise ValueError("hybrid source local configurations differ")
    team_config = TeamSharedFuturePredictorConfig.from_dict(
        dict(_mapping(team_saved, "team_model_config"))
    )
    reference = LocalActionConditionedFuturePredictor(local_config).to(device)
    reference.load_state_dict(_mapping(own_saved, "model"), strict=True)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    hybrid = ProtectedHybridFuturePredictor(local_config, team_config).to(device)
    hybrid.load_sources(
        own_state_dict=_mapping(own_saved, "model"),
        team_state_dict=_mapping(team_saved, "model"),
    )
    if any(parameter.requires_grad for parameter in hybrid.parameters()):
        raise RuntimeError("protected hybrid unexpectedly has trainable parameters")

    artifact_path = (
        ROOT / str(_mapping(raw, "artifacts")["pca_statistics"])
    ).resolve(strict=True)
    artifact_sha256 = file_sha256(artifact_path)
    if artifact_sha256 != own_saved.get("future_artifacts_sha256") or (
        artifact_sha256 != team_saved.get("future_artifacts_sha256")
    ):
        raise ValueError("hybrid source PCA/statistics identities differ")
    artifact = load_s2_artifact(artifact_path, device=device)
    vision = _vision(raw).to(device).eval()
    dataset = _dataset(raw)
    evaluation = _mapping(raw, "evaluation")
    selected = _validation_indices(
        dataset,
        windows_per_episode=int(evaluation.get("windows_per_episode", 4)),
    )
    selection_payload = [
        {
            "task_id": task_id,
            "dataset_index": index,
            "episode_index": dataset.sample_lineage(index).episode_index,
            "decision_t": dataset.sample_lineage(index).decision_t,
        }
        for task_id, indices in selected.items()
        for index in indices
    ]
    selection_sha256 = hashlib.sha256(
        json.dumps(selection_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    batch_size = int(evaluation.get("batch_size", 4))
    batches_by_task = {
        task_id: _validation_batches(indices, batch_size=batch_size)
        for task_id, indices in selected.items()
    }
    total_batches = sum(len(batches) for batches in batches_by_task.values())
    completed_batches = 0
    pca = _mapping(raw, "pca")
    records: dict[str, list[dict[str, float | int | bool]]] = {}
    exact_by_task: dict[str, dict[str, float | bool]] = {}
    with torch.inference_mode():
        for task_id, indices in selected.items():
            task_records: list[dict[str, float | int | bool]] = []
            task_exact = _empty_exact()
            batches = batches_by_task[task_id]
            loader = DataLoader(
                dataset,
                batch_sampler=batches,
                num_workers=int(evaluation.get("num_workers", 2)),
                pin_memory=True,
            )
            for batch_index, raw_batch in enumerate(loader, start=1):
                grouped = grouped_s2_batch(raw_batch)
                current_visual, target_visual = encode_local_visual_targets(
                    vision,
                    grouped,
                    artifact,
                    device=device,
                    grid_height=int(pca["grid_height"]),
                    grid_width=int(pca["grid_width"]),
                )
                current_shared, target_shared, persistence_shared = (
                    encode_shared_visual_targets(
                        vision,
                        grouped,
                        artifact,
                        device=device,
                        grid_height=int(pca["grid_height"]),
                        grid_width=int(pca["grid_width"]),
                    )
                )
                current_state = grouped["current_state"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                actions = grouped["candidate_actions"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                valid_agents = grouped["valid_agent_mask"].to(
                    device=device, dtype=torch.bool
                )
                state_valid = grouped["future_state_valid_mask"].to(
                    device=device, dtype=torch.bool
                )
                visual_valid = grouped["future_agent_visual_valid_mask"].to(
                    device=device, dtype=torch.bool
                )
                shared_valid = grouped["future_shared_visual_valid_mask"].to(
                    device=device, dtype=torch.bool
                )
                target_state = normalized_state_delta(grouped, artifact, device=device)
                reference_own = reference(
                    current_state,
                    current_visual,
                    actions,
                    valid_agents,
                    valid_agents,
                )
                normal = hybrid(
                    current_state,
                    current_visual,
                    current_shared,
                    actions,
                    valid_agents,
                )
                exact = exact_own_difference(reference_own, normal)
                task_exact = _merge_exact(task_exact, exact)
                reference_loss = masked_future_prediction_losses(
                    reference_own[0],
                    target_state,
                    state_valid,
                    reference_own[1],
                    target_visual,
                    visual_valid,
                )
                hybrid_own_loss = masked_future_prediction_losses(
                    normal.own_state,
                    target_state,
                    state_valid,
                    normal.own_visual,
                    target_visual,
                    visual_valid,
                )
                normal_peer = masked_peer_future_prediction_losses(
                    normal.peer_state,
                    target_state,
                    state_valid,
                    normal.peer_visual,
                    target_visual,
                    visual_valid,
                    valid_agents,
                )
                normal_shared = masked_shared_future_prediction_losses(
                    normal.shared_visual,
                    target_shared,
                    shared_valid,
                    valid_agents,
                )
                shuffled = hybrid(
                    current_state,
                    current_visual,
                    current_shared,
                    actions,
                    valid_agents,
                    actions_by_focal=peer_actions_shuffled_by_focal(
                        actions, valid_agents
                    ),
                )
                shuffled_peer = masked_peer_future_prediction_losses(
                    shuffled.peer_state,
                    target_state,
                    state_valid,
                    shuffled.peer_visual,
                    target_visual,
                    visual_valid,
                    valid_agents,
                )
                shuffled_shared = masked_shared_future_prediction_losses(
                    shuffled.shared_visual,
                    target_shared,
                    shared_valid,
                    valid_agents,
                )
                persistence_state = normalized_persistence_state(
                    artifact,
                    batch_size=actions.shape[0],
                    agents=actions.shape[1],
                    device=device,
                )
                persistence_visual = normalized_persistence_visual(
                    artifact,
                    batch_size=actions.shape[0],
                    agents=actions.shape[1],
                    grid_tokens=local_config.visual_grid_tokens,
                    device=device,
                )
                persistence_peer = masked_peer_future_prediction_losses(
                    persistence_state[:, None].expand(
                        -1, actions.shape[1], -1, -1, -1
                    ),
                    target_state,
                    state_valid,
                    persistence_visual[:, None].expand(
                        -1, actions.shape[1], -1, -1, -1, -1
                    ),
                    target_visual,
                    visual_valid,
                    valid_agents,
                )
                persistence_shared_loss = masked_shared_future_prediction_losses(
                    persistence_shared[:, None].expand(
                        -1, actions.shape[1], -1, -1, -1
                    ),
                    target_shared,
                    shared_valid,
                    valid_agents,
                )
                normal_per = normal_peer["per_trajectory"] + normal_shared[
                    "per_trajectory"
                ]
                shuffled_per = shuffled_peer["per_trajectory"] + shuffled_shared[
                    "per_trajectory"
                ]
                persistence_per = persistence_peer[
                    "per_trajectory"
                ] + persistence_shared_loss["per_trajectory"]
                for row in range(grouped["episode_index"].shape[0]):
                    normal_loss = float(normal_per[row])
                    shuffled_loss = float(shuffled_per[row])
                    task_records.append(
                        {
                            "episode_index": int(grouped["episode_index"][row]),
                            "episode_seed": int(grouped["episode_seed"][row]),
                            "decision_t": int(grouped["decision_t"][row]),
                            "own_normal_loss": float(
                                hybrid_own_loss["per_trajectory"][row]
                            ),
                            "reference_own_loss": float(
                                reference_loss["per_trajectory"][row]
                            ),
                            "own_loss_exact": bool(
                                torch.equal(
                                    reference_loss["per_trajectory"][row],
                                    hybrid_own_loss["per_trajectory"][row],
                                )
                            ),
                            "peer_shared_normal_loss": normal_loss,
                            "peer_shared_shuffled_loss": shuffled_loss,
                            "peer_shared_persistence_loss": float(
                                persistence_per[row]
                            ),
                            "peer_shuffle_delta": shuffled_loss - normal_loss,
                        }
                    )
                completed_batches += 1
                progress = {
                    "event": "hybrid_validation_progress",
                    "program": "evaluate_s2_r4_hybrid_checkpoint.py",
                    "task_id": task_id,
                    "batch": batch_index,
                    "batches": len(batches),
                    "completed_batches": completed_batches,
                    "total_batches": total_batches,
                    "completed_fraction": completed_batches / total_batches,
                    "windows": len(task_records),
                    "own_max_abs_diff": task_exact["max_abs_diff"],
                    "peer_shared_loss": float(normal_per.mean()),
                    "persistence_loss": float(persistence_per.mean()),
                    "peer_shuffle_delta": float(
                        shuffled_per.mean() - normal_per.mean()
                    ),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                print(json.dumps(progress, sort_keys=True), flush=True)
                _append_jsonl(progress_log, progress)
            records[task_id] = task_records
            exact_by_task[task_id] = task_exact

    bootstrap_samples = int(evaluation.get("bootstrap_samples", 10000))
    bootstrap_seed = int(evaluation.get("bootstrap_seed", 40404))
    per_task: dict[str, dict[str, Any]] = {}
    for task_index, (task_id, task_records) in enumerate(records.items()):
        metrics = _task_metrics(
            task_records,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + task_index,
        )
        reference_loss = _episode_mean(task_records, "reference_own_loss")
        hybrid_loss = float(metrics["own"]["normal_composite_future_loss"])
        metrics["protected_own"] = {
            **exact_by_task[task_id],
            "reference_composite_future_loss": reference_loss,
            "hybrid_composite_future_loss": hybrid_loss,
            "loss_elementwise_exact": all(
                bool(record["own_loss_exact"]) for record in task_records
            ),
            "loss_difference": hybrid_loss - reference_loss,
        }
        per_task[task_id] = metrics

    action_equivalence = _action_equivalence_smoke(raw, team_saved, device=device)
    source_hashes_after = {
        "protected_own": file_sha256(own_path),
        "team": file_sha256(team_path),
    }
    sources_stable = source_hashes_before == source_hashes_after
    diagnostic = build_diagnostic(
        per_task,
        action_equivalence=action_equivalence,
        sources_stable=sources_stable,
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "round_id": "s2-r4-hybrid",
        "candidate_id": "HYBRID",
        "model_kind": model_kind,
        "training_performed": False,
        "hybrid_manifest": str(manifest_path),
        "hybrid_manifest_sha256": file_sha256(manifest_path),
        "sources": {
            "before_sha256": source_hashes_before,
            "after_sha256": source_hashes_after,
            "stable": sources_stable,
        },
        "comparison_contract": {
            "task_vocabulary": list(dataset.task_vocabulary),
            "validation_split": "validation",
            "validation_selection_sha256": selection_sha256,
            "future_artifacts_sha256": artifact_sha256,
            "windows_per_episode": int(evaluation["windows_per_episode"]),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "own_action_preserved_during_peer_shuffle": True,
        },
        "per_task": per_task,
        "action_equivalence": action_equivalence,
        "diagnostic": diagnostic,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    dataset.close()
    print(json.dumps(diagnostic, sort_keys=True), flush=True)
    return 0 if diagnostic["passed"] else 4


def build_diagnostic(
    per_task: Mapping[str, Mapping[str, Any]],
    *,
    action_equivalence: Mapping[str, Any],
    sources_stable: bool,
) -> dict[str, Any]:
    exact_five_task_set = set(per_task) == REQUIRED_TASKS
    own_exact = exact_five_task_set and all(
        bool(_mapping(metrics, "protected_own").get("state_elementwise_exact"))
        and bool(_mapping(metrics, "protected_own").get("visual_elementwise_exact"))
        and bool(_mapping(metrics, "protected_own").get("loss_elementwise_exact"))
        and float(_mapping(metrics, "protected_own").get("max_abs_diff", 1.0))
        == 0.0
        and float(_mapping(metrics, "protected_own").get("loss_difference", 1.0))
        == 0.0
        for metrics in per_task.values()
    )
    team_capability = exact_five_task_set and all(
        float(_mapping(metrics, "peer_shared")["normal_composite_future_loss"])
        < float(
            _mapping(metrics, "peer_shared")[
                "persistence_composite_future_loss"
            ]
        )
        for metrics in per_task.values()
    )
    peer_causality = exact_five_task_set and all(
        float(_mapping(metrics, "peer_shared")["shuffle_delta"]) > 0.0
        and float(
            _mapping(
                _mapping(metrics, "peer_shared"),
                "shuffle_delta_bootstrap_95",
            )["lower"]
        )
        > 0.0
        for metrics in per_task.values()
    )
    off_path_safe = bool(action_equivalence.get("passed")) and sources_stable
    passed = own_exact and team_capability and peer_causality and off_path_safe
    if passed:
        conclusion = "pass_existing_team_compatible"
        next_action = "report_r4_pass_no_r5_requested"
    elif not own_exact or not off_path_safe:
        conclusion = "fail_hybrid_wiring_or_source_integrity"
        next_action = "stop_before_r5_fix_hybrid"
    else:
        conclusion = "fail_old_team_incompatible_with_protected_own"
        next_action = "enter_s2_r5_retrain_team_from_protected_p0"
    return {
        "passed": passed,
        "conclusion": conclusion,
        "next_action": next_action,
        "checks": {
            "exact_five_task_set": exact_five_task_set,
            "protected_own_exact_on_every_task": own_exact,
            "team_beats_persistence_on_every_task": team_capability,
            "peer_shuffle_delta_and_ci_positive_on_every_task": peer_causality,
            "off_path_and_sources_unchanged": off_path_safe,
        },
        "special_rule": (
            "R4 passes only when protected own is bit-exact, every task beats "
            "persistence, every peer-action-shuffle delta and paired-bootstrap "
            "95% lower bound is positive, and off-path/source hashes stay exact."
        ),
    }


def _source_checkpoint(path: Path, *, candidate: str) -> Mapping[str, Any]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, Mapping) or saved.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError(f"invalid S2-R4 source checkpoint: {path}")
    method = _mapping(saved, "method")
    if method.get("candidate_id") != candidate:
        raise ValueError(f"unexpected S2-R4 source candidate: {path}")
    return saved


def _empty_exact() -> dict[str, float | bool]:
    return {
        "state_elementwise_exact": True,
        "visual_elementwise_exact": True,
        "state_max_abs_diff": 0.0,
        "visual_max_abs_diff": 0.0,
        "max_abs_diff": 0.0,
    }


def _merge_exact(
    current: Mapping[str, float | bool],
    observed: Mapping[str, float | bool],
) -> dict[str, float | bool]:
    return {
        "state_elementwise_exact": bool(current["state_elementwise_exact"])
        and bool(observed["state_elementwise_exact"]),
        "visual_elementwise_exact": bool(current["visual_elementwise_exact"])
        and bool(observed["visual_elementwise_exact"]),
        "state_max_abs_diff": max(
            float(current["state_max_abs_diff"]),
            float(observed["state_max_abs_diff"]),
        ),
        "visual_max_abs_diff": max(
            float(current["visual_max_abs_diff"]),
            float(observed["visual_max_abs_diff"]),
        ),
        "max_abs_diff": max(
            float(current["max_abs_diff"]), float(observed["max_abs_diff"])
        ),
    }


def _episode_mean(
    records: Sequence[Mapping[str, float | int | bool]],
    key: str,
) -> float:
    episodes: dict[int, list[float]] = {}
    for record in records:
        episodes.setdefault(int(record["episode_index"]), []).append(
            float(record[key])
        )
    return float(np.mean([np.mean(values) for values in episodes.values()]))


def _json_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
