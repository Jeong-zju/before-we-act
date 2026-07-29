#!/usr/bin/env python3
"""Evaluate one S2-R3 predictor with paired own-action shuffle."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from models.static_rgb_act import StaticRGBMoEACTConfig  # noqa: E402
from models.wam_multimodal import (  # noqa: E402
    AgentFactorizedFlowWAM,
    LocalActionConditionedFuturePredictor,
    LocalFuturePredictorConfig,
)
from scripts.train_s2_r3_future_predictor import (  # noqa: E402
    CHECKPOINT_FORMAT,
    FLOW_FORMAT,
)
from scripts.train_static_rgb_act_moe import (  # noqa: E402
    _append_jsonl,
    _load_yaml,
    _mapping,
    _vision,
)
from train.s2_future_prediction import (  # noqa: E402
    encode_local_visual_targets,
    file_sha256,
    load_s2_artifact,
    masked_future_prediction_losses,
    normalized_state_delta,
)
from train.s2_grouped_trajectory import (  # noqa: E402
    S2GroupedTrajectoryDataset,
    grouped_s2_batch,
)
from train.s2_model_registry import validate_s2_candidate  # noqa: E402


EVALUATION_FORMAT = "wam.robofactory.s2_r3.action_shuffle_evaluation/1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-log", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    raw = _load_yaml(config_path)
    candidate_id, model_kind, action_conditioning = validate_s2_candidate(
        _mapping(raw, "round")
    )
    checkpoint_config = _mapping(raw, "checkpoint")
    checkpoint_path = (
        args.checkpoint.expanduser().resolve(strict=True)
        if args.checkpoint is not None
        else (ROOT / str(checkpoint_config["output"])).resolve(strict=True)
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (ROOT / str(checkpoint_config["evaluation"])).resolve()
    )
    progress_log = (
        args.progress_log.expanduser().resolve()
        if args.progress_log is not None
        else (ROOT / str(checkpoint_config["evaluation_progress"])).resolve()
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite S2 evaluation {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_log.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("S2-R3 evaluation requires exactly one visible GPU")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if saved.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not an S2-R3 local predictor")
    method = _mapping(saved, "method")
    if (
        method.get("candidate_id") != candidate_id
        or method.get("model_kind") != model_kind
        or method.get("action_conditioning") is not action_conditioning
        or method.get("world_predictor_path") != "strictly_off_path"
    ):
        raise ValueError("S2-R3 checkpoint method differs from runtime config")
    model_config = LocalFuturePredictorConfig.from_dict(
        dict(_mapping(saved, "model_config"))
    )
    configured_model = LocalFuturePredictorConfig.from_dict(
        dict(_mapping(raw, "model"))
    )
    if configured_model != model_config:
        raise ValueError("S2-R3 checkpoint/config model mismatch")
    model = LocalActionConditionedFuturePredictor(model_config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    artifact_path = (
        ROOT / str(_mapping(raw, "artifacts")["pca_statistics"])
    ).resolve(strict=True)
    artifact_sha256 = file_sha256(artifact_path)
    if artifact_sha256 != saved.get("future_artifacts_sha256"):
        raise ValueError("S2-R3 checkpoint PCA/statistics identity changed")
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
        json.dumps(
            selection_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    batch_size = int(evaluation.get("batch_size", 4))
    pca = _mapping(raw, "pca")
    records: dict[str, list[dict[str, float | int]]] = {}
    with torch.inference_mode():
        for task_id, indices in selected.items():
            task_records: list[dict[str, float | int]] = []
            batches = _validation_batches(indices, batch_size=batch_size)
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
                visual_valid = grouped[
                    "future_agent_visual_valid_mask"
                ].to(device=device, dtype=torch.bool)
                target_state = normalized_state_delta(
                    grouped, artifact, device=device
                )
                if action_conditioning:
                    normal_actions = actions
                    shuffled_actions = actions.roll(1, dims=0)
                    action_mask = valid_agents
                else:
                    normal_actions = torch.zeros_like(actions)
                    shuffled_actions = torch.zeros_like(actions)
                    action_mask = torch.zeros_like(valid_agents)
                predicted_state, predicted_visual = model(
                    current_state,
                    current_visual,
                    normal_actions,
                    valid_agents,
                    action_mask,
                )
                normal = masked_future_prediction_losses(
                    predicted_state,
                    target_state,
                    state_valid,
                    predicted_visual,
                    target_visual,
                    visual_valid,
                )
                shuffled_state, shuffled_visual = model(
                    current_state,
                    current_visual,
                    shuffled_actions,
                    valid_agents,
                    action_mask,
                )
                shuffled = masked_future_prediction_losses(
                    shuffled_state,
                    target_state,
                    state_valid,
                    shuffled_visual,
                    target_visual,
                    visual_valid,
                )
                for row in range(grouped["episode_index"].shape[0]):
                    normal_loss = float(normal["per_trajectory"][row])
                    shuffled_loss = float(shuffled["per_trajectory"][row])
                    task_records.append(
                        {
                            "episode_index": int(
                                grouped["episode_index"][row]
                            ),
                            "episode_seed": int(grouped["episode_seed"][row]),
                            "decision_t": int(grouped["decision_t"][row]),
                            "normal_loss": normal_loss,
                            "shuffled_loss": shuffled_loss,
                            "shuffle_delta": shuffled_loss - normal_loss,
                        }
                    )
                progress = {
                    "event": "validation_progress",
                    "program": "evaluate_s2_r3_future_predictor.py",
                    "candidate_id": candidate_id,
                    "task_id": task_id,
                    "batch": batch_index,
                    "batches": len(batches),
                    "samples": len(task_records),
                    "normal_loss": float(normal["loss"]),
                    "shuffle_delta": float(
                        shuffled["loss"] - normal["loss"]
                    ),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                print(json.dumps(progress, sort_keys=True), flush=True)
                _append_jsonl(progress_log, progress)
            records[task_id] = task_records

    bootstrap_samples = int(evaluation.get("bootstrap_samples", 10000))
    bootstrap_seed = int(evaluation.get("bootstrap_seed", 30303))
    per_task = {
        task_id: _task_metrics(
            task_records,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + task_index,
        )
        for task_index, (task_id, task_records) in enumerate(records.items())
    }
    equivalence = _action_equivalence_smoke(
        raw,
        saved,
        device=device,
    )
    training = _mapping(saved, "training")
    comparison_contract = {
        "task_vocabulary": list(dataset.task_vocabulary),
        "validation_split": "validation",
        "validation_selection_sha256": selection_sha256,
        "future_artifacts_sha256": artifact_sha256,
        "base_flow_sha256": _mapping(saved, "frozen_parent")[
            "flow_checkpoint_sha256"
        ],
        "dinov3_weights_sha256": vision.artifact_sha256,
        "dinov3_config_sha256": vision.config_sha256,
        "initial_model_sha256": saved["initial_model_sha256"],
        "model_config": model_config.to_dict(),
        "training_seed": int(training["seed"]),
        "training_updates": int(saved["update"]),
        "training_batch_size": int(training["batch_size"]),
        "future_horizons": list(model_config.future_horizons),
    }
    payload = {
        "format_version": EVALUATION_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "model_kind": model_kind,
        "action_conditioning": action_conditioning,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "comparison_contract": comparison_contract,
        "statistics": {
            "unit": "episode",
            "paired_action_shuffle": True,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": 0.95,
        },
        "per_task": per_task,
        "action_equivalence": equivalence,
        "frozen_parent": {
            "passed": bool(
                equivalence["flow_checkpoint_stable"]
                and equivalence["dinov3_artifacts_stable"]
                and equivalence["predictor_checkpoint_excludes_flow_and_dinov3"]
            ),
            "flow_checkpoint_stable": equivalence[
                "flow_checkpoint_stable"
            ],
            "dinov3_artifacts_stable": equivalence[
                "dinov3_artifacts_stable"
            ],
            "predictor_checkpoint_excludes_flow_and_dinov3": equivalence[
                "predictor_checkpoint_excludes_flow_and_dinov3"
            ],
        },
        "leakage_contract": {
            "future_targets_forwarded_as_inputs": False,
            "validation_split_used_for_fit": False,
            "pca_and_normalization_fit_split": artifact["fit_split"],
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    dataset.close()
    print(
        json.dumps(
            {
                "evaluation": str(output),
                "candidate_id": candidate_id,
                "action_equivalence": equivalence["passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _dataset(config: Mapping[str, object]) -> S2GroupedTrajectoryDataset:
    data = _mapping(config, "data")
    manifests = [
        (ROOT / str(value)).resolve(strict=True)
        for value in data["manifests"]  # type: ignore[index]
    ]
    return S2GroupedTrajectoryDataset(
        manifests,
        split="validation",
        stride=int(data.get("validation_stride", data.get("stride", 1))),
        hdf5_cache_size=int(data.get("hdf5_cache_size", 4)),
    )


def _validation_indices(
    dataset: S2GroupedTrajectoryDataset,
    *,
    windows_per_episode: int,
) -> dict[str, list[int]]:
    if windows_per_episode <= 0:
        raise ValueError("evaluation.windows_per_episode must be positive")
    result: dict[str, list[int]] = {}
    for task_index, contract in enumerate(dataset.contracts):
        episodes: dict[int, list[int]] = defaultdict(list)
        for index in dataset.task_indices(task_index):
            episodes[dataset.sample_lineage(index).episode_index].append(index)
        selected: list[int] = []
        for episode_index in sorted(episodes):
            values = episodes[episode_index]
            count = min(windows_per_episode, len(values))
            if count == 1:
                offsets = [len(values) // 2]
            else:
                offsets = [
                    round(position * (len(values) - 1) / (count - 1))
                    for position in range(count)
                ]
            selected.extend(values[offset] for offset in offsets)
        if len(selected) < 2:
            raise RuntimeError(
                f"task {contract.task_id} has fewer than two held-out windows"
            )
        result[contract.task_id] = selected
    return result


def _validation_batches(
    indices: Sequence[int],
    *,
    batch_size: int,
) -> list[list[int]]:
    if batch_size < 2:
        raise ValueError("S2 action shuffle evaluation batch_size must be >=2")
    batches = [
        list(indices[start : start + batch_size])
        for start in range(0, len(indices), batch_size)
    ]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-1].insert(0, batches[-2].pop())
    if any(len(batch) < 2 for batch in batches):
        raise RuntimeError("paired action shuffle requires at least two samples")
    return batches


def _task_metrics(
    records: Sequence[Mapping[str, float | int]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    episodes: dict[int, list[Mapping[str, float | int]]] = defaultdict(list)
    for record in records:
        episodes[int(record["episode_index"])].append(record)
    episode_rows = []
    for episode_index in sorted(episodes):
        values = episodes[episode_index]
        normal = float(np.mean([float(item["normal_loss"]) for item in values]))
        shuffled = float(
            np.mean([float(item["shuffled_loss"]) for item in values])
        )
        episode_rows.append(
            {
                "episode_index": episode_index,
                "windows": len(values),
                "normal_loss": normal,
                "shuffled_loss": shuffled,
                "shuffle_delta": shuffled - normal,
            }
        )
    deltas = np.asarray(
        [float(value["shuffle_delta"]) for value in episode_rows],
        dtype=np.float64,
    )
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(
        0,
        len(deltas),
        size=(bootstrap_samples, len(deltas)),
    )
    means = deltas[draws].mean(axis=1)
    normal_loss = float(
        np.mean([float(value["normal_loss"]) for value in episode_rows])
    )
    shuffled_loss = float(
        np.mean([float(value["shuffled_loss"]) for value in episode_rows])
    )
    return {
        "episodes": len(episode_rows),
        "windows": len(records),
        "normal_composite_future_loss": normal_loss,
        "shuffled_composite_future_loss": shuffled_loss,
        "shuffle_delta": shuffled_loss - normal_loss,
        "shuffle_delta_bootstrap_95": {
            "lower": float(np.quantile(means, 0.025)),
            "upper": float(np.quantile(means, 0.975)),
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "episode_rows": episode_rows,
    }


def _action_equivalence_smoke(
    config: Mapping[str, Any],
    predictor_checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    parent = _mapping(config, "parent")
    flow_path = (ROOT / str(parent["flow_checkpoint"])).resolve(strict=True)
    vision_config = _mapping(config, "vision")
    vision_weights = (
        ROOT / str(vision_config["weights_path"])
    ).resolve(strict=True)
    vision_json = (
        ROOT / str(vision_config["config_path"])
    ).resolve(strict=True)
    before = {
        "flow": file_sha256(flow_path),
        "vision_weights": file_sha256(vision_weights),
        "vision_config": file_sha256(vision_json),
    }
    saved_flow = torch.load(flow_path, map_location=device, weights_only=False)
    if saved_flow.get("format_version") != FLOW_FORMAT:
        raise ValueError("action-equivalence parent is not S1-R1 Flow")
    flow_config = StaticRGBMoEACTConfig.from_dict(
        dict(_mapping(saved_flow, "model_config"))
    )
    flow = AgentFactorizedFlowWAM(flow_config).to(device)
    flow.load_state_dict(saved_flow["model"], strict=True)
    flow.eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator(device=device).manual_seed(7123)
    vision_tokens = torch.randn(
        2,
        4,
        flow_config.vision_dim,
        generator=generator,
        device=device,
    )
    state = torch.randn(
        2,
        flow_config.state_dim,
        generator=generator,
        device=device,
    )
    initial = torch.randn(
        2,
        flow_config.horizon,
        flow_config.action_dim,
        generator=generator,
        device=device,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        parent_actions = flow.generate_actions(
            vision_tokens,
            state,
            initial_actions=initial,
            solver_steps=4,
            solver="euler",
        )
        predictor_disabled_actions = flow.generate_actions(
            vision_tokens,
            state,
            initial_actions=initial,
            solver_steps=4,
            solver="euler",
        )
    after = {
        "flow": file_sha256(flow_path),
        "vision_weights": file_sha256(vision_weights),
        "vision_config": file_sha256(vision_json),
    }
    predictor_keys = set(_mapping(predictor_checkpoint, "model"))
    excluded = not any(
        key.startswith(("flow.", "vision.", "dinov3."))
        for key in predictor_keys
    )
    exact = bool(torch.equal(parent_actions, predictor_disabled_actions))
    result = {
        "passed": bool(
            exact
            and before == after
            and excluded
            and before["flow"]
            == _mapping(predictor_checkpoint, "frozen_parent")[
                "flow_checkpoint_sha256"
            ]
        ),
        "elementwise_exact": exact,
        "maximum_absolute_difference": float(
            (parent_actions - predictor_disabled_actions).abs().max()
        ),
        "flow_checkpoint_stable": before["flow"] == after["flow"],
        "dinov3_artifacts_stable": (
            before["vision_weights"] == after["vision_weights"]
            and before["vision_config"] == after["vision_config"]
        ),
        "predictor_checkpoint_excludes_flow_and_dinov3": excluded,
        "before_sha256": before,
        "after_sha256": after,
    }
    del flow, saved_flow
    return result


if __name__ == "__main__":
    raise SystemExit(main())
