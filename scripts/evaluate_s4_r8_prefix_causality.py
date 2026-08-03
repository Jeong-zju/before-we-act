#!/usr/bin/env python3
"""Evaluate the S4-R8 prefix-causality gates on held-out trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
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

from scripts.evaluate_s2_r5_protected_team import (  # noqa: E402
    _dataset as _validation_dataset,
    _validation_batches,
    _validation_indices,
)
from scripts.s4_r8_model_io import build_s4_r8_model  # noqa: E402
from scripts.train_s2_r4_future_predictor import (  # noqa: E402
    _validate_artifact_dataset,
)
from scripts.train_s3_r6_world_action_flow import _model_inputs  # noqa: E402
from scripts.train_s4_r7_world_utility import _future_targets  # noqa: E402
from scripts.train_static_rgb_act_moe import (  # noqa: E402
    _append_jsonl,
    _load_yaml,
    _mapping,
    _seed_everything,
    _vision,
)
from train.s2_future_prediction import file_sha256, load_s2_artifact  # noqa: E402
from train.s2_grouped_trajectory import grouped_s2_batch  # noqa: E402
from train.s4_future_feature_cache import (  # noqa: E402
    S4ProjectedFutureFeatureCache,
)
from train.s4_joint_losses import (  # noqa: E402
    s4_own_state_loss,
    s4_own_visual_loss,
    s4_peer_state_loss,
    s4_peer_visual_loss,
    s4_shared_visual_loss,
)
from train.s4_model_registry import validate_s4_r8_candidate  # noqa: E402


CHECKPOINT_FORMAT = "wam.robofactory.s4_r8.horizon_causal.checkpoint/1"
PREFIX_SUFFIX_FORMAT = "wam.robofactory.s4_r8.prefix_suffix_exact/1"
PREFIX_SHUFFLE_FORMAT = "wam.robofactory.s4_r8.prefix_shuffle_by_source_horizon/1"
SOURCES = ("own", "peer", "shared")
HORIZONS = (1, 25, 50, 100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix-suffix-output", type=Path, required=True)
    parser.add_argument("--prefix-shuffle-output", type=Path, required=True)
    parser.add_argument("--progress-log", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    suffix_output = args.prefix_suffix_output.expanduser().resolve()
    shuffle_output = args.prefix_shuffle_output.expanduser().resolve()
    progress_log = args.progress_log.expanduser().resolve()
    if suffix_output.exists() or shuffle_output.exists():
        raise FileExistsError("refusing to overwrite an R8 prefix audit")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("S4-R8 prefix evaluation requires one visible CUDA GPU")
    device = torch.device(args.device)
    raw = _load_yaml(config_path)
    candidate_id, model_kind, aggregator = validate_s4_r8_candidate(raw)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(saved, Mapping)
        or saved.get("format_version") != CHECKPOINT_FORMAT
    ):
        raise ValueError("checkpoint is not a registered S4-R8 policy")
    method = _mapping(saved, "method")
    if (
        method.get("round_id") != "s4-r8"
        or method.get("candidate_id") != candidate_id
        or method.get("model_kind") != model_kind
        or method.get("action_prefix_aggregator") != aggregator
    ):
        raise ValueError("R8 checkpoint/config method identity differs")

    cache_root = os.environ.get("S4_R8_FUTURE_FEATURE_CACHE", "")
    cache_sha256 = os.environ.get("S4_R8_FUTURE_FEATURE_CACHE_SHA256", "")
    if not cache_root or len(cache_sha256) != 64:
        raise ValueError("R8 prefix evaluation requires the shared future cache")
    evaluation = _mapping(raw, "evaluation")
    seed = int(evaluation.get("prefix_bootstrap_seed", 80_808))
    _seed_everything(seed)
    model, legacy_reference, parent_identity = build_s4_r8_model(raw, device=device)
    if dict(_mapping(saved, "parent_identity")) != parent_identity:
        raise ValueError("R8 prefix evaluation ancestor identity differs")
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    del legacy_reference
    provider = model.active_parent.future_predictor

    artifact_path = (ROOT / str(_mapping(raw, "artifacts")["pca_statistics"])).resolve(
        strict=True
    )
    artifact = load_s2_artifact(artifact_path, device=device)
    dataset = _validation_dataset(raw)
    _validate_artifact_dataset(artifact, dataset)
    cache = S4ProjectedFutureFeatureCache(
        cache_root,
        manifests=[contract.manifest_path for contract in dataset.contracts],
        expected_features_sha256=cache_sha256,
        expected_pca_sha256=str(_mapping(raw, "parent")["expected_pca_sha256"]),
        expected_vision_weights_sha256=str(
            _mapping(raw, "vision")["expected_weights_sha256"]
        ),
    )
    vision = _vision(raw).to(device).eval()
    selected = _validation_indices(
        dataset,
        windows_per_episode=int(evaluation.get("prefix_windows_per_episode", 2)),
    )
    batch_size = int(evaluation.get("prefix_batch_size", 4))
    if batch_size < 2:
        raise ValueError("R8 prefix shuffle batch size must be at least two")
    workers = int(evaluation.get("offline_num_workers", 2))
    pca = _mapping(raw, "pca")
    records: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    suffix_max = {(source, horizon): 0.0 for source in SOURCES for horizon in HORIZONS}
    prefix_min = {
        (source, horizon): float("inf") for source in SOURCES for horizon in HORIZONS
    }
    exact_batch_done = False

    with torch.inference_mode():
        for task_index, task_id in enumerate(selected):
            ordered = _interleave_episodes(dataset, selected[task_id])
            batches = _validation_batches(ordered, batch_size=batch_size)
            loader = DataLoader(
                dataset,
                batch_sampler=batches,
                num_workers=workers,
                pin_memory=True,
            )
            for batch_index, raw_batch in enumerate(loader, start=1):
                grouped = grouped_s2_batch(raw_batch)
                episodes = grouped["episode_index"].tolist()
                if any(
                    left == right
                    for left, right in zip(episodes, episodes[-1:] + episodes[:-1])
                ):
                    raise RuntimeError(
                        "R8 prefix donors must come from different held-out episodes"
                    )
                inputs = _model_inputs(
                    vision, grouped, artifact, device=device, pca=pca
                )
                targets = _future_targets(
                    grouped,
                    artifact,
                    future_feature_cache=cache,
                    current_local=inputs["local_visual"],
                    current_shared=inputs["shared_visual"],
                    device=device,
                )
                actions = grouped["candidate_actions"].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                valid = inputs["valid"]
                team_actions = actions[:, None].expand(-1, actions.shape[1], -1, -1, -1)
                baseline = provider(
                    inputs["state"],
                    inputs["local_visual"],
                    inputs["shared_visual"],
                    actions,
                    valid,
                    actions_by_focal=team_actions,
                )
                baseline_losses = _source_horizon_losses(baseline, targets, valid)
                for horizon_index, horizon in enumerate(HORIZONS):
                    donor_actions = actions.clone()
                    donor_actions[:, :, :horizon] = torch.roll(
                        actions[:, :, :horizon], shifts=1, dims=0
                    )
                    own_shuffled = provider(
                        inputs["state"],
                        inputs["local_visual"],
                        inputs["shared_visual"],
                        donor_actions,
                        valid,
                        actions_by_focal=team_actions,
                    )
                    donor_team = team_actions.clone()
                    donor_team[:, :, :, :horizon] = torch.roll(
                        team_actions[:, :, :, :horizon], shifts=1, dims=0
                    )
                    team_shuffled = provider(
                        inputs["state"],
                        inputs["local_visual"],
                        inputs["shared_visual"],
                        actions,
                        valid,
                        actions_by_focal=donor_team,
                    )
                    shuffled = {
                        "own": _source_horizon_losses(own_shuffled, targets, valid)[
                            ("own", horizon)
                        ],
                        "peer": _source_horizon_losses(team_shuffled, targets, valid)[
                            ("peer", horizon)
                        ],
                        "shared": _source_horizon_losses(team_shuffled, targets, valid)[
                            ("shared", horizon)
                        ],
                    }
                    for source in SOURCES:
                        normal = baseline_losses[(source, horizon)]
                        changed = shuffled[source]
                        for row, episode in enumerate(episodes):
                            records[(source, horizon)].append(
                                {
                                    "task_id": task_id,
                                    "episode_index": int(episode),
                                    "normal_loss": float(normal[row]),
                                    "prefix_shuffled_loss": float(changed[row]),
                                    "delta": float(changed[row] - normal[row]),
                                }
                            )
                    if not exact_batch_done:
                        exact = _exact_interventions(
                            provider,
                            inputs,
                            actions,
                            team_actions,
                            valid,
                            baseline,
                            horizon_index=horizon_index,
                            horizon=horizon,
                        )
                        for source in SOURCES:
                            suffix_max[(source, horizon)] = max(
                                suffix_max[(source, horizon)],
                                exact[source]["suffix_max_abs_diff"],
                            )
                            prefix_min[(source, horizon)] = min(
                                prefix_min[(source, horizon)],
                                exact[source]["prefix_max_abs_diff"],
                            )
                exact_batch_done = True
                _append_jsonl(
                    progress_log,
                    {
                        "event": "prefix_causality_batch",
                        "program": "evaluate_s4_r8_prefix_causality.py",
                        "round_id": "s4-r8",
                        "candidate_id": candidate_id,
                        "task": task_id,
                        "task_index": task_index,
                        "batch": batch_index,
                        "batches": len(batches),
                        "detail": (
                            f"held-out prefix shuffle {task_id} "
                            f"{batch_index}/{len(batches)}"
                        ),
                        "created_at": _now(),
                    },
                )

    suffix_rows = {
        f"{source}@{horizon}": {
            "source": source,
            "horizon": horizon,
            "suffix_max_abs_diff": suffix_max[(source, horizon)],
            "suffix_elementwise_exact": suffix_max[(source, horizon)] == 0.0,
            "legal_prefix_max_abs_diff": prefix_min[(source, horizon)],
            "legal_prefix_changes_output": prefix_min[(source, horizon)] > 0.0,
        }
        for source in SOURCES
        for horizon in HORIZONS
    }
    suffix_report = {
        "format_version": PREFIX_SUFFIX_FORMAT,
        "round_id": "s4-r8",
        "candidate_id": candidate_id,
        "model_kind": model_kind,
        "checkpoint_sha256": checkpoint_sha256,
        "action_prefix_aggregator": aggregator,
        "fp32_eval": True,
        "groups": suffix_rows,
        "passed": all(
            row["suffix_elementwise_exact"] is True
            and row["legal_prefix_changes_output"] is True
            for row in suffix_rows.values()
        ),
        "created_at": _now(),
    }
    bootstrap_samples = int(evaluation.get("bootstrap_samples", 10_000))
    shuffle_rows = {
        f"{source}@{horizon}": _bootstrap_group(
            records[(source, horizon)],
            samples=bootstrap_samples,
            seed=seed + source_index * 1000 + horizon,
            source=source,
            horizon=horizon,
        )
        for source_index, source in enumerate(SOURCES)
        for horizon in HORIZONS
    }
    shuffle_report = {
        "format_version": PREFIX_SHUFFLE_FORMAT,
        "round_id": "s4-r8",
        "candidate_id": candidate_id,
        "model_kind": model_kind,
        "checkpoint_sha256": checkpoint_sha256,
        "action_prefix_aggregator": aggregator,
        "shuffle_protocol": (
            "within_task_different_episode_replace_legal_action_prefix_only"
        ),
        "bootstrap_unit": "episode",
        "bootstrap_samples": bootstrap_samples,
        "groups": shuffle_rows,
        "passed": all(
            float(row["episode_bootstrap_95_lower"]) > 0.0
            for row in shuffle_rows.values()
        ),
        "created_at": _now(),
    }
    _atomic_json(suffix_output, suffix_report)
    _atomic_json(shuffle_output, shuffle_report)
    dataset.close()
    print(
        json.dumps(
            {
                "prefix_suffix_exact": str(suffix_output),
                "prefix_shuffle": str(shuffle_output),
                "suffix_passed": suffix_report["passed"],
                "shuffle_passed": shuffle_report["passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _interleave_episodes(dataset: Any, indices: Sequence[int]) -> list[int]:
    by_episode: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        by_episode[dataset.sample_lineage(index).episode_index].append(index)
    ordered = []
    for offset in range(max(len(values) for values in by_episode.values())):
        ordered.extend(
            values[offset]
            for _, values in sorted(by_episode.items())
            if offset < len(values)
        )
    return ordered


def _source_horizon_losses(
    prediction: Any,
    targets: Mapping[str, Tensor],
    valid: Tensor,
) -> dict[tuple[str, int], Tensor]:
    result: dict[tuple[str, int], Tensor] = {}
    for index, horizon in enumerate(HORIZONS):
        future = slice(index, index + 1)
        per_item: dict[str, list[Tensor]] = {source: [] for source in SOURCES}
        for row in range(valid.shape[0]):
            item = slice(row, row + 1)
            own_state = s4_own_state_loss(
                prediction.own_state[item, :, future],
                targets["state"][item, :, future],
                targets["state_valid"][item, :, future],
                valid[item],
            )
            own_visual = s4_own_visual_loss(
                prediction.own_visual[item, :, future],
                targets["local_visual"][item, :, future],
                targets["local_visual_valid"][item, :, future],
                valid[item],
            )
            peer_state = s4_peer_state_loss(
                prediction.peer_state[item, :, :, future],
                targets["state"][item, :, future],
                targets["state_valid"][item, :, future],
                valid[item],
            )
            peer_visual = s4_peer_visual_loss(
                prediction.peer_visual[item, :, :, future],
                targets["local_visual"][item, :, future],
                targets["local_visual_valid"][item, :, future],
                valid[item],
            )
            shared_visual = s4_shared_visual_loss(
                prediction.shared_visual[item, :, future],
                targets["shared_visual"][item, future],
                targets["shared_visual_valid"][item, future],
                valid[item],
            )
            per_item["own"].append((own_state + own_visual) / 2.0)
            per_item["peer"].append((peer_state + peer_visual) / 2.0)
            per_item["shared"].append(shared_visual)
        for source in SOURCES:
            result[(source, horizon)] = torch.stack(per_item[source]).cpu()
    return result


def _exact_interventions(
    provider: torch.nn.Module,
    inputs: Mapping[str, Tensor],
    actions: Tensor,
    team_actions: Tensor,
    valid: Tensor,
    baseline: Any,
    *,
    horizon_index: int,
    horizon: int,
) -> dict[str, dict[str, float]]:
    suffix_actions = actions.clone()
    suffix_actions[:, :, horizon:, 0] += 7.0
    own_suffix = provider(
        inputs["state"],
        inputs["local_visual"],
        inputs["shared_visual"],
        suffix_actions,
        valid,
        actions_by_focal=team_actions,
    )
    prefix_actions = actions.clone()
    prefix_actions[:, :, :horizon, 0] += 0.125
    own_prefix = provider(
        inputs["state"],
        inputs["local_visual"],
        inputs["shared_visual"],
        prefix_actions,
        valid,
        actions_by_focal=team_actions,
    )
    suffix_team = team_actions.clone()
    suffix_team[:, :, :, horizon:, 0] += 7.0
    team_suffix = provider(
        inputs["state"],
        inputs["local_visual"],
        inputs["shared_visual"],
        actions,
        valid,
        actions_by_focal=suffix_team,
    )
    prefix_team = team_actions.clone()
    prefix_team[:, :, :, :horizon, 0] += 0.125
    team_prefix = provider(
        inputs["state"],
        inputs["local_visual"],
        inputs["shared_visual"],
        actions,
        valid,
        actions_by_focal=prefix_team,
    )
    fields = {
        "own": ("own_state", "own_visual", own_suffix, own_prefix),
        "peer": ("peer_state", "peer_visual", team_suffix, team_prefix),
        "shared": ("shared_visual", None, team_suffix, team_prefix),
    }
    result = {}
    for source, (first, second, suffix, prefix) in fields.items():
        names = (first,) if second is None else (first, second)
        suffix_diff = max(
            float(
                (
                    getattr(suffix, name).select(
                        -2 if name.endswith("state") else -3,
                        horizon_index,
                    )
                    - getattr(baseline, name).select(
                        -2 if name.endswith("state") else -3,
                        horizon_index,
                    )
                )
                .abs()
                .max()
            )
            for name in names
        )
        prefix_diff = max(
            float(
                (
                    getattr(prefix, name).select(
                        -2 if name.endswith("state") else -3,
                        horizon_index,
                    )
                    - getattr(baseline, name).select(
                        -2 if name.endswith("state") else -3,
                        horizon_index,
                    )
                )
                .abs()
                .max()
            )
            for name in names
        )
        result[source] = {
            "suffix_max_abs_diff": suffix_diff,
            "prefix_max_abs_diff": prefix_diff,
        }
    return result


def _bootstrap_group(
    rows: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
    source: str,
    horizon: int,
) -> dict[str, object]:
    by_episode: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_episode[(str(row["task_id"]), int(row["episode_index"]))].append(
            float(row["delta"])
        )
    episode_rows = [
        {
            "task_id": task,
            "episode_index": episode,
            "windows": len(values),
            "mean_delta": float(np.mean(values)),
        }
        for (task, episode), values in sorted(by_episode.items())
    ]
    values = np.asarray(
        [float(row["mean_delta"]) for row in episode_rows], dtype=np.float64
    )
    if values.size == 0 or samples <= 0:
        raise RuntimeError("R8 prefix bootstrap has no episode observations")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "source": source,
        "horizon": horizon,
        "episodes": len(values),
        "mean_delta": float(values.mean()),
        "episode_bootstrap_95_lower": float(np.quantile(means, 0.025)),
        "episode_bootstrap_95_upper": float(np.quantile(means, 0.975)),
        "passed": float(np.quantile(means, 0.025)) > 0.0,
        "episode_rows": episode_rows,
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
