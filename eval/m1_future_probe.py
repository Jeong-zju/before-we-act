"""Leakage-resistant H=8 linear probes for the Phase M1 latent WAM.

The future-latent probe compares two representations under an identical,
deterministic ridge protocol:

* the action-conditioned H=8 latent predicted by the M1 future head under the
  same demonstration's expert action chunk;
* the frozen visual teacher's pooled representation of the current frame; and
* a matched-information baseline containing that current-frame latent plus the
  identical expert action chunk.

Only training data fit probe weights.  Validation data select ridge strength
and the event decision threshold; the held-out test labels are read only after
all choices have been frozen.  Future RGB is never passed to either feature
extractor.  Labels are obtained exclusively through
``M1WindowDataset.probe_labels``, an explicitly offline-only API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
import yaml

from eval.m1_statistics import (
    paired_balanced_accuracy_comparison,
    paired_rmse_comparison,
)
from train.m1_checkpointing import checkpoint_tree_sha256, load_m1_checkpoint
from eval.m1_vision_contract import (
    validate_loaded_checkpoint_vision,
    validate_training_summary_vision,
)
from train.m1_manifest_dataset import (
    M1_PROBE_HORIZON,
    M1ManifestIndex,
    M1WindowDataset,
)


FORMAT_VERSION = "wam.multimodal.m1.future_probe/1"
PRIMARY_VARIANT = "state_vision_future"
CANONICAL_SPLITS = ("train", "validation", "test")
PROBE_ACTION_SOURCE = "same_demonstration_expert_action_chunk"
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1_000.0)

# These fixed caps make the formal CPU evaluation bounded while retaining
# hundreds of paired test examples in each class and task.
FORMAL_OBJECT_SAMPLES = {
    "train": 1_536,
    "validation": 768,
    "test": 1_536,
}
FORMAL_EVENT_SAMPLES = {
    "train": 768,
    "validation": 384,
    "test": 768,
}


class ProbeDataset(Protocol):
    """Narrow surface used by selection and feature extraction."""

    task_order: Sequence[str]
    records: Sequence[Any]
    decision_window_indices: Sequence[int]

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]: ...

    def probe_labels(self, index: int) -> Mapping[str, torch.Tensor]: ...

    def sample_lineage(self, index: int) -> Any: ...


@dataclass(frozen=True)
class ProbeSelection:
    """Deterministic, split-local indices for the two probe targets."""

    object_indices: tuple[int, ...]
    event_indices: tuple[int, ...]
    union_indices: tuple[int, ...]
    object_tasks: tuple[str, ...]
    event_labels: tuple[int, ...]
    sample_ids: Mapping[int, str]

    def summary(self) -> dict[str, Any]:
        object_counts: dict[str, int] = {}
        for task in self.object_tasks:
            object_counts[task] = object_counts.get(task, 0) + 1
        event_counts = {
            "0": int(sum(value == 0 for value in self.event_labels)),
            "1": int(sum(value == 1 for value in self.event_labels)),
        }
        return {
            "object_samples": len(self.object_indices),
            "event_samples": len(self.event_indices),
            "union_samples": len(self.union_indices),
            "object_task_counts": dict(sorted(object_counts.items())),
            "event_class_counts": event_counts,
            "object_sample_ids_sha256": _json_sha256(
                [self.sample_ids[index] for index in self.object_indices]
            ),
            "event_sample_ids_sha256": _json_sha256(
                [self.sample_ids[index] for index in self.event_indices]
            ),
            "union_sample_ids_sha256": _json_sha256(
                [self.sample_ids[index] for index in self.union_indices]
            ),
        }


@dataclass(frozen=True)
class ProbeFeatureSet:
    """Representations and offline targets in a frozen sample order."""

    predicted_h8: np.ndarray
    current_frame: np.ndarray
    current_frame_action: np.ndarray
    center_xy: np.ndarray
    event_active: np.ndarray
    sample_ids: tuple[str, ...]

    def subset(self, positions: Sequence[int]) -> "ProbeFeatureSet":
        selected = np.asarray(tuple(int(value) for value in positions), dtype=np.int64)
        return ProbeFeatureSet(
            predicted_h8=self.predicted_h8[selected],
            current_frame=self.current_frame[selected],
            current_frame_action=self.current_frame_action[selected],
            center_xy=self.center_xy[selected],
            event_active=self.event_active[selected],
            sample_ids=tuple(self.sample_ids[index] for index in selected.tolist()),
        )


@dataclass(frozen=True)
class RidgeModel:
    """A centered/scaled deterministic ridge model."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    coefficients: np.ndarray
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = _finite_matrix(features, "features")
        if values.shape[1] != self.feature_mean.shape[0]:
            raise ValueError("ridge feature width changed between fit and predict")
        return (
            (values - self.feature_mean) / self.feature_scale
        ) @ self.coefficients + self.target_mean


def select_probe_indices(
    dataset: ProbeDataset,
    *,
    object_samples: int,
    event_samples: int,
    seed: int,
    event_task: str = "visual_event_stop",
) -> ProbeSelection:
    """Select task-balanced object rows and class-balanced event rows.

    Selection is restricted to the dataset's decision-window indices when
    available.  Event labels are used only for this offline stratification and
    never appear in a model input.
    """

    if int(object_samples) <= 0 or int(event_samples) <= 0:
        raise ValueError("probe sample caps must be positive")
    candidates = tuple(int(value) for value in dataset.decision_window_indices)
    if not candidates:
        candidates = tuple(range(len(dataset)))
    if not candidates:
        raise ValueError("probe dataset contains no candidate windows")

    task_by_path = {
        str(Path(record.path).resolve()): str(record.task_id)
        for record in dataset.records
    }
    catalog: list[tuple[int, str, int, str]] = []
    for index in candidates:
        lineage = dataset.sample_lineage(index)
        path = str(Path(lineage.path).resolve())
        if path not in task_by_path:
            raise ValueError("sample lineage path is absent from manifest records")
        labels = dataset.probe_labels(index)
        if set(labels) != {"h8_center_xy", "h8_event_active"}:
            raise ValueError("offline probe label contract changed")
        event = int(bool(torch.as_tensor(labels["h8_event_active"]).item()))
        sample_id = f"{path}:{int(lineage.decision_t)}"
        catalog.append((index, task_by_path[path], event, sample_id))

    rng = np.random.default_rng(int(seed))
    by_task: dict[str, list[tuple[int, str, int, str]]] = {}
    for row in catalog:
        by_task.setdefault(row[1], []).append(row)
    expected_tasks = tuple(str(value) for value in dataset.task_order)
    missing_tasks = [task for task in expected_tasks if not by_task.get(task)]
    if missing_tasks:
        raise ValueError(f"probe selection has no rows for tasks {missing_tasks}")
    object_rows = _balanced_rows(
        {task: by_task[task] for task in expected_tasks},
        maximum=int(object_samples),
        rng=rng,
    )

    event_groups: dict[str, list[tuple[int, str, int, str]]] = {"0": [], "1": []}
    for row in catalog:
        if row[1] == event_task:
            event_groups[str(row[2])].append(row)
    if not event_groups["0"] or not event_groups["1"]:
        raise ValueError(
            f"event probe task {event_task!r} must contain both H8 event classes"
        )
    event_rows = _balanced_rows(
        event_groups,
        maximum=int(event_samples),
        rng=rng,
        require_equal=True,
    )

    object_indices = tuple(row[0] for row in object_rows)
    event_indices = tuple(row[0] for row in event_rows)
    union_indices = tuple(dict.fromkeys((*object_indices, *event_indices)))
    sample_ids = {row[0]: row[3] for row in catalog if row[0] in set(union_indices)}
    event_labels_by_index = {row[0]: row[2] for row in catalog}
    result = ProbeSelection(
        object_indices=object_indices,
        event_indices=event_indices,
        union_indices=union_indices,
        object_tasks=tuple(row[1] for row in object_rows),
        event_labels=tuple(event_labels_by_index[index] for index in event_indices),
        sample_ids=sample_ids,
    )
    if (
        result.summary()["event_class_counts"]["0"]
        != result.summary()["event_class_counts"]["1"]
    ):
        raise RuntimeError("event probe selection is not class-balanced")
    return result


@torch.inference_mode()
def extract_probe_features(
    model: Any,
    dataset: ProbeDataset,
    selection: ProbeSelection,
    *,
    device: str | torch.device,
    batch_size: int = 64,
) -> ProbeFeatureSet:
    """Extract a target-consistent, same-demonstration offline probe.

    The future latent and the H8 labels are paired with the expert action chunk
    from the same demonstration window.  The matched baseline receives that
    identical action chunk, so the future head cannot win merely because it
    was given oracle action information unavailable to the current-frame-only
    representation.
    """

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if getattr(model, "future_head", None) is None:
        raise ValueError("H8 probe requires a trained future visual-latent head")
    horizons = tuple(int(value) for value in model.future_horizons)
    if M1_PROBE_HORIZON not in horizons:
        raise ValueError("future head does not contain the canonical H=8 output")
    if getattr(model, "vision_encoder", None) is None:
        raise ValueError("current-frame baseline requires the frozen visual teacher")

    target_device = torch.device(device)
    h8_index = horizons.index(M1_PROBE_HORIZON)
    predicted_values: list[np.ndarray] = []
    current_values: list[np.ndarray] = []
    current_action_values: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    events: list[np.ndarray] = []
    sample_ids: list[str] = []
    deployable_keys = (
        "states",
        "state_valid_mask",
        "past_actions",
        "images",
        "task_index",
        "action_targets",
    )

    indices = selection.union_indices
    for start in range(0, len(indices), int(batch_size)):
        batch_indices = indices[start : start + int(batch_size)]
        deployable_samples = []
        for index in batch_indices:
            sample = dataset[index]
            missing = [name for name in deployable_keys if name not in sample]
            if missing:
                raise KeyError(f"M1 probe sample is missing deployable keys {missing}")
            # Construct a new mapping so future_images/future_states cannot be
            # forwarded accidentally even though the training dataset exposes
            # them for supervised losses.
            deployable_samples.append({name: sample[name] for name in deployable_keys})
        batch = default_collate(deployable_samples)
        states = batch["states"].to(target_device)
        valid_mask = batch["state_valid_mask"].to(target_device)
        past_actions = batch["past_actions"].to(target_device)
        images = batch["images"].to(target_device)
        task_index = batch["task_index"].to(target_device)

        encoding = model.encode(
            states,
            past_actions,
            valid_mask,
            images,
            task_index,
        )
        expert_actions = batch["action_targets"].to(
            target_device, dtype=encoding.planning_features.dtype
        )
        if (
            expert_actions.ndim != 3
            or expert_actions.shape[0] != encoding.planning_features.shape[0]
            or expert_actions.shape[1] != M1_PROBE_HORIZON
        ):
            raise ValueError("same-demonstration expert action chunk is non-canonical")
        if not bool(torch.isfinite(expert_actions).all()):
            raise ValueError(
                "same-demonstration expert action chunk contains NaN or Inf"
            )
        predicted = model.future_head(
            encoding.planning_features,
            encoding.visual_tokens,
            expert_actions,
        )[:, h8_index]

        # Baseline is deliberately the final observed frame only, not the
        # two-frame visual context consumed by the multimodal model.
        current_images = images[:, -1]
        current_output = model.vision_encoder(current_images)
        current = current_output.pooled_latent
        if current.ndim == 3:
            current = current.mean(dim=1)
        if current.ndim != 2 or current.shape[-1] != predicted.shape[-1]:
            raise ValueError("current/predicted teacher-latent dimensions differ")
        current_action = torch.cat(
            (current.to(dtype=expert_actions.dtype), expert_actions.flatten(1)),
            dim=1,
        )

        # Offline labels are deliberately requested only after feature
        # extraction.  They cannot affect the model, the deployed action flow,
        # or the current-frame baseline.
        batch_labels = [dataset.probe_labels(index) for index in batch_indices]

        predicted_values.append(predicted.detach().cpu().float().numpy())
        current_values.append(current.detach().cpu().float().numpy())
        current_action_values.append(current_action.detach().cpu().float().numpy())
        centers.append(
            np.stack(
                [
                    torch.as_tensor(value["h8_center_xy"]).cpu().numpy()
                    for value in batch_labels
                ]
            ).astype(np.float32)
        )
        events.append(
            np.asarray(
                [
                    int(bool(torch.as_tensor(value["h8_event_active"]).item()))
                    for value in batch_labels
                ],
                dtype=np.int8,
            )
        )
        sample_ids.extend(selection.sample_ids[index] for index in batch_indices)

    return ProbeFeatureSet(
        predicted_h8=np.concatenate(predicted_values, axis=0),
        current_frame=np.concatenate(current_values, axis=0),
        current_frame_action=np.concatenate(current_action_values, axis=0),
        center_xy=np.concatenate(centers, axis=0),
        event_active=np.concatenate(events, axis=0),
        sample_ids=tuple(sample_ids),
    )


def fit_ridge_with_validation(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    classification: bool = False,
) -> tuple[RidgeModel, float | None, dict[str, Any]]:
    """Fit on train and select alpha/threshold exclusively on validation."""

    train_x = _finite_matrix(train_features, "train_features")
    validation_x = _finite_matrix(validation_features, "validation_features")
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("train/validation feature widths differ")
    train_y = _target_matrix(train_targets, train_x.shape[0], "train_targets")
    validation_y = _target_matrix(
        validation_targets, validation_x.shape[0], "validation_targets"
    )
    if train_y.shape[1] != validation_y.shape[1]:
        raise ValueError("train/validation target widths differ")
    alpha_values = tuple(float(value) for value in alphas)
    if not alpha_values or any(
        not np.isfinite(value) or value <= 0 for value in alpha_values
    ):
        raise ValueError("ridge alphas must be finite and positive")
    if classification:
        if train_y.shape[1] != 1 or not _binary(train_y) or not _binary(validation_y):
            raise ValueError("classification ridge requires binary scalar targets")
        train_fit_y = train_y * 2.0 - 1.0
    else:
        train_fit_y = train_y

    feature_mean = train_x.mean(axis=0)
    feature_scale = train_x.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    normalized = (train_x - feature_mean) / feature_scale
    target_mean = train_fit_y.mean(axis=0)
    centered_targets = train_fit_y - target_mean
    gram = normalized.T @ normalized
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    projected = eigenvectors.T @ (normalized.T @ centered_targets)

    candidates: list[tuple[float, float, float | None, RidgeModel]] = []
    for alpha in alpha_values:
        coefficients = eigenvectors @ (projected / (eigenvalues[:, None] + alpha))
        model = RidgeModel(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            coefficients=coefficients,
            alpha=alpha,
        )
        validation_prediction = model.predict(validation_x)
        if classification:
            threshold, score = _best_balanced_threshold(
                validation_prediction[:, 0], validation_y[:, 0].astype(np.int8)
            )
            objective = -score
        else:
            threshold = None
            errors = np.linalg.norm(validation_prediction - validation_y, axis=1)
            objective = float(np.sqrt(np.mean(np.square(errors))))
        candidates.append((objective, alpha, threshold, model))
    best = min(
        candidates,
        key=lambda value: (
            value[0],
            value[1],
            float("inf") if value[2] is None else abs(value[2]),
        ),
    )
    objective, alpha, threshold, model = best
    validation_metric = -objective if classification else objective
    return (
        model,
        threshold,
        {
            "selected_alpha": alpha,
            "validation_metric": float(validation_metric),
            "validation_metric_name": (
                "balanced_accuracy" if classification else "center_xy_euclidean_rmse"
            ),
            "selected_threshold": threshold,
            "alphas": list(alpha_values),
            "fit_samples": int(train_x.shape[0]),
            "validation_samples": int(validation_x.shape[0]),
            "test_samples_used_for_fit_or_selection": 0,
        },
    )


def evaluate_feature_sets(
    train: ProbeFeatureSet,
    validation: ProbeFeatureSet,
    test: ProbeFeatureSet,
    *,
    train_selection: ProbeSelection,
    validation_selection: ProbeSelection,
    test_selection: ProbeSelection,
) -> dict[str, Any]:
    """Fit both probes for one train seed and evaluate its held-out test rows."""

    train_position = {
        index: position for position, index in enumerate(train_selection.union_indices)
    }
    validation_position = {
        index: position
        for position, index in enumerate(validation_selection.union_indices)
    }
    test_position = {
        index: position for position, index in enumerate(test_selection.union_indices)
    }

    def subset(
        features: ProbeFeatureSet,
        positions: Mapping[int, int],
        indices: Sequence[int],
    ) -> ProbeFeatureSet:
        return features.subset([positions[index] for index in indices])

    train_object = subset(train, train_position, train_selection.object_indices)
    validation_object = subset(
        validation, validation_position, validation_selection.object_indices
    )
    test_object = subset(test, test_position, test_selection.object_indices)
    train_event = subset(train, train_position, train_selection.event_indices)
    validation_event = subset(
        validation, validation_position, validation_selection.event_indices
    )
    test_event = subset(test, test_position, test_selection.event_indices)

    object_model, _, object_model_fit = fit_ridge_with_validation(
        train_object.predicted_h8,
        train_object.center_xy,
        validation_object.predicted_h8,
        validation_object.center_xy,
    )
    object_baseline, _, object_baseline_fit = fit_ridge_with_validation(
        train_object.current_frame,
        train_object.center_xy,
        validation_object.current_frame,
        validation_object.center_xy,
    )
    object_action_baseline, _, object_action_baseline_fit = fit_ridge_with_validation(
        train_object.current_frame_action,
        train_object.center_xy,
        validation_object.current_frame_action,
        validation_object.center_xy,
    )
    object_model_prediction = object_model.predict(test_object.predicted_h8)
    object_baseline_prediction = object_baseline.predict(test_object.current_frame)
    object_action_baseline_prediction = object_action_baseline.predict(
        test_object.current_frame_action
    )
    object_model_errors = np.linalg.norm(
        object_model_prediction - test_object.center_xy, axis=1
    )
    object_baseline_errors = np.linalg.norm(
        object_baseline_prediction - test_object.center_xy, axis=1
    )
    object_action_baseline_errors = np.linalg.norm(
        object_action_baseline_prediction - test_object.center_xy, axis=1
    )

    event_model, event_model_threshold, event_model_fit = fit_ridge_with_validation(
        train_event.predicted_h8,
        train_event.event_active,
        validation_event.predicted_h8,
        validation_event.event_active,
        classification=True,
    )
    event_baseline, event_baseline_threshold, event_baseline_fit = (
        fit_ridge_with_validation(
            train_event.current_frame,
            train_event.event_active,
            validation_event.current_frame,
            validation_event.event_active,
            classification=True,
        )
    )
    (
        event_action_baseline,
        event_action_baseline_threshold,
        event_action_baseline_fit,
    ) = fit_ridge_with_validation(
        train_event.current_frame_action,
        train_event.event_active,
        validation_event.current_frame_action,
        validation_event.event_active,
        classification=True,
    )
    assert event_model_threshold is not None
    assert event_baseline_threshold is not None
    assert event_action_baseline_threshold is not None
    event_model_predictions = (
        event_model.predict(test_event.predicted_h8)[:, 0] >= event_model_threshold
    ).astype(np.int8)
    event_baseline_predictions = (
        event_baseline.predict(test_event.current_frame)[:, 0]
        >= event_baseline_threshold
    ).astype(np.int8)
    event_action_baseline_predictions = (
        event_action_baseline.predict(test_event.current_frame_action)[:, 0]
        >= event_action_baseline_threshold
    ).astype(np.int8)
    event_labels = test_event.event_active.astype(np.int8)

    return {
        "object_model_errors": object_model_errors.tolist(),
        "object_baseline_errors": object_baseline_errors.tolist(),
        "object_action_baseline_errors": object_action_baseline_errors.tolist(),
        "event_model_predictions": event_model_predictions.tolist(),
        "event_baseline_predictions": event_baseline_predictions.tolist(),
        "event_action_baseline_predictions": event_action_baseline_predictions.tolist(),
        "event_labels": event_labels.tolist(),
        "object_test_sample_ids_sha256": _json_sha256(test_object.sample_ids),
        "event_test_sample_ids_sha256": _json_sha256(test_event.sample_ids),
        "ridge": {
            "object_predicted_h8": object_model_fit,
            "object_current_frame": object_baseline_fit,
            "object_current_frame_action": object_action_baseline_fit,
            "event_predicted_h8": event_model_fit,
            "event_current_frame": event_baseline_fit,
            "event_current_frame_action": event_action_baseline_fit,
        },
        "feature_sha256": {
            "train_predicted_h8": _array_sha256(train.predicted_h8),
            "train_current_frame": _array_sha256(train.current_frame),
            "train_current_frame_action": _array_sha256(train.current_frame_action),
            "validation_predicted_h8": _array_sha256(validation.predicted_h8),
            "validation_current_frame": _array_sha256(validation.current_frame),
            "validation_current_frame_action": _array_sha256(
                validation.current_frame_action
            ),
            "test_predicted_h8": _array_sha256(test.predicted_h8),
            "test_current_frame": _array_sha256(test.current_frame),
            "test_current_frame_action": _array_sha256(test.current_frame_action),
        },
    }


def cluster_aware_probe_comparisons(
    per_seed: Mapping[str, Mapping[str, Any]],
    *,
    train_seeds: Sequence[int],
    confidence: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Compare probes without counting repeated test samples as new evidence.

    All three checkpoints are evaluated on the same held-out sample IDs.  The
    cluster-level object error is the RMS error across training seeds for one
    sample; the cluster-level event prediction is the three-seed majority
    vote.  Bootstrap and McNemar inference therefore see each held-out sample
    exactly once.  In addition, every training seed must independently pass
    both probe gates, preventing one strong seed from hiding a failed seed.
    """

    seeds = tuple(int(value) for value in train_seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("cluster-aware H8 comparison requires three train seeds")
    keys = tuple(str(value) for value in seeds)
    if set(per_seed) != set(keys):
        raise ValueError("per-seed probe results do not match the train seeds")

    object_hashes = {
        str(per_seed[key].get("object_test_sample_ids_sha256", "")) for key in keys
    }
    event_hashes = {
        str(per_seed[key].get("event_test_sample_ids_sha256", "")) for key in keys
    }
    if len(object_hashes) != 1 or "" in object_hashes:
        raise ValueError("object probe sample IDs differ across train seeds")
    if len(event_hashes) != 1 or "" in event_hashes:
        raise ValueError("event probe sample IDs differ across train seeds")

    object_model = _seed_matrix(per_seed, keys, "object_model_errors", np.float64)
    object_baseline = _seed_matrix(per_seed, keys, "object_baseline_errors", np.float64)
    object_action_baseline = _seed_matrix(
        per_seed, keys, "object_action_baseline_errors", np.float64
    )
    event_model = _seed_matrix(per_seed, keys, "event_model_predictions", np.int8)
    event_baseline = _seed_matrix(per_seed, keys, "event_baseline_predictions", np.int8)
    event_action_baseline = _seed_matrix(
        per_seed, keys, "event_action_baseline_predictions", np.int8
    )
    event_labels = _seed_matrix(per_seed, keys, "event_labels", np.int8)
    if (
        object_model.shape != object_baseline.shape
        or object_model.shape != object_action_baseline.shape
    ):
        raise ValueError("per-seed object probe evidence is not paired")
    if (
        event_model.shape != event_baseline.shape
        or event_model.shape != event_action_baseline.shape
        or event_model.shape != event_labels.shape
    ):
        raise ValueError("per-seed event probe evidence is not paired")
    if not np.all(event_labels == event_labels[0]):
        raise ValueError("event labels differ across train seeds")

    per_seed_comparisons: dict[str, Any] = {}
    every_seed_passed = True
    for offset, key in enumerate(keys):
        object_vs_current = paired_rmse_comparison(
            object_model[offset],
            object_baseline[offset],
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + 10 + 2 * offset,
        )
        object_vs_current_action = paired_rmse_comparison(
            object_model[offset],
            object_action_baseline[offset],
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + 110 + 2 * offset,
        )
        event_vs_current = paired_balanced_accuracy_comparison(
            event_model[offset],
            event_baseline[offset],
            event_labels[offset],
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + 11 + 2 * offset,
        )
        event_vs_current_action = paired_balanced_accuracy_comparison(
            event_model[offset],
            event_action_baseline[offset],
            event_labels[offset],
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + 111 + 2 * offset,
        )
        object_current_passed = _object_comparison_passed(object_vs_current)
        object_action_passed = _object_comparison_passed(object_vs_current_action)
        event_current_passed = _event_comparison_passed(event_vs_current)
        event_action_passed = _event_comparison_passed(event_vs_current_action)
        object_passed = bool(object_current_passed and object_action_passed)
        event_passed = bool(event_current_passed and event_action_passed)
        seed_passed = bool(object_passed and event_passed)
        every_seed_passed = bool(every_seed_passed and seed_passed)
        per_seed_comparisons[key] = {
            "passed": seed_passed,
            "object_significantly_better": object_passed,
            "event_significantly_better": event_passed,
            "object_vs_current_frame": object_vs_current,
            "object_vs_current_frame_action": object_vs_current_action,
            "event_vs_current_frame": event_vs_current,
            "event_vs_current_frame_action": event_vs_current_action,
        }

    # Collapse only the repeated train-seed dimension.  Sample order remains
    # the selection's deterministic held-out order.
    clustered_object_model = np.sqrt(np.mean(np.square(object_model), axis=0))
    clustered_object_baseline = np.sqrt(np.mean(np.square(object_baseline), axis=0))
    clustered_object_action_baseline = np.sqrt(
        np.mean(np.square(object_action_baseline), axis=0)
    )
    clustered_event_model = (event_model.sum(axis=0) >= 2).astype(np.int8)
    clustered_event_baseline = (event_baseline.sum(axis=0) >= 2).astype(np.int8)
    clustered_event_action_baseline = (event_action_baseline.sum(axis=0) >= 2).astype(
        np.int8
    )
    clustered_event_labels = event_labels[0]
    object_vs_current = paired_rmse_comparison(
        clustered_object_model,
        clustered_object_baseline,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    object_vs_current_action = paired_rmse_comparison(
        clustered_object_model,
        clustered_object_action_baseline,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed + 100,
    )
    event_vs_current = paired_balanced_accuracy_comparison(
        clustered_event_model,
        clustered_event_baseline,
        clustered_event_labels,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    event_vs_current_action = paired_balanced_accuracy_comparison(
        clustered_event_model,
        clustered_event_action_baseline,
        clustered_event_labels,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed + 101,
    )
    object_passed = bool(
        _object_comparison_passed(object_vs_current)
        and _object_comparison_passed(object_vs_current_action)
    )
    event_passed = bool(
        _event_comparison_passed(event_vs_current)
        and _event_comparison_passed(event_vs_current_action)
    )
    robust_passed = bool(object_passed and event_passed and every_seed_passed)
    return {
        "aggregate": {
            "object_model_errors": clustered_object_model.tolist(),
            "object_baseline_errors": clustered_object_baseline.tolist(),
            "object_action_baseline_errors": (
                clustered_object_action_baseline.tolist()
            ),
            "event_model_predictions": clustered_event_model.tolist(),
            "event_baseline_predictions": clustered_event_baseline.tolist(),
            "event_action_baseline_predictions": (
                clustered_event_action_baseline.tolist()
            ),
            "event_labels": clustered_event_labels.tolist(),
        },
        "comparisons": {
            # Backward-compatible aliases remain the current-frame-only
            # comparisons; the composite gates below require both baselines.
            "object": object_vs_current,
            "event": event_vs_current,
            "object_vs_current_frame": object_vs_current,
            "object_vs_current_frame_action": object_vs_current_action,
            "event_vs_current_frame": event_vs_current,
            "event_vs_current_frame_action": event_vs_current_action,
            "object_significantly_better": object_passed,
            "event_significantly_better": event_passed,
            "all_train_seeds_significantly_better": every_seed_passed,
            "robust_improvement": robust_passed,
            "per_seed": per_seed_comparisons,
            "inference_unit": "held_out_sample_id_cluster",
            "train_seed_aggregation": {
                "object": "per_sample_root_mean_square_error",
                "event": "per_sample_three_seed_majority_vote",
            },
            "formal_requires_every_train_seed": True,
            "formal_requires_both_baselines": True,
        },
        "cluster_evidence": {
            "train_seed_count": len(seeds),
            "object_unique_sample_clusters": int(object_model.shape[1]),
            "event_unique_sample_clusters": int(event_model.shape[1]),
            "object_sample_ids_sha256": next(iter(object_hashes)),
            "event_sample_ids_sha256": next(iter(event_hashes)),
            "pooled_seed_sample_rows_used_for_inference": False,
        },
    }


def run_future_probe(
    config_path: str | Path,
    *,
    project_root: str | Path,
    checkpoint_root: str | Path | None = None,
    training_summary_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    train_seeds: Sequence[int] | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    verify_hdf5_sha256: bool = True,
    object_samples: Mapping[str, int] = FORMAL_OBJECT_SAMPLES,
    event_samples: Mapping[str, int] = FORMAL_EVENT_SAMPLES,
    formal_protocol: bool = True,
) -> dict[str, Any]:
    """Run the manifest-bound probe for all three canonical M1 train seeds."""

    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve(strict=True)
    config = _load_yaml(config_file)
    configured_seeds = tuple(int(value) for value in config["training"]["seeds"])
    seeds = tuple(
        int(value)
        for value in (train_seeds if train_seeds is not None else configured_seeds)
    )
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("formal H8 probing requires three distinct train seeds")
    canonical_manifest = _resolve(root, str(config["data"]["manifest"]))
    manifest_file = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else canonical_manifest
    )
    expected_manifest_sha256 = str(config["data"]["expected_manifest_sha256"])
    if _sha256_file(manifest_file) != expected_manifest_sha256:
        raise ValueError("M1 probe manifest hash differs from the pinned config")
    checkpoints = (
        Path(checkpoint_root).resolve()
        if checkpoint_root is not None
        else _resolve(root, str(config["training"]["checkpoint_root"]))
    )
    canonical_training_summary = (
        _resolve(root, str(config["training"]["report_root"])) / "training_summary.json"
    )
    training_summary_file = (
        Path(training_summary_path).resolve()
        if training_summary_path is not None
        else canonical_training_summary
    )
    if formal_protocol:
        override_names = [
            name
            for name, value in (
                ("checkpoint_root", checkpoint_root),
                ("training_summary_path", training_summary_path),
                ("manifest_path", manifest_path),
                ("train_seeds", train_seeds),
            )
            if value is not None
        ]
        if override_names:
            raise ValueError(
                "formal H8 probe forbids diagnostic path/seed overrides: "
                f"{override_names}"
            )
        if seeds != configured_seeds:
            raise ValueError("formal H8 probe train seeds differ from config")
        if dict(object_samples) != FORMAL_OBJECT_SAMPLES:
            raise ValueError("formal H8 probe object sample caps changed")
        if dict(event_samples) != FORMAL_EVENT_SAMPLES:
            raise ValueError("formal H8 probe event sample caps changed")
        if not verify_hdf5_sha256:
            raise ValueError("formal H8 probe requires HDF5 hash verification")
    if int(config["data"]["action_horizon"]) != M1_PROBE_HORIZON:
        raise ValueError("M1 probe requires same-demo 8-step action chunks")

    training_summary_sha256 = _sha256_file(training_summary_file)
    training_summary = _load_json(training_summary_file)
    primary_bindings = _validate_training_summary_binding(
        training_summary,
        config=config,
        project_root=root,
        config_file=config_file,
        config_sha256=_sha256_file(config_file),
        manifest_sha256=expected_manifest_sha256,
        checkpoint_root=checkpoints,
        train_seeds=seeds,
        formal_protocol=formal_protocol,
    )

    manifest = M1ManifestIndex.from_path(
        manifest_file,
        verify_hdf5_sha256=verify_hdf5_sha256,
        verify_hdf5_contract=True,
    )
    datasets = {
        split: _dataset(manifest, split=split, config=config)
        for split in CANONICAL_SPLITS
    }
    selections = {
        split: select_probe_indices(
            datasets[split],
            object_samples=int(object_samples[split]),
            event_samples=int(event_samples[split]),
            seed=19_071 + offset,
        )
        for offset, split in enumerate(CANONICAL_SPLITS)
    }

    per_seed: dict[str, Any] = {}
    checkpoint_hashes: dict[str, str] = {}
    strict_reload: dict[str, bool] = {}
    target_device = torch.device(device)
    for train_seed in seeds:
        checkpoint = checkpoints / PRIMARY_VARIANT / f"seed_{train_seed}"
        binding = primary_bindings[str(train_seed)]
        if checkpoint.resolve() != Path(str(binding["checkpoint"])).resolve():
            raise ValueError(
                "primary checkpoint path differs from training summary for "
                f"seed {train_seed}"
            )
        tree_before = checkpoint_tree_sha256(checkpoint)
        if tree_before != binding["checkpoint_tree_sha256"]:
            raise ValueError(
                "primary checkpoint tree differs from training summary for "
                f"seed {train_seed}"
            )
        model, _, _, _, metadata = load_m1_checkpoint(
            checkpoint,
            device=target_device,
            expected_schema_version=str(config["data"]["schema_version"]),
        )
        schema = metadata["schema"]
        lineage = metadata["dataset_manifest"]
        if schema.get("model_variant") != PRIMARY_VARIANT:
            raise ValueError("future probe checkpoint is not state_vision_future")
        if int(schema.get("train_seed", -1)) != train_seed:
            raise ValueError("future probe checkpoint train seed mismatch")
        validate_loaded_checkpoint_vision(config, model, metadata)
        if lineage.get("manifest_sha256") != manifest.manifest_sha256:
            raise ValueError("checkpoint/dataset manifest lineage mismatch")
        if lineage.get("split") != "train":
            raise ValueError("M1 checkpoint was not bound to the training split")
        if model.config.use_state is not True or model.config.use_vision is not True:
            raise ValueError("H8 probe requires the state+vision model")
        if model.config.capacity_control != "future_head":
            raise ValueError("H8 probe checkpoint does not contain the future head")

        features = {
            split: extract_probe_features(
                model,
                datasets[split],
                selections[split],
                device=target_device,
                batch_size=batch_size,
            )
            for split in CANONICAL_SPLITS
        }
        result = evaluate_feature_sets(
            features["train"],
            features["validation"],
            features["test"],
            train_selection=selections["train"],
            validation_selection=selections["validation"],
            test_selection=selections["test"],
        )
        tree_after = checkpoint_tree_sha256(checkpoint)
        if tree_after != tree_before:
            raise RuntimeError(
                f"primary checkpoint changed during probe for seed {train_seed}"
            )
        checkpoint_hashes[str(train_seed)] = tree_after
        strict_reload[str(train_seed)] = True
        result["checkpoint"] = str(checkpoint)
        result["checkpoint_tree_sha256"] = checkpoint_hashes[str(train_seed)]
        per_seed[str(train_seed)] = result

    for train_seed in seeds:
        checkpoint = checkpoints / PRIMARY_VARIANT / f"seed_{train_seed}"
        if checkpoint_tree_sha256(checkpoint) != checkpoint_hashes[str(train_seed)]:
            raise RuntimeError(
                f"primary checkpoint changed before probe completion for seed {train_seed}"
            )

    confidence = float(config["statistics"]["confidence"])
    bootstrap_samples = int(config["statistics"]["bootstrap_samples"])
    bootstrap_seed = int(config["statistics"]["bootstrap_seed"])
    clustered = cluster_aware_probe_comparisons(
        per_seed,
        train_seeds=seeds,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    robust_improvement = bool(clustered["comparisons"]["robust_improvement"])
    selection_summaries = {
        split: selections[split].summary() for split in CANONICAL_SPLITS
    }
    split_lineage = {
        split: datasets[split].checkpoint_lineage() for split in CANONICAL_SPLITS
    }
    split_ids = {
        split: {
            selections[split].sample_ids[index]
            for index in selections[split].union_indices
        }
        for split in CANONICAL_SPLITS
    }
    disjoint = all(
        split_ids[left].isdisjoint(split_ids[right])
        for offset, left in enumerate(CANONICAL_SPLITS)
        for right in CANONICAL_SPLITS[offset + 1 :]
    )
    evidence_complete = bool(
        manifest.hdf5_contract_verified
        and (manifest.hdf5_sha256_verified or not formal_protocol)
        and disjoint
        and all(strict_reload.values())
        and all(
            value["event_class_counts"]["0"] == value["event_class_counts"]["1"]
            for value in selection_summaries.values()
        )
    )
    training_summary_stable = (
        _sha256_file(training_summary_file) == training_summary_sha256
    )
    evidence_complete = bool(evidence_complete and training_summary_stable)
    passed = bool(formal_protocol and evidence_complete and robust_improvement)
    return {
        "format_version": FORMAT_VERSION,
        "phase": "M1",
        "formal_protocol": bool(formal_protocol),
        "passed": passed,
        "claim_allowed": passed,
        "baseline": "current_frame_only",
        "required_baselines": [
            "current_frame_only",
            "current_frame_plus_same_demonstration_action_chunk",
        ],
        "probe_action_source": PROBE_ACTION_SOURCE,
        "probe_action_protocol": {
            "mode": "offline_teacher_forced_same_demonstration",
            "action_chunk_horizon": M1_PROBE_HORIZON,
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
        "horizon": M1_PROBE_HORIZON,
        "train_seeds": list(seeds),
        **clustered["aggregate"],
        "comparisons": clustered["comparisons"],
        "cluster_evidence": clustered["cluster_evidence"],
        "per_seed": per_seed,
        "selection": selection_summaries,
        "split_lineage": split_lineage,
        "checkpoint_sha256": checkpoint_hashes,
        "strict_reload": strict_reload,
        "artifact_sha256": {
            "config": _sha256_file(config_file),
            "dataset_manifest": manifest.manifest_sha256,
            "training_summary": training_summary_sha256,
            "visual_backbone": str(
                config["initialization"]["expected_vision_weights_sha256"]
            ),
        },
        "training_summary": str(training_summary_file),
        "training_summary_sha256": training_summary_sha256,
        "training_summary_stable_during_probe": training_summary_stable,
        "primary_checkpoint_tree_sha256": checkpoint_hashes,
        "primary_checkpoint_trees_stable_during_probe": True,
        "no_future_target_leakage": {
            "passed": evidence_complete,
            "feature_model_inputs": [
                "states",
                "state_valid_mask",
                "past_actions",
                "images.current_history",
                "task_index",
                "same_demonstration_expert_action_chunk",
            ],
            "future_head_input_signature": [
                "planning_features",
                "visual_tokens",
                "candidate_actions",
            ],
            "current_frame_baseline_frames": 1,
            "current_frame_baseline_uses_same_frozen_teacher": True,
            "candidate_actions_source": PROBE_ACTION_SOURCE,
            "dataset_action_targets_read": True,
            "dataset_action_targets_forwarded_to_future_head": True,
            "matched_baseline_receives_identical_action_targets": True,
            "target_action_pairing": "same_demonstration_window",
            "deployed_causal_claim_allowed": False,
            "future_images_forwarded": False,
            "future_states_forwarded": False,
            "probe_labels_forwarded": False,
            "probe_labels_accessor": "M1WindowDataset.probe_labels",
            "test_samples_used_for_fit_or_selection": 0,
            "train_validation_test_sample_ids_disjoint": disjoint,
            "manifest_hdf5_sha256_verified": manifest.hdf5_sha256_verified,
            "manifest_hdf5_contract_verified": manifest.hdf5_contract_verified,
        },
    }


def _dataset(
    manifest: M1ManifestIndex,
    *,
    split: str,
    config: Mapping[str, Any],
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


def _validate_training_summary_binding(
    value: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    project_root: Path,
    config_file: Path,
    config_sha256: str,
    manifest_sha256: str,
    checkpoint_root: Path,
    train_seeds: Sequence[int],
    formal_protocol: bool,
) -> dict[str, dict[str, str]]:
    """Return primary checkpoint bindings after fail-closed summary checks."""

    if value.get("format_version") != "wam.multimodal.m1.training/1":
        raise ValueError("unsupported M1 training summary")
    if config is not None:
        validate_training_summary_vision(value, config, project_root=project_root)
    if value.get("config_sha256") != config_sha256:
        raise ValueError("training summary does not bind the probe config")
    if _resolve(project_root, str(value.get("config", ""))) != config_file:
        raise ValueError("training summary config path differs from probe config")
    if value.get("manifest_sha256") != manifest_sha256:
        raise ValueError("training summary does not bind the probe manifest")
    seeds = tuple(int(item) for item in train_seeds)
    if tuple(int(item) for item in value.get("train_seeds", ())) != seeds:
        raise ValueError("training summary train seeds differ from probe seeds")
    if PRIMARY_VARIANT not in value.get("variants", ()):
        raise ValueError("training summary has no primary future-latent variant")
    if _resolve(project_root, str(value.get("checkpoint_root", ""))) != checkpoint_root:
        raise ValueError("training summary checkpoint root differs from probe root")
    if formal_protocol and (
        value.get("formal_protocol") is not True or value.get("passed") is not True
    ):
        raise ValueError("formal H8 probe requires passed formal training evidence")

    hashes = value.get("checkpoint_sha256")
    strict_reload = value.get("strict_reload")
    reports = value.get("reports")
    if not isinstance(hashes, Mapping) or not isinstance(strict_reload, Mapping):
        raise ValueError("training summary lacks checkpoint/strict-reload evidence")
    primary_hashes = hashes.get(PRIMARY_VARIANT)
    primary_strict = strict_reload.get(PRIMARY_VARIANT)
    if not isinstance(primary_hashes, Mapping) or not isinstance(
        primary_strict, Mapping
    ):
        raise ValueError("training summary lacks primary checkpoint evidence")
    if not isinstance(reports, list):
        raise ValueError("training summary run reports are missing")

    primary_reports: dict[str, Mapping[str, Any]] = {}
    for report in reports:
        if not isinstance(report, Mapping) or report.get("variant") != PRIMARY_VARIANT:
            continue
        key = str(int(report.get("train_seed", -1)))
        if key in primary_reports:
            raise ValueError(f"duplicate primary training report for seed {key}")
        primary_reports[key] = report
    seed_keys = {str(seed) for seed in seeds}
    if set(primary_hashes) != seed_keys or set(primary_strict) != seed_keys:
        raise ValueError(
            "primary checkpoint evidence does not cover exactly three seeds"
        )
    if set(primary_reports) != seed_keys:
        raise ValueError("primary run reports do not cover exactly three seeds")

    bindings: dict[str, dict[str, str]] = {}
    for seed in seeds:
        key = str(seed)
        report = primary_reports[key]
        expected_path = checkpoint_root / PRIMARY_VARIANT / f"seed_{seed}"
        report_path = _resolve(project_root, str(report.get("checkpoint", "")))
        if report_path != expected_path:
            raise ValueError(f"primary checkpoint report path mismatch for seed {seed}")
        summary_tree = str(primary_hashes[key])
        report_tree = str(report.get("checkpoint_tree_sha256", ""))
        if not _is_sha256(summary_tree) or summary_tree != report_tree:
            raise ValueError(f"primary checkpoint hash mismatch for seed {seed}")
        report_strict = report.get("strict_reload")
        summary_strict = primary_strict[key]
        if (
            not isinstance(report_strict, Mapping)
            or report_strict.get("passed") is not True
            or not isinstance(summary_strict, Mapping)
            or summary_strict.get("passed") is not True
        ):
            raise ValueError(f"primary checkpoint seed {seed} lacks strict reload")
        bindings[key] = {
            "checkpoint": str(expected_path),
            "checkpoint_tree_sha256": summary_tree,
        }
    return bindings


def _balanced_rows(
    groups: Mapping[str, Sequence[tuple[int, str, int, str]]],
    *,
    maximum: int,
    rng: np.random.Generator,
    require_equal: bool = False,
) -> list[tuple[int, str, int, str]]:
    names = tuple(sorted(groups))
    if not names:
        raise ValueError("cannot balance empty probe groups")
    if require_equal:
        per_group = min(maximum // len(names), *(len(groups[name]) for name in names))
        if per_group <= 0:
            raise ValueError("probe cap is too small to represent every group")
        quotas = {name: per_group for name in names}
    else:
        base, remainder = divmod(maximum, len(names))
        quotas = {
            name: min(len(groups[name]), base + int(offset < remainder))
            for offset, name in enumerate(names)
        }
        unused = maximum - sum(quotas.values())
        while unused > 0:
            eligible = [name for name in names if quotas[name] < len(groups[name])]
            if not eligible:
                break
            for name in eligible:
                if unused <= 0:
                    break
                quotas[name] += 1
                unused -= 1
    selected: list[tuple[int, str, int, str]] = []
    for name in names:
        rows = list(groups[name])
        order = rng.permutation(len(rows))[: quotas[name]].tolist()
        selected.extend(rows[index] for index in order)
    selected.sort(key=lambda row: row[3])
    return selected


def _best_balanced_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int8)
    if values.ndim != 1 or truth.ndim != 1 or values.shape != truth.shape:
        raise ValueError("threshold scores/labels must be paired 1-D arrays")
    if not np.isfinite(values).all() or not _binary(truth.reshape(-1, 1)):
        raise ValueError("threshold inputs must be finite with binary labels")
    unique = np.unique(values)
    if unique.size == 1:
        candidates = np.asarray([unique[0] - 1.0, unique[0], unique[0] + 1.0])
    else:
        midpoints = 0.5 * (unique[:-1] + unique[1:])
        candidates = np.concatenate(
            (
                [np.nextafter(unique[0], -np.inf)],
                midpoints,
                [np.nextafter(unique[-1], np.inf)],
            )
        )
    positives = truth == 1
    negatives = ~positives
    if not positives.any() or not negatives.any():
        raise ValueError("balanced threshold selection requires both classes")
    best_threshold = float(candidates[0])
    best_score = -1.0
    for threshold in candidates.tolist():
        prediction = values >= threshold
        score = 0.5 * (
            float(prediction[positives].mean()) + float((~prediction[negatives]).mean())
        )
        if score > best_score or (
            score == best_score
            and (abs(threshold), threshold) < (abs(best_threshold), best_threshold)
        ):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def _finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _target_matrix(value: np.ndarray, rows: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[0] != rows or result.shape[1] == 0:
        raise ValueError(f"{name} must align with its feature rows")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _binary(value: np.ndarray) -> bool:
    return bool(np.isin(np.asarray(value), (0, 1)).all())


def _seed_matrix(
    per_seed: Mapping[str, Mapping[str, Any]],
    keys: Sequence[str],
    field: str,
    dtype: np.dtype[Any] | type[Any],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for key in keys:
        if field not in per_seed[key]:
            raise ValueError(f"per-seed probe evidence lacks {field}")
        row = np.asarray(per_seed[key][field], dtype=dtype)
        if row.ndim != 1 or not row.size:
            raise ValueError(f"per-seed probe field {field} must be non-empty 1-D")
        if expected_shape is None:
            expected_shape = row.shape
        elif row.shape != expected_shape:
            raise ValueError(f"per-seed probe field {field} has inconsistent rows")
        rows.append(row)
    result = np.stack(rows, axis=0)
    if not np.isfinite(result).all():
        raise ValueError(f"per-seed probe field {field} contains NaN or Inf")
    return result


def _object_comparison_passed(value: Mapping[str, Any]) -> bool:
    return bool(
        float(value["baseline_minus_model_rmse"]) > 0.0
        and float(value["ci_lower"]) > 0.0
    )


def _event_comparison_passed(value: Mapping[str, Any]) -> bool:
    mcnemar = value.get("mcnemar")
    return bool(
        isinstance(mcnemar, Mapping)
        and float(value["model_minus_baseline_balanced_accuracy"]) > 0.0
        and float(value["ci_lower"]) > 0.0
        and float(mcnemar["p_value_two_sided"]) < 0.05
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must contain a mapping: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("M1 probe config must contain a mapping")
    return value


__all__ = [
    "CANONICAL_SPLITS",
    "FORMAL_EVENT_SAMPLES",
    "FORMAL_OBJECT_SAMPLES",
    "FORMAT_VERSION",
    "ProbeFeatureSet",
    "ProbeSelection",
    "PROBE_ACTION_SOURCE",
    "RidgeModel",
    "cluster_aware_probe_comparisons",
    "evaluate_feature_sets",
    "extract_probe_features",
    "fit_ridge_with_validation",
    "run_future_probe",
    "select_probe_indices",
]
