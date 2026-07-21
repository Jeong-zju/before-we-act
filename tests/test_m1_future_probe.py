from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eval.m1_future_probe import (
    PROBE_ACTION_SOURCE,
    ProbeFeatureSet,
    ProbeSelection,
    _validate_training_summary_binding,
    cluster_aware_probe_comparisons,
    evaluate_feature_sets,
    extract_probe_features,
    fit_ridge_with_validation,
    select_probe_indices,
)


class _CatalogDataset:
    task_order = (
        "visual_event_stop",
        "visual_target_select",
        "visual_obstacle_avoid",
    )

    def __init__(self, root: Path) -> None:
        self.records = []
        self._rows = []
        index = 0
        for task_offset, task in enumerate(self.task_order):
            path = root / f"{task}.hdf5"
            self.records.append(SimpleNamespace(path=path, task_id=task))
            for step in range(20):
                event = step % 2 if task == "visual_event_stop" else 1
                self._rows.append((path, step, task_offset, event))
                index += 1
        self.decision_window_indices = tuple(range(index))

    def __len__(self) -> int:
        return len(self._rows)

    def sample_lineage(self, index: int) -> SimpleNamespace:
        path, step, _, _ = self._rows[index]
        return SimpleNamespace(path=path, decision_t=step)

    def probe_labels(self, index: int) -> dict[str, torch.Tensor]:
        _, step, task, event = self._rows[index]
        return {
            "h8_center_xy": torch.tensor([task, step], dtype=torch.float32),
            "h8_event_active": torch.tensor(bool(event)),
        }


def test_probe_selection_is_task_and_event_balanced_and_deterministic(
    tmp_path: Path,
) -> None:
    dataset = _CatalogDataset(tmp_path)
    left = select_probe_indices(
        dataset, object_samples=12, event_samples=10, seed=77
    )
    right = select_probe_indices(
        dataset, object_samples=12, event_samples=10, seed=77
    )

    assert left == right
    summary = left.summary()
    assert summary["object_task_counts"] == {
        "visual_event_stop": 4,
        "visual_obstacle_avoid": 4,
        "visual_target_select": 4,
    }
    assert summary["event_class_counts"] == {"0": 5, "1": 5}
    assert set(left.union_indices) == set(left.object_indices) | set(left.event_indices)


def _synthetic_split(seed: int, rows: int, prefix: str) -> ProbeFeatureSet:
    rng = np.random.default_rng(seed)
    event = np.tile(np.asarray([0, 1], dtype=np.int8), rows // 2)
    if event.size < rows:
        event = np.concatenate((event, np.asarray([0], dtype=np.int8)))
    center = rng.normal(size=(rows, 2))
    predicted = np.column_stack(
        (center, event * 2.0 - 1.0, rng.normal(scale=0.02, size=rows))
    )
    current = rng.normal(size=(rows, 4))
    return ProbeFeatureSet(
        predicted_h8=predicted.astype(np.float32),
        current_frame=current.astype(np.float32),
        current_frame_action=np.concatenate((current, current), axis=1).astype(
            np.float32
        ),
        center_xy=center.astype(np.float32),
        event_active=event,
        sample_ids=tuple(f"{prefix}:{index}" for index in range(rows)),
    )


def _selection(rows: int) -> ProbeSelection:
    indices = tuple(range(rows))
    labels = tuple(index % 2 for index in indices)
    return ProbeSelection(
        object_indices=indices,
        event_indices=indices,
        union_indices=indices,
        object_tasks=tuple("visual_event_stop" for _ in indices),
        event_labels=labels,
        sample_ids={index: f"sample:{index}" for index in indices},
    )


def test_ridge_probes_select_on_validation_and_keep_test_out_of_fit() -> None:
    train = _synthetic_split(1, 120, "train")
    validation = _synthetic_split(2, 60, "validation")
    test = _synthetic_split(3, 80, "test")
    result = evaluate_feature_sets(
        train,
        validation,
        test,
        train_selection=_selection(120),
        validation_selection=_selection(60),
        test_selection=_selection(80),
    )

    assert len(result["object_model_errors"]) == 80
    assert np.mean(result["object_model_errors"]) < 0.02
    assert np.mean(result["object_baseline_errors"]) > 0.5
    assert result["event_model_predictions"] == result["event_labels"]
    for value in result["ridge"].values():
        assert value["fit_samples"] == 120
        assert value["validation_samples"] == 60
        assert value["test_samples_used_for_fit_or_selection"] == 0


def test_ridge_rejects_nonbinary_classifier_targets() -> None:
    features = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError, match="binary"):
        fit_ridge_with_validation(
            features,
            np.asarray([0, 1, 2, 0]),
            features,
            np.asarray([0, 1, 0, 1]),
            classification=True,
        )


class _ExtractDataset:
    task_order = ("visual_event_stop",)

    def __init__(self, root: Path) -> None:
        path = root / "episode.hdf5"
        self.records = [SimpleNamespace(path=path, task_id="visual_event_stop")]
        self.decision_window_indices = (0, 1)
        self._path = path

    def __len__(self) -> int:
        return 2

    def sample_lineage(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(path=self._path, decision_t=index)

    def probe_labels(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "h8_center_xy": torch.tensor([index, index + 1], dtype=torch.float32),
            "h8_event_active": torch.tensor(bool(index)),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        first = 10 + index
        latest = 30 + index
        return {
            "states": torch.zeros(2, 22),
            "state_valid_mask": torch.ones(2, dtype=torch.bool),
            "past_actions": torch.zeros(1, 8),
            "past_action_valid_mask": torch.ones(1, dtype=torch.bool),
            "images": torch.stack(
                (
                    torch.full((1, 3, 4, 4), first, dtype=torch.uint8),
                    torch.full((1, 3, 4, 4), latest, dtype=torch.uint8),
                )
            ),
            "task_index": torch.tensor(0, dtype=torch.long),
            "action_targets": torch.full((8, 8), float(index + 1)),
            "future_states": torch.full((8, 22), float("nan")),
            "future_images": torch.zeros(4, 1, 3, 4, 4, dtype=torch.uint8),
            "future_image_novelty_mask": torch.ones(4, 1, dtype=torch.bool),
            "future_horizons": torch.tensor((1, 2, 4, 8)),
        }


class _FakeVision:
    def __call__(self, images: torch.Tensor) -> SimpleNamespace:
        values = images.float().mean(dim=(-3, -2, -1))
        return SimpleNamespace(pooled_latent=values.unsqueeze(-1).repeat(1, 1, 4))


class _FakeFutureHead:
    def __call__(
        self,
        planning: torch.Tensor,
        visual: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        batch = actions.shape[0]
        result = torch.zeros(batch, 4, 4, device=actions.device)
        result[:, 3] = actions.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        return result


class _FakeModel:
    future_horizons = (1, 2, 4, 8)
    planning_feature_dim = 4

    def __init__(self) -> None:
        self.future_head = _FakeFutureHead()
        self.vision_encoder = _FakeVision()
        self.seen_images: torch.Tensor | None = None

    def encode(
        self,
        states: torch.Tensor,
        past_actions: torch.Tensor,
        valid_mask: torch.Tensor,
        images: torch.Tensor,
        task_index: torch.Tensor,
    ) -> SimpleNamespace:
        del states, past_actions, valid_mask, task_index
        self.seen_images = images.detach().clone()
        return SimpleNamespace(
            planning_features=torch.zeros(images.shape[0], 4),
            visual_tokens=torch.zeros(images.shape[0], 1, 4),
        )


def test_feature_extraction_uses_same_demo_actions_and_matched_baseline(
    tmp_path: Path,
) -> None:
    dataset = _ExtractDataset(tmp_path)
    selection = select_probe_indices(
        dataset, object_samples=2, event_samples=2, seed=1
    )
    model = _FakeModel()
    result = extract_probe_features(
        model, dataset, selection, device="cpu", batch_size=2
    )

    assert model.seen_images is not None
    assert model.seen_images.shape == (2, 2, 1, 3, 4, 4)
    # The baseline is the latest frame (30/31), not the history average.
    assert sorted(result.current_frame[:, 0].tolist()) == [30.0, 31.0]
    # The H8 feature and matched baseline receive the identical same-demo
    # expert action chunk (all ones/all twos).
    assert sorted(result.predicted_h8[:, 0].tolist()) == [1.0, 2.0]
    assert result.current_frame_action.shape == (2, 68)
    assert sorted(result.current_frame_action[:, -1].tolist()) == [1.0, 2.0]
    assert np.isfinite(result.predicted_h8).all()
    assert PROBE_ACTION_SOURCE == "same_demonstration_expert_action_chunk"


def _seed_probe_result(*, object_model_error: float) -> dict[str, object]:
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 100)
    model = labels.copy()
    baseline = labels.copy()
    for pair in range(100):
        if pair % 10 == 0:
            model[2 * pair : 2 * pair + 2] = 1 - labels[
                2 * pair : 2 * pair + 2
            ]
        if pair % 10 < 4:
            baseline[2 * pair : 2 * pair + 2] = 1 - labels[
                2 * pair : 2 * pair + 2
            ]
    return {
        "object_model_errors": [object_model_error] * 200,
        "object_baseline_errors": [0.3] * 200,
        "object_action_baseline_errors": [0.35] * 200,
        "event_model_predictions": model.tolist(),
        "event_baseline_predictions": baseline.tolist(),
        "event_action_baseline_predictions": baseline.tolist(),
        "event_labels": labels.tolist(),
        "object_test_sample_ids_sha256": "a" * 64,
        "event_test_sample_ids_sha256": "b" * 64,
    }


def test_cluster_aware_probe_requires_every_training_seed_to_pass() -> None:
    per_seed = {
        "101": _seed_probe_result(object_model_error=0.1),
        "202": _seed_probe_result(object_model_error=0.1),
        # The sample-cluster aggregate still improves, but this seed regresses.
        "303": _seed_probe_result(object_model_error=0.4),
    }
    result = cluster_aware_probe_comparisons(
        per_seed,
        train_seeds=(101, 202, 303),
        confidence=0.95,
        bootstrap_samples=300,
        bootstrap_seed=9,
    )

    assert result["comparisons"]["object_significantly_better"] is True
    assert result["comparisons"]["event_significantly_better"] is True
    assert result["comparisons"]["per_seed"]["303"]["passed"] is False
    assert result["comparisons"]["all_train_seeds_significantly_better"] is False
    assert result["comparisons"]["robust_improvement"] is False
    assert len(result["aggregate"]["object_model_errors"]) == 200
    assert result["cluster_evidence"]["pooled_seed_sample_rows_used_for_inference"] is False


def test_cluster_aware_probe_must_beat_matched_action_baseline() -> None:
    per_seed = {
        str(seed): _seed_probe_result(object_model_error=0.1)
        for seed in (101, 202, 303)
    }
    for result in per_seed.values():
        result["object_action_baseline_errors"] = [0.05] * 200
        result["event_action_baseline_predictions"] = list(
            result["event_model_predictions"]
        )
    comparison = cluster_aware_probe_comparisons(
        per_seed,
        train_seeds=(101, 202, 303),
        confidence=0.95,
        bootstrap_samples=300,
        bootstrap_seed=17,
    )["comparisons"]

    assert comparison["object_vs_current_frame"]["ci_lower"] > 0.0
    assert comparison["object_significantly_better"] is False
    assert comparison["event_significantly_better"] is False
    assert comparison["robust_improvement"] is False


def _training_summary(root: Path) -> tuple[dict[str, object], Path, Path]:
    config = (root / "m1.yaml").resolve()
    checkpoint_root = (root / "checkpoints").resolve()
    seeds = (101, 202, 303)
    hashes = {str(seed): f"{offset:x}" * 64 for offset, seed in enumerate(seeds, 1)}
    strict = {str(seed): {"passed": True} for seed in seeds}
    reports = [
        {
            "variant": "state_vision_future",
            "train_seed": seed,
            "checkpoint": str(
                checkpoint_root / "state_vision_future" / f"seed_{seed}"
            ),
            "checkpoint_tree_sha256": hashes[str(seed)],
            "strict_reload": {"passed": True},
        }
        for seed in seeds
    ]
    return (
        {
            "format_version": "wam.multimodal.m1.training/1",
            "formal_protocol": True,
            "passed": True,
            "config": str(config),
            "config_sha256": "c" * 64,
            "manifest_sha256": "d" * 64,
            "checkpoint_root": str(checkpoint_root),
            "train_seeds": list(seeds),
            "variants": ["state_vision_future"],
            "checkpoint_sha256": {"state_vision_future": hashes},
            "strict_reload": {"state_vision_future": strict},
            "reports": reports,
        },
        config,
        checkpoint_root,
    )


def test_training_summary_binding_rejects_stale_primary_checkpoint_hash(
    tmp_path: Path,
) -> None:
    summary, config, checkpoint_root = _training_summary(tmp_path)
    bindings = _validate_training_summary_binding(
        summary,
        project_root=tmp_path.resolve(),
        config_file=config,
        config_sha256="c" * 64,
        manifest_sha256="d" * 64,
        checkpoint_root=checkpoint_root,
        train_seeds=(101, 202, 303),
        formal_protocol=True,
    )
    assert set(bindings) == {"101", "202", "303"}

    stale = copy.deepcopy(summary)
    stale["checkpoint_sha256"]["state_vision_future"]["202"] = "e" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_training_summary_binding(
            stale,
            project_root=tmp_path.resolve(),
            config_file=config,
            config_sha256="c" * 64,
            manifest_sha256="d" * 64,
            checkpoint_root=checkpoint_root,
            train_seeds=(101, 202, 303),
            formal_protocol=True,
        )
