from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from deployment.bicoord_care.cache_dino import (
    _dino_source_manifest,
    _valid_cache,
)
from deployment.bicoord_care.config import (
    DINO_HIDDEN_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    TASK_TEXT,
    TASKS,
)
from deployment.bicoord_care.data import BiCoordEpisode
from deployment.bicoord_care.preprocessing import (
    DINO_NORMALIZATION_ID,
    IMAGE_PREPROCESS_ID,
)
from deployment.bicoord_care.stage_common import (
    artifact,
    publish_result,
    read_json,
)
from deployment.bicoord_care.train_b0h import (
    _arguments as b0h_arguments,
    _smoke_episode_subset,
    _validate_cache_receipt,
    _publish_supervisor_result,
)
from deployment.bicoord_care.evaluate_b0h import _progress_paths as b0h_progress_paths
from deployment.bicoord_care.evaluate_bcore import _progress_paths as bcore_progress_paths


def _episode() -> BiCoordEpisode:
    return BiCoordEpisode(
        path="/fixture/episode0.hdf5",
        task="cook",
        task_text=TASK_TEXT["cook"],
        episode_id=0,
        length=3,
        hdf5_sha256="a" * 64,
    )


def _write_cache(path: Path, episode: BiCoordEpisode, model_sha256: str) -> None:
    values = np.zeros((episode.length, DINO_HIDDEN_SIZE), dtype=np.float16)
    np.savez(
        path,
        source_identity=np.asarray(episode.source_identity),
        dino_model=np.asarray("/model"),
        dino_source_sha256=np.asarray(model_sha256),
        image_height=np.asarray(IMAGE_HEIGHT),
        image_width=np.asarray(IMAGE_WIDTH),
        image_preprocess_id=np.asarray(IMAGE_PREPROCESS_ID),
        dino_normalization_id=np.asarray(DINO_NORMALIZATION_ID),
        strict_dino_contract=np.asarray(True),
        view_head=values,
        view_wrist_0=values,
        view_wrist_1=values,
    )


def test_local_dino_source_manifest_hashes_weight_artifact(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "preprocessor_config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"upstream-weights")
    first = _dino_source_manifest(tmp_path)
    second = _dino_source_manifest(tmp_path)
    assert first == second
    assert len(first["sha256"]) == 64
    assert {row["name"] for row in first["files"]} == {
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    }


def test_cache_validation_rejects_model_or_contract_drift(tmp_path: Path) -> None:
    episode = _episode()
    path = tmp_path / "cache.npz"
    _write_cache(path, episode, "b" * 64)
    assert _valid_cache(path, episode, model_sha256="b" * 64)
    assert not _valid_cache(path, episode, model_sha256="c" * 64)
    with np.load(path, allow_pickle=False) as source:
        values = {name: np.asarray(source[name]) for name in source.files}
    values["strict_dino_contract"] = np.asarray(False)
    np.savez(path, **values)
    assert not _valid_cache(path, episode, model_sha256="b" * 64)


def test_stage_result_identity_cannot_be_overridden(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"status": "real"}))
    args = argparse.Namespace(
        result=tmp_path / "result.json", config_sha256="d" * 64
    )
    with pytest.raises(ValueError):
        publish_result(
            args,
            stage="dataset_audit",
            artifacts=[artifact(evidence)],
            status="SMOKE",
        )
    assert not args.result.exists()
    publish_result(
        args,
        stage="dataset_audit",
        artifacts=[artifact(evidence)],
        status="PASSED",
    )
    assert read_json(args.result)["status"] == "PASSED"


def test_b0h_supervisor_cli_maps_to_existing_training_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    benchmark = tmp_path / "benchmark"
    dataset = tmp_path / "dataset"
    for path in (repo, benchmark, dataset):
        path.mkdir()
    run = tmp_path / "run"
    args = b0h_arguments(
        [
            "smoke-train",
            "--repo", str(repo),
            "--benchmark-repo", str(benchmark),
            "--dataset", str(dataset),
            "--run", str(run),
            "--dino-model", str(tmp_path / "dino"),
            "--result", str(tmp_path / "candidate.json"),
            "--config-sha256", "a" * 64,
            "--updates", "5",
            "--global-batch", "48",
            "--auto-resume",
        ]
    )
    assert args.stage == "smoke" and args.updates == 5
    assert args.data_root == dataset.resolve()
    assert args.normalization == (run / "artifacts/dataset_audit/normalization.json").resolve()
    assert args.visual_cache == (run / "artifacts/dino_cache").resolve()
    assert args.output == (run / "artifacts/b0h_smoke_train").resolve()
    assert args.auto_resume is True


def test_b0h_supervisor_cli_rejects_batch_drift(tmp_path: Path) -> None:
    for name in ("repo", "benchmark", "dataset"):
        (tmp_path / name).mkdir()
    with pytest.raises(ValueError, match="global batch"):
        b0h_arguments(
            [
                "formal-train",
                "--repo", str(tmp_path / "repo"),
                "--benchmark-repo", str(tmp_path / "benchmark"),
                "--dataset", str(tmp_path / "dataset"),
                "--run", str(tmp_path / "run"),
                "--result", str(tmp_path / "result.json"),
                "--config-sha256", "b" * 64,
                "--global-batch", "16",
            ]
        )


def test_b0h_publishes_hashed_supervisor_result(tmp_path: Path) -> None:
    output = tmp_path / "b0h"
    output.mkdir()
    checkpoint = output / "checkpoint_latest.pt"
    checkpoint.write_bytes(b"real-checkpoint-bytes")
    receipt = output / "checkpoint_receipt.json"
    receipt.write_text(json.dumps({"status": "PASSED_SMOKE"}))
    (output / "config.json").write_text(json.dumps({"stage": "smoke"}))
    (output / "status.json").write_text(json.dumps({"status": "PASSED_SMOKE"}))
    result = tmp_path / "result.json"
    args = argparse.Namespace(
        result=result,
        config_sha256="c" * 64,
        stage="smoke",
        output=output,
        updates=5,
    )
    _publish_supervisor_result(
        args, checkpoint=checkpoint, receipt=receipt, rank=0
    )
    value = read_json(result)
    assert value["stage"] == "b0h_smoke_train"
    assert value["model_contract"]["d_model"] == 384
    assert value["model_contract"]["gripper_encoding"] == "continuous_absolute_drive_target"
    checkpoint_rows = [row for row in value["artifacts"] if row.get("kind") == "checkpoint"]
    assert len(checkpoint_rows) == 1
    assert checkpoint_rows[0]["sha256"] == value["checkpoint_sha256"]


def test_b0h_smoke_uses_18_minimum_sources_even_with_formal_dino_receipt() -> None:
    episodes: list[BiCoordEpisode] = []
    for task_index, task in enumerate(TASK_TEXT):
        for episode_id in (9, 1):
            episodes.append(
                BiCoordEpisode(
                    path=f"/dataset/{task}/episode{episode_id}.hdf5",
                    task=task,
                    task_text=TASK_TEXT[task],
                    episode_id=episode_id,
                    length=3,
                    hdf5_sha256=(f"{task_index:02x}" * 32)[:64]
                    if episode_id == 1
                    else (f"{task_index + 18:02x}" * 32)[:64],
                )
            )
    # The visual cache is formal/full (1,800 rows), but smoke selection must
    # still be exactly one minimum source per task.
    selected = [
        next(episode for episode in episodes if episode.task == task and episode.episode_id == 1)
        for task in TASK_TEXT
    ]
    receipt = {
        "schema": "before-we-act.bicoord.dino-cache/1",
        "status": "PASSED",
        "episodes": 1_800,
        "episodes_per_task": {task: 100 for task in TASK_TEXT},
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "feature_width": DINO_HIDDEN_SIZE,
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "files": [
            {
                "task": episode.task,
                "episode_id": episode.episode_id,
                "source_identity": episode.source_identity,
            }
            for episode in selected
        ],
    }
    observed = _smoke_episode_subset(list(reversed(episodes)), receipt)
    assert len(observed) == 18
    assert [episode.task for episode in observed] == list(TASK_TEXT)
    assert [episode.episode_id for episode in observed] == [1] * 18

    # A receipt naming the non-minimum source is not silently accepted.
    bad = dict(receipt)
    bad["files"] = [
        {
            "task": episode.task,
            "episode_id": 9,
            "source_identity": next(
                candidate.source_identity
                for candidate in episodes
                if candidate.task == episode.task and candidate.episode_id == 9
            ),
        }
        for episode in selected
    ]
    with pytest.raises(ValueError, match="exactly once"):
        _smoke_episode_subset(episodes, bad)


def test_closed_loop_progress_roots_are_stage_isolated(tmp_path: Path) -> None:
    b0h_smoke, b0h_smoke_receipt = b0h_progress_paths(
        tmp_path, "smoke-closed-loop", "cook"
    )
    bcore_smoke, bcore_smoke_receipt = bcore_progress_paths(
        tmp_path, "smoke-closed-loop", "cook"
    )
    b0h_probe, _ = b0h_progress_paths(tmp_path, "probe", "cook")
    bcore_formal, _ = bcore_progress_paths(tmp_path, "validation20", "cook")
    assert b0h_smoke != bcore_smoke
    assert b0h_smoke_receipt.parent.name == "b0h_smoke_closed_loop"
    assert bcore_smoke_receipt.parent.name == "bcore_smoke_closed_loop"
    assert b0h_probe.parent.name == "b0h_probe"
    assert bcore_formal.parent.name == "bcore_validation20"


def test_formal_dino_receipt_drives_18_smoke_episodes_but_1800_formal(
    tmp_path: Path,
) -> None:
    episodes: list[BiCoordEpisode] = []
    rows: list[dict[str, object]] = []
    cache_root = tmp_path / "dino_cache"
    for task in TASKS:
        for episode_id in range(100):
            identity = hashlib.sha256(f"{task}:{episode_id}".encode()).hexdigest()
            episode = BiCoordEpisode(
                path=str(tmp_path / "dataset" / task / f"episode{episode_id}.hdf5"),
                task=task,
                task_text=TASK_TEXT[task],
                episode_id=episode_id,
                length=3,
                hdf5_sha256=identity,
            )
            episodes.append(episode)
            cache_path = cache_root / task / f"{identity}.npz"
            if episode_id == 0:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(f"selected:{task}".encode())
                digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            else:
                # Formal selection does not rehash all 1,800 artifacts during
                # this unit test; production cache construction already does.
                digest = "f" * 64
            rows.append(
                {
                    "task": task,
                    "episode_id": episode_id,
                    "source_identity": identity,
                    "path": str(cache_path.resolve()),
                    "sha256": digest,
                }
            )
    receipt_path = cache_root / "cache_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "before-we-act.bicoord.dino-cache/1",
                "status": "PASSED",
                "episodes": 1_800,
                "episodes_per_task": {task: 100 for task in TASKS},
                "image_height": IMAGE_HEIGHT,
                "image_width": IMAGE_WIDTH,
                "feature_width": DINO_HIDDEN_SIZE,
                "image_preprocess_id": IMAGE_PREPROCESS_ID,
                "dino_normalization_id": DINO_NORMALIZATION_ID,
                "strict_dino_contract": True,
                "files": rows,
            }
        )
    )
    _, smoke = _validate_cache_receipt(
        receipt_path,
        stage="smoke",
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        episodes=list(reversed(episodes)),
    )
    assert len(smoke) == 18
    assert [episode.task for episode in smoke] == list(TASKS)
    assert all(episode.episode_id == 0 for episode in smoke)

    _, formal = _validate_cache_receipt(
        receipt_path,
        stage="formal",
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        episodes=episodes,
    )
    assert len(formal) == 1_800
    assert formal == episodes
