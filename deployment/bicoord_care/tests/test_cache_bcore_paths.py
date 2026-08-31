from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from deployment.bicoord_care import cache_bcore
from deployment.bicoord_care.evaluate_b0h import _checkpoint_from_dependencies
from deployment.bicoord_care.evaluate_bcore import _checkpoint as bcore_checkpoint
from deployment.bicoord_care.config import TASKS, TASK_TEXT
from deployment.bicoord_care.data import BiCoordEpisode
from deployment.bicoord_care.preprocessing import (
    DINO_NORMALIZATION_ID,
    IMAGE_PREPROCESS_ID,
)
from deployment.bicoord_care.stage_common import RESULT_SCHEMA, sha256_file


CONFIG_SHA = "a" * 64


def _stage_result(path: Path, stage: str, artifacts: list[tuple[Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"path": str(source.resolve()), "sha256": sha256_file(source), "kind": kind}
        for source, kind in artifacts
    ]
    path.write_text(
        json.dumps(
            {
                "schema": RESULT_SCHEMA,
                "stage": stage,
                "status": "PASSED",
                "benchmark_adapter": "BiCoord",
                "config_sha256": CONFIG_SHA,
                "artifacts": rows,
            }
        )
    )


def _run_layout(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    audit = run / "artifacts" / "dataset_audit"
    dino = run / "artifacts" / "dino_cache"
    formal = run / "artifacts" / "b0h_formal"
    smoke = run / "artifacts" / "b0h_smoke_train"
    for directory in (audit, dino, formal, smoke):
        directory.mkdir(parents=True)
    (audit / "normalization.json").write_text("{}")
    (dino / "cache_receipt.json").write_text("{}")
    (formal / "checkpoint_latest.pt").write_bytes(b"formal")
    (smoke / "checkpoint_latest.pt").write_bytes(b"smoke")
    results = run / "stage_results"
    _stage_result(
        results / "dataset_audit.json",
        "dataset_audit",
        [(audit / "normalization.json", "normalization")],
    )
    _stage_result(
        results / "dino_cache.json",
        "dino_cache",
        [(dino / "cache_receipt.json", "dino_cache")],
    )
    _stage_result(
        results / "b0h_formal.json",
        "b0h_formal",
        [(formal / "checkpoint_latest.pt", "checkpoint")],
    )
    _stage_result(
        results / "b0h_smoke_train.json",
        "b0h_smoke_train",
        [(smoke / "checkpoint_latest.pt", "checkpoint")],
    )
    return run


def test_generic_arguments_resolve_artifacts_and_prefer_formal_b0h(tmp_path: Path) -> None:
    run = _run_layout(tmp_path)
    args = cache_bcore._arguments(
        [
            "cache-all",
            "--run",
            str(run),
            "--dataset",
            str(tmp_path / "dataset"),
            "--config-sha256",
            CONFIG_SHA,
        ]
    )
    assert args.output == run / "artifacts" / "bcore_cache"
    assert args.normalization == run / "artifacts" / "dataset_audit" / "normalization.json"
    assert args.visual_cache == run / "artifacts" / "dino_cache"
    assert args.b0h_checkpoint == run / "artifacts" / "b0h_formal" / "checkpoint_latest.pt"


def test_generic_smoke_arguments_resolve_only_smoke_b0h(tmp_path: Path) -> None:
    run = _run_layout(tmp_path)
    args = cache_bcore._arguments(
        [
            "cache-all",
            "--run",
            str(run),
            "--dataset",
            str(tmp_path / "dataset"),
            "--config-sha256",
            CONFIG_SHA,
            "--smoke",
        ]
    )
    assert args.b0h_checkpoint == (
        run / "artifacts" / "b0h_smoke_train" / "checkpoint_latest.pt"
    )
    assert args.output == run / "artifacts" / "bcore_smoke_cache"
    assert args.stage_name == "bcore_smoke_cache"


def test_smoke_result_path_cannot_implicitly_change_cache_mode(tmp_path: Path) -> None:
    run = _run_layout(tmp_path)
    with pytest.raises(ValueError, match="require explicit --smoke"):
        cache_bcore._arguments(
            [
                "cache-all",
                "--run",
                str(run),
                "--dataset",
                str(tmp_path / "dataset"),
                "--config-sha256",
                CONFIG_SHA,
                "--result",
                str(run / "workers" / "bcore_smoke_cache" / "rank_0.json"),
            ]
        )


def test_b0h_probe_excludes_smoke_checkpoint_and_fails_closed(tmp_path: Path) -> None:
    run = _run_layout(tmp_path)
    args = argparse.Namespace(
        run=run,
        operation="probe",
        config_sha256=CONFIG_SHA,
    )
    formal = run / "artifacts" / "b0h_formal" / "checkpoint_latest.pt"
    assert _checkpoint_from_dependencies(args) == formal

    # Corrupting the formal artifact cannot cause a fallback to the still
    # valid smoke result.
    formal.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="b0h_formal"):
        _checkpoint_from_dependencies(args)


def test_b0h_smoke_closed_loop_uses_only_smoke_checkpoint(tmp_path: Path) -> None:
    run = _run_layout(tmp_path)
    args = argparse.Namespace(
        run=run,
        operation="smoke-closed-loop",
        config_sha256=CONFIG_SHA,
    )
    assert _checkpoint_from_dependencies(args) == (
        run / "artifacts" / "b0h_smoke_train" / "checkpoint_latest.pt"
    )


def test_bcore_formal_evaluator_never_falls_back_to_smoke_or_unhashed_field(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    smoke = run / "artifacts" / "bcore_smoke_train" / "deployment_checkpoint.pt"
    formal = run / "artifacts" / "bcore_select" / "deployment_checkpoint.pt"
    smoke.parent.mkdir(parents=True)
    formal.parent.mkdir(parents=True)
    smoke.write_bytes(b"smoke")
    formal.write_bytes(b"formal")
    results = run / "stage_results"
    _stage_result(
        results / "bcore_smoke_train.json",
        "bcore_smoke_train",
        [(smoke, "deployment_checkpoint")],
    )
    _stage_result(
        results / "bcore_select.json",
        "bcore_select",
        [(formal, "deployment_checkpoint")],
    )
    formal_result = json.loads((results / "bcore_select.json").read_text())
    # This field is deliberately un-hashed and points at the still-valid smoke
    # model.  It must not become a fallback candidate.
    formal_result["checkpoint"] = str(smoke.resolve())
    (results / "bcore_select.json").write_text(json.dumps(formal_result))
    args = argparse.Namespace(
        run=run, operation="validation20", config_sha256=CONFIG_SHA
    )
    assert bcore_checkpoint(args) == formal
    formal.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="bcore_select"):
        bcore_checkpoint(args)


def _episodes() -> list[BiCoordEpisode]:
    return [
        BiCoordEpisode(
            path=f"/{task}/episode0.hdf5",
            task=task,
            task_text=TASK_TEXT[task],
            episode_id=0,
            length=2,
            hdf5_sha256=hashlib.sha256(task.encode()).hexdigest(),
        )
        for task in TASKS
    ]


def _multi_episodes(tmp_path: Path) -> list[BiCoordEpisode]:
    rows: list[BiCoordEpisode] = []
    # Shuffled input plus an episode-id tie proves that both ordering keys are
    # used.  ``a`` must win over ``b`` for episode 2 in every task.
    for task in reversed(TASKS):
        for episode_id, suffix in ((7, "z"), (2, "b"), (2, "a")):
            path = tmp_path / task / f"episode_{episode_id}_{suffix}.hdf5"
            identity = hashlib.sha256(str(path).encode()).hexdigest()
            rows.append(
                BiCoordEpisode(
                    path=str(path),
                    task=task,
                    task_text=TASK_TEXT[task],
                    episode_id=episode_id,
                    length=2,
                    hdf5_sha256=identity,
                )
            )
    return rows


def test_smoke_selection_is_exactly_minimum_source_per_task(tmp_path: Path) -> None:
    discovered = _multi_episodes(tmp_path)
    selected = cache_bcore._select_cache_episodes(discovered, formal=False)
    assert len(selected) == len(TASKS) == 18
    assert tuple(episode.task for episode in selected) == TASKS
    assert all(episode.episode_id == 2 for episode in selected)
    assert all(Path(episode.path).name == "episode_2_a.hdf5" for episode in selected)

    formal = cache_bcore._select_cache_episodes(discovered, formal=True)
    assert len(formal) == len(discovered)
    assert formal[:3] == sorted(
        [episode for episode in discovered if episode.task == TASKS[0]],
        key=lambda episode: (episode.episode_id, episode.path),
    )


def _write_episode_cache(
    output: Path, episode: BiCoordEpisode, *, b0h_sha: str
) -> None:
    decoded, base, marker = cache_bcore._cached_paths(output, episode)
    decoded.parent.mkdir(parents=True, exist_ok=True)
    np.save(decoded, np.zeros((1, 2, 100, 384), dtype=np.float16))
    np.save(base, np.zeros((1, 2, 100, 7), dtype=np.float16))
    marker.write_text(
        json.dumps(
            {
                "status": "PASSED",
                "source_identity": episode.source_identity,
                "decisions": 1,
                "b0h_checkpoint_sha256": b0h_sha,
            }
        )
    )


def test_bcore_aggregate_receipt_and_complete_worker_contract(tmp_path: Path) -> None:
    episodes = _episodes()
    output = tmp_path / "bcore"
    output.mkdir()
    for episode in episodes:
        decoded, base, marker = cache_bcore._cached_paths(output, episode)
        decoded.parent.mkdir(parents=True, exist_ok=True)
        np.save(decoded, np.zeros((1, 2, 100, 384), dtype=np.float16))
        np.save(base, np.zeros((1, 2, 100, 7), dtype=np.float16))
        marker.write_text(
            json.dumps(
                {
                    "status": "PASSED",
                    "source_identity": episode.source_identity,
                    "decisions": 1,
                    "b0h_checkpoint_sha256": "b" * 64,
                }
            )
        )
    normalization = tmp_path / "normalization.json"
    normalization.write_text("normalization")
    visual = tmp_path / "visual"
    visual.mkdir()
    visual_receipt = visual / "cache_receipt.json"
    visual_receipt.write_text("visual")
    for rank in range(2):
        assigned = episodes[rank::2]
        (output / f"rank_{rank:02d}_receipt.json").write_text(
            json.dumps(
                {
                    "schema": cache_bcore.BCORE_CACHE_SHARD_SCHEMA,
                    "status": "PASSED",
                    "rank": rank,
                    "world_size": 2,
                    "formal": True,
                    "config_sha256": CONFIG_SHA,
                    "dataset_revision": cache_bcore.DATASET_REVISION,
                    "episodes": len(assigned),
                    "samples": len(assigned) * 2,
                    "task_tokens": {
                        episode.task: [0.0] * 384 for episode in assigned
                    },
                    "b0h_checkpoint_sha256": "b" * 64,
                    "normalization_receipt_sha256": sha256_file(normalization),
                    "visual_cache_receipt_sha256": sha256_file(visual_receipt),
                }
            )
        )
    aggregate = cache_bcore._try_finalize(
        output=output,
        episodes=episodes,
        world=2,
        b0h_sha="b" * 64,
        formal=True,
        visual_cache=visual,
        normalization=normalization,
        config_sha256=CONFIG_SHA,
    )
    assert aggregate is not None
    assert aggregate["status"] == "PASSED"
    assert aggregate["cache_complete"] is True
    assert aggregate["dataset_revision"] == cache_bcore.DATASET_REVISION
    assert aggregate["config_sha256"] == CONFIG_SHA
    assert set(aggregate["rank_receipt_sha256"]) == {
        "rank_00_receipt.json",
        "rank_01_receipt.json",
    }


def test_smoke_finalize_uses_same_source_slice_as_build_and_workers(
    tmp_path: Path,
) -> None:
    discovered = _multi_episodes(tmp_path / "dataset")
    selected = cache_bcore._select_cache_episodes(discovered, formal=False)
    output = tmp_path / "bcore_smoke_cache"
    output.mkdir()
    b0h_sha = "c" * 64
    for episode in selected:
        _write_episode_cache(output, episode, b0h_sha=b0h_sha)

    normalization = tmp_path / "normalization.json"
    normalization.write_text("normalization")
    visual = tmp_path / "visual"
    visual.mkdir()
    visual_receipt = visual / "cache_receipt.json"
    visual_receipt.write_text("visual")
    sources = [cache_bcore._episode_source_row(episode) for episode in selected]
    for rank in range(4):
        assigned = selected[rank::4]
        assigned_sources = sources[rank::4]
        (output / f"rank_{rank:02d}_receipt.json").write_text(
            json.dumps(
                {
                    "schema": cache_bcore.BCORE_CACHE_SHARD_SCHEMA,
                    "status": "PASSED",
                    "rank": rank,
                    "world_size": 4,
                    "formal": False,
                    "config_sha256": CONFIG_SHA,
                    "dataset_revision": cache_bcore.DATASET_REVISION,
                    "episodes": len(assigned),
                    "samples": len(assigned) * 2,
                    "assigned_episode_source_identities": [
                        row["source_identity"] for row in assigned_sources
                    ],
                    "assigned_episode_sources": assigned_sources,
                    "episode_selection": cache_bcore.SMOKE_EPISODE_SELECTION,
                    "task_tokens": {
                        episode.task: [0.0] * 384 for episode in assigned
                    },
                    "b0h_checkpoint_sha256": b0h_sha,
                    "normalization_receipt_sha256": sha256_file(normalization),
                    "visual_cache_receipt_sha256": sha256_file(visual_receipt),
                }
            )
        )

    # Pass the full 54-row discovery result.  Finalization must independently
    # derive the same exact 18-row subset used by workers/build.
    aggregate = cache_bcore._try_finalize(
        output=output,
        episodes=discovered,
        world=4,
        b0h_sha=b0h_sha,
        formal=False,
        visual_cache=visual,
        normalization=normalization,
        config_sha256=CONFIG_SHA,
    )
    assert aggregate is not None
    assert aggregate["episodes"] == 18
    assert aggregate["episodes_per_task"] == {task: 1 for task in TASKS}
    assert aggregate["episode_selection"] == cache_bcore.SMOKE_EPISODE_SELECTION
    assert aggregate["episode_sources"] == sources
    assert aggregate["episode_source_identities"] == [
        episode.source_identity for episode in selected
    ]


def test_incomplete_worker_result_cannot_publish_passed(tmp_path: Path) -> None:
    output = tmp_path / "cache"
    output.mkdir()
    incomplete = {
        "schema": cache_bcore.BCORE_CACHE_SCHEMA,
        "status": "PASSED",
        "cache_complete": False,
        "world_size": 4,
        "config_sha256": CONFIG_SHA,
    }
    with pytest.raises(ValueError, match="aggregate completion"):
        cache_bcore._complete_worker_result(
            stage="bcore_smoke_cache",
            config_sha256=CONFIG_SHA,
            output=output,
            rank=0,
            world=4,
            receipt=incomplete,
        )
