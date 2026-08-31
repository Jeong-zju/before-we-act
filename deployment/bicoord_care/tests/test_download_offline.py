from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from deployment.bicoord_care import download
from deployment.bicoord_care.config import DATASET_REVISION, TASKS, TOTAL_EPISODES


def _offline_tree(root: Path) -> None:
    files: dict[str, dict[str, object]] = {}
    metadata_root = root / ".cache" / "huggingface" / "download"
    for task in TASKS:
        for episode in range(100):
            relative = f"{task}/demo_clean/data/episode{episode}.hdf5"
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
            object_id = f"{TASKS.index(task):02d}{episode:04d}".ljust(40, "a")
            files[relative] = {"size": 0, "blob_id": object_id}
            sidecar = metadata_root / f"{relative}.metadata"
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(f"{DATASET_REVISION}\n{object_id}\n0\n")
    tree = root / ".cache" / "huggingface" / "trees" / f"{DATASET_REVISION}.json"
    tree.parent.mkdir(parents=True, exist_ok=True)
    tree.write_text(json.dumps({"format_version": 1, "files": files}))


def test_complete_local_snapshot_is_verified_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _offline_tree(dataset)
    repo = tmp_path / "repo"
    benchmark = tmp_path / "benchmark"
    repo.mkdir()
    benchmark.mkdir()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(
        download,
        "_token",
        lambda: (_ for _ in ()).throw(AssertionError("offline path looked up token")),
    )
    monkeypatch.setattr(download, "EXPECTED_SNAPSHOT_FILES", TOTAL_EPISODES)
    args = argparse.Namespace(
        repo=repo,
        benchmark_repo=benchmark,
        dataset=dataset,
        run=tmp_path / "run",
        dino_model=tmp_path / "dino",
        result=tmp_path / "result.json",
        config_sha256="a" * 64,
        auto_resume=True,
        operation="download",
    )
    result = download.run(args)
    assert result["status"] == "PASSED"
    assert result["episodes"] == TOTAL_EPISODES
    assert result["token_source"] == "local_snapshot_verified_without_token"
    assert result["reused_existing_snapshot"] is True
    receipt = json.loads((dataset / "dataset_receipt.json").read_text())
    assert receipt["snapshot_files"] == TOTAL_EPISODES
    assert receipt["metadata_sidecars_verified"] == TOTAL_EPISODES
    assert receipt["resolved_revision"] == DATASET_REVISION
    assert (dataset / ".bicoord_download_intent.json").is_file()


def test_verified_receipt_and_intent_are_reused_byte_for_byte_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _offline_tree(dataset)
    repo = tmp_path / "repo"
    benchmark = tmp_path / "benchmark"
    repo.mkdir()
    benchmark.mkdir()
    monkeypatch.setattr(download, "EXPECTED_SNAPSHOT_FILES", TOTAL_EPISODES)
    args = argparse.Namespace(
        repo=repo,
        benchmark_repo=benchmark,
        dataset=dataset,
        run=tmp_path / "run",
        dino_model=tmp_path / "dino",
        result=tmp_path / "result.json",
        config_sha256="a" * 64,
        auto_resume=True,
        operation="download",
    )
    first = download.run(args)
    assert first["reused_existing_snapshot"] is True
    receipt_path = dataset / "dataset_receipt.json"
    intent_path = dataset / ".bicoord_download_intent.json"
    receipt_bytes = receipt_path.read_bytes()
    intent_bytes = intent_path.read_bytes()
    monkeypatch.setattr(
        download,
        "_token",
        lambda: (_ for _ in ()).throw(AssertionError("offline path looked up token")),
    )
    args.result = tmp_path / "result-second.json"
    second = download.run(args)
    assert second["reused_existing_snapshot"] is True
    assert receipt_path.read_bytes() == receipt_bytes
    assert intent_path.read_bytes() == intent_bytes


def test_passed_receipt_with_changed_tree_provenance_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _offline_tree(dataset)
    repo = tmp_path / "repo"
    benchmark = tmp_path / "benchmark"
    repo.mkdir()
    benchmark.mkdir()
    monkeypatch.setattr(download, "EXPECTED_SNAPSHOT_FILES", TOTAL_EPISODES)
    args = argparse.Namespace(
        repo=repo,
        benchmark_repo=benchmark,
        dataset=dataset,
        run=tmp_path / "run",
        dino_model=tmp_path / "dino",
        result=tmp_path / "result.json",
        config_sha256="a" * 64,
        auto_resume=True,
        operation="download",
    )
    download.run(args)
    receipt_path = dataset / "dataset_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["tree_manifest_sha256"] = "f" * 64
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    tampered = receipt_path.read_bytes()
    args.result = tmp_path / "second-result.json"
    with pytest.raises(RuntimeError, match="provenance differs at tree_manifest_sha256"):
        download.run(args)
    assert receipt_path.read_bytes() == tampered
    assert not args.result.exists()


def test_failed_receipt_is_never_promoted_to_passed(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _offline_tree(dataset)
    (tmp_path / "repo").mkdir()
    (tmp_path / "benchmark").mkdir()
    (dataset / "dataset_receipt.json").write_text(
        json.dumps(
            {
                "schema": "before-we-act.bicoord-dataset-download/1",
                "status": "FAILED",
                "dataset_repo_id": "GradiusTwinbee/BiCoord",
                "dataset_revision": DATASET_REVISION,
            }
        )
    )
    with pytest.raises(RuntimeError, match="not PASSED"):
        download.run(
            argparse.Namespace(
                repo=tmp_path / "repo",
                benchmark_repo=tmp_path / "benchmark",
                dataset=dataset,
                run=tmp_path / "run",
                dino_model=tmp_path / "dino",
                result=tmp_path / "result.json",
                config_sha256="a" * 64,
                auto_resume=True,
                operation="download",
            )
        )
    assert json.loads((dataset / "dataset_receipt.json").read_text())["status"] == "FAILED"
    assert not (dataset / ".bicoord_download_intent.json").exists()


@pytest.mark.parametrize(
    "value",
    [
        {"dataset_repo_id": "GradiusTwinbee/BiCoord", "dataset_revision": DATASET_REVISION},
        {
            "schema": "wrong",
            "dataset_repo_id": "GradiusTwinbee/BiCoord",
            "dataset_revision": DATASET_REVISION,
            "state": "DOWNLOADING",
        },
        {
            "schema": download.DOWNLOAD_INTENT_SCHEMA,
            "dataset_repo_id": "GradiusTwinbee/BiCoord",
            "dataset_revision": DATASET_REVISION,
            "state": "VERIFIED",
            "tree_manifest_sha256": "0" * 64,
            "snapshot_files": 0,
        },
        {
            "schema": download.DOWNLOAD_INTENT_SCHEMA,
            "dataset_repo_id": "GradiusTwinbee/BiCoord",
            "dataset_revision": DATASET_REVISION,
            "state": "ROLLED_BACK",
        },
    ],
)
def test_download_intent_state_machine_rejects_missing_or_invalid_states(
    tmp_path: Path, value: dict[str, object]
) -> None:
    path = tmp_path / "intent.json"
    path.write_text(json.dumps(value))
    assert not download._valid_existing_intent(path)


def test_verified_intent_cannot_be_downgraded_or_rebound(tmp_path: Path) -> None:
    path = tmp_path / "intent.json"
    evidence = {
        "tree_manifest_sha256": "a" * 64,
        "snapshot_files": 9_057,
    }
    download.atomic_json(path, download._intent(evidence))
    with pytest.raises(RuntimeError, match="differs"):
        download._finalize_intent(
            path,
            {"tree_manifest_sha256": "b" * 64, "snapshot_files": 9_057},
        )
    assert json.loads(path.read_text())["state"] == "VERIFIED"


def test_snapshot_manifest_rejects_symlinked_parent(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "episode0.hdf5").write_bytes(b"")
    (dataset / "task").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic"):
        download._safe_snapshot_path(dataset, "task/episode0.hdf5")


def test_download_tree_rejects_noncanonical_object_identity(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _offline_tree(dataset)
    tree_path = dataset / ".cache" / "huggingface" / "trees" / f"{DATASET_REVISION}.json"
    tree = json.loads(tree_path.read_text())
    first = next(iter(tree["files"].values()))
    first["blob_id"] = "not-a-git-object"
    tree_path.write_text(json.dumps(tree))
    with pytest.raises(RuntimeError, match="object identity"):
        download._verify_local_snapshot(dataset, expected_files=TOTAL_EPISODES)


def test_offline_snapshot_rejects_metadata_revision_drift(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _offline_tree(dataset)
    sidecar = next(
        (dataset / ".cache" / "huggingface" / "download").rglob("*.metadata")
    )
    lines = sidecar.read_text().splitlines()
    sidecar.write_text("0" * 40 + "\n" + "\n".join(lines[1:]) + "\n")
    with pytest.raises(RuntimeError, match="metadata provenance differs"):
        download._verify_local_snapshot(dataset, expected_files=TOTAL_EPISODES)
