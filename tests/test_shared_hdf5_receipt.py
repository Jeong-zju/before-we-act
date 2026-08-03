from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from train.shared_hdf5_receipt import (
    EXPECTED_TASKS,
    create_shared_hdf5_receipt,
    file_sha256,
    validate_shared_hdf5_receipt,
)


def _dataset_and_proof(root: Path) -> tuple[list[Path], Path, str]:
    manifests: list[Path] = []
    proof_rows: list[dict[str, str]] = []
    for task_index, task_id in enumerate(EXPECTED_TASKS):
        task_root = root / "datasets" / task_id
        hdf5_root = task_root / "hdf5"
        hdf5_root.mkdir(parents=True)
        episodes = []
        for episode_index in range(150):
            payload = f"{task_id}:{episode_index}".encode()
            episode = hdf5_root / f"episode_{episode_index:06d}.hdf5"
            episode.write_bytes(payload)
            os.utime(episode, ns=(1_700_000_000_000_000_000,) * 2)
            episodes.append(
                {
                    "episode_index": episode_index,
                    "hdf5_path": f"hdf5/{episode.name}",
                    "hdf5_sha256": hashlib.sha256(payload).hexdigest(),
                    "hdf5_size_bytes": len(payload),
                }
            )
        manifest = task_root / "training_manifest.json"
        manifest.write_text(
            json.dumps({"task": {"id": task_id}, "episodes": episodes}) + "\n"
        )
        manifests.append(manifest)
        proof_rows.append(
            {
                "task_id": task_id,
                "path": str(manifest),
                "sha256": file_sha256(manifest),
            }
        )
    proof = root / "accepted_policy.pt"
    torch.save(
        {
            "format_version": "wam.robofactory.s3_r6.world_action_flow.checkpoint/1",
            "method": {
                "micro_round": "R6L",
                "candidate_id": "P1",
                "model_kind": "s3_r6l_protected_local_gated",
            },
            "data": {"manifests": proof_rows},
            "source": {"git_commit": "a" * 40},
        },
        proof,
    )
    return manifests, proof, file_sha256(proof)


def test_shared_hdf5_receipt_reuses_proof_and_fails_on_stat_change(
    tmp_path: Path,
) -> None:
    manifests, proof, proof_sha256 = _dataset_and_proof(tmp_path)
    receipt = tmp_path / "receipt.json"
    payload = create_shared_hdf5_receipt(
        manifests,
        proof_checkpoint=proof,
        expected_proof_sha256=proof_sha256,
        output=receipt,
    )
    assert len(payload["files"]) == 750
    receipt_sha256 = file_sha256(receipt)
    validated = validate_shared_hdf5_receipt(
        receipt,
        manifests,
        expected_proof_sha256=proof_sha256,
        expected_receipt_sha256=receipt_sha256,
    )
    assert validated["proof"]["sha256"] == proof_sha256

    changed = Path(payload["files"][0]["path"])
    changed.write_bytes(changed.read_bytes())
    with pytest.raises(ValueError, match="HDF5 identity changed"):
        validate_shared_hdf5_receipt(
            receipt,
            manifests,
            expected_proof_sha256=proof_sha256,
            expected_receipt_sha256=receipt_sha256,
        )
