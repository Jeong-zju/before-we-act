#!/usr/bin/env python3
"""Download and byte-verify the four pinned MARS-Control datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .common import DATASET_REPOS, atomic_json, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/mars_control"))
    parser.add_argument("--token-file", type=Path, default=Path("/workspace/.secrets/hf_token"))
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    if not token:
        raise RuntimeError("Hugging Face token file is empty")
    api = HfApi(token=token)
    for task, (repo_id, revision) in DATASET_REPOS.items():
        output = args.data_root / task
        output.mkdir(parents=True, exist_ok=True)
        info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise RuntimeError(f"{task}: revision drift: {info.sha}")
        siblings = {item.rfilename: item for item in info.siblings}
        hdf5_names = sorted(
            name for name in siblings
            if name.startswith("motionplanning/") and name.endswith(".h5") and "/." not in name
        )
        if len(hdf5_names) != 10:
            raise RuntimeError(f"{task}: expected 10 promoted HDF5 shards, found {len(hdf5_names)}")
        rows = []
        for name in hdf5_names:
            item = siblings[name]
            local = Path(hf_hub_download(
                repo_id, name, repo_type="dataset", revision=revision,
                local_dir=str(output), token=token,
            ))
            expected_size = int(item.size or 0)
            if local.stat().st_size != expected_size:
                raise RuntimeError(f"{task}: size mismatch: {name}")
            digest = sha256(local)
            expected_hash = item.lfs.sha256 if item.lfs else None
            if expected_hash and digest != expected_hash:
                raise RuntimeError(f"{task}: sha256 mismatch: {name}")
            sidecar = name[:-3] + "json"
            if sidecar in siblings:
                hf_hub_download(repo_id, sidecar, repo_type="dataset", revision=revision,
                                local_dir=str(output), token=token)
            rows.append({"path": name, "bytes": expected_size, "sha256": digest})
        receipt = {
            "schema": "mars-control.latent-tom.dataset.v1",
            "status": "complete",
            "task": task,
            "repo_id": repo_id,
            "revision": revision,
            "formal_shards": rows,
            "formal_episodes": 150,
            "bytes_total": sum(row["bytes"] for row in rows),
            "training_policy": "all_data_no_split",
        }
        atomic_json(output / "download_receipt.json", receipt)
        print({"task": task, "shards": len(rows), "bytes": receipt["bytes_total"]}, flush=True)


if __name__ == "__main__":
    main()
