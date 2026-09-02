#!/usr/bin/env python3
"""Download and audit the pinned four-task MARS-Control corpus.

Only the ten promoted successful shards per task are downloaded.  The HF
revisions also contain failed ``.parts`` files; accepting those would silently
change the CARE training corpus.  Files are placed in the EnvID layout used by
the official CARE loaders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_download


TASKS = {
    "place_cube_in_cup": ("PlaceCubeInCup-rf", "Jeong-zju/mars-control-place-cube-in-cup-rf", "3878150bec8f4830e1a57a01a13762a10abc8d52"),
    "strike_cube_hard": ("StrikeCubeHard-rf", "Jeong-zju/mars-control-strike-cube-hard-rf", "bc7051cb0560058bf426e792871faa1ca8a4f78f"),
    "three_robots_place_shoes": ("ThreeRobotsPlaceShoes-rf", "Jeong-zju/mars-control-three-robots-place-shoes-rf", "ad231c7eff530f71f0c5302b6c03c7164bbcc896"),
    "four_robots_stack_cube": ("FourRobotsStackCube-rf", "Jeong-zju/mars-control-four-robots-stack-cube-rf", "3fa4833f5e34c3565da04af99c62d516e048fcfc"),
}
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def link_or_copy(source: Path, target: Path) -> None:
    # huggingface_hub may return a relative symlink into its blob cache.  Resolve
    # it before hard-linking so the destination cannot inherit a broken link.
    source = source.resolve(strict=True)
    if source == target.resolve(strict=False):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == source.stat().st_size and sha256(target) == sha256(source):
            return
        raise RuntimeError(f"refusing to overwrite mismatched file: {target}")
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def audit_shard(path: Path, expected_episodes: int = 15) -> int:
    with h5py.File(path, "r", swmr=True) as handle:
        names = sorted((name for name in handle if name.startswith("traj_")), key=lambda x: int(x.rsplit("_", 1)[-1]))
        if len(names) != expected_episodes:
            raise RuntimeError(f"{path}: expected {expected_episodes} trajectories, got {len(names)}")
        for name in names:
            group = handle[name]
            success = np.asarray(group["success"])
            if not len(success) or not bool(success[-1]):
                raise RuntimeError(f"{path}:{name} is not a successful trajectory")
    return len(names)


def audit_sidecar(path: Path, expected_episodes: int = 15) -> int:
    value = json.loads(path.read_text(encoding="utf-8"))
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        raise RuntimeError(f"{path}: expected {expected_episodes} sidecar episodes")
    ids = sorted(int(row["episode_id"]) for row in episodes)
    if ids != list(range(expected_episodes)):
        raise RuntimeError(f"{path}: sidecar episode IDs do not match H5 trajectories")
    for row in episodes:
        seed = row.get("episode_seed", row.get("reset_kwargs", {}).get("seed"))
        if seed is None:
            raise RuntimeError(f"{path}: missing episode seed for trajectory {row['episode_id']}")
    return len(episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/mars_control"))
    parser.add_argument("--token-file", type=Path, default=Path("/workspace/.secrets/hf_token"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    token = args.token or (args.token_file.read_text(encoding="utf-8").strip() if args.token_file.is_file() else "")
    if not token:
        raise RuntimeError("Hugging Face token is missing (HF_TOKEN or --token-file)")
    api = HfApi(token=token)
    raw_root = args.data_root / "raw"
    receipts = []
    for task, (env_id, repo, revision) in TASKS.items():
        info = api.dataset_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise RuntimeError(f"{task}: revision drift: {info.sha}")
        siblings = {item.rfilename: item for item in info.siblings}
        names = sorted(name for name in siblings if name.startswith("motionplanning/") and name.endswith(".h5") and "/." not in name)
        if len(names) != 10 or names != [f"motionplanning/{task}.shard{i:02d}.h5" for i in range(10)]:
            raise RuntimeError(f"{task}: formal shard set drift: {names}")
        task_root = raw_root / env_id / "motionplanning"
        task_root.mkdir(parents=True, exist_ok=True)
        rows = []
        with tempfile.TemporaryDirectory(prefix=f"mars-{task}-", dir=str(args.data_root)) as staging:
            for name in names:
                target = task_root / Path(name).name
                expected_size = int(siblings[name].size or 0)
                if target.is_file() and not target.is_symlink() and target.stat().st_size == expected_size:
                    local = target
                else:
                    local = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision, local_dir=staging, token=token))
                if local.stat().st_size != expected_size:
                    raise RuntimeError(f"{task}: size mismatch {name}")
                digest = sha256(local)
                if siblings[name].lfs and siblings[name].lfs.sha256 and digest != siblings[name].lfs.sha256:
                    raise RuntimeError(f"{task}: sha256 mismatch {name}")
                link_or_copy(local, target)
                count = audit_shard(target)
                sidecar = str(Path(name).with_suffix(".json"))
                if sidecar not in siblings:
                    raise RuntimeError(f"{task}: missing formal sidecar {sidecar}")
                side_target = task_root / Path(sidecar).name
                if side_target.is_file() and not side_target.is_symlink():
                    side = side_target
                else:
                    side = Path(hf_hub_download(repo, sidecar, repo_type="dataset", revision=revision, local_dir=staging, token=token))
                link_or_copy(side, side_target)
                side_count = audit_sidecar(side_target)
                rows.append({
                    "path": str(target),
                    "bytes": expected_size,
                    "sha256": digest,
                    "episodes": count,
                    "sidecar_path": str(side_target),
                    "sidecar_sha256": sha256(side_target),
                    "sidecar_episodes": side_count,
                })
        receipt = {"schema": "before-we-act.mars-control.dataset.v2", "status": "PASSED", "task": task, "env_id": env_id, "repo_id": repo, "revision": revision, "formal_shards": rows, "formal_episodes": 150, "training_policy": "all_data_no_split"}
        atomic_json(args.data_root / task / "download_receipt.json", receipt)
        receipts.append(receipt)
        print(json.dumps({"task": task, "shards": 10, "episodes": 150, "bytes": sum(row["bytes"] for row in rows)}), flush=True)
    atomic_json(args.data_root / "download_receipt.json", {"schema": "before-we-act.mars-control.dataset.bundle.v2", "status": "PASSED", "tasks": receipts, "episodes": 600, "training_policy": "all_data_no_split"})
    print(json.dumps({"status": "PASSED", "tasks": 4, "episodes": 600}), flush=True)


if __name__ == "__main__":
    main()
