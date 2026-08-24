#!/usr/bin/env python3
"""Fail-fast host, GPU, storage, secret, and pinned-data preflight."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from huggingface_hub import HfApi

from download_datasets import REPOS, TOKEN_FILE


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("Hugging Face secret file is empty")
    if TOKEN_FILE.stat().st_mode & 0o077:
        raise PermissionError(f"secret permissions are too broad: {oct(TOKEN_FILE.stat().st_mode & 0o777)}")

    gpu_lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    if len(gpu_lines) != 4:
        raise RuntimeError(f"formal run requires exactly four GPUs, found {len(gpu_lines)}")
    gpus = []
    for line in gpu_lines:
        index, name, total, free = (item.strip() for item in line.split(",", 3))
        if "A100" not in name:
            raise RuntimeError(f"GPU {index} is not an A100: {name}")
        gpus.append({"index": int(index), "name": name, "memory_total_mib": int(total), "memory_free_mib": int(free)})
    compute = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    ).strip()
    if compute:
        raise RuntimeError(f"GPU compute processes already exist before launch: {compute}")

    api = HfApi(token=token)
    datasets = {}
    missing_bytes = 0
    root = Path("/workspace/datasets/robofactory_multitask")
    for task, (repo_id, revision) in REPOS.items():
        info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise RuntimeError(f"revision drift for {task}: {info.sha} != {revision}")
        total = sum(int(item.size or 0) for item in info.siblings if item.rfilename.endswith(".hdf5"))
        present = sum(
            path.stat().st_size for path in (root / task / "hdf5").glob("*.hdf5") if path.is_file()
        )
        missing = max(0, total - present)
        datasets[task] = {"repo_id": repo_id, "revision": revision, "bytes": total, "present_bytes": present}
        missing_bytes += missing

    free_bytes = shutil.disk_usage("/workspace").free
    reserve_bytes = max(100 * 1024**3, missing_bytes // 10)
    if free_bytes < missing_bytes + reserve_bytes:
        raise RuntimeError(
            f"insufficient /workspace storage: free={free_bytes}, missing_data={missing_bytes}, reserve={reserve_bytes}"
        )
    payload = {
        "schema": "bwa.openvla.preflight.v1", "status": "complete",
        "gpus": gpus, "datasets": datasets, "missing_dataset_bytes": missing_bytes,
        "workspace_free_bytes": free_bytes, "required_reserve_bytes": reserve_bytes,
        "openvla_commit": subprocess.check_output(
            ["git", "-C", "/workspace/repos/openvla-oft", "rev-parse", "HEAD"], text=True
        ).strip(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if payload["openvla_commit"] != "e4287e94541f459edc4feabc4e181f537cd569a8":
        raise RuntimeError(f"OpenVLA commit mismatch: {payload['openvla_commit']}")
    atomic_json(Path("/workspace/bwa_vla_runs/audit/preflight.json"), payload)


if __name__ == "__main__":
    main()
