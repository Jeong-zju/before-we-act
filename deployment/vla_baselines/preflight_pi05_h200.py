#!/usr/bin/env python3
"""Fail-fast audit for the one-H200 pi0.5 LoRA deployment."""
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

OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
ROBOFACTORY_COMMIT = "5868242322414a91454e22f1dd9641f613ba1bcf"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def git_head(path: str) -> str:
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token or TOKEN_FILE.stat().st_mode & 0o077:
        raise RuntimeError("Hugging Face token is missing or not mode 0600")
    gpu_lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    if len(gpu_lines) != 1:
        raise RuntimeError(f"expected exactly one GPU, found {len(gpu_lines)}")
    index, name, total, free = (value.strip() for value in gpu_lines[0].split(",", 3))
    if "H200" not in name or int(total) < 140_000:
        raise RuntimeError(f"expected a 141-GiB H200, found {name} ({total} MiB)")
    openpi_head = git_head("/workspace/repos/openpi")
    robofactory_head = git_head("/workspace/repos/RoboFactory")
    if openpi_head != OPENPI_COMMIT or robofactory_head != ROBOFACTORY_COMMIT:
        raise RuntimeError(f"source revision drift: openpi={openpi_head}, RoboFactory={robofactory_head}")

    api = HfApi(token=token)
    datasets = {}
    required = 0
    for task, (repo_id, revision) in REPOS.items():
        info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise RuntimeError(f"dataset revision drift for {task}: {info.sha}")
        size = sum(int(item.size or 0) for item in info.siblings if item.rfilename.endswith(".hdf5"))
        datasets[task] = {"repo_id": repo_id, "revision": revision, "bytes": size}
        required += size
    free_bytes = shutil.disk_usage("/workspace").free
    reserve = 120 * 1024**3
    if free_bytes < required + reserve:
        raise RuntimeError(f"insufficient disk: free={free_bytes}, datasets={required}, reserve={reserve}")
    atomic_json(
        Path("/workspace/bwa_pi05_runs/audit/preflight.json"),
        {
            "schema": "bwa.pi05.h200.preflight.v1",
            "status": "complete",
            "gpu": {"index": int(index), "name": name, "memory_total_mib": int(total), "memory_free_mib": int(free)},
            "openpi_commit": openpi_head,
            "robofactory_commit": robofactory_head,
            "datasets": datasets,
            "dataset_bytes": required,
            "workspace_free_bytes": free_bytes,
            "workspace_is_persistent_volume": False,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
