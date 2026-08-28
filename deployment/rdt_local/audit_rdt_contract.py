#!/usr/bin/env python3
"""Audit H100/runtime, full-parameter optimizer wiring, and local RDT adapter."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


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
    repo = Path("/workspace/repos/rdt-1b")
    sys.path.insert(0, str(repo))
    token = Path("/workspace/.secrets/hf_token")
    if not token.is_file() or not token.read_text().strip() or token.stat().st_mode & 0o077:
        raise RuntimeError("Hugging Face token is missing, empty, or not mode 0600")
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    if len(gpu_rows) != 4 or any("H100" not in row for row in gpu_rows):
        raise RuntimeError(f"expected exactly four H100 GPUs, got {gpu_rows}")
    train_source = (repo / "train/train.py").read_text()
    tree = ast.parse(train_source)
    full_optimizer = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "params_to_optimize" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "rdt"
        and node.value.func.attr == "parameters"
        for node in ast.walk(tree)
    )
    if not full_optimizer:
        raise RuntimeError("official trainer no longer wires rdt.parameters() into the optimizer")
    from data.hdf5_vla_dataset import HDF5VLADataset

    dataset = HDF5VLADataset()
    sample = dataset.get_item(0)
    if len(dataset._episodes) != 900 or len(dataset._tasks) != 6:
        raise RuntimeError("RDT adapter does not expose all 900 episodes / six tasks")
    if sample["state"].shape != (1, 128) or sample["actions"].shape != (64, 128):
        raise RuntimeError(f"unexpected RDT tensor contract: {sample['state'].shape}, {sample['actions'].shape}")
    payload = {
        "schema": "bwa.rdt.contract.v1", "status": "complete",
        "protocol": "shared_weights_decentralized_local_rgb_qpos_action",
        "training_split": "all_episodes_ignore_manifest_split",
        "episodes": 900, "tasks": dataset._tasks, "local_streams": len(dataset),
        "optimizer_scope": "all_rdt_parameters", "optimizer_ast_verified": True,
        "frozen_condition_encoders": ["siglip", "t5_precomputed"],
        "gpus": gpu_rows, "rdt_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "adapter_sha256": hashlib.sha256((repo / "data/hdf5_vla_dataset.py").read_bytes()).hexdigest(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(Path("/workspace/bwa_rdt_runs/audit/rdt_contract.json"), payload)


if __name__ == "__main__":
    main()
