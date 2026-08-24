#!/usr/bin/env python3
"""Build a compact, hashed receipt after all formal training and Validation20 stages."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path("/workspace/bwa_vla_runs/formal")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files(policy: str, final: Path) -> list[Path]:
    if policy == "rdt":
        return [path for name in ("config.json", "pytorch_model.bin", "ema/model.safetensors") if (path := final / name).is_file()]
    if policy == "openvla":
        names = ("config.json", "dataset_statistics.json", "action_head--latest_checkpoint.pt",
                 "proprio_projector--latest_checkpoint.pt", "lora_adapter/adapter_model.safetensors")
        return [path for name in names if (path := final / name).is_file()] + sorted(final.glob("model-*.safetensors"))
    roots = (final / "params", final / "assets")
    return sorted(path for root in roots for path in root.rglob("*") if path.is_file())


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
    layout = {"openvla": ROOT / "openvla_oft"}
    policies = {}
    for policy, policy_root in layout.items():
        final = (policy_root / "final").resolve(strict=True)
        validation = json.loads((ROOT / policy / "validation20/summary.json").read_text())
        if validation.get("status") != "complete" or validation.get("total_episodes") != 120:
            raise RuntimeError(f"incomplete Validation20: {policy}")
        files = selected_files(policy, final)
        if not files:
            raise RuntimeError(f"no inference artifacts selected: {policy}")
        artifacts = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]
        policies[policy] = {
            "checkpoint": str(final), "artifacts": artifacts,
            "validation20": str(ROOT / policy / "validation20/summary.json"),
            "macro_success_rate": validation["macro_success_rate"],
        }
    atomic_json(
        Path("/workspace/bwa_vla_runs/final_report.json"),
        {
            "schema": "bwa.vla.formal_run.v1", "status": "complete", "policies": policies,
            "dataset": "/workspace/datasets/robofactory_multitask", "episodes": 900,
            "validation_episodes": 120,
            "policy_contract": "shared_weights_decentralized_local_rgb_qpos_to_local_action8",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
