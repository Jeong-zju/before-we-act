from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from local_dataset import TASKS


RUN = Path("/workspace/bwa_latent_tom_runs")
FORMAL = RUN / "formal"
ROOT = Path("/workspace/repos/before-we-act/deployment/latent_tom_local")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def commit(path: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path}", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    fixed_audit_path = RUN / "validation20_fixed_audit.json"
    fixed_audit = json.loads(fixed_audit_path.read_text())
    if fixed_audit.get("status") != "complete":
        raise RuntimeError("fixed Validation20 audit incomplete")
    train = json.loads((FORMAL / "status.json").read_text())
    checkpoint_path = FORMAL / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    if int(checkpoint["step"]) != 300000 or "model" not in checkpoint or "ema_model" not in checkpoint:
        raise RuntimeError("final checkpoint is incomplete")
    receipts = {path.stem: {"path": str(path), "sha256": digest(path)}
                for path in sorted((RUN / "supervisor/receipts").glob("*.json"))}
    source_files = (
        "local_dataset.py", "local_policy.py", "train_local.py", "evaluate_closed_loop.py",
        "run_validation20_fixed.sh", "audit_validation20_fixed.py",
        "finalize_validation20_fixed.py", "verify_delivery_fixed.py",
        "audit_runtime_isolation.py", "audit_policy_contract.py",
    )
    report = {
        "schema": "bwa.latent_tom.final_report.fixed_validation.v2",
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "contract": "shared_weights_local_rgb_qpos_task_to_local_action8",
            "deployment": "same_checkpoint_per_actor_strict_local_rgb_qpos_to_local_action8",
        },
        "checkpoint": {
            "path": str(checkpoint_path), "sha256": digest(checkpoint_path),
            "step": int(checkpoint["step"]), "raw_weights": True, "ema_weights": True,
            "ema_optimization_step": int(checkpoint["ema_optimization_step"]),
        },
        "training": {
            "all_episodes": bool(train["all_episodes"]), "episodes": 900,
            "indexed_local_timesteps": int(train["indexed_local_timesteps"]),
            "steps": int(train["step"]), "batch_size": 512, "workers": 16,
            "precision": "bfloat16", "learning_rate": 1e-4,
            "action_horizon": 40, "observation_steps": 2,
        },
        "validation20_fixed": fixed_audit["fixed_result"],
        "superseded_buggy_validation20": fixed_audit["original_buggy_result"],
        "root_cause": fixed_audit["root_cause"],
        "source": {
            "official_latent_tom": "a51d929027799a53d54e7d7d2ba90e2703642b4a",
            "before_we_act": commit("/workspace/repos/before-we-act"),
            "robofactory": commit("/workspace/repos/RoboFactory"),
            "pipeline_sha256": digest(Path("/workspace/bwa_latent_tom_pipeline/pipeline.json")),
            "adaptation_files": {name: digest(ROOT / name) for name in source_files},
        },
        "supervisor_stage_receipts": receipts,
    }
    atomic_json(RUN / "final_report_fixed_validation_v2.json", report)
    print(json.dumps({"status": "complete", "successes": report["validation20_fixed"]["successes"],
                      "episodes": report["validation20_fixed"]["episodes"]}))


if __name__ == "__main__":
    main()
