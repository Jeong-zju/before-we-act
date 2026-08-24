from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from local_dataset import TASKS


RUN = Path("/workspace/bwa_latent_tom_runs")
FORMAL = RUN / "formal"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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
    train = json.loads((FORMAL / "status.json").read_text())
    policy = json.loads((RUN / "audit/policy_contract.json").read_text())
    runtime = json.loads((RUN / "audit/runtime_isolation_fixed_v2.json").read_text())
    fixed = json.loads((RUN / "validation20_fixed_audit.json").read_text())
    report = json.loads((RUN / "final_report_fixed_validation_v2.json").read_text())
    checkpoint_path = FORMAL / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    require(train["status"] == "complete" and train["all_episodes"], "training completeness")
    require(int(train["step"]) == 300000 and int(train["indexed_local_timesteps"]) == 892107, "training scope")
    require(int(checkpoint["step"]) == 300000 and int(checkpoint["ema_optimization_step"]) == 300000, "checkpoint step")
    require("model" in checkpoint and "ema_model" in checkpoint, "checkpoint raw/EMA")
    require(checkpoint["contract"] == "shared_weights_local_rgb_qpos_task_to_local_action8", "checkpoint contract")
    require(policy["status"] == "complete" and policy["inputs"] == ["own_rgb_history", "own_qpos_history", "task_id"], "local policy inputs")
    require(runtime["status"] == "complete" and runtime["checkpoint_step"] == 300000, "runtime audit")
    require(float(runtime["encoder_cross_actor_max_abs_diff"]) == 0.0 and float(runtime["action_cross_actor_max_abs_diff"]) == 0.0, "cross-actor influence")
    fixed_result = fixed["fixed_result"]
    require(fixed["status"] == "complete" and fixed_result["episodes"] == 120 and fixed_result["successes"] > 0, "fixed Validation20")
    summary = json.loads(Path(fixed_result["path"]).read_text())
    require(summary["status"] == "complete" and summary["episodes_per_task"] == 20 and summary["total_episodes"] == 120, "summary protocol")
    task_rows = {}
    for task in TASKS:
        row = summary["tasks"][task]
        detail = row["episodes_detail"]
        require(len(detail) == 20 and [int(item["episode"]) for item in detail] == list(range(20)), f"{task}: episodes")
        require(not any(item.get("error") for item in detail), f"{task}: errors")
        task_rows[task] = {"episodes": 20, "successes": int(row["successes"]), "success_rate": float(row["success_rate"])}
    checkpoint_sha = digest(checkpoint_path)
    require(report["checkpoint"]["sha256"] == checkpoint_sha == fixed["checkpoint"]["sha256"], "checkpoint hash chain")
    require(report["validation20_fixed"]["sha256"] == fixed_result["sha256"] == digest(Path(fixed_result["path"])), "validation hash chain")
    required = (
        "download_six_datasets", "audit_decentralized_training_contract", "audit_strict_local_policy_contract",
        "compute_latent_tom_normalization", "build_local_320x240_cache", "latent_tom_train_smoke",
        "latent_tom_checkpoint_reload_smoke", "latent_tom_train_full_all_episodes",
        "latent_tom_final_checkpoint_reload", "latent_tom_validation20_six_tasks", "latent_tom_final_report",
        "audit_runtime_actor_isolation", "audit_latent_tom_delivery", "latent_tom_validation20_fixed_v2",
        "audit_latent_tom_validation20_fixed_v2", "audit_runtime_actor_isolation_fixed_v2",
        "latent_tom_final_report_fixed_v2",
    )
    receipt_root = RUN / "supervisor/receipts"
    receipts = {}
    for name in required:
        path = receipt_root / f"{name}.json"
        payload = json.loads(path.read_text())
        require(payload["status"] == "complete", f"receipt: {name}")
        receipts[name] = {"attempt": payload["attempt"], "sha256": digest(path)}
    output = {
        "schema": "bwa.latent_tom.delivery_audit.fixed_validation.v2", "status": "complete",
        "audited_at": datetime.now(timezone.utc).isoformat(), "checkpoint": {"step": 300000, "sha256": checkpoint_sha, "raw": True, "ema": True},
        "training": {"episodes": 900, "timesteps": 892107}, "strict_local_inputs": policy["inputs"],
        "runtime_actor_isolation": runtime, "validation20_fixed": {"episodes": 120, "successes": fixed_result["successes"], "micro_success_rate": fixed_result["micro_success_rate"], "per_task": task_rows, "sha256": fixed_result["sha256"]},
        "superseded_buggy_validation20": fixed["original_buggy_result"], "root_cause": fixed["root_cause"],
        "stage_receipts": receipts,
    }
    atomic_json(RUN / "delivery_audit_fixed_validation_v2.json", output)
    print(json.dumps({"status": "complete", "successes": fixed_result["successes"], "episodes": 120}))


if __name__ == "__main__":
    main()
