from __future__ import annotations

import hashlib
import json
import tempfile
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from local_dataset import TASKS


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def main():
    run_root = Path("/workspace/bwa_latent_tom_runs")
    formal = run_root / "formal"
    data = Path("/workspace/datasets/robofactory_multitask")
    report_path = run_root / "final_report.json"
    summary_path = formal / "validation20/summary.json"
    checkpoint_path = formal / "last.pt"
    train = json.loads((formal / "status.json").read_text())
    policy_audit = json.loads((run_root / "audit/policy_contract.json").read_text())
    runtime_audit = json.loads((run_root / "audit/runtime_isolation.json").read_text())
    validation = json.loads(summary_path.read_text())
    report = json.loads(report_path.read_text())

    require(train["status"] == "complete", "training status is not complete")
    require(train["all_episodes"] is True, "training did not declare all episodes")
    require(int(train["step"]) == 300000, f"training stopped at step {train['step']}")
    require(int(train["indexed_local_timesteps"]) == 892107, "local timestep count changed")
    require(policy_audit["status"] == "complete", "strict-local policy audit incomplete")
    require(policy_audit["shared_checkpoint"] is True, "checkpoint is not shared")
    require(policy_audit["inputs"] == ["own_rgb_history", "own_qpos_history", "task_id"],
            "strict-local inputs changed")
    require(runtime_audit["status"] == "complete", "runtime actor isolation audit incomplete")
    require(int(runtime_audit["checkpoint_step"]) == 300000, "runtime audit did not use final checkpoint")
    require(float(runtime_audit["encoder_cross_actor_max_abs_diff"]) == 0.0 and
            float(runtime_audit["action_cross_actor_max_abs_diff"]) == 0.0,
            "runtime batch inference has cross-actor influence")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    require(int(checkpoint["step"]) == 300000, "checkpoint is not final step")
    require(checkpoint["contract"] == "shared_weights_local_rgb_qpos_task_to_local_action8",
            "checkpoint contract mismatch")
    require("model" in checkpoint and "ema_model" in checkpoint, "raw or EMA weights missing")
    require(int(checkpoint.get("ema_optimization_step", -1)) == 300000,
            "EMA optimization step mismatch")

    require(validation["status"] == "complete", "Validation20 status is not complete")
    require(int(validation["episodes_per_task"]) == 20, "Validation20 per-task count mismatch")
    require(int(validation["total_episodes"]) == 120, "Validation20 total count mismatch")
    require(set(validation["tasks"]) == set(TASKS), "Validation20 task set mismatch")
    validation_rows = {}
    for task in TASKS:
        task_result = validation["tasks"][task]
        detail = task_result["episodes_detail"]
        require(task_result["status"] == "complete", f"{task}: validation incomplete")
        require(int(task_result["episodes"]) == 20 and len(detail) == 20,
                f"{task}: expected exactly 20 episodes")
        require(sorted(int(row["episode"]) for row in detail) == list(range(20)),
                f"{task}: episode identifiers are incomplete or duplicated")
        require(not any(row.get("error") for row in detail), f"{task}: episode errors present")
        require(task_result["policy_contract"] ==
                "shared_weights_strict_local_rgb_qpos_to_local_action8",
                f"{task}: policy contract mismatch")
        validation_rows[task] = {"episodes": 20, "successes": int(task_result["successes"]),
                                 "success_rate": float(task_result["success_rate"])}

    dataset_rows = {}
    for task in TASKS:
        receipt_path = data / task / "download_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        require(receipt["status"] == "complete", f"{task}: dataset incomplete")
        require(int(receipt["episodes_total"]) == 150, f"{task}: expected 150 episodes")
        require(bool(receipt.get("revision")), f"{task}: pinned revision missing")
        dataset_rows[task] = {"episodes": 150, "revision": receipt["revision"],
                              "receipt_sha256": digest(receipt_path)}

    checkpoint_sha = digest(checkpoint_path)
    validation_sha = digest(summary_path)
    require(report["status"] == "complete" and int(report["episodes"]) == 900,
            "final report episode/status mismatch")
    require(report["checkpoint"]["sha256"] == checkpoint_sha,
            "final report checkpoint hash mismatch")
    require(report["validation20"]["sha256"] == validation_sha,
            "final report validation hash mismatch")
    require(int(report["validation20"]["episodes"]) == 120,
            "final report validation count mismatch")

    required_stage_receipts = (
        "download_six_datasets", "audit_decentralized_training_contract",
        "audit_strict_local_policy_contract", "compute_latent_tom_normalization",
        "build_local_320x240_cache", "latent_tom_train_smoke",
        "latent_tom_checkpoint_reload_smoke", "latent_tom_train_full_all_episodes",
        "latent_tom_final_checkpoint_reload", "latent_tom_validation20_six_tasks",
        "latent_tom_final_report", "audit_runtime_actor_isolation",
    )
    receipt_root = run_root / "supervisor/receipts"
    stage_rows = {}
    for name in required_stage_receipts:
        path = receipt_root / f"{name}.json"
        receipt = json.loads(path.read_text())
        require(receipt["status"] == "complete", f"stage receipt incomplete: {name}")
        stage_rows[name] = {"sha256": digest(path), "attempt": receipt["attempt"]}

    output = {"schema": "bwa.latent_tom.delivery_audit.v1", "status": "complete",
              "training_step": 300000, "episodes": 900, "indexed_local_timesteps": 892107,
              "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha,
                             "ema": True, "contract": checkpoint["contract"]},
              "validation20": {"path": str(summary_path), "sha256": validation_sha,
                               "total_episodes": 120, "tasks": validation_rows},
              "datasets": dataset_rows, "stage_receipts": stage_rows,
              "strict_local_inputs": policy_audit["inputs"],
              "forbidden_inputs": policy_audit["forbidden"],
              "runtime_actor_isolation": runtime_audit,
              "audited_at": datetime.now(timezone.utc).isoformat()}
    path = run_root / "delivery_audit.json"
    atomic_json(path, output)
    print(json.dumps({"status": "complete", "output": str(path), "episodes": 900,
                      "validation_episodes": 120}))


if __name__ == "__main__":
    main()
