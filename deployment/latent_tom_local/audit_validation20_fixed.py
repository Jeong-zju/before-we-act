from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


TASKS = (
    "lift_barrier",
    "camera_alignment",
    "long_pipeline_delivery",
    "take_photo",
    "pass_shoe",
    "place_food",
)
BASE = Path("/workspace/bwa_latent_tom_runs")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate(summary: dict) -> dict:
    if summary.get("status") != "complete" or summary.get("total_episodes") != 120:
        raise RuntimeError("fixed Validation20 is incomplete")
    if tuple(summary.get("tasks", {}).keys()) != TASKS:
        # JSON is emitted in sorted order, so compare sets while retaining the
        # canonical order used in the report below.
        if set(summary.get("tasks", {})) != set(TASKS):
            raise RuntimeError("fixed Validation20 task set mismatch")
    per_task = {}
    for task in TASKS:
        row = summary["tasks"][task]
        details = row.get("episodes_detail", [])
        ids = [int(item["episode"]) for item in details]
        if ids != list(range(20)) or len(set(ids)) != 20:
            raise RuntimeError(f"{task}: episode ids are not exactly 0..19")
        errors = [item for item in details if item.get("error")]
        if errors:
            raise RuntimeError(f"{task}: {len(errors)} rollout errors")
        successes = sum(bool(item.get("success")) for item in details)
        if successes != int(row.get("successes", -1)):
            raise RuntimeError(f"{task}: success count mismatch")
        per_task[task] = {
            "episodes": 20,
            "successes": successes,
            "success_rate": successes / 20,
        }
    return per_task


def main() -> None:
    original_path = BASE / "formal/validation20/summary.json"
    fixed_path = BASE / "formal/validation20_fixed/summary.json"
    original = json.loads(original_path.read_text())
    fixed = json.loads(fixed_path.read_text())
    per_task = validate(fixed)
    fixed_successes = sum(row["successes"] for row in per_task.values())
    if fixed_successes <= 0:
        raise RuntimeError("fixed Validation20 still has zero successes")
    original_successes = sum(int(original["tasks"][task]["successes"]) for task in TASKS)
    checkpoint_path = BASE / "formal/last.pt"
    source_root = Path("/workspace/repos/before-we-act/deployment/latent_tom_local")
    payload = {
        "schema": "bwa.latent_tom.validation20_fixed_audit.v1",
        "status": "complete",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "root_cause": {
            "category": "inference_action_standardization_mismatch",
            "description": (
                "DDIM clip_sample=True clipped mean/std-standardized actions to [-1,1], "
                "making targets beyond one standard deviation unreachable; the closed-gripper "
                "target was the clearest affected dimension. The fixed evaluator uses "
                "clip_sample=False and deterministic rollout-seeded diffusion noise."
            ),
            "checkpoint_retrained": False,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "step": 300000,
            "sha256": sha256(checkpoint_path),
            "contract": "shared_weights_local_rgb_qpos_task_to_local_action8",
        },
        "protocol": {
            "episodes_per_task": 20,
            "total_episodes": 120,
            "episode_ids": list(range(20)),
            "seed_base": 20260820,
            "sim_backend": "cpu",
            "diffusion_steps": 20,
            "replan_interval": 8,
            "policy_contract": "same_checkpoint_per_actor_strict_local_rgb_qpos_to_local_action8",
        },
        "original_buggy_result": {
            "path": str(original_path),
            "sha256": sha256(original_path),
            "successes": original_successes,
            "episodes": 120,
        },
        "fixed_result": {
            "path": str(fixed_path),
            "sha256": sha256(fixed_path),
            "successes": fixed_successes,
            "episodes": 120,
            "micro_success_rate": fixed_successes / 120,
            "macro_success_rate": sum(row["success_rate"] for row in per_task.values()) / 6,
            "per_task": per_task,
        },
        "source": {
            "local_policy.py": sha256(source_root / "local_policy.py"),
            "evaluate_closed_loop.py": sha256(source_root / "evaluate_closed_loop.py"),
        },
    }
    atomic_json(BASE / "validation20_fixed_audit.json", payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
