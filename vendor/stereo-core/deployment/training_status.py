#!/usr/bin/env python3
"""Emit one JSON record describing the formal train/evaluation pipeline."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


RUN_ROOT = Path("/workspace/runs/no_wrist_stereo_core_120k")
TRAIN_LOG = Path("/workspace/logs/no_wrist_train_formal.log")
TOTAL_UPDATES = 120_000
BEIJING = ZoneInfo("Asia/Shanghai")
EVAL_TASKS = (
    "lift_barrier",
    "camera_alignment",
    "three_robots_stack_cube",
    "long_pipeline_delivery",
    "take_photo",
)
EPISODES_PER_TASK = 100


def last_records() -> tuple[dict, int | None]:
    latest: dict = {}
    saved_update: int | None = None
    if not TRAIN_LOG.exists():
        return latest, saved_update
    for raw_line in TRAIN_LOG.read_text(errors="replace").splitlines():
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if "update" in record:
            latest = record
        if "saved_update" in record:
            saved_update = int(record["saved_update"])
    return latest, saved_update


def trainer_running() -> bool:
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = cmdline.read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"stereo_core/train_no_wrist_pair.py" in command:
            return True
    return False


def gpu_status() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        fields = subprocess.check_output(command, text=True, timeout=10).strip().split(", ")
        return {
            "utilization_percent": int(fields[0]),
            "memory_used_mib": int(fields[1]),
            "memory_total_mib": int(fields[2]),
            "temperature_c": int(fields[3]),
            "power_w": float(fields[4]),
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def evaluation_status(eval_root: Path, now: datetime, final_checkpoint: Path) -> dict:
    task_status: dict[str, dict] = {}
    total_episodes = 0
    total_successes = 0
    current_task: str | None = None

    for task in EVAL_TASKS:
        output = eval_root / f"{task}.json"
        log = eval_root / f"{task}.log"
        rows_by_seed: dict[int, dict] = {}
        if output.exists():
            try:
                payload = json.loads(output.read_text())
                for row in payload.get("rows", []):
                    if "seed" in row:
                        rows_by_seed[int(row["seed"])] = row
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass
        if log.exists():
            for raw_line in log.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if row.get("task") == task and "seed" in row:
                    rows_by_seed[int(row["seed"])] = row

        episodes = len(rows_by_seed)
        successes = sum(bool(row.get("success")) for row in rows_by_seed.values())
        complete = episodes >= EPISODES_PER_TASK
        if current_task is None and not complete and (log.exists() or total_episodes > 0):
            current_task = task
        task_status[task] = {
            "episodes": episodes,
            "successes": successes,
            "complete": complete,
        }
        total_episodes += episodes
        total_successes += successes

    if current_task is None and total_episodes < len(EVAL_TASKS) * EPISODES_PER_TASK:
        current_task = EVAL_TASKS[0]

    elapsed_seconds = (
        max(0.0, now.timestamp() - final_checkpoint.stat().st_mtime)
        if final_checkpoint.exists()
        else 0.0
    )
    episodes_per_hour = total_episodes * 3600 / elapsed_seconds if elapsed_seconds > 0 else None
    remaining = len(EVAL_TASKS) * EPISODES_PER_TASK - total_episodes
    eta = (
        now + timedelta(hours=remaining / episodes_per_hour)
        if episodes_per_hour and remaining > 0
        else None
    )
    return {
        "current_task": current_task,
        "tasks": task_status,
        "episodes": total_episodes,
        "total_episodes": len(EVAL_TASKS) * EPISODES_PER_TASK,
        "progress_percent": round(100 * total_episodes / (len(EVAL_TASKS) * EPISODES_PER_TASK), 2),
        "successes_so_far": total_successes,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "episodes_per_hour_naive": round(episodes_per_hour, 2) if episodes_per_hour else None,
        "estimated_complete_beijing_naive": eta.isoformat(timespec="minutes") if eta else None,
    }


def main() -> None:
    now = datetime.now(BEIJING)
    latest, saved_update = last_records()
    final_checkpoint = RUN_ROOT / "checkpoint_120000.pt"
    eval_root = RUN_ROOT / "frozen100"
    evaluation_complete = (eval_root / ".complete").exists()
    running = trainer_running()

    if evaluation_complete:
        phase = "complete"
    elif final_checkpoint.exists():
        phase = "evaluating"
    elif running:
        phase = "training"
    else:
        phase = "waiting_or_recovering"

    update = int(latest.get("update", 0))
    eta_hours = latest.get("eta_hours")
    result = {
        "observed_at_beijing": now.isoformat(timespec="seconds"),
        "phase": phase,
        "trainer_running": running,
        "update": update,
        "total_updates": TOTAL_UPDATES,
        "progress_percent": round(100 * update / TOTAL_UPDATES, 3),
        "saved_update": saved_update,
        "updates_per_hour": latest.get("updates_per_hour"),
        "eta_hours": eta_hours,
        "estimated_training_complete_beijing": (
            (now + timedelta(hours=float(eta_hours))).isoformat(timespec="minutes")
            if eta_hours is not None and not final_checkpoint.exists()
            else None
        ),
        "loss": latest.get("loss"),
        "action_loss": latest.get("action"),
        "checkpoint_latest_bytes": (
            (RUN_ROOT / "checkpoint_latest.pt").stat().st_size
            if (RUN_ROOT / "checkpoint_latest.pt").exists()
            else None
        ),
        "train_log_age_seconds": (
            round(now.timestamp() - TRAIN_LOG.stat().st_mtime, 1) if TRAIN_LOG.exists() else None
        ),
        "gpu": gpu_status(),
        "evaluation_complete": evaluation_complete,
    }
    if final_checkpoint.exists():
        result["evaluation"] = evaluation_status(eval_root, now, final_checkpoint)
    summary = eval_root / "summary.json"
    if summary.exists():
        result["evaluation_summary"] = json.loads(summary.read_text())
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
