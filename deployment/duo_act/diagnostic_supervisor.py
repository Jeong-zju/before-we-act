"""Persistent causal/prior/action-runtime diagnostic supervisor."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


REPO = Path("/workspace/repos/before-we-act")
RUN = Path("/workspace/runs/duobench-act")
DATA = RUN / "data_unclipped"
EXPERIMENT = RUN / "causal_lag1_prior"
FROZEN_CONFIG = REPO / "configs" / "duobench_act_causal_lag1_prior_v1.json"
FINAL = EXPERIMENT / "final.pt"
OLD = RUN / "formal" / "final.pt"
OUTPUT = EXPERIMENT / "ablations"
STATUS = EXPERIMENT / "diagnostic_status.json"
PYTHON = "/venv/main/bin/python"
TASKS = ("ball_maze", "spring_door", "transfer_gate")
MODES = ("first", "open30", "ensemble")
TASK_INDEX = {"ball_maze": 0, "spring_door": 7, "transfer_gate": 9}


def write_status(stage: str, **extra) -> None:
    payload = {"stage": stage, "updated_at": time.time(), **extra}
    temporary = STATUS.with_suffix(".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, STATUS)


def training_is_running() -> bool:
    marker = "deployment.duo_act.train"
    experiment = str(EXPERIMENT)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        if marker in command and experiment in command:
            return True
    return False


def train_command() -> list[str]:
    return [
        PYTHON,
        "-m",
        "deployment.duo_act.train",
        "--config",
        str(FROZEN_CONFIG),
    ]


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": f"{REPO}:/workspace/repos/duobench/src",
            "CUDA_VISIBLE_DEVICES": "0",
            "MUJOCO_GL": "egl",
            "DUOBENCH_PREFIX": "/workspace/datasets/duobench_assets",
            "HF_HOME": "/workspace/.hf_home",
            "WANDB_MODE": "disabled",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        }
    )
    return env


def ensure_training() -> None:
    while not FINAL.is_file():
        if training_is_running():
            write_status("training", final_ready=False)
            time.sleep(30)
            continue
        write_status("resuming_training", command=train_command())
        log = EXPERIMENT / "logs" / "supervised_resume.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            result = subprocess.run(
                train_command(),
                cwd=REPO,
                env=runtime_env(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode and not FINAL.is_file():
            write_status("training_retry", returncode=result.returncode)
            time.sleep(30)


def jobs():
    for checkpoint_name, checkpoint in (("old", OLD), ("new", FINAL)):
        for task in TASKS:
            for mode in MODES:
                output = OUTPUT / f"{checkpoint_name}_{task}_{mode}.json"
                yield checkpoint_name, checkpoint, task, mode, output


def launch_job(job):
    checkpoint_name, checkpoint, task, mode, output = job
    command = [
        PYTHON,
        "-m",
        "deployment.duo_act.ablate_rollout",
        "--checkpoint",
        str(checkpoint),
        "--data",
        str(DATA),
        "--task",
        task,
        "--mode",
        mode,
        "--episodes",
        "3",
        "--episode-start",
        "0",
        "--no-clip",
        "--output",
        str(output),
    ]
    log = output.with_suffix(".log").open("a")
    process = subprocess.Popen(
        command,
        cwd=REPO,
        env=runtime_env(),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return job, process, log


def run_ablations() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pending = []
    for job in jobs():
        output = job[-1]
        if output.is_file():
            try:
                rows = json.loads(output.read_text())
            except (OSError, json.JSONDecodeError):
                rows = []
            if len(rows) == 3 and all(
                row.get("state_gripper_encoding")
                == "physical_width_gt_0.9_to_binary"
                and not row.get("gym_box_clip", True)
                for row in rows
            ):
                continue
        pending.append(job)

    active = []
    while pending or active:
        while pending and len(active) < 3:
            active.append(launch_job(pending.pop(0)))
        write_status(
            "ablations",
            pending=len(pending),
            active=[f"{item[0][0]}:{item[0][2]}:{item[0][3]}" for item in active],
        )
        job, process, log = active.pop(0)
        returncode = process.wait()
        log.close()
        if returncode:
            raise RuntimeError(
                f"ablation {job[0]}:{job[2]}:{job[3]} exited {returncode}"
            )


def aggregate() -> dict:
    rows = []
    for checkpoint_name, _, task, mode, output in jobs():
        for row in json.loads(output.read_text()):
            rows.append({"checkpoint": checkpoint_name, **row})
    cells = {}
    for checkpoint in ("old", "new"):
        cells[checkpoint] = {}
        for task in TASKS:
            cells[checkpoint][task] = {}
            for mode in MODES:
                selected = [
                    row
                    for row in rows
                    if row["checkpoint"] == checkpoint
                    and row["task"] == task
                    and row["mode"] == mode
                ]
                cells[checkpoint][task][mode] = {
                    "episodes": len(selected),
                    "successes": sum(int(row["success"]) for row in selected),
                    "success_rate": float(np.mean([row["success"] for row in selected])),
                    "mean_max_stage_progress": float(
                        np.mean([row["max_stage_progress"] for row in selected])
                    ),
                    "max_stage_progress": float(
                        np.max([row["max_stage_progress"] for row in selected])
                    ),
                    "transitional_gripper_observations": sum(
                        int(row["transitional_gripper_observations"])
                        for row in selected
                    ),
                }
    result = {
        "schema": "duobench-act-causal-prior-three-task-ablation-v1",
        "status": "complete",
        "seeds_by_task": {
            task: [
                20260820 + TASK_INDEX[task] * 1000 + episode
                for episode in range(3)
            ]
            for task in TASKS
        },
        "tasks": list(TASKS),
        "modes": list(MODES),
        "runtime_contract": {
            "state_gripper": "physical_width_gt_0.9_to_binary",
            "gym_box_clip": False,
            "controller_saturation": "MuJoCo_FR3_actuator_ctrlrange",
        },
        "checkpoints": {"old": str(OLD), "new": str(FINAL)},
        "cells": cells,
        "rows": rows,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    try:
        ensure_training()
        if not OLD.is_file():
            raise FileNotFoundError(OLD)
        run_ablations()
        summary = aggregate()
        write_status("complete", state="complete", summary=str(OUTPUT / "summary.json"))
        print(json.dumps({"status": "complete", "cells": summary["cells"]}), flush=True)
    except Exception as error:
        write_status("failed", state="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
