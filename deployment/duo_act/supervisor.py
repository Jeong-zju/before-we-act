from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from pathlib import Path


REPO = Path(os.environ.get("DUO_ACT_REPO", "/workspace/repos/before-we-act"))
RUN = Path(os.environ.get("DUO_ACT_RUN", "/workspace/runs/duobench-act"))
DATASET = Path(os.environ.get("DUO_ACT_DATASET", "/workspace/datasets/duobench"))
DATA = RUN / "data_unclipped"
FROZEN_CONFIG = REPO / "configs" / "duobench_act_causal_lag1_prior_v1.json"
EXPERIMENT = RUN / "causal_lag1_prior"
LOG = RUN / "logs"
STATUS = RUN / "status.json"
PYTHON = "/venv/main/bin/python"


def status(stage: str, state: str = "running", **extra):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "duobench-act-supervisor-v1", "stage": stage, "state": state,
        "updated_at": time.time(), "gpu_schedule": "one RTX 5090; exclusive formal training; up to three validation simulators",
        **extra,
    }
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, STATUS)


def run(stage: str, command: list[str], retries: int = 1):
    LOG.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{REPO}:/workspace/repos/duobench/src",
        "CUDA_VISIBLE_DEVICES": "0", "MUJOCO_GL": "egl",
        "DUOBENCH_PREFIX": "/workspace/datasets/duobench_assets",
        "HF_HOME": "/workspace/.hf_home", "WANDB_MODE": "disabled",
        "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
        "LD_LIBRARY_PATH": "/venv/main/lib/python3.12/site-packages/mujoco:" + env.get("LD_LIBRARY_PATH", ""),
    })
    for attempt in range(1, retries + 1):
        status(stage, attempt=attempt, command=command)
        with (LOG / f"{stage}.log").open("a") as output:
            output.write(json.dumps({"event": "launch", "attempt": attempt, "command": command, "time": time.time()}) + "\n")
            output.flush()
            process = subprocess.Popen(
                command, cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
            )
            returncode = process.wait()
        if returncode == 0:
            return
        if attempt < retries:
            time.sleep(min(60, 10 * attempt))
    raise RuntimeError(f"{stage} exited {returncode} after {retries} attempt(s)")


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        if not (DATA / "manifest.json").is_file():
            run("data_prepare", [
                PYTHON, "-m", "deployment.duo_act.prepare", "--dataset", str(DATASET),
                "--output", str(DATA), "--image-size", "224", "--jobs", "6",
            ], retries=2)
        if not (RUN / "audit_unclipped.json").is_file():
            run("data_audit", [PYTHON, "-m", "deployment.duo_act.audit", "--data", str(DATA), "--output", str(RUN / "audit_unclipped.json")])
        if not (RUN / "preflight_state_binary.json").is_file():
            run("interface_preflight", [PYTHON, "-m", "deployment.duo_act.preflight", "--data", str(DATA), "--output", str(RUN / "preflight_state_binary.json")])
        smoke = RUN / "smoke" / "final.pt"
        if not smoke.is_file():
            run("smoke_train", [
                PYTHON, "-m", "deployment.duo_act.train", "--data", str(DATA), "--output", str(smoke.parent),
                "--updates", "5", "--batch-size", "40", "--workers", "8", "--horizon", "100",
                "--action-lag", "1", "--save-every", "5", "--smoke",
            ])
        smoke_summary = RUN / "smoke" / "validation" / "summary.json"
        if not smoke_summary.is_file():
            run("smoke_validation", [
                PYTHON, "-m", "deployment.duo_act.validation_launcher", "--checkpoint", str(smoke),
                "--data", str(DATA), "--output", str(smoke_summary.parent), "--episodes", "1",
                "--max-steps", "2", "--workers", "1",
            ])
        formal = EXPERIMENT / "final.pt"
        if not formal.is_file():
            run("formal_train", [
                PYTHON, "-m", "deployment.duo_act.train", "--config", str(FROZEN_CONFIG),
            ], retries=3)
        validation = EXPERIMENT / "validation20_open30" / "summary.json"
        if not validation.is_file():
            run("validation20", [
                PYTHON, "-m", "deployment.duo_act.validation_launcher", "--checkpoint", str(formal),
                "--data", str(DATA), "--output", str(validation.parent), "--episodes", "20", "--workers", "3",
                "--mode", "open30",
            ], retries=3)
        status("complete", "complete", checkpoint=str(formal), summary=str(validation))
    except Exception as error:
        status("failed", "failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
