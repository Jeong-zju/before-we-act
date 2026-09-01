from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .common import OPENPI_REVISION, POLICY_CONTRACT, atomic_json, sha256_tree


REPO = Path(os.environ.get("DUO_PI05_REPO", "/workspace/repos/before-we-act"))
OPENPI = Path(os.environ.get("DUO_PI05_OPENPI", "/workspace/repos/openpi"))
DUO = Path(os.environ.get("DUO_PI05_DUOBENCH", "/workspace/repos/duobench"))
RCS = Path(os.environ.get("DUO_PI05_RCS", "/workspace/repos/robot-control-stack"))
DATASET = Path(os.environ.get("DUO_PI05_DATASET", "/workspace/datasets/duobench"))
RUN = Path(os.environ.get("DUO_PI05_RUN", "/workspace/runs/pi05_duo"))
DATA = RUN / "prepared"
PYTHON = os.environ.get("DUO_PI05_PYTHON", "/workspace/venvs/openpi/bin/python")
SIM_PYTHON = os.environ.get("DUO_PI05_SIM_PYTHON", "/workspace/venvs/duobench/bin/python")
ACTIVE: subprocess.Popen | None = None
STOP = False


def now() -> str: return datetime.now(timezone.utc).isoformat()


def write_state(stage: str, status: str = "running", **extra) -> None:
    atomic_json(RUN / "state.json", {"schema": "duobench.pi05.supervisor.v1", "stage": stage, "status": status, "updated_at": now(), "policy_contract": POLICY_CONTRACT, "gpu_schedule": "four-GPU JAX data-parallel training; four-GPU validation waves; supervisor never terminates foreign PIDs", **extra})


def load(path: Path) -> dict:
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return {}


def valid(stage: str) -> bool:
    checks = {
        "environment": lambda: OPENPI.is_dir() and DUO.is_dir() and RCS.is_dir() and (OPENPI / ".git").is_dir(),
        "download": lambda: load(DATASET / "download_receipt.json").get("status") == "complete",
        "prepare": lambda: load(DATA / "manifest.json").get("total_episodes") == 550 and load(DATA / "manifest.json").get("total_policy_samples") == 285438,
        "normalization": lambda: load(RUN / "assets/norm_stats.json").get("schema") == "duobench.pi05.exact-normalization.v1",
        "audit": lambda: load(RUN / "audit.json").get("passed") is True,
        "preflight": lambda: load(RUN / "preflight.json").get("passed") is True,
        "smoke_train": lambda: load(RUN / "smoke/status.json").get("status") == "complete" and load(RUN / "smoke/status.json").get("updates") == 2 and Path(load(RUN / "smoke/status.json").get("checkpoint", "")).is_dir(),
        "smoke_isolation": lambda: load(RUN / "smoke/isolation.json").get("passed") is True,
        "smoke_validation": lambda: load(RUN / "smoke/validation/summary.json").get("total_episodes") == 11,
        "formal_train": lambda: load(RUN / "formal/status.json").get("status") == "complete" and load(RUN / "formal/status.json").get("updates") == 25000 and Path(load(RUN / "formal/status.json").get("checkpoint", "")).is_dir(),
        "formal_isolation": lambda: load(RUN / "formal/isolation.json").get("passed") is True,
        "validation20": lambda: load(RUN / "formal/validation20/summary.json").get("status") == "complete" and load(RUN / "formal/validation20/summary.json").get("total_episodes") == 220,
        "finalize": lambda: load(RUN / "final_report.json").get("status") == "complete",
    }
    return checks[stage]()


def receipt(stage: str) -> Path: return RUN / "receipts" / f"{stage}.json"
def completed(stage: str) -> bool: return load(receipt(stage)).get("status") == "complete" and valid(stage)


def gpu_pids() -> list[int]:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    values = []
    for line in result.stdout.splitlines():
        try: values.append(int(line.strip()))
        except ValueError: pass
    return values


def run_stage(stage: str, command: list[str], *, gpus: tuple[int, ...], attempts: int = 3) -> None:
    global ACTIVE
    if completed(stage): return
    log = RUN / "logs" / f"{stage}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    lock = RUN / "gpu.lock" if gpus else None
    for attempt in range(1, attempts + 1):
        lock_file = None
        try:
            if lock is not None:
                lock.parent.mkdir(parents=True, exist_ok=True); lock_file = lock.open("a+"); fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                while gpu_pids() and not STOP:
                    write_state(stage, "waiting_for_gpu", gpus=list(gpus), foreign_gpu_pids=gpu_pids(), attempt=attempt); time.sleep(15)
            env = os.environ.copy(); env.update({"PYTHONPATH": f"{REPO}:{OPENPI / 'src'}:{DUO / 'src'}:{RCS / 'python'}", "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)) if gpus else "", "OPENPI_DUOBENCH_ROOT": str(DATA), "DUO_PI05_CHECKPOINT": str(RUN / "formal/final"), "DUOBENCH_PREFIX": "/workspace/datasets/duobench_assets", "HF_HOME": "/workspace/.hf_home", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_ALLOCATOR": "platform", "MUJOCO_GL": "egl", "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
            write_state(stage, attempt=attempt, command=command, gpus=list(gpus), log=str(log))
            with log.open("ab", buffering=0) as stream:
                stream.write((json.dumps({"event": "launch", "attempt": attempt, "command": command, "time": now()}) + "\n").encode())
                ACTIVE = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True); code = ACTIVE.wait(); ACTIVE = None
            if code == 0 and valid(stage):
                atomic_json(receipt(stage), {"schema": "duobench.pi05.stage.v1", "status": "complete", "stage": stage, "attempt": attempt, "completed_at": now(), "log": str(log)}); return
            error = f"exit={code}, artifact_valid={valid(stage)}"
        except Exception as exc:
            ACTIVE = None; error = repr(exc)
        finally:
            if lock_file is not None: fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN); lock_file.close()
        write_state(stage, "retrying", attempt=attempt, error=error, log=str(log));
        if attempt < attempts: time.sleep(min(180, 20 * attempt))
    raise RuntimeError(f"stage {stage} failed after {attempts} attempts: {error}")


def handle(_signum, _frame) -> None:
    global STOP
    STOP = True
    if ACTIVE is not None:
        try: os.killpg(ACTIVE.pid, signal.SIGTERM)
        except ProcessLookupError: pass


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True); signal.signal(signal.SIGTERM, handle); signal.signal(signal.SIGINT, handle)
    smoke = RUN / "smoke"; formal = RUN / "formal"
    smoke_checkpoint = smoke / "checkpoints/pi05_duobench_lora/smoke/1"
    formal_checkpoint = formal / "checkpoints/pi05_duobench_lora/all550_4gpu_dp_b128_25k/24999"
    stages = [
        ("environment", [PYTHON, "-c", f"import torch,jax; assert torch.cuda.device_count()==4 and len(jax.devices())==4; print(jax.devices(), torch.cuda.get_device_name(0))"], (0, 1, 2, 3), 3),
        ("download", [PYTHON, "-m", "deployment.duo_pi05.download", "--output", str(DATASET), "--workers", "24"], (), 12),
        ("prepare", [PYTHON, "-m", "deployment.duo_act.prepare", "--dataset", str(DATASET), "--output", str(DATA), "--image-size", "224", "--jobs", "11"], (), 4),
        ("normalization", [PYTHON, "-m", "deployment.duo_pi05.compute_norm", "--data", str(DATA), "--output", str(RUN / "assets/norm_stats.json")], (), 4),
        ("audit", [PYTHON, "-m", "deployment.duo_pi05.audit_contract", "--data", str(DATA), "--norm", str(RUN / "assets/norm_stats.json"), "--output", str(RUN / "audit.json")], (), 3),
        ("preflight", [PYTHON, "-m", "deployment.duo_pi05.preflight", "--data", str(DATA), "--norm", str(RUN / "assets/norm_stats.json"), "--output", str(RUN / "preflight.json")], (0, 1, 2, 3), 3),
        ("smoke_train", [PYTHON, "-m", "deployment.duo_pi05.train_stage", "--openpi", str(OPENPI), "--checkpoint-base-dir", str(smoke / "checkpoints"), "--assets-base-dir", str(smoke / "assets"), "--exp-name", "smoke", "--updates", "2", "--workers", "0", "--save-every", "2", "--keep-period", "2", "--smoke"], (0, 1, 2, 3), 4),
        ("smoke_isolation", [PYTHON, "-m", "deployment.duo_pi05.isolation_audit", "--checkpoint", str(smoke_checkpoint), "--output", str(smoke / "isolation.json")], (0,), 3),
        ("smoke_validation", [PYTHON, "-m", "deployment.duo_pi05.validation_launcher", "--checkpoint", str(smoke_checkpoint), "--data", str(DATA), "--output", str(smoke / "validation"), "--episodes", "1", "--workers", "4", "--smoke"], (0, 1, 2, 3), 4),
        ("formal_train", [PYTHON, "-m", "deployment.duo_pi05.train_stage", "--openpi", str(OPENPI), "--checkpoint-base-dir", str(formal / "checkpoints"), "--assets-base-dir", str(formal / "assets"), "--exp-name", "all550_4gpu_dp_b128_25k", "--updates", "25000", "--workers", "12", "--save-every", "1000", "--keep-period", "5000"], (0, 1, 2, 3), 20),
        ("formal_isolation", [PYTHON, "-m", "deployment.duo_pi05.isolation_audit", "--checkpoint", str(formal_checkpoint), "--output", str(formal / "isolation.json")], (0,), 3),
        ("validation20", [PYTHON, "-m", "deployment.duo_pi05.validation_launcher", "--checkpoint", str(formal_checkpoint), "--data", str(DATA), "--output", str(formal / "validation20"), "--episodes", "20", "--workers", "4"], (0, 1, 2, 3), 20),
        ("finalize", [PYTHON, "-m", "deployment.duo_pi05.finalize"], (), 3),
    ]
    try:
        for stage, command, gpus, attempts in stages:
            if STOP: return
            run_stage(stage, command, gpus=gpus, attempts=attempts)
        write_state("complete", "complete", final_report=str(RUN / "final_report.json"))
    except KeyboardInterrupt:
        write_state("stopped", "stopped")
    except Exception as error:
        write_state("failed", "failed", error=repr(error), traceback=traceback.format_exc()); raise


if __name__ == "__main__": main()
