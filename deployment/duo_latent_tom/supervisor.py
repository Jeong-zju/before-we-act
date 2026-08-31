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

from .common import FROZEN_CONFIG, POLICY_CONTRACT, atomic_json


REPO = Path(os.environ.get("DUO_LATENT_TOM_REPO", "/workspace/repos/before-we-act"))
RUN = Path(os.environ.get("DUO_LATENT_TOM_RUN", "/workspace/runs/duobench-latent-tom"))
DATASET = Path(os.environ.get("DUO_LATENT_TOM_DATASET", "/workspace/datasets/duobench"))
DATA = Path(os.environ.get("DUO_LATENT_TOM_PREPARED", "/workspace/runs/duobench-latent-tom/data"))
PYTHON = os.environ.get("DUO_LATENT_TOM_PYTHON", "/venv/main/bin/python")
LATENT = Path(os.environ.get("LATENT_TOM_ROOT", "/workspace/repos/LatentToM"))
DUO = Path(os.environ.get("DUOBENCH_ROOT", "/workspace/repos/duobench"))
RCS = Path(os.environ.get("RCS_ROOT", "/workspace/repos/robot-control-stack"))
ACTIVE: subprocess.Popen | None = None
STOP = False


def now() -> str: return datetime.now(timezone.utc).isoformat()


def write_state(stage: str, status: str = "running", **extra) -> None:
    atomic_json(RUN / "state.json", {"schema": "duobench.latent-tom.supervisor.v1", "stage": stage, "status": status, "updated_at": now(), "gpu_schedule": "exclusive GPU 0 for train/inference; CPU-only data stages; no foreign-process termination", "policy_contract": POLICY_CONTRACT, **extra})


def load_json(path: Path) -> dict:
    try: return json.loads(path.read_text())
    except Exception: return {}


def valid(stage: str) -> bool:
    tests = {
        "environment_preflight": lambda: True,
        "download": lambda: load_json(DATASET / "download_receipt.json").get("status") == "complete",
        "prepare": lambda: load_json(DATA / "manifest.json").get("total_episodes") == 550 and "qpos_min" in load_json(DATA / "manifest.json").get("normalization", {}),
        "data_audit": lambda: load_json(RUN / "audit.json").get("passed") is True,
        "interface_preflight": lambda: load_json(RUN / "preflight.json").get("passed") is True,
        "smoke_train": lambda: load_json(RUN / "smoke/smoke_status.json").get("step") == 2 and (RUN / "smoke/final.pt").is_file(),
        "smoke_checkpoint": lambda: load_json(RUN / "smoke/checkpoint_smoke.json").get("status") == "complete",
        "smoke_isolation": lambda: load_json(RUN / "smoke/isolation.json").get("status") == "complete",
        "smoke_validation": lambda: load_json(RUN / "smoke/validation/summary.json").get("total_episodes") == 11,
        "formal_train": lambda: load_json(RUN / "formal/status.json").get("step") == 60000 and (RUN / "formal/final.pt").is_file(),
        "formal_checkpoint": lambda: load_json(RUN / "formal/checkpoint_smoke.json").get("checkpoint_step") == 60000,
        "formal_isolation": lambda: load_json(RUN / "formal/isolation.json").get("checkpoint_step") == 60000,
        "validation20": lambda: load_json(RUN / "formal/validation20/summary.json").get("total_episodes") == 220 and load_json(RUN / "formal/validation20/summary.json").get("diffusion_steps") == 100,
        "finalize": lambda: load_json(RUN / "final_report.json").get("status") == "complete",
    }
    return bool(tests[stage]())


def receipt(stage: str) -> Path: return RUN / "receipts" / f"{stage}.json"


def completed(stage: str) -> bool:
    return load_json(receipt(stage)).get("status") == "complete" and valid(stage)


def foreign_gpu_pids() -> list[int]:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False)
    values = []
    for line in result.stdout.splitlines():
        try: values.append(int(line.strip()))
        except ValueError: pass
    return values


def run_stage(stage: str, command: list[str], *, gpu: bool, attempts: int) -> None:
    global ACTIVE
    if completed(stage): return
    log = RUN / "logs" / f"{stage}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": f"{REPO}:{LATENT}:{DUO / 'src'}", "CUDA_VISIBLE_DEVICES": "0", "MUJOCO_GL": "egl", "DUOBENCH_PREFIX": str(DUO), "HF_HOME": "/workspace/.hf_home", "HF_XET_HIGH_PERFORMANCE": "1", "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "LD_LIBRARY_PATH": f"/venv/main/lib/python3.12/site-packages/mujoco:{environment.get('LD_LIBRARY_PATH', '')}"})
    for attempt in range(1, attempts + 1):
        if STOP: raise KeyboardInterrupt
        lock_stream = None
        try:
            if gpu:
                lock_path = RUN / "gpu0.lock"; lock_path.parent.mkdir(parents=True, exist_ok=True)
                lock_stream = lock_path.open("a+"); fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                while foreign_gpu_pids() and not STOP:
                    write_state(stage, "waiting_for_gpu", foreign_gpu_pids=foreign_gpu_pids(), attempt=attempt); time.sleep(30)
            write_state(stage, attempt=attempt, command=command, log=str(log), gpu=[0] if gpu else [])
            with log.open("ab", buffering=0) as stream:
                stream.write((json.dumps({"event": "launch", "attempt": attempt, "command": command, "time": now()}) + "\n").encode())
                ACTIVE = subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                code = ACTIVE.wait(); ACTIVE = None
            if code == 0 and valid(stage):
                atomic_json(receipt(stage), {"schema": "duobench.latent-tom.stage.v1", "status": "complete", "stage": stage, "completed_at": now(), "attempt": attempt, "log": str(log)})
                return
            error = f"exit={code}, artifact_valid={valid(stage)}"
        except Exception as exc:
            ACTIVE = None; error = repr(exc)
        finally:
            if lock_stream is not None:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN); lock_stream.close()
        write_state(stage, "retrying", attempt=attempt, error=error, log=str(log))
        if attempt < attempts: time.sleep(min(300, 20 * attempt))
    raise RuntimeError(f"{stage} failed after {attempts} attempts: {error}")


def handle(_signum, _frame) -> None:
    global STOP
    STOP = True
    if ACTIVE is not None:
        try: os.killpg(ACTIVE.pid, signal.SIGTERM)
        except ProcessLookupError: pass


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True); signal.signal(signal.SIGTERM, handle); signal.signal(signal.SIGINT, handle)
    formal = RUN / "formal/final.pt"
    stages = [
        ("environment_preflight", [PYTHON, "-c", "import torch,torchvision,diffusers,av,pyarrow,mujoco,gymnasium,rcs,duobench; assert torch.cuda.is_available() and torch.cuda.device_count()==1 and not torch.backends.cudnn.deterministic; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))"], True, 3),
        ("download", [PYTHON, "-u", "-m", "deployment.duo_latent_tom.download", "--output", str(DATASET), "--workers", "24"], False, 12),
        ("prepare", [PYTHON, "-u", "-m", "deployment.duo_act.prepare", "--dataset", str(DATASET), "--output", str(DATA), "--image-size", "224", "--jobs", "11"], False, 4),
        ("data_audit", [PYTHON, "-u", "-m", "deployment.duo_act.audit", "--data", str(DATA), "--output", str(RUN / "audit.json")], False, 3),
        ("interface_preflight", [PYTHON, "-u", "-m", "deployment.duo_latent_tom.preflight", "--data", str(DATA), "--output", str(RUN / "preflight.json")], True, 3),
        ("smoke_train", [PYTHON, "-u", "-m", "deployment.duo_latent_tom.train", "--config", str(FROZEN_CONFIG), "--data", str(DATA), "--output", str(RUN / "smoke"), "--smoke"], True, 3),
        ("smoke_checkpoint", [PYTHON, "-u", "-m", "deployment.duo_latent_tom.checkpoint_smoke", "--checkpoint", str(RUN / "smoke/final.pt"), "--output", str(RUN / "smoke/checkpoint_smoke.json")], True, 3),
        ("smoke_isolation", [PYTHON, "-u", "-m", "deployment.duo_latent_tom.audit_isolation", "--checkpoint", str(RUN / "smoke/final.pt"), "--output", str(RUN / "smoke/isolation.json")], True, 3),
        ("smoke_validation", [PYTHON, "-u", "-m", "deployment.duo_latent_tom.validation_launcher", "--checkpoint", str(RUN / "smoke/final.pt"), "--data", str(DATA), "--output", str(RUN / "smoke/validation"), "--episodes", "1", "--workers", "3", "--smoke"], True, 3),
        ("formal_train", [PYTHON, "-u", "-m", "deployment.duo_latent_tom.train", "--config", str(FROZEN_CONFIG), "--data", str(DATA), "--output", str(RUN / "formal"), "--resume"], True, 20),
        ("formal_checkpoint", [PYTHON, "-u", "-m", "deployment.duo_latent_tom.checkpoint_smoke", "--checkpoint", str(formal), "--output", str(RUN / "formal/checkpoint_smoke.json")], True, 3),
        ("formal_isolation", [PYTHON, "-u", "-m", "deployment.duo_latent_tom.audit_isolation", "--checkpoint", str(formal), "--output", str(RUN / "formal/isolation.json")], True, 3),
        ("validation20", [PYTHON, "-u", "-m", "deployment.duo_latent_tom.validation_launcher", "--checkpoint", str(formal), "--data", str(DATA), "--output", str(RUN / "formal/validation20"), "--episodes", "20", "--workers", "3"], True, 20),
        ("finalize", [PYTHON, "-u", "-m", "deployment.duo_latent_tom.finalize", "--run", str(RUN)], False, 3),
    ]
    try:
        for stage, command, gpu, attempts in stages: run_stage(stage, command, gpu=gpu, attempts=attempts)
        write_state("complete", "complete", final_report=str(RUN / "final_report.json"))
    except KeyboardInterrupt:
        write_state("stopped", "stopped"); raise
    except Exception as error:
        write_state("failed", "failed", error=repr(error), traceback=traceback.format_exc()); raise


if __name__ == "__main__": main()
