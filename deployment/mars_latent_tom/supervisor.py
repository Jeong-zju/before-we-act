from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .common import FROZEN_CONFIG, POLICY_CONTRACT, atomic_json


RUN = Path(os.environ.get("MARS_LATENT_TOM_RUN_ROOT", "/workspace/runs/mars_latent_tom"))
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("MARS_LATENT_TOM_DATA_ROOT", "/workspace/datasets/mars_control"))
PYTHON = os.environ.get("MARS_LATENT_TOM_PYTHON", "/venv/main/bin/python")
RF = Path(os.environ.get("ROBOFACTORY_ROOT", "/workspace/repos/RoboFactory"))
stop = False; active = None


def now(): return datetime.now(timezone.utc).isoformat()
def state(stage, status="running", **extra): atomic_json(RUN / "state.json", {"schema": "mars-control.latent-tom.supervisor.v1", "stage": stage, "status": status, "updated_at": now(), **extra})
def receipt(name): return RUN / "receipts" / f"{name}.json"
def completed(name):
    try: return json.loads(receipt(name).read_text()).get("status") == "complete"
    except Exception: return False


def run(name, command, env=None):
    global active
    log = RUN / "logs" / f"{name}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy(); merged.update({"PYTHONPATH": f"{ROOT}:{RF}:/workspace/repos/LatentToM", "CUDA_VISIBLE_DEVICES": "0", "HF_HOME": "/workspace/.hf_home", "TOKENIZERS_PARALLELISM": "false"}); merged.update(env or {})
    state(name, command=command, log=str(log), gpu=[0], contract=POLICY_CONTRACT)
    with log.open("ab", buffering=0) as stream:
        active = subprocess.Popen(command, cwd=ROOT, env=merged, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        code = active.wait()
    active = None
    if code: raise RuntimeError(f"{name} exited with code {code}")
    atomic_json(receipt(name), {"schema": "mars-control.latent-tom.stage.v1", "status": "complete", "stage": name, "completed_at": now(), "log": str(log)})


def handle(_signum, _frame):
    global stop
    stop = True
    if active:
        try: os.killpg(active.pid, signal.SIGTERM)
        except ProcessLookupError: pass


def main():
    global stop
    RUN.mkdir(parents=True, exist_ok=True); signal.signal(signal.SIGTERM, handle); signal.signal(signal.SIGINT, handle)
    stages = [
        ("preflight", [PYTHON, "-c", "import torch,h5py; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.__version__,torch.cuda.get_device_name(0))"]),
        ("download", [PYTHON, "-m", "deployment.mars_latent_tom.download", "--data-root", str(DATA)]),
        ("audit_and_stats", [PYTHON, "-m", "deployment.mars_latent_tom.audit", "--data-root", str(DATA), "--output", str(RUN / "audit.json"), "--stats", str(RUN / "normalization.json")]),
        ("smoke_train", [PYTHON, "-m", "deployment.mars_latent_tom.train", "--config", str(FROZEN_CONFIG), "--data-root", str(DATA), "--stats", str(RUN / "normalization.json"), "--output", str(RUN / "smoke"), "--smoke"]),
        ("smoke_validation", [PYTHON, "-m", "deployment.mars_latent_tom.evaluate", "--config", str(FROZEN_CONFIG), "--checkpoint", str(RUN / "smoke/last.pt"), "--output", str(RUN / "smoke/validation20"), "--config-root", str(RF / "configs/table"), "--smoke"]),
        ("formal_train", [PYTHON, "-m", "deployment.mars_latent_tom.train", "--config", str(FROZEN_CONFIG), "--data-root", str(DATA), "--stats", str(RUN / "normalization.json"), "--output", str(RUN / "formal"), "--resume"]),
        ("validation20", [PYTHON, "-m", "deployment.mars_latent_tom.evaluate", "--config", str(FROZEN_CONFIG), "--checkpoint", str(RUN / "formal/last.pt"), "--output", str(RUN / "formal/validation20"), "--config-root", str(RF / "configs/table")]),
    ]
    while not stop:
        progressed = False
        for name, command in stages:
            if stop: break
            if completed(name): continue
            progressed = True
            while not stop:
                try: run(name, command); break
                except Exception as exc: state(name, "retrying", error=repr(exc)); time.sleep(20)
        if not progressed: state("complete", "complete"); break
    state("stopped" if stop else "complete", "stopped" if stop else "complete")


if __name__ == "__main__": main()
