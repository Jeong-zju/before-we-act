from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .common import FROZEN_CONFIG, atomic_json

RUN = Path(os.environ.get("MARS_GAUDP_RUN_ROOT", "/workspace/runs/mars_gaudp_fp32_v2"))
ROOT = Path(os.environ.get("MARS_GAUDP_REPO", "/workspace/repos/before-we-act"))
RF = Path(os.environ.get("MARS_GAUDP_ROBOFACTORY", "/workspace/repos/RoboFactory"))
DATA = Path(os.environ.get("MARS_GAUDP_DATA_ROOT", "/workspace/datasets/mars_control"))
CACHE = Path(os.environ.get("MARS_GAUDP_CACHE_ROOT", str(RUN / "cache")))
WEIGHT = Path(os.environ.get("MARS_GAUDP_WEIGHT", "/workspace/repos/Policy-Lightning/weights/re10k.ckpt"))
PY = os.environ.get("MARS_GAUDP_PYTHON", "/venv/main/bin/python")
active = None
stopping = False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state(stage: str, status: str = "running", **kwargs) -> None:
    atomic_json(RUN / "state.json", {"schema": "mars-control.gaudp.supervisor.v2", "stage": stage, "status": status, "updated_at": now(), **kwargs})


def done(name: str) -> bool:
    try:
        return json.loads((RUN / "receipts" / f"{name}.json").read_text()).get("status") == "complete"
    except Exception:
        return False


def run(name: str, cmd: list[str], env: dict[str, str] | None = None) -> None:
    global active
    log = RUN / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    merged.update({
        "PYTHONPATH": f"{ROOT}:/workspace/repos/Policy-Lightning:{RF}/robofactory/policy/Diffusion-Policy",
        "MARS_GAUDP_DATA_ROOT": str(DATA),
        "MARS_GAUDP_CACHE_ROOT": str(CACHE),
        "MARS_GAUDP_RUN_ROOT": str(RUN),
        "MARS_GAUDP_WEIGHT": str(WEIGHT),
        "CUDA_VISIBLE_DEVICES": "0",
        "TOKENIZERS_PARALLELISM": "false",
    })
    merged.update(env or {})
    state(name, command=cmd, log=str(log), gpu=[0])
    with log.open("ab", buffering=0) as stream:
        active = subprocess.Popen(cmd, cwd=ROOT, env=merged, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        code = active.wait()
    active = None
    if code:
        raise RuntimeError(f"{name} exited with code {code}")
    atomic_json(RUN / "receipts" / f"{name}.json", {"stage": name, "status": "complete", "completed_at": now(), "log": str(log)})


def sig(_signum, _frame) -> None:
    global stopping
    stopping = True
    if active:
        try:
            os.killpg(active.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def main() -> None:
    global stopping
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "receipts").mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM, sig)
    signal.signal(signal.SIGINT, sig)
    stages = [
        ("verify_frozen_config", [PY, "-m", "deployment.mars_gaudp.verify_frozen_config", "--config", str(FROZEN_CONFIG)]),
        ("preflight", [PY, "-c", "import torch,h5py,diffusers; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.__version__,torch.cuda.get_device_name(0))"]),
        ("precompute_cache", [PY, "-m", "deployment.mars_gaudp.precompute", "--data-root", str(DATA), "--cache-root", str(CACHE), "--weight", str(WEIGHT), "--batch-size", "120", "--output-hw", "30", "40"]),
        ("cache_parity", [PY, "-m", "deployment.mars_gaudp.cache_parity", "--data-root", str(DATA), "--cache-root", str(CACHE), "--weight", str(WEIGHT), "--output", str(RUN / "cache_parity.json")]),
        ("audit", [PY, "-m", "deployment.mars_gaudp.audit"]),
        ("smoke_train", [PY, "-m", "deployment.mars_gaudp.train", "--data-root", str(DATA), "--cache-root", str(CACHE), "--output", str(RUN / "smoke"), "--smoke", "--batch-size", "8", "--workers", "0"]),
        ("smoke_eval", [PY, "-m", "deployment.mars_gaudp.run_validation", "--checkpoint", str(RUN / "smoke" / "last.pt"), "--noposplat-weight", str(WEIGHT), "--output-root", str(RUN / "smoke" / "validation"), "--robofactory-root", str(RF), "--inference-steps", "100", "--smoke"]),
        ("formal_train", [PY, "-m", "deployment.mars_gaudp.train", "--data-root", str(DATA), "--cache-root", str(CACHE), "--output", str(RUN / "formal"), "--steps", "60000", "--batch-size", "64", "--workers", "16"]),
        ("verify_checkpoint", [PY, "-m", "deployment.mars_gaudp.verify_frozen_config", "--config", str(FROZEN_CONFIG), "--checkpoint", str(RUN / "formal" / "last.pt")]),
        ("inference_compare", [PY, "-m", "deployment.mars_gaudp.compare_inference", "--checkpoint", str(RUN / "formal" / "last.pt"), "--noposplat-weight", str(WEIGHT), "--robofactory-root", str(RF), "--output", str(RUN / "inference_comparison.json"), "--ensemble-decay", "0.01"]),
        ("validation20", [PY, "-m", "deployment.mars_gaudp.run_validation", "--checkpoint", str(RUN / "formal" / "last.pt"), "--noposplat-weight", str(WEIGHT), "--output-root", str(RUN / "validation20"), "--robofactory-root", str(RF), "--selection-file", str(RUN / "inference_comparison.json"), "--ensemble-decay", "0.01"]),
        ("finalize", [PY, "-m", "deployment.mars_gaudp.finalize"]),
    ]
    while not stopping:
        progressed = False
        for name, cmd in stages:
            if stopping:
                break
            if done(name):
                continue
            progressed = True
            while not stopping:
                try:
                    run(name, cmd)
                    break
                except Exception as exc:
                    state(name, "retrying", error=repr(exc))
                    time.sleep(30)
        if not progressed and not stopping:
            state("complete", "complete")
            time.sleep(60)
    state("stopped", "stopped")


if __name__ == "__main__":
    main()
