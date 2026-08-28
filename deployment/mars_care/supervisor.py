from __future__ import annotations

import json, os, signal, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .common import TASKS
from .dataset import audit_raw


ROOT = Path(os.environ.get("MARS_CARE_REPO", "/workspace/repos/before-we-act"))
RF = Path(os.environ.get("MARS_ROBOFACTORY_ROOT", "/workspace/repos/RoboFactory"))
PY = os.environ.get("MARS_PYTHON", "/workspace/venvs/mars/bin/python")
RAW = Path(os.environ.get("MARS_DATA_ROOT", "/workspace/datasets/mars_control/raw"))
RUN = Path(os.environ.get("MARS_CARE_RUN_ROOT", "/workspace/runs/mars_care"))
REPAIR = RUN / "repair"
LOG = RUN / "logs"; STATUS = RUN / "status.json"; NORM = REPAIR / "data/normalization.json"
LEGACY = RUN / "formal/final.pt"
active: set[subprocess.Popen] = set(); stop_requested = False


def now(): return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(value, indent=2) + "\n"); os.replace(tmp, path)


def status(state: str, stage: str, detail: str, **extra):
    current = {}
    try: current = json.loads(STATUS.read_text())
    except Exception: pass
    atomic_json(STATUS, {**current, "schema": "mars-care-supervisor-v1", "state": state, "stage": stage, "detail": detail, "updated_at": now(), "pid": os.getpid(), **extra})


def heartbeat():
    while not stop_requested:
        try:
            current = json.loads(STATUS.read_text()); current["heartbeat_at"] = now(); atomic_json(STATUS, current)
        except Exception: pass
        time.sleep(20)


def environment(extra=None):
    value = dict(os.environ); value.update({"PYTHONPATH": f"{ROOT}:{RF}:{value.get('PYTHONPATH','')}", "HF_HOME": "/workspace/.hf_home", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8", "WANDB_MODE": "disabled"})
    if extra: value.update(extra)
    return value


def run(name: str, argv: list[str], cwd: Path = ROOT, extra_env=None):
    LOG.mkdir(parents=True, exist_ok=True); path = LOG / f"{name}.log"
    with path.open("a") as stream:
        stream.write(f"\n[{now()}] RUN {argv!r}\n"); stream.flush()
        process = subprocess.Popen(argv, cwd=cwd, env=environment(extra_env), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True); active.add(process)
        code = process.wait(); active.discard(process)
    if code: raise RuntimeError(f"{name} exited {code}; see {path}")


def shard_ok(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            names = [x for x in handle if x.startswith("traj_")]
            return len(names) == 15 and all(bool(np.asarray(handle[x]["success"])[-1]) for x in names)
    except Exception: return False


def collection_progress() -> tuple[int, int]:
    complete = staged = 0
    for task in TASKS:
        directory = RAW / task.env_id / "motionplanning"
        for shard in range(10):
            name = f"{task.name}.shard{shard:02d}"
            if shard_ok(directory / f"{name}.h5"):
                complete += 15
                continue
            parts = directory / f".{name}.parts"
            for path in parts.glob("seed_*.h5"):
                try:
                    with h5py.File(path, "r") as handle:
                        names = [key for key in handle if key.startswith("traj_")]
                        staged += int(len(names) == 1 and bool(np.asarray(handle[names[0]]["success"])[-1]))
                except Exception:
                    pass
    return complete, staged


def collect_data():
    commands = []
    for task_id, task in enumerate(TASKS):
        directory = RAW / task.env_id / "motionplanning"; directory.mkdir(parents=True, exist_ok=True)
        for shard in range(10):
            target = directory / f"{task.name}.shard{shard:02d}.h5"
            if shard_ok(target): continue
            name = f"data_{task.name}_{shard:02d}"
            argv = [PY, "-m", "deployment.mars_care.expert_shard", "--config", task.config, "--episodes", "15", "--seed", str(task_id * 100000 + shard * 1000), "--record-dir", str(RAW), "--name", f"{task.name}.shard{shard:02d}", "--render-device", "DEVICE", "--attempt-timeout", os.environ.get("MARS_EXPERT_ATTEMPT_TIMEOUT", "180")]
            commands.append((name, target, argv))
    # The SAPIEN 3.0 Vulkan backend in this container is process-global in
    # practice: even explicit cuda:0/cuda:1 contexts can lose the device.
    for wave_start in range(0, len(commands), 1):
        jobs = []
        for slot, (name, target, argv) in enumerate(commands[wave_start:wave_start + 1]):
            argv = [f"cuda:{slot}" if value == "DEVICE" else value for value in argv]
            log = (LOG / f"{name}.log").open("a")
            process = subprocess.Popen(argv, cwd=RF, env=environment({"CUDA_VISIBLE_DEVICES": str(slot % 2)}), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active.add(process); jobs.append((name, target, process, log))
        failures = []
        for name, target, process, log in jobs:
            code = process.wait(); active.discard(process); log.close()
            if code or not shard_ok(target): failures.append(f"{name}:exit={code}:valid={shard_ok(target)}")
        if failures: raise RuntimeError("data generation failures: " + ", ".join(failures))
        complete, staged = collection_progress()
        status("running", "data_generation", f"official experts: {complete}/600 complete, {staged} staged", trajectories_complete=complete, trajectories_staged=staged, trajectories_target=600)


def validate_all():
    for task_id, task in enumerate(TASKS):
        output = REPAIR / "validation20" / f"{task.name}.json"
        try:
            saved = json.loads(output.read_text())
            if saved.get("episodes") == 20 and saved.get("policy_runtime") == "temporal_ensemble_v2": continue
        except Exception: pass
        argv = [PY, "-m", "deployment.mars_care.evaluate", "--checkpoint", str(REPAIR / "formal/final.pt"), "--task", task.name, "--robofactory-root", str(RF), "--output", str(output), "--episodes", "20", "--seed-start", str(20260824 + task_id * 1000)]
        log = (LOG / f"repair_validation20_{task.name}.log").open("a")
        process = subprocess.Popen(argv, cwd=ROOT, env=environment({"CUDA_VISIBLE_DEVICES": "0"}), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active.add(process); code = process.wait(); active.discard(process); log.close()
        if code: raise RuntimeError(f"validation20 {task.name} exited {code}")


def handle_signal(_number, _frame):
    global stop_requested; stop_requested = True; status("stopping", "signal", "graceful stop requested")
    for process in list(active):
        try: os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError: pass


def main():
    RUN.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, handle_signal); signal.signal(signal.SIGINT, handle_signal)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        status("running", "preflight", "checking official commit, environment and GPUs")
        if subprocess.check_output(["git", "-C", str(RF), "rev-parse", "HEAD"], text=True).strip() != "2d34fb38c80cb06550a5dbf99abac2c89f4336ed": raise RuntimeError("RoboFactory official commit drift")
        run("preflight", [PY, "-c", "import torch,mani_skill,h5py; assert torch.cuda.device_count()==2; print(torch.__version__,torch.cuda.get_device_name(0))"], RF)
        marker = RUN / "assets.complete"
        if not marker.is_file():
            required = ("assets/scenes/table/table.glb", "assets/objects/teamug_annotated/teamugobj.obj", "assets/objects/hammer_annotated/base.glb", "assets/objects/single_shoe_annotated/base.glb", "assets/objects/box/mobility.urdf")
            if not all((RF / path).is_file() and (RF / path).stat().st_size for path in required):
                status("running", "assets", "downloading official RoboFactory assets")
                run("assets", [PY, "script/download_assets.py"], RF)
            marker.write_text(now() + "\n")
        try: audit_raw(RAW)
        except Exception:
            status("running", "data_generation", "official experts: 4 tasks x 150 successful trajectories", trajectories_target=600)
            collect_data()
        status("running", "repair_data_audit", "v2 residual-action normalization over all 600 trajectories", repair="temporal_ensemble+residual+local_history+task_balance")
        if not NORM.is_file():
            run("repair_data_audit", [PY, "-m", "deployment.mars_care.dataset", "--raw-root", str(RAW), "--audit-output", str(REPAIR / "data/audit.json"), "--normalization", str(NORM)])
        smoke_checkpoint = REPAIR / "smoke/final.pt"
        if not smoke_checkpoint.is_file():
            status("running", "repair_training_smoke", "10k all-data DDP repair smoke on GPU0+GPU1", gpus=[0, 1], steps=10000)
            run("repair_training_smoke", [PY, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "-m", "deployment.mars_care.train", "--raw-root", str(RAW), "--normalization", str(NORM), "--output", str(REPAIR / "smoke"), "--steps", "10000", "--batch-size", "96", "--workers", "12", "--save-every", "5000", "--init-checkpoint", str(LEGACY), "--init-vision-only"])
        smoke_report = REPAIR / "smoke/closed_loop.json"
        if not smoke_report.is_file():
            status("running", "repair_closed_loop_smoke", "strict-local audit and one successful closed loop per task")
            run("repair_closed_loop_smoke", [PY, "-m", "deployment.mars_care.smoke", "--checkpoint", str(smoke_checkpoint), "--robofactory-root", str(RF), "--output", str(smoke_report), "--closed-loop"], extra_env={"CUDA_VISIBLE_DEVICES": "0"})
        if not (REPAIR / "formal/final.pt").is_file():
            status("running", "repair_formal_training", "repaired CARE all-data DDP training on GPU0+GPU1", gpus=[0, 1], steps=100000)
            run("repair_formal_training", [PY, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "-m", "deployment.mars_care.train", "--raw-root", str(RAW), "--normalization", str(NORM), "--output", str(REPAIR / "formal"), "--steps", "100000", "--batch-size", "96", "--workers", "12", "--save-every", "5000", "--init-checkpoint", str(smoke_checkpoint)])
        status("running", "repair_validation20", "four tasks x 20 repaired closed-loop episodes", gpus=[0], episodes_target=80)
        validate_all()
        results = [json.loads((REPAIR / "validation20" / f"{task.name}.json").read_text()) for task in TASKS]
        summary = {"status": "complete", "completed_at": now(), "checkpoint": str(REPAIR / "formal/final.pt"), "episodes": 80, "successes": sum(x["successes"] for x in results), "macro_success_rate": float(np.mean([x["success_rate"] for x in results])), "repair": "temporal_ensemble+residual+local_history+task_balance", "tasks": {x["task"]: {"successes": x["successes"], "episodes": x["episodes"], "success_rate": x["success_rate"]} for x in results}}
        atomic_json(REPAIR / "validation20/summary.json", summary); status("complete", "complete", "repaired CARE training and validation20 complete", summary=str(REPAIR / "validation20/summary.json"))
    except Exception as error:
        status("failed", "failed", repr(error)); raise


if __name__ == "__main__": main()
