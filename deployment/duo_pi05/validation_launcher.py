from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .common import EVALUATOR_REVISION, POLICY_CONTRACT, TASKS, atomic_json, checkpoint_identity


def stop_group(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass


def wait_socket(path: Path, process: subprocess.Popen, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    while not path.is_socket():
        if process.poll() is not None: raise RuntimeError(f"policy worker exited {process.returncode}")
        if time.monotonic() > deadline: raise TimeoutError("policy worker startup timeout")
        time.sleep(1)


def task_complete(path: Path, episodes: int, checkpoint_sha: str) -> bool:
    try:
        row = json.loads(path.read_text())
        return row.get("status") == "complete" and row.get("episodes") == episodes and row.get("checkpoint_sha256") == checkpoint_sha and row.get("evaluator_revision") == EVALUATOR_REVISION
    except (FileNotFoundError, json.JSONDecodeError): return False


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--data", required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--episodes", type=int, default=20); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    if args.workers < 1 or args.workers > 4: raise ValueError("workers must be in [1,4]")
    args.output.mkdir(parents=True, exist_ok=True); checkpoint_sha = checkpoint_identity(args.checkpoint); target = 1 if args.smoke else args.episodes; pending = [task for task in TASKS if not task_complete(args.output / f"{task}.json", target, checkpoint_sha)]
    sim_python = os.environ.get("DUO_PI05_SIM_PYTHON", sys.executable)
    # A wave contains at most one renderer/policy pair per GPU.  Completed task
    # receipts are skipped, so an interrupted Validation20 resumes safely.
    for wave_start in range(0, len(pending), args.workers):
        wave = pending[wave_start : wave_start + args.workers]; workers = []; evaluators = []
        try:
            for gpu, task in enumerate(wave):
                socket_path = Path(f"/tmp/duo-pi05-{os.getpid()}-{gpu}.sock"); socket_path.unlink(missing_ok=True)
                # RCS is deliberately loaded from the simulator venv's built
                # wheel.  Adding the RCS source tree here shadows its compiled
                # ``rcs._core`` extension with an incomplete namespace package.
                env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OPENPI_ROOT": "/workspace/repos/openpi", "DUO_PI05_CHECKPOINT": args.checkpoint, "PYTHONPATH": "/workspace/repos/before-we-act:/workspace/repos/openpi/src:/workspace/repos/duobench/src", "HF_HOME": "/workspace/.hf_home", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_ALLOCATOR": "platform", "MUJOCO_GL": "egl", "DUOBENCH_PREFIX": "/workspace/datasets/duobench_assets", "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false", "TF_CPP_MIN_LOG_LEVEL": "2"})
                log = (args.output / "logs" / f"{task}.worker.log"); log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open("ab", buffering=0); worker = subprocess.Popen([sys.executable, "-m", "deployment.duo_pi05.rpc_server", "--checkpoint", args.checkpoint, "--socket", str(socket_path)], env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True); worker._duo_log = stream  # type: ignore[attr-defined]
                workers.append((task, gpu, socket_path, worker)); wait_socket(socket_path, worker)
                output = args.output / f"{task}.json"; eval_log = (args.output / "logs" / f"{task}.eval.log").open("ab", buffering=0)
                command = [sys.executable, "-m", "deployment.duo_pi05.evaluate", "--socket", str(socket_path), "--task", task, "--output", str(output), "--episodes", str(1 if args.smoke else args.episodes), "--smoke" if args.smoke else ""]
                command = [item for item in command if item]
                if args.smoke: command += ["--max-steps", "2"]
                command[0] = sim_python
                evaluator = subprocess.Popen(command, env=env, stdout=eval_log, stderr=subprocess.STDOUT, start_new_session=True); evaluator._duo_log = eval_log  # type: ignore[attr-defined]
                evaluators.append((task, evaluator))
            failed = []
            for task, process in evaluators:
                code = process.wait()
                if code: failed.append((task, code))
            if failed: raise RuntimeError(f"validation task failures: {failed}")
        finally:
            for _task, process in evaluators: stop_group(process)
            for _task, _gpu, socket_path, process in workers:
                if process.poll() is None:
                    try:
                        payload = pickle.dumps({"op": "shutdown"}, protocol=5)
                        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                            conn.settimeout(10); conn.connect(str(socket_path)); conn.sendall(struct.pack("!Q", len(payload)) + payload)
                    except OSError: pass
                stop_group(process); socket_path.unlink(missing_ok=True); getattr(process, "_duo_log", None) and process._duo_log.close()
        for task, _gpu, _socket, _worker in workers:
            report = json.loads((args.output / f"{task}.json").read_text())
            if report.get("episodes") != (1 if args.smoke else args.episodes) or report.get("checkpoint_sha256") != checkpoint_sha: raise RuntimeError(f"invalid validation receipt for {task}")
    reports = {task: json.loads((args.output / f"{task}.json").read_text()) for task in TASKS}
    rows = [row for task in TASKS for row in reports[task]["rows"]]
    summary = {"schema": "duobench.pi05.validation-summary.v1", "status": "complete", "benchmark": "DuoBench", "policy": "pi0.5_lora", "policy_contract": POLICY_CONTRACT, "evaluator_revision": EVALUATOR_REVISION, "episodes_per_task": target, "total_episodes": len(rows), "successes": sum(int(row["success"]) for row in rows), "macro_success_rate": float(np.mean([reports[task]["success_rate"] for task in TASKS])), "normalized_final_stage_progress": float(np.mean([row["final_stage_progress"] for row in rows])), "tasks": {task: {"episodes": reports[task]["episodes"], "successes": reports[task]["successes"], "success_rate": reports[task]["success_rate"], "max_steps": reports[task]["max_steps"]} for task in TASKS}, "rows": rows, "checkpoint": args.checkpoint, "checkpoint_sha256": checkpoint_sha, "gpu_schedule": "four GPU waves; one shared-weight local-only policy worker per task; CPU MuJoCo/EGL simulator", "smoke": args.smoke}
    atomic_json(args.output / "summary.json", summary); print(json.dumps({key: summary[key] for key in ("status", "total_episodes", "successes", "macro_success_rate")}), flush=True)


if __name__ == "__main__": main()
