#!/usr/bin/env python3
"""Launch Validation20 workers with exact-PID lifecycle and GPU isolation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

TASKS = ("lift_barrier", "camera_alignment", "long_pipeline_delivery", "take_photo", "pass_shoe", "place_food")
# RoboFactory's CPU simulation backend still uses Vulkan for RGB rendering.  Running
# several renderer instances at once can lose a Vulkan device on this host, so keep
# closed-loop tasks isolated.  Policy inference still uses a GPU, while completed
# task receipts make retries resume without re-running successful episodes.
WAVES = tuple(((task, gpu),) for task, gpu in (
    ("lift_barrier", 0),
    ("camera_alignment", 1),
    ("long_pipeline_delivery", 2),
    ("take_photo", 3),
    ("pass_shoe", 0),
    ("place_food", 1),
))
PARALLEL_WAVES = (
    (("lift_barrier", 0), ("camera_alignment", 1), ("long_pipeline_delivery", 2), ("take_photo", 3)),
    (("pass_shoe", 0), ("place_food", 1)),
)
PYTHONS = {
    "rdt": "/workspace/venvs/rdt/bin/python",
    "openvla": "/workspace/venvs/openvla/bin/python",
    "pi05": "/workspace/repos/openpi/.venv/bin/python",
    "gaudp": "/venv/main/bin/python",
}
SCRIPT_ROOT = Path("/workspace/repos/before-we-act/deployment/vla_baselines")
# RoboFactory/SAPIEN is installed in the dedicated Python 3.9 environment.
# Keep this overrideable for hosts with a different simulator installation.
SIM_PYTHON = os.environ.get("BWA_SIM_PYTHON", "/workspace/venvs/robofactory/bin/python")
children: list[tuple[subprocess.Popen, int]] = []
stopping = False


def proc_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text()
    return int(raw[raw.rfind(")") + 2 :].split()[19])


def same_process(child: tuple[subprocess.Popen, int]) -> bool:
    process, ticks = child
    try:
        return proc_start_ticks(process.pid) == ticks
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return False


def spawn(argv: list[str], *, env: dict[str, str], log_path: Path) -> tuple[subprocess.Popen, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab", buffering=0)
    process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    process._bwa_log = log  # type: ignore[attr-defined]
    child = (process, proc_start_ticks(process.pid))
    children.append(child)
    return child


def stop(child: tuple[subprocess.Popen, int], timeout: int = 45) -> None:
    process, _ = child
    if same_process(child):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while same_process(child) and time.monotonic() < deadline:
        # An exited-but-unreaped child remains in /proc with the same start
        # ticks.  Poll/reap it so shutdown does not spend the full timeout on
        # every failed worker.
        if process.poll() is not None:
            break
        time.sleep(0.2)
    if same_process(child):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    log = getattr(process, "_bwa_log", None)
    if log is not None:
        log.close()


def stop_all() -> None:
    for child in reversed(children):
        stop(child)


def on_signal(signum, _frame) -> None:
    global stopping
    stopping = True
    stop_all()
    raise SystemExit(128 + signum)


def rpc_shutdown(path: Path) -> None:
    payload = pickle.dumps({"op": "shutdown"}, protocol=5)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(10)
        conn.connect(str(path))
        conn.sendall(struct.pack("!Q", len(payload)) + payload)
        size = struct.unpack("!Q", conn.recv(8))[0]
        while size:
            block = conn.recv(size)
            if not block:
                break
            size -= len(block)


def task_complete(path: Path, target_episodes: int) -> bool:
    try:
        data = json.loads(path.read_text())
        return data.get("status") == "complete" and data.get("episodes") == target_episodes
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def render_preflight(env: dict[str, str]) -> None:
    """Fail before loading a policy when the SAPIEN Vulkan device is unusable."""
    probe_env = env.copy()
    probe_env["CUDA_VISIBLE_DEVICES"] = "0"
    render_device = "cpu" if env.get("BWA_RENDER_ICD") == "cpu" else "cuda:0"
    code = (
        "import sapien\n"
        f"device = sapien.Device({render_device!r})\n"
        "render_system = sapien.render.RenderSystem(device)\n"
        "print('bwa-sapien-render-preflight-ok', flush=True)\n"
    )
    result = subprocess.run(
        [SIM_PYTHON, "-c", code],
        env=probe_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        detail = result.stdout[-4000:].strip()
        raise RuntimeError(f"SAPIEN Vulkan render preflight failed ({result.returncode}): {detail}")


def run_wave(
    policy: str,
    checkpoint: str,
    root: Path,
    assignments,
    *,
    episodes: int,
    seed: int,
    formal: bool,
    max_steps_override: int | None,
) -> None:
    base_env = os.environ.copy()
    driver_lib = "/opt/nvidia-drivers/lib64"
    inherited_ld_path = base_env.get("LD_LIBRARY_PATH", "")
    use_cpu_vulkan = base_env.get("BWA_RENDER_ICD") == "cpu"
    nvidia_icd = Path("/etc/vulkan/icd.d/nvidia_icd.json")
    lavapipe_icd = Path("/usr/share/vulkan/icd.d/lvp_icd.json")
    selected_icd = lavapipe_icd if use_cpu_vulkan else nvidia_icd
    if not selected_icd.is_file():
        raise RuntimeError(f"requested Vulkan ICD is missing: {selected_icd}")
    Path("/tmp/bwa-xdg-runtime").mkdir(mode=0o700, exist_ok=True)
    base_env.update(
        HF_HOME="/workspace/.hf_home",
        OPENVLA_ROBOFACTORY_ROOT="/workspace/datasets/robofactory_multitask",
        OPENPI_ROBOFACTORY_ROOT="/workspace/datasets/robofactory_multitask",
        RDT_ROBOFACTORY_DATASET="/workspace/datasets/robofactory_multitask",
        PYTHONPATH="/workspace/repos/RoboFactory:/workspace/repos/before-we-act",
        TOKENIZERS_PARALLELISM="false",
        WANDB_MODE="disabled",
        TF_CPP_MIN_LOG_LEVEL="2",
        # JAX otherwise preallocates ~75% of HBM when the policy worker starts.
        # On a one-GPU host SAPIEN's NVIDIA Vulkan renderer shares that H200;
        # preallocation starves Vulkan and surfaces as vk::DeviceLost before
        # the first observation.  Inference uses demand allocation instead.
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_ALLOCATOR="platform",
        # Prefer the instance's hardware Vulkan ICD when available.  Some
        # headless GPU images expose NVIDIA Vulkan but do not ship Mesa's
        # lavapipe JSON; other images have only lavapipe.  Select an ICD that
        # exists instead of forcing a nonexistent path (which makes SAPIEN
        # fail before an environment can be created).
        VK_DRIVER_FILES=str(selected_icd),
        VK_ICD_FILENAMES=str(selected_icd),
        XDG_RUNTIME_DIR="/tmp/bwa-xdg-runtime",
        # Vast mounts the matching NVIDIA userspace driver here.  It must
        # precede the image's stale GL/Vulkan libraries or the ICD exports no
        # usable vkCreateInstance even though CUDA continues to work.
        LD_LIBRARY_PATH=(
            driver_lib + (":" + inherited_ld_path if inherited_ld_path else "")
            if Path(driver_lib).is_dir()
            else inherited_ld_path
        ),
    )
    if policy == "openvla":
        # Final DDP checkpoint merging can exceed NCCL's barrier timeout.  If
        # that happened, materialize the inference model once, offline, under
        # this supervisor-managed validation stage before starting workers.
        subprocess.run(
            [PYTHONS["openvla"], str(SCRIPT_ROOT / "prepare_openvla_checkpoint.py"),
             "--checkpoint", checkpoint],
            env=base_env, check=True,
        )
    render_preflight(base_env)
    active = []
    for task, gpu in assignments:
        result_path = root / f"{task}.json"
        if task_complete(result_path, episodes):
            continue
        socket_path = Path(f"/tmp/bwa-{policy}-{task}-{os.getpid()}.sock")
        socket_path.unlink(missing_ok=True)
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        worker = spawn(
            [PYTHONS[policy], str(SCRIPT_ROOT / "policy_rpc_server.py"), "--policy", policy,
             "--checkpoint", checkpoint, "--socket", str(socket_path)],
            env=env, log_path=root / "logs" / f"{task}.worker.log",
        )
        active.append({"task": task, "gpu": gpu, "socket": socket_path, "worker": worker, "result": result_path, "env": env})

    deadline = time.monotonic() + 900
    pending = list(active)
    while pending and not stopping:
        for item in list(pending):
            if item["socket"].is_socket():
                pending.remove(item)
            elif not same_process(item["worker"]):
                code = item["worker"][0].wait()
                raise RuntimeError(f"{item['task']} policy worker exited during load: {code}")
        if pending and time.monotonic() >= deadline:
            raise TimeoutError("policy worker readiness timeout: " + ", ".join(row["task"] for row in pending))
        time.sleep(1)

    for item in active:
        evaluator = [SIM_PYTHON, str(SCRIPT_ROOT / "evaluate_vla_closed_loop.py"), "--policy", policy,
             "--socket", str(item["socket"]), "--task", item["task"], "--output", str(item["result"]),
             "--episodes", str(episodes), "--seed", str(seed), "--sim-backend", "cpu",
             "--temporal-ensemble-decay", "0.01", "--cpu-threads", "8"]
        if max_steps_override is not None:
            evaluator.extend(["--max-steps-override", str(max_steps_override)])
        if formal:
            evaluator.append("--formal")
        item["eval"] = spawn(
            evaluator,
            env=item["env"], log_path=root / "logs" / f"{item['task']}.eval.log",
        )

    failures = []
    for item in active:
        process = item["eval"][0]
        code = process.wait()
        if code:
            failures.append((item["task"], code))
    for item in active:
        try:
            rpc_shutdown(item["socket"])
        except Exception:
            pass
        stop(item["worker"])
        item["socket"].unlink(missing_ok=True)
    if failures:
        raise RuntimeError(f"Validation20 evaluator failures: {failures}")


def atomic_json(path: Path, payload: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=tuple(PYTHONS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--max-steps-override", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        if args.episodes != 1 or args.max_steps_override is None or args.max_steps_override < 1:
            raise ValueError("closed-loop smoke requires --episodes 1 and a positive --max-steps-override")
    elif args.episodes != 20 or args.seed != 20260820 or args.max_steps_override is not None:
        raise ValueError("formal Validation20 parameters are frozen")
    root = Path(args.output_root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        visible_gpu_count = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
        single_gpu = visible_gpu_count == 1 or int(os.environ.get("BWA_GPU_COUNT", "1")) == 1
        waves = tuple(((task, 0),) for task in TASKS) if args.policy == "gaudp" or single_gpu else (
            PARALLEL_WAVES if os.environ.get("BWA_VALIDATION_PARALLEL") == "1" else WAVES
        )
        try:
            for wave in waves:
                run_wave(
                    args.policy,
                    args.checkpoint,
                    root,
                    wave,
                    episodes=args.episodes,
                    seed=args.seed,
                    formal=not args.smoke,
                    max_steps_override=args.max_steps_override,
                )
        except Exception:
            if waves is not PARALLEL_WAVES:
                raise
            # Some Vulkan hosts cannot sustain concurrent renderer instances.
            # Preserve completed task receipts and automatically retry only the
            # unfinished tasks with isolated renderer/policy workers.
            stop_all()
            children.clear()
            for wave in WAVES:
                run_wave(
                    args.policy,
                    args.checkpoint,
                    root,
                    wave,
                    episodes=args.episodes,
                    seed=args.seed,
                    formal=not args.smoke,
                    max_steps_override=args.max_steps_override,
                )
        rows = {task: json.loads((root / f"{task}.json").read_text()) for task in TASKS}
        if any(row.get("status") != "complete" or row.get("episodes") != args.episodes for row in rows.values()):
            raise RuntimeError("closed-loop task artifact incomplete")
        atomic_json(
            root / "summary.json",
            {
                "schema": "bwa.vla.closed_loop_smoke.v1" if args.smoke else "bwa.vla.validation20.v1",
                "baseline": args.policy, "status": "complete",
                "episodes_per_task": args.episodes, "total_episodes": args.episodes * len(TASKS),
                "macro_success_rate": sum(row["success_rate"] for row in rows.values()) / len(rows),
                "tasks": rows, "seed_base": args.seed, "sim_backend": "cpu",
                "temporal_ensemble_decay": 0.01,
                "max_steps_override": args.max_steps_override,
                "policy_contract": "shared_weights_decentralized_local_rgb_qpos_to_local_action8",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    finally:
        stop_all()


if __name__ == "__main__":
    main()
