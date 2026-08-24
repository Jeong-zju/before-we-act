#!/usr/bin/env python3
"""Runtime gate for OpenVLA's strict one-arm-per-request contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import socket
import struct
import subprocess
import tempfile
import time

import numpy as np


def rpc(path: str, value: dict) -> dict:
    payload = pickle.dumps(value, protocol=5)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(900)
        conn.connect(path)
        conn.sendall(struct.pack("!Q", len(payload)) + payload)
        size = struct.unpack("!Q", _recv_exact(conn, 8))[0]
        response = pickle.loads(_recv_exact(conn, size))
    return response


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    blocks = []
    while size:
        block = conn.recv(size)
        if not block:
            raise EOFError("policy worker disconnected")
        blocks.append(block)
        size -= len(block)
    return b"".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default="/workspace/venvs/openvla/bin/python")
    args = parser.parse_args()
    socket_path = f"/tmp/bwa-openvla-isolation-{os.getpid()}.sock"
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        HF_HOME="/workspace/.hf_home",
        OPENVLA_ROBOFACTORY_ROOT="/workspace/datasets/robofactory_multitask",
        PYTHONPATH="/workspace/repos/before-we-act:/workspace/repos/RoboFactory",
        TOKENIZERS_PARALLELISM="false",
        WANDB_MODE="disabled",
        TF_CPP_MIN_LOG_LEVEL="2",
    )
    worker = subprocess.Popen(
        [args.python, "/workspace/repos/before-we-act/deployment/vla_baselines/policy_rpc_server.py",
         "--policy", "openvla", "--checkpoint", args.checkpoint, "--socket", socket_path],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 900
        while not Path(socket_path).is_socket():
            if worker.poll() is not None:
                detail = worker.stdout.read().decode(errors="replace")[-8000:] if worker.stdout else ""
                raise RuntimeError(
                    f"OpenVLA worker exited during isolation audit: {worker.returncode}\n{detail}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError("OpenVLA isolation worker readiness timeout")
            time.sleep(1)
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        state = np.zeros((9,), dtype=np.float32)
        local = {"agent": 0, "task": "lift_barrier", "prompt": "Lift the barrier together",
                 "image": image, "state": state}
        rejected = rpc(socket_path, {"op": "infer", "observations": [local, dict(local, agent=1) ]})
        if rejected.get("ok") or "exactly one arm-local" not in rejected.get("error", ""):
            raise RuntimeError(f"worker accepted a multi-arm request: {rejected}")
        first = rpc(socket_path, {"op": "infer", "observations": [local]})
        second = rpc(socket_path, {"op": "infer", "observations": [dict(local, agent=1)]})
        if not first.get("ok") or not second.get("ok"):
            raise RuntimeError(f"valid local request failed: {first} / {second}")
        a0, a1 = np.asarray(first["chunks"][0]), np.asarray(second["chunks"][0])
        if a0.shape != a1.shape or not np.isfinite(a0).all() or not np.isfinite(a1).all():
            raise RuntimeError(f"invalid action chunks: {a0.shape} / {a1.shape}")
        if not np.array_equal(a0, a1):
            raise RuntimeError("policy output depends on agent id despite identical local observation")
        payload = {
            "schema": "bwa.openvla.runtime_isolation.v1", "status": "complete",
            "policy_contract": "shared_weights_decentralized_local_rgb_qpos_to_local_action8",
            "multi_arm_request_rejected": True, "identical_local_observation_identical_action": True,
            "action_shape": list(a0.shape),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        try:
            if Path(socket_path).is_socket():
                rpc(socket_path, {"op": "shutdown"})
        except Exception:
            pass
        try:
            worker.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid, 15)
            worker.wait(timeout=20)
        Path(socket_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
