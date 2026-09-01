from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import sys
from pathlib import Path

import numpy as np

from deployment.duo_act.action_target import canonicalize_controller_action


def recv_exact(conn: socket.socket, size: int) -> bytes:
    blocks = []
    while size:
        block = conn.recv(size)
        if not block: raise EOFError("policy client disconnected")
        blocks.append(block); size -= len(block)
    return b"".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--socket", required=True); args = parser.parse_args()
    repo = Path(os.environ.get("OPENPI_ROOT", "/workspace/repos/openpi")); sys.path.insert(0, str(repo)); os.chdir(repo)
    from openpi.policies import policy_config
    from openpi.training import config

    train_config = config.get_config("pi05_duobench_lora")
    policy = policy_config.create_trained_policy(train_config, args.checkpoint, pytorch_device="cpu")
    # JAX policies choose their visible CUDA device from CUDA_VISIBLE_DEVICES.
    # ``pytorch_device`` is intentionally irrelevant for this upstream model;
    # retaining the argument makes accidental PyTorch model replacement fail.
    path = Path(args.socket); path.unlink(missing_ok=True); path.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); server.bind(str(path)); os.chmod(path, 0o600); server.listen(4)
    running = True
    while running:
        conn, _ = server.accept()
        with conn:
            try:
                size = struct.unpack("!Q", recv_exact(conn, 8))[0]
                if size > 256 * 1024 * 1024: raise ValueError("RPC request too large")
                request = pickle.loads(recv_exact(conn, size)); op = request.get("op")
                if op == "ping": response = {"ok": True, "policy_contract": train_config.policy_metadata}
                elif op == "shutdown": response = {"ok": True}; running = False
                elif op == "reset": response = {"ok": True}
                elif op == "infer":
                    observation = request.get("observation")
                    if not isinstance(observation, dict) or set(observation) != {"head", "wrist", "state", "prompt"}:
                        raise ValueError("decentralized request must contain exactly head,wrist,state,prompt")
                    # Arrays cross the NumPy-1/NumPy-2 venv boundary as raw
                    # bytes, avoiding pickle references to private NumPy
                    # module names (for example ``numpy._core.numeric``).
                    if not all(isinstance(observation[key], bytes) for key in ("head", "wrist", "state")):
                        raise ValueError("RPC array fields must be raw bytes")
                    head = np.frombuffer(observation["head"], np.uint8).reshape(224, 224, 3)
                    wrist = np.frombuffer(observation["wrist"], np.uint8).reshape(224, 224, 3)
                    state = np.frombuffer(observation["state"], np.float32)
                    for name, image in (("head", head), ("wrist", wrist)):
                        if image.shape != (224, 224, 3) or image.dtype != np.uint8: raise ValueError(f"{name} image contract failed: {image.shape}/{image.dtype}")
                    if state.shape != (8,) or not np.isfinite(state).all() or not np.isin(state[7], (0.0, 1.0)): raise ValueError("local state contract failed")
                    if not isinstance(observation["prompt"], str) or not observation["prompt"]: raise ValueError("prompt contract failed")
                    output = policy.infer({
                        "head": head, "wrist": wrist, "state": state,
                        "prompt": observation["prompt"],
                    })
                    chunk = canonicalize_controller_action(np.asarray(output["actions"], np.float32))
                    if chunk.shape != (16, 8) or not np.isfinite(chunk).all(): raise ValueError(f"policy action chunk contract failed: {chunk.shape}")
                    response = {"ok": True, "chunk": chunk.astype(np.float32, copy=False).tobytes()}
                else: raise ValueError(f"unknown RPC operation: {op!r}")
            except Exception as error: response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            payload = pickle.dumps(response, protocol=5); conn.sendall(struct.pack("!Q", len(payload)) + payload)
    server.close(); path.unlink(missing_ok=True)


if __name__ == "__main__": main()
