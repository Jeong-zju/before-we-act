from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import socket
import struct
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torchvision.transforms.v2 import functional as TVF

from deployment.duo_act.action_target import canonicalize_controller_action

from .common import EVALUATOR_REVISION, MAX_STEPS, POLICY_CONTRACT, PROMPTS, TASKS, atomic_json, checkpoint_identity


def recv_exact(conn: socket.socket, size: int) -> bytes:
    result = bytearray()
    while size:
        block = conn.recv(size)
        if not block: raise EOFError("policy RPC disconnected")
        result.extend(block); size -= len(block)
    return bytes(result)


def rpc(path: str, request: dict) -> dict:
    payload = pickle.dumps(request, protocol=5)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(900); conn.connect(path); conn.sendall(struct.pack("!Q", len(payload)) + payload)
        size = struct.unpack("!Q", recv_exact(conn, 8))[0]; response = pickle.loads(recv_exact(conn, size))
    if not response.get("ok"): raise RuntimeError(response.get("error", "policy RPC failed"))
    return response


def make_env(task: str):
    module = __import__(f"duobench.tasks.{task}", fromlist=["*"])
    cls = getattr(module, "".join(part.title() for part in task.split("_")) + "EnvConfig")
    from rcs._core.sim import SimConfig
    from rcs.envs.base import ControlMode, RelativeTo
    cfg = cls().config(); cfg.headless = True; cfg.control_mode = ControlMode.JOINTS; cfg.relative_to = RelativeTo.NONE
    cfg.sim_cfg = SimConfig(async_control=True, realtime=False, frequency=30); cfg.wrapper_cfg.binary_gripper = True
    return gym.make(f"duobench/{task}", cfg=cfg)


def local_state(observation: dict, arm: int) -> np.ndarray:
    key = "left" if arm == 0 else "right"; own = observation[key]
    joints = np.asarray(own["joints"], np.float32); width = np.asarray(own["gripper"], np.float32).reshape(-1)
    if joints.shape != (7,) or width.shape != (1,): raise ValueError(f"runtime state shape drift: {joints.shape}/{width.shape}")
    return np.concatenate((joints, np.asarray([float(width[0] > 0.9)], np.float32)))


def local_views(observation: dict, arm: int) -> tuple[np.ndarray, np.ndarray]:
    head = np.asarray(observation["frames"]["head"]["rgb"]["data"], np.uint8)
    wrist = np.asarray(observation["frames"]["left_wrist" if arm == 0 else "right_wrist"]["rgb"]["data"], np.uint8)
    if head.ndim != 3 or wrist.ndim != 3 or head.shape[-1] != 3 or wrist.shape[-1] != 3: raise ValueError(f"runtime RGB drift: {head.shape}/{wrist.shape}")
    views = TVF.resize(torch.from_numpy(np.stack((head, wrist)).copy()).permute(0, 3, 1, 2), (224, 224), antialias=True)
    return views[0].permute(1, 2, 0).numpy().astype(np.uint8), views[1].permute(1, 2, 0).numpy().astype(np.uint8)


class TemporalEnsembler:
    def __init__(self, decay: float = 0.01): self.decay, self.history = float(decay), [deque() for _ in range(2)]
    def update(self, step: int, arm: int, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, np.float32); history = self.history[arm]; history.append((step, chunk))
        retained = [(start, value) for start, value in history if step - start < len(value)]
        history.clear(); history.extend(retained)
        candidates = np.asarray([value[step - start] for start, value in history], np.float32)
        weights = np.exp(-self.decay * np.arange(len(candidates) - 1, -1, -1, dtype=np.float32)); weights /= weights.sum()
        return canonicalize_controller_action(np.sum(candidates * weights[:, None], axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--socket", required=True); parser.add_argument("--task", choices=TASKS, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--episodes", type=int, default=20); parser.add_argument("--seed-start", type=int, default=20260820); parser.add_argument("--max-steps", type=int); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    max_steps = int(args.max_steps or MAX_STEPS[args.task]); checkpoint = os.environ.get("DUO_PI05_CHECKPOINT", "")
    checkpoint_sha = checkpoint_identity(checkpoint) if checkpoint and Path(checkpoint).exists() else None
    previous = {}
    journal = args.output.with_suffix(".jsonl")
    if journal.is_file():
        for line in journal.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get("evaluator_revision") == EVALUATOR_REVISION and row.get("checkpoint_sha256") == checkpoint_sha: previous[int(row["seed"])] = row
            except (ValueError, KeyError, json.JSONDecodeError): pass
    env = make_env(args.task); rows = []
    try:
        for episode in range(args.episodes):
            seed = args.seed_start + TASKS.index(args.task) * 1000 + episode
            if seed in previous: rows.append(previous[seed]); continue
            observation, _ = env.reset(seed=seed); rpc(args.socket, {"op": "reset"}); ensemble = TemporalEnsembler(); trace = hashlib.sha256(); success = False; final_progress = max_progress = 0.0; started = time.perf_counter(); controls = 0; inference_times = []
            while controls < max_steps and not success:
                chunks = []
                for arm in (0, 1):
                    head, wrist = local_views(observation, arm); state = local_state(observation, arm)
                    started_infer = time.perf_counter(); response = rpc(args.socket, {"op": "infer", "observation": {"head": head.tobytes(), "wrist": wrist.tobytes(), "state": state.astype(np.float32, copy=False).tobytes(), "prompt": PROMPTS[args.task]}}); inference_times.append(time.perf_counter() - started_infer)
                    chunk = np.frombuffer(response["chunk"], np.float32).reshape(16, 8).copy()
                    chunks.append(chunk)
                action = {}
                for arm, key in enumerate(("left", "right")):
                    local = ensemble.update(controls, arm, chunks[arm]); trace.update(local.astype(np.float32).tobytes()); action[key] = {"joints": local[:7], "gripper": np.asarray([local[7]], np.float32)}
                observation, reward, terminated, truncated, info = env.step(action); controls += 1; final_progress = float(reward); max_progress = max(max_progress, final_progress); success = bool(info.get("success", False))
                if success or bool(np.asarray(terminated).all()) or bool(np.asarray(truncated).all()): break
            row = {"task": args.task, "seed": seed, "success": success, "steps": controls, "max_steps": max_steps, "final_stage_progress": final_progress, "max_stage_progress": max_progress, "replans": controls, "mean_inference_seconds": float(np.mean(inference_times)) if inference_times else None, "p95_inference_seconds": float(np.quantile(inference_times, .95)) if inference_times else None, "action_trace_sha256": trace.hexdigest(), "wall_seconds": time.perf_counter() - started, "checkpoint_sha256": checkpoint_sha, "evaluator_revision": EVALUATOR_REVISION}
            rows.append(row); args.output.parent.mkdir(parents=True, exist_ok=True)
            with journal.open("a", encoding="utf-8") as stream: stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    finally: env.close()
    result = {"schema": "duobench.pi05.validation-task.v1", "status": "complete", "task": args.task, "episodes": len(rows), "successes": sum(int(row["success"]) for row in rows), "success_rate": float(np.mean([row["success"] for row in rows])), "rows": rows, "checkpoint": os.environ.get("DUO_PI05_CHECKPOINT"), "checkpoint_sha256": checkpoint_sha, "evaluator_revision": EVALUATOR_REVISION, "policy_contract": POLICY_CONTRACT, "weights": "upstream_pi05_lora", "action_horizon": 16, "temporal_ensemble_decay": .01, "replan_interval": 1, "state_gripper_encoding": "physical_width_gt_0.9_to_binary", "action_encoding": "controller_ctrlrange_absolute_joint7_binary_gripper1", "rgb_preprocessing": "independent_torchvision_v2_uint8_bilinear_antialias_resize_224", "max_steps": max_steps, "smoke": args.smoke}
    atomic_json(args.output, result)


if __name__ == "__main__": main()
