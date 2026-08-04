#!/usr/bin/env python3
"""Real-parent gate-zero, candidate-bank, temporal and latency audit for R10."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stereo_core.bwa_perception import build_perception_extension  # noqa: E402
from stereo_core.evaluate_no_wrist_pair import (  # noqa: E402
    TemporalChunkEnsembler,
    denormalize_action_chunks,
)
from stereo_core.no_wrist_pair_model import NoWristPAIRRoute  # noqa: E402


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def compare(left, right):
    exact = torch.equal(left, right)
    maximum = float((left.float() - right.float()).abs().max()) if left.numel() else 0.0
    return {"exact": exact, "max_abs": maximum}


def temporal(chunks, stats):
    values = denormalize_action_chunks(chunks, stats).float().cpu().numpy()
    ensemble = TemporalChunkEnsembler((0,))
    return np.stack([ensemble.append_and_select(step, values)["panda-0"] for step in range(12)])


def timed_forward(model, inputs, repeats, device):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            model(*inputs)
    values = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            model(*inputs)
            torch.cuda.synchronize(device)
            values.append((time.perf_counter_ns() - started) / 1e6)
    return {"samples": repeats, "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--extension-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latency-repeats", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.latency_repeats < 1000:
        raise ValueError("R10 latency acceptance requires at least 1000 samples per path")

    device = torch.device(args.device)
    parent = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
    candidate = torch.load(args.extension_checkpoint, map_location="cpu", weights_only=False)
    config = parent["config"]
    model = NoWristPAIRRoute(
        config.get("state_dim", 9), config.get("action_dim", 8),
        horizon=config.get("horizon", 100), d_model=config.get("d_model", 384),
        enc_layers=config.get("enc_layers", 4), dec_layers=config.get("dec_layers", 7),
        roles=config.get("roles", 4), role_rank=config.get("role_rank", 32),
        dino_model=config["dino_model"],
    ).to(device)
    model.load_state_dict(parent["model"], strict=True)
    model.eval()
    parent_hash_before = tensor_hash(model.state_dict())
    generator = torch.Generator().manual_seed(101001)
    global_rgb = torch.rand(1, 3, 480, 640, generator=generator).to(device)
    local_rgb = torch.rand(1, 3, 480, 640, generator=generator).to(device)
    qpos = torch.rand(1, config.get("state_dim", 9), generator=generator).to(device)
    inputs = (global_rgb, local_rgb, qpos)
    stats = {key: torch.as_tensor(value, device=device) for key, value in parent["stats"].items()}

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        base_forward = model(*inputs, return_routing=True)
        base_bank = model.propose_core_bank(*inputs)
    base_temporal = temporal(base_forward[0], stats)
    base_latency = timed_forward(model, inputs, args.latency_repeats, device)

    extension = build_perception_extension(candidate["config"]["bridge"]).to(device)
    extension.load_state_dict(candidate["extension"], strict=True)
    trained_gate = float(torch.tanh(extension.perception_gate.detach()))
    with torch.no_grad():
        extension.raw_perception_gate.zero_()
    model.register_perception_extension(extension)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        zero_forward = model(*inputs, return_routing=True)
        zero_bank = model.propose_core_bank(*inputs)
    zero_temporal = temporal(zero_forward[0], stats)
    candidate_latency = timed_forward(model, inputs, args.latency_repeats, device)
    parent_subset = {name: value for name, value in model.state_dict().items() if not name.startswith("perception_extension.")}
    parent_hash_after = tensor_hash(parent_subset)
    checks = {
        "base_chunk": compare(base_forward[0], zero_forward[0]),
        "dense_routes": compare(base_forward[4], zero_forward[4]),
        "bank_chunks": compare(base_bank.chunks, zero_bank.chunks),
        "bank_routes": compare(base_bank.routes, zero_bank.routes),
        "temporal_output": {
            "exact": bool(np.array_equal(base_temporal, zero_temporal)),
            "max_abs": float(np.abs(base_temporal - zero_temporal).max()),
        },
        "parent_state": {"exact": parent_hash_before == parent_hash_after},
    }
    ratio = candidate_latency["p95_ms"] / base_latency["p95_ms"]
    result = {
        "schema_version": 1,
        "candidate_id": candidate["candidate_id"],
        "trained_gate": trained_gate,
        "gate_for_audit": 0.0,
        "checks": checks,
        "latency": {"base": base_latency, "candidate": candidate_latency, "p95_ratio": ratio},
        "privileged_inputs": False,
    }
    result["gate_zero_passed"] = all(value["exact"] for value in checks.values())
    result["latency_passed"] = ratio <= 1.15
    result["passed"] = result["gate_zero_passed"] and result["latency_passed"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
