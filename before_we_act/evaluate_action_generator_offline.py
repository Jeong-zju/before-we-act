from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from before_we_act.action_generator.base import JointActionGenerator, load_r12_config
from before_we_act.data.action_windows import CachedActionWindows
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config


def subset(data, indices, device):
    return {key: value.index_select(0, indices).to(device) for key, value in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--belief-config", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    config = load_r12_config(args.config)
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved["candidate_id"] != config.candidate_id:
        raise ValueError("R12 checkpoint candidate differs")
    model = JointActionGenerator(config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    belief_config = load_r11_config(args.belief_config)
    belief_saved = torch.load(args.belief_checkpoint, map_location="cpu", weights_only=False)
    belief = PredictiveBeliefModel(belief_config).to(device)
    belief.load_state_dict(belief_saved["model"], strict=True)
    belief.eval()
    dataset = CachedActionWindows(args.cache, "validation")
    count = min(args.samples, len(dataset))
    indices = torch.arange(count)
    batch = subset(dataset.data, indices, device)
    generator = torch.Generator(device=device).manual_seed(20260805)
    noise = torch.randn((count, 100, 32), device=device, generator=generator)
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        belief_state = belief(batch)["belief"]
        proposals = model.sample(belief_state, noise=noise)
    prediction = proposals.actions[:, 0].permute(0, 2, 1, 3)
    target = batch["joint_actions"]
    mask = (
        batch["agent_mask"][:, None, :, None]
        & batch["action_step_mask"][:, :, None, None].bool()
    ).expand_as(target)
    mse = ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1)
    timing_belief = type(belief_state)(
        tokens=belief_state.tokens[:1],
        agent_tokens=belief_state.agent_tokens[:1],
        consensus_token=belief_state.consensus_token[:1],
        uncertainty=belief_state.uncertainty[:1],
        agent_mask=belief_state.agent_mask[:1],
    )
    timing_noise = noise[:1]
    latencies = []
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for _ in range(3):
            model.sample(timing_belief, noise=timing_noise)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(10):
            started = time.perf_counter_ns()
            model.sample(timing_belief, noise=timing_noise)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter_ns() - started) / 1e6)
    values = torch.tensor(latencies)
    result = {
        "schema_version": 1,
        "round": "R12",
        "candidate_id": config.candidate_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_update": int(saved["update"]),
        "validation_samples": count,
        "normalized_action_mse": float(mse),
        "finite": bool(torch.isfinite(prediction).all()),
        "normalized_abs_max": float(prediction.abs().max()),
        "absent_agent_zero": bool((prediction.masked_select(~batch["agent_mask"][:, None, :, None])).eq(0).all()),
        "latency_ms": {
            "samples": len(latencies),
            "p50": float(torch.quantile(values, 0.5)),
            "p95": float(torch.quantile(values, 0.95)),
        },
        "core_free_runtime": bool(saved.get("core_free_runtime")),
        "quality_threshold": None,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
