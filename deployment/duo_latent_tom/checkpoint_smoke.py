from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import POLICY_CONTRACT, atomic_json, load_config
from .policy import LocalLatentToMPolicy


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    config = load_config(); saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("policy_contract") != POLICY_CONTRACT: raise ValueError("checkpoint policy contract mismatch")
    model = LocalLatentToMPolicy.from_config(config).to(args.device); model.load_state_dict(saved.get("ema_model", saved["model"]), strict=True); model.set_stats(saved["stats"]); model.eval()
    obs = {"image": torch.zeros((1, 2, 3, 224, 448), dtype=torch.uint8, device=args.device), "qpos": torch.zeros((1, 2, 8), device=args.device), "task": torch.nn.functional.one_hot(torch.tensor([0], device=args.device), num_classes=11).float(), "arm_id": torch.tensor([[1.0, 0.0]], device=args.device)}
    torch.manual_seed(123); torch.cuda.manual_seed_all(123)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): output = model.predict_chunk(obs, steps=2)
    passed = output.shape == (1, 40, 8) and bool(torch.isfinite(output).all()) and bool(model.scheduler.config.clip_sample)
    report = {"schema": "duobench.latent-tom.checkpoint-smoke.v2", "status": "complete" if passed else "failed", "checkpoint_step": int(saved["step"]), "shape": list(output.shape), "finite": bool(torch.isfinite(output).all()), "ddim_clip_sample": bool(model.scheduler.config.clip_sample), "normalization_mode": model.normalization_mode, "policy_contract": POLICY_CONTRACT}
    atomic_json(args.output, report); print(json.dumps(report))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
