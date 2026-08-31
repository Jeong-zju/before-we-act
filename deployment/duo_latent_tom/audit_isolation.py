from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import FROZEN_CONFIG, POLICY_CONTRACT, atomic_json, load_config
from .policy import LocalLatentToMPolicy


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    config = load_config(); payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("policy_contract") != POLICY_CONTRACT: raise ValueError("checkpoint policy contract mismatch")
    model = LocalLatentToMPolicy.from_config(config).to(args.device); model.load_state_dict(payload.get("ema_model", payload["model"])); model.set_stats(payload["stats"]); model.eval()
    torch.manual_seed(7)
    image = torch.randint(0, 256, (2, 2, 3, 224, 448), dtype=torch.uint8, device=args.device)
    qpos = torch.randn((2, 2, 8), device=args.device); task = torch.nn.functional.one_hot(torch.tensor([2, 2], device=args.device), num_classes=11).float()
    with torch.inference_mode():
        first = model._encode({"image": image, "qpos": qpos, "task": task})
        second = model._encode({"image": image, "qpos": qpos, "task": task})
    encoder_diff = float((first - second).abs().max().cpu())
    # There is no peer tensor in the policy API.  Equal local rows must remain
    # equal even when the other batch row is changed, documenting the row-wise
    # batching invariant used by decentralized inference.
    image_alt = image.clone(); image_alt[1].zero_(); qpos_alt = qpos.clone(); qpos_alt[1].zero_()
    with torch.inference_mode():
        local_a = model._encode({"image": image, "qpos": qpos, "task": task})[0]
        local_b = model._encode({"image": image_alt, "qpos": qpos_alt, "task": task})[0]
    cross_actor_diff = float((local_a - local_b).abs().max().cpu())
    passed = encoder_diff == 0.0 and cross_actor_diff == 0.0
    report = {"schema": "duobench.latent-tom.runtime-isolation.v1", "status": "complete" if passed else "failed", "checkpoint_step": int(payload["step"]), "encoder_repeat_max_abs_diff": encoder_diff, "cross_actor_input_perturbation_max_abs_diff_expected_zero": cross_actor_diff, "policy_contract": POLICY_CONTRACT}
    atomic_json(args.output, report); print(json.dumps(report))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
