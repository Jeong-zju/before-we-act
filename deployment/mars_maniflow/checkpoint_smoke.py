from __future__ import annotations

import argparse
import json

import torch

from modeling import load_policy, model_config


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); policy, payload = load_policy(args.checkpoint, "cuda:0")
    obs = {"head_cam": torch.zeros(2, 2, 240, 320, 3, dtype=torch.uint8, device="cuda"),
           "agent_pos": torch.zeros(2, 2, 9, device="cuda")}
    torch.manual_seed(7); base = policy.predict_action(obs)["action"]
    changed = {key: value.clone() for key, value in obs.items()}; changed["head_cam"][0].fill_(255); changed["agent_pos"][0].fill_(1)
    torch.manual_seed(7); perturbed = policy.predict_action(changed)["action"]
    isolation = float((base[1] - perturbed[1]).abs().max())
    if base.shape != (2, 15, 8) or not torch.isfinite(base).all() or isolation != 0.0:
        raise RuntimeError(f"checkpoint/runtime isolation smoke failed: shape={base.shape}, isolation={isolation}")
    result = {"status": "complete", "step": int(payload["step"]), "shape": list(base.shape),
              "weights": "ema", "cross_actor_max_abs_diff": isolation,
              "contract": model_config()["policy_contract"]}
    with open(args.output, "w") as handle: json.dump(result, handle, indent=2, sort_keys=True); handle.write("\n")


if __name__ == "__main__": main()
