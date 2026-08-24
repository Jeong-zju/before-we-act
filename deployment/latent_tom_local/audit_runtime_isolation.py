from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from local_policy import LocalLatentToMPolicy


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model = LocalLatentToMPolicy().to(device)
    model.load_state_dict(payload.get("ema_model", payload["model"])); model.eval()

    generator = torch.Generator(device="cpu").manual_seed(20260822)
    image = torch.randint(0, 256, (2, 2, 3, 240, 320), dtype=torch.uint8, generator=generator)
    qpos = torch.randn((2, 2, 9), generator=generator)
    task = torch.zeros((2, 6)); task[0, 0] = 1; task[1, 1] = 1
    baseline = {"image": image.to(device), "qpos": qpos.to(device), "task": task.to(device)}
    perturbed = {key: value.clone() for key, value in baseline.items()}
    perturbed["image"][1].copy_(255 - perturbed["image"][1])
    perturbed["qpos"][1].add_(7.0)
    perturbed["task"][1].zero_(); perturbed["task"][1, 5] = 1

    with torch.no_grad():
        cond_a = model._encode(baseline)
        cond_b = model._encode(perturbed)
        torch.manual_seed(314159)
        action_a = model.predict_chunk(baseline, steps=2)
        torch.manual_seed(314159)
        action_b = model.predict_chunk(perturbed, steps=2)
    cond_actor0_diff = float((cond_a[0] - cond_b[0]).abs().max().cpu())
    action_actor0_diff = float((action_a[0] - action_b[0]).abs().max().cpu())
    peer_cond_change = float((cond_a[1] - cond_b[1]).abs().max().cpu())
    if cond_actor0_diff != 0.0 or action_actor0_diff != 0.0:
        raise RuntimeError({"encoder_cross_actor_diff": cond_actor0_diff,
                            "action_cross_actor_diff": action_actor0_diff})
    if peer_cond_change <= 0.0:
        raise RuntimeError("peer perturbation was not effective")
    result = {"schema": "bwa.latent_tom.runtime_isolation.v1", "status": "complete",
              "checkpoint_step": int(payload["step"]),
              "weights": "ema" if "ema_model" in payload else "raw",
              "batching_semantics": "independent_rows_shared_weights_no_cross_actor_communication",
              "encoder_cross_actor_max_abs_diff": cond_actor0_diff,
              "action_cross_actor_max_abs_diff": action_actor0_diff,
              "perturbed_actor_encoder_max_abs_change": peer_cond_change,
              "output_shape": list(action_a.shape)}
    atomic_json(Path(args.output), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
