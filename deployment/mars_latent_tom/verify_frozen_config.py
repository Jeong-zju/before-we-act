from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import FROZEN_CONFIG, POLICY_CONTRACT, load_frozen_config, sha256
from .policy import LocalLatentToMPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen MARS-Control LatentToM contract.")
    parser.add_argument("--config", type=Path, default=FROZEN_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    frozen = load_frozen_config(args.config)
    model = LocalLatentToMPolicy.from_frozen_config(frozen)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    state_elements = sum(value.numel() for value in model.state_dict().values())
    if trainable != frozen["model"]["trainable_parameters"]:
        raise RuntimeError("trainable parameter count drift")
    if state_elements != frozen["model"]["state_elements_including_buffers"]:
        raise RuntimeError("model state element count drift")
    if model.scheduler.config.clip_sample is not False:
        raise RuntimeError("DDIM clip_sample drift")
    if frozen["optimization"]["samples_seen"] != (
        frozen["optimization"]["optimizer_updates"] * frozen["optimization"]["global_batch_size"]
    ):
        raise RuntimeError("sample budget arithmetic drift")

    checkpoint_path = args.checkpoint or Path(frozen["artifacts"]["formal_checkpoint"])
    checkpoint = None
    if checkpoint_path.is_file():
        if sha256(checkpoint_path) != frozen["artifacts"]["formal_checkpoint_sha256"]:
            raise RuntimeError("formal checkpoint SHA-256 drift")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract") != POLICY_CONTRACT:
            raise RuntimeError("checkpoint policy contract drift")
        if int(checkpoint.get("step", -1)) != frozen["optimization"]["optimizer_updates"]:
            raise RuntimeError("checkpoint optimizer-update budget drift")
        model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]), strict=True)

    print(json.dumps({
        "schema": "mars-control.latent-tom.frozen-config-audit.v1",
        "status": "PASS",
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "checkpoint": str(checkpoint_path) if checkpoint is not None else None,
        "checkpoint_verified": checkpoint is not None,
        "trainable_parameters": trainable,
        "state_elements_including_buffers": state_elements,
        "optimizer_updates": frozen["optimization"]["optimizer_updates"],
        "samples_seen": frozen["optimization"]["samples_seen"],
        "policy_contract": POLICY_CONTRACT,
    }, indent=2))


if __name__ == "__main__":
    main()
