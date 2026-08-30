"""Verify that the local GauDP policy contract matches the frozen run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import FROZEN_CONFIG, POLICY_CONTRACT, load_frozen_config, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen MARS-Control GauDP contract.")
    parser.add_argument("--config", type=Path, default=FROZEN_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    frozen = load_frozen_config(args.config)
    data = frozen["data"]
    model = frozen["model"]
    diffusion = frozen["diffusion"]
    opt = frozen["optimization"]
    val = frozen["validation20"]
    if data["total_episodes"] != 600 or data["local_streams"] != 1650:
        raise RuntimeError("dataset episode/local-stream contract drift")
    if data["sample_budget"] != opt["optimizer_updates"] * opt["global_batch_size"]:
        raise RuntimeError("sample budget arithmetic drift")
    if model["observation_steps"] != 3 or model["action_horizon"] != 8 or model["action_steps"] != 6:
        raise RuntimeError("temporal policy contract drift")
    if model["denoiser"]["global_cond_dim"] != 521 * 3:
        raise RuntimeError("global conditioning dimension drift")
    if diffusion["num_train_timesteps"] != 100 or diffusion["prediction_type"] != "epsilon":
        raise RuntimeError("diffusion contract drift")
    if val["episodes_per_task"] != 20 or val["replan_interval"] != 1:
        raise RuntimeError("validation protocol drift")

    source_root = Path(__file__).resolve().parent
    verified_sources = []
    for filename, expected_digest in frozen["artifacts"]["source_sha256"].items():
        source_path = source_root / filename
        if sha256(source_path) != expected_digest:
            raise RuntimeError(f"frozen source drift: {filename}")
        verified_sources.append(filename)
    stage_root = source_root.parents[2] / ".gaudp-stage"
    verified_upstream_sources = []
    for filename, expected_digest in frozen["upstream"]["source_sha256"].items():
        source_path = stage_root / filename
        if sha256(source_path) != expected_digest:
            raise RuntimeError(f"frozen upstream source drift: {filename}")
        verified_upstream_sources.append(filename)

    checkpoint_path = args.checkpoint or Path(frozen["artifacts"]["formal_checkpoint"])
    checkpoint = None
    if checkpoint_path.is_file():
        if sha256(checkpoint_path) != frozen["artifacts"]["formal_checkpoint_sha256"]:
            raise RuntimeError("formal checkpoint SHA-256 drift")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract") != POLICY_CONTRACT:
            raise RuntimeError("checkpoint policy contract drift")
        if int(checkpoint.get("step", -1)) != opt["optimizer_updates"]:
            raise RuntimeError("checkpoint optimizer-update budget drift")
        config = checkpoint.get("config", {})
        for key in ("obs_steps", "horizon", "action_steps", "diffusion_train_steps", "batch_size", "seed"):
            expected = {"obs_steps": 3, "horizon": 8, "action_steps": 6, "diffusion_train_steps": 100,
                        "batch_size": opt["global_batch_size"], "seed": opt["seed"]}[key]
            if config.get(key) != expected:
                raise RuntimeError(f"checkpoint config drift: {key}")

    print(json.dumps({
        "schema": "mars-control.gaudp.frozen-config-audit.v1",
        "status": "PASS",
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "checkpoint": str(checkpoint_path) if checkpoint is not None else None,
        "checkpoint_verified": checkpoint is not None,
        "policy_contract": POLICY_CONTRACT,
        "optimizer_updates": opt["optimizer_updates"],
        "sample_budget": data["sample_budget"],
        "verified_sources": verified_sources,
        "verified_upstream_sources": verified_upstream_sources,
        "validation_successes": 0,
        "validation_episodes": 80,
    }, indent=2))


if __name__ == "__main__":
    main()
