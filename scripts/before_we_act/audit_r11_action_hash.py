#!/usr/bin/env python3
"""Prove that loading the off-path R11 belief component cannot change W10 actions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from before_we_act.data.raw_team_windows import TASKS  # noqa: E402
from before_we_act.team_belief.base import PredictiveBeliefModel, load_r11_config  # noqa: E402
from stereo_core.no_wrist_pair_model import NoWristPAIRRoute  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_parent(payload: dict, device: torch.device) -> NoWristPAIRRoute:
    config = payload["config"]
    model = NoWristPAIRRoute(
        config.get("state_dim", 9), config.get("action_dim", 8),
        horizon=config.get("horizon", 100), d_model=config.get("d_model", 384),
        enc_layers=config.get("enc_layers", 4), dec_layers=config.get("dec_layers", 7),
        roles=config.get("roles", 4), role_rank=config.get("role_rank", 32),
        dino_model=config["dino_model"],
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


def canaries(data_root: Path):
    for task in TASKS:
        manifest = json.loads((data_root / task / "training_manifest.json").read_text(encoding="utf-8"))
        episode = next(row for row in manifest["episodes"] if row["split"] == "test")
        with h5py.File(data_root / task / episode["hdf5_path"], "r") as handle:
            agents = sorted(handle["data/observation/agents"].keys())
            global_rgb = np.asarray(handle["data/observation/images/global"][0])
            local_rgb = np.asarray(handle["data/observation/images/agent_0"][0])
            qpos = np.asarray(handle[f"data/observation/agents/{agents[0]}/qpos"][0])
        yield task, global_rgb, local_rgb, qpos


def tensor_image(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value.copy()).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    parent_path = Path(args.parent_checkpoint)
    parent_file_before = sha256(parent_path)
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent = build_parent(parent_payload, device)
    parent_state_before = state_hash(parent)
    inputs = []
    outputs_before = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for task, global_rgb, local_rgb, qpos in canaries(Path(args.data_root)):
            batch = (
                tensor_image(global_rgb, device), tensor_image(local_rgb, device),
                torch.from_numpy(qpos.copy()).unsqueeze(0).to(device).float(),
            )
            inputs.append((task, batch))
            outputs_before[task] = parent(*batch)[0].detach().cpu()

    config = load_r11_config(args.config)
    belief_payload = torch.load(args.belief_checkpoint, map_location="cpu", weights_only=False)
    belief = PredictiveBeliefModel(config).to(device)
    belief.load_state_dict(belief_payload["model"], strict=True)
    belief.eval()
    # The belief object is deliberately not registered on or called by W10.  Keeping
    # both live in the same process proves that its presence has no action-path effect.
    outputs_after = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for task, batch in inputs:
            outputs_after[task] = parent(*batch)[0].detach().cpu()
    checks = {}
    for task in TASKS:
        left, right = outputs_before[task], outputs_after[task]
        checks[task] = {
            "exact": bool(torch.equal(left, right)),
            "max_abs": float((left.float() - right.float()).abs().max()),
            "sha256_before": hashlib.sha256(left.contiguous().numpy().tobytes()).hexdigest(),
            "sha256_after": hashlib.sha256(right.contiguous().numpy().tobytes()).hexdigest(),
        }
    result = {
        "schema_version": 1,
        "round": "R11",
        "candidate_id": config.candidate_id,
        "integration": "strictly_off_path_not_registered_on_w10",
        "canary_source": "first observation from one frozen test episode per task",
        "checks": checks,
        "parent_checkpoint_sha256_before": parent_file_before,
        "parent_checkpoint_sha256_after": sha256(parent_path),
        "parent_state_sha256_before": parent_state_before,
        "parent_state_sha256_after": state_hash(parent),
    }
    result["action_hash_equal"] = all(row["exact"] for row in checks.values())
    result["parent_immutable"] = (
        result["parent_checkpoint_sha256_before"] == result["parent_checkpoint_sha256_after"]
        and result["parent_state_sha256_before"] == result["parent_state_sha256_after"]
    )
    result["passed"] = result["action_hash_equal"] and result["parent_immutable"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"action_hash_equal": result["action_hash_equal"], "parent_immutable": result["parent_immutable"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
