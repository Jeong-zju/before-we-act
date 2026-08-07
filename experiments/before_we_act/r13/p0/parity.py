#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


def config():
    return SimpleNamespace(
        multitask=False,
        tasks=(),
        task_dim=0,
        action_dims=(),
        action_dim=96,
        obs_shape={"state": (96,)},
        obs="state",
        latent_dim=96,
        mlp_dim=384,
        num_bins=1,
        episodic=True,
        dropout=0.0,
        num_q=2,
        log_std_min=-10.0,
        log_std_max=2.0,
        tau=0.01,
        simnorm_dim=8,
        num_enc_layers=2,
        enc_dim=384,
        num_channels=32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.upstream) / "tdmpc2"))
    Official = importlib.import_module("common.world_model").WorldModel
    from before_we_act.upstream_components.r13_tdmpc2.tdmpc2.common.world_model import (
        WorldModel as Local,
    )

    torch.manual_seed(1300)
    official = Official(config()).eval()
    local = Local(config()).eval()
    local.load_state_dict(official.state_dict(), strict=True)
    device = torch.device(args.device)
    official, local = official.to(device), local.to(device)
    generator = torch.Generator().manual_seed(1301)
    state = torch.randn(8, 96, generator=generator).to(device)
    action = torch.randn(8, 96, generator=generator).to(device)
    with torch.no_grad():
        left_z, right_z = official.encode(state, None), local.encode(state, None)
        pairs = {
            "encode": (left_z, right_z),
            "dynamics": (official.next(left_z, action, None), local.next(right_z, action, None)),
            "reward": (official.reward(left_z, action, None), local.reward(right_z, action, None)),
            "value": (
                official.Q(left_z, action, None, return_type="all"),
                local.Q(right_z, action, None, return_type="all"),
            ),
            "termination": (
                official.termination(left_z, None, unnormalized=True),
                local.termination(right_z, None, unnormalized=True),
            ),
        }
    exact = {name: bool(torch.equal(left, right)) for name, (left, right) in pairs.items()}
    result = {
        "schema_version": 1,
        "round": "R13",
        "candidate_id": "p0",
        "symbols": ["WorldModel.encode", "WorldModel.next", "WorldModel.reward", "WorldModel.Q", "WorldModel.termination"],
        "exact": exact,
        "max_abs": {name: float((left - right).abs().max()) for name, (left, right) in pairs.items()},
        "passed": all(exact.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
