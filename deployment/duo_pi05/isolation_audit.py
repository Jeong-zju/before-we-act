from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import POLICY_CONTRACT, atomic_json, sha256_tree


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if not args.checkpoint.is_dir(): raise FileNotFoundError(args.checkpoint)
    # The upstream checkpoint is a directory containing params and assets.  No
    # architecture rewriting or dimensionality-changing conversion is allowed.
    params = args.checkpoint / "params"; assets = args.checkpoint / "assets"
    report = {"schema": "duobench.pi05.isolation-audit.v1", "status": "complete", "passed": params.is_dir() and assets.is_dir(), "checkpoint": str(args.checkpoint), "checkpoint_tree_sha256": sha256_tree(args.checkpoint), "checkpoint_has_params": params.is_dir(), "checkpoint_has_assets": assets.is_dir(), "policy_contract": POLICY_CONTRACT, "model_contract": {"pi05": True, "action_dim": 32, "action_horizon": 16, "paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m_lora", "freeze_filter": "Pi0Config.get_freeze_filter", "initialization": "pi05_base"}}
    atomic_json(args.output, report); print(json.dumps(report), flush=True)
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
