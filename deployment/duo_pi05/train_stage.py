from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from .common import atomic_json, sha256_tree


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--openpi", type=Path, required=True); parser.add_argument("--checkpoint-base-dir", type=Path, required=True); parser.add_argument("--assets-base-dir", type=Path, required=True); parser.add_argument("--exp-name", required=True); parser.add_argument("--updates", type=int, required=True); parser.add_argument("--workers", type=int, required=True); parser.add_argument("--save-every", type=int, required=True); parser.add_argument("--keep-period", type=int, required=True); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    asset = args.assets_base_dir / "pi05_duobench_lora/duobench"; asset.mkdir(parents=True, exist_ok=True)
    norm = Path(os.environ.get("DUO_PI05_RUN", "/workspace/runs/pi05_duo")) / "assets/norm_stats.json"
    if not norm.is_file(): raise FileNotFoundError(norm)
    shutil.copy2(norm, asset / "norm_stats.json")
    checkpoint_root = args.checkpoint_base_dir / "pi05_duobench_lora" / args.exp_name
    has_checkpoint = any(path.is_dir() and path.name.isdigit() for path in checkpoint_root.glob("*"))
    command = [
        os.environ.get("DUO_PI05_PYTHON", "/workspace/venvs/openpi/bin/python"),
        str(args.openpi / "scripts/train.py"), "pi05_duobench_lora",
        "--checkpoint-base-dir", str(args.checkpoint_base_dir),
        "--assets-base-dir", str(args.assets_base_dir), "--exp-name", args.exp_name,
        "--batch-size", "128", "--num-workers", str(args.workers),
        "--num-train-steps", str(args.updates), "--save-interval", str(args.save_every),
        "--keep-period", str(args.keep_period), "--fsdp-devices", "1", "--no-wandb-enabled",
        *( ["--resume"] if has_checkpoint else ["--overwrite"] ),
    ]
    code = subprocess.call(command, cwd=args.openpi, env=os.environ.copy())
    if code: raise SystemExit(code)
    checkpoint = args.checkpoint_base_dir / "pi05_duobench_lora" / args.exp_name / str(args.updates - 1)
    if not checkpoint.is_dir(): raise FileNotFoundError(f"OpenPI checkpoint missing: {checkpoint}")
    run = Path(os.environ.get("DUO_PI05_RUN", "/workspace/runs/pi05_duo")); stage = "smoke" if args.smoke else "formal"
    status = {"schema": "duobench.pi05.training-status.v1", "status": "complete", "updates": args.updates, "step": args.updates - 1, "checkpoint_step": args.updates - 1, "checkpoint": str(checkpoint), "checkpoint_tree_sha256": sha256_tree(checkpoint), "all_550_demonstrations_no_split": True, "global_batch_size": 128, "devices": 4, "model": {"pi05": True, "action_dim": 32, "action_horizon": 16, "paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m_lora", "freeze_filter": "Pi0Config.get_freeze_filter"}, "smoke": args.smoke}
    atomic_json(run / stage / "status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__": main()
