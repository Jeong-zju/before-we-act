#!/usr/bin/env python3
"""Read-only branch/source/foundation preflight for one deployed R11 candidate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from before_we_act.r11_vendor import (
    validate_asset_bundle_receipt,
    validate_asset_receipt,
    verify_vendor_checkout,
)
from before_we_act.train_r11_candidate import atomic_json, sha256_file


EXPECTED_FILES = {
    "A": ("r11_vjepa21_ac_refine.py", "a-vjepa21-ac-refine.json", "vjepa2"),
    "B": ("r11_dreamzero_wan22_wam.py", "b-dreamzero-wan22-wam.json", "dreamzero"),
    "C": ("r11_cosmos_policy_latent.py", "c-cosmos-policy-latent.json", "cosmos-predict2.5"),
    "D": ("r11_lawam_subgoal_flow.py", "d-lawam-latent-subgoal.json", "lawam"),
}


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def validate_assets(config: dict) -> dict:
    candidate = config["candidate"]
    if candidate == "A":
        return {
            name: validate_asset_receipt(asset["receipt"], {"source_url": asset["source_url"]})
            for name, asset in config["assets"].items()
        }
    foundation = config["foundation"]
    if candidate == "B":
        expected = {
            "repositories": {
                foundation["wan22_repo"]: foundation["wan22_revision"],
                foundation["wan21_support_repo"]: foundation["wan21_support_revision"],
                foundation["tokenizer_repo"]: foundation["tokenizer_revision"],
            },
            "license": "Apache-2.0",
        }
    elif candidate == "C":
        expected = {
            "repositories": {foundation["repo"]: foundation["revision"]},
            "license": "NVIDIA Open Model License Agreement",
            "license_acceptance": "verified_noninteractive",
            "attribution": "Built on NVIDIA Cosmos",
        }
    else:
        expected = {
            "repositories": foundation["repositories"],
            "licenses": foundation["licenses"],
            "task_sft_checkpoint": "none",
        }
    return validate_asset_bundle_receipt(foundation["receipt"], expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=tuple(EXPECTED_FILES), required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-assets", action="store_true")
    args = parser.parse_args()

    root = args.worktree.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    run_manifest_path = args.run_manifest.resolve(strict=True)
    config = json.loads(config_path.read_text())
    manifest = json.loads(run_manifest_path.read_text())
    candidate = args.candidate
    expected_model_file, expected_config_file, expected_vendor = EXPECTED_FILES[candidate]
    if config.get("candidate") != candidate:
        raise ValueError("candidate config identity differs")
    if manifest["candidates"][candidate]["model"] != config.get("model"):
        raise ValueError("candidate model differs from immutable run manifest")
    branch = git(root, "branch", "--show-current")
    commit = git(root, "rev-parse", "HEAD")
    if branch != manifest["branches"][candidate]:
        raise ValueError("deployed candidate branch differs")
    if git(root, "status", "--porcelain"):
        raise ValueError("deployed candidate worktree is dirty")
    base = manifest["base"]["commit"]
    subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", base, commit],
        check=True,
    )
    changed = git(root, "diff", "--name-only", f"{base}..{commit}").splitlines()
    model_files = [
        Path(path).name
        for path in changed
        if path.startswith("before_we_act/r11_")
        and Path(path).name
        not in {
            "r11_data.py",
            "r11_vendor.py",
            "r11_registry.py",
        }
    ]
    if sorted(set(model_files)) != [expected_model_file]:
        raise ValueError(f"candidate model diff is cross-contaminated: {model_files}")
    config_files = [
        Path(path).name
        for path in changed
        if path.startswith("configs/before_we_act/r11/")
    ]
    if sorted(set(config_files)) != [expected_config_file]:
        raise ValueError(f"candidate config diff is cross-contaminated: {config_files}")
    third_party_roots = {
        Path(path).parts[2]
        for path in changed
        if path.startswith("third_party/r11/") and len(Path(path).parts) > 2
    }
    if third_party_roots != {expected_vendor}:
        raise ValueError(f"candidate source diff is cross-contaminated: {third_party_roots}")

    vendor = Path(os.environ.get(config["vendor_env"], config["vendor_default"])).resolve()
    source = verify_vendor_checkout(root / config["source_receipt"], vendor)
    assets = None if args.skip_assets else validate_assets(config)
    payload = {
        "format_version": "before-we-act.r11.candidate_preflight/1",
        "status": "PASSED",
        "candidate": candidate,
        "model": config["model"],
        "branch": branch,
        "commit": commit,
        "base_commit": base,
        "upstream_commit": source["upstream_commit"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source": source,
        "assets": assets,
        "assets_checked": not args.skip_assets,
        "completed_at_epoch": time.time(),
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload | {"source": "verified", "assets": "verified" if assets else "skipped"}, sort_keys=True))


if __name__ == "__main__":
    main()
