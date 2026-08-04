#!/usr/bin/env python3
"""Fail closed when an R11 branch touches the frozen W10 action path."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


ALLOWED_PREFIXES = (
    "before_we_act/",
    "configs/before_we_act/r11_belief/",
    "experiments/before_we_act/r11/",
    "scripts/before_we_act/",
    "tests/before_we_act/",
    "third_party/r11_components/",
    "LICENSES/upstream_components/",
    "docs/plans/20260725_P1_MULTI_ROBOT_MODEL_ARCHITECTURE_ACTION_GENERATION_ROADMAP_V2.0_ZH.md",
    "requirements/",
)
ACTION_PATH_PREFIXES = ("stereo_core/", "train/", "eval/", "rollout/", "inference/")


def run(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    files = [line for line in run("git", "diff", "--name-only", f"{args.parent}...{args.head}").splitlines() if line]
    action_files = [path for path in files if path.startswith(ACTION_PATH_PREFIXES)]
    unexpected = [path for path in files if not path.startswith(ALLOWED_PREFIXES)]
    result = {
        "schema_version": 1,
        "round": "R11",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parent": args.parent,
        "head": run("git", "rev-parse", args.head).strip(),
        "changed_files": files,
        "action_path_files": action_files,
        "unexpected_files": unexpected,
        "classification": "strictly_off_path" if not action_files and not unexpected else "action_affecting_or_out_of_scope",
        "passed": not action_files and not unexpected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
