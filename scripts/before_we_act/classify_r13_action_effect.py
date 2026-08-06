#!/usr/bin/env python3
"""Fail closed when an R13 candidate modifies the frozen W12 action path."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


ALLOWED_PREFIXES = (
    "before_we_act/contracts.py",
    "before_we_act/data/world_windows.py",
    "before_we_act/world_model/",
    "before_we_act/train_team_world.py",
    "before_we_act/evaluate_team_world.py",
    "configs/before_we_act/r13_world/",
    "experiments/before_we_act/r13/",
    "scripts/before_we_act/",
    "tests/before_we_act/",
    "third_party/r13_components/",
    "LICENSES/upstream_components/",
    "docs/plans/20260725_P1_MULTI_ROBOT_MODEL_ARCHITECTURE_ACTION_GENERATION_ROADMAP_V2.0_ZH.md",
    "requirements/",
)
FROZEN_ACTION_PATHS = (
    "before_we_act/action_generator/",
    "before_we_act/train_action_generator",
    "before_we_act/evaluate_action_generator",
    "configs/before_we_act/r12_action/",
)


def git(*args: str) -> str:
    return subprocess.run(("git", *args), check=True, text=True, capture_output=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    files = [row for row in git("diff", "--name-only", f"{args.parent}...{args.head}").splitlines() if row]
    action_files = [row for row in files if row.startswith(FROZEN_ACTION_PATHS)]
    unexpected = [row for row in files if not row.startswith(ALLOWED_PREFIXES)]
    result = {
        "schema_version": 1,
        "round": "R13",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parent": args.parent,
        "head": git("rev-parse", args.head).strip(),
        "changed_files": files,
        "frozen_action_path_files": action_files,
        "unexpected_files": unexpected,
        "planner_enabled": False,
        "rerank_enabled": False,
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
