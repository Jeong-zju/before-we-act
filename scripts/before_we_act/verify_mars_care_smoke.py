#!/usr/bin/env python3
"""Issue the end-to-end CARE-on-MARS smoke receipt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from before_we_act.care_training_data import atomic_json, sha256_file
from before_we_act.mars_temporal_data import MARS_TASKS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    families = sorted((args.root / "families").glob("*/*.json"))
    if len(families) != 4:
        raise RuntimeError(f"CARE smoke expected 4 families, got {len(families)}")
    tasks = set()
    restore_rerender = 0.0
    for path in families:
        row = json.loads(path.read_text())
        tasks.add(row["task"])
        if row.get("branch_count") != 24:
            raise RuntimeError(f"incomplete branch family: {path}")
        if not all(item.get("valid") is True for item in row["candidate_legality"]):
            raise RuntimeError(f"illegal candidate family: {path}")
        for branch in row["branches"]:
            if branch.get("candidate_valid") is not True:
                raise RuntimeError(f"invalid branch: {path}")
            if set(branch.get("outcomes", {})) != {"8", "16", "32", "64"}:
                raise RuntimeError(f"incomplete horizons: {path}")
            if float(branch["candidate0_reference_action_max_abs_error"]) > 1e-6:
                raise RuntimeError(f"candidate-0 drift: {path}")
            if float(branch["replay_teammate_action_max_abs_error"]) > 1e-6:
                raise RuntimeError(f"replay drift: {path}")
            restore_rerender = max(
                restore_rerender,
                float(branch["restore_observation_max_abs_error"]),
            )
    if tasks != set(MARS_TASKS):
        raise RuntimeError(f"four-task coverage drift: {tasks}")
    scorer = json.loads((args.root / "scorer" / "status.json").read_text())
    if scorer.get("status") != "PASSED_SMOKE" or int(scorer.get("update", -1)) != 4:
        raise RuntimeError("CARE scorer resume smoke did not reach update 4")
    if int(scorer.get("resume_start_update", -1)) != 2:
        raise RuntimeError("CARE scorer resume did not start from update 2")
    validation = {}
    for task in MARS_TASKS:
        row = json.loads((args.root / "closed_loop" / f"{task}.json").read_text())
        if row.get("status") != "complete" or row.get("episodes") != 1:
            raise RuntimeError(f"closed-loop smoke incomplete: {task}")
        if int(row["rows"][0]["steps"]) != 2:
            raise RuntimeError(f"closed-loop smoke step drift: {task}")
        validation[task] = {
            "steps": 2,
            "overrides": int(row["rows"][0]["overrides"]),
        }
    deployment = args.root / "care_smoke_deployment.pt"
    atomic_json(
        args.output,
        {
            "format_version": "before-we-act.care-mars-end-to-end-smoke/1",
            "status": "PASSED",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "families": 4,
            "branches": 96,
            "scorer_train_updates": 2,
            "scorer_resume_updates": 4,
            "closed_loop": validation,
            "maximum_rerender_pixel_error": restore_rerender,
            "deployment_checkpoint_sha256": sha256_file(deployment),
        },
    )


if __name__ == "__main__":
    main()
