#!/usr/bin/env python3
"""Deep, read-only integrity audit for the five fixed-revision R10 datasets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import h5py


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def audit_file(path: Path) -> tuple[int, int]:
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        action_agents = data["action"]["agents"]
        observation_agents = data["observation"]["agents"]
        images = data["observation"]["images"]
        if "global" not in images:
            raise ValueError("missing global RGB")
        agents = sorted(action_agents)
        if not agents or agents != sorted(observation_agents):
            raise ValueError("action/observation agent sets differ or are empty")
        lengths = set()
        for agent in agents:
            if not agent.startswith("panda_"):
                raise ValueError(f"unexpected agent key {agent}")
            index = agent.removeprefix("panda_")
            image_key = f"agent_{index}"
            commanded = action_agents[agent]["commanded"]
            qpos = observation_agents[agent]["qpos"]
            if image_key not in images:
                raise ValueError(f"missing RGB {image_key}")
            lengths.update((len(commanded), len(qpos), len(images[image_key])))
            if commanded.shape[1:] != (8,) or qpos.shape[1:] != (9,):
                raise ValueError(
                    f"invalid state/action shape for {agent}: "
                    f"qpos={qpos.shape}, action={commanded.shape}"
                )
            for sample in (0, -1):
                if qpos[sample].shape != (9,) or commanded[sample].shape != (8,):
                    raise ValueError(f"unreadable state/action endpoint for {agent}")
                if images[image_key][sample].shape != (480, 640, 3):
                    raise ValueError(f"invalid RGB endpoint for {image_key}")
        lengths.add(len(images["global"]))
        if len(lengths) != 1 or next(iter(lengths)) <= 0:
            raise ValueError(f"time dimensions differ: {sorted(lengths)}")
        for sample in (0, -1):
            if images["global"][sample].shape != (480, 640, 3):
                raise ValueError("invalid global RGB endpoint")
        return next(iter(lengths)), len(agents)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/workspace/datasets/robofactory_multitask"),
    )
    parser.add_argument("--expected-files", type=int, default=750)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started_at = utc_now()
    started = time.monotonic()
    paths = sorted(args.data_root.glob("*/hdf5/episode_*.hdf5"))
    errors, by_task, apparent_bytes, total_steps = [], {}, 0, 0
    for path in paths:
        task = path.relative_to(args.data_root).parts[0]
        try:
            steps, agents = audit_file(path)
            row = by_task.setdefault(task, {"files": 0, "steps": 0, "agent_streams": 0})
            row["files"] += 1
            row["steps"] += steps
            row["agent_streams"] += agents
            total_steps += steps
            apparent_bytes += path.stat().st_size
        except Exception as error:  # fail-closed summary retains every bad path
            errors.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
    passed = len(paths) == args.expected_files and not errors
    payload = {
        "schema_version": 1,
        "audit": "r10-five-task-hdf5-deep-read",
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "data_root": str(args.data_root.resolve()),
        "expected_files": args.expected_files,
        "observed_files": len(paths),
        "apparent_hdf5_bytes": apparent_bytes,
        "total_episode_steps": total_steps,
        "tasks": by_task,
        "checks": [
            "all files open with h5py",
            "action and qpos exist for every panda agent",
            "global and matching agent RGB streams exist",
            "time dimensions agree and are non-empty",
            "first and last state/action/RGB samples are readable",
            "qpos/action are 9-D/8-D and RGB is 480x640x3",
        ],
        "errors": errors,
        "passed": passed,
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "passed": passed,
                "observed_files": len(paths),
                "expected_files": args.expected_files,
                "error_count": len(errors),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
