"""Create the one frozen, random, training-disjoint 100-seed protocol per task."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from five_task_contract import CANONICAL_TASKS


def main() -> None:
    out = Path("/workspace/RoboFactory/runs/strict640x480_v2/heldout_seeds")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260728)
    # Formal demonstrations use seeds below one million.  This high range is
    # intentionally sampled without replacement and shared by every method.
    candidates = range(1_000_000_000, 2_000_000_000)
    for task in CANONICAL_TASKS:
        seeds = rng.sample(candidates, 100)
        payload = {
            "version": "strict640x480-v2",
            "task": task,
            "seeds": seeds,
            "selection_method": "frozen random sample without replacement; Python Random(20260728)",
            "training_seed_range": [400000, 800999],
            "training_seed_overlap": False,
        }
        raw = json.dumps(payload, indent=2).encode() + b"\n"
        (out / f"{task}.json").write_bytes(raw)
        print(task, hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
