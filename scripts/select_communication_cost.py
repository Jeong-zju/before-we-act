"""Freeze the private-gates validation Pareto communication work point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.calibration import ParetoPoint, select_pareto_utopia  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-sweep", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.validation_sweep)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if str(payload.get("split", "")) not in {"val", "validation"}:
        raise ValueError("Pareto selection may only consume a validation sweep")
    points = [ParetoPoint.from_mapping(item) for item in payload["points"]]
    result = {
        **select_pareto_utopia(points),
        "source": str(source.resolve()),
        "split": "validation",
        "frozen_for_test": True,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
