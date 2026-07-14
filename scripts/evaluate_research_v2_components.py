"""Evaluate frozen Research-v2 component predictions stored in one NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.research_v2 import (  # noqa: E402
    binary_calibration_metrics,
    grouped_branch_regret,
    proposal_oracle_coverage,
    return_quantile_metrics,
    vpi_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--frozen-validation-config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.split == "test" and not args.frozen_validation_config:
        raise ValueError("test evaluation requires a frozen validation config")
    arrays = np.load(args.predictions)
    report = {
        "split": args.split,
        "return": return_quantile_metrics(arrays["return_quantiles"], arrays["return_target"]),
        "constraint": binary_calibration_metrics(
            arrays["constraint_probability"], arrays["constraint_target"]
        ),
        "branch": grouped_branch_regret(
            arrays["branch_predicted_score"], arrays["branch_return"], arrays["branch_group_id"]
        ),
        "proposal_topk_coverage": proposal_oracle_coverage(
            arrays["proposal_topk_codes"], arrays["proposal_oracle_code"]
        ),
        "vpi": vpi_calibration(arrays["predicted_vpi"], arrays["realized_communication_value"]),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
