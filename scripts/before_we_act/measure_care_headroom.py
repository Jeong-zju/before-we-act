"""Measure whether a CARE candidate family leaves any room to be selected.

The selector executes an alternative only when its calibrated lower bound clears
the improvement margin::

    Qhat^{tau_lo}_{A,k} - q_{1-alpha} > delta

Two quantities decide whether that can ever happen, and neither depends on how
well the scorer is trained:

* **headroom** -- the realized advantage of the best candidate over the nominal
  action, ``max_k A_k``.  If no candidate is ever better, a perfect scorer still
  has nothing to select.
* **the irreducible radius** -- the conformal correction that survives even a
  perfect predictor, driven by the spread between matched repeats of the same
  candidate.  A radius wider than the headroom suppresses every override.

Reporting these before training turns "the selector never fired" from a result
that costs a full training run into a gate that costs one branch-collection
pass.  The archived RoboFactory corpus fails it: ``max|A| = 0.0057`` against a
radius of ``0.0239``.

Reads the branch-family JSONs written by the branch collectors and emits a
machine-readable verdict.  No model, checkpoint, or GPU is required.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from before_we_act.care_belief import CARE_HORIZONS
from before_we_act.care_training_data import (
    DEFAULT_UTILITY_WEIGHTING,
    UTILITY_WEIGHTINGS,
    ordinary_utility,
)


REPORT_VERSION = "before-we-act.care-headroom-report/1"
# The calibrated radii the deployed selectors actually used, for reference.
KNOWN_RADII = {"robofactory_a6r1": 0.02387029491364956, "mars_v3_h16": 0.0329}


def _branch(
    family: Mapping[str, Any], candidate: int, regime: str, repeat: int
) -> Mapping[str, Any] | None:
    rows = [
        row
        for row in family["branches"]
        if int(row["candidate_id"]) == candidate
        and str(row["regime"]) == regime
        and int(row["repeat_id"]) == repeat
    ]
    return rows[0] if len(rows) == 1 else None


def family_advantages(
    family: Mapping[str, Any],
    horizon: int,
    *,
    weighting: str = DEFAULT_UTILITY_WEIGHTING,
) -> dict[int, dict[int, dict[str, float]]]:
    """Return ``{repeat: {candidate: {direct, response, total}}}`` for one horizon."""

    key = str(horizon)
    result: dict[int, dict[int, dict[str, float]]] = {}
    repeats = sorted({int(row["repeat_id"]) for row in family["branches"]})
    candidates = sorted({int(row["candidate_id"]) for row in family["branches"]})
    for repeat in repeats:
        reference_reactive = _branch(family, 0, "reactive", repeat)
        reference_replay = _branch(family, 0, "replay", repeat)
        if reference_reactive is None or reference_replay is None:
            continue
        if key not in reference_reactive["outcomes"] or key not in reference_replay["outcomes"]:
            continue
        nominal_reactive = ordinary_utility(reference_reactive["outcomes"][key], weighting)
        nominal_replay = ordinary_utility(reference_replay["outcomes"][key], weighting)
        rows: dict[int, dict[str, float]] = {}
        for candidate in candidates:
            if candidate == 0:
                continue
            reactive = _branch(family, candidate, "reactive", repeat)
            replay = _branch(family, candidate, "replay", repeat)
            if reactive is None or replay is None:
                continue
            if key not in reactive["outcomes"] or key not in replay["outcomes"]:
                continue
            direct = ordinary_utility(replay["outcomes"][key], weighting) - nominal_replay
            total = ordinary_utility(reactive["outcomes"][key], weighting) - nominal_reactive
            rows[candidate] = {
                "direct": direct,
                "response": total - direct,
                "total": total,
            }
        if rows:
            result[repeat] = rows
    return result


def _conformal_quantile(values: Sequence[float], coverage: float) -> float:
    """Split-conformal quantile with the finite-sample rank correction."""

    if not len(values):
        return math.inf
    adjusted = min(1.0, math.ceil((len(values) + 1) * coverage) / len(values))
    return float(np.quantile(np.asarray(values, dtype=np.float64), adjusted, method="higher"))


def horizon_summary(
    families: Sequence[Mapping[str, Any]],
    horizon: int,
    *,
    coverage: float,
    reference_radius: float | None,
    weighting: str = DEFAULT_UTILITY_WEIGHTING,
) -> dict[str, Any]:
    totals: list[float] = []
    family_best: list[float] = []
    family_repeat_spread: list[float] = []
    positive = 0
    exact_zero = 0
    counted = 0
    used_families = 0

    for family in families:
        advantages = family_advantages(family, horizon, weighting=weighting)
        if not advantages:
            continue
        used_families += 1
        best_per_repeat: list[float] = []
        spread_per_family: list[float] = []
        by_candidate: dict[int, list[float]] = defaultdict(list)
        for rows in advantages.values():
            values = [row["total"] for row in rows.values()]
            totals.extend(values)
            counted += len(values)
            positive += sum(1 for value in values if value > 0.0)
            exact_zero += sum(1 for value in values if value == 0.0)
            best_per_repeat.append(max(values))
            for candidate, row in rows.items():
                by_candidate[candidate].append(row["total"])
        family_best.append(float(np.mean(best_per_repeat)))
        # Half the gap between matched repeats is the noise a perfect predictor
        # still has to cover.
        for values in by_candidate.values():
            if len(values) >= 2:
                spread_per_family.append(0.5 * (max(values) - min(values)))
        if spread_per_family:
            family_repeat_spread.append(max(spread_per_family))

    if not counted:
        return {"horizon": horizon, "families": 0, "status": "NO_DATA"}

    magnitudes = np.abs(np.asarray(totals, dtype=np.float64))
    best = np.asarray(family_best, dtype=np.float64)
    irreducible = _conformal_quantile(family_repeat_spread, coverage)
    headroom = float(_conformal_quantile(best.tolist(), coverage)) if len(best) else 0.0

    summary: dict[str, Any] = {
        "horizon": horizon,
        "families": used_families,
        "candidate_rows": counted,
        "mean_abs_total": float(magnitudes.mean()),
        "median_abs_total": float(np.median(magnitudes)),
        "max_abs_total": float(magnitudes.max()),
        "fraction_positive": positive / counted,
        "fraction_exactly_zero": exact_zero / counted,
        "mean_best_candidate": float(best.mean()),
        "max_best_candidate": float(best.max()),
        "irreducible_radius": irreducible,
        "matched_repeat_families": len(family_repeat_spread),
    }

    # A perfect scorer overrides only where the best candidate clears the radius.
    def clearance(radius: float) -> dict[str, Any]:
        if not math.isfinite(radius):
            return {"radius": radius, "oracle_override_rate": 0.0, "signal_to_radius": 0.0}
        return {
            "radius": radius,
            "oracle_override_rate": float(np.mean(best > radius)),
            "signal_to_radius": float(best.max() / radius) if radius > 0 else math.inf,
        }

    summary["against_irreducible_radius"] = clearance(irreducible)
    if reference_radius is not None:
        summary["against_reference_radius"] = clearance(reference_radius)
    return summary


def load_families(roots: Iterable[Path]) -> list[dict[str, Any]]:
    families: list[dict[str, Any]] = []
    for root in roots:
        paths = sorted(root.rglob("*.json")) if root.is_dir() else [root]
        for path in paths:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping) and "branches" in value:
                families.append(dict(value))
    return families


def build_report(
    families: Sequence[Mapping[str, Any]],
    *,
    horizons: Sequence[int],
    coverage: float,
    reference_radius: float | None,
    primary_horizon: int,
    weighting: str = DEFAULT_UTILITY_WEIGHTING,
) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for family in families:
        by_task[str(family.get("task", "unknown"))].append(family)

    horizon_rows = {
        str(horizon): horizon_summary(
            families,
            horizon,
            coverage=coverage,
            reference_radius=reference_radius,
            weighting=weighting,
        )
        for horizon in horizons
    }
    task_rows = {
        task: horizon_summary(
            rows,
            primary_horizon,
            coverage=coverage,
            reference_radius=reference_radius,
            weighting=weighting,
        )
        for task, rows in sorted(by_task.items())
    }

    primary = horizon_rows.get(str(primary_horizon), {})
    verdict = "NO_DATA"
    reason = "no usable branch family was found"
    if primary.get("candidate_rows"):
        against = primary.get("against_reference_radius") or primary["against_irreducible_radius"]
        ratio = float(against["signal_to_radius"])
        rate = float(against["oracle_override_rate"])
        if ratio <= 1.0:
            verdict = "BLOCKED"
            reason = (
                "the best realized candidate never reaches the calibration radius, "
                "so even a perfect scorer would never override"
            )
        elif rate < 0.05:
            verdict = "MARGINAL"
            reason = (
                f"only {rate:.1%} of decision points admit any override; the "
                "candidate family leaves too little room to measure a gain"
            )
        else:
            verdict = "PASS"
            reason = (
                f"{rate:.1%} of decision points admit an override with headroom "
                f"{ratio:.1f}x the calibration radius"
            )

    # The deployed horizon was fixed at 16 without measuring the alternatives.
    # Recommend the horizon that admits the most overrides, so the choice is
    # made from the corpus rather than inherited.
    ranked = []
    for row in horizon_rows.values():
        if not row.get("candidate_rows"):
            continue
        against = row.get("against_reference_radius") or row["against_irreducible_radius"]
        # Rank by how often an override is admissible, then by how far the best
        # candidate clears the radius. Ties on rate would otherwise be broken by
        # the horizon's numeric value, which carries no information.
        ranked.append(
            (
                float(against["oracle_override_rate"]),
                float(against["signal_to_radius"]),
                int(row["horizon"]),
            )
        )
    recommended = max(ranked)[2] if ranked else None

    return {
        "report_version": REPORT_VERSION,
        "utility_weighting": weighting,
        "recommended_primary_horizon": recommended,
        "families": len(families),
        "tasks": sorted(by_task),
        "coverage": coverage,
        "primary_horizon": primary_horizon,
        "reference_radius": reference_radius,
        "verdict": verdict,
        "reason": reason,
        "by_horizon": horizon_rows,
        "by_task_primary_horizon": task_rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        type=Path,
        nargs="+",
        required=True,
        help="branch-family JSON files, or directories searched recursively",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument(
        "--utility-weighting",
        choices=sorted(UTILITY_WEIGHTINGS),
        default=DEFAULT_UTILITY_WEIGHTING,
        help="archived drops the collision/drop component; safety_weighted "
        "restores it",
    )
    parser.add_argument("--primary-horizon", type=int, default=16)
    parser.add_argument(
        "--reference-radius",
        type=float,
        default=None,
        help=(
            "calibration radius to judge against; omit to use the irreducible "
            f"radius implied by matched repeats. Known values: {KNOWN_RADII}"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.primary_horizon not in CARE_HORIZONS:
        raise SystemExit(f"primary horizon must be one of {CARE_HORIZONS}")
    families = load_families(args.families)
    if not families:
        raise SystemExit("no branch-family JSON was found under the requested paths")
    report = build_report(
        families,
        horizons=CARE_HORIZONS,
        coverage=args.coverage,
        reference_radius=args.reference_radius,
        primary_horizon=args.primary_horizon,
        weighting=args.utility_weighting,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("verdict", "reason", "families")}, indent=2))
    print(json.dumps(report["by_horizon"][str(args.primary_horizon)], indent=2, sort_keys=True))
    return 0 if report["verdict"] in {"PASS", "MARGINAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
