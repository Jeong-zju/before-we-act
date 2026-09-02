"""Protocol-isolated CARE-v2 selector with deployment-facing safeguards.

CARE still selects the candidate with the largest conservative lower utility
bound and falls back to the frozen reference candidate.  This module only
aligns that decision with the actual execution contract:

* physical candidate legality is applied before ``argmax``;
* a one-class safety corpus uses legality-only gating instead of a random head;
* conformal corrections may be scalar or task-conditioned (Mondrian CARE);
* candidate zero is an exact, always-available fail-closed reference.

The frozen v1 selector is deliberately untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from before_we_act.care_belief import (
    CAREBeliefOutput,
    CARECalibration,
)


SAFETY_GATE_MODES = ("legality_only", "learned_probability")
REASON_REFERENCE_BELOW_DELTA = 0
REASON_OVERRIDE = 1
REASON_NO_ELIGIBLE_NONREFERENCE = 2
REASON_SELECTOR_DISABLED = 3
SELECTION_REASON_NAMES: Mapping[int, str] = {
    REASON_REFERENCE_BELOW_DELTA: "reference_below_delta",
    REASON_OVERRIDE: "override",
    REASON_NO_ELIGIBLE_NONREFERENCE: "reference_no_eligible_nonreference",
    REASON_SELECTOR_DISABLED: "reference_selector_disabled",
}


@dataclass(frozen=True)
class CARESelectionV2:
    """Auditable batch result from the isolated v2 selector."""

    selected: torch.Tensor
    best_lower: torch.Tensor
    masked_lower: torch.Tensor
    illegal: torch.Tensor
    learned_unsafe: torch.Tensor
    rejected_illegal_count: torch.Tensor
    rejected_safety_count: torch.Tensor
    reason_code: torch.Tensor

    def reason_names(self) -> list[str]:
        return [SELECTION_REASON_NAMES[int(value)] for value in self.reason_code.cpu()]


def _canonical_correction(
    lower: torch.Tensor,
    correction: float | torch.Tensor | None,
    calibration: CARECalibration,
) -> torch.Tensor:
    """Return a finite, non-negative correction for every batch row."""

    value = calibration.lower_correction if correction is None else correction
    result = torch.as_tensor(value, dtype=lower.dtype, device=lower.device)
    if result.ndim == 0:
        result = result.expand(lower.shape[0])
    if result.shape != (lower.shape[0],):
        raise ValueError("CARE v2 lower correction must be scalar or [batch]")
    if not torch.isfinite(result).all() or bool((result < 0).any()):
        raise ValueError("CARE v2 lower correction must be finite and non-negative")
    return result


def _validate_calibration(calibration: CARECalibration) -> None:
    values = (
        calibration.lower_correction,
        calibration.selector_delta,
        calibration.hard_safety_probability_max,
        calibration.nominal_simultaneous_coverage,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("CARE v2 calibration contains NaN/Inf")
    if calibration.lower_correction < 0.0 or calibration.selector_delta < 0.0:
        raise ValueError("CARE v2 correction/delta must be non-negative")
    if not 0.0 < calibration.hard_safety_probability_max < 1.0:
        raise ValueError("CARE v2 safety threshold must lie in (0,1)")
    if not 0.0 < calibration.nominal_simultaneous_coverage <= 1.0:
        raise ValueError("CARE v2 nominal coverage must lie in (0,1]")


def select_care_candidate_v2(
    output: CAREBeliefOutput,
    calibration: CARECalibration,
    candidate_legality: torch.Tensor,
    *,
    variant: str = "care",
    safety_gate_mode: str,
    lower_correction: float | torch.Tensor | None = None,
    selector_enabled: bool = True,
) -> CARESelectionV2:
    """Select independently for every batch row, failing closed to reference.

    ``candidate_legality`` is the physical controller certificate produced by
    the same candidate canonicalization used at execution.  It is intentionally
    applied before selection so one illegal proposal cannot suppress another
    legal, beneficial proposal.
    """

    _validate_calibration(calibration)
    if safety_gate_mode not in SAFETY_GATE_MODES:
        raise ValueError(f"unsupported CARE v2 safety gate: {safety_gate_mode}")
    if output.quantiles.ndim != 4 or output.hard_safety_logit.ndim != 2:
        raise ValueError("CARE v2 selector output rank differs")
    batch, candidates = output.quantiles.shape[:2]
    if output.hard_safety_logit.shape != (batch, candidates):
        raise ValueError("CARE v2 selector safety shape differs")
    if candidate_legality.shape != (batch, candidates):
        raise ValueError("CARE v2 legality must be [batch,candidate]")
    if candidate_legality.dtype != torch.bool:
        raise ValueError("CARE v2 legality must be boolean")
    if not bool(candidate_legality[:, 0].all()):
        raise ValueError("CARE v2 reference candidate must always be physically legal")
    if variant not in {"care", "reactive_only", "replay_only", "capacity"}:
        raise ValueError(f"unsupported CARE v2 scorer variant: {variant}")

    component = 0 if variant == "replay_only" else 2
    raw_lower = output.quantiles[:, :, component, 0].float()
    if not torch.isfinite(raw_lower).all():
        raise ValueError("CARE v2 lower utility contains NaN/Inf")
    correction = _canonical_correction(raw_lower, lower_correction, calibration)
    lower = raw_lower - correction[:, None]
    illegal = ~candidate_legality.to(device=lower.device)
    if safety_gate_mode == "learned_probability":
        learned_unsafe = (
            output.hard_safety_logit.float().sigmoid()
            > calibration.hard_safety_probability_max
        )
    else:
        learned_unsafe = torch.zeros_like(illegal)

    # The frozen reference is the fail-closed action.  Neither conformal
    # correction nor a noisy learned head may remove it from the candidate set.
    illegal = illegal.clone()
    learned_unsafe = learned_unsafe.clone()
    illegal[:, 0] = False
    learned_unsafe[:, 0] = False
    lower = lower.masked_fill(illegal | learned_unsafe, -torch.inf)
    lower[:, 0] = 0.0

    best_lower, best = lower.max(1)
    eligible_nonreference = (~(illegal | learned_unsafe))[:, 1:].any(1)
    override = selector_enabled & (best != 0) & (
        best_lower > calibration.selector_delta
    )
    selected = torch.where(override, best, torch.zeros_like(best))
    reason = torch.full_like(selected, REASON_REFERENCE_BELOW_DELTA)
    reason = torch.where(
        ~eligible_nonreference,
        torch.full_like(reason, REASON_NO_ELIGIBLE_NONREFERENCE),
        reason,
    )
    reason = torch.where(
        override, torch.full_like(reason, REASON_OVERRIDE), reason
    )
    if not selector_enabled:
        reason.fill_(REASON_SELECTOR_DISABLED)

    return CARESelectionV2(
        selected=selected,
        best_lower=best_lower,
        masked_lower=lower,
        illegal=illegal,
        learned_unsafe=learned_unsafe,
        rejected_illegal_count=illegal[:, 1:].sum(1),
        rejected_safety_count=learned_unsafe[:, 1:].sum(1),
        reason_code=reason,
    )


__all__ = [
    "CARESelectionV2",
    "REASON_NO_ELIGIBLE_NONREFERENCE",
    "REASON_OVERRIDE",
    "REASON_REFERENCE_BELOW_DELTA",
    "REASON_SELECTOR_DISABLED",
    "SAFETY_GATE_MODES",
    "SELECTION_REASON_NAMES",
    "select_care_candidate_v2",
]
