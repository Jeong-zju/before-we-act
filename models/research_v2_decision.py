"""Calibrated risk and counterfactual VPI for Research-v2.

The world model predicts conditional outcomes.  Posterior probabilities are
therefore deliberately kept outside this module's conditional risk tensor and
are only introduced by :func:`counterfactual_vpi`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class CalibrationV2:
    """Frozen validation-only transformations used by the online planner.

    The names accepted by :meth:`from_mapping` include the fields emitted by
    the original stage-A calibration checkpoint.  Optional bias/communication
    fields make the contract forward-compatible with a fitted calibration
    stage without making old bundles unreadable.
    """

    quantile_scale: float = 1.0
    quantile_bias: float = 0.0
    constraint_temperature: float = 1.0
    constraint_logit_bias: float = 0.0
    posterior_temperature: float = 1.0
    posterior_variance_scale: float = 1.0
    communication_price_frozen: bool = False
    communication_price: float | None = None

    def __post_init__(self) -> None:
        finite = (
            self.quantile_scale,
            self.quantile_bias,
            self.constraint_temperature,
            self.constraint_logit_bias,
            self.posterior_temperature,
            self.posterior_variance_scale,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("calibration parameters must be finite")
        if self.quantile_scale <= 0:
            raise ValueError("quantile_scale must be positive")
        if self.constraint_temperature <= 0 or self.posterior_temperature <= 0:
            raise ValueError("calibration temperatures must be positive")
        if self.posterior_variance_scale <= 0:
            raise ValueError("posterior_variance_scale must be positive")
        if self.communication_price is not None and (
            not math.isfinite(float(self.communication_price))
            or float(self.communication_price) < 0
        ):
            raise ValueError("communication_price must be finite and non-negative")
        if self.communication_price_frozen and self.communication_price is None:
            raise ValueError("a frozen communication price requires communication_price")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CalibrationV2":
        """Parse a checkpoint ``extra`` mapping, rejecting silent bad values."""

        communication_price = payload.get(
            "communication_price", payload.get("communication_cost")
        )
        return cls(
            quantile_scale=float(payload.get("quantile_scale", 1.0)),
            quantile_bias=float(payload.get("quantile_bias", 0.0)),
            constraint_temperature=float(payload.get("constraint_temperature", 1.0)),
            constraint_logit_bias=float(
                payload.get("constraint_logit_bias", payload.get("constraint_bias", 0.0))
            ),
            posterior_temperature=float(payload.get("posterior_temperature", 1.0)),
            posterior_variance_scale=float(
                payload.get("posterior_variance_scale", 1.0)
            ),
            communication_price_frozen=bool(
                payload.get("communication_price_frozen", False)
            ),
            communication_price=(
                None if communication_price is None else float(communication_price)
            ),
        )

    def as_dict(self) -> dict[str, float | bool | None]:
        return {
            "quantile_scale": self.quantile_scale,
            "quantile_bias": self.quantile_bias,
            "constraint_temperature": self.constraint_temperature,
            "constraint_logit_bias": self.constraint_logit_bias,
            "posterior_temperature": self.posterior_temperature,
            "posterior_variance_scale": self.posterior_variance_scale,
            "communication_price_frozen": self.communication_price_frozen,
            "communication_price": self.communication_price,
        }


@dataclass(frozen=True)
class RiskV2Config:
    lambda_tail: float = 0.5
    lambda_constraint: float = 1.0
    lambda_epistemic: float = 1.0
    lambda_control: float = 0.05
    lambda_residual_uncertainty: float = 0.1
    quantile_10_index: int = 0
    quantile_50_index: int = 1
    enforce_monotone_quantiles: bool = True

    def __post_init__(self) -> None:
        if min(
            self.lambda_tail,
            self.lambda_constraint,
            self.lambda_epistemic,
            self.lambda_control,
            self.lambda_residual_uncertainty,
        ) < 0:
            raise ValueError("risk weights must be non-negative")
        if min(self.quantile_10_index, self.quantile_50_index) < 0:
            raise ValueError("risk quantile indices must be non-negative")


def calibrated_posterior_probabilities(
    logits: torch.Tensor,
    *,
    active_code_mask: torch.Tensor,
    calibration: CalibrationV2 | None = None,
) -> torch.Tensor:
    """Apply the frozen posterior temperature before masking and softmax."""

    if logits.ndim < 2:
        raise ValueError("posterior logits must end in a code dimension")
    active = active_code_mask.to(device=logits.device, dtype=torch.bool)
    if active.ndim != 1 or active.shape[0] != logits.shape[-1]:
        raise ValueError("active_code_mask/posterior code dimension mismatch")
    if not bool(active.any()):
        raise ValueError("posterior requires at least one active code")
    cal = calibration or CalibrationV2()
    calibrated = logits / cal.posterior_temperature
    calibrated = calibrated.masked_fill(~active.reshape(*([1] * (logits.ndim - 1)), -1), -torch.inf)
    return calibrated.softmax(dim=-1)


def candidate_hypothesis_risk(
    *,
    ensemble_return_quantiles: torch.Tensor,
    ensemble_constraint_logits: torch.Tensor,
    ego_actions: torch.Tensor,
    hypothesis_residual_variance: torch.Tensor | None = None,
    config: RiskV2Config | None = None,
    calibration: CalibrationV2 | None = None,
    epistemic_available: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Compute conditional ``G[k,m]`` with explicit calibration.

    Shapes are ``quantiles=[E,B,K,M,Q]``, constraints ``[E,B,K,M]`` and
    ego actions ``[B,K,H,A]``.  Residual variance may be ``[B,M]`` or
    ``[B,K,M]``.  A one-member (or explicitly non-independent) ensemble has
    no epistemic estimate; its epistemic contribution is explicitly disabled
    and reported instead of being presented as a meaningful zero estimate.
    """

    cfg = config or RiskV2Config()
    cal = calibration or CalibrationV2()
    if ensemble_return_quantiles.ndim != 5:
        raise ValueError("ensemble_return_quantiles must have shape [E,B,K,M,Q]")
    if ensemble_constraint_logits.shape != ensemble_return_quantiles.shape[:-1]:
        raise ValueError("constraint ensemble shape mismatch")
    E, B, K, M, Q = ensemble_return_quantiles.shape
    if E < 1:
        raise ValueError("world ensemble cannot be empty")
    if ego_actions.ndim != 4 or ego_actions.shape[:2] != (B, K):
        raise ValueError("ego_actions must have shape [B,K,H,A]")
    if max(cfg.quantile_10_index, cfg.quantile_50_index) >= Q:
        raise ValueError("risk quantile index is unavailable")

    calibrated_quantiles = (
        ensemble_return_quantiles * cal.quantile_scale + cal.quantile_bias
    )
    crossing = calibrated_quantiles[..., 1:] < calibrated_quantiles[..., :-1]
    if cfg.enforce_monotone_quantiles:
        # A cumulative maximum is inexpensive and preserves the meaning of the
        # lower-quantile head while guaranteeing a non-decreasing sequence.
        calibrated_quantiles = calibrated_quantiles.cummax(dim=-1).values
    q10 = calibrated_quantiles[..., cfg.quantile_10_index].mean(dim=0)
    q50_members = calibrated_quantiles[..., cfg.quantile_50_index]
    q50 = q50_members.mean(dim=0)
    tail = (q50 - q10).clamp_min(0.0)

    calibrated_constraint_logits = (
        ensemble_constraint_logits / cal.constraint_temperature
        + cal.constraint_logit_bias
    )
    constraint = calibrated_constraint_logits.sigmoid().mean(dim=0)

    available = E >= 2 if epistemic_available is None else bool(epistemic_available)
    if available and E < 2:
        raise ValueError("epistemic risk requires at least two independent world models")
    epistemic = (
        q50_members.var(dim=0, unbiased=False)
        if available
        else q50.new_zeros((B, K, M))
    )
    control = ego_actions.square().mean(dim=(-1, -2)).unsqueeze(-1).expand(B, K, M)

    if hypothesis_residual_variance is None:
        residual_uncertainty = q50.new_zeros((B, K, M))
    else:
        residual_uncertainty = hypothesis_residual_variance.to(
            device=q50.device, dtype=q50.dtype
        )
        if residual_uncertainty.shape == (B, M):
            residual_uncertainty = residual_uncertainty.unsqueeze(1).expand(B, K, M)
        elif residual_uncertainty.shape != (B, K, M):
            raise ValueError(
                "hypothesis_residual_variance must have shape [B,M] or [B,K,M]"
            )
        if not torch.isfinite(residual_uncertainty).all() or (residual_uncertainty < 0).any():
            raise ValueError("hypothesis residual variance must be finite and non-negative")

    effective_epistemic_weight = cfg.lambda_epistemic if available else 0.0
    risk = (
        -q50
        + cfg.lambda_tail * tail
        + cfg.lambda_constraint * constraint
        + effective_epistemic_weight * epistemic
        + cfg.lambda_control * control
        + cfg.lambda_residual_uncertainty * residual_uncertainty
    )
    return {
        "G": risk,
        "calibrated_return_quantiles": calibrated_quantiles,
        "q10": q10,
        "q50": q50,
        "tail": tail,
        "constraint_probability": constraint,
        "epistemic_variance": epistemic,
        "epistemic_available": torch.tensor(available, device=q50.device),
        "epistemic_weight_applied": q50.new_tensor(effective_epistemic_weight),
        "ensemble_size": torch.tensor(E, device=q50.device),
        "control_cost": control,
        "residual_uncertainty": residual_uncertainty,
        "quantile_crossing_rate_before_projection": crossing.to(q50.dtype).mean(),
    }


def counterfactual_vpi(
    G: torch.Tensor,
    hypothesis_weights: torch.Tensor,
    *,
    tail_weight: torch.Tensor | None = None,
    tail_risk: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute min-of-expectation, expectation-of-min, and VPI.

    ``hypothesis_weights`` need not be renormalized over a truncated top-M
    support.  Pass the omitted posterior probability as ``tail_weight`` and a
    conservative per-candidate ``tail_risk``.  Since the omitted plans were not
    evaluated, the reveal calculation grants them no speculative information
    benefit; it keeps the tail risk of the selected no-communication plan.
    """

    if G.ndim != 3:
        raise ValueError("G must have shape [B,K,M]")
    B, K, M = G.shape
    weights = hypothesis_weights.to(device=G.device, dtype=G.dtype)
    if weights.shape != (B, M):
        raise ValueError("hypothesis_weights must have shape [B,M]")
    if (weights < 0).any() or not torch.isfinite(weights).all():
        raise ValueError("hypothesis weights must be finite and non-negative")

    if tail_weight is None:
        tail = G.new_zeros(B)
    else:
        tail = tail_weight.to(device=G.device, dtype=G.dtype)
        if tail.shape != (B,):
            raise ValueError("tail_weight must have shape [B]")
        if (tail < 0).any() or not torch.isfinite(tail).all():
            raise ValueError("tail_weight must be finite and non-negative")
    if tail_risk is None:
        if bool((tail > 1e-7).any()):
            raise ValueError("non-zero tail_weight requires tail_risk")
        tail_cost = G.new_zeros((B, K))
    else:
        tail_cost = tail_risk.to(device=G.device, dtype=G.dtype)
        if tail_cost.shape != (B, K):
            raise ValueError("tail_risk must have shape [B,K]")
        if not torch.isfinite(tail_cost).all():
            raise ValueError("tail_risk must be finite")

    total_mass = weights.sum(dim=-1) + tail
    if bool((total_mass <= 0).any()):
        raise ValueError("hypothesis and tail mass must have a positive sum")
    weights = weights / total_mass.unsqueeze(-1)
    tail = tail / total_mass
    expected_by_candidate = (G * weights.unsqueeze(1)).sum(dim=-1)
    expected_by_candidate = expected_by_candidate + tail.unsqueeze(-1) * tail_cost
    G_no, no_index = expected_by_candidate.min(dim=-1)
    best_by_hypothesis, reveal_indices = G.min(dim=1)
    selected_tail_risk = tail_cost.gather(1, no_index.unsqueeze(-1)).squeeze(-1)
    G_reveal = (best_by_hypothesis * weights).sum(dim=-1)
    G_reveal = G_reveal + tail * selected_tail_risk
    # Mathematically non-negative with the common hypothesis support above;
    # clamp only protects against floating-point roundoff.
    vpi = (G_no - G_reveal).clamp_min(0.0)
    return {
        "G_no": G_no,
        "G_reveal": G_reveal,
        "VPI": vpi,
        "no_comm_plan_index": no_index,
        "reveal_plan_index_by_hypothesis": reveal_indices,
        "normalized_hypothesis_weights": weights,
        "normalized_tail_weight": tail,
        "selected_tail_risk": selected_tail_risk,
    }
