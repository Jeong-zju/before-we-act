"""Fail-closed residual safety contracts shared by export and deployment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class ResidualSafetyConfig:
    """Held-out calibrated limits for the optional belief residual."""

    enabled: bool = False
    max_residual_l2: float = 1.0e9
    max_belief_entropy: float = 1.0e9
    max_temporal_residual_delta_l2: float = 1.0e9
    progress_inactivity_l2: float = 0.02
    progress_patience_steps: int = 8
    progress_recovery_steps: int = 32

    def __post_init__(self) -> None:
        positive = (
            self.max_residual_l2,
            self.max_belief_entropy,
            self.max_temporal_residual_delta_l2,
            self.progress_inactivity_l2,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("residual safety limits must be positive")
        if self.progress_patience_steps < 1 or self.progress_recovery_steps < 1:
            raise ValueError("progress watchdog windows must be positive")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, object] | None
    ) -> "ResidualSafetyConfig":
        if not values:
            return cls()
        fields = {
            name: values[name]
            for name in cls.__dataclass_fields__
            if name in values
        }
        return cls(**fields)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calibrated_residual_safety(
    calibration: Mapping[str, object],
) -> ResidualSafetyConfig:
    """Build conservative limits from held-out, non-closed-loop statistics."""

    target = calibration["target_residual_l2"]
    entropy = calibration["belief_entropy"]
    temporal = calibration["temporal_residual_delta_l2"]
    if not isinstance(target, Mapping) or not isinstance(entropy, Mapping):
        raise TypeError("deployment safety calibration contract differs")
    if not isinstance(temporal, Mapping):
        raise TypeError("deployment safety temporal contract differs")
    target_p99 = float(target["p99"])
    entropy_p99 = float(entropy["p99"])
    temporal_p99 = float(temporal["p99"])
    return ResidualSafetyConfig(
        enabled=True,
        max_residual_l2=max(1e-4, 1.10 * target_p99),
        max_belief_entropy=min(
            1.0, entropy_p99 + max(0.02, 0.10 * (1.0 - entropy_p99))
        ),
        max_temporal_residual_delta_l2=max(
            1e-4, 2.0 * temporal_p99 + 1e-4
        ),
    )


class DeploymentProgressWatchdog:
    """Select the frozen base while the residual route is making no progress."""

    def __init__(self, config: ResidualSafetyConfig) -> None:
        self.config = config
        self.inactivity_streak = 0
        self.recovery_remaining = 0

    def choose_base(
        self, *, candidate_inactive: bool, base_inactive: bool
    ) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "disabled"
        if self.recovery_remaining > 0:
            self.recovery_remaining -= 1
            return True, "recovery_window"
        # Only attribute a stall to the residual when the frozen base would
        # move. If both routes hold position, the residual has not caused the
        # inactivity and an automatic override could break a legitimate wait.
        if candidate_inactive and not base_inactive:
            self.inactivity_streak += 1
        else:
            self.inactivity_streak = 0
        if self.inactivity_streak >= self.config.progress_patience_steps:
            self.recovery_remaining = self.config.progress_recovery_steps - 1
            self.inactivity_streak = 0
            return True, "residual_induced_stall"
        return False, "candidate"


__all__ = [
    "ResidualSafetyConfig",
    "DeploymentProgressWatchdog",
    "calibrated_residual_safety",
]
