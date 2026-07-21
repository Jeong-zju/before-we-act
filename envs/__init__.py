"""Simulation environments and environment-only runtime utilities.

This package intentionally has no dependency on :mod:`data` or :mod:`models`.
Dataset collection may depend on this package, but the dependency must never
point in the opposite direction.
"""

from envs.visual_required_env import (
    CAMERA_NAMES,
    VISUAL_REQUIRED_TASKS,
    VisualRequiredEnv,
    VisualRequiredEnvConfig,
)

__all__ = [
    "CAMERA_NAMES",
    "VISUAL_REQUIRED_TASKS",
    "VisualRequiredEnv",
    "VisualRequiredEnvConfig",
]
