"""Minimal joint-action adapter of the official CVPR 2026 AAC code.

The upstream implementation operates on multiple sampled end-effector chunks.
W12 emits normalized joint-position chunks, so the same Gaussian/Bernoulli
entropy elbow and mean-nearest sample rules are applied per arm here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AACSelection:
    chunk_size: int
    sample_index: int
    step_entropy: np.ndarray
    chunk_mean_entropy: np.ndarray


def _safe_logdet_cov(values: np.ndarray, eps: float = 1e-6) -> float:
    values = np.asarray(values)
    samples, dimensions = values.shape
    if samples <= 1:
        return float("-inf")
    covariance = np.cov(values, rowvar=False) + eps * np.eye(dimensions)
    try:
        factor = np.linalg.cholesky(covariance)
        return float(2.0 * np.sum(np.log(np.diag(factor))))
    except np.linalg.LinAlgError:
        sign, logdet = np.linalg.slogdet(covariance)
        return float(logdet) if sign > 0 else float("-inf")


def gaussian_entropy_from_samples(values: np.ndarray) -> float:
    values = np.asarray(values)
    samples, dimensions = values.shape
    if samples <= 1:
        return 0.0
    logdet = _safe_logdet_cov(values)
    return float(0.5 * (dimensions * np.log(2 * np.pi * np.e) + logdet))


def bernoulli_entropy_from_samples(values: np.ndarray) -> float:
    probability = float(np.clip(np.asarray(values).mean(), 1e-9, 1 - 1e-9))
    return float(
        -probability * np.log(probability)
        - (1.0 - probability) * np.log(1.0 - probability)
    )


def select_joint_action_chunk(
    normalized_actions: np.ndarray,
    *,
    selection_horizon: int = 16,
    minimum_chunk_size: int = 2,
) -> AACSelection:
    """Select the entropy elbow and representative sampled W12 action chunk.

    Args:
        normalized_actions: ``[samples, agents, horizon, 8]`` joint targets.
    """

    actions = np.asarray(normalized_actions, dtype=np.float64)
    if actions.ndim != 4 or actions.shape[-1] != 8:
        raise ValueError("AAC actions must be [samples,agents,horizon,8]")
    samples, agents, horizon, _ = actions.shape
    if samples < 2 or agents < 1:
        raise ValueError("AAC requires at least two samples and one agent")
    selected_horizon = min(int(selection_horizon), int(horizon))
    if not 2 <= int(minimum_chunk_size) <= selected_horizon:
        raise ValueError("AAC minimum chunk size differs from the available horizon")
    if not bool(np.isfinite(actions).all()):
        raise ValueError("AAC received non-finite actions")

    step_entropy = []
    for timestep in range(selected_horizon):
        total = 0.0
        for agent in range(agents):
            total += gaussian_entropy_from_samples(
                actions[:, agent, timestep, :7]
            )
            total += bernoulli_entropy_from_samples(
                actions[:, agent, timestep, 7] > 0.0
            )
        step_entropy.append(total)
    step_entropy_array = np.asarray(step_entropy, dtype=np.float64)
    chunk_mean = np.cumsum(step_entropy_array) / np.arange(
        1, selected_horizon + 1, dtype=np.float64
    )
    entropy_elbow = int(np.argmax(np.diff(chunk_mean))) + 1
    chunk_size = max(entropy_elbow, int(minimum_chunk_size))

    flattened = actions[:, :, :chunk_size].reshape(samples, -1)
    mean = flattened.mean(axis=0)
    sample_index = int(np.argmin(np.linalg.norm(flattened - mean, axis=1)))
    return AACSelection(
        chunk_size=chunk_size,
        sample_index=sample_index,
        step_entropy=step_entropy_array,
        chunk_mean_entropy=chunk_mean,
    )
