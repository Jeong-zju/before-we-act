"""Proprioception-only runtime adapter for the accepted action prior."""

from __future__ import annotations

from collections import deque
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch

from models.wam import ActionPrior, RWMARWorldModel, WorldModelSequenceInputs


class ActionPriorPolicy:
    def __init__(
        self,
        world_model: RWMARWorldModel,
        prior: ActionPrior,
        *,
        fixed_actions: Mapping[int, float] | None = None,
        distillation_callback: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], None
        ]
        | None = None,
        clock: Any = time.perf_counter,
    ) -> None:
        if prior.config.feature_dim != world_model.planning_feature_dim:
            raise ValueError("action-prior feature dimension does not match world model")
        if prior.config.action_dim != world_model.config.action_dim:
            raise ValueError("action-prior action dimension does not match world model")
        self.world_model = world_model.eval()
        self.prior = prior.eval()
        self.device = next(world_model.parameters()).device
        self.dtype = next(world_model.parameters()).dtype
        self.fixed_actions = {
            int(index): float(value) for index, value in (fixed_actions or {}).items()
        }
        self.distillation_callback = distillation_callback
        for index, value in self.fixed_actions.items():
            if not 0 <= index < world_model.config.action_dim:
                raise ValueError("fixed action index is out of range")
            if not -1.0 <= value <= 1.0:
                raise ValueError("fixed action values must be in [-1,1]")
        horizon = world_model.config.history_horizon
        self._states: deque[np.ndarray] = deque(maxlen=horizon)
        self._actions: deque[np.ndarray] = deque(maxlen=max(horizon - 1, 1))
        self.clock = clock
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self) -> None:
        self._states.clear()
        self._actions.clear()
        self.last_diagnostics = {}

    @torch.inference_mode()
    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        keys = {str(key) for key in observation}
        if "privileged_state" in observation:
            raise RuntimeError("privileged_state leakage into action prior")
        if "proprioception" not in observation:
            raise KeyError("action prior requires observation['proprioception']")
        state = np.asarray(observation["proprioception"], dtype=np.float32)
        expected = self.world_model.config.state_dim
        if state.shape != (expected,) or not np.isfinite(state).all():
            raise ValueError(f"proprioception must be finite with shape {(expected,)}")
        self._states.append(state.copy())
        start = self.clock()
        hidden, current_state, features = self.world_model.encode_planning_history(
            self._history()
        )
        if self.distillation_callback is not None:
            self.distillation_callback(features, hidden, current_state)
        action = self.prior.deterministic_action(features)[0]
        action = action.clamp(-1.0, 1.0).clone()
        for index, value in self.fixed_actions.items():
            action[index] = value
        result = action.cpu().numpy().astype(np.float32)
        self._actions.append(result.copy())
        self.last_diagnostics = {
            "latency_ms": float((self.clock() - start) * 1000.0),
            "executed_mode": "action_prior",
            "planned_mode": "none",
            "plan_executed": False,
            "deadline_exceeded": False,
            "fallback_reason": "none",
            "observation_keys": sorted(keys),
            "privileged_state_seen": False,
        }
        return result

    def _history(self) -> WorldModelSequenceInputs:
        config = self.world_model.config
        count = len(self._states)
        offset = config.history_horizon - count
        states = torch.zeros(
            1,
            config.history_horizon,
            config.state_dim,
            device=self.device,
            dtype=self.dtype,
        )
        states[0, offset:] = torch.as_tensor(
            np.stack(self._states), device=self.device, dtype=self.dtype
        )
        past_actions = torch.zeros(
            1,
            config.history_horizon - 1,
            config.action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        needed = max(count - 1, 0)
        if needed:
            past_actions[0, offset:] = torch.as_tensor(
                np.stack(list(self._actions)[-needed:]),
                device=self.device,
                dtype=self.dtype,
            )
        valid_mask = torch.zeros(
            1, config.history_horizon, device=self.device, dtype=torch.bool
        )
        valid_mask[0, offset:] = True
        return WorldModelSequenceInputs(states, past_actions, valid_mask)


__all__ = ["ActionPriorPolicy"]
