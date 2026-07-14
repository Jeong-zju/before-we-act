"""Independent local history/belief runtime for Research-v2."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from data.local_observation import LocalHistoryBuffer, LocalObservationPacket, LocalObservationSpec
from models.research_v2 import BeliefEncoderV2
from policies.research_v2 import (
    DecentralizedPairCoordinatorV2,
    LocalDecisionV2,
    LocalPlannerV2,
    PairDecisionV2,
)


class LocalRuntimeV2:
    """One robot's fully independent observation history and model instance."""

    def __init__(
        self,
        agent_id: int,
        *,
        spec: LocalObservationSpec,
        belief: BeliefEncoderV2,
        planner: LocalPlannerV2,
    ) -> None:
        if agent_id != planner.agent_id:
            raise ValueError("runtime/planner agent ids differ")
        self.agent_id = int(agent_id)
        self.spec = spec
        self.belief = belief.eval()
        self.planner = planner
        self.buffer = LocalHistoryBuffer(
            spec, action_dim=planner.tokenizer.cfg.action_dim, history=belief.cfg.history
        )
        if self.buffer.local_dim != belief.cfg.local_dim:
            raise ValueError("local packet contract differs from belief checkpoint")
        self.last_action: np.ndarray | None = None

    def reset(self, *, episode_sequence: int = 0) -> None:
        self.buffer.reset()
        self.last_action = None
        self.planner.reset(episode_sequence=episode_sequence)

    @torch.inference_mode()
    def observe(self, packet: LocalObservationPacket) -> torch.Tensor:
        self.buffer.append(packet, previous_action=self.last_action)
        device = next(self.belief.parameters()).device
        dtype = next(self.belief.parameters()).dtype
        batch = self.buffer.as_torch(device=str(device), add_batch_dimension=True)
        out = self.belief(
            batch["local_history"].to(dtype),
            batch["history_mask"],
            torch.tensor([self.agent_id], device=device),
            object_observation=batch["object_observation_history"].to(dtype),
            object_valid=batch["object_valid_history"],
            object_age=batch["object_age_history"].to(dtype),
            object_confidence=batch["object_confidence_history"].to(dtype),
        )
        return out["belief"][0]

    def commit(self, decision: LocalDecisionV2) -> None:
        if decision.agent_id != self.agent_id:
            raise ValueError("cannot commit another robot's decision")
        self.last_action = decision.action.detach().cpu().numpy().astype(np.float32)


class DecentralizedRuntimePairV2:
    """Single-process simulator adapter; no observation tensor is ever joined."""

    def __init__(self, runtime0: LocalRuntimeV2, runtime1: LocalRuntimeV2):
        if runtime0.agent_id != 0 or runtime1.agent_id != 1:
            raise ValueError("pair runtime requires local runtimes 0 and 1")
        self.runtimes = (runtime0, runtime1)
        self.coordinator = DecentralizedPairCoordinatorV2(runtime0.planner, runtime1.planner)

    def reset(self, *, episode_sequence: int = 0) -> None:
        for runtime in self.runtimes:
            runtime.reset(episode_sequence=episode_sequence)

    def step(self, packets: Sequence[LocalObservationPacket]) -> PairDecisionV2:
        if len(packets) != 2:
            raise ValueError("pair runtime requires one packet per robot")
        beliefs = (
            self.runtimes[0].observe(packets[0]),
            self.runtimes[1].observe(packets[1]),
        )
        decision = self.coordinator.step(beliefs)
        self.runtimes[0].commit(decision.agents[0])
        self.runtimes[1].commit(decision.agents[1])
        return decision
