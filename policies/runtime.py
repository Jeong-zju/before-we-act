"""Checkpoint-to-sensor runtime for decentralized FE-PC-WAM.

This module is intentionally small: it turns each robot's own
``LocalObservationPacket`` into the exact history used offline, encodes four
belief slots with shared weights, and passes one ego input to each local
planner.  The pair wrapper only exists for single-process simulation; on real
robots the two ``LocalAgentPlanner`` instances run on separate machines and
exchange the same ``PlanMessage`` packets through the transport layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from data.local_observation import (
    LocalHistoryBuffer,
    LocalObservationPacket,
    LocalObservationSpec,
)
from models.communication import CommunicationConfig, VPICommunicationTrigger
from models.decentralized import (
    EgoLocalWAM,
    EgoLocalWAMConfig,
    LocalIntentionConfig,
    LocalIntentionPosterior,
)
from models.free_energy import FreeEnergyConfig, FreeEnergyEvaluator
from models.plan_tokenizer import (
    ActionOnlyPlanTokenizer,
    ActionOnlyPlanTokenizerConfig,
    PlanCodeSupport,
)
from models.slot_encoder import LocalBeliefSlotEncoder, LocalBeliefSlotEncoderConfig
from policies.decentralized import (
    DecentralizedPairCoordinator,
    DecentralizedPolicyConfig,
    LocalPlannerInput,
    PairDecision,
)
from train.checkpoint import file_sha256, load_checkpoint, require_plan_code_support


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    policy: DecentralizedPolicyConfig = field(default_factory=DecentralizedPolicyConfig)
    progress_target: float = 1.0
    force_limit: float = 1.0
    alpha_goal: float = 1.0
    alpha_safety: float = 2.0
    alpha_collab: float = 1.0
    alpha_unc: float = 0.5
    alpha_ctrl: float = 0.05
    lambda_bits: float = 1e-4
    lambda_delay: float = 0.05
    delay_steps: float = 1.0
    delta_margin: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.progress_target):
            raise ValueError("progress_target must be finite")
        if self.force_limit <= 0 or self.lambda_bits < 0 or self.lambda_delay < 0:
            raise ValueError("runtime limits/cost weights are invalid")
        if self.delay_steps < 0:
            raise ValueError("delay_steps cannot be negative")


class DecentralizedRuntime:
    """Two local runtimes sharing weights but never sharing observations."""

    def __init__(
        self,
        *,
        spec: LocalObservationSpec,
        belief: LocalBeliefSlotEncoder,
        coordinator: DecentralizedPairCoordinator,
    ) -> None:
        self.spec = spec
        self.belief = belief
        self.coordinator = coordinator
        history = int(belief.cfg.history)
        action_dim = int(coordinator.planners[0].tokenizer.cfg.action_dim)
        expected_local_dim = spec.model_observation_dim + action_dim
        if expected_local_dim != belief.cfg.local_dim:
            raise ValueError(
                "local observation spec/action dimension do not match the belief checkpoint"
            )
        self.buffers = (
            LocalHistoryBuffer(spec, action_dim=action_dim, history=history),
            LocalHistoryBuffer(spec, action_dim=action_dim, history=history),
        )
        self._last_actions: tuple[np.ndarray, np.ndarray] | None = None
        self._device = next(belief.parameters()).device
        self._dtype = next(belief.parameters()).dtype
        self.belief.eval()

    @classmethod
    def from_checkpoints(
        cls,
        *,
        plan_checkpoint: str | Path,
        belief_checkpoint: str | Path,
        wam_checkpoint: str | Path,
        intention_checkpoint: str | Path,
        config: RuntimeConfig | None = None,
    ) -> "DecentralizedRuntime":
        cfg = config or RuntimeConfig()
        device = _resolve_device(cfg.device)
        plan_state = load_checkpoint(plan_checkpoint, expected_stage="plan", map_location=device)
        belief_state = load_checkpoint(
            belief_checkpoint, expected_stage="belief", map_location=device
        )
        wam_state = load_checkpoint(
            wam_checkpoint, expected_stage=("wam", "wam_robust"), map_location=device
        )
        intention_state = load_checkpoint(
            intention_checkpoint, expected_stage="intention", map_location=device
        )
        states = (plan_state, belief_state, wam_state, intention_state)
        support = PlanCodeSupport.from_dict(require_plan_code_support(plan_state))
        _require_same_support(support, states[1:])
        _require_compatible_dataset_contract(states)
        _require_upstream_file(belief_state, "plan", plan_checkpoint)
        _require_upstream_file(wam_state, "plan", plan_checkpoint)
        _require_upstream_file(wam_state, "belief", belief_checkpoint)
        _require_upstream_file(intention_state, "plan", plan_checkpoint)
        _require_upstream_file(intention_state, "belief", belief_checkpoint)
        if wam_state["stage"] == "wam":
            _require_upstream_file(intention_state, "wam", wam_checkpoint)
        else:
            _require_upstream_file(wam_state, "intention", intention_checkpoint)

        tokenizer = ActionOnlyPlanTokenizer(
            ActionOnlyPlanTokenizerConfig(**plan_state["model_config"])
        ).to(device)
        tokenizer.load_state_dict(plan_state["model_state_dict"], strict=True)
        belief = LocalBeliefSlotEncoder(
            LocalBeliefSlotEncoderConfig(**belief_state["model_config"])
        ).to(device)
        belief.load_state_dict(belief_state["model_state_dict"], strict=True)
        wam = EgoLocalWAM(EgoLocalWAMConfig(**wam_state["model_config"])).to(device)
        wam.load_state_dict(wam_state["model_state_dict"], strict=True)
        intention = LocalIntentionPosterior(
            LocalIntentionConfig(**intention_state["model_config"])
        ).to(device)
        intention.load_state_dict(intention_state["model_state_dict"], strict=True)
        for model in (tokenizer, belief, wam, intention):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        normalization = plan_state.get("normalization", {})
        try:
            action_mean = torch.as_tensor(normalization["action_mean"])
            action_std = torch.as_tensor(normalization["action_std"])
        except KeyError as exc:
            raise ValueError(" plan checkpoint lacks action normalization") from exc

        free_energy = FreeEnergyEvaluator(
            FreeEnergyConfig(
                goal_y=cfg.progress_target,
                force_limit=cfg.force_limit,
                alpha_goal=cfg.alpha_goal,
                alpha_safety=cfg.alpha_safety,
                alpha_collab=cfg.alpha_collab,
                alpha_unc=cfg.alpha_unc,
                alpha_ctrl=cfg.alpha_ctrl,
            )
        )
        communication = VPICommunicationTrigger(
            CommunicationConfig(
                codebook_size=tokenizer.cfg.codebook_size,
                residual_dim=tokenizer.cfg.latent_dim,
                lambda_bits=cfg.lambda_bits,
                lambda_delay=cfg.lambda_delay,
                delay_steps=cfg.delay_steps,
                delta_margin=cfg.delta_margin,
            )
        )
        coordinator = DecentralizedPairCoordinator.from_shared_components(
            tokenizer=tokenizer,
            wam=wam,
            intention=intention,
            support=support,
            free_energy=free_energy,
            communication=communication,
            config=cfg.policy,
            action_mean=action_mean,
            action_std=action_std,
        )
        spec_state = plan_state["dataset"].get("local_observation_spec")
        if not isinstance(spec_state, Mapping):
            raise ValueError(
                " checkpoint lacks local_observation_spec; retrain with the current pipeline"
            )
        spec = LocalObservationSpec(**dict(spec_state))
        return cls(spec=spec, belief=belief, coordinator=coordinator)

    def reset(self, *, seed: int | None = None) -> None:
        """Reset histories and controller state for a new paired episode."""

        for buffer in self.buffers:
            buffer.reset()
        self.coordinator.reset(seed=seed)
        self._last_actions = None

    @torch.inference_mode()
    def step(self, packets: Sequence[LocalObservationPacket]) -> PairDecision:
        if len(packets) != 2:
            raise ValueError("runtime step requires one local packet per robot")
        for agent_id, packet in enumerate(packets):
            previous = None if self._last_actions is None else self._last_actions[agent_id]
            self.buffers[agent_id].append(packet, previous_action=previous)

        local_inputs = []
        for agent_id, buffer in enumerate(self.buffers):
            model_input = buffer.as_torch(device=str(self._device), add_batch_dimension=True)
            slots = self.belief(
                model_input["local_history"].to(dtype=self._dtype),
                model_input["history_mask"],
                torch.tensor([agent_id], device=self._device, dtype=torch.long),
                object_observation=model_input["object_observation_history"].to(
                    dtype=self._dtype
                ),
                object_valid=model_input["object_valid_history"],
                object_age=model_input["object_age_history"].to(dtype=self._dtype),
                object_confidence=model_input["object_confidence_history"].to(
                    dtype=self._dtype
                ),
            )["slots"][0]
            metadata = torch.zeros(
                self.coordinator.planners[agent_id].intention.cfg.message_metadata_dim,
                device=self._device,
                dtype=self._dtype,
            )
            local_inputs.append(LocalPlannerInput(slots, metadata))

        decision = self.coordinator.step(local_inputs)
        self._last_actions = (
            decision.agents[0].action.detach().cpu().numpy().astype(np.float32),
            decision.agents[1].action.detach().cpu().numpy().astype(np.float32),
        )
        return decision


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _require_same_support(
    expected: PlanCodeSupport, checkpoints: Sequence[Mapping[str, object]]
) -> None:
    for checkpoint in checkpoints:
        actual = PlanCodeSupport.from_dict(require_plan_code_support(checkpoint))
        for name in ("counts", "probabilities", "residual_mean", "residual_std"):
            if not torch.equal(getattr(expected, name), getattr(actual, name)):
                raise ValueError(
                    f"{checkpoint.get('stage')} checkpoint plan support differs from plan checkpoint"
                )
        if expected.min_count != actual.min_count:
            raise ValueError("checkpoint plan support min_count differs")


def _require_compatible_dataset_contract(
    checkpoints: Sequence[Mapping[str, object]],
) -> None:
    first = checkpoints[0].get("dataset")
    if not isinstance(first, Mapping):
        raise ValueError(" checkpoint lacks dataset contract metadata")
    keys = (
        "schema_version",
        "history",
        "horizon",
        "model_observation_dim",
        "local_history_dim",
        "action_dim",
        "local_observation_spec",
        "local_contact_semantics",
        "local_force_semantics",
        "local_force_units",
        "local_force_scale_newtons",
        "local_sensor_provenance",
    )
    for checkpoint in checkpoints[1:]:
        dataset = checkpoint.get("dataset")
        if not isinstance(dataset, Mapping):
            raise ValueError(" checkpoint lacks dataset contract metadata")
        for key in keys:
            if dataset.get(key) != first.get(key):
                raise ValueError(
                    f"checkpoint dataset contract differs for {key!r}: "
                    f"{first.get(key)!r} vs {dataset.get(key)!r}"
                )


def _require_upstream_file(
    checkpoint: Mapping[str, object],
    name: str,
    path: str | Path,
) -> None:
    upstream = checkpoint.get("upstream")
    reference = upstream.get(name) if isinstance(upstream, Mapping) else None
    expected = reference.get("sha256") if isinstance(reference, Mapping) else None
    actual = file_sha256(path)
    if expected != actual:
        raise ValueError(
            f"{checkpoint.get('stage')} checkpoint lineage mismatch for {name}: "
            f"expected {expected!r}, actual {actual!r}"
        )
