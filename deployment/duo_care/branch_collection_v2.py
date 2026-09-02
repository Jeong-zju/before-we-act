"""Policy-independent same-snapshot CARE branch kernel for DuoBench.

The kernel owns CARE's causal semantics and delegates only two benchmark/model
boundaries:

``ProposalProvider``
    A frozen, strict-local policy adapter.  It supplies reference/base chunks,
    local memory, and proposal-time qpos.  ACT, DINO--Transformer, and future
    reference policies can implement the same protocol.

``BranchEnvironment``
    A DuoBench simulator adapter with exact snapshot/restore (including task
    wrapper state and RNG), physical stepping, and privileged *offline-only*
    outcome metrics.

Only one focal arm is intervened on, only for the first control step.  The
other arm either reacts through the frozen policy or replays its exact
candidate-zero physical commands.  This is the registered CARE direct/response
factorization, not a new method variant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SHA256,
    summarize_action_canonicalization,
)

from deployment.duo_care.branch_signal import (
    BRANCH_CANDIDATES,
    BRANCH_REPEATS,
    BranchGate,
    HORIZONS,
    derive_branch_seed,
    outcome_table,
    stable_tree_hash,
)
from before_we_act.care_behavior_candidates import BehaviorCandidateConfig
from before_we_act.care_candidate_family import (
    BEHAVIOR_FAMILY,
    CANDIDATE_FAMILIES,
    FIXED_FAMILY,
)
from deployment.duo_care.candidates import (
    ACTION_DIM,
    ACTION_DIM,
    CandidateAudit,
    behavior_candidate_family,
    candidate_family,
    canonicalize_encoded_chunk,
    decoded_absolute_chunk,
    encoded_relative_chunk,
)


@dataclass
class Proposal:
    """One frozen proposal-provider output in the DuoBench action contract."""

    reference_encoded: np.ndarray
    base_encoded: np.ndarray
    qpos: np.ndarray
    memory: np.ndarray
    memory_mask: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, *, agents: int, horizon: int = 100) -> None:
        reference = np.asarray(self.reference_encoded)
        base = np.asarray(self.base_encoded)
        qpos = np.asarray(self.qpos)
        memory = np.asarray(self.memory)
        mask = np.asarray(self.memory_mask)
        if reference.shape != (agents, horizon, ACTION_DIM):
            raise ValueError(f"reference proposal must be [{agents},{horizon},8], got {reference.shape}")
        if base.shape != reference.shape:
            raise ValueError(f"base proposal differs: {base.shape}/{reference.shape}")
        if qpos.shape != (agents, ACTION_DIM):
            raise ValueError(f"proposal qpos must be [{agents},8], got {qpos.shape}")
        if memory.ndim != 3 or memory.shape[0] != agents:
            raise ValueError(f"local memory must be [agents,tokens,width], got {memory.shape}")
        if mask.shape != memory.shape[:2]:
            raise ValueError(f"memory mask differs: {mask.shape}/{memory.shape[:2]}")
        if not all(np.isfinite(value).all() for value in (reference, base, qpos, memory)):
            raise ValueError("proposal contains a non-finite value")


@dataclass
class StepResult:
    """Physical step plus the privileged labels used only for branch outcomes."""

    observation: Any
    info: Mapping[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    progress: float
    success: bool
    executed_absolute: np.ndarray
    collision_or_drop: bool = False
    robot_conflict: bool = False
    duplicate_work: bool = False
    active: tuple[bool, ...] = ()
    all_joint_changes_below_threshold: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def metrics(self, *, domain_violation: bool = False) -> dict[str, Any]:
        return {
            "reward": float(self.reward),
            "progress": float(np.clip(self.progress, 0.0, 1.0)),
            "success": bool(self.success),
            "collision_or_drop": bool(self.collision_or_drop or domain_violation),
            "hard_safety_violation": bool(self.collision_or_drop or domain_violation),
            "robot_conflict": bool(self.robot_conflict),
            "duplicate_work": bool(self.duplicate_work),
            "active": tuple(bool(value) for value in self.active),
            "all_joint_changes_below_threshold": bool(self.all_joint_changes_below_threshold),
        }


@runtime_checkable
class ProposalProvider(Protocol):
    """Frozen policy boundary; implementations must remain strict-local."""

    @property
    def agent_count(self) -> int: ...

    @property
    def action_horizon(self) -> int: ...

    def new_runtime(self, task: str) -> Any: ...

    def clone_runtime(self, runtime: Any) -> Any: ...

    def propose(self, observation: Any, runtime: Any, task: str) -> Proposal: ...

    def append_executed_action(self, runtime: Any, encoded_action: np.ndarray) -> None: ...


@runtime_checkable
class BranchEnvironment(Protocol):
    """Exact-state DuoBench simulator boundary."""

    @property
    def joint_low(self) -> np.ndarray: ...

    @property
    def joint_high(self) -> np.ndarray: ...

    def reset(self, seed: int) -> tuple[Any, Mapping[str, Any]]: ...

    def capture(self, observation: Any, info: Mapping[str, Any]) -> Any: ...

    def restore(self, snapshot: Any, branch_seed: int) -> tuple[Any, Mapping[str, Any]]: ...

    def snapshot_hash(self, snapshot: Any) -> str: ...

    def local_observation_hash(self, observation: Any) -> str: ...

    def progress(self, observation: Any, info: Mapping[str, Any]) -> float: ...

    def step_absolute(self, absolute_action: np.ndarray) -> StepResult: ...


@dataclass(frozen=True)
class KernelConfig:
    action_horizon: int = 100
    rollout_horizon: int = 64
    restore_tolerance: float = 0.0
    candidate0_tolerance: float = 1e-6
    active_joint_l2_threshold: float = 0.02
    fail_on_domain_canonicalization: bool = False
    candidate_family: str = FIXED_FAMILY
    intervention_steps: int = 1

    def __post_init__(self) -> None:
        if self.action_horizon < self.rollout_horizon:
            raise ValueError("CARE action horizon must cover the branch horizon")
        if self.rollout_horizon < max(HORIZONS):
            raise ValueError("CARE branch horizon must cover every registered outcome")
        if self.candidate_family not in CANDIDATE_FAMILIES:
            raise ValueError(f"unknown CARE candidate family: {self.candidate_family}")
        if not 1 <= self.intervention_steps <= self.rollout_horizon:
            raise ValueError("CARE commitment must lie inside the branch horizon")
        if self.candidate_family == BEHAVIOR_FAMILY and self.intervention_steps < 4:
            raise ValueError(
                "behavior candidates need a commitment window of at least four "
                f"steps to be distinguishable; got {self.intervention_steps}"
            )

    def behavior_config(self) -> BehaviorCandidateConfig | None:
        """Behavior magnitudes scale with the window they are committed to."""
        if self.candidate_family != BEHAVIOR_FAMILY:
            return None
        return BehaviorCandidateConfig(
            action_horizon=self.action_horizon,
            action_dim=ACTION_DIM,
            intervention_steps=self.intervention_steps,
            wait_steps=max(1, self.intervention_steps // 2),
            grip_shift_steps=max(1, self.intervention_steps // 2),
        )


class ReferenceTerminatedBeforeAnchor(RuntimeError):
    """The frozen reference reached a terminal state before its nominal anchor."""

    def __init__(
        self,
        *,
        task: str,
        episode_seed: int,
        requested_anchor: int,
        terminal_step: int,
    ) -> None:
        self.task = str(task)
        self.episode_seed = int(episode_seed)
        self.requested_anchor = int(requested_anchor)
        self.terminal_step = int(terminal_step)
        super().__init__(
            "reference ended before CARE anchor "
            f"{self.task}:{self.episode_seed}:{self.terminal_step} "
            f"(requested_anchor={self.requested_anchor})"
        )


def clip_anchor_for_reference(
    *, requested_anchor: int, terminal_step: int, branch_horizon: int
) -> tuple[int, int]:
    """Clip a nominal anchor using only the same-seed reference length."""

    reference_length = int(terminal_step) + 1
    if reference_length <= int(branch_horizon):
        raise RuntimeError(
            "same-seed reference is too short for the registered CARE branch "
            f"horizon: reference_length={reference_length}, "
            f"branch_horizon={branch_horizon}"
        )
    adjusted_anchor = min(int(requested_anchor), reference_length - int(branch_horizon))
    if adjusted_anchor < 1 or adjusted_anchor >= int(requested_anchor):
        raise RuntimeError(
            "reference reachability did not produce a strictly earlier valid "
            f"anchor: requested={requested_anchor}, adjusted={adjusted_anchor}, "
            f"reference_length={reference_length}"
        )
    return adjusted_anchor, reference_length


@dataclass
class AnchorSnapshot:
    task: str
    episode_seed: int
    anchor_step: int
    focal_agent: int
    sampling_stratum: str
    snapshot_id: str
    environment_state: Any
    runtime: Any
    observation: Any
    info: Mapping[str, Any]
    state_hash: str
    observation_hash: str
    start_progress: float


def _clone_runtime(provider: ProposalProvider, runtime: Any) -> Any:
    clone = getattr(provider, "clone_runtime", None)
    return clone(runtime) if callable(clone) else deepcopy(runtime)


def _canonical_reference(
    proposal: Proposal,
    env: BranchEnvironment,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Canonicalize provider chunks before either collection or scoring."""

    count = proposal.reference_encoded.shape[0]
    references: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    diagnostics: dict[str, Any] = {}
    for arm in range(count):
        reference, rc, rm = canonicalize_encoded_chunk(
            proposal.reference_encoded[arm], proposal.qpos[arm], env.joint_low, env.joint_high
        )
        base, bc, bm = canonicalize_encoded_chunk(
            proposal.base_encoded[arm], proposal.qpos[arm], env.joint_low, env.joint_high
        )
        references.append(reference)
        bases.append(base)
        diagnostics[str(arm)] = {
            "reference_changed_values": rc,
            "reference_max_abs_change": rm,
            "base_changed_values": bc,
            "base_max_abs_change": bm,
        }
    return np.stack(references), np.stack(bases), diagnostics


def _reference_action(
    proposal: Proposal,
    reference_encoded: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [decoded_absolute_chunk(reference_encoded[arm], proposal.qpos[arm])[0] for arm in range(len(reference_encoded))]
    ).astype(np.float32)


def _encoded_executed(absolute: np.ndarray, proposal_qpos: np.ndarray) -> np.ndarray:
    rows = []
    for arm in range(len(absolute)):
        rows.append(encoded_relative_chunk(np.asarray(absolute[arm], np.float32)[None], proposal_qpos[arm])[0])
    return np.stack(rows).astype(np.float32)


def advance_to_anchor(
    env: BranchEnvironment,
    provider: ProposalProvider,
    *,
    task: str,
    episode_seed: int,
    anchor_step: int,
    focal_agent: int,
    sampling_stratum: str,
    snapshot_id: str | None = None,
    config: KernelConfig = KernelConfig(),
) -> AnchorSnapshot:
    """Run frozen candidate zero and capture an exact pre-decision snapshot."""

    if not 0 <= int(focal_agent) < int(provider.agent_count):
        raise ValueError("focal agent is outside the proposal provider")
    if int(provider.action_horizon) != config.action_horizon:
        raise ValueError("proposal/kernel action horizons differ")
    observation, info = env.reset(int(episode_seed))
    runtime = provider.new_runtime(task)
    for step in range(int(anchor_step)):
        proposal = provider.propose(observation, runtime, task)
        proposal.validate(agents=provider.agent_count, horizon=config.action_horizon)
        reference, _base, _diagnostics = _canonical_reference(proposal, env)
        action = _reference_action(proposal, reference)
        result = env.step_absolute(action)
        physical = np.asarray(result.executed_absolute, dtype=np.float32)
        if physical.shape != action.shape or not np.isfinite(physical).all():
            raise RuntimeError("environment did not return finite executed absolute actions")
        encoded = _encoded_executed(physical, proposal.qpos)
        provider.append_executed_action(runtime, encoded)
        observation, info = result.observation, result.info
        if result.success or result.terminated or result.truncated:
            raise ReferenceTerminatedBeforeAnchor(
                task=task,
                episode_seed=episode_seed,
                requested_anchor=anchor_step,
                terminal_step=step,
            )
    identity = snapshot_id or hashlib.sha256(
        f"{task}|{episode_seed}|{anchor_step}|{focal_agent}|{sampling_stratum}".encode()
    ).hexdigest()
    state = env.capture(observation, info)
    return AnchorSnapshot(
        task=task,
        episode_seed=int(episode_seed),
        anchor_step=int(anchor_step),
        focal_agent=int(focal_agent),
        sampling_stratum=str(sampling_stratum),
        snapshot_id=identity,
        environment_state=state,
        runtime=_clone_runtime(provider, runtime),
        observation=deepcopy(observation),
        info=deepcopy(dict(info)),
        state_hash=str(env.snapshot_hash(state)),
        observation_hash=str(env.local_observation_hash(observation)),
        start_progress=float(np.clip(env.progress(observation, info), 0.0, 1.0)),
    )


def _restore_anchor(
    env: BranchEnvironment,
    provider: ProposalProvider,
    anchor: AnchorSnapshot,
    branch_seed: int,
) -> tuple[Any, Mapping[str, Any], Any, dict[str, Any]]:
    observation, info = env.restore(anchor.environment_state, int(branch_seed))
    state_hash = str(env.snapshot_hash(env.capture(observation, info)))
    observation_hash = str(env.local_observation_hash(observation))
    diagnostics = {
        "expected_state_hash": anchor.state_hash,
        "restored_state_hash": state_hash,
        "expected_local_observation_hash": anchor.observation_hash,
        "restored_local_observation_hash": observation_hash,
        "state_hash_match": state_hash == anchor.state_hash,
        "local_observation_hash_match": observation_hash == anchor.observation_hash,
    }
    if not diagnostics["state_hash_match"] or not diagnostics["local_observation_hash_match"]:
        raise RuntimeError(f"same-snapshot restore failed for {anchor.snapshot_id}: {diagnostics}")
    return observation, info, _clone_runtime(provider, anchor.runtime), diagnostics


def _run_branch(
    env: BranchEnvironment,
    provider: ProposalProvider,
    anchor: AnchorSnapshot,
    *,
    candidate_id: int,
    regime: str,
    repeat_id: int,
    candidate_chunks: np.ndarray,
    preview_reference: np.ndarray,
    teammate_reference_actions: Sequence[np.ndarray] | None,
    candidate_audit: CandidateAudit,
    config: KernelConfig,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    if regime not in {"reactive", "replay"}:
        raise ValueError("CARE branch regime must be reactive or replay")
    branch_seed = derive_branch_seed(anchor.snapshot_id, repeat_id)
    observation, _info, runtime, restore_diagnostics = _restore_anchor(
        env, provider, anchor, branch_seed
    )
    metrics: list[dict[str, Any]] = []
    executed: list[np.ndarray] = []
    requested: list[np.ndarray] = []
    trace = hashlib.sha256()
    provider_reference_error = 0.0
    replay_error = 0.0
    domain_violation = False
    status = "VALID"
    frozen_absolute: np.ndarray | None = None
    for branch_step in range(config.rollout_horizon):
        proposal = provider.propose(observation, runtime, anchor.task)
        proposal.validate(agents=provider.agent_count, horizon=config.action_horizon)
        reference, _base, canonicalization = _canonical_reference(proposal, env)
        action = _reference_action(proposal, reference)
        if branch_step == 0:
            provider_reference_error = float(
                np.max(np.abs(reference[anchor.focal_agent] - preview_reference))
            )
            if provider_reference_error > config.candidate0_tolerance:
                raise RuntimeError(
                    f"restored provider proposal drifted by {provider_reference_error} for {anchor.snapshot_id}"
                )
            # Decode the whole candidate once, against the anchor pose it was
            # encoded relative to, and replay it open loop for the commitment
            # window. Re-decoding each step against a moving pose would let the
            # policy's own re-planning erase the intervention, which is exactly
            # what left the one-step protocol with no measurable advantage.
            frozen_absolute = decoded_absolute_chunk(
                candidate_chunks[candidate_id], proposal.qpos[anchor.focal_agent]
            )
            domain_violation = bool(
                config.fail_on_domain_canonicalization
                and candidate_audit.max_abs_canonicalization > config.candidate0_tolerance
            )
        if frozen_absolute is not None and branch_step < int(config.intervention_steps):
            action[anchor.focal_agent] = frozen_absolute[branch_step]
        if regime == "replay":
            if teammate_reference_actions is None or branch_step >= len(teammate_reference_actions):
                status = "INVALID_REPLAY_LOG"
                break
            logged = np.asarray(teammate_reference_actions[branch_step], dtype=np.float32)
            if logged.shape != action.shape:
                raise ValueError("teammate replay action shape differs")
            for arm in range(provider.agent_count):
                if arm == anchor.focal_agent:
                    continue
                action[arm] = logged[arm]
                replay_error = max(replay_error, float(np.max(np.abs(action[arm] - logged[arm]))))
        if not np.isfinite(action).all():
            domain_violation = True
            status = "INVALID_NONFINITE_ACTION"
            break
        before = proposal.qpos[:, :7]
        requested.append(action.copy())
        result = env.step_absolute(action)
        physical = np.asarray(result.executed_absolute, dtype=np.float32)
        if physical.shape != action.shape or not np.isfinite(physical).all():
            raise RuntimeError("environment did not return finite executed absolute actions")
        encoded = _encoded_executed(physical, proposal.qpos)
        provider.append_executed_action(runtime, encoded)
        trace.update(physical.tobytes())
        executed.append(physical.copy())
        if not result.active:
            movement = np.linalg.norm(physical[:, :7] - before, axis=1)
            result.active = tuple(bool(value >= config.active_joint_l2_threshold) for value in movement)
            result.all_joint_changes_below_threshold = bool(np.all(movement < config.active_joint_l2_threshold))
        metrics.append(result.metrics(domain_violation=domain_violation))
        observation = result.observation
        if result.success or result.terminated or result.truncated:
            status = "SUCCESS_TERMINATION" if result.success else "PREMATURE_TERMINATION"
            break
    if not metrics:
        raise RuntimeError(f"CARE branch produced no steps: {anchor.snapshot_id}/{candidate_id}/{regime}/{repeat_id}")
    outcomes = outcome_table(metrics, start_progress=anchor.start_progress)
    action_canonicalization = summarize_action_canonicalization(
        np.stack(requested), np.stack(executed)
    )
    row = {
        "candidate_id": int(candidate_id),
        "regime": regime,
        "repeat_id": int(repeat_id),
        "branch_seed": int(branch_seed),
        "status": status,
        "candidate_valid": bool(candidate_audit.valid),
        "candidate_failures": list(candidate_audit.failures),
        "intervention_steps_requested": int(config.intervention_steps),
        "intervention_steps_applied": int(config.intervention_steps),
        "steps": len(metrics),
        "outcomes": outcomes,
        "success": bool(metrics[-1]["success"]),
        "hard_safety_violation": bool(any(row["hard_safety_violation"] for row in metrics)),
        "action_trace_sha256": trace.hexdigest(),
        "provider_candidate0_max_abs_error": provider_reference_error,
        "replay_teammate_action_max_abs_error": replay_error,
        "restore_diagnostics": restore_diagnostics,
        "action_canonicalization": action_canonicalization,
    }
    return row, executed


def collect_from_anchor(
    env: BranchEnvironment,
    provider: ProposalProvider,
    anchor: AnchorSnapshot,
    *,
    config: KernelConfig = KernelConfig(),
    gate: BranchGate = BranchGate(),
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Collect the 24 registered branches from one exact anchor snapshot."""

    preview_runtime = _clone_runtime(provider, anchor.runtime)
    preview = provider.propose(deepcopy(anchor.observation), preview_runtime, anchor.task)
    preview.validate(agents=provider.agent_count, horizon=config.action_horizon)
    references, bases, canonicalization = _canonical_reference(preview, env)
    focal = anchor.focal_agent
    behavior_config = config.behavior_config()
    if behavior_config is None:
        chunks, candidate_audits = candidate_family(
            references[focal],
            bases[focal],
            preview.qpos[focal],
            joint_low=env.joint_low,
            joint_high=env.joint_high,
            current_gripper=float(preview.qpos[focal, 7]),
        )
    else:
        chunks, candidate_audits = behavior_candidate_family(
            references[focal],
            bases[focal],
            preview.qpos[focal],
            joint_low=env.joint_low,
            joint_high=env.joint_high,
            config=behavior_config,
            current_gripper=float(preview.qpos[focal, 7]),
        )
    if not all(row.valid for row in candidate_audits):
        raise RuntimeError(f"illegal CARE candidate family {anchor.snapshot_id}: {candidate_audits}")
    branches: list[dict[str, Any]] = []
    for repeat_id in BRANCH_REPEATS:
        reference_reactive, teammate_trace = _run_branch(
            env, provider, anchor,
            candidate_id=0, regime="reactive", repeat_id=repeat_id,
            candidate_chunks=chunks, preview_reference=references[focal],
            teammate_reference_actions=None, candidate_audit=candidate_audits[0], config=config,
        )
        branches.append(reference_reactive)
        reference_replay, _ = _run_branch(
            env, provider, anchor,
            candidate_id=0, regime="replay", repeat_id=repeat_id,
            candidate_chunks=chunks, preview_reference=references[focal],
            teammate_reference_actions=teammate_trace, candidate_audit=candidate_audits[0], config=config,
        )
        branches.append(reference_replay)
        # Preserve the registered ordering: all reactive counterfactuals, then
        # all teammate-replay counterfactuals, for each repeat.
        for candidate_id in range(1, BRANCH_CANDIDATES):
            row, _ = _run_branch(
                env, provider, anchor,
                candidate_id=candidate_id, regime="reactive", repeat_id=repeat_id,
                candidate_chunks=chunks, preview_reference=references[focal],
                teammate_reference_actions=None, candidate_audit=candidate_audits[candidate_id], config=config,
            )
            branches.append(row)
        for candidate_id in range(1, BRANCH_CANDIDATES):
            row, _ = _run_branch(
                env, provider, anchor,
                candidate_id=candidate_id, regime="replay", repeat_id=repeat_id,
                candidate_chunks=chunks, preview_reference=references[focal],
                teammate_reference_actions=teammate_trace, candidate_audit=candidate_audits[candidate_id], config=config,
            )
            branches.append(row)
    family = {
        "format_version": "before-we-act.care-duobench-branch-family/2",
        "snapshot_id": anchor.snapshot_id,
        "task": anchor.task,
        "episode_seed": anchor.episode_seed,
        "anchor_step": anchor.anchor_step,
        "focal_agent": focal,
        "sampling_stratum": anchor.sampling_stratum,
        "branch_count": len(branches),
        "branches": branches,
        "candidate_legality": [
            {
                "candidate_id": index,
                "valid": audit.valid,
                "failures": list(audit.failures),
                "first_joint_delta_linf": audit.first_joint_delta_linf,
                "changed_values": audit.changed_values,
                "max_abs_canonicalization": audit.max_abs_canonicalization,
            }
            for index, audit in enumerate(candidate_audits)
        ],
        "snapshot_state_sha256": anchor.state_hash,
        "snapshot_local_observation_sha256": anchor.observation_hash,
        "start_progress": anchor.start_progress,
        "intervention_contract": (
            "single_focal_arm_open_loop_prefix_"
            f"{int(config.intervention_steps)}_control_steps"
        ),
        "candidate_contract": "reference_base_hold1_timewarp075_timewarp125_freeze_gripper",
        "proposal_provider_contract": "frozen_strict_local_model_independent_v1",
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "prebranch_diagnostics": {
            "provider": dict(preview.diagnostics),
            "canonicalization": canonicalization,
        },
    }
    gate_report = gate.check(family)
    family["branch_gate"] = gate_report
    if gate_report["status"] != "PASSED":
        raise RuntimeError(f"CARE branch gate failed for {anchor.snapshot_id}: {gate_report['errors']}")
    arrays = {
        "memory": np.asarray(preview.memory[focal], dtype=np.float16),
        "memory_mask": np.asarray(preview.memory_mask[focal], dtype=bool),
        "candidate_chunks": chunks.astype(np.float32),
        "focal_agent": np.asarray([focal], dtype=np.int64),
    }
    return family, arrays


__all__ = [
    "AnchorSnapshot",
    "BranchEnvironment",
    "KernelConfig",
    "Proposal",
    "ProposalProvider",
    "ReferenceTerminatedBeforeAnchor",
    "StepResult",
    "advance_to_anchor",
    "clip_anchor_for_reference",
    "collect_from_anchor",
]
