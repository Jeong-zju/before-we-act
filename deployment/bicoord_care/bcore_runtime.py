"""Strict-local BiCoord runtime for the real CARE B-core/TUNE policy.

One shared :class:`PredictiveTeamBeliefPolicy` is evaluated as a two-row
batch.  Each row contains the shared head camera plus only that arm's wrist,
qpos, and executed-action history; no peer wrist/state/action or arm ID is
accepted by the model path.  The preview/commit API lets CARE branch rollouts
execute a counterfactual command while retaining the exact reference policy
context, without copying or replacing the policy module.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from before_we_act.predictive_team_belief_policy import PredictiveTeamBeliefPolicy
from before_we_act.team_belief.predictive_core import TeamBeliefConfig
from before_we_act.temporal_history_data import task_text_tensor

from .bcore_data import (
    BICOORD_CARE_MEMORY_SEMANTICS,
    BICOORD_CARE_MEMORY_TOKENS,
    BICOORD_CARE_MEMORY_WIDTH,
    BICOORD_FUTURE_OFFSETS_STEPS,
    BICOORD_SOURCE_FREQUENCY_HZ,
)
from .config import (
    ACTION_DIM,
    ACTION_ENCODING,
    ACTION_HORIZON,
    D_MODEL,
    DECODER_LAYERS,
    ENCODER_LAYERS,
    HISTORY_LAYERS,
    HISTORY_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    ROLES,
    ROLE_RANK,
    STATE_DIM,
    TASKS,
    TASK_TEXT,
    GRIPPER_ENCODING,
    GRIPPER_NATIVE_RANGE,
    validate_native_gripper_vector,
)
from .data import project_local_observation
from .preprocessing import resize_rgb_batch
from .runtime import AbsoluteChunkEnsemble, B0HRuntime, _HistoryRow, _as_frame
from .train_bcore import validate_deployment_payload


def _copy_array(value: np.ndarray | None) -> np.ndarray | None:
    return None if value is None else np.asarray(value).copy()


def _ensemble_state(ensemble: AbsoluteChunkEnsemble) -> dict[str, Any]:
    return {
        "step": int(ensemble.step),
        "decay": float(ensemble.decay),
        "values": {
            arm: [(int(born), np.asarray(row, dtype=np.float32).copy()) for born, row in values]
            for arm, values in ensemble.values.items()
        },
    }


def _restore_ensemble(ensemble: AbsoluteChunkEnsemble, state: Mapping[str, Any]) -> None:
    if float(state.get("decay", -1.0)) != ensemble.decay:
        raise ValueError("B-core runtime ensemble decay differs")
    step = int(state.get("step", -1))
    values = state.get("values")
    if step < 0 or not isinstance(values, Mapping) or set(map(int, values)) != {0, 1}:
        raise ValueError("invalid B-core ensemble snapshot")
    restored: dict[int, list[tuple[int, np.ndarray]]] = {0: [], 1: []}
    for arm in (0, 1):
        source = values.get(arm, values.get(str(arm)))
        if not isinstance(source, Sequence):
            raise ValueError("invalid B-core ensemble arm snapshot")
        for item in source:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise ValueError("invalid B-core ensemble row snapshot")
            born = int(item[0])
            row = np.asarray(item[1], dtype=np.float32)
            if row.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(row).all():
                raise ValueError("invalid B-core ensemble chunk snapshot")
            restored[arm].append((born, row.copy()))
    ensemble.step = step
    ensemble.values = restored


@dataclass(frozen=True)
class BcoreContext:
    """Reference context exposed to CARE branch/paired validation."""

    reference_plan: np.ndarray
    base_plan: np.ndarray
    reference_chunk: np.ndarray
    base_chunk: np.ndarray
    memory: np.ndarray
    memory_mask: np.ndarray
    current_qpos: np.ndarray
    belief_mu: np.ndarray
    event_memory: np.ndarray
    event_mask: np.ndarray
    belief_sigma: np.ndarray
    belief_reliability: np.ndarray
    residual_gate: np.ndarray
    residual: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = {
            "reference_plan": (2, ACTION_HORIZON, ACTION_DIM),
            "base_plan": (2, ACTION_HORIZON, ACTION_DIM),
            "reference_chunk": (2, ACTION_HORIZON, ACTION_DIM),
            "base_chunk": (2, ACTION_HORIZON, ACTION_DIM),
            "memory": (2, BICOORD_CARE_MEMORY_TOKENS, BICOORD_CARE_MEMORY_WIDTH),
            "memory_mask": (2, BICOORD_CARE_MEMORY_TOKENS),
            "current_qpos": (2, STATE_DIM),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"B-core context {name} differs: {value.shape} != {shape}")
            if name != "memory_mask" and not np.isfinite(value).all():
                raise ValueError(f"B-core context {name} is non-finite")
        if self.memory_mask.dtype != np.bool_:
            raise TypeError("B-core context memory mask must be boolean")

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_plan": self.reference_plan,
            "base_plan": self.base_plan,
            "reference_chunk": self.reference_chunk,
            "base_chunk": self.base_chunk,
            "memory": self.memory,
            "memory_mask": self.memory_mask,
            "current_qpos": self.current_qpos,
            "belief_mu": self.belief_mu,
            "event_memory": self.event_memory,
            "event_mask": self.event_mask,
            "belief_sigma": self.belief_sigma,
            "belief_reliability": self.belief_reliability,
            "residual_gate": self.residual_gate,
            "residual": self.residual,
            "diagnostics": dict(self.diagnostics),
        }


class BiCoordBcoreRuntime(B0HRuntime):
    """One shared B-core policy with two independent local histories."""

    model: PredictiveTeamBeliefPolicy

    def __init__(
        self,
        model: PredictiveTeamBeliefPolicy,
        stats: Mapping[str, Any],
        *,
        device: torch.device,
        ensemble_decay: float = 0.01,
    ) -> None:
        super().__init__(model, stats, device=device)
        self.ensemble = AbsoluteChunkEnsemble(ensemble_decay)
        self.base_ensemble = AbsoluteChunkEnsemble(ensemble_decay)
        self.task: str | None = None
        self._preview_after_observation: Mapping[str, Any] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        dino_model: str | Path | None = None,
        ensemble_decay: float = 0.01,
    ) -> "BiCoordBcoreRuntime":
        path = Path(checkpoint).expanduser().resolve(strict=True)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(saved, Mapping):
            raise ValueError("BiCoord B-core checkpoint is not a mapping")
        config = validate_deployment_payload(saved)
        values = config.get("n2_config")
        if not isinstance(values, Mapping):
            raise ValueError("BiCoord B-core deployment has no N2 config")
        belief_values = dict(values)
        for key in ("future_offsets_steps", "future_offsets_seconds"):
            if key in belief_values:
                belief_values[key] = tuple(belief_values[key])
        belief_config = TeamBeliefConfig(**belief_values)
        if (
            belief_config.source_frequency_hz != BICOORD_SOURCE_FREQUENCY_HZ
            or tuple(belief_config.future_offsets_steps) != BICOORD_FUTURE_OFFSETS_STEPS
        ):
            raise ValueError("BiCoord B-core runtime temporal anchors differ")
        model_name = str(dino_model or config.get("dino_model") or "")
        if not model_name:
            raise ValueError("BiCoord B-core deployment has no pinned DINO model path")
        model = PredictiveTeamBeliefPolicy(
            belief_config,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            horizon=ACTION_HORIZON,
            d_model=D_MODEL,
            enc_layers=ENCODER_LAYERS,
            dec_layers=DECODER_LAYERS,
            roles=ROLES,
            role_rank=ROLE_RANK,
            history_layers=HISTORY_LAYERS,
            dino_model=model_name,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            strict_dino_contract=True,
            include_teacher=False,
            residual_safety={"enabled": False},
        )
        model.load_state_dict(saved["model"], strict=True)
        if model.belief_core.teacher_branch is not None:
            raise RuntimeError("privileged teacher was reconstructed at deployment")
        return cls(
            model,
            saved["stats"],
            device=torch.device(device),
            ensemble_decay=ensemble_decay,
        )

    def reset(self, task: str | None = None) -> None:
        if task is not None and task not in TASKS:
            raise ValueError(f"unknown BiCoord task: {task}")
        super().reset()
        self.base_ensemble.reset()
        self.task = task
        self._preview_after_observation = None

    def snapshot_state(self) -> dict[str, Any]:
        """Deep-copy runtime state only; model tensors are never duplicated."""

        return {
            "schema": "before-we-act.bicoord.bcore-runtime-state/1",
            "task": self.task,
            "rows": {
                arm: [
                    {
                        "visual": row.visual.copy(),
                        "state": row.state.copy(),
                        "action": _copy_array(row.action),
                    }
                    for row in self.rows[arm]
                ]
                for arm in (0, 1)
            },
            "pending_actions": {
                arm: _copy_array(self.pending_actions[arm]) for arm in (0, 1)
            },
            "ensemble": _ensemble_state(self.ensemble),
            "base_ensemble": _ensemble_state(self.base_ensemble),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "before-we-act.bicoord.bcore-runtime-state/1":
            raise ValueError("invalid BiCoord B-core runtime-state schema")
        task = state.get("task")
        if task is not None and task not in TASKS:
            raise ValueError("invalid task in B-core runtime snapshot")
        rows = state.get("rows")
        pending = state.get("pending_actions")
        if not isinstance(rows, Mapping) or not isinstance(pending, Mapping):
            raise ValueError("invalid B-core history snapshot")
        restored_rows: dict[int, deque[_HistoryRow]] = {
            0: deque(maxlen=HISTORY_STEPS),
            1: deque(maxlen=HISTORY_STEPS),
        }
        for arm in (0, 1):
            source = rows.get(arm, rows.get(str(arm)))
            if not isinstance(source, Sequence) or len(source) > HISTORY_STEPS:
                raise ValueError("invalid B-core arm history snapshot")
            for item in source:
                if not isinstance(item, Mapping):
                    raise ValueError("invalid B-core history row")
                visual = np.asarray(item.get("visual"), dtype=np.float32)
                qpos = np.asarray(item.get("state"), dtype=np.float32)
                action_value = item.get("action")
                action = (
                    None
                    if action_value is None
                    else np.asarray(action_value, dtype=np.float32)
                )
                if visual.shape != (2, 768) or qpos.shape != (STATE_DIM,):
                    raise ValueError("invalid B-core history tensor shape")
                if action is not None and action.shape != (ACTION_DIM,):
                    raise ValueError("invalid B-core executed-action shape")
                if not np.isfinite(visual).all() or not np.isfinite(qpos).all() or (
                    action is not None and not np.isfinite(action).all()
                ):
                    raise ValueError("non-finite B-core history snapshot")
                restored_rows[arm].append(
                    _HistoryRow(visual.copy(), qpos.copy(), _copy_array(action))
                )
        restored_pending: dict[int, np.ndarray | None] = {}
        for arm in (0, 1):
            value = pending.get(arm, pending.get(str(arm)))
            if value is None:
                restored_pending[arm] = None
            else:
                array = np.asarray(value, dtype=np.float32)
                if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
                    raise ValueError("invalid pending B-core action snapshot")
                restored_pending[arm] = array.copy()
        _restore_ensemble(self.ensemble, state["ensemble"])
        _restore_ensemble(self.base_ensemble, state["base_ensemble"])
        self.rows = restored_rows
        self.pending_actions = restored_pending
        self.task = task
        self._preview_after_observation = None

    def _temporal_batch(
        self,
        head: np.ndarray,
        wrists: Sequence[np.ndarray],
        qraw: np.ndarray,
        task: str,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        if len(wrists) != 2 or qraw.shape != (2, STATE_DIM):
            raise ValueError("B-core runtime requires two local wrist/qpos streams")
        features = self._features(head, wrists)
        if features.shape != (3, 768):
            raise ValueError(f"B-core pooled DINO features differ: {features.shape}")
        qnorm = (qraw - self.q_mean[None]) / self.q_std[None]
        task_bytes, task_mask = task_text_tensor(TASK_TEXT[task])
        rows: list[dict[str, torch.Tensor]] = []
        for arm in (0, 1):
            previous = self.pending_actions[arm]
            self.rows[arm].append(
                _HistoryRow(
                    visual=np.stack((features[0], features[arm + 1])).astype(np.float32),
                    state=qnorm[arm].astype(np.float32),
                    action=(
                        None
                        if previous is None
                        else ((previous - self.a_mean) / self.a_std).astype(np.float32)
                    ),
                )
            )
            self.pending_actions[arm] = None
            history = list(self.rows[arm])
            offset = HISTORY_STEPS - len(history)
            visual = np.zeros((HISTORY_STEPS, 2, 768), dtype=np.float32)
            qpos = np.zeros((HISTORY_STEPS, STATE_DIM), dtype=np.float32)
            action = np.zeros((HISTORY_STEPS, ACTION_DIM), dtype=np.float32)
            hmask = np.zeros(HISTORY_STEPS, dtype=bool)
            amask = np.zeros(HISTORY_STEPS, dtype=bool)
            for index, row in enumerate(history, start=offset):
                visual[index] = row.visual
                qpos[index] = row.state
                hmask[index] = True
                if row.action is not None:
                    action[index] = row.action
                    amask[index] = True
            rows.append(
                {
                    "history_visual_raw": torch.from_numpy(visual),
                    "history_qpos": torch.from_numpy(qpos),
                    "history_action": torch.from_numpy(action),
                    "history_mask": torch.from_numpy(hmask),
                    "action_history_mask": torch.from_numpy(amask),
                    "task_bytes": task_bytes,
                    "task_text_mask": task_mask,
                    "episode_reset": torch.tensor(len(history) == 1, dtype=torch.bool),
                }
            )
        batch = {
            key: torch.stack([row[key] for row in rows]).to(self.device)
            for key in rows[0]
        }
        head_tensor = resize_rgb_batch(head, IMAGE_HEIGHT, IMAGE_WIDTH)
        wrist_tensors = [resize_rgb_batch(frame, IMAGE_HEIGHT, IMAGE_WIDTH) for frame in wrists]
        batch["global_rgb"] = torch.stack((head_tensor, head_tensor)).to(
            self.device
        ).float().div_(255)
        batch["local_rgb"] = torch.stack(wrist_tensors).to(self.device).float().div_(255)
        return batch, qnorm

    @staticmethod
    def _coerce_actions(actions: Mapping[int, Any] | np.ndarray) -> np.ndarray:
        if isinstance(actions, Mapping):
            if set(map(int, actions)) != {0, 1}:
                raise ValueError("B-core executed actions require arm keys 0 and 1")
            value = np.stack(
                (
                    np.asarray(actions.get(0, actions.get("0")), dtype=np.float32),
                    np.asarray(actions.get(1, actions.get("1")), dtype=np.float32),
                )
            )
        else:
            value = np.asarray(actions, dtype=np.float32)
        if value.shape != (2, ACTION_DIM) or not np.isfinite(value).all():
            raise ValueError("B-core executed actions must be finite [2,7]")
        validate_native_gripper_vector(value, context="B-core executed actions")
        return value

    @torch.inference_mode()
    def act_with_context(
        self,
        observation: Mapping[str, Any],
        task: str | None = None,
        *,
        belief_enabled: bool = True,
        commit: bool = False,
    ) -> BcoreContext:
        task = task or self.task
        if task is None:
            raise ValueError("B-core runtime requires reset(task) or an explicit task")
        if task not in TASKS:
            raise ValueError(f"unknown BiCoord task: {task}")
        if self.task != task:
            self.reset(task)
        before = self.snapshot_state()
        local = [project_local_observation(observation, arm) for arm in (0, 1)]
        heads = [_as_frame(item["head_rgb"]) for item in local]
        if not np.array_equal(heads[0], heads[1]):
            raise ValueError("BiCoord arms disagree on the shared head camera")
        wrists = [_as_frame(item["wrist_rgb"]) for item in local]
        qraw = np.stack(
            [np.asarray(item["state"], dtype=np.float32) for item in local]
        )
        if qraw.shape != (2, STATE_DIM) or not np.isfinite(qraw).all():
            raise ValueError("BiCoord runtime qpos must be finite [2,7]")
        try:
            batch, _qnorm = self._temporal_batch(heads[0], wrists, qraw, task)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                output = self.model(**batch, belief_enabled=belief_enabled)
            reference_chunk = (
                output.prediction.float().cpu().numpy() * self.a_std[None, None]
                + self.a_mean[None, None]
            ).astype(np.float32)
            base_chunk = (
                output.base_prediction.float().cpu().numpy() * self.a_std[None, None]
                + self.a_mean[None, None]
            ).astype(np.float32)
            if (
                reference_chunk.shape != (2, ACTION_HORIZON, ACTION_DIM)
                or base_chunk.shape != (2, ACTION_HORIZON, ACTION_DIM)
                or not np.isfinite(reference_chunk).all()
                or not np.isfinite(base_chunk).all()
            ):
                raise ValueError("B-core predicted chunks must be finite [2,100,7]")
            reference_plan_map = self.ensemble.add_and_plan(
                {arm: reference_chunk[arm] for arm in (0, 1)}
            )
            base_plan_map = self.base_ensemble.add_and_plan(
                {arm: base_chunk[arm] for arm in (0, 1)}
            )
            reference_plan = np.stack(
                (reference_plan_map[0], reference_plan_map[1])
            )
            base_plan = np.stack((base_plan_map[0], base_plan_map[1]))
            belief_mu = output.belief.mu.detach().float().cpu().numpy()
            event_memory = output.belief.event_memory.detach().float().cpu().numpy()
            event_mask = output.belief.event_mask.detach().bool().cpu().numpy()
            memory = np.concatenate((belief_mu, event_memory), axis=1)
            memory_mask = np.concatenate(
                (np.ones(belief_mu.shape[:2], dtype=bool), event_mask), axis=1
            )
            residual = (
                output.belief_residual.detach().float().cpu().numpy()
                * self.a_std[None, None]
            )
            reference_gripper = reference_chunk[..., -1]
            base_gripper = base_chunk[..., -1]
            reference_plan_gripper = reference_plan[..., -1]
            low, high = GRIPPER_NATIVE_RANGE
            diagnostics = {
                "policy_family": "PredictiveTeamBeliefPolicy",
                "method_family": "CARE",
                "benchmark_adapter": "BiCoord",
                "task": task,
                "belief_enabled": bool(belief_enabled),
                "action_encoding": ACTION_ENCODING,
                "gripper_encoding": GRIPPER_ENCODING,
                "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
                "strictly_decentralized": True,
                "shared_checkpoint_for_both_arms": True,
                "arm_id_input": False,
                "peer_runtime_input": False,
                "teacher_present": False,
                "source_frequency_hz": BICOORD_SOURCE_FREQUENCY_HZ,
                "future_offsets_steps": list(BICOORD_FUTURE_OFFSETS_STEPS),
                "care_memory_tokens": BICOORD_CARE_MEMORY_TOKENS,
                "care_memory_width": BICOORD_CARE_MEMORY_WIDTH,
                "care_memory_semantics": BICOORD_CARE_MEMORY_SEMANTICS,
                "belief_reliability_mean": float(
                    output.belief.reliability.float().mean().cpu()
                ),
                "belief_sigma_mean": float(output.belief.sigma.float().mean().cpu()),
                "belief_event_slots_valid": int(event_mask.sum()),
                "residual_gate_mean": float(output.residual_gate.float().mean().cpu()),
                "residual_norm_mean": float(
                    np.linalg.norm(residual, axis=-1).mean()
                ),
                # The unexecuted chunk/plan tails remain unchanged.  Their
                # range is evidence, not a clipping trigger; the actual row
                # is checked fail-closed by record_executed_actions below.
                "reference_chunk_gripper_min": float(reference_gripper.min()),
                "reference_chunk_gripper_max": float(reference_gripper.max()),
                "reference_chunk_gripper_oob_count": int(
                    np.count_nonzero(
                        (reference_gripper < low) | (reference_gripper > high)
                    )
                ),
                "base_chunk_gripper_min": float(base_gripper.min()),
                "base_chunk_gripper_max": float(base_gripper.max()),
                "base_chunk_gripper_oob_count": int(
                    np.count_nonzero((base_gripper < low) | (base_gripper > high))
                ),
                "reference_plan_gripper_min": float(reference_plan_gripper.min()),
                "reference_plan_gripper_max": float(reference_plan_gripper.max()),
                "reference_plan_gripper_oob_count": int(
                    np.count_nonzero(
                        (reference_plan_gripper < low)
                        | (reference_plan_gripper > high)
                    )
                ),
                "executed_gripper_oob_count": 0,
                "policy_output_clipping": False,
            }
            context = BcoreContext(
                reference_plan=reference_plan.astype(np.float32),
                base_plan=base_plan.astype(np.float32),
                reference_chunk=reference_chunk,
                base_chunk=base_chunk,
                memory=memory.astype(np.float32),
                memory_mask=memory_mask,
                current_qpos=qraw.copy(),
                belief_mu=belief_mu.astype(np.float32),
                event_memory=event_memory.astype(np.float32),
                event_mask=event_mask,
                belief_sigma=output.belief.sigma.detach().float().cpu().numpy(),
                belief_reliability=output.belief.reliability.detach().float().cpu().numpy(),
                residual_gate=output.residual_gate.detach().float().cpu().numpy(),
                residual=residual.astype(np.float32),
                diagnostics=diagnostics,
            )
            after_observation = self.snapshot_state()
        except BaseException:
            self.restore_state(before)
            raise
        self.restore_state(before)
        self._preview_after_observation = after_observation
        if commit:
            self.record_executed_actions(context.reference_plan[:, 0])
        return context

    def record_executed_actions(
        self, actions: Mapping[int, Any] | np.ndarray
    ) -> None:
        """Commit the last preview and the actual controller command."""

        value = self._coerce_actions(actions)
        if self._preview_after_observation is None:
            raise RuntimeError("record_executed_actions requires an uncommitted preview")
        state = self._preview_after_observation
        self.restore_state(state)
        for arm in (0, 1):
            # Preserve native controller-equivalent values.  No clipping or
            # gripper threshold is legal at this boundary.
            self.pending_actions[arm] = value[arm].copy()
        self._preview_after_observation = None

    commit = record_executed_actions
    set_pending_actions = record_executed_actions

    @torch.inference_mode()
    def act(
        self,
        observation: Mapping[str, Any],
        task: str | None = None,
        *,
        belief_enabled: bool = True,
    ) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
        context = self.act_with_context(
            observation, task, belief_enabled=belief_enabled, commit=True
        )
        actions = {
            arm: context.reference_plan[arm, 0].copy() for arm in (0, 1)
        }
        return actions, dict(context.diagnostics)


# Both spellings are accepted by the existing evaluator import bridge.
BicoordBcoreRuntime = BiCoordBcoreRuntime


__all__ = [
    "BcoreContext",
    "BiCoordBcoreRuntime",
    "BicoordBcoreRuntime",
]
