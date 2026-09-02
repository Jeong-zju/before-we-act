"""DuoBench adapters for the model-independent CARE branch kernel.

This module is deliberately a thin benchmark boundary.  The formal provider
is the Duo DINO ``PredictiveTeamBeliefPolicy``; CARE receives its legal local
belief memory plus the full B-core and frozen B0-H proposal chunks.  A separate
B0-H-only provider remains available for diagnostics but is not accepted by
the formal launcher.  The simulator
adapter keeps the MuJoCo dynamic state *and* task-wrapper state so every
reactive/replay branch starts from one exact snapshot.

The adapter is kept separate from :mod:`branch_collection_v2` so that the
kernel can still be tested with a fake provider/environment and so that an ACT
checkpoint cannot accidentally become the formal DuoBench provider.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import random
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import gymnasium as gym
import mujoco
import numpy as np
import torch

from before_we_act.temporal_history_data import task_text_tensor
from before_we_act.temporal_history_policy import TemporalHistoryPolicy
from before_we_act.predictive_team_belief_policy import PredictiveTeamBeliefPolicy
from before_we_act.team_belief.predictive_core import TeamBeliefConfig
from deployment.duo_dino_reference.bcore_data import (
    DUO_CARE_MEMORY_SEMANTICS,
    DUO_CARE_MEMORY_TOKENS,
    DUO_CARE_MEMORY_WIDTH,
    validate_b0h_payload,
)
from deployment.duo_dino_reference.bcore_runtime import validate_bcore_payload

from deployment.duo_dino_reference.data import (
    ACTION_DIM,
    ACTION_HORIZON,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HISTORY_STEPS,
    STATE_DIM,
    TASKS,
    TASK_TEXT,
    resize_rgb_batch,
)
from deployment.duo_dino_reference.preprocessing import (
    IMAGE_PREPROCESS_ID,
    DINO_NORMALIZATION_ID,
)
from deployment.duo_care.branch_collection_v2 import (
    BranchEnvironment,
    Proposal,
    ProposalProvider,
    StepResult,
)
from deployment.duo_care.branch_signal import stable_tree_hash
from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SHA256,
    CONTROLLER_JOINT_HIGH,
    CONTROLLER_JOINT_LOW,
    canonicalize_controller_action_with_audit,
)


JOINT_DIM = 7

# Ordered, task-owned predicates are used only to add within-stage resolution
# to the benchmark's public stage number.  They are privileged branch labels;
# none are exposed to the deployed policy or CARE scorer input.
_PROGRESS_PREDICATES: dict[str, tuple[str, ...]] = {
    "ball_maze": (
        "left_arm_contact", "right_arm_contact", "both_arms_contact",
        "maze_lifted", "ball_left_start", "ball_near_goal", "ball_at_goal",
    ),
    "bin_sort": ("left_box_picked", "right_box_picked", "left_box_placed", "right_box_placed"),
    "block_balance": (
        "beam_grasped", "beam_on_cube", "both_cubes_grasped",
        "both_cubes_on_beam", "hands_retracted", "beam_balanced",
    ),
    "carry_pot": ("one_arm_handle_contact", "both_arms_handle_contact", "pot_lifted", "pot_on_stove"),
    "hinge_chest": ("door_open", "object_picked", "object_placed"),
    "join_blocks": ("approaching_done", "blocks_connected", "wall_connected", "holding_on_wall"),
    "pour_marbles": (
        "one_cup_grasped", "both_cups_grasped", "both_cups_lifted",
        "one_marble_in_target_cup", "all_marbles_in_target_cup",
        "cups_placed", "cups_upright", "done",
    ),
    "transfer_cube": (
        "grasped_and_lifted", "both_grippers_contact", "successfully_transferred",
        "correctly_placed", "arms_retracted",
    ),
    "transfer_gate": ("object_picked", "object_passed_ring", "object_handover", "object_on_mat"),
    "transfer_reorient": (
        "grasped_and_lifted", "both_grippers_contact", "successfully_transferred", "inserted",
    ),
}


def _frame(value: Any) -> np.ndarray:
    """Convert an RCS camera value to contiguous ``uint8 HxWx3``."""

    if isinstance(value, Mapping):
        if "rgb" in value:
            value = value["rgb"]
        if isinstance(value, Mapping):
            value = value.get("data", value.get("image", value))
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected RGB frame HxWx3, got {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(array.max(initial=0)) <= 1.0:
            array = np.rint(array * 255.0)
        array = np.asarray(array, dtype=np.uint8)
    return np.ascontiguousarray(array)


def _arm_entry(observation: Mapping[str, Any], arm: int) -> Mapping[str, Any]:
    key = "left" if int(arm) == 0 else "right"
    value = observation.get(key)
    if isinstance(value, Mapping):
        return value
    agents = observation.get("agent") or observation.get("agents")
    if isinstance(agents, Mapping):
        for candidate in (f"panda-{arm}", f"panda_{arm}", str(arm)):
            if isinstance(agents.get(candidate), Mapping):
                return agents[candidate]
    raise KeyError(f"DuoBench observation has no arm {arm}")


def arm_qpos(observation: Mapping[str, Any], arm: int) -> np.ndarray:
    entry = _arm_entry(observation, arm)
    if "joints" in entry:
        joints = np.asarray(entry["joints"], dtype=np.float32).reshape(-1)
        grip = np.asarray(entry.get("gripper", [0.0]), dtype=np.float32).reshape(-1)
        if joints.size < JOINT_DIM or grip.size < 1:
            raise ValueError(f"invalid arm-{arm} qpos shape: {joints.shape}/{grip.shape}")
        return np.concatenate((joints[:JOINT_DIM], grip[:1])).astype(np.float32)
    for key in ("qpos", "joint_positions", "position"):
        if key in entry:
            value = np.asarray(entry[key], dtype=np.float32).reshape(-1)
            if value.size >= ACTION_DIM:
                return value[:ACTION_DIM].astype(np.float32)
    raise KeyError(f"DuoBench observation has no qpos for arm {arm}")


def arm_frames(observation: Mapping[str, Any], arm: int) -> tuple[np.ndarray, np.ndarray]:
    frames = observation.get("frames")
    wrist_name = "left_wrist" if int(arm) == 0 else "right_wrist"
    if isinstance(frames, Mapping) and frames.get("head") is not None and frames.get(wrist_name) is not None:
        return _frame(frames["head"]), _frame(frames[wrist_name])
    head = observation.get("head") or observation.get("head_camera")
    wrist = observation.get(wrist_name)
    sensor = observation.get("sensor_data")
    if isinstance(sensor, Mapping):
        head = head or sensor.get("head_camera_global", sensor.get("head"))
        wrist = wrist or sensor.get(
            f"head_camera_agent{arm}", sensor.get("left" if int(arm) == 0 else "right")
        )
    if head is None or wrist is None:
        raise KeyError(f"DuoBench observation has no head/own-wrist frame for arm {arm}")
    return _frame(head), _frame(wrist)


@dataclass
class _ProviderRuntime:
    task: str
    visual: list[deque]
    qpos: list[deque]
    actions: list[deque]
    reference_chunks: list[list[tuple[int, np.ndarray]]]
    base_chunks: list[list[tuple[int, np.ndarray]]]
    last_qpos: np.ndarray | None = None
    step: int = 0

    @classmethod
    def create(cls, task: str) -> "_ProviderRuntime":
        return cls(
            task=task,
            visual=[deque(maxlen=HISTORY_STEPS - 1) for _ in range(2)],
            qpos=[deque(maxlen=HISTORY_STEPS - 1) for _ in range(2)],
            actions=[deque(maxlen=HISTORY_STEPS) for _ in range(2)],
            reference_chunks=[[] for _ in range(2)],
            base_chunks=[[] for _ in range(2)],
        )


def _consolidated_plan(
    history: list[tuple[int, np.ndarray]],
    *,
    step: int,
    newest: np.ndarray,
    decay: float = 0.01,
) -> np.ndarray:
    """Build the deployed 100-step absolute temporal-ensemble plan."""

    chunk = np.asarray(newest, dtype=np.float32)
    if chunk.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"Duo DINO chunk differs: {chunk.shape}")
    history.append((int(step), chunk.copy()))
    history[:] = [row for row in history if 0 <= int(step) - row[0] < len(row[1])]
    rows: list[np.ndarray] = []
    for offset in range(ACTION_HORIZON):
        absolute_step = int(step) + offset
        available = [
            (proposal, value[absolute_step - proposal])
            for proposal, value in history
            if 0 <= absolute_step - proposal < len(value)
        ]
        if not available:
            raise RuntimeError("newest DINO proposal did not cover its own horizon")
        ages = np.asarray([absolute_step - proposal for proposal, _ in available], dtype=np.float64)
        weights = np.exp(-float(decay) * ages)
        weights /= weights.sum()
        rows.append(
            np.sum(
                np.stack([value for _proposal, value in available]) * weights[:, None],
                axis=0,
            )
        )
    return np.asarray(rows, dtype=np.float32)


class DuoDinoProposalProvider(ProposalProvider):
    """Frozen, strictly local DINO proposal provider for DuoBench CARE."""

    agent_count = 2

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        dino_model: str | None = None,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> None:
        path = Path(checkpoint).resolve(strict=True)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("format") != "before-we-act.duobench.dino-b0h/1":
            raise ValueError(
                "formal Duo CARE provider only accepts a TemporalHistoryPolicy "
                "DINO B0-H checkpoint (ACT is not a registered provider)"
            )
        config = dict(saved.get("config", {}))
        if (
            config.get("policy_family") != "TemporalHistoryPolicy"
            # Do not default a missing method tag.  A legacy/metadata-light
            # checkpoint must fail closed instead of being silently promoted
            # to the formal CARE reference.
            or config.get("method_family") != "CARE"
            or config.get("architecture") != "TemporalHistoryPolicy_hidden_residual"
            or config.get("vision_backbone") != "dinov3_vitb16_frozen"
            or config.get("strict_dino_contract") is not True
        ):
            raise ValueError(
                "formal Duo CARE provider requires the project-owned frozen-DINO "
                "TemporalHistoryPolicy B0-H checkpoint; ACT/ConvNeXt is forbidden"
            )
        if config.get("action_encoding") != "absolute_joint7_binary_gripper1":
            raise ValueError("Duo DINO checkpoint does not use absolute joint7+binary gripper actions")
        contract = str(config.get("policy_contract", ""))
        if "strictly_decentralized" not in contract or "own_wrist" not in contract:
            raise ValueError("Duo DINO checkpoint does not carry the strict-local contract")
        if config.get("image_preprocess_id") != IMAGE_PREPROCESS_ID:
            raise ValueError(
                "Duo DINO checkpoint does not carry the registered image preprocess_id"
            )
        if config.get("dino_normalization_id") != DINO_NORMALIZATION_ID:
            raise ValueError("Duo DINO checkpoint has an incompatible DINO normalization id")
        self.device = torch.device(device)
        self.checkpoint = path
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self._action_horizon = int(config.get("horizon", ACTION_HORIZON))
        self.dino_model = str(dino_model or config.get("dino_model", ""))
        if not self.dino_model:
            raise ValueError("DINO model path is absent from checkpoint")
        self.image_height = int(image_height or config.get("image_height", DEFAULT_IMAGE_HEIGHT))
        self.image_width = int(image_width or config.get("image_width", DEFAULT_IMAGE_WIDTH))
        self.model = TemporalHistoryPolicy(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            variant="hidden_residual",
            horizon=self._action_horizon,
            d_model=int(config.get("d_model", 384)),
            enc_layers=int(config.get("enc_layers", 4)),
            dec_layers=int(config.get("dec_layers", 7)),
            roles=int(config.get("roles", 4)),
            role_rank=int(config.get("role_rank", 32)),
            history_layers=int(config.get("history_layers", 2)),
            dino_model=self.dino_model,
            image_height=self.image_height,
            image_width=self.image_width,
            strict_dino_contract=True,
        ).to(self.device)
        self.model.load_state_dict(saved["model"], strict=True)
        self.model.eval()
        stats = saved.get("stats", {})
        self.q_mean = torch.as_tensor(stats.get("q_mean"), dtype=torch.float32, device=self.device)
        self.q_std = torch.as_tensor(stats.get("q_std"), dtype=torch.float32, device=self.device)
        self.a_mean = torch.as_tensor(stats.get("a_mean"), dtype=torch.float32, device=self.device)
        self.a_std = torch.as_tensor(stats.get("a_std"), dtype=torch.float32, device=self.device)
        for name, value in (("q_mean", self.q_mean), ("q_std", self.q_std), ("a_mean", self.a_mean), ("a_std", self.a_std)):
            if tuple(value.shape) != (ACTION_DIM,) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"invalid DINO normalization {name}: {tuple(value.shape)}")
            if name.endswith("std") and bool(torch.any(value <= 0)):
                raise ValueError(f"non-positive DINO normalization {name}")
        # This provider is retained only for B0-H diagnostics.  It must not
        # present itself as the formal PredictiveTeamBeliefPolicy B-core used
        # by the CARE branch launcher.
        self.reference_policy_family = "TemporalHistoryPolicy"
        self.diagnostic_only = True
        self.vision = "dinov3_vitb16_frozen"
        self.action_encoding = "joint_residual_gripper_absolute"

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    def new_runtime(self, task: str) -> _ProviderRuntime:
        if task not in TASKS:
            raise ValueError(f"unknown DuoBench task: {task}")
        return _ProviderRuntime.create(task)

    def clone_runtime(self, runtime: _ProviderRuntime) -> _ProviderRuntime:
        return deepcopy(runtime)

    def _history_batch(
        self,
        observation: Mapping[str, Any],
        runtime: _ProviderRuntime,
        task: str,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        qraw = np.stack([arm_qpos(observation, arm) for arm in range(2)]).astype(np.float32)
        qnorm = (torch.as_tensor(qraw, device=self.device) - self.q_mean) / self.q_std
        visual = torch.zeros((2, HISTORY_STEPS, 2, 768), dtype=torch.float16, device=self.device)
        qpos = torch.zeros((2, HISTORY_STEPS, STATE_DIM), dtype=torch.float32, device=self.device)
        actions = torch.zeros((2, HISTORY_STEPS, ACTION_DIM), dtype=torch.float32, device=self.device)
        hmask = torch.zeros((2, HISTORY_STEPS), dtype=torch.bool, device=self.device)
        amask = torch.zeros((2, HISTORY_STEPS), dtype=torch.bool, device=self.device)
        heads: list[torch.Tensor] = []
        wrists: list[torch.Tensor] = []
        for arm in range(2):
            head, wrist = arm_frames(observation, arm)
            heads.append(resize_rgb_batch(head, self.image_height, self.image_width))
            wrists.append(resize_rgb_batch(wrist, self.image_height, self.image_width))
            if runtime.visual[arm]:
                first = HISTORY_STEPS - 1 - len(runtime.visual[arm])
                visual[arm, first:-1] = torch.stack(tuple(runtime.visual[arm])).to(self.device)
                qpos[arm, first:-1] = torch.stack(tuple(runtime.qpos[arm])).to(self.device)
                hmask[arm, first:-1] = True
            qpos[arm, -1] = qnorm[arm]
            hmask[arm, -1] = True
            if runtime.actions[arm]:
                first = HISTORY_STEPS - len(runtime.actions[arm])
                actions[arm, first:] = torch.stack(tuple(runtime.actions[arm])).to(self.device)
                amask[arm, first:] = True
        task_bytes, text_mask = task_text_tensor(TASK_TEXT[task])
        temporal = {
            "history_visual_raw": visual,
            "history_qpos": qpos,
            "history_action": actions,
            "history_mask": hmask,
            "action_history_mask": amask,
            "task_bytes": task_bytes.unsqueeze(0).expand(2, -1).to(self.device),
            "task_text_mask": text_mask.unsqueeze(0).expand(2, -1).to(self.device),
            "episode_reset": torch.tensor(
                [not runtime.visual[arm] and not runtime.actions[arm] for arm in range(2)],
                dtype=torch.bool,
                device=self.device,
            ),
        }
        return temporal, qraw, qnorm

    @torch.inference_mode()
    def propose(self, observation: Any, runtime: _ProviderRuntime, task: str) -> Proposal:
        if task not in TASKS or runtime.task != task:
            raise ValueError("provider/runtime task mismatch")
        temporal, qraw, _qnorm = self._history_batch(observation, runtime, task)
        heads: list[torch.Tensor] = []
        wrists: list[torch.Tensor] = []
        for arm in range(2):
            head, wrist = arm_frames(observation, arm)
            heads.append(resize_rgb_batch(head, self.image_height, self.image_width))
            wrists.append(resize_rgb_batch(wrist, self.image_height, self.image_width))
        global_rgb = torch.stack(heads).to(self.device).float().div_(255.0)
        local_rgb = torch.stack(wrists).to(self.device).float().div_(255.0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            context = self.model._decode_action_context(
                global_rgb,
                local_rgb,
                temporal["history_visual_raw"],
                temporal["history_qpos"],
                temporal["history_action"],
                temporal["history_mask"],
                temporal["action_history_mask"],
                temporal["task_bytes"],
                temporal["task_text_mask"],
                temporal["episode_reset"],
                None,
            )
            base = self.model.out(context.decoded)
            residual = self.model.hidden_residual(
                torch.cat(
                    (
                        context.decoded,
                        context.history_summary.unsqueeze(1).expand(-1, self.action_horizon, -1),
                    ),
                    dim=-1,
                )
            )
            prediction = base + residual
        # Keep the exact hidden history used by the frozen action policy as
        # CARE's legal local memory.  Padding is represented by the same mask
        # consumed by the temporal policy.
        memory = context.history.float()
        memory_mask = temporal["history_mask"]
        reference_raw = (prediction.float() * self.a_std + self.a_mean).detach().cpu().numpy()
        base_raw = (base.float() * self.a_std + self.a_mean).detach().cpu().numpy()
        reference = np.stack(
            [
                _consolidated_plan(
                    runtime.reference_chunks[arm],
                    step=runtime.step,
                    newest=reference_raw[arm],
                )
                for arm in range(2)
            ]
        ).astype(np.float32)
        base_np = np.stack(
            [
                _consolidated_plan(
                    runtime.base_chunks[arm],
                    step=runtime.step,
                    newest=base_raw[arm],
                )
                for arm in range(2)
            ]
        ).astype(np.float32)
        reference[:, :, :JOINT_DIM] -= qraw[:, None, :JOINT_DIM]
        base_np[:, :, :JOINT_DIM] -= qraw[:, None, :JOINT_DIM]
        runtime.last_qpos = qraw.copy()
        # The action provider itself never sees the peer's observation.  The
        # batched tensor is only two independent rows of the shared module.
        for arm in range(2):
            runtime.visual[arm].append(context.current_visual_raw[arm].detach().float().cpu())
            runtime.qpos[arm].append(((torch.as_tensor(qraw[arm], device=self.device) - self.q_mean) / self.q_std).detach().float().cpu())
        return Proposal(
            reference_encoded=reference,
            base_encoded=base_np,
            qpos=qraw,
            memory=memory.detach().cpu().numpy().astype(np.float32),
            memory_mask=memory_mask.detach().cpu().numpy().astype(bool),
            diagnostics={
                "reference_policy_family": self.reference_policy_family,
                "vision": self.vision,
                "action_encoding": self.action_encoding,
                "strictly_decentralized": True,
                "strict_local": True,
                "method_family": "CARE",
                "preprocess_id": IMAGE_PREPROCESS_ID,
                "image_preprocess_id": IMAGE_PREPROCESS_ID,
                "dino_normalization_id": DINO_NORMALIZATION_ID,
                "strict_dino_contract": True,
                "diagnostic_only": self.diagnostic_only,
                "checkpoint_sha256": self.checkpoint_sha256,
            },
        )

    def append_executed_action(self, runtime: _ProviderRuntime, encoded_action: np.ndarray) -> None:
        value = np.asarray(encoded_action, dtype=np.float32)
        if value.shape != (2, ACTION_DIM) or not np.isfinite(value).all():
            raise ValueError(f"executed action must be [2,8] finite, got {value.shape}")
        if runtime.last_qpos is None:
            raise RuntimeError("append_executed_action called before propose")
        absolute = value.copy()
        absolute[:, :JOINT_DIM] += runtime.last_qpos[:, :JOINT_DIM]
        normalized = (torch.as_tensor(absolute, device=self.device) - self.a_mean) / self.a_std
        for arm in range(2):
            runtime.actions[arm].append(normalized[arm].detach().float().cpu())
        runtime.step += 1


class DuoBcoreProposalProvider(DuoDinoProposalProvider):
    """Formal branch provider backed by a real PredictiveTeamBeliefPolicy.

    ``reference_encoded`` is the complete B-core proposal.  ``base_encoded``
    is the complete selector-off B0-H proposal embedded in that same policy;
    it is never the raw pre-hidden-residual action head.  CARE memory is the
    runtime belief ``mu`` plus its sparse event-memory slots rather than a
    generic B0-H history tensor.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        b0h_checkpoint: str | Path,
        device: str | torch.device = "cuda:0",
        dino_model: str | None = None,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> None:
        path = Path(checkpoint).resolve(strict=True)
        base_path = Path(b0h_checkpoint).resolve(strict=True)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(saved, Mapping):
            raise ValueError("Duo B-core checkpoint is not a mapping")
        config = dict(validate_bcore_payload(saved))
        b0h = torch.load(base_path, map_location="cpu", weights_only=False)
        if not isinstance(b0h, Mapping):
            raise ValueError("Duo B0-H checkpoint is not a mapping")
        b0h_config = dict(validate_b0h_payload(b0h))
        base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
        source_sha = str(
            saved.get("source_b0h_checkpoint_sha256")
            or config.get("source_b0h_checkpoint_sha256")
            or ""
        )
        if source_sha != base_sha:
            raise ValueError("Duo B-core was not derived from the supplied formal B0-H")
        # A provenance string alone is insufficient: verify the entire frozen
        # backbone embedded in B-core remains byte-identical tensor-for-tensor.
        bcore_state = saved.get("model")
        b0h_state = b0h.get("model")
        if not isinstance(bcore_state, Mapping) or not isinstance(b0h_state, Mapping):
            raise ValueError("Duo B-core/B0-H checkpoints have no model state")
        drifted = [
            key
            for key, value in b0h_state.items()
            if key not in bcore_state
            or not torch.equal(value.detach().cpu(), bcore_state[key].detach().cpu())
        ]
        if drifted:
            raise ValueError(f"Duo B-core frozen B0-H backbone drifted: {drifted[:4]}")

        self.device = torch.device(device)
        self.checkpoint = path
        self.b0h_checkpoint = base_path
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.b0h_checkpoint_sha256 = base_sha
        self._action_horizon = int(config.get("horizon", ACTION_HORIZON))
        self.dino_model = str(dino_model or config.get("dino_model", ""))
        if not self.dino_model:
            raise ValueError("DINO model path is absent from Duo B-core checkpoint")
        self.image_height = int(
            image_height or config.get("image_height", DEFAULT_IMAGE_HEIGHT)
        )
        self.image_width = int(
            image_width or config.get("image_width", DEFAULT_IMAGE_WIDTH)
        )
        n2 = dict(config.get("n2_config", {}))
        if not n2:
            raise ValueError("Duo B-core checkpoint has no belief-core config")
        for key in ("future_offsets_steps", "future_offsets_seconds"):
            if key in n2:
                n2[key] = tuple(n2[key])
        belief_config = TeamBeliefConfig(**n2)
        if (
            belief_config.source_frequency_hz != 30
            or belief_config.future_offsets_steps != (6, 12, 24, 48)
            or belief_config.future_offsets_seconds != (0.2, 0.4, 0.8, 1.6)
        ):
            raise ValueError("Duo B-core does not preserve the 30-Hz future-time contract")
        if config.get("image_preprocess_id") != IMAGE_PREPROCESS_ID:
            raise ValueError(
                "Duo B-core checkpoint does not carry the registered image preprocess_id"
            )
        if config.get("dino_normalization_id") != DINO_NORMALIZATION_ID:
            raise ValueError("Duo B-core checkpoint has an incompatible DINO normalization id")
        self.model = PredictiveTeamBeliefPolicy(
            belief_config,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            horizon=self._action_horizon,
            d_model=int(config.get("d_model", 384)),
            enc_layers=int(config.get("enc_layers", 4)),
            dec_layers=int(config.get("dec_layers", 7)),
            roles=int(config.get("roles", 4)),
            role_rank=int(config.get("role_rank", 32)),
            history_layers=int(config.get("history_layers", 2)),
            dino_model=self.dino_model,
            image_height=self.image_height,
            image_width=self.image_width,
            strict_dino_contract=True,
            include_teacher=False,
            residual_safety=config.get("residual_safety", {"enabled": False}),
        ).to(self.device)
        self.model.load_state_dict(saved["model"], strict=True)
        self.model.eval()
        stats = saved.get("stats", {})
        self.q_mean = torch.as_tensor(
            stats.get("q_mean"), dtype=torch.float32, device=self.device
        )
        self.q_std = torch.as_tensor(
            stats.get("q_std"), dtype=torch.float32, device=self.device
        )
        self.a_mean = torch.as_tensor(
            stats.get("a_mean"), dtype=torch.float32, device=self.device
        )
        self.a_std = torch.as_tensor(
            stats.get("a_std"), dtype=torch.float32, device=self.device
        )
        for name, value in (
            ("q_mean", self.q_mean),
            ("q_std", self.q_std),
            ("a_mean", self.a_mean),
            ("a_std", self.a_std),
        ):
            if tuple(value.shape) != (ACTION_DIM,) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"invalid B-core normalization {name}")
            if name.endswith("std") and bool(torch.any(value <= 0)):
                raise ValueError(f"non-positive B-core normalization {name}")
        self.reference_policy_family = "PredictiveTeamBeliefPolicy"
        self.vision = "dinov3_vitb16_frozen"
        self.action_encoding = "joint_residual7_gripper_absolute1"

    @torch.inference_mode()
    def propose(self, observation: Any, runtime: _ProviderRuntime, task: str) -> Proposal:
        if task not in TASKS or runtime.task != task:
            raise ValueError("provider/runtime task mismatch")
        temporal, qraw, _qnorm = self._history_batch(observation, runtime, task)
        heads: list[torch.Tensor] = []
        wrists: list[torch.Tensor] = []
        for arm in range(2):
            head, wrist = arm_frames(observation, arm)
            heads.append(resize_rgb_batch(head, self.image_height, self.image_width))
            wrists.append(resize_rgb_batch(wrist, self.image_height, self.image_width))
        global_rgb = torch.stack(heads).to(self.device).float().div_(255.0)
        local_rgb = torch.stack(wrists).to(self.device).float().div_(255.0)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            output = self.model(
                global_rgb,
                local_rgb,
                **temporal,
                belief_enabled=True,
            )
        reference_raw = (
            output.prediction.float() * self.a_std + self.a_mean
        ).detach().cpu().numpy()
        base_raw = (
            output.base_prediction.float() * self.a_std + self.a_mean
        ).detach().cpu().numpy()
        reference = np.stack(
            [
                _consolidated_plan(
                    runtime.reference_chunks[arm],
                    step=runtime.step,
                    newest=reference_raw[arm],
                )
                for arm in range(2)
            ]
        ).astype(np.float32)
        base = np.stack(
            [
                _consolidated_plan(
                    runtime.base_chunks[arm],
                    step=runtime.step,
                    newest=base_raw[arm],
                )
                for arm in range(2)
            ]
        ).astype(np.float32)
        reference[:, :, :JOINT_DIM] -= qraw[:, None, :JOINT_DIM]
        base[:, :, :JOINT_DIM] -= qraw[:, None, :JOINT_DIM]
        runtime.last_qpos = qraw.copy()
        for arm in range(2):
            runtime.visual[arm].append(
                output.current_visual_raw[arm].detach().float().cpu()
            )
            runtime.qpos[arm].append(
                (
                    (torch.as_tensor(qraw[arm], device=self.device) - self.q_mean)
                    / self.q_std
                )
                .detach()
                .float()
                .cpu()
            )
        memory_tensor = torch.cat(
            (output.belief.mu, output.belief.event_memory), dim=1
        ).float()
        memory_mask_tensor = torch.cat(
            (
                torch.ones(
                    output.belief.mu.shape[:2],
                    dtype=torch.bool,
                    device=output.belief.mu.device,
                ),
                output.belief.event_mask,
            ),
            dim=1,
        )
        if tuple(memory_tensor.shape) != (
            2,
            DUO_CARE_MEMORY_TOKENS,
            DUO_CARE_MEMORY_WIDTH,
        ) or tuple(memory_mask_tensor.shape) != (2, DUO_CARE_MEMORY_TOKENS):
            raise RuntimeError(
                "Duo B-core CARE memory contract differs: "
                f"{tuple(memory_tensor.shape)}/{tuple(memory_mask_tensor.shape)}"
            )
        memory = memory_tensor.detach().cpu().numpy().astype(np.float32)
        memory_mask = memory_mask_tensor.detach().cpu().numpy().astype(bool)
        return Proposal(
            reference_encoded=reference,
            base_encoded=base,
            qpos=qraw,
            memory=memory,
            memory_mask=memory_mask,
            diagnostics={
                "reference_policy_family": self.reference_policy_family,
                "base_policy_family": "TemporalHistoryPolicy",
                "vision": self.vision,
                "vision_backbone": self.vision,
                "action_encoding": self.action_encoding,
                "strictly_decentralized": True,
                "strict_local": True,
                "method_family": "CARE",
                "preprocess_id": IMAGE_PREPROCESS_ID,
                "image_preprocess_id": IMAGE_PREPROCESS_ID,
                "dino_normalization_id": DINO_NORMALIZATION_ID,
                "strict_dino_contract": True,
                "diagnostic_only": False,
                "act_provider_allowed": False,
                "bcore_checkpoint_sha256": self.checkpoint_sha256,
                "b0h_checkpoint_sha256": self.b0h_checkpoint_sha256,
                "memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
                "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
                "belief_tokens": int(self.model.team_belief_config.n_belief_tokens),
                "event_capacity": int(self.model.team_belief_config.event_capacity),
                "valid_event_slots": output.belief.event_mask.sum(1).tolist(),
                "base_semantics": "complete_TemporalHistoryPolicy_hidden_residual_prediction",
                "source_frequency_hz": self.model.team_belief_config.source_frequency_hz,
                "future_offsets_steps": list(
                    self.model.team_belief_config.future_offsets_steps
                ),
                "future_offsets_seconds": list(
                    self.model.team_belief_config.future_offsets_seconds
                ),
            },
        )


@dataclass
class _DuoSnapshot:
    sim_state: np.ndarray
    sim_schema: Mapping[str, Any]
    sim_extras: dict[str, Any]
    wrapper_state: dict[str, dict[str, Any]]
    rng_state: dict[str, Any]
    observation: Any
    info: Mapping[str, Any]
    step_count: int


def _walk_env(root: Any) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    seen: set[int] = set()

    def visit(value: Any, path: str) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        result.append((path, value))
        if hasattr(value, "envs") and isinstance(value.envs, Mapping):
            for key in sorted(value.envs, key=str):
                visit(value.envs[key], f"{path}.envs[{key}]")
        if hasattr(value, "env"):
            visit(value.env, f"{path}.env")

    visit(root, "env")
    return result


_MUTABLE_NAMES = frozenset(
    {
        "prev_action",
        "_last_gripper_cmd",
        "_absolute_action",
        "_last_action",
        "_origin",
        "initial_obs",
        "_elapsed_steps",
        "_replay_state",
        "_np_random",
        "np_random",
    }
)


def _copy_rng(value: Any) -> Any:
    if isinstance(value, np.random.Generator):
        return {"kind": "generator", "state": deepcopy(value.bit_generator.state)}
    if isinstance(value, np.random.RandomState):
        return {"kind": "random_state", "state": deepcopy(value.get_state())}
    return deepcopy(value)


def _restore_rng(target: Any, value: Any) -> None:
    if isinstance(value, Mapping) and value.get("kind") == "generator" and isinstance(target, np.random.Generator):
        target.bit_generator.state = deepcopy(value["state"])
    elif isinstance(value, Mapping) and value.get("kind") == "random_state" and isinstance(target, np.random.RandomState):
        target.set_state(deepcopy(value["state"]))


class DuoBenchEnvironment(BranchEnvironment):
    """Exact-state RCS/MuJoCo adapter with privileged offline labels."""

    def __init__(self, task: str, *, image_size: int = DEFAULT_IMAGE_HEIGHT) -> None:
        if task not in TASKS:
            raise ValueError(task)
        if int(image_size) != DEFAULT_IMAGE_HEIGHT:
            raise ValueError(
                "formal Duo CARE branch collection requires the registered 224x224 "
                "runtime resize target"
            )
        self.task = task
        self.env = self._make_env(task, image_size)
        self.step_count = 0
        self._observation: Any = None
        self._info: Mapping[str, Any] = {}
        # ``Sim.reset`` is required to reset RCS' C++ controller callback
        # clocks.  Those clocks are not represented by ``Sim.get_state``.  A
        # reset also drops the controller targets, so the first physical step
        # after every restore must re-issue both joint and gripper commands.
        # Keep this flag outside the captured wrapper tree so same-snapshot
        # hashing still compares the source state itself.
        self._force_reissue_after_restore = False
        # RCS camera buffers are asynchronous: rendering the same MuJoCo state
        # twice can differ by a handful of uint8 pixels.  That harmless sensor
        # jitter nevertheless changes a vision policy's later action and makes
        # candidate-zero reactive/replay traces fail the causal gate.  Cache
        # the first observation for each physical state so paired branches
        # consume one canonical sensor frame.  The key excludes wrapper/RNG
        # metadata and therefore never aliases different physical states.
        self._observation_cache: OrderedDict[
            str, tuple[Any, Mapping[str, Any]]
        ] = OrderedDict()
        self._sim = self.env.get_wrapper_attr("sim")
        # CARE plans and emitted actions use the physical MuJoCo actuator
        # ctrlrange.  RCS exposes a narrower API Box, but that Box is not the
        # controller-equivalent action contract of the released trajectories.
        self._joint_low = CONTROLLER_JOINT_LOW.copy()
        self._joint_high = CONTROLLER_JOINT_HIGH.copy()
        self._last_action_canonicalization: Mapping[str, Any] | None = None

    @staticmethod
    def _make_env(task: str, image_size: int) -> gym.Env:
        # Importing task modules registers all DuoBench IDs.  The config is
        # intentionally identical to the baseline closed-loop evaluator.
        __import__(f"duobench.tasks.{task}")
        module = __import__(f"duobench.tasks.{task}", fromlist=["*"])
        class_name = "".join(part.title() for part in task.split("_")) + "EnvConfig"
        cfg = getattr(module, class_name)().config()
        from rcs._core.sim import SimConfig
        from rcs.envs.base import ControlMode, RelativeTo

        cfg.headless = True
        cfg.control_mode = ControlMode.JOINTS
        cfg.relative_to = RelativeTo.NONE
        cfg.sim_cfg = SimConfig(async_control=True, realtime=False, frequency=30)
        cfg.wrapper_cfg.binary_gripper = True
        # Preserve DuoBench's native 1280x720 camera projection.  The provider
        # applies the exact converter-equivalent uint8 resize independently to
        # each view.  Rendering the simulator directly at 224x224 is not pixel
        # equivalent and would make branch collection differ from Validation20.
        return gym.make(f"duobench/{task}", cfg=cfg)

    @property
    def joint_low(self) -> np.ndarray:
        return self._joint_low

    @property
    def joint_high(self) -> np.ndarray:
        return self._joint_high

    def reset(self, seed: int) -> tuple[Any, Mapping[str, Any]]:
        observation, info = self.env.reset(seed=int(seed))
        self.step_count = 0
        self._force_reissue_after_restore = False
        self._observation_cache.clear()
        self._observation, self._info = observation, info
        return observation, info

    def _canonical_sensor_observation(
        self, observation: Any, info: Mapping[str, Any]
    ) -> tuple[Any, Mapping[str, Any]]:
        state = self.capture(observation, info)
        key = stable_tree_hash({"sim_state": state.sim_state, "sim_extras": state.sim_extras})
        cached = self._observation_cache.get(key)
        if cached is None:
            cached = (deepcopy(observation), deepcopy(dict(info)))
            self._observation_cache[key] = cached
            while len(self._observation_cache) > 96:
                self._observation_cache.popitem(last=False)
        else:
            self._observation_cache.move_to_end(key)
        return deepcopy(cached[0]), deepcopy(dict(cached[1]))

    def _clear_camera_buffers(self) -> None:
        """Discard frames rendered before the restored MuJoCo state.

        RCS simulation cameras keep a C++ ring buffer.  Its timestamps and
        pixels are deliberately not part of ``Sim.get_state``; returning one
        of those frames after a restore would make the policy observation
        depend on whichever counterfactual happened to run previously.
        """

        cleared: set[int] = set()
        for _path, obj in _walk_env(self.env):
            camera_set = getattr(obj, "camera_set", None)
            if camera_set is None or id(camera_set) in cleared:
                continue
            clear = getattr(camera_set, "clear_buffer", None)
            if callable(clear):
                clear()
                cleared.add(id(camera_set))

    def _force_controller_reissue(self) -> None:
        """Invalidate only command de-duplication caches before one step.

        ``RobotWrapper`` and ``GripperWrapper`` intentionally suppress an
        unchanged command.  After ``Sim.reset`` the hidden C++ target has been
        reset even though the Python cache was restored, so suppression would
        be incorrect.  Clearing these two caches makes the *same* physical
        command cross the controller boundary again; it does not alter the
        command, observation, policy state, or CARE intervention.
        """

        for _path, obj in _walk_env(self.env):
            if "prev_action" in vars(obj):
                vars(obj)["prev_action"] = None
            if "_last_gripper_cmd" in vars(obj):
                vars(obj)["_last_gripper_cmd"] = None

    def _wrapper_state(self) -> dict[str, dict[str, Any]]:
        state: dict[str, dict[str, Any]] = {}
        for path, obj in _walk_env(self.env):
            attrs: dict[str, Any] = {}
            for name in _MUTABLE_NAMES:
                if name in vars(obj):
                    value = vars(obj)[name]
                    if name in {"_np_random", "np_random"}:
                        attrs[name] = _copy_rng(value)
                    else:
                        try:
                            attrs[name] = deepcopy(value)
                        except Exception:
                            pass
            tracker = getattr(obj, "stage_tracker", None)
            if tracker is not None:
                tracker_attrs: dict[str, Any] = {}
                for name, value in vars(tracker).items():
                    if name in {"cfg", "sim"}:
                        continue
                    try:
                        tracker_attrs[name] = deepcopy(value)
                    except Exception:
                        pass
                attrs["__stage_tracker__"] = tracker_attrs
            if attrs:
                state[path] = attrs
        return state

    def _rng_state(self) -> dict[str, Any]:
        wrappers: dict[str, Any] = {}
        for path, obj in _walk_env(self.env):
            for name in ("_np_random", "np_random"):
                if name in vars(obj):
                    wrappers[f"{path}:{name}"] = _copy_rng(vars(obj)[name])
        return {
            "python": random.getstate(),
            "numpy": deepcopy(np.random.get_state()),
            "torch": torch.get_rng_state().clone(),
            "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            "wrappers": wrappers,
        }

    def capture(self, observation: Any, info: Mapping[str, Any]) -> _DuoSnapshot:
        sim = self._sim
        extras: dict[str, Any] = {}
        data = sim.data
        for name in (
            "time",
            "qacc",
            "qacc_warmstart",
            "act",
            "ctrl",
            "qfrc_applied",
            "xfrc_applied",
            "mocap_pos",
            "mocap_quat",
            "userdata",
            "eq_active",
        ):
            if hasattr(data, name):
                value = getattr(data, name)
                try:
                    extras[name] = np.array(value, copy=True)
                except Exception:
                    pass
        return _DuoSnapshot(
            sim_state=np.array(sim.get_state(), copy=True),
            sim_schema=deepcopy(sim.get_state_schema()),
            sim_extras=extras,
            wrapper_state=self._wrapper_state(),
            rng_state=self._rng_state(),
            observation=deepcopy(observation),
            info=deepcopy(dict(info)),
            step_count=int(self.step_count),
        )

    def _restore_wrapper_state(self, state: Mapping[str, Mapping[str, Any]]) -> None:
        objects = dict(_walk_env(self.env))
        for path, attrs in state.items():
            obj = objects.get(path)
            if obj is None:
                raise RuntimeError(f"wrapper path disappeared during restore: {path}")
            for name, value in attrs.items():
                if name == "__stage_tracker__":
                    tracker = getattr(obj, "stage_tracker", None)
                    if tracker is None:
                        raise RuntimeError(f"stage tracker disappeared during restore: {path}")
                    for key, item in value.items():
                        setattr(tracker, key, deepcopy(item))
                elif name in {"_np_random", "np_random"}:
                    if name in vars(obj):
                        _restore_rng(vars(obj)[name], value)
                else:
                    setattr(obj, name, deepcopy(value))

    def _restore_rng(self, state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda"):
            torch.cuda.set_rng_state_all(state["cuda"])
        objects = dict(_walk_env(self.env))
        for key, value in state.get("wrappers", {}).items():
            path, name = key.rsplit(":", 1)
            obj = objects.get(path)
            if obj is not None and name in vars(obj):
                _restore_rng(vars(obj)[name], value)

    def restore(self, snapshot: _DuoSnapshot, branch_seed: int) -> tuple[Any, Mapping[str, Any]]:
        # Reset C++ callback clocks/targets before restoring the public MuJoCo
        # state.  ``set_state`` alone cannot rewind those callbacks and caused
        # branch-order-dependent candidate-zero traces in DuoBench.
        self._sim.reset()
        self._sim.set_state(snapshot.sim_state, snapshot.sim_schema)
        data = self._sim.data
        for name, value in snapshot.sim_extras.items():
            if hasattr(data, name):
                target = getattr(data, name)
                try:
                    target[...] = value
                except Exception:
                    try:
                        setattr(data, name, deepcopy(value))
                    except Exception:
                        pass
        mujoco.mj_forward(self._sim.model, self._sim.data)
        # ``mj_forward`` recomputes a few acceleration/force buffers.  Restore
        # the captured warm-start/control values once more so the next
        # ``mj_step`` begins from the exact same integrator state.
        for name, value in snapshot.sim_extras.items():
            if hasattr(data, name):
                target = getattr(data, name)
                try:
                    target[...] = value
                except Exception:
                    try:
                        setattr(data, name, deepcopy(value))
                    except Exception:
                        pass
        self._restore_wrapper_state(snapshot.wrapper_state)
        self._restore_rng(snapshot.rng_state)
        self._clear_camera_buffers()
        # Derive an independent repeat stream after restoring the exact source
        # RNG state; this affects only stochastic task internals, never model
        # inputs or privileged labels.
        random.seed(int(branch_seed))
        np.random.seed(int(branch_seed) % (2**32))
        torch.manual_seed(int(branch_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(branch_seed))
        self.step_count = int(snapshot.step_count)
        self._force_reissue_after_restore = True
        self._observation = deepcopy(snapshot.observation)
        self._info = deepcopy(dict(snapshot.info))
        return deepcopy(snapshot.observation), deepcopy(dict(snapshot.info))

    def snapshot_hash(self, snapshot: _DuoSnapshot) -> str:
        return stable_tree_hash(
            {
                "sim_state": snapshot.sim_state,
                "sim_schema": snapshot.sim_schema,
                "sim_extras": snapshot.sim_extras,
                "wrapper_state": snapshot.wrapper_state,
                "step_count": snapshot.step_count,
            }
        )

    def local_observation_hash(self, observation: Any) -> str:
        return stable_tree_hash(observation)

    def _progress(self, observation: Any, info: Mapping[str, Any]) -> float:
        value = info.get("stage", 0)
        maximum = info.get("max_stage", 1)
        try:
            stage = float(np.asarray(value).reshape(-1)[0])
            stages = max(float(np.asarray(maximum).reshape(-1)[0]), 1.0)
            if stage >= stages:
                return 1.0
            internal: Mapping[str, Any] = {}
            for _path, obj in _walk_env(self.env):
                tracker = getattr(obj, "stage_tracker", None)
                if tracker is not None and isinstance(getattr(tracker, "internal_state", None), Mapping):
                    internal = tracker.internal_state
                    break
            names = _PROGRESS_PREDICATES.get(self.task, tuple(internal))
            present = [bool(internal[name]) for name in names if name in internal]
            within = float(np.mean(present)) if present else 0.0
            # The public discrete stage remains the dominant term.  The
            # bounded half-stage predicate fraction prevents label collapse
            # at h8/h16 without changing success semantics.
            return float(np.clip((stage + 0.5 * within) / stages, 0.0, 1.0))
        except Exception:
            return 0.0

    def progress(self, observation: Any, info: Mapping[str, Any]) -> float:
        return self._progress(observation, info)

    def step_absolute(self, absolute_action: np.ndarray) -> StepResult:
        action = np.asarray(absolute_action, dtype=np.float32)
        if action.shape != (2, ACTION_DIM) or not np.isfinite(action).all():
            raise ValueError(f"DuoBench absolute action must be [2,8], got {action.shape}")
        before = np.stack([arm_qpos(self._observation, arm) for arm in range(2)])
        command, action_audit = canonicalize_controller_action_with_audit(action)
        self._last_action_canonicalization = action_audit
        if self._force_reissue_after_restore:
            self._force_controller_reissue()
            self._force_reissue_after_restore = False
        payload = {
            "left": {"joints": command[0, :JOINT_DIM].copy(), "gripper": np.asarray([command[0, JOINT_DIM]], np.float32)},
            "right": {"joints": command[1, :JOINT_DIM].copy(), "gripper": np.asarray([command[1, JOINT_DIM]], np.float32)},
        }
        observation, reward, terminated, truncated, info = self.env.step(payload)
        observation, info = self._canonical_sensor_observation(observation, info)
        self.step_count += 1
        self._observation, self._info = observation, info
        after = np.stack([arm_qpos(observation, arm) for arm in range(2)])
        movement = np.linalg.norm(after[:, :JOINT_DIM] - before[:, :JOINT_DIM], axis=1)
        collisions = []
        for arm, key in enumerate(("left", "right")):
            row = info.get(key, {}) if isinstance(info, Mapping) else {}
            collisions.append(bool(row.get("collision", False)) if isinstance(row, Mapping) else False)
        # A conservative interaction flag based on end-effector proximity is
        # useful on tasks whose wrapper does not expose a named conflict bit.
        conflict = False
        try:
            poses = [np.asarray(observation[key]["tquat"], dtype=np.float32)[:3] for key in ("left", "right")]
            conflict = bool(np.linalg.norm(poses[0] - poses[1]) < 0.07)
        except Exception:
            pass
        success = bool(info.get("success", False)) if isinstance(info, Mapping) else False
        term = bool(np.asarray(terminated).all())
        trunc = bool(np.asarray(truncated).all())
        return StepResult(
            observation=observation,
            info=info,
            reward=float(np.asarray(reward).mean()),
            terminated=term,
            truncated=trunc,
            progress=self._progress(observation, info),
            success=success,
            executed_absolute=command,
            collision_or_drop=bool(any(collisions)),
            robot_conflict=conflict,
            duplicate_work=False,
            active=tuple(bool(value >= 0.02) for value in movement),
            all_joint_changes_below_threshold=bool(np.all(movement < 0.02)),
            diagnostics={
                "collision_by_arm": collisions,
                "step": self.step_count,
                "task": self.task,
                "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
                "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
                "action_canonicalization": action_audit,
            },
        )

    def close(self) -> None:
        self.env.close()


__all__ = [
    "DuoBenchEnvironment",
    "DuoBcoreProposalProvider",
    "DuoDinoProposalProvider",
    "IMAGE_PREPROCESS_ID",
    "arm_frames",
    "arm_qpos",
]
