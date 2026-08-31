"""BiCoord adapter for CARE's predictive team-belief (B-core/TUNE) stage.

The source benchmark stores both arms in one HDF5 episode.  This module keeps
that representation at the data boundary while exposing the same strict local
runtime stream used by :class:`PredictiveTeamBeliefPolicy`: head RGB, the
focal wrist RGB, focal qpos and focal executed actions.  Peer observations,
peer state and future anchors are returned in a separate teacher namespace and
are never part of ``RUNTIME_FIELDS``.

BiCoord records an observation at row ``t`` and the command selected for that
observation at row ``t + 1``.  Consequently a legal decision has
``0 <= t < T - 1`` and every action target below starts at ``t + 1``.  This
one-row lag is deliberately repeated in the cache, teacher targets and
sampler instead of being inferred by callers.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from before_we_act.temporal_history_data import task_text_tensor
from before_we_act.team_belief.predictive_core import (
    FUTURE_OFFSETS_SECONDS,
    TeamBeliefConfig,
)

from .config import (
    ACTION_DIM,
    ACTION_ENCODING,
    ACTION_HORIZON,
    ARM_COUNT,
    BASE_SAMPLES_PER_TASK,
    DATASET_REVISION,
    DINO_HIDDEN_SIZE,
    EFFECTIVE_BATCH,
    EPISODES_PER_TASK,
    EXTRA_SAMPLES_PER_UPDATE,
    HISTORY_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    STATE_DIM,
    TASKS,
    TASK_TEXT,
    TOTAL_EPISODES,
    FORMAL_SEEDS,
    GRIPPER_ENCODING,
    GRIPPER_NATIVE_RANGE,
)
from .data import (
    BiCoordEpisode,
    BiCoordTemporalRequest,
    BiCoordVisualCache,
    discover_bicoord_episodes,
    load_normalization_receipt,
)
from .hdf5_data import BiCoordHDF5Reader, sha256_file
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID


BCORE_CACHE_SCHEMA = "before-we-act.bicoord.bcore-cache/1"
BCORE_TRAINING_FORMAT = "before-we-act.bicoord.dino-bcore-training/1"
BCORE_DEPLOYMENT_FORMAT = "before-we-act.bicoord.dino-bcore-deployment/1"
BCORE_UPDATES = 120_000
BCORE_SEEDS = tuple(int(value) for value in FORMAL_SEEDS)
DATA_SEED = 20260901
TEAMMATE_ACTION_HORIZON = 16

# B-core rotates six extra paired situations per update over 18 task
# buckets.  Compute the complete circular period explicitly; integer
# division happens to work for today's 18/6 values but is not the definition
# and regresses to one for non-divisible batch contracts.
BALANCE_CYCLE_UPDATES = len(TASKS) // math.gcd(
    len(TASKS), EFFECTIVE_BATCH // ARM_COUNT - len(TASKS)
)

# The official GO1 -> LeRobot conversion records BiCoord at 15 FPS.  The
# predictive core's registered horizons are in seconds, therefore the exact
# row offsets are 3/6/12/24.  Do not substitute the old DuoBench 20 Hz values
# or infer a simulator control-loop rate from raw HDF5 timestamps.
BICOORD_SOURCE_FREQUENCY_HZ = 15
BICOORD_FUTURE_OFFSETS_STEPS = (3, 6, 12, 24)

BICOORD_BELIEF_CONFIG = TeamBeliefConfig(
    n_belief_tokens=16,
    n_evidence_queries=4,
    event_capacity=4,
    temporal_layers=2,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    source_frequency_hz=BICOORD_SOURCE_FREQUENCY_HZ,
    future_offsets_steps=BICOORD_FUTURE_OFFSETS_STEPS,
    future_offsets_seconds=FUTURE_OFFSETS_SECONDS,
)

# CARE consumes persistent belief tokens followed by event slots.  Keep these
# values explicit in every receipt so branch collection and deployment cannot
# silently drop the event memory.
BICOORD_CARE_MEMORY_TOKENS = (
    BICOORD_BELIEF_CONFIG.n_belief_tokens + BICOORD_BELIEF_CONFIG.event_capacity
)
BICOORD_CARE_MEMORY_WIDTH = BICOORD_BELIEF_CONFIG.d_model
BICOORD_CARE_MEMORY_SEMANTICS = (
    "PredictiveTeamBeliefPolicy.belief.mu+belief.event_memory"
)


def canonical_json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _decision_count(episode: BiCoordEpisode) -> int:
    value = int(episode.length) - 1
    if value < 1:
        raise ValueError(f"BiCoord episode has no legal lag-one decision: {episode.path}")
    return value


def _episode_key(episode: BiCoordEpisode) -> str:
    return str(episode.source_identity)


def _sha256(path: str | Path) -> str:
    return sha256_file(path)


def _validate_b0h_contract(config: Mapping[str, Any]) -> None:
    expected = {
        "policy_family": "TemporalHistoryPolicy",
        "method_family": "CARE",
        "architecture": "TemporalHistoryPolicy_hidden_residual",
        "benchmark_adapter": "BiCoord",
        "vision_backbone": "dinov3_vitb16_frozen",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "variant": "hidden_residual",
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "horizon": ACTION_HORIZON,
        "history_steps": HISTORY_STEPS,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"BiCoord B0-H checkpoint differs at {key}: {config.get(key)!r} != {value!r}"
            )
    contract = str(config.get("policy_contract", ""))
    if "strictly_decentralized" not in contract or "own_wrist" not in contract:
        raise ValueError("BiCoord B0-H checkpoint is not strictly decentralized")
    if config.get("shared_weights") is not True:
        raise ValueError("BiCoord B0-H checkpoint does not declare shared weights")


def validate_b0h_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed on ACT/legacy or metadata-only B0-H checkpoints."""

    if payload.get("format") not in (
        "before-we-act.bicoord.dino-b0h/1",
        "before-we-act.bicoord.dino-b0h-checkpoint/1",
    ):
        raise ValueError("BiCoord B-core requires a formal BiCoord DINO B0-H checkpoint")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("BiCoord B0-H checkpoint has no config mapping")
    _validate_b0h_contract(config)
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("BiCoord B0-H checkpoint has no model state")
    keys = tuple(str(key) for key in state)
    required_prefixes = (
        "vision.",
        "history_encoder.",
        "history_action.",
        "decoder.",
        "hidden_residual.",
    )
    missing = [
        prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in keys)
    ]
    if missing:
        raise ValueError(f"BiCoord B0-H state is missing architecture keys: {missing}")
    lowered = " ".join(str(value).lower() for value in config.values())
    if any(marker in lowered for marker in ("convnext", "resnet18", "actpolicy")):
        raise ValueError("legacy ACT/ConvNeXt markers are forbidden in BiCoord B0-H")
    return config


class BiCoordBcoreContextCache:
    """Bounded LRU over frozen B0-H hidden/action contexts."""

    def __init__(self, root: str | Path, *, limit: int = 8) -> None:
        self.root = Path(root).expanduser().resolve()
        self.limit = max(0, int(limit))
        receipt_path = self.root / "cache_receipt.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(receipt_path)
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid BiCoord B-core cache receipt: {receipt_path}") from error
        if receipt.get("schema") != BCORE_CACHE_SCHEMA or receipt.get("status") != "PASSED":
            raise ValueError("BiCoord B-core context cache is incomplete")
        expected = {
            "policy_family": "TemporalHistoryPolicy",
            "method_family": "CARE",
            "downstream_policy_family": "PredictiveTeamBeliefPolicy",
            "benchmark_adapter": "BiCoord",
            "vision_backbone": "dinov3_vitb16_frozen",
            "image_preprocess_id": IMAGE_PREPROCESS_ID,
            "dino_normalization_id": DINO_NORMALIZATION_ID,
            "strict_dino_contract": True,
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
            "strictly_decentralized": True,
            "act_provider_allowed": False,
            "source_frequency_hz": BICOORD_SOURCE_FREQUENCY_HZ,
            "action_lag_rows": 1,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(
                    f"BiCoord B-core cache differs at {key}: {receipt.get(key)!r} != {value!r}"
                )
        task_path = self.root / "task_tokens.json"
        if not task_path.is_file():
            raise FileNotFoundError(task_path)
        task_tokens = json.loads(task_path.read_text(encoding="utf-8"))
        if not isinstance(task_tokens, Mapping) or set(task_tokens) != set(TASKS):
            raise ValueError("BiCoord B-core task-token cache differs")
        self.task_tokens = {
            task: torch.tensor(task_tokens[task], dtype=torch.float32) for task in TASKS
        }
        if any(tuple(value.shape) != (BICOORD_CARE_MEMORY_WIDTH,) for value in self.task_tokens.values()):
            raise ValueError("BiCoord B-core task token must be 384-wide")
        self.receipt = receipt
        self.values: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def paths(self, episode: BiCoordEpisode) -> tuple[Path, Path, Path]:
        root = self.root / episode.task
        key = _episode_key(episode)
        return (
            root / f"{key}.decoded.npy",
            root / f"{key}.base_action.npy",
            root / f"{key}.complete.json",
        )

    def load(self, episode: BiCoordEpisode) -> tuple[np.ndarray, np.ndarray]:
        key = _episode_key(episode)
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        decoded_path, base_path, marker_path = self.paths(episode)
        if not decoded_path.is_file() or not base_path.is_file() or not marker_path.is_file():
            raise FileNotFoundError(f"incomplete BiCoord B-core cache for {key}")
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid BiCoord B-core episode marker: {marker_path}") from error
        if marker.get("status") != "PASSED" or marker.get("source_identity") != key:
            raise ValueError(f"BiCoord B-core cache marker drift: {marker_path}")
        decoded = np.load(decoded_path, mmap_mode="r", allow_pickle=False)
        base = np.load(base_path, mmap_mode="r", allow_pickle=False)
        expected_decoded = (_decision_count(episode), ARM_COUNT, ACTION_HORIZON, BICOORD_CARE_MEMORY_WIDTH)
        expected_base = (_decision_count(episode), ARM_COUNT, ACTION_HORIZON, ACTION_DIM)
        if decoded.shape != expected_decoded:
            raise ValueError(f"BiCoord decoded context shape differs: {decoded.shape} != {expected_decoded}")
        if base.shape != expected_base:
            raise ValueError(f"BiCoord base-action context shape differs: {base.shape} != {expected_base}")
        if decoded.dtype != np.float16 or base.dtype != np.float16:
            raise ValueError("BiCoord B-core context cache must be float16")
        self.values[key] = (decoded, base)
        while self.limit and len(self.values) > self.limit:
            self.values.popitem(last=False)
        return decoded, base


class _BiCoordNumericCache:
    """Bounded LRU over the native paired 7-D source rows.

    B-core never needs JPEG bytes: all legal visual evidence already lives in
    the frozen DINO cache.  Keeping the small joint/gripper arrays in a worker-
    local LRU also avoids reopening an HDF5 file for every history row during
    the 3 x 120k-update protocol.
    """

    def __init__(self, *, limit: int = 32) -> None:
        self.limit = max(0, int(limit))
        self.values: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, episode: BiCoordEpisode, reader: BiCoordHDF5Reader) -> np.ndarray:
        key = _episode_key(episode)
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        with reader._open() as handle:
            left = np.asarray(handle["joint_action/left_arm"], dtype=np.float32)
            right = np.asarray(handle["joint_action/right_arm"], dtype=np.float32)
            left_gripper = np.asarray(
                handle["joint_action/left_gripper"], dtype=np.float32
            )[:, None]
            right_gripper = np.asarray(
                handle["joint_action/right_gripper"], dtype=np.float32
            )[:, None]
        value = np.stack(
            (
                np.concatenate((left, left_gripper), axis=1),
                np.concatenate((right, right_gripper), axis=1),
            ),
            axis=1,
        )
        expected = (int(episode.length), ARM_COUNT, STATE_DIM)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(
                f"invalid BiCoord numeric source for {episode.path}: "
                f"{value.shape} != {expected}"
            )
        self.values[key] = value
        while self.limit and len(self.values) > self.limit:
            self.values.popitem(last=False)
        return value


@dataclass(frozen=True)
class BiCoordBeliefRequest:
    episode_index: int
    arm: int
    time_index: int
    sample_key: str
    task: str


class BiCoordTeamBeliefDataset(Dataset):
    """Paired local rows plus a removable privileged teacher namespace."""

    RUNTIME_FIELDS = frozenset(
        {
            "runtime_visual_tokens",
            "runtime_visual_mask",
            "history_qpos",
            "history_action",
            "history_mask",
            "action_history_mask",
            "task_bytes",
            "task_text_mask",
            "task_token",
            "episode_reset_mask",
            "decoded_action_hidden",
            "base_action",
        }
    )
    TEACHER_FIELDS = frozenset(
        {
            "teacher_current_visual_tokens",
            "teacher_current_visual_mask",
            "teacher_future_visual_tokens",
            "teacher_future_visual_mask",
            "teacher_future_anchor_mask",
            "teacher_agent_state",
            "teacher_agent_mask",
            "teacher_relative_agent_role",
        }
    )

    def __init__(
        self,
        episodes: Sequence[BiCoordEpisode] | str | Path,
        normalization: Mapping[str, Any] | str | Path,
        visual_cache_root: str | Path,
        bcore_cache_root: str | Path,
        *,
        image_height: int = IMAGE_HEIGHT,
        image_width: int = IMAGE_WIDTH,
        cache_limit: int = 8,
    ) -> None:
        if isinstance(episodes, (str, Path)):
            episodes = discover_bicoord_episodes(episodes, require_formal=False)
        self.episodes = list(episodes)
        if not self.episodes:
            raise ValueError("BiCoord B-core dataset cannot be empty")
        self.normalization = load_normalization_receipt(normalization, require_formal=False)
        self.q_mean = torch.tensor(self.normalization["qpos_mean"], dtype=torch.float32)
        self.q_std = torch.tensor(self.normalization["qpos_std"], dtype=torch.float32)
        self.a_mean = torch.tensor(self.normalization["action_mean"], dtype=torch.float32)
        self.a_std = torch.tensor(self.normalization["action_std"], dtype=torch.float32)
        if self.q_mean.shape != (STATE_DIM,) or self.a_mean.shape != (ACTION_DIM,):
            raise ValueError("BiCoord B-core normalization must be seven-dimensional")
        if torch.any(self.q_std <= 0) or torch.any(self.a_std <= 0):
            raise ValueError("BiCoord B-core normalization standard deviations must be positive")
        self.visual = BiCoordVisualCache(visual_cache_root, limit=cache_limit, require_receipt=True)
        self.context = BiCoordBcoreContextCache(bcore_cache_root, limit=max(2, cache_limit // 2))
        # Image dimensions belong to the upstream B0-H cache construction.
        # B-core consumes frozen 768-D features and must never decode/re-resize
        # JPEGs on its hot training path.
        if int(image_height) != IMAGE_HEIGHT or int(image_width) != IMAGE_WIDTH:
            raise ValueError("BiCoord B-core image contract must remain 224x224")
        self.numeric = _BiCoordNumericCache(limit=max(8, cache_limit * 2))
        self._readers = {
            episode.source_identity: BiCoordHDF5Reader(
                episode.path,
                task=episode.task,
                episode_id=episode.episode_id,
                stage_path=episode.stage_path,
                instruction_path=episode.instruction_path,
            )
            for episode in self.episodes
        }

    def __len__(self) -> int:
        return sum(_decision_count(episode) * ARM_COUNT for episode in self.episodes)

    def __getitem__(self, request: BiCoordBeliefRequest | BiCoordTemporalRequest | tuple[Any, ...]) -> dict[str, Any]:
        if isinstance(request, BiCoordTemporalRequest):
            request = BiCoordBeliefRequest(
                request.episode_index,
                request.arm,
                request.time_index,
                request.sample_key,
                request.task,
            )
        elif not isinstance(request, BiCoordBeliefRequest):
            request = BiCoordBeliefRequest(*request)
        if not 0 <= request.episode_index < len(self.episodes):
            raise IndexError(request.episode_index)
        episode = self.episodes[request.episode_index]
        if request.task != episode.task or request.arm not in (0, 1):
            raise ValueError("BiCoord B-core sample identity drift")
        t = int(request.time_index)
        if not 0 <= t < _decision_count(episode):
            raise IndexError(f"time index {t} has no lag-one target")
        ego = int(request.arm)
        peer = 1 - ego
        reader = self._readers[episode.source_identity]
        numeric = self.numeric.load(episode, reader)
        visual = self.visual.load(episode)
        decoded, base = self.context.load(episode)

        # B-core receives the exact same two-view causal history as B0-H.  The
        # extra singleton patch dimension is the interface expected by the
        # predictive core's multi-view compressor.
        observation_first = max(0, t - HISTORY_STEPS + 1)
        observation_indices = np.arange(observation_first, t + 1, dtype=np.int64)
        observation_offset = HISTORY_STEPS - len(observation_indices)
        action_first = max(0, t - HISTORY_STEPS)
        action_indices = np.arange(action_first, t, dtype=np.int64)
        action_offset = HISTORY_STEPS - len(action_indices)

        history_visual = torch.zeros(
            HISTORY_STEPS, 2, DINO_HIDDEN_SIZE, dtype=torch.float32
        )
        history_visual[observation_offset:, 0] = torch.from_numpy(
            np.array(visual["view_head"][observation_indices], dtype=np.float32, copy=True)
        )
        history_visual[observation_offset:, 1] = torch.from_numpy(
            np.array(
                visual[f"view_wrist_{ego}"][observation_indices],
                dtype=np.float32,
                copy=True,
            )
        )
        history_qpos = torch.zeros(HISTORY_STEPS, STATE_DIM, dtype=torch.float32)
        history_qpos[observation_offset:] = (
            torch.from_numpy(np.array(numeric[observation_indices, ego], copy=True))
            - self.q_mean
        ) / self.q_std
        history_action = torch.zeros(HISTORY_STEPS, ACTION_DIM, dtype=torch.float32)
        if len(action_indices):
            # Source row i+1 is the command paired with observation row i.
            history_action[action_offset:] = (
                torch.from_numpy(np.array(numeric[action_indices + 1, ego], copy=True))
                - self.a_mean
            ) / self.a_std
        history_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        history_mask[observation_offset:] = True
        action_history_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        action_history_mask[action_offset:] = True
        runtime_visual = history_visual.unsqueeze(-2)
        runtime_mask = history_mask[:, None, None].expand(-1, 2, 1).clone()
        episode_reset_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        episode_reset_mask[observation_offset] = True

        current = torch.zeros(3, 1, DINO_HIDDEN_SIZE, dtype=torch.float32)
        current[0, 0] = torch.from_numpy(np.array(visual["view_head"][t], dtype=np.float32, copy=True))
        current[1, 0] = torch.from_numpy(np.array(visual[f"view_wrist_{ego}"][t], dtype=np.float32, copy=True))
        current[2, 0] = torch.from_numpy(np.array(visual[f"view_wrist_{peer}"][t], dtype=np.float32, copy=True))
        current_mask = torch.ones(3, 1, dtype=torch.bool)

        offsets = BICOORD_BELIEF_CONFIG.future_offsets_steps
        future = torch.zeros(len(offsets), 3, 1, DINO_HIDDEN_SIZE, dtype=torch.float32)
        future_mask = torch.zeros(len(offsets), 3, 1, dtype=torch.bool)
        anchor_mask = torch.zeros(len(offsets), dtype=torch.bool)
        teammate_delta = torch.zeros(len(offsets), STATE_DIM, dtype=torch.float32)
        teammate_now_raw = torch.from_numpy(np.array(numeric[t, peer], copy=True))
        for slot, delta in enumerate(offsets):
            target = t + int(delta)
            if target >= episode.length:
                continue
            future[slot, 0, 0] = torch.from_numpy(np.array(visual["view_head"][target], dtype=np.float32, copy=True))
            future[slot, 1, 0] = torch.from_numpy(np.array(visual[f"view_wrist_{ego}"][target], dtype=np.float32, copy=True))
            future[slot, 2, 0] = torch.from_numpy(np.array(visual[f"view_wrist_{peer}"][target], dtype=np.float32, copy=True))
            future_mask[slot] = True
            anchor_mask[slot] = True
            future_peer = torch.from_numpy(np.array(numeric[target, peer], copy=True))
            teammate_delta[slot] = (future_peer - teammate_now_raw) / self.q_std

        ego_state = (
            torch.from_numpy(np.array(numeric[t, ego], copy=True)) - self.q_mean
        ) / self.q_std
        peer_state = (teammate_now_raw - self.q_mean) / self.q_std
        teacher_agent_state = torch.stack((ego_state, peer_state)).float()

        # The peer's executed command has the same t -> t+1 lag as the ego
        # target.  Never use row t as a command merely because it is available.
        teammate_end = min(episode.length, t + 1 + TEAMMATE_ACTION_HORIZON)
        teammate_rows = numeric[t + 1 : teammate_end, peer]
        teammate_action = torch.zeros(TEAMMATE_ACTION_HORIZON, ACTION_DIM, dtype=torch.float32)
        teammate_action_mask = torch.zeros(TEAMMATE_ACTION_HORIZON, dtype=torch.bool)
        if len(teammate_rows):
            source = torch.from_numpy(np.array(teammate_rows, dtype=np.float32, copy=True))
            teammate_action[: len(source)] = (source - self.a_mean) / self.a_std
            teammate_action_mask[: len(source)] = True

        target_end = min(episode.length, t + 1 + ACTION_HORIZON)
        target_source = torch.from_numpy(
            np.array(numeric[t + 1 : target_end, ego], dtype=np.float32, copy=True)
        )
        if not len(target_source):
            raise RuntimeError("BiCoord B-core causal request has no lag-one target")
        target = (target_source - self.a_mean) / self.a_std
        action = torch.empty(ACTION_HORIZON, ACTION_DIM, dtype=torch.float32)
        action[: len(target)] = target
        action[len(target) :] = target[-1]
        action_mask = torch.zeros(ACTION_HORIZON, dtype=torch.bool)
        action_mask[: len(target)] = True
        task_bytes, task_text_mask = task_text_tensor(TASK_TEXT[episode.task])

        phase = float(t / max(_decision_count(episode) - 1, 1))
        output: dict[str, Any] = {
                "history_visual_raw": history_visual,
                "history_qpos": history_qpos,
                "history_action": history_action,
                "history_mask": history_mask,
                "action_history_mask": action_history_mask,
                "task_bytes": task_bytes,
                "task_text_mask": task_text_mask,
                "episode_reset": torch.tensor(t == 0, dtype=torch.bool),
                "action": action,
                "action_mask": action_mask,
                "runtime_visual_tokens": runtime_visual,
                "runtime_visual_mask": runtime_mask,
                "episode_reset_mask": episode_reset_mask,
                "task_token": self.context.task_tokens[episode.task].clone(),
                "decoded_action_hidden": torch.from_numpy(
                    np.array(decoded[t, ego], dtype=np.float32, copy=True)
                ),
                "base_action": torch.from_numpy(
                    np.array(base[t, ego], dtype=np.float32, copy=True)
                ),
                "teacher_current_visual_tokens": current,
                "teacher_current_visual_mask": current_mask,
                "teacher_future_visual_tokens": future,
                "teacher_future_visual_mask": future_mask,
                "teacher_future_anchor_mask": anchor_mask,
                "teacher_agent_state": teacher_agent_state,
                "teacher_agent_mask": torch.ones(ARM_COUNT, dtype=torch.bool),
                "teacher_relative_agent_role": torch.tensor((0, 1), dtype=torch.long),
                "teammate_delta": teammate_delta,
                "teammate_action": teammate_action,
                "teammate_action_mask": teammate_action_mask,
                "pair_id": torch.tensor(request.episode_index * 1_000_000 + t, dtype=torch.long),
                "phase_bin": torch.tensor(min(3, int(phase * 4)), dtype=torch.long),
                "task_index": torch.tensor(TASKS.index(episode.task), dtype=torch.long),
                "sample_key": request.sample_key,
                "episode_index": torch.tensor(request.episode_index, dtype=torch.long),
                "episode_id": torch.tensor(episode.episode_id, dtype=torch.long),
                "time_index": torch.tensor(t, dtype=torch.long),
                "arm": torch.tensor(ego, dtype=torch.long),
                "source_identity": episode.source_identity,
            }
        return output


class BiCoordPairedSituationBatchSampler(Sampler[list[BiCoordBeliefRequest]]):
    """Deterministic paired task-balanced batches (48 rows/update).

    Each task contributes one paired situation (two arm-local rows) and six
    extra paired situations rotate across tasks.  Thus every update has 24
    situations/48 rows, every task is represented, and over a three-update
    cycle each task receives exactly one extra pair (two extra arm rows).
    """

    BASE_PAIRS_PER_TASK = 1
    EXTRA_PAIRS_PER_UPDATE = EFFECTIVE_BATCH // ARM_COUNT - len(TASKS)

    def __init__(
        self,
        episodes: Sequence[BiCoordEpisode],
        *,
        updates: int,
        data_seed: int = DATA_SEED,
        start_update: int = 0,
    ) -> None:
        if EFFECTIVE_BATCH != 48 or self.EXTRA_PAIRS_PER_UPDATE != 6:
            raise ValueError("BiCoord B-core batch contract drifted")
        if updates < 1 or not 0 <= start_update <= updates:
            raise ValueError("invalid BiCoord B-core update interval")
        self.episodes = list(episodes)
        self.updates = int(updates)
        self.data_seed = int(data_seed)
        self.start_update = int(start_update)
        self.by_task: dict[str, list[int]] = defaultdict(list)
        for index, episode in enumerate(self.episodes):
            if _decision_count(episode) < 1:
                raise ValueError(f"episode has no legal decisions: {episode.path}")
            self.by_task[episode.task].append(index)
        if set(self.by_task) != set(TASKS):
            raise ValueError("BiCoord B-core sampler requires all 18 task buckets")

    def __len__(self) -> int:
        return self.updates - self.start_update

    def _extra_tasks(self, update: int) -> tuple[str, ...]:
        start = (update - 1) * self.EXTRA_PAIRS_PER_UPDATE
        return tuple(
            TASKS[(start + offset) % len(TASKS)] for offset in range(self.EXTRA_PAIRS_PER_UPDATE)
        )

    def requests_for_update(self, update: int) -> list[BiCoordBeliefRequest]:
        if not 1 <= update <= self.updates:
            raise IndexError(update)
        rng = random.Random(self.data_seed + 1_000_003 * update)
        extra = Counter(self._extra_tasks(update))
        pairs: list[list[BiCoordBeliefRequest]] = []
        for task in TASKS:
            used: set[tuple[int, int]] = set()
            for _ in range(self.BASE_PAIRS_PER_TASK + extra[task]):
                while True:
                    episode_index = rng.choice(self.by_task[task])
                    episode = self.episodes[episode_index]
                    time_index = rng.randrange(_decision_count(episode))
                    situation = (episode_index, time_index)
                    if situation not in used:
                        used.add(situation)
                        break
                pair: list[BiCoordBeliefRequest] = []
                for arm in (0, 1):
                    identity = (
                        f"{episode.source_identity}:{episode.episode_id}:{arm}:{time_index}:bcore"
                    )
                    pair.append(
                        BiCoordBeliefRequest(
                            episode_index,
                            arm,
                            time_index,
                            hashlib.sha256(identity.encode()).hexdigest(),
                            task,
                        )
                    )
                pairs.append(pair)
        rng.shuffle(pairs)
        rows = [row for pair in pairs for row in pair]
        expected_counts = Counter(
            {
                task: ARM_COUNT * (self.BASE_PAIRS_PER_TASK + extra[task])
                for task in TASKS
            }
        )
        if len(rows) != EFFECTIVE_BATCH or Counter(row.task for row in rows) != expected_counts:
            raise AssertionError("BiCoord B-core task balance failed")
        return rows

    def __iter__(self) -> Iterable[list[BiCoordBeliefRequest]]:
        for update in range(self.start_update + 1, self.updates + 1):
            yield self.requests_for_update(update)

    def cursor_receipt(self, completed_update: int) -> dict[str, Any]:
        if not 0 <= completed_update <= self.updates:
            raise ValueError(completed_update)
        next_update = completed_update + 1
        rows = self.requests_for_update(next_update) if next_update <= self.updates else []
        return {
            "format_version": "before-we-act.bicoord.bcore-cursor/1",
            "data_seed": self.data_seed,
            "completed_update": int(completed_update),
            "next_update": next_update if rows else None,
            "next_sample_keys": [row.sample_key for row in rows],
            "effective_batch": EFFECTIVE_BATCH,
            "paired_arms": True,
            "base_pairs_per_task": self.BASE_PAIRS_PER_TASK,
            "extra_pairs_per_update": self.EXTRA_PAIRS_PER_UPDATE,
            "balance_cycle_updates": BALANCE_CYCLE_UPDATES,
            "action_lag_rows": 1,
        }

    def validate_cursor(self, receipt: Mapping[str, Any]) -> int:
        completed = int(receipt["completed_update"])
        if dict(receipt) != self.cursor_receipt(completed):
            raise ValueError("BiCoord B-core resume sample cursor drifted")
        return completed


def fixed_diagnostic_requests(
    episodes: Sequence[BiCoordEpisode],
) -> list[BiCoordBeliefRequest]:
    """Deterministic all-task offline rows; no closed-loop outcomes involved."""

    rows: list[BiCoordBeliefRequest] = []
    for task in TASKS:
        candidates = [
            (index, episode) for index, episode in enumerate(episodes) if episode.task == task
        ]
        if not candidates:
            continue
        ordinals = (0, len(candidates) // 2, len(candidates) - 1)
        for ordinal in ordinals:
            episode_index, episode = candidates[ordinal]
            decision_count = _decision_count(episode)
            time_index = min(
                decision_count - 1,
                max(0, int(round((ordinal % 3 + 1) * decision_count / 4))),
            )
            for arm in (0, 1):
                key = f"diagnostic:{episode.source_identity}:{arm}:{time_index}"
                rows.append(
                    BiCoordBeliefRequest(
                        episode_index,
                        arm,
                        time_index,
                        key,
                        task,
                    )
                )
    return rows


# Generic aliases used by launchers and tests.
TeamBeliefDataset = BiCoordTeamBeliefDataset
PairedSituationBatchSampler = BiCoordPairedSituationBatchSampler


__all__ = [
    "BCORE_CACHE_SCHEMA",
    "BCORE_DEPLOYMENT_FORMAT",
    "BCORE_SEEDS",
    "BCORE_TRAINING_FORMAT",
    "BCORE_UPDATES",
    "BICOORD_BELIEF_CONFIG",
    "BICOORD_CARE_MEMORY_SEMANTICS",
    "BICOORD_CARE_MEMORY_TOKENS",
    "BICOORD_CARE_MEMORY_WIDTH",
    "BICOORD_FUTURE_OFFSETS_STEPS",
    "BICOORD_SOURCE_FREQUENCY_HZ",
    "BiCoordBcoreContextCache",
    "BiCoordBeliefRequest",
    "BiCoordPairedSituationBatchSampler",
    "BiCoordTeamBeliefDataset",
    "DATA_SEED",
    "PairedSituationBatchSampler",
    "TeamBeliefDataset",
    "TEAMMATE_ACTION_HORIZON",
    "canonical_json_hash",
    "fixed_diagnostic_requests",
    "sha256_file",
    "validate_b0h_payload",
]
