"""DuoBench projection for CARE's predictive team-belief (B-core) stage.

Runtime rows are strictly decentralized: shared head DINO, the focal arm's
own wrist DINO, local qpos8, executed local action8 and legal task text.  The
peer wrist/state/action and future anchors live only in the removable training
teacher/targets.  Every sample is drawn from the full 550-demo corpus.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from before_we_act.team_belief.predictive_core import (
    FUTURE_OFFSETS_SECONDS,
    TeamBeliefConfig,
)
from .data import (
    ACTION_DIM,
    ACTION_HORIZON,
    ACTION_LAG_ROWS,
    EFFECTIVE_BATCH,
    HISTORY_STEPS,
    STATE_DIM,
    TASKS,
    DuoTemporalEpisode,
    DuoVisualCache,
    _load_task_arrays,
    load_manifest,
)
from .preprocessing import DINO_NORMALIZATION_ID, IMAGE_PREPROCESS_ID


BCORE_CACHE_SCHEMA = "before-we-act.duobench.bcore-cache/1"
BCORE_TRAINING_FORMAT = "before-we-act.duobench.dino-bcore-training/1"
BCORE_DEPLOYMENT_FORMAT = "before-we-act.duobench.dino-bcore-deployment/1"
BCORE_UPDATES = 120_000
BCORE_SEEDS = (20260815, 20260816, 20260817)
DATA_SEED = 20260815
TEAMMATE_ACTION_HORIZON = 16
DUO_FUTURE_OFFSETS_STEPS = (6, 12, 24, 48)


DUO_BELIEF_CONFIG = TeamBeliefConfig(
    n_belief_tokens=16,
    n_evidence_queries=4,
    event_capacity=4,
    temporal_layers=2,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    source_frequency_hz=30,
    future_offsets_steps=DUO_FUTURE_OFFSETS_STEPS,
    future_offsets_seconds=FUTURE_OFFSETS_SECONDS,
)

# CARE reads the complete legal B-core memory used by the registered MARS /
# RoboFactory path: the persistent belief tokens followed by the sparse event
# slots.  Keeping this contract next to the frozen Duo B-core architecture
# prevents branch collection, offline preparation, and paired validation from
# silently disagreeing about whether event memory is present.
DUO_CARE_MEMORY_TOKENS = (
    DUO_BELIEF_CONFIG.n_belief_tokens + DUO_BELIEF_CONFIG.event_capacity
)
DUO_CARE_MEMORY_WIDTH = DUO_BELIEF_CONFIG.d_model
DUO_CARE_MEMORY_SEMANTICS = (
    "PredictiveTeamBeliefPolicy.belief.mu+belief.event_memory"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb", buffering=0) as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_b0h_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    """Fail closed on ACT-like or metadata-only B0-H payloads."""

    if payload.get("format") != "before-we-act.duobench.dino-b0h/1":
        raise ValueError("Duo B-core requires the formal DINO B0-H checkpoint")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Duo B0-H checkpoint has no config mapping")
    required = {
        "policy_family": "TemporalHistoryPolicy",
        # CARE is the method family; it is not a substitute for the concrete
        # TemporalHistoryPolicy policy_family above.  Require the field
        # explicitly so omitted metadata cannot silently inherit a default.
        "method_family": "CARE",
        "architecture": "TemporalHistoryPolicy_hidden_residual",
        "image_preprocess_id": IMAGE_PREPROCESS_ID,
        "dino_normalization_id": DINO_NORMALIZATION_ID,
        "strict_dino_contract": True,
        "variant": "hidden_residual",
        "vision_backbone": "dinov3_vitb16_frozen",
        "action_encoding": "absolute_joint7_binary_gripper1",
        "action_lag_rows": ACTION_LAG_ROWS,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "horizon": ACTION_HORIZON,
        "history_steps": HISTORY_STEPS,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(
                f"Duo B0-H checkpoint differs at {key}: "
                f"{config.get(key)!r} != {expected!r}"
            )
    contract = str(config.get("policy_contract", ""))
    if "strictly_decentralized" not in contract or "own_wrist" not in contract:
        raise ValueError("Duo B0-H checkpoint is not strictly decentralized")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Duo B0-H checkpoint has no model state")
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
        raise ValueError(f"Duo B0-H state is missing architecture keys: {missing}")
    lowered = " ".join(str(value).lower() for value in config.values())
    if any(marker in lowered for marker in ("convnext", "resnet18", "actpolicy")):
        raise ValueError("legacy ACT/ConvNeXt markers are forbidden in Duo B0-H")
    return config


class DuoBcoreContextCache:
    """Small LRU over cached B0-H decoded hidden/base action arrays."""

    def __init__(self, root: str | Path, *, limit: int = 4) -> None:
        self.root = Path(root)
        self.limit = int(limit)
        receipt_path = self.root / "cache_receipt.json"
        self.receipt = json.loads(receipt_path.read_text())
        if (
            self.receipt.get("schema") != BCORE_CACHE_SCHEMA
            or self.receipt.get("status") != "PASSED"
        ):
            raise ValueError("Duo B-core context cache is incomplete")
        # The cache is generated from B0-H but consumed by the independent
        # PredictiveTeamBeliefPolicy.  Keep both identities explicit and
        # require the CARE method tag; otherwise a generic/legacy cache could
        # be relabeled at the training boundary.
        expected_receipt = {
            "policy_family": "TemporalHistoryPolicy",
            "method_family": "CARE",
            "downstream_policy_family": "PredictiveTeamBeliefPolicy",
            "vision_backbone": "dinov3_vitb16_frozen",
            "image_preprocess_id": IMAGE_PREPROCESS_ID,
            "dino_normalization_id": DINO_NORMALIZATION_ID,
            "strict_dino_contract": True,
            "action_encoding": "absolute_joint7_binary_gripper1",
            "action_lag_rows": ACTION_LAG_ROWS,
            "strictly_decentralized": True,
            "act_provider_allowed": False,
        }
        for key, expected in expected_receipt.items():
            if self.receipt.get(key) != expected:
                raise ValueError(
                    f"Duo B-core context cache differs at {key}: "
                    f"{self.receipt.get(key)!r} != {expected!r}"
                )
        task_tokens = json.loads((self.root / "task_tokens.json").read_text())
        if set(task_tokens) != set(TASKS):
            raise ValueError("Duo B-core task-token cache differs")
        self.task_tokens = {
            task: torch.tensor(task_tokens[task], dtype=torch.float32)
            for task in TASKS
        }
        if any(tuple(value.shape) != (384,) for value in self.task_tokens.values()):
            raise ValueError("Duo B-core task token must be 384-wide")
        self.values: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def paths(self, episode: DuoTemporalEpisode) -> tuple[Path, Path]:
        root = self.root / episode.task
        return (
            root / f"{episode.cache_key}.decoded.npy",
            root / f"{episode.cache_key}.base_action.npy",
        )

    def load(self, episode: DuoTemporalEpisode) -> tuple[np.ndarray, np.ndarray]:
        key = episode.cache_key
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        decoded_path, base_path = self.paths(episode)
        decoded = np.load(decoded_path, mmap_mode="r")
        base = np.load(base_path, mmap_mode="r")
        if decoded.shape != (episode.length - ACTION_LAG_ROWS, 2, ACTION_HORIZON, 384):
            raise ValueError(f"Duo B-core decoded cache shape differs: {decoded.shape}")
        if base.shape != (episode.length - ACTION_LAG_ROWS, 2, ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"Duo B-core base cache shape differs: {base.shape}")
        if decoded.dtype != np.float16 or base.dtype != np.float16:
            raise ValueError("Duo B-core context cache must be float16")
        self.values[key] = (decoded, base)
        while self.limit > 0 and len(self.values) > self.limit:
            self.values.popitem(last=False)
        return decoded, base


@dataclass(frozen=True)
class DuoBeliefRequest:
    episode_index: int
    arm: int
    time_index: int
    sample_key: str
    task: str


class DuoTeamBeliefDataset(Dataset):
    """Strict-local runtime row plus a structurally separate team teacher."""

    RUNTIME_FIELDS = frozenset(
        {
            "runtime_visual_tokens",
            "runtime_visual_mask",
            "history_qpos",
            "history_action",
            "history_mask",
            "action_history_mask",
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
        prepared_root: str | Path,
        episodes: Sequence[DuoTemporalEpisode],
        visual_cache_root: str | Path,
        bcore_cache_root: str | Path,
        *,
        cache_limit: int = 8,
    ) -> None:
        self.root = Path(prepared_root)
        self.episodes = list(episodes)
        manifest = load_manifest(self.root)
        norm = manifest["normalization"]
        self.q_mean = torch.tensor(norm["qpos_mean"], dtype=torch.float32)
        self.q_std = torch.tensor(norm["qpos_std"], dtype=torch.float32)
        self.a_mean = torch.tensor(norm["action_mean"], dtype=torch.float32)
        self.a_std = torch.tensor(norm["action_std"], dtype=torch.float32)
        self.visual = DuoVisualCache(visual_cache_root, limit=cache_limit)
        self.context = DuoBcoreContextCache(bcore_cache_root, limit=max(2, cache_limit // 2))
        self.task_data = {
            task: _load_task_arrays(self.root, task)
            for task in TASKS
            if any(episode.task == task for episode in self.episodes)
        }

    def __len__(self) -> int:
        return sum((episode.length - ACTION_LAG_ROWS) * 2 for episode in self.episodes)

    def __getitem__(self, request: DuoBeliefRequest | tuple) -> dict:
        if not isinstance(request, DuoBeliefRequest):
            request = DuoBeliefRequest(*request)
        episode = self.episodes[request.episode_index]
        if request.task != episode.task or request.arm not in (0, 1):
            raise ValueError("Duo B-core sample identity drift")
        t = int(request.time_index)
        decision_length = episode.length - ACTION_LAG_ROWS
        if decision_length < 1:
            raise ValueError("Duo episode has no legal lag-one decision")
        if not 0 <= t < decision_length:
            raise IndexError(t)
        ego = int(request.arm)
        peer = 1 - ego
        absolute = episode.start + t
        arrays = self.task_data[episode.task]
        state = arrays["state"].reshape(-1, 2, STATE_DIM)
        action_source = arrays["action"].reshape(-1, 2, ACTION_DIM)
        visual = self.visual.load(episode)
        decoded, base = self.context.load(episode)

        first = max(0, t - HISTORY_STEPS + 1)
        observation_indices = np.arange(first, t + 1)
        observation_offset = HISTORY_STEPS - len(observation_indices)
        action_first = max(1, t - HISTORY_STEPS + 1)
        action_indices = np.arange(action_first, t + 1)
        action_offset = HISTORY_STEPS - len(action_indices)

        runtime_visual = torch.zeros(HISTORY_STEPS, 2, 1, 768)
        runtime_mask = torch.zeros(HISTORY_STEPS, 2, 1, dtype=torch.bool)
        history_qpos = torch.zeros(HISTORY_STEPS, STATE_DIM)
        history_action = torch.zeros(HISTORY_STEPS, ACTION_DIM)
        history_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        action_history_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        local_visual_indices = episode.start + observation_indices
        runtime_visual[observation_offset:, 0, 0] = torch.from_numpy(
            np.array(visual["view_head"][observation_indices], dtype=np.float32, copy=True)
        )
        runtime_visual[observation_offset:, 1, 0] = torch.from_numpy(
            np.array(
                visual[f"view_wrist_{ego}"][observation_indices],
                dtype=np.float32,
                copy=True,
            )
        )
        runtime_mask[observation_offset:] = True
        qpos = np.array(state[local_visual_indices, ego], dtype=np.float32, copy=True)
        history_qpos[observation_offset:] = (
            torch.from_numpy(qpos) - self.q_mean
        ) / self.q_std
        history_mask[observation_offset:] = True
        if len(action_indices):
            past = np.array(
                action_source[episode.start + action_indices, ego],
                dtype=np.float32,
                copy=True,
            )
            history_action[action_offset:] = (
                torch.from_numpy(past) - self.a_mean
            ) / self.a_std
            action_history_mask[action_offset:] = True
        reset_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        reset_mask[observation_offset] = True

        current = torch.zeros(3, 1, 768)
        current[0, 0] = torch.from_numpy(
            np.array(visual["view_head"][t], dtype=np.float32, copy=True)
        )
        current[1, 0] = torch.from_numpy(
            np.array(visual[f"view_wrist_{ego}"][t], dtype=np.float32, copy=True)
        )
        current[2, 0] = torch.from_numpy(
            np.array(visual[f"view_wrist_{peer}"][t], dtype=np.float32, copy=True)
        )
        current_mask = torch.ones(3, 1, dtype=torch.bool)
        offsets = DUO_BELIEF_CONFIG.future_offsets_steps
        future = torch.zeros(len(offsets), 3, 1, 768)
        future_mask = torch.zeros(len(offsets), 3, 1, dtype=torch.bool)
        anchor_mask = torch.zeros(len(offsets), dtype=torch.bool)
        teammate_delta = torch.zeros(len(offsets), STATE_DIM)
        teammate_now = torch.from_numpy(
            np.array(state[absolute, peer], dtype=np.float32, copy=True)
        )
        for slot, delta in enumerate(offsets):
            target = t + int(delta)
            if target >= episode.length:
                continue
            future[slot, 0, 0] = torch.from_numpy(
                np.array(visual["view_head"][target], dtype=np.float32, copy=True)
            )
            future[slot, 1, 0] = torch.from_numpy(
                np.array(visual[f"view_wrist_{ego}"][target], dtype=np.float32, copy=True)
            )
            future[slot, 2, 0] = torch.from_numpy(
                np.array(visual[f"view_wrist_{peer}"][target], dtype=np.float32, copy=True)
            )
            future_mask[slot] = True
            anchor_mask[slot] = True
            teammate_future = torch.from_numpy(
                np.array(
                    state[episode.start + target, peer], dtype=np.float32, copy=True
                )
            )
            teammate_delta[slot] = (teammate_future - teammate_now) / self.q_std

        ego_state = (
            torch.from_numpy(np.array(state[absolute, ego], dtype=np.float32, copy=True))
            - self.q_mean
        ) / self.q_std
        peer_state = (teammate_now - self.q_mean) / self.q_std
        teammate_end = min(episode.end, absolute + TEAMMATE_ACTION_HORIZON)
        teammate_source = torch.from_numpy(
            np.array(
                action_source[absolute:teammate_end, peer],
                dtype=np.float32,
                copy=True,
            )
        )
        teammate_action = torch.zeros(TEAMMATE_ACTION_HORIZON, ACTION_DIM)
        teammate_action_mask = torch.zeros(TEAMMATE_ACTION_HORIZON, dtype=torch.bool)
        teammate_action[: len(teammate_source)] = (
            teammate_source - self.a_mean
        ) / self.a_std
        teammate_action_mask[: len(teammate_source)] = True

        target_start = absolute + ACTION_LAG_ROWS
        end = min(episode.end, target_start + ACTION_HORIZON)
        source = torch.from_numpy(
            np.array(action_source[target_start:end, ego], dtype=np.float32, copy=True)
        )
        normalized = (source - self.a_mean) / self.a_std
        target_action = torch.empty(ACTION_HORIZON, ACTION_DIM)
        target_action[: len(source)] = normalized
        target_action[len(source) :] = normalized[-1]
        target_mask = torch.zeros(ACTION_HORIZON, dtype=torch.bool)
        target_mask[: len(source)] = True
        phase = float(t / max(episode.length - 1, 1))
        return {
            "runtime_visual_tokens": runtime_visual,
            "runtime_visual_mask": runtime_mask,
            "history_qpos": history_qpos,
            "history_action": history_action,
            "history_mask": history_mask,
            "action_history_mask": action_history_mask,
            "episode_reset_mask": reset_mask,
            "task_token": self.context.task_tokens[episode.task].clone(),
            "decoded_action_hidden": torch.from_numpy(
                np.array(decoded[t, ego], dtype=np.float32, copy=True)
            ),
            # This is the complete frozen B0-H prediction, not its raw
            # pre-hidden-residual action head.
            "base_action": torch.from_numpy(
                np.array(base[t, ego], dtype=np.float32, copy=True)
            ),
            "teacher_current_visual_tokens": current,
            "teacher_current_visual_mask": current_mask,
            "teacher_future_visual_tokens": future,
            "teacher_future_visual_mask": future_mask,
            "teacher_future_anchor_mask": anchor_mask,
            "teacher_agent_state": torch.stack((ego_state, peer_state)),
            "teacher_agent_mask": torch.ones(2, dtype=torch.bool),
            "teacher_relative_agent_role": torch.tensor((0, 1), dtype=torch.long),
            "teammate_delta": teammate_delta,
            "teammate_action": teammate_action,
            "teammate_action_mask": teammate_action_mask,
            "action": target_action,
            "action_mask": target_mask,
            "pair_id": torch.tensor(request.episode_index * 1_000_000 + t),
            "phase_bin": torch.tensor(min(3, int(phase * 4)), dtype=torch.long),
            "task_index": torch.tensor(TASKS.index(episode.task), dtype=torch.long),
            "sample_key": request.sample_key,
        }


class DuoPairedSituationBatchSampler(Sampler[list[DuoBeliefRequest]]):
    """Batch48: two paired situations/task plus two rotating extra pairs."""

    def __init__(
        self,
        episodes: Sequence[DuoTemporalEpisode],
        *,
        updates: int,
        data_seed: int = DATA_SEED,
        start_update: int = 0,
    ) -> None:
        if EFFECTIVE_BATCH != 48:
            raise ValueError("Duo B-core sampler is frozen to batch48")
        if not 0 <= start_update <= updates:
            raise ValueError("invalid Duo B-core update interval")
        self.episodes = list(episodes)
        self.updates = int(updates)
        self.data_seed = int(data_seed)
        self.start_update = int(start_update)
        self.by_task: dict[str, list[int]] = defaultdict(list)
        for index, episode in enumerate(self.episodes):
            self.by_task[episode.task].append(index)
        if set(self.by_task) != set(TASKS):
            raise ValueError("Duo B-core sampler requires all 11 tasks")

    def __len__(self) -> int:
        return self.updates - self.start_update

    def requests_for_update(self, update: int) -> list[DuoBeliefRequest]:
        if not 1 <= update <= self.updates:
            raise IndexError(update)
        rng = random.Random(self.data_seed + 1_000_003 * update)
        extra_tasks = {
            TASKS[((update - 1) * 2 + offset) % len(TASKS)] for offset in range(2)
        }
        pairs: list[list[DuoBeliefRequest]] = []
        for task in TASKS:
            used: set[tuple[int, int]] = set()
            for _ in range(2 + int(task in extra_tasks)):
                while True:
                    episode_index = rng.choice(self.by_task[task])
                    episode = self.episodes[episode_index]
                    time_index = rng.randrange(episode.length - ACTION_LAG_ROWS)
                    if (episode_index, time_index) not in used:
                        used.add((episode_index, time_index))
                        break
                pair = []
                for arm in (0, 1):
                    identity = (
                        f"{episode.source_identity}:{episode.episode_id}:"
                        f"{arm}:{time_index}:duo-bcore"
                    )
                    pair.append(
                        DuoBeliefRequest(
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
        expected = Counter(
            {task: 4 + 2 * int(task in extra_tasks) for task in TASKS}
        )
        if len(rows) != EFFECTIVE_BATCH or Counter(row.task for row in rows) != expected:
            raise AssertionError("Duo B-core batch balance failed")
        return rows

    def __iter__(self) -> Iterable[list[DuoBeliefRequest]]:
        for update in range(self.start_update + 1, self.updates + 1):
            yield self.requests_for_update(update)

    def cursor_receipt(self, completed_update: int) -> dict:
        next_update = completed_update + 1
        keys = (
            [row.sample_key for row in self.requests_for_update(next_update)]
            if next_update <= self.updates
            else []
        )
        return {
            "format_version": "before-we-act.duobench.bcore-cursor/1",
            "data_seed": self.data_seed,
            "completed_update": int(completed_update),
            "next_update": next_update if keys else None,
            "next_sample_keys": keys,
            "effective_batch": EFFECTIVE_BATCH,
            "paired_arms": True,
            "base_pairs_per_task": 2,
            "extra_pairs_per_update": 2,
            "balance_cycle_updates": len(TASKS),
        }

    def validate_cursor(self, receipt: Mapping[str, object]) -> None:
        completed = int(receipt["completed_update"])
        if dict(receipt) != self.cursor_receipt(completed):
            raise ValueError("Duo B-core resume sample cursor drifted")


def fixed_diagnostic_requests(
    episodes: Sequence[DuoTemporalEpisode],
) -> list[DuoBeliefRequest]:
    """Frozen all-task offline selection rows; never use closed-loop outcomes."""

    rows: list[DuoBeliefRequest] = []
    for task in TASKS:
        candidates = [
            (index, episode)
            for index, episode in enumerate(episodes)
            if episode.task == task
        ]
        for ordinal in (0, len(candidates) // 2, len(candidates) - 1):
            episode_index, episode = candidates[ordinal]
            time_index = min(
                episode.length - 1,
                max(0, int(round((ordinal % 3 + 1) * episode.length / 4))),
            )
            for arm in (0, 1):
                rows.append(
                    DuoBeliefRequest(
                        episode_index,
                        arm,
                        time_index,
                        f"diagnostic:{episode.cache_key}:{arm}:{time_index}",
                        task,
                    )
                )
    return rows


__all__ = [
    "BCORE_CACHE_SCHEMA",
    "BCORE_DEPLOYMENT_FORMAT",
    "BCORE_SEEDS",
    "BCORE_TRAINING_FORMAT",
    "BCORE_UPDATES",
    "DATA_SEED",
    "DUO_BELIEF_CONFIG",
    "DUO_CARE_MEMORY_SEMANTICS",
    "DUO_CARE_MEMORY_TOKENS",
    "DUO_CARE_MEMORY_WIDTH",
    "DUO_FUTURE_OFFSETS_STEPS",
    "DuoBcoreContextCache",
    "DuoBeliefRequest",
    "DuoPairedSituationBatchSampler",
    "DuoTeamBeliefDataset",
    "TEAMMATE_ACTION_HORIZON",
    "canonical_json_hash",
    "fixed_diagnostic_requests",
    "sha256_file",
    "validate_b0h_payload",
]
