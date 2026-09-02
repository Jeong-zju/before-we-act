"""DuoBench data contract for the official temporal CARE reference policy.

Each demonstration is projected into two independent arm streams.  A stream
contains shared head RGB, that arm's wrist RGB, local qpos8, and its own past
absolute action8.  Peer wrist/proprioception/action and simulator state never
enter a sample.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from before_we_act.temporal_history_data import task_text_tensor
from deployment.duo_act.protocol import (
    FORMAL_DATASET_REVISION,
    FORMAL_EPISODES_PER_TASK,
    VALIDATION_HORIZON_METHOD,
    VALIDATION_HORIZON_QUANTILE,
    VALIDATION_MAX_STEPS,
)
from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SCHEMA,
    ACTION_TARGET_CONTRACT_SHA256,
    validate_action_target_contract,
)

from .preprocessing import IMAGE_PREPROCESS_ID, resize_rgb_batch


TASKS = (
    "ball_maze",
    "bin_sort",
    "block_balance",
    "carry_pot",
    "hinge_chest",
    "join_blocks",
    "pour_marbles",
    "spring_door",
    "transfer_cube",
    "transfer_gate",
    "transfer_reorient",
)

# The frozen byte encoder accepts at most 64 UTF-8 bytes.  These texts retain
# the full task identity without injecting arm identity or privileged stage.
TASK_TEXT = {
    "ball_maze": "Lift and tilt the maze to move the ball to the target",
    "bin_sort": "Sort both cubes into their matching bowls",
    "block_balance": "Balance the beam and place both blocks on it",
    "carry_pot": "Carry the pot by both handles onto the stove",
    "hinge_chest": "Open the chest and place the cube inside",
    "join_blocks": "Join both blocks and attach them to the wall peg",
    "pour_marbles": "Pour the marbles into the other cup and replace both cups",
    "spring_door": "Open the door place the box inside and close it",
    "transfer_cube": "Hand over the cube and place it in the bowl",
    "transfer_gate": "Hand over the box through the gate and place it on the mat",
    "transfer_reorient": "Hand over reorient and insert the block in the socket",
}

HISTORY_STEPS = 16
ACTION_HORIZON = 100
# DuoBench's released rows are post-action observations.  Keep the current
# observation at row t and supervise the command issued for the next row.
# This is a versioned CARE-v2 contract; v1 remains untouched for provenance.
ACTION_LAG_ROWS = 1
STATE_DIM = 8
ACTION_DIM = 8
EFFECTIVE_BATCH = 48
BASE_SAMPLES_PER_TASK = 4
EXTRA_SAMPLES_PER_UPDATE = 4
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_IMAGE_WIDTH = 224


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DuoTemporalEpisode:
    task: str
    task_id: int
    episode_id: int
    start: int
    end: int
    length: int
    cache_key: str
    source_identity: str


@dataclass(frozen=True)
class DuoTemporalRequest:
    episode_list_index: int
    arm: int
    time_index: int
    sample_key: str
    task: str


def _load_task_arrays(root: Path, task: str) -> dict[str, np.ndarray]:
    task_root = root / task
    result: dict[str, np.ndarray] = {}
    for name in ("state", "action", "head", "left", "right", "episodes"):
        path = task_root / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = np.load(path, mmap_mode="r")
    state = result["state"]
    action = result["action"]
    episodes = result["episodes"]
    if state.ndim != 2 or state.shape[1] != 16 or action.shape != state.shape:
        raise ValueError(f"{task}: expected state/action [N,16], got {state.shape}/{action.shape}")
    if episodes.shape != (len(state),):
        raise ValueError(f"{task}: episode index shape differs: {episodes.shape}")
    for name in ("head", "left", "right"):
        value = result[name]
        if value.ndim != 4 or value.shape[0] != len(state) or value.shape[-1] != 3:
            raise ValueError(f"{task}/{name}: expected [N,H,W,3], got {value.shape}")
        if value.dtype != np.uint8:
            raise ValueError(f"{task}/{name}: expected uint8 RGB, got {value.dtype}")
    return result


def validate_prepared_manifest_contract(manifest: Mapping) -> dict:
    """Fail closed on formal data/preprocessing/horizon provenance."""

    if manifest.get("schema") != "duobench-act-prepared-v1":
        raise ValueError("unexpected Duo prepared-data schema")
    if manifest.get("dataset_revision") != FORMAL_DATASET_REVISION:
        raise ValueError("formal Duo dataset revision differs")
    # Preparation stores the successful RCS Pin-IK output separately and
    # exposes only the controller-equivalent target (pinned MuJoCo ctrlrange
    # saturation + binary gripper) to formal training.  Require the exact
    # serialized contract before a cache or trainer can consume the data.
    contract = manifest.get("action_target_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("formal Duo manifest lacks action-target contract")
    try:
        validate_action_target_contract(contract)
    except ValueError as error:
        raise ValueError(f"formal Duo action-target contract differs: {error}") from error
    if (
        contract.get("schema") != ACTION_TARGET_CONTRACT_SCHEMA
        or contract.get("id") != ACTION_TARGET_CONTRACT_ID
        or contract.get("sha256") != ACTION_TARGET_CONTRACT_SHA256
    ):
        raise ValueError("formal Duo action-target contract identity differs")
    audit = manifest.get("action_target_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("formal Duo manifest lacks action-target audit metadata")
    if (
        audit.get("schema") != "before-we-act.duobench.action-target-audit/1"
        or audit.get("status") != "PASSED"
        or audit.get("path") != "action_target_audit.json"
        or audit.get("contract_id") != ACTION_TARGET_CONTRACT_ID
        or audit.get("contract_sha256") != ACTION_TARGET_CONTRACT_SHA256
        or not isinstance(audit.get("sha256"), str)
        or len(audit["sha256"]) != 64
    ):
        raise ValueError("formal Duo action-target audit metadata differs")
    if int(manifest.get("image_size", -1)) != DEFAULT_IMAGE_HEIGHT:
        raise ValueError("formal Duo image size differs")
    preprocessing = manifest.get("image_preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ValueError("formal Duo manifest lacks image preprocessing metadata")
    expected_preprocessing = {
        "id": IMAGE_PREPROCESS_ID,
        "training_video_resolution": [224, 224],
        "training_decode_resize": "none_already_converter_resized",
        "runtime_source_resolution": [720, 1280],
        "runtime_resize": "torchvision_v2_uint8_bilinear_antialias_true",
        "views_resized_independently": True,
    }
    for key, expected in expected_preprocessing.items():
        if preprocessing.get(key) != expected:
            raise ValueError(f"formal Duo image preprocessing differs at {key}")
    horizon = manifest.get("validation_horizon")
    if not isinstance(horizon, Mapping):
        raise ValueError("formal Duo manifest lacks validation horizon metadata")
    if (
        horizon.get("method") != VALIDATION_HORIZON_METHOD
        or float(horizon.get("quantile", -1)) != VALIDATION_HORIZON_QUANTILE
        or horizon.get("per_task_max_steps") != VALIDATION_MAX_STEPS
    ):
        raise ValueError("formal Duo validation horizon contract differs")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, Mapping) or tuple(tasks) != TASKS:
        raise ValueError("formal Duo task order/coverage differs")
    for task in TASKS:
        row = tasks.get(task, {})
        if (
            int(row.get("episodes", -1)) != FORMAL_EPISODES_PER_TASK
            or int(row.get("validation_max_steps", -1)) != VALIDATION_MAX_STEPS[task]
            or row.get("validation_horizon_method") != VALIDATION_HORIZON_METHOD
            or float(row.get("validation_horizon_quantile", -1))
            != VALIDATION_HORIZON_QUANTILE
        ):
            raise ValueError(f"{task}: formal episode/horizon contract differs")
        task_audit = row.get("action_target_audit")
        if not isinstance(task_audit, Mapping):
            raise ValueError(f"{task}: missing action-target audit")
        # Per-task summaries generated by prepare.py carry the same contract
        # identity; an older unreceipted artifact is never formal input.
        if (
            task_audit.get("contract_sha256") != ACTION_TARGET_CONTRACT_SHA256
            or task_audit.get("contract_id") != ACTION_TARGET_CONTRACT_ID
            or task_audit.get("action_encoding")
            != "absolute_joint7_binary_gripper1"
        ):
            raise ValueError(f"{task}: action-target audit identity differs")
    return dict(manifest)


def _validate_action_target_receipt(root: Path, manifest: Mapping) -> None:
    """Verify the on-disk action receipt bound by a formal manifest.

    A manifest-only check is insufficient: an attacker (or an interrupted
    copy) could replace ``action_target_audit.json`` while retaining its old
    metadata.  Every formal data consumer therefore verifies the receipt
    bytes and its embedded contract before opening any arrays.
    """

    metadata = manifest.get("action_target_audit")
    if not isinstance(metadata, Mapping):
        raise ValueError("formal Duo manifest lacks action-target audit metadata")
    path_value = metadata.get("path")
    if path_value != "action_target_audit.json":
        raise ValueError("formal Duo action-target audit path differs")
    receipt_path = root / str(path_value)
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    if digest != metadata.get("sha256"):
        raise ValueError("formal Duo action-target audit receipt hash differs")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("formal Duo action-target audit receipt is invalid JSON") from error
    if not isinstance(receipt, Mapping):
        raise ValueError("formal Duo action-target audit receipt is not an object")
    if (
        receipt.get("schema") != "before-we-act.duobench.action-target-audit/1"
        or receipt.get("status") != "PASSED"
        or receipt.get("contract_id") != ACTION_TARGET_CONTRACT_ID
        or receipt.get("contract_sha256") != ACTION_TARGET_CONTRACT_SHA256
    ):
        raise ValueError("formal Duo action-target audit receipt identity differs")
    embedded = receipt.get("contract")
    if not isinstance(embedded, Mapping):
        raise ValueError("formal Duo action-target receipt lacks embedded contract")
    try:
        validate_action_target_contract(embedded)
    except ValueError as error:
        raise ValueError("formal Duo action-target receipt contract differs") from error
    tasks = receipt.get("tasks")
    if not isinstance(tasks, Mapping) or tuple(tasks) != TASKS:
        raise ValueError("formal Duo action-target receipt task coverage differs")


def load_manifest(root: str | Path, *, require_formal: bool = False) -> dict:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text())
    if require_formal:
        validate_prepared_manifest_contract(manifest)
        _validate_action_target_receipt(Path(root), manifest)
    normalization = manifest.get("normalization", {})
    if normalization.get("action_encoding") != "absolute_joint7_binary_gripper1":
        raise ValueError(
            "Duo DINO reference requires un-clipped absolute joint7 + binary "
            "gripper1 targets; rebuild data with deployment.duo_act.prepare"
        )
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
        value = np.asarray(normalization.get(key), dtype=np.float32)
        if value.shape != (8,) or not np.isfinite(value).all():
            raise ValueError(f"invalid Duo normalization {key}: {value.shape}")
        if key.endswith("std") and np.any(value <= 0):
            raise ValueError(f"non-positive Duo normalization {key}")
    if require_formal and (
        normalization.get("action_target_contract_id") != ACTION_TARGET_CONTRACT_ID
        or normalization.get("action_target_contract_sha256")
        != ACTION_TARGET_CONTRACT_SHA256
    ):
        raise ValueError("formal Duo normalization is not tied to action-target contract")
    return manifest


def load_duo_episodes(
    root: str | Path,
    *,
    require_formal: bool = True,
) -> list[DuoTemporalEpisode]:
    """Reconstruct episode boundaries without allowing cross-episode history."""

    prepared = Path(root).resolve(strict=True)
    manifest = load_manifest(prepared, require_formal=require_formal)
    records: list[DuoTemporalEpisode] = []
    counts: Counter[str] = Counter()
    revision = str(manifest.get("dataset_revision", "unknown"))
    for task_id, task in enumerate(TASKS):
        arrays = _load_task_arrays(prepared, task)
        episode_ids = np.asarray(arrays["episodes"])
        if len(episode_ids) == 0:
            raise ValueError(f"{task}: empty corpus")
        starts = np.r_[0, np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1]
        ends = np.r_[starts[1:], len(episode_ids)]
        seen: set[int] = set()
        for start, end in zip(starts, ends, strict=True):
            episode_id = int(episode_ids[start])
            if episode_id in seen:
                raise ValueError(f"{task}: non-contiguous episode id {episode_id}")
            seen.add(episode_id)
            length = int(end - start)
            identity = {
                "dataset_revision": revision,
                "task": task,
                "episode_id": episode_id,
                "start": int(start),
                "end": int(end),
            }
            digest = _canonical_hash(identity)
            records.append(
                DuoTemporalEpisode(
                    task=task,
                    task_id=task_id,
                    episode_id=episode_id,
                    start=int(start),
                    end=int(end),
                    length=length,
                    cache_key=digest,
                    source_identity=digest,
                )
            )
            counts[task] += 1
    if require_formal:
        expected = Counter({task: FORMAL_EPISODES_PER_TASK for task in TASKS})
        if counts != expected or len(records) != 550:
            raise ValueError(f"formal Duo corpus must be 50 episodes/task: {counts}")
    return records


class DuoVisualCache:
    """Small LRU over per-episode head/left/right frozen DINO pools."""

    KEYS = ("view_head", "view_wrist_0", "view_wrist_1")

    def __init__(self, root: str | Path, limit: int = 32):
        self.root = Path(root)
        self.limit = int(limit)
        self.values: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def path_for(self, episode: DuoTemporalEpisode) -> Path:
        return self.root / episode.task / f"{episode.cache_key}.npz"

    def load(self, episode: DuoTemporalEpisode) -> dict[str, np.ndarray]:
        if episode.cache_key in self.values:
            self.values.move_to_end(episode.cache_key)
            return self.values[episode.cache_key]
        path = self.path_for(episode)
        if not path.is_file():
            raise FileNotFoundError(f"missing Duo DINO cache: {path}")
        with np.load(path, allow_pickle=False) as source:
            identity = str(source["source_identity"].item())
            if identity != episode.source_identity:
                raise ValueError(f"Duo DINO cache source drift: {path}")
            result = {key: np.asarray(source[key]) for key in self.KEYS}
        for key, value in result.items():
            if value.shape != (episode.length, 768) or value.dtype != np.float16:
                raise ValueError(f"invalid cache {path}/{key}: {value.shape}/{value.dtype}")
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite cache {path}/{key}")
        self.values[episode.cache_key] = result
        while self.limit > 0 and len(self.values) > self.limit:
            self.values.popitem(last=False)
        return result


class DuoTemporalDataset(Dataset):
    """Legal local history and H=100 absolute target for one Duo arm."""

    MODEL_INPUT_FIELDS = frozenset(
        {
            "global_rgb",
            "local_rgb",
            "history_visual_raw",
            "history_qpos",
            "history_action",
            "history_mask",
            "action_history_mask",
            "task_bytes",
            "task_text_mask",
            "episode_reset",
        }
    )
    TARGET_FIELDS = frozenset({"action", "action_mask"})

    def __init__(
        self,
        prepared_root: str | Path,
        episodes: Sequence[DuoTemporalEpisode],
        visual_cache_root: str | Path,
        *,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        cache_limit: int = 32,
    ) -> None:
        self.root = Path(prepared_root)
        self.episodes = list(episodes)
        manifest = load_manifest(self.root)
        norm = manifest["normalization"]
        self.q_mean = torch.tensor(norm["qpos_mean"], dtype=torch.float32)
        self.q_std = torch.tensor(norm["qpos_std"], dtype=torch.float32)
        self.a_mean = torch.tensor(norm["action_mean"], dtype=torch.float32)
        self.a_std = torch.tensor(norm["action_std"], dtype=torch.float32)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        # Validate the rectangular DINO grid before workers are spawned.
        resize_rgb_batch(
            torch.zeros(1, 16, 16, 3, dtype=torch.uint8),
            self.image_height,
            self.image_width,
        )
        self.visual_cache = DuoVisualCache(visual_cache_root, cache_limit)
        self.task_data = {
            task: _load_task_arrays(self.root, task)
            for task in TASKS
            if any(episode.task == task for episode in self.episodes)
        }

    def __len__(self) -> int:
        return sum(episode.length * 2 for episode in self.episodes)

    def __getitem__(self, request: DuoTemporalRequest | tuple) -> dict:
        if not isinstance(request, DuoTemporalRequest):
            request = DuoTemporalRequest(*request)
        episode = self.episodes[request.episode_list_index]
        if request.task != episode.task or request.arm not in (0, 1):
            raise ValueError(f"sample identity mismatch: {request}")
        decision_length = episode.length - ACTION_LAG_ROWS
        if decision_length < 1:
            raise ValueError("Duo episode has no legal lag-one decision")
        if not 0 <= request.time_index < decision_length:
            raise IndexError(request.time_index)
        arm = request.arm
        local_t = request.time_index
        absolute_t = episode.start + local_t
        arrays = self.task_data[episode.task]
        state = arrays["state"].reshape(-1, 2, 8)
        action = arrays["action"].reshape(-1, 2, 8)
        cache = self.visual_cache.load(episode)

        observation_first = max(0, local_t - HISTORY_STEPS + 1)
        observation_indices = np.arange(observation_first, local_t + 1)
        observation_offset = HISTORY_STEPS - len(observation_indices)
        # At decision t the newest causally available executed command is row
        # t (the command whose post-action observation is row t).  Row zero is
        # omitted so reset-time deployment, which has no policy action yet,
        # matches training.
        action_first = max(1, local_t - HISTORY_STEPS + 1)
        action_indices = np.arange(action_first, local_t + 1)
        action_offset = HISTORY_STEPS - len(action_indices)

        history_visual = torch.zeros(HISTORY_STEPS, 2, 768, dtype=torch.float16)
        history_qpos = torch.zeros(HISTORY_STEPS, STATE_DIM)
        history_action = torch.zeros(HISTORY_STEPS, ACTION_DIM)
        history_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        action_history_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        wrist_key = f"view_wrist_{arm}"
        history_visual[observation_offset:, 0] = torch.from_numpy(
            cache["view_head"][observation_indices]
        )
        history_visual[observation_offset:, 1] = torch.from_numpy(
            cache[wrist_key][observation_indices]
        )
        qpos = np.array(
            state[episode.start + observation_indices, arm], dtype=np.float32, copy=True
        )
        history_qpos[observation_offset:] = (
            torch.from_numpy(qpos) - self.q_mean
        ) / self.q_std
        history_mask[observation_offset:] = True
        if len(action_indices):
            past = np.array(
                action[episode.start + action_indices, arm], dtype=np.float32, copy=True
            )
            history_action[action_offset:] = (
                torch.from_numpy(past) - self.a_mean
            ) / self.a_std
            action_history_mask[action_offset:] = True

        target_start = absolute_t + ACTION_LAG_ROWS
        future_end = min(episode.end, target_start + ACTION_HORIZON)
        future = np.array(
            action[target_start:future_end, arm], dtype=np.float32, copy=True
        )
        valid = len(future)
        normalized = (torch.from_numpy(future) - self.a_mean) / self.a_std
        target = torch.empty(ACTION_HORIZON, ACTION_DIM)
        target[:valid] = normalized
        target[valid:] = normalized[-1]
        action_mask = torch.zeros(ACTION_HORIZON, dtype=torch.bool)
        action_mask[:valid] = True

        head = np.array(arrays["head"][absolute_t], dtype=np.uint8, copy=True)
        wrist_name = "left" if arm == 0 else "right"
        wrist = np.array(arrays[wrist_name][absolute_t], dtype=np.uint8, copy=True)
        global_rgb = resize_rgb_batch(head, self.image_height, self.image_width)
        local_rgb = resize_rgb_batch(wrist, self.image_height, self.image_width)
        task_bytes, task_text_mask = task_text_tensor(TASK_TEXT[episode.task])
        return {
            "global_rgb": global_rgb,
            "local_rgb": local_rgb,
            "history_visual_raw": history_visual,
            "history_qpos": history_qpos,
            "history_action": history_action,
            "history_mask": history_mask,
            "action_history_mask": action_history_mask,
            "task_bytes": task_bytes,
            "task_text_mask": task_text_mask,
            "episode_reset": torch.tensor(local_t == 0, dtype=torch.bool),
            "action": target,
            "action_mask": action_mask,
            "task": episode.task,
            "episode_id": torch.tensor(episode.episode_id),
            "arm": torch.tensor(arm),
            "time_index": torch.tensor(local_t),
            "sample_key": request.sample_key,
        }


class DuoBalancedDistributedBatchSampler(Sampler[list[DuoTemporalRequest]]):
    """Matched-compute batch48 with exact task balance over 11 updates.

    Each update starts with four rows/task.  Four extras rotate through task
    IDs, so every task receives exactly four extras over any aligned 11-update
    cycle.  Four-GPU formal DDP therefore has 12 rows/rank while retaining
    exact long-run task equality.
    """

    def __init__(
        self,
        episodes: Sequence[DuoTemporalEpisode],
        *,
        updates: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        start_update: int = 0,
    ) -> None:
        if updates < 1 or not 0 <= start_update <= updates:
            raise ValueError("invalid Duo update interval")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed rank")
        if EFFECTIVE_BATCH % world_size:
            raise ValueError(f"world size must divide effective batch {EFFECTIVE_BATCH}")
        self.episodes = list(episodes)
        self.updates = int(updates)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_update = int(start_update)
        self.by_task: dict[str, list[int]] = defaultdict(list)
        for index, episode in enumerate(self.episodes):
            self.by_task[episode.task].append(index)
        if set(self.by_task) != set(TASKS):
            raise ValueError(f"expected all 11 task buckets, got {sorted(self.by_task)}")

    def __len__(self) -> int:
        return self.updates - self.start_update

    def requests_for_update(self, update: int) -> list[DuoTemporalRequest]:
        if not 1 <= update <= self.updates:
            raise IndexError(update)
        rng = random.Random(self.seed + 1_000_003 * update)
        rows: list[DuoTemporalRequest] = []
        extra_tasks = {
            TASKS[((update - 1) * EXTRA_SAMPLES_PER_UPDATE + offset) % len(TASKS)]
            for offset in range(EXTRA_SAMPLES_PER_UPDATE)
        }
        for task in TASKS:
            for _ in range(BASE_SAMPLES_PER_TASK + int(task in extra_tasks)):
                episode_index = rng.choice(self.by_task[task])
                episode = self.episodes[episode_index]
                arm = rng.randrange(2)
                time_index = rng.randrange(episode.length - ACTION_LAG_ROWS)
                identity = (
                    f"{episode.source_identity}:{episode.episode_id}:"
                    f"{arm}:{time_index}"
                )
                rows.append(
                    DuoTemporalRequest(
                        episode_index,
                        arm,
                        time_index,
                        hashlib.sha256(identity.encode()).hexdigest(),
                        task,
                    )
                )
        rng.shuffle(rows)
        counts = Counter(row.task for row in rows)
        expected = Counter(
            {
                task: BASE_SAMPLES_PER_TASK + int(task in extra_tasks)
                for task in TASKS
            }
        )
        if len(rows) != EFFECTIVE_BATCH or counts != expected:
            raise AssertionError(f"Duo batch balance failure: {counts}")
        return rows

    def __iter__(self) -> Iterator[list[DuoTemporalRequest]]:
        for update in range(self.start_update + 1, self.updates + 1):
            yield self.requests_for_update(update)[self.rank :: self.world_size]

    def cursor_receipt(self, completed_update: int) -> dict:
        if not 0 <= completed_update <= self.updates:
            raise ValueError(completed_update)
        next_update = completed_update + 1
        keys = (
            [row.sample_key for row in self.requests_for_update(next_update)]
            if next_update <= self.updates
            else []
        )
        return {
            "format_version": "before-we-act.duobench.b0h-cursor/1",
            "seed": self.seed,
            "completed_update": completed_update,
            "next_update": next_update if keys else None,
            "next_sample_keys": keys,
            "effective_batch": EFFECTIVE_BATCH,
            "base_samples_per_task": BASE_SAMPLES_PER_TASK,
            "extra_samples_per_update": EXTRA_SAMPLES_PER_UPDATE,
            "balance_cycle_updates": len(TASKS),
            "extra_task_ids_next_update": (
                [TASKS.index(task) for task in TASKS if task in {
                    TASKS[((next_update - 1) * EXTRA_SAMPLES_PER_UPDATE + offset) % len(TASKS)]
                    for offset in range(EXTRA_SAMPLES_PER_UPDATE)
                }]
                if keys
                else []
            ),
        }

    def validate_cursor(self, receipt: Mapping) -> int:
        completed = int(receipt["completed_update"])
        expected = self.cursor_receipt(completed)
        if dict(receipt) != expected:
            raise ValueError("Duo B0-H resume sample cursor drifted")
        return completed


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "DEFAULT_IMAGE_HEIGHT",
    "DEFAULT_IMAGE_WIDTH",
    "EFFECTIVE_BATCH",
    "HISTORY_STEPS",
    "BASE_SAMPLES_PER_TASK",
    "EXTRA_SAMPLES_PER_UPDATE",
    "STATE_DIM",
    "TASKS",
    "TASK_TEXT",
    "DuoBalancedDistributedBatchSampler",
    "DuoTemporalDataset",
    "DuoTemporalEpisode",
    "DuoTemporalRequest",
    "DuoVisualCache",
    "load_duo_episodes",
    "load_manifest",
    "resize_rgb_batch",
    "validate_prepared_manifest_contract",
]
