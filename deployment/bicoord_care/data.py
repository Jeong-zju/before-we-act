"""BiCoord CARE data projection.

This adapter intentionally keeps the source benchmark untouched.  A raw
BiCoord HDF5 row contains both arms, but every training/runtime sample is a
single-arm stream: shared head RGB, the focal wrist RGB, focal seven
dimensional proprioception, and focal executed-action history.  The other arm
is never copied into model input fields.  Future target actions are the only
teacher-side values and are returned under the target key.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from before_we_act.temporal_history_data import task_text_tensor

from .config import (
    ACTION_DIM,
    ACTION_ENCODING,
    ACTION_HORIZON,
    ARM_COUNT,
    BALANCE_CYCLE_UPDATES,
    BASE_SAMPLES_PER_TASK,
    DATASET_REVISION,
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
    VALIDATION_MAX_STEPS,
    canonical_json_hash,
    GRIPPER_ENCODING,
    GRIPPER_NATIVE_RANGE,
    validate_native_gripper_vector,
)
from .hdf5_data import (
    BiCoordHDF5Reader,
    discover_all_episode_files,
    episode_number,
    sha256_file,
    validate_hdf5_schema,
)
from .preprocessing import decode_bicoord_jpeg_rgb, resize_rgb_batch


NORMALIZATION_SCHEMA = "before-we-act.bicoord.normalization/1"
DINO_CACHE_SCHEMA = "before-we-act.bicoord.dino-cache/1"
PREPARED_SCHEMA = "before-we-act.bicoord.prepared/1"
DATA_CURSOR_SCHEMA = "before-we-act.bicoord.b0h-cursor/1"

# These names are used by stage/audit code to detect accidental leakage when a
# caller forwards a raw environment dictionary to the policy.  The projector
# below intentionally copies only legal local values.
LEGAL_RUNTIME_FIELDS = frozenset(
    {"head_rgb", "wrist_rgb", "state", "action_history", "task_text", "reset"}
)
FORBIDDEN_RUNTIME_FIELDS = frozenset(
    {
        "peer_rgb", "peer_wrist_rgb", "peer_state", "peer_qpos", "peer_action",
        "all_qpos", "all_actions", "joint_action_vector", "stage", "stage_text",
        "task_id", "sim_state", "privileged_state", "future_action",
    }
)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _normalization_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = value.get("normalization")
    return nested if isinstance(nested, Mapping) else value


def project_local_observation(observation: Mapping[str, Any], arm: int) -> dict[str, Any]:
    """Project a raw BiCoord observation to one legal decentralized stream.

    The function is deliberately strict about required values and refuses
    dictionaries containing obvious peer/privileged channels.  It is intended
    for rollout adapters, while :class:`BiCoordTemporalDataset` performs the
    equivalent projection directly from HDF5.
    """

    arm = int(arm)
    if arm not in (0, 1):
        raise ValueError("BiCoord arm must be 0 or 1")
    if not isinstance(observation, Mapping):
        raise TypeError("BiCoord observation must be a mapping")
    # Native simulator observation.  This is a trusted boundary which selects
    # one arm before constructing a model dictionary; neither the returned
    # mapping nor any model tensor contains the peer values.
    cameras = observation.get("observation")
    joint_action = observation.get("joint_action")
    if isinstance(cameras, Mapping) and isinstance(joint_action, Mapping):
        side = "left" if arm == 0 else "right"
        try:
            head = cameras["head_camera"]["rgb"]
            wrist = cameras[f"{side}_camera"]["rgb"]
            joints = np.asarray(joint_action[f"{side}_arm"], dtype=np.float32).reshape(-1)
            gripper = np.asarray(joint_action[f"{side}_gripper"], dtype=np.float32).reshape(-1)
        except (KeyError, TypeError) as error:
            raise KeyError("native BiCoord observation is missing local fields") from error
        if joints.shape != (6,) or gripper.size != 1:
            raise ValueError("native BiCoord local state must be six joints plus one gripper")
        state = np.concatenate((joints, gripper[:1]))
    else:
        lowered = {str(key).lower() for key in observation}
        leaked = [marker for marker in FORBIDDEN_RUNTIME_FIELDS if marker in lowered]
        if leaked:
            raise ValueError(f"forbidden peer/privileged observation fields: {sorted(leaked)}")
        head = observation.get("head_rgb", observation.get("head_camera"))
        wrist_key = "left_rgb" if arm == 0 else "right_rgb"
        wrist = observation.get(wrist_key, observation.get("wrist_rgb"))
        state = observation.get("state")
    if head is None or wrist is None or state is None:
        raise KeyError("local BiCoord observation requires head, own wrist, and state")
    state_array = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_array.shape != (STATE_DIM,) or not np.isfinite(state_array).all():
        raise ValueError("local BiCoord state must be finite shape [7]")
    validate_native_gripper_vector(state_array, context="local BiCoord state")
    action_history = observation.get("action_history", ())
    history_array = np.asarray(action_history, dtype=np.float32)
    if history_array.size and (history_array.ndim != 2 or history_array.shape[-1] != ACTION_DIM):
        raise ValueError("local BiCoord action history must have trailing dimension 7")
    if history_array.size:
        for row in history_array.reshape(-1, ACTION_DIM):
            validate_native_gripper_vector(row, context="local BiCoord action history")
    return {
        "head_rgb": head,
        "wrist_rgb": wrist,
        "state": state_array,
        "action_history": history_array.reshape(-1, ACTION_DIM),
        "task_text": str(observation.get("task_text", "")),
        "reset": bool(observation.get("reset", False)),
    }


def validate_local_sample(sample: Mapping[str, Any], *, arm: int | None = None) -> None:
    """Audit a model sample for strict decentralized input provenance."""

    keys = {str(key) for key in sample}
    forbidden = sorted(key for key in keys if key.lower() in FORBIDDEN_RUNTIME_FIELDS)
    if forbidden:
        raise ValueError(f"sample contains forbidden peer/privileged fields: {forbidden}")
    required = set(BiCoordTemporalDataset.MODEL_INPUT_FIELDS) if "BiCoordTemporalDataset" in globals() else set()
    missing = required - keys
    if missing:
        raise ValueError(f"sample is missing model input fields: {sorted(missing)}")
    if arm is not None and "arm" in sample and int(np.asarray(sample["arm"]).item()) != int(arm):
        raise ValueError("sample arm identity differs from requested local arm")


def _episode_sidecar(path: Path, name: str, episode_id: int) -> Path | None:
    """Locate an instruction/stage sidecar next to ``data/episodeN.hdf5``."""

    candidates = [
        path.parent.parent / name / f"episode{episode_id}.json",
        path.parent.parent / name / f"episode_{episode_id}.json",
        path.parent / name / f"episode{episode_id}.json",
        path.parent / name / f"episode_{episode_id}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


@dataclass(frozen=True)
class BiCoordEpisode:
    """One paired demonstration and its immutable source identity."""

    path: str
    task: str
    task_text: str
    episode_id: int
    length: int
    hdf5_sha256: str
    stage_path: str | None = None
    instruction_path: str | None = None
    seed: int | None = None

    @property
    def task_id(self) -> int:
        return TASKS.index(self.task)

    @property
    def source_identity(self) -> str:
        return self.hdf5_sha256

    @property
    def relative_path(self) -> str:
        return Path(self.path).name

    @property
    def arms(self) -> tuple[int, int]:
        return (0, 1)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"task_id": self.task_id, "source_identity": self.source_identity}


@dataclass(frozen=True)
class BiCoordTemporalRequest:
    episode_index: int
    arm: int
    time_index: int
    sample_key: str
    task: str

    # Names used by a few generic loaders in the CARE scaffold.
    @property
    def episode_list_index(self) -> int:
        return self.episode_index


def _validate_episode_record(record: BiCoordEpisode) -> None:
    if record.task not in TASKS:
        raise ValueError(f"unsupported BiCoord task: {record.task}")
    if record.length < 2:
        raise ValueError(f"BiCoord episode must contain at least two rows: {record.path}")
    if len(record.hdf5_sha256) != 64:
        raise ValueError(f"invalid HDF5 SHA-256 for {record.path}")
    if record.task_text != TASK_TEXT[record.task]:
        raise ValueError(f"canonical task text drift for {record.task}")


def discover_bicoord_episodes(
    root: str | Path,
    *,
    require_formal: bool = True,
    verify_schema: bool = False,
    source_hashes: Mapping[str, str] | None = None,
) -> list[BiCoordEpisode]:
    """Discover deterministic task/episode records from a raw HF snapshot.

    ``require_formal=False`` is useful for smoke tests and fixture datasets;
    formal callers fail closed unless all 100 episodes of all 18 tasks are
    present.  Hashes are computed from source bytes and can optionally be
    checked against a previously recorded manifest.
    """

    files_by_task = discover_all_episode_files(root, require_complete=require_formal)
    records: list[BiCoordEpisode] = []
    counts: Counter[str] = Counter()
    for task in TASKS:
        for path in files_by_task[task]:
            ep_id = episode_number(path)
            digest = sha256_file(path)
            if source_hashes is not None:
                expected = source_hashes.get(str(path)) or source_hashes.get(path.name)
                if expected is not None and str(expected) != digest:
                    raise ValueError(f"BiCoord source hash drift: {path}")
            if verify_schema:
                metadata = validate_hdf5_schema(path, check_images=False)
                length = int(metadata["length"])
            else:
                # HDF5 metadata access is cheap and avoids materializing any
                # image payload.  The full schema can be requested above.
                reader = BiCoordHDF5Reader(path)
                length = reader.length
            stage = _episode_sidecar(path, "stages", ep_id)
            instruction = _episode_sidecar(path, "instructions", ep_id)
            record = BiCoordEpisode(
                path=str(path.resolve()),
                task=task,
                task_text=TASK_TEXT[task],
                episode_id=ep_id,
                length=int(length),
                hdf5_sha256=digest,
                stage_path=str(stage) if stage else None,
                instruction_path=str(instruction) if instruction else None,
            )
            _validate_episode_record(record)
            records.append(record)
            counts[task] += 1
    if require_formal:
        expected = Counter({task: EPISODES_PER_TASK for task in TASKS})
        if counts != expected or len(records) != TOTAL_EPISODES:
            raise ValueError(f"formal BiCoord corpus must contain 100 episodes/task: {counts}")
    # Discovery order is part of the receipt and must not depend on filesystem
    # traversal order or HDF5 hash order.
    records.sort(key=lambda item: (TASKS.index(item.task), item.episode_id, item.path))
    return records


# Common aliases used by preparation/supervisor code.
load_bicoord_episodes = discover_bicoord_episodes


def episode_manifest(episodes: Sequence[BiCoordEpisode], *, root: str | Path | None = None) -> dict[str, Any]:
    rows = [item.as_dict() for item in episodes]
    counts = Counter(item.task for item in episodes)
    return {
        "schema": PREPARED_SCHEMA,
        "dataset_revision": DATASET_REVISION,
        "dataset_repo_id": "GradiusTwinbee/BiCoord",
        "root": str(Path(root).resolve()) if root is not None else None,
        "tasks": list(TASKS),
        "task_text": dict(TASK_TEXT),
        "episodes": len(episodes),
        "episodes_per_task": {task: int(counts.get(task, 0)) for task in TASKS},
        "records_sha256": canonical_json_hash(rows),
        "records": rows,
    }


def load_normalization_receipt(
    path_or_root: str | Path | Mapping[str, Any],
    *,
    require_formal: bool = True,
) -> dict[str, Any]:
    """Read and validate the shared state/action normalization receipt."""

    if isinstance(path_or_root, Mapping):
        receipt = dict(path_or_root)
        source_path: Path | None = None
    else:
        source = Path(path_or_root)
        if source.is_dir():
            candidates = (source / "normalization.json", source / "normalization_receipt.json", source / "manifest.json")
            source_path = next((item for item in candidates if item.is_file()), None)
            if source_path is None:
                raise FileNotFoundError(f"no normalization receipt under {source}")
        else:
            source_path = source
        try:
            value = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid BiCoord normalization JSON: {source_path}") from error
        if not isinstance(value, Mapping):
            raise ValueError("normalization receipt must be a JSON object")
        receipt = dict(value)
        # Prepared manifests nest normalization under this key.  Preserve the
        # outer provenance fields while exposing a flat receipt to callers.
        nested = receipt.get("normalization")
        if isinstance(nested, Mapping):
            receipt = {**receipt, **dict(nested)}

    required = ("qpos_mean", "qpos_std", "action_mean", "action_std")
    for key in required:
        array = np.asarray(receipt.get(key), dtype=np.float64)
        if array.shape != (STATE_DIM,) or not np.isfinite(array).all():
            raise ValueError(f"BiCoord normalization {key} must be finite shape [{STATE_DIM}]")
        if key.endswith("std") and np.any(array <= 0):
            raise ValueError(f"BiCoord normalization {key} must be strictly positive")
        receipt[key] = array.astype(np.float32).tolist()
    if receipt.get("state_dim", STATE_DIM) != STATE_DIM or receipt.get("action_dim", ACTION_DIM) != ACTION_DIM:
        raise ValueError("BiCoord normalization dimensions differ from native 7D contract")
    if receipt.get("action_encoding", ACTION_ENCODING) != ACTION_ENCODING:
        raise ValueError(f"BiCoord action encoding differs: {receipt.get('action_encoding')!r}")
    if receipt.get("state_encoding", ACTION_ENCODING) != ACTION_ENCODING:
        raise ValueError(f"BiCoord state encoding differs: {receipt.get('state_encoding')!r}")
    if receipt.get("gripper_encoding", GRIPPER_ENCODING) != GRIPPER_ENCODING:
        raise ValueError(f"BiCoord gripper encoding differs: {receipt.get('gripper_encoding')!r}")
    try:
        native_range = tuple(
            float(value)
            for value in receipt.get("gripper_native_range", GRIPPER_NATIVE_RANGE)
        )
    except (TypeError, ValueError):
        native_range = ()
    if native_range != GRIPPER_NATIVE_RANGE:
        raise ValueError(f"BiCoord gripper native range differs: {native_range!r}")
    for key, expected in (
        ("gripper_thresholding", False),
        ("gripper_reparameterization", False),
    ):
        if receipt.get(key, expected) is not expected:
            raise ValueError(f"BiCoord normalization {key} is enabled")
    if require_formal:
        if receipt.get("schema") != NORMALIZATION_SCHEMA:
            raise ValueError("formal BiCoord normalization receipt schema differs")
        if receipt.get("status") != "PASSED":
            raise ValueError("formal BiCoord normalization receipt is not PASSED")
        if int(receipt.get("episodes", receipt.get("total_episodes", -1))) != TOTAL_EPISODES:
            raise ValueError("formal BiCoord normalization does not cover all 1800 episodes")
        if tuple(receipt.get("tasks", TASKS)) != TASKS:
            raise ValueError("formal BiCoord normalization task order differs")
        if receipt.get("dataset_revision") not in (None, DATASET_REVISION):
            raise ValueError("BiCoord normalization dataset revision differs")
    if source_path is not None:
        receipt.setdefault("receipt_path", str(source_path.resolve()))
        receipt.setdefault("receipt_sha256", sha256_file(source_path))
    return receipt


def compute_normalization(
    episodes: Sequence[BiCoordEpisode],
    *,
    status: str | None = None,
) -> dict[str, Any]:
    """Compute causal statistics over all episodes without loading images."""

    if not episodes:
        raise ValueError("cannot normalize an empty BiCoord corpus")
    counts = Counter(item.task for item in episodes)
    formal_coverage = (
        len(episodes) == TOTAL_EPISODES
        and counts == Counter({task: EPISODES_PER_TASK for task in TASKS})
    )
    if status is None:
        status = "PASSED" if formal_coverage else "SMOKE"
    if status == "PASSED" and not formal_coverage:
        raise ValueError("PASSED normalization requires all 1800 BiCoord episodes")
    if status not in ("PASSED", "SMOKE"):
        raise ValueError(f"invalid normalization status: {status}")
    sums_q = np.zeros(STATE_DIM, dtype=np.float64)
    sums_a = np.zeros(ACTION_DIM, dtype=np.float64)
    sums_q2 = np.zeros(STATE_DIM, dtype=np.float64)
    sums_a2 = np.zeros(ACTION_DIM, dtype=np.float64)
    mins_q = np.full(STATE_DIM, np.inf)
    mins_a = np.full(ACTION_DIM, np.inf)
    maxs_q = np.full(STATE_DIM, -np.inf)
    maxs_a = np.full(ACTION_DIM, -np.inf)
    rows = 0
    for episode in episodes:
        reader = BiCoordHDF5Reader(episode.path, episode_id=episode.episode_id)
        with reader._open() as handle:  # one bounded handle per episode
            left = np.asarray(handle["joint_action/left_arm"], dtype=np.float64)
            right = np.asarray(handle["joint_action/right_arm"], dtype=np.float64)
            lg = np.asarray(handle["joint_action/left_gripper"], dtype=np.float64)[:, None]
            rg = np.asarray(handle["joint_action/right_gripper"], dtype=np.float64)[:, None]
        state = np.stack((np.concatenate((left[:-1], lg[:-1]), axis=1), np.concatenate((right[:-1], rg[:-1]), axis=1)))
        action = np.stack((np.concatenate((left[1:], lg[1:]), axis=1), np.concatenate((right[1:], rg[1:]), axis=1)))
        q = state.reshape(-1, STATE_DIM)
        a = action.reshape(-1, ACTION_DIM)
        sums_q += q.sum(0); sums_a += a.sum(0)
        sums_q2 += np.square(q).sum(0); sums_a2 += np.square(a).sum(0)
        mins_q = np.minimum(mins_q, q.min(0)); mins_a = np.minimum(mins_a, a.min(0))
        maxs_q = np.maximum(maxs_q, q.max(0)); maxs_a = np.maximum(maxs_a, a.max(0))
        rows += len(q)
    qmean = sums_q / rows; amean = sums_a / rows
    qstd = np.maximum(np.sqrt(np.maximum(sums_q2 / rows - qmean * qmean, 0.0)), 1e-4)
    astd = np.maximum(np.sqrt(np.maximum(sums_a2 / rows - amean * amean, 0.0)), 1e-4)
    receipt: dict[str, Any] = {
        "schema": NORMALIZATION_SCHEMA,
        "status": status,
        "dataset_repo_id": "GradiusTwinbee/BiCoord",
        "dataset_revision": DATASET_REVISION,
        "tasks": list(TASKS),
        "episodes": len(episodes),
        "episodes_per_task": {task: int(counts.get(task, 0)) for task in TASKS},
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_encoding": ACTION_ENCODING,
        "action_encoding": ACTION_ENCODING,
        # The seventh channel is a native continuous simulator drive target.
        # Keep these fields explicit in every receipt so a downstream loader
        # cannot silently reinterpret historical values as binary labels.
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
        "recording_alignment": {
            "source_row_semantics": "joint_action_row_is_observation_state",
            "policy_decision_pair": "observation_row_i_to_action_row_i_plus_1",
            "observation_row_offset": 0,
            "action_row_offset": 1,
            "action_lag_rows": 1,
        },
        "qpos_mean": qmean.astype(np.float32).tolist(),
        "qpos_std": qstd.astype(np.float32).tolist(),
        "qpos_min": mins_q.astype(np.float32).tolist(),
        "qpos_max": maxs_q.astype(np.float32).tolist(),
        "action_mean": amean.astype(np.float32).tolist(),
        "action_std": astd.astype(np.float32).tolist(),
        "action_min": mins_a.astype(np.float32).tolist(),
        "action_max": maxs_a.astype(np.float32).tolist(),
        "rows_per_arm": rows // ARM_COUNT,
        "all_demonstrations": True,
    }
    receipt["receipt_payload_sha256"] = _sha256_json({k: v for k, v in receipt.items() if k != "receipt_payload_sha256"})
    return receipt


def write_normalization_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    """Atomically write a validated normalization receipt."""

    value = load_normalization_receipt(receipt, require_formal=receipt.get("status") == "PASSED")
    value.pop("receipt_path", None); value.pop("receipt_sha256", None)
    expected = value.get("receipt_payload_sha256")
    payload = {key: item for key, item in value.items() if key != "receipt_payload_sha256"}
    actual = _sha256_json(payload)
    if expected is not None and expected != actual:
        raise ValueError("normalization payload hash differs")
    value["receipt_payload_sha256"] = actual
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


class BiCoordVisualCache:
    """Bounded LRU for per-episode frozen DINO features."""

    KEYS = ("view_head", "view_wrist_0", "view_wrist_1")

    def __init__(self, root: str | Path, *, limit: int = 32, require_receipt: bool = False) -> None:
        self.root = Path(root).expanduser().resolve()
        self.limit = max(0, int(limit))
        self.values: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        receipt_path = self.root / "cache_receipt.json"
        self.receipt: dict[str, Any] | None = None
        if receipt_path.is_file():
            try:
                self.receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid DINO cache receipt: {receipt_path}") from error
        elif require_receipt:
            raise FileNotFoundError(receipt_path)

    def path_for(self, episode: BiCoordEpisode) -> Path:
        candidates = (
            self.root / episode.task / f"{episode.hdf5_sha256}.npz",
            self.root / episode.task / f"episode{episode.episode_id}.npz",
            self.root / episode.task / f"episode_{episode.episode_id}.npz",
            self.root / f"{episode.hdf5_sha256}.npz",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        # Return canonical hash path for a useful error message.
        return candidates[0]

    def load(self, episode: BiCoordEpisode) -> dict[str, np.ndarray]:
        if episode.source_identity in self.values:
            self.values.move_to_end(episode.source_identity)
            return self.values[episode.source_identity]
        path = self.path_for(episode)
        if not path.is_file():
            raise FileNotFoundError(f"missing BiCoord DINO cache: {path}")
        with np.load(path, allow_pickle=False) as source:
            identity_name = "source_identity" if "source_identity" in source.files else "source_hdf5_sha256"
            if identity_name in source.files:
                identity = str(np.asarray(source[identity_name]).item())
                if identity != episode.source_identity:
                    raise ValueError(f"BiCoord DINO cache source drift: {path}")
            result = {key: np.asarray(source[key]) for key in self.KEYS if key in source.files}
        if set(result) != set(self.KEYS):
            raise ValueError(f"BiCoord DINO cache views differ: {path}")
        for key, value in result.items():
            if value.shape != (episode.length, 768) or value.dtype != np.float16:
                raise ValueError(f"invalid BiCoord cache {path}/{key}: {value.shape}/{value.dtype}")
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite BiCoord cache values: {path}/{key}")
        self.values[episode.source_identity] = result
        while self.limit and len(self.values) > self.limit:
            self.values.popitem(last=False)
        return result


class BiCoordTemporalDataset(Dataset):
    """Causal local-arm history and 100-step absolute target."""

    MODEL_INPUT_FIELDS = frozenset(
        {
            "global_rgb", "local_rgb", "history_visual_raw", "history_qpos",
            "history_action", "history_mask", "action_history_mask", "task_bytes",
            "task_text_mask", "episode_reset",
        }
    )
    TARGET_FIELDS = frozenset({"action", "action_mask"})
    AUDIT_ONLY_FIELDS = frozenset(
        {"task", "sample_key", "episode_index", "episode_id", "time_index", "arm", "source_identity"}
    )

    def __init__(
        self,
        episodes: Sequence[BiCoordEpisode] | str | Path,
        normalization: Mapping[str, Any] | str | Path,
        visual_cache_root: str | Path | None = None,
        *,
        image_height: int = IMAGE_HEIGHT,
        image_width: int = IMAGE_WIDTH,
        cache_limit: int = 32,
        require_visual_cache: bool = False,
    ) -> None:
        if isinstance(episodes, (str, Path)):
            episodes = discover_bicoord_episodes(episodes, require_formal=False)
        self.episodes = list(episodes)
        if not self.episodes:
            raise ValueError("BiCoord dataset cannot be empty")
        for episode in self.episodes:
            _validate_episode_record(episode)
        norm = load_normalization_receipt(normalization, require_formal=False)
        self.q_mean = torch.tensor(norm["qpos_mean"], dtype=torch.float32)
        self.q_std = torch.tensor(norm["qpos_std"], dtype=torch.float32)
        self.a_mean = torch.tensor(norm["action_mean"], dtype=torch.float32)
        self.a_std = torch.tensor(norm["action_std"], dtype=torch.float32)
        if self.q_mean.shape != (STATE_DIM,) or self.a_mean.shape != (ACTION_DIM,):
            raise ValueError("BiCoord normalization must be seven dimensional")
        if torch.any(self.q_std <= 0) or torch.any(self.a_std <= 0):
            raise ValueError("BiCoord normalization standard deviations must be positive")
        self.image_height = int(image_height); self.image_width = int(image_width)
        # Validate dimensions before DataLoader workers are spawned.
        resize_rgb_batch(torch.zeros(1, 16, 16, 3, dtype=torch.uint8), self.image_height, self.image_width)
        self.visual_cache = (
            BiCoordVisualCache(visual_cache_root, limit=cache_limit, require_receipt=require_visual_cache)
            if visual_cache_root is not None else None
        )
        self.require_visual_cache = bool(require_visual_cache)
        self._readers = {item.source_identity: BiCoordHDF5Reader(item.path, task=item.task, episode_id=item.episode_id, stage_path=item.stage_path, instruction_path=item.instruction_path) for item in self.episodes}

    def __len__(self) -> int:
        return sum((episode.length - 1) * ARM_COUNT for episode in self.episodes)

    def _request(self, request: BiCoordTemporalRequest | tuple[Any, ...]) -> BiCoordTemporalRequest:
        if not isinstance(request, BiCoordTemporalRequest):
            request = BiCoordTemporalRequest(*request)
        if not 0 <= request.episode_index < len(self.episodes):
            raise IndexError(request.episode_index)
        episode = self.episodes[request.episode_index]
        if request.task != episode.task or request.arm not in (0, 1):
            raise ValueError(f"BiCoord sample identity mismatch: {request}")
        if not 0 <= request.time_index < episode.length - 1:
            raise IndexError(f"time index {request.time_index} has no next action row")
        return request

    def __getitem__(self, request: BiCoordTemporalRequest | tuple[Any, ...]) -> dict[str, Any]:
        request = self._request(request)
        episode = self.episodes[request.episode_index]
        arm = int(request.arm); t = int(request.time_index)
        reader = self._readers[episode.source_identity]

        observation_first = max(0, t - HISTORY_STEPS + 1)
        observation_indices = np.arange(observation_first, t + 1, dtype=np.int64)
        observation_offset = HISTORY_STEPS - len(observation_indices)
        # Raw row i+1 is the command paired with observation row i.  Commands
        # already executed before observation t therefore occupy rows
        # [t-H+1, t], inclusive.
        action_first = max(0, t - HISTORY_STEPS)
        action_indices = np.arange(action_first, t, dtype=np.int64)
        action_offset = HISTORY_STEPS - len(action_indices)

        history_visual = torch.zeros(HISTORY_STEPS, 2, 768, dtype=torch.float16)
        if self.visual_cache is not None:
            cache = self.visual_cache.load(episode)
            wrist_key = f"view_wrist_{arm}"
            history_visual[observation_offset:, 0] = torch.from_numpy(cache["view_head"][observation_indices])
            history_visual[observation_offset:, 1] = torch.from_numpy(cache[wrist_key][observation_indices])
        elif self.require_visual_cache:
            raise FileNotFoundError("BiCoord visual cache is required for formal model input")

        history_qpos = torch.zeros(HISTORY_STEPS, STATE_DIM)
        for offset, index in enumerate(observation_indices, start=observation_offset):
            qpos = torch.from_numpy(reader.state(int(index), arm))
            history_qpos[offset] = (qpos - self.q_mean) / self.q_std
        history_action = torch.zeros(HISTORY_STEPS, ACTION_DIM)
        for offset, index in enumerate(action_indices, start=action_offset):
            action = torch.from_numpy(reader.action(int(index) + 1, arm))
            history_action[offset] = (action - self.a_mean) / self.a_std
        history_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool); history_mask[observation_offset:] = True
        action_history_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool); action_history_mask[action_offset:] = True

        target_indices = np.arange(t + 1, min(episode.length, t + 1 + ACTION_HORIZON), dtype=np.int64)
        target_values = np.stack([reader.action(int(index), arm) for index in target_indices], axis=0).astype(np.float32)
        valid = len(target_values)
        if valid < 1:
            raise RuntimeError("BiCoord causal request has no future action")
        target = torch.from_numpy((target_values - self.a_mean.numpy()) / self.a_std.numpy())
        padded = torch.empty(ACTION_HORIZON, ACTION_DIM, dtype=torch.float32)
        padded[:valid] = target
        padded[valid:] = target[-1]
        action_mask = torch.zeros(ACTION_HORIZON, dtype=torch.bool); action_mask[:valid] = True

        head = resize_rgb_batch(decode_bicoord_jpeg_rgb(reader.frame_bytes("head_camera", t)), self.image_height, self.image_width)
        wrist_camera = "left_camera" if arm == 0 else "right_camera"
        wrist = resize_rgb_batch(decode_bicoord_jpeg_rgb(reader.frame_bytes(wrist_camera, t)), self.image_height, self.image_width)
        task_bytes, task_text_mask = task_text_tensor(TASK_TEXT[episode.task])
        return {
            "global_rgb": head,
            "local_rgb": wrist,
            "history_visual_raw": history_visual,
            "history_qpos": history_qpos,
            "history_action": history_action,
            "history_mask": history_mask,
            "action_history_mask": action_history_mask,
            "task_bytes": task_bytes,
            "task_text_mask": task_text_mask,
            "episode_reset": torch.tensor(t == 0, dtype=torch.bool),
            "action": padded,
            "action_mask": action_mask,
            "task": episode.task,
            "episode_index": torch.tensor(request.episode_index, dtype=torch.long),
            "episode_id": torch.tensor(episode.episode_id, dtype=torch.long),
            "arm": torch.tensor(arm, dtype=torch.long),
            "time_index": torch.tensor(t, dtype=torch.long),
            "sample_key": request.sample_key,
            "source_identity": episode.source_identity,
        }


class BiCoordBalancedDistributedBatchSampler(Sampler[list[BiCoordTemporalRequest]]):
    """Deterministic task-balanced global batches with DDP slicing."""

    def __init__(
        self,
        episodes: Sequence[BiCoordEpisode],
        *,
        updates: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        start_update: int = 0,
    ) -> None:
        if updates < 1 or not 0 <= start_update <= updates:
            raise ValueError("invalid BiCoord update interval")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed rank")
        if EFFECTIVE_BATCH % world_size:
            raise ValueError(f"world size must divide BiCoord effective batch {EFFECTIVE_BATCH}")
        self.episodes = list(episodes); self.updates = int(updates); self.seed = int(seed)
        self.rank = int(rank); self.world_size = int(world_size); self.start_update = int(start_update)
        self.by_task: dict[str, list[int]] = defaultdict(list)
        for index, episode in enumerate(self.episodes):
            if episode.length < 2: raise ValueError(f"episode too short for causal sampler: {episode.path}")
            self.by_task[episode.task].append(index)
        if set(self.by_task) != set(TASKS):
            raise ValueError(f"expected all 18 BiCoord task buckets, got {sorted(self.by_task)}")

    def __len__(self) -> int:
        return self.updates - self.start_update

    def _extra_tasks(self, update: int) -> tuple[str, ...]:
        start = (update - 1) * EXTRA_SAMPLES_PER_UPDATE
        return tuple(TASKS[(start + offset) % len(TASKS)] for offset in range(EXTRA_SAMPLES_PER_UPDATE))

    def requests_for_update(self, update: int) -> list[BiCoordTemporalRequest]:
        if not 1 <= update <= self.updates: raise IndexError(update)
        rng = random.Random(self.seed + 1_000_003 * update)
        extra = Counter(self._extra_tasks(update))
        rows: list[BiCoordTemporalRequest] = []
        for task in TASKS:
            count = BASE_SAMPLES_PER_TASK + extra[task]
            for _ in range(count):
                episode_index = rng.choice(self.by_task[task]); episode = self.episodes[episode_index]
                arm = rng.randrange(ARM_COUNT); time_index = rng.randrange(episode.length - 1)
                identity = f"{episode.source_identity}:{episode.episode_id}:{arm}:{time_index}"
                rows.append(BiCoordTemporalRequest(episode_index, arm, time_index, hashlib.sha256(identity.encode()).hexdigest(), task))
        rng.shuffle(rows)
        if len(rows) != EFFECTIVE_BATCH:
            raise AssertionError(f"BiCoord batch size failure: {len(rows)}")
        return rows

    def __iter__(self) -> Iterator[list[BiCoordTemporalRequest]]:
        for update in range(self.start_update + 1, self.updates + 1):
            yield self.requests_for_update(update)[self.rank :: self.world_size]

    def cursor_receipt(self, completed_update: int) -> dict[str, Any]:
        if not 0 <= completed_update <= self.updates: raise ValueError(completed_update)
        next_update = completed_update + 1
        rows = self.requests_for_update(next_update) if next_update <= self.updates else []
        return {
            "format_version": DATA_CURSOR_SCHEMA,
            "seed": self.seed,
            "completed_update": completed_update,
            "next_update": next_update if rows else None,
            "next_sample_keys": [row.sample_key for row in rows],
            "effective_batch": EFFECTIVE_BATCH,
            "base_samples_per_task": BASE_SAMPLES_PER_TASK,
            "extra_samples_per_update": EXTRA_SAMPLES_PER_UPDATE,
            "balance_cycle_updates": BALANCE_CYCLE_UPDATES,
        }

    def validate_cursor(self, receipt: Mapping[str, Any]) -> int:
        completed = int(receipt["completed_update"])
        if dict(receipt) != self.cursor_receipt(completed):
            raise ValueError("BiCoord resume sample cursor drifted")
        return completed


# Short aliases make generic CARE launchers easier to reuse.
BalancedDistributedBatchSampler = BiCoordBalancedDistributedBatchSampler
TemporalDataset = BiCoordTemporalDataset


__all__ = [
    "ACTION_DIM", "ACTION_HORIZON", "ACTION_ENCODING", "BASE_SAMPLES_PER_TASK",
    "BiCoordBalancedDistributedBatchSampler", "BiCoordEpisode", "BiCoordTemporalDataset",
    "BiCoordTemporalRequest", "BiCoordVisualCache", "BalancedDistributedBatchSampler",
    "DINO_CACHE_SCHEMA", "DATA_CURSOR_SCHEMA", "NORMALIZATION_SCHEMA", "TemporalDataset",
    "FORBIDDEN_RUNTIME_FIELDS", "LEGAL_RUNTIME_FIELDS", "project_local_observation",
    "validate_local_sample",
    "compute_normalization", "discover_bicoord_episodes", "episode_manifest",
    "load_bicoord_episodes", "load_normalization_receipt", "sha256_file",
    "write_normalization_receipt",
]
