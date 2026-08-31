"""Lazy readers and schema checks for the official BiCoord HDF5 files.

The dataset stores compressed JPEGs in variable-length-looking, fixed-width
HDF5 string arrays.  Opening all 1,800 files or decoding all camera frames at
once is both slow and memory hungry, so this module keeps file handles scoped to
one operation and decodes one frame on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np

from .config import ARM_COUNT, JOINT_DIM, TASKS, TASK_TEXT
from .preprocessing import decode_bicoord_jpeg_rgb


EPISODE_RE = re.compile(r"^episode[_-]?(\d+)\.hdf5$", re.IGNORECASE)
CAMERA_NAMES: tuple[str, ...] = (
    "head_camera",
    "left_camera",
    "right_camera",
    "front_camera",
)
ARM_NAMES: tuple[str, ...] = ("left", "right")


def sha256_file(path: str | Path) -> str:
    """Hash a file in bounded chunks (safe for 50+ GiB dataset trees)."""

    digest = hashlib.sha256()
    with Path(path).open("rb", buffering=0) as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def episode_number(path: str | Path) -> int:
    match = EPISODE_RE.match(Path(path).name)
    if match is None:
        raise ValueError(f"not a BiCoord episode filename: {path}")
    return int(match.group(1))


def _candidate_task_roots(root: Path, task: str) -> tuple[Path, ...]:
    """Return possible task roots for both HF and local benchmark layouts."""

    roots = [
        root / task / "demo_clean",
        root / task,
        root / "data" / task / "demo_clean",
        root / "data" / task,
    ]
    # A caller may pass ``.../task/demo_clean/data`` directly.
    if root.name == "data":
        roots.extend((root.parent, root.parent / "demo_clean"))
    if root.name == "demo_clean":
        roots.append(root)
    return tuple(dict.fromkeys(path for path in roots if path.is_dir()))


def _episode_files_in(path: Path) -> list[Path]:
    for candidate in (path / "data", path):
        if not candidate.is_dir():
            continue
        files = [item for item in candidate.iterdir() if item.is_file() and EPISODE_RE.match(item.name)]
        if files:
            return sorted(files, key=episode_number)
    return []


def discover_episode_files(root: str | Path, task: str) -> list[Path]:
    """Discover raw episode files for one task without following symlink loops."""

    root = Path(root).expanduser().resolve()
    if task not in TASKS:
        raise ValueError(f"unsupported BiCoord task: {task}")
    for task_root in _candidate_task_roots(root, task):
        files = _episode_files_in(task_root)
        if files:
            return files

    # Last-resort recursive search handles snapshots with an extra repository
    # directory while still restricting matches to the requested task subtree.
    matches: list[Path] = []
    for path in root.glob(f"**/{task}/**/episode*.hdf5"):
        if path.is_file() and EPISODE_RE.match(path.name):
            matches.append(path.resolve())
    return sorted(set(matches), key=episode_number)


def discover_all_episode_files(root: str | Path, *, require_complete: bool = False) -> dict[str, list[Path]]:
    result = {task: discover_episode_files(root, task) for task in TASKS}
    if require_complete:
        bad = {task: len(paths) for task, paths in result.items() if len(paths) != 100}
        if bad:
            raise ValueError(f"formal BiCoord corpus requires 100 episodes/task: {bad}")
    return result


def _dataset(handle: h5py.File, path: str) -> h5py.Dataset:
    try:
        value = handle[path]
    except KeyError as error:
        raise ValueError(f"missing BiCoord HDF5 dataset {path}") from error
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"BiCoord HDF5 path is not a dataset: {path}")
    return value


def _as_jpeg_bytes(value: Any) -> bytes:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            value = value.reshape(-1)[0]
        value = value.item()
    if isinstance(value, np.bytes_):
        return bytes(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"unsupported HDF5 JPEG scalar: {type(value)!r}")


def validate_hdf5_schema(path: str | Path, *, check_images: bool = False) -> dict[str, Any]:
    """Validate one source file and return immutable shape metadata."""

    path = Path(path).expanduser().resolve(strict=True)
    with h5py.File(path, "r", swmr=True) as handle:
        left_arm = _dataset(handle, "joint_action/left_arm")
        right_arm = _dataset(handle, "joint_action/right_arm")
        left_gripper = _dataset(handle, "joint_action/left_gripper")
        right_gripper = _dataset(handle, "joint_action/right_gripper")
        lengths = {
            int(left_arm.shape[0]),
            int(right_arm.shape[0]),
            int(left_gripper.shape[0]),
            int(right_gripper.shape[0]),
        }
        if len(lengths) != 1:
            raise ValueError(f"BiCoord arm streams have different lengths: {sorted(lengths)}")
        length = lengths.pop()
        if length < 1:
            raise ValueError(f"empty BiCoord episode: {path}")
        for name, value in (("left_arm", left_arm), ("right_arm", right_arm)):
            if value.ndim != 2 or value.shape[1] != JOINT_DIM:
                raise ValueError(f"{name} must have shape [T,6], got {value.shape}")
            if not np.issubdtype(value.dtype, np.number):
                raise ValueError(f"{name} must be numeric, got {value.dtype}")
        for name, value in (("left_gripper", left_gripper), ("right_gripper", right_gripper)):
            if value.ndim != 1 or not np.issubdtype(value.dtype, np.number):
                raise ValueError(f"{name} must be numeric [T], got {value.shape}/{value.dtype}")
        vector = _dataset(handle, "joint_action/vector")
        if vector.shape != (length, ARM_COUNT * (JOINT_DIM + 1)) or not np.issubdtype(vector.dtype, np.number):
            raise ValueError(f"joint_action/vector must be [T,14], got {vector.shape}/{vector.dtype}")
        # The redundant vector is useful corruption evidence.  Compare in
        # bounded chunks so a malformed source cannot silently reorder arms or
        # place the gripper in a different channel.
        for first in range(0, length, 4096):
            last = min(length, first + 4096)
            expected = np.concatenate(
                (
                    np.asarray(left_arm[first:last]),
                    np.asarray(left_gripper[first:last])[:, None],
                    np.asarray(right_arm[first:last]),
                    np.asarray(right_gripper[first:last])[:, None],
                ),
                axis=1,
            )
            observed = np.asarray(vector[first:last])
            if not np.array_equal(observed, expected, equal_nan=False):
                raise ValueError("joint_action/vector differs from per-arm datasets")
        cameras: dict[str, tuple[int, ...]] = {}
        for camera in CAMERA_NAMES:
            dataset = _dataset(handle, f"observation/{camera}/rgb")
            if dataset.ndim != 1 or dataset.shape[0] != length:
                raise ValueError(f"{camera}/rgb must be [T] JPEG bytes, got {dataset.shape}")
            cameras[camera] = tuple(int(v) for v in dataset.shape)
            if check_images:
                # Decode just first and last frame to catch truncated files;
                # full validation remains streaming in the dataset worker.
                for index in sorted({0, length - 1}):
                    decode_bicoord_jpeg_rgb(_as_jpeg_bytes(dataset[index]))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "length": length,
        "state_dim": JOINT_DIM + 1,
        "action_dim": JOINT_DIM + 1,
        "arms": ARM_COUNT,
        "cameras": list(CAMERA_NAMES),
    }


@dataclass(frozen=True)
class StageSegment:
    start: int
    end: int
    global_text: str
    left_text: str
    right_text: str


def load_stage_segments(path: str | Path, *, length: int | None = None) -> tuple[StageSegment, ...]:
    """Load optional ``stages/episodeN.json`` metadata and validate alignment."""

    path = Path(path)
    if not path.is_file():
        return ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid BiCoord stage file: {path}") from error
    if not isinstance(raw, list):
        raise ValueError(f"BiCoord stage file must contain a list: {path}")
    result: list[StageSegment] = []
    previous_end = 0
    for row in raw:
        if not isinstance(row, Sequence) or len(row) < 5:
            raise ValueError(f"invalid BiCoord stage row: {row!r}")
        start, end = int(row[0]), int(row[1])
        if start != previous_end or end <= start or (length is not None and end > length):
            raise ValueError(f"non-contiguous/out-of-range BiCoord stage: {row!r}")
        result.append(StageSegment(start, end, str(row[2]), str(row[3]), str(row[4])))
        previous_end = end
    if length is not None and result and previous_end != length:
        raise ValueError(f"BiCoord stages end at {previous_end}, episode has {length} frames")
    return tuple(result)


class BiCoordHDF5Reader:
    """A small, reopenable reader for one BiCoord episode."""

    def __init__(
        self,
        path: str | Path,
        *,
        task: str | None = None,
        episode_id: int | None = None,
        stage_path: str | Path | None = None,
        instruction_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve(strict=True)
        self.task = task
        self.episode_id = episode_number(self.path) if episode_id is None else int(episode_id)
        self._length: int | None = None
        self.stage_path = Path(stage_path) if stage_path is not None else None
        self.instruction_path = Path(instruction_path) if instruction_path is not None else None

    @property
    def length(self) -> int:
        if self._length is None:
            with h5py.File(self.path, "r", swmr=True) as handle:
                self._length = int(_dataset(handle, "joint_action/left_arm").shape[0])
        return self._length

    def _open(self) -> h5py.File:
        return h5py.File(self.path, "r", swmr=True)

    @staticmethod
    def _arm_name(arm: int) -> str:
        if int(arm) not in (0, 1):
            raise ValueError(f"BiCoord arm must be 0 or 1, got {arm}")
        return ARM_NAMES[int(arm)]

    def state(self, index: int, arm: int) -> np.ndarray:
        return self._joint_row(index, arm, prefix="joint_action")

    def action(self, index: int, arm: int) -> np.ndarray:
        # The released source has no separate command dataset.  ``joint_action``
        # is the controller-equivalent absolute target used by all baselines.
        return self._joint_row(index, arm, prefix="joint_action")

    def _joint_row(self, index: int, arm: int, *, prefix: str) -> np.ndarray:
        index = int(index)
        if not 0 <= index < self.length:
            raise IndexError(index)
        name = self._arm_name(arm)
        with self._open() as handle:
            joints = np.asarray(_dataset(handle, f"{prefix}/{name}_arm")[index], dtype=np.float32)
            gripper = float(_dataset(handle, f"{prefix}/{name}_gripper")[index])
        if joints.shape != (JOINT_DIM,) or not np.isfinite(joints).all() or not np.isfinite(gripper):
            raise ValueError(f"invalid BiCoord arm row at {self.path}:{index}:{arm}")
        # Preserve source values; binary conversion is a separate explicit
        # contract step and never clips joint coordinates.
        return np.concatenate((joints, np.asarray([gripper], dtype=np.float32)))

    def frame_bytes(self, camera: str, index: int) -> bytes:
        camera = str(camera)
        if camera not in CAMERA_NAMES:
            raise ValueError(f"unsupported BiCoord camera {camera!r}")
        index = int(index)
        if not 0 <= index < self.length:
            raise IndexError(index)
        with self._open() as handle:
            return _as_jpeg_bytes(_dataset(handle, f"observation/{camera}/rgb")[index])

    def frame(self, camera: str, index: int) -> np.ndarray:
        return decode_bicoord_jpeg_rgb(self.frame_bytes(camera, index))

    def stages(self) -> tuple[StageSegment, ...]:
        if self.stage_path is None:
            return ()
        return load_stage_segments(self.stage_path, length=self.length)

    def instruction(self) -> str:
        """Return canonical task text, never an episode-specific privileged label."""

        if self.task in TASK_TEXT:
            return TASK_TEXT[self.task]
        if self.instruction_path and self.instruction_path.is_file():
            try:
                value = json.loads(self.instruction_path.read_text(encoding="utf-8"))
                if isinstance(value, Mapping):
                    text = value.get("full_description")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
            except (OSError, json.JSONDecodeError):
                pass
        return ""

    def iter_frames(self, camera: str, indices: Sequence[int] | None = None) -> Iterator[np.ndarray]:
        values = range(self.length) if indices is None else indices
        for index in values:
            yield self.frame(camera, int(index))


__all__ = [
    "ARM_NAMES",
    "CAMERA_NAMES",
    "BiCoordHDF5Reader",
    "StageSegment",
    "discover_all_episode_files",
    "discover_episode_files",
    "episode_number",
    "load_stage_segments",
    "sha256_file",
    "validate_hdf5_schema",
]
