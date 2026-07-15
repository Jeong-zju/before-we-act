"""Episode-safe HDF5 sequence windows for proprioceptive WAM training."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.trajectory import PROPRIO_WAM_SCHEMA_VERSION


@dataclass(frozen=True)
class EpisodeSequenceIndex:
    file_index: int
    decision_t: int


@dataclass(frozen=True)
class _EpisodeRecord:
    path: Path
    num_steps: int
    seed: int
    episode_index: int
    state_dim: int
    action_dim: int
    layout: str


PreloadProgressCallback = Callable[[Mapping[str, int]], None]


def discover_episode_paths(data_dir: str | Path) -> list[Path]:
    """Find completed episode files below either a dataset or HDF5 directory."""

    root = Path(data_dir)
    if root.is_file():
        paths = [root] if root.name.endswith(".hdf5") else []
    else:
        paths = sorted(root.rglob("episode_*.hdf5"))
    paths = [path for path in paths if ".partial." not in path.name]
    if not paths:
        raise FileNotFoundError(f"no episode_*.hdf5 files found below {root}")
    return paths


def split_episode_paths(
    paths: Sequence[str | Path],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, tuple[Path, ...]]:
    """Split whole seed groups so no episode or seed crosses a partition."""

    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=np.float64
    )
    if np.any(fractions < 0.0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError(
            "train/validation/test fractions must be non-negative and sum to 1"
        )
    groups: dict[tuple[str, int | str], list[Path]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with h5py.File(path, "r") as file:
            raw_seed = int(file.attrs.get("seed", -1))
        key: tuple[str, int | str]
        key = ("seed", raw_seed) if raw_seed >= 0 else ("path", str(path.resolve()))
        groups.setdefault(key, []).append(path)

    keys = list(groups)
    np.random.default_rng(seed).shuffle(keys)
    raw_counts = fractions * len(keys)
    counts = np.floor(raw_counts).astype(np.int64)
    for index in np.argsort(-(raw_counts - counts))[: len(keys) - int(counts.sum())]:
        counts[index] += 1
    if keys and counts[0] == 0:
        donor = int(np.argmax(counts))
        counts[donor] -= 1
        counts[0] += 1
    for index in (1, 2):
        if len(keys) >= 3 and fractions[index] > 0.0 and counts[index] == 0:
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[index] += 1

    result: dict[str, list[Path]] = {"train": [], "validation": [], "test": []}
    cursor = 0
    for name, count in zip(result, counts, strict=True):
        for key in keys[cursor : cursor + int(count)]:
            result[name].extend(groups[key])
        cursor += int(count)
    return {name: tuple(sorted(items)) for name, items in result.items()}


class ProprioSequenceDataset(Dataset):
    """Return right-padded history and future windows from one episode only.

    A sample at transition ``t`` ends its state history at ``observation[t]``.
    ``past_actions`` contains only actions that led between history states, so
    the candidate action at ``t`` is never leaked into the belief history.
    Future labels are right-padded and identified by ``forecast_mask``.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        paths: Sequence[str | Path] | None = None,
        history_horizon: int = 32,
        forecast_horizon: int = 16,
        stride: int = 1,
        state_dim: int = 22,
        action_dim: int = 8,
        allow_legacy_wam: bool = True,
        hdf5_cache_size: int = 8,
    ) -> None:
        if paths is None:
            if data_dir is None:
                raise ValueError("data_dir or paths must be provided")
            resolved_paths = discover_episode_paths(data_dir)
        else:
            resolved_paths = [Path(path) for path in paths]
        if not resolved_paths:
            raise ValueError("paths cannot be empty")
        for name, value in (
            ("history_horizon", history_horizon),
            ("forecast_horizon", forecast_horizon),
            ("stride", stride),
            ("state_dim", state_dim),
            ("action_dim", action_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if hdf5_cache_size < 0:
            raise ValueError("hdf5_cache_size cannot be negative")

        self.history_horizon = int(history_horizon)
        self.forecast_horizon = int(forecast_horizon)
        self.stride = int(stride)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hdf5_cache_size = int(hdf5_cache_size)
        self._cache_pid = os.getpid()
        self._cache: OrderedDict[str, h5py.File] = OrderedDict()
        self.records = [
            self._inspect_episode(path, allow_legacy_wam=allow_legacy_wam)
            for path in sorted(resolved_paths)
        ]
        self.paths = [record.path for record in self.records]
        self.index = [
            EpisodeSequenceIndex(file_index=file_index, decision_t=decision_t)
            for file_index, record in enumerate(self.records)
            for decision_t in range(0, record.num_steps, self.stride)
        ]
        if not self.index:
            raise RuntimeError("no transitions are available in the selected episodes")

    def _inspect_episode(self, path: Path, *, allow_legacy_wam: bool) -> _EpisodeRecord:
        return _inspect_episode_record(
            path,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            allow_legacy_wam=allow_legacy_wam,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = self.index[item]
        record = self.records[sample.file_index]
        t = sample.decision_t
        history_start = max(0, t - self.history_horizon + 1)
        history_count = t - history_start + 1
        history_offset = self.history_horizon - history_count
        forecast_end = min(record.num_steps, t + self.forecast_horizon)
        forecast_count = forecast_end - t

        states = np.zeros((self.history_horizon, self.state_dim), dtype=np.float32)
        past_actions = np.zeros(
            (max(0, self.history_horizon - 1), self.action_dim), dtype=np.float32
        )
        valid_mask = np.zeros(self.history_horizon, dtype=np.bool_)
        past_action_mask = np.zeros(max(0, self.history_horizon - 1), dtype=np.bool_)
        candidate_actions = np.zeros(
            (self.forecast_horizon, self.action_dim), dtype=np.float32
        )
        executed_actions = np.zeros_like(candidate_actions)
        target_states = np.zeros(
            (self.forecast_horizon, self.state_dim), dtype=np.float32
        )
        forecast_mask = np.zeros(self.forecast_horizon, dtype=np.bool_)

        with self._open_hdf5(record.path) as file:
            states[history_offset:] = self._read_states(
                file, record.layout, history_start, t + 1, next_state=False
            )
            valid_mask[history_offset:] = True
            if history_count > 1:
                past_actions[history_offset : self.history_horizon - 1] = (
                    self._read_actions(
                        file,
                        record.layout,
                        history_start,
                        t,
                        executed=False,
                    )
                )
                past_action_mask[history_offset : self.history_horizon - 1] = True
            candidate_actions[:forecast_count] = self._read_actions(
                file, record.layout, t, forecast_end, executed=False
            )
            executed_actions[:forecast_count] = self._read_actions(
                file, record.layout, t, forecast_end, executed=True
            )
            target_states[:forecast_count] = self._read_states(
                file, record.layout, t, forecast_end, next_state=True
            )
            forecast_mask[:forecast_count] = True
            scalars = {
                name: self._read_scalar(file, name, t, forecast_end, record.layout)
                for name in (
                    "reward",
                    "done",
                    "success",
                    "failure",
                    "response_progress",
                    "coordination_error",
                )
            }

        result = {
            "states": torch.from_numpy(states),
            "past_actions": torch.from_numpy(past_actions),
            "valid_mask": torch.from_numpy(valid_mask),
            "past_action_mask": torch.from_numpy(past_action_mask),
            "candidate_actions": torch.from_numpy(candidate_actions),
            "executed_actions": torch.from_numpy(executed_actions),
            "target_states": torch.from_numpy(target_states),
            "forecast_mask": torch.from_numpy(forecast_mask),
            "episode_index": torch.tensor(record.episode_index, dtype=torch.int64),
            "episode_seed": torch.tensor(record.seed, dtype=torch.int64),
            "decision_t": torch.tensor(t, dtype=torch.int64),
        }
        output_names = {
            "reward": "rewards",
            "done": "dones",
            "success": "successes",
            "failure": "failures",
        }
        for name, values in scalars.items():
            padded = np.zeros((self.forecast_horizon, 1), dtype=np.float32)
            padded[:forecast_count, 0] = values
            result[output_names.get(name, name)] = torch.from_numpy(padded)
        return result

    @contextmanager
    def _open_hdf5(self, path: Path) -> Iterator[h5py.File]:
        if self.hdf5_cache_size == 0:
            with h5py.File(path, "r") as file:
                yield file
            return
        if self._cache_pid != os.getpid():
            self.close()
            self._cache_pid = os.getpid()
        key = str(path.resolve())
        file = self._cache.pop(key, None)
        if file is None:
            file = h5py.File(path, "r")
        self._cache[key] = file
        while len(self._cache) > self.hdf5_cache_size:
            _, stale = self._cache.popitem(last=False)
            stale.close()
        yield file

    def _read_states(
        self,
        file: h5py.File,
        layout: str,
        start: int,
        stop: int,
        *,
        next_state: bool,
    ) -> np.ndarray:
        return _read_states(file, layout, start, stop, next_state=next_state)

    def _read_actions(
        self,
        file: h5py.File,
        layout: str,
        start: int,
        stop: int,
        *,
        executed: bool,
    ) -> np.ndarray:
        return _read_actions(file, layout, start, stop, executed=executed)

    @staticmethod
    def _read_scalar(
        file: h5py.File,
        name: str,
        start: int,
        stop: int,
        layout: str,
    ) -> np.ndarray:
        return _read_scalar(file, name, start, stop, layout)

    def close(self) -> None:
        for file in self._cache.values():
            file.close()
        self._cache.clear()

    def __getstate__(self) -> Mapping[str, object]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["_cache_pid"] = os.getpid()
        return state

    def __del__(self) -> None:
        if hasattr(self, "_cache"):
            self.close()


class InMemoryOneStepDataset(Dataset):
    """Preload the minimal one-step baseline fields and reuse them across epochs.

    The sequence dataset intentionally exposes full history and forecast windows for
    later WAM models. Phase 0 baselines consume only the final history state and the
    first forecast target, so rebuilding those overlapping windows on every epoch is
    unnecessary. This dataset reads each episode once, stores only the seven tensors
    used by the baseline losses, and preserves the existing batch field shapes.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        paths: Sequence[str | Path] | None = None,
        state_dim: int = 22,
        action_dim: int = 8,
        allow_legacy_wam: bool = True,
        progress: PreloadProgressCallback | None = None,
    ) -> None:
        if paths is None:
            if data_dir is None:
                raise ValueError("data_dir or paths must be provided")
            resolved_paths = discover_episode_paths(data_dir)
        else:
            resolved_paths = [Path(path) for path in paths]
        if not resolved_paths:
            raise ValueError("paths cannot be empty")
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        records = [
            _inspect_episode_record(
                path,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                allow_legacy_wam=allow_legacy_wam,
            )
            for path in sorted(resolved_paths)
        ]
        self.paths = [record.path for record in records]
        self.num_episodes = len(records)
        sample_count = sum(record.num_steps for record in records)
        self._tensors = {
            "states": torch.empty(
                (sample_count, 1, self.state_dim), dtype=torch.float32
            ),
            "candidate_actions": torch.empty(
                (sample_count, 1, self.action_dim), dtype=torch.float32
            ),
            "target_states": torch.empty(
                (sample_count, 1, self.state_dim), dtype=torch.float32
            ),
            "rewards": torch.empty((sample_count, 1, 1), dtype=torch.float32),
            "dones": torch.empty((sample_count, 1, 1), dtype=torch.float32),
            "successes": torch.empty((sample_count, 1, 1), dtype=torch.float32),
            "failures": torch.empty((sample_count, 1, 1), dtype=torch.float32),
        }

        cursor = 0
        for episode_number, record in enumerate(records, start=1):
            stop = cursor + record.num_steps
            with h5py.File(record.path, "r") as file:
                states = _read_states(
                    file, record.layout, 0, record.num_steps, next_state=False
                )
                actions = _read_actions(
                    file, record.layout, 0, record.num_steps, executed=False
                )
                target_states = _read_states(
                    file, record.layout, 0, record.num_steps, next_state=True
                )
                rewards = _read_scalar(
                    file, "reward", 0, record.num_steps, record.layout
                )
                dones = _read_scalar(file, "done", 0, record.num_steps, record.layout)
                successes = _read_scalar(
                    file, "success", 0, record.num_steps, record.layout
                )
                failures = _read_scalar(
                    file, "failure", 0, record.num_steps, record.layout
                )
            self._tensors["states"][cursor:stop, 0].copy_(torch.from_numpy(states))
            self._tensors["candidate_actions"][cursor:stop, 0].copy_(
                torch.from_numpy(actions)
            )
            self._tensors["target_states"][cursor:stop, 0].copy_(
                torch.from_numpy(target_states)
            )
            self._tensors["rewards"][cursor:stop, 0, 0].copy_(torch.from_numpy(rewards))
            self._tensors["dones"][cursor:stop, 0, 0].copy_(torch.from_numpy(dones))
            self._tensors["successes"][cursor:stop, 0, 0].copy_(
                torch.from_numpy(successes)
            )
            self._tensors["failures"][cursor:stop, 0, 0].copy_(
                torch.from_numpy(failures)
            )
            cursor = stop
            if progress is not None:
                progress(
                    {
                        "episode": episode_number,
                        "episodes": self.num_episodes,
                        "samples": cursor,
                    }
                )

    def __len__(self) -> int:
        return int(self._tensors["states"].shape[0])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        return {name: tensor[item] for name, tensor in self._tensors.items()}

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for tensor in self._tensors.values()
        )

    def close(self) -> None:
        """Match the sequence dataset lifecycle API; no files remain open."""


def _inspect_episode_record(
    path: Path,
    *,
    state_dim: int,
    action_dim: int,
    allow_legacy_wam: bool,
) -> _EpisodeRecord:
    with h5py.File(path, "r") as file:
        profile = str(file.attrs.get("schema_profile", ""))
        version = str(file.attrs.get("schema_version", ""))
        if "data/observation/state" in file:
            layout = "wam_proprio"
            if version != PROPRIO_WAM_SCHEMA_VERSION:
                raise ValueError(
                    f"{path} has proprio fields but schema version {version!r}; "
                    f"expected {PROPRIO_WAM_SCHEMA_VERSION!r}"
                )
            state_shape = file["data/observation/state"].shape
            action_shape = file["data/commanded_action"].shape
        elif allow_legacy_wam and profile in {"wam", "world_action_model"}:
            layout = "legacy_wam"
            state_shape = (
                file["data/observation/agent_0"].shape[0],
                file["data/observation/agent_0"].shape[1]
                + file["data/observation/agent_1"].shape[1],
            )
            action_shape = file["data/action"].shape
        else:
            raise ValueError(
                f"{path} is neither {PROPRIO_WAM_SCHEMA_VERSION} nor a supported "
                "legacy WAM episode"
            )
        if len(state_shape) != 2 or state_shape[1] != state_dim:
            raise ValueError(
                f"{path} state shape {state_shape} does not match (*,{state_dim})"
            )
        if len(action_shape) != 2 or action_shape[1] != action_dim:
            raise ValueError(
                f"{path} action shape {action_shape} does not match (*,{action_dim})"
            )
        num_steps = int(file.attrs.get("num_steps", state_shape[0]))
        if num_steps != state_shape[0] or num_steps != action_shape[0]:
            raise ValueError(f"{path} contains inconsistent transition counts")
        return _EpisodeRecord(
            path=path,
            num_steps=num_steps,
            seed=int(file.attrs.get("seed", -1)),
            episode_index=int(file.attrs.get("episode_index", -1)),
            state_dim=state_shape[1],
            action_dim=action_shape[1],
            layout=layout,
        )


def _read_states(
    file: h5py.File,
    layout: str,
    start: int,
    stop: int,
    *,
    next_state: bool,
) -> np.ndarray:
    if layout == "wam_proprio":
        prefix = "next_observation" if next_state else "observation"
        return np.asarray(file[f"data/{prefix}/state"][start:stop], dtype=np.float32)
    prefix = "next_observation" if next_state else "observation"
    return np.concatenate(
        (
            file[f"data/{prefix}/agent_0"][start:stop],
            file[f"data/{prefix}/agent_1"][start:stop],
        ),
        axis=-1,
        dtype=np.float32,
    )


def _read_actions(
    file: h5py.File,
    layout: str,
    start: int,
    stop: int,
    *,
    executed: bool,
) -> np.ndarray:
    if layout == "wam_proprio":
        name = "executed_action" if executed else "commanded_action"
        return np.asarray(file[f"data/{name}"][start:stop], dtype=np.float32)
    return np.asarray(file["data/action"][start:stop], dtype=np.float32)


def _read_scalar(
    file: h5py.File,
    name: str,
    start: int,
    stop: int,
    layout: str,
) -> np.ndarray:
    del layout
    path = f"data/{name}"
    if path not in file:
        return np.zeros(stop - start, dtype=np.float32)
    return np.asarray(file[path][start:stop], dtype=np.float32).reshape(-1)


__all__ = [
    "EpisodeSequenceIndex",
    "InMemoryOneStepDataset",
    "PreloadProgressCallback",
    "ProprioSequenceDataset",
    "discover_episode_paths",
    "split_episode_paths",
]
