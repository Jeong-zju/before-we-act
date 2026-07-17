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
    later WAM models. baseline baselines consume only the final history state and the
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


class InMemoryProprioSequenceDataset(Dataset):
    """RAM-backed episode-safe sequence windows for recurrent world model training.

    Raw transitions are loaded exactly once.  Histories and forecasts are sliced
    lazily, avoiding both random HDF5 access across 10,000 files and materializing
    millions of overlapping 32+16 step windows.
    """

    _FIELDS = (
        "states",
        "next_states",
        "commanded_actions",
        "executed_actions",
        "rewards",
        "dones",
        "successes",
        "failures",
        "response_progress",
        "coordination_error",
    )
    CACHE_FORMAT_VERSION = "wam.in_memory_proprio_sequence/1"

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
        allow_legacy_wam: bool = False,
        planning_discount: float | None = None,
        action_prior_behavior_weights: Mapping[str, float] | None = None,
        action_prior_require_success: bool = True,
        action_prior_min_return_quantile: float = 0.0,
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
        for name, value in (
            ("history_horizon", history_horizon),
            ("forecast_horizon", forecast_horizon),
            ("stride", stride),
            ("state_dim", state_dim),
            ("action_dim", action_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        self.history_horizon = int(history_horizon)
        self.forecast_horizon = int(forecast_horizon)
        self.stride = int(stride)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        if planning_discount is not None and not 0.0 < planning_discount <= 1.0:
            raise ValueError("planning_discount must be in (0,1]")
        if not 0.0 <= action_prior_min_return_quantile < 1.0:
            raise ValueError("action_prior_min_return_quantile must be in [0,1)")
        behavior_weights = {
            str(name): float(weight)
            for name, weight in (action_prior_behavior_weights or {}).items()
        }
        if any(weight < 0.0 for weight in behavior_weights.values()):
            raise ValueError("action-prior behavior weights must be non-negative")
        self.planning_discount = planning_discount
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
        self._episode_seeds = np.asarray(
            [record.seed for record in records], dtype=np.int64
        )
        self._episode_indices = np.asarray(
            [record.episode_index for record in records], dtype=np.int64
        )
        self._episode_offsets = np.zeros(self.num_episodes + 1, dtype=np.int64)
        self._episode_offsets[1:] = np.cumsum(
            [record.num_steps for record in records], dtype=np.int64
        )
        transition_count = int(self._episode_offsets[-1])
        self._tensors = {
            "states": torch.empty((transition_count, self.state_dim)),
            "next_states": torch.empty((transition_count, self.state_dim)),
            "commanded_actions": torch.empty((transition_count, self.action_dim)),
            "executed_actions": torch.empty((transition_count, self.action_dim)),
            "rewards": torch.empty((transition_count, 1)),
            "dones": torch.empty((transition_count, 1)),
            "successes": torch.empty((transition_count, 1)),
            "failures": torch.empty((transition_count, 1)),
            "response_progress": torch.empty((transition_count, 1)),
            "coordination_error": torch.empty((transition_count, 1)),
        }
        if planning_discount is not None:
            self._tensors.update(
                {
                    "returns_to_go": torch.empty((transition_count, 1)),
                    "episode_returns": torch.empty((transition_count, 1)),
                    "action_prior_weights": torch.empty((transition_count, 1)),
                }
            )
        episode_returns = np.zeros(self.num_episodes, dtype=np.float32)
        episode_successes = np.zeros(self.num_episodes, dtype=np.bool_)
        episode_behavior_weights = np.zeros(self.num_episodes, dtype=np.float32)
        episode_behavior_ids: list[str] = ["unknown"] * self.num_episodes
        sample_episode: list[np.ndarray] = []
        sample_decision: list[np.ndarray] = []
        for episode_number, record in enumerate(records, start=1):
            start = int(self._episode_offsets[episode_number - 1])
            stop = int(self._episode_offsets[episode_number])
            with h5py.File(record.path, "r") as file:
                arrays = {
                    "states": _read_states(
                        file, record.layout, 0, record.num_steps, next_state=False
                    ),
                    "next_states": _read_states(
                        file, record.layout, 0, record.num_steps, next_state=True
                    ),
                    "commanded_actions": _read_actions(
                        file, record.layout, 0, record.num_steps, executed=False
                    ),
                    "executed_actions": _read_actions(
                        file, record.layout, 0, record.num_steps, executed=True
                    ),
                }
                for field, source in (
                    ("rewards", "reward"),
                    ("dones", "done"),
                    ("successes", "success"),
                    ("failures", "failure"),
                    ("response_progress", "response_progress"),
                    ("coordination_error", "coordination_error"),
                ):
                    arrays[field] = _read_scalar(
                        file, source, 0, record.num_steps, record.layout
                    )[:, None]
                behavior_id = str(file.attrs.get("behavior_id", "unknown"))
                episode_return = float(
                    file.attrs.get("total_reward", arrays["rewards"].sum())
                )
            for field, array in arrays.items():
                self._tensors[field][start:stop].copy_(torch.from_numpy(array))
            if planning_discount is not None:
                returns_to_go = _discounted_returns(
                    arrays["rewards"].reshape(-1), planning_discount
                )
                success = bool(np.asarray(arrays["successes"]).max(initial=0.0) >= 0.5)
                behavior_weight = behavior_weights.get(
                    behavior_id, 1.0 if not behavior_weights else 0.0
                )
                if action_prior_require_success and not success:
                    behavior_weight = 0.0
                self._tensors["returns_to_go"][start:stop, 0].copy_(
                    torch.from_numpy(returns_to_go)
                )
                self._tensors["episode_returns"][start:stop, 0].fill_(episode_return)
                self._tensors["action_prior_weights"][start:stop, 0].fill_(
                    behavior_weight
                )
                episode_returns[episode_number - 1] = episode_return
                episode_successes[episode_number - 1] = success
                episode_behavior_weights[episode_number - 1] = behavior_weight
                episode_behavior_ids[episode_number - 1] = behavior_id
            decisions = np.arange(0, record.num_steps, self.stride, dtype=np.int32)
            sample_episode.append(
                np.full(decisions.shape, episode_number - 1, dtype=np.int32)
            )
            sample_decision.append(decisions)
            if progress is not None:
                progress(
                    {
                        "episode": episode_number,
                        "episodes": self.num_episodes,
                        "samples": stop,
                    }
                )
        return_threshold: float | None = None
        if planning_discount is not None and action_prior_min_return_quantile > 0.0:
            eligible = episode_returns[
                episode_successes & (episode_behavior_weights > 0.0)
            ]
            if eligible.size == 0:
                raise RuntimeError("no successful action-prior episodes are eligible")
            threshold = float(
                np.quantile(eligible, action_prior_min_return_quantile)
            )
            return_threshold = threshold
            for episode, episode_return in enumerate(episode_returns):
                if episode_return >= threshold:
                    continue
                start = int(self._episode_offsets[episode])
                stop = int(self._episode_offsets[episode + 1])
                self._tensors["action_prior_weights"][start:stop].zero_()
        if planning_discount is not None:
            self.planning_metadata = {
                "discount": float(planning_discount),
                "action_prior_require_success": bool(action_prior_require_success),
                "action_prior_min_return_quantile": float(
                    action_prior_min_return_quantile
                ),
                "eligible_episodes": int(
                    sum(
                        float(
                            self._tensors["action_prior_weights"][
                                int(self._episode_offsets[index])
                            ]
                        )
                        > 0.0
                        for index in range(self.num_episodes)
                    )
                ),
                "behavior_weights": behavior_weights,
            }
        self._sample_episode = np.concatenate(sample_episode)
        self._sample_decision = np.concatenate(sample_decision)
        if planning_discount is not None:
            eligible_episodes = []
            for index in range(self.num_episodes):
                offset = int(self._episode_offsets[index])
                weight = float(self._tensors["action_prior_weights"][offset])
                if weight <= 0.0:
                    continue
                eligible_episodes.append(
                    {
                        "episode_index": int(self._episode_indices[index]),
                        "seed": int(self._episode_seeds[index]),
                        "behavior_id": episode_behavior_ids[index],
                        "episode_return": float(episode_returns[index]),
                        "weight": weight,
                    }
                )
            eligible_set = {
                int(item["episode_index"]) for item in eligible_episodes
            }
            self.action_quality_metadata = {
                "discount": float(planning_discount),
                "require_success": bool(action_prior_require_success),
                "minimum_return_quantile": float(
                    action_prior_min_return_quantile
                ),
                "return_threshold": return_threshold,
                "behavior_weights": behavior_weights,
                "eligible_episode_count": len(eligible_episodes),
                "eligible_window_count": int(
                    sum(
                        int(episode_index) in eligible_set
                        for episode_index in self._episode_indices[
                            self._sample_episode
                        ]
                    )
                ),
                "eligible_episodes": eligible_episodes,
            }

    def __len__(self) -> int:
        return int(self._sample_episode.size)

    def complete_forecast_indices(
        self,
        *,
        require_positive_action_quality: bool = False,
    ) -> np.ndarray:
        """Return windows whose full forecast remains inside one episode.

        Joint WAM action chunks may not use padded tail actions.  This method
        exposes the episode-safe index calculation without materializing every
        overlapping history window merely to inspect its mask.
        """

        episode_lengths = np.diff(self._episode_offsets)
        remaining = episode_lengths[self._sample_episode] - self._sample_decision
        selected = remaining >= self.forecast_horizon
        if require_positive_action_quality:
            if self.planning_discount is None:
                raise ValueError(
                    "action-quality filtering requires planning_discount"
                )
            absolute = (
                self._episode_offsets[self._sample_episode]
                + self._sample_decision
            )
            quality = self._tensors["action_prior_weights"][absolute, 0].numpy()
            selected &= quality > 0.0
        return np.flatnonzero(selected).astype(np.int64, copy=False)

    @property
    def episode_seeds(self) -> np.ndarray:
        """Return a defensive copy for portable split/lineage manifests."""

        return self._episode_seeds.copy()

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode = int(self._sample_episode[item])
        decision_t = int(self._sample_decision[item])
        episode_start = int(self._episode_offsets[episode])
        episode_stop = int(self._episode_offsets[episode + 1])
        num_steps = episode_stop - episode_start
        history_start_t = max(0, decision_t - self.history_horizon + 1)
        history_count = decision_t - history_start_t + 1
        history_offset = self.history_horizon - history_count
        forecast_stop_t = min(num_steps, decision_t + self.forecast_horizon)
        forecast_count = forecast_stop_t - decision_t
        history_start = episode_start + history_start_t
        current_stop = episode_start + decision_t + 1
        forecast_start = episode_start + decision_t
        forecast_stop = episode_start + forecast_stop_t

        states = torch.zeros(self.history_horizon, self.state_dim)
        states[history_offset:].copy_(
            self._tensors["states"][history_start:current_stop]
        )
        past_actions = torch.zeros(max(self.history_horizon - 1, 0), self.action_dim)
        if history_count > 1:
            past_actions[history_offset:].copy_(
                self._tensors["commanded_actions"][history_start : current_stop - 1]
            )
        valid_mask = torch.zeros(self.history_horizon, dtype=torch.bool)
        valid_mask[history_offset:] = True
        past_action_mask = torch.zeros(
            max(self.history_horizon - 1, 0), dtype=torch.bool
        )
        if history_count > 1:
            past_action_mask[history_offset:] = True

        result: dict[str, torch.Tensor] = {
            "states": states,
            "past_actions": past_actions,
            "valid_mask": valid_mask,
            "past_action_mask": past_action_mask,
            "forecast_mask": torch.arange(self.forecast_horizon) < forecast_count,
            "episode_index": torch.tensor(
                int(self._episode_indices[episode]), dtype=torch.int64
            ),
            "episode_seed": torch.tensor(
                int(self._episode_seeds[episode]), dtype=torch.int64
            ),
            "decision_t": torch.tensor(decision_t, dtype=torch.int64),
        }
        forecast_fields = {
            "candidate_actions": ("commanded_actions", self.action_dim),
            "executed_actions": ("executed_actions", self.action_dim),
            "target_states": ("next_states", self.state_dim),
            "rewards": ("rewards", 1),
            "dones": ("dones", 1),
            "successes": ("successes", 1),
            "failures": ("failures", 1),
            "response_progress": ("response_progress", 1),
            "coordination_error": ("coordination_error", 1),
        }
        for output_name, (field, width) in forecast_fields.items():
            padded = torch.zeros(self.forecast_horizon, width)
            padded[:forecast_count].copy_(
                self._tensors[field][forecast_start:forecast_stop]
            )
            result[output_name] = padded
        if self.planning_discount is not None:
            current = episode_start + decision_t
            for name in (
                "returns_to_go",
                "episode_returns",
                "action_prior_weights",
            ):
                result[name] = self._tensors[name][current].clone()
            result["action_quality_weights"] = result[
                "action_prior_weights"
            ].clone()
        return result

    @property
    def nbytes(self) -> int:
        tensor_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in self._tensors.values()
        )
        return int(
            tensor_bytes
            + self._episode_offsets.nbytes
            + self._episode_seeds.nbytes
            + self._episode_indices.nbytes
            + self._sample_episode.nbytes
            + self._sample_decision.nbytes
        )

    def cache_payload(self) -> dict[str, object]:
        """Return a tensor-only payload compatible with safe ``torch.load``."""

        return {
            "format_version": self.CACHE_FORMAT_VERSION,
            "history_horizon": self.history_horizon,
            "forecast_horizon": self.forecast_horizon,
            "stride": self.stride,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "planning_discount": self.planning_discount,
            "paths": [str(path) for path in self.paths],
            "num_episodes": self.num_episodes,
            "episode_seeds": torch.from_numpy(self._episode_seeds.copy()),
            "episode_indices": torch.from_numpy(self._episode_indices.copy()),
            "episode_offsets": torch.from_numpy(self._episode_offsets.copy()),
            "sample_episode": torch.from_numpy(self._sample_episode.copy()),
            "sample_decision": torch.from_numpy(self._sample_decision.copy()),
            "tensors": self._tensors,
            "planning_metadata": getattr(self, "planning_metadata", None),
            "action_quality_metadata": getattr(
                self, "action_quality_metadata", None
            ),
        }

    @classmethod
    def from_cache_payload(
        cls, payload: Mapping[str, object]
    ) -> "InMemoryProprioSequenceDataset":
        if payload.get("format_version") != cls.CACHE_FORMAT_VERSION:
            raise ValueError("unsupported in-memory sequence cache format")
        required_tensors = (
            "episode_seeds",
            "episode_indices",
            "episode_offsets",
            "sample_episode",
            "sample_decision",
        )
        if any(not isinstance(payload.get(name), torch.Tensor) for name in required_tensors):
            raise ValueError("in-memory sequence cache arrays are incomplete")
        tensors = payload.get("tensors")
        if not isinstance(tensors, Mapping) or any(
            not isinstance(value, torch.Tensor) for value in tensors.values()
        ):
            raise ValueError("in-memory sequence cache tensors are incomplete")
        dataset = cls.__new__(cls)
        dataset.history_horizon = int(payload["history_horizon"])
        dataset.forecast_horizon = int(payload["forecast_horizon"])
        dataset.stride = int(payload["stride"])
        dataset.state_dim = int(payload["state_dim"])
        dataset.action_dim = int(payload["action_dim"])
        raw_discount = payload.get("planning_discount")
        dataset.planning_discount = (
            None if raw_discount is None else float(raw_discount)
        )
        dataset.paths = [Path(str(path)) for path in payload["paths"]]
        dataset.num_episodes = int(payload["num_episodes"])
        dataset._episode_seeds = payload["episode_seeds"].numpy()
        dataset._episode_indices = payload["episode_indices"].numpy()
        dataset._episode_offsets = payload["episode_offsets"].numpy()
        dataset._sample_episode = payload["sample_episode"].numpy()
        dataset._sample_decision = payload["sample_decision"].numpy()
        dataset._tensors = dict(tensors)
        planning_metadata = payload.get("planning_metadata")
        if planning_metadata is not None:
            dataset.planning_metadata = planning_metadata
        action_quality_metadata = payload.get("action_quality_metadata")
        if action_quality_metadata is not None:
            dataset.action_quality_metadata = action_quality_metadata
        return dataset

    def close(self) -> None:
        """Match the disk-backed dataset lifecycle API."""


def _discounted_returns(rewards: np.ndarray, discount: float) -> np.ndarray:
    result = np.empty_like(np.asarray(rewards, dtype=np.float32))
    running = 0.0
    for index in range(result.size - 1, -1, -1):
        running = float(rewards[index]) + discount * running
        result[index] = running
    return result


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
    "InMemoryProprioSequenceDataset",
    "PreloadProgressCallback",
    "ProprioSequenceDataset",
    "discover_episode_paths",
    "split_episode_paths",
]
