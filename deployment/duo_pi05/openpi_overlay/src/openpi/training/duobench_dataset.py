from __future__ import annotations

import bisect
import os
from pathlib import Path
from typing import SupportsIndex

import numpy as np

TASKS = ("ball_maze", "bin_sort", "block_balance", "carry_pot", "hinge_chest", "join_blocks", "pour_marbles", "spring_door", "transfer_cube", "transfer_gate", "transfer_reorient")
PROMPTS = {
    "ball_maze": "pick up the board and tilt it so the ball roles onto the red square",
    "bin_sort": "use the left arm to place the white cube in the white bowl; use the right arm to place the black cube in the black bowl",
    "block_balance": "place the beam on the cube and then place the other blocks on the beam simultaneously using one arm for each cube",
    "carry_pot": "use two arms to carry the pot at the handle on the stove",
    "hinge_chest": "open the box with the right arm and place the cube inside the box with the left arm",
    "join_blocks": "join the two blocks using the peg on the left block and join the free socket of the right block with the peg on the wall",
    "pour_marbles": "grasp and lift both cups, then pour the marbles from one cup into the other and place the cups back to their original location inside the green square",
    "spring_door": "use the left arm to open the microwave door, then use the right arm to place the box inside the microwave, and close the door again",
    "transfer_cube": "grasp the white cube with the right arm, hand it over to the left arm and place it in the white bowl with the left arm",
    "transfer_gate": "use the right arm to pick up the white box, and hand it over to the left arm through the hoop, then place it on the green mat with the left arm",
    "transfer_reorient": "grasp the block with the right arm, hand it over to the left arm such that the left arm can easily insert the piece later, then insert the block into the socket with the left arm",
}


class DuoBenchDataset:
    """Eleven equal-probability virtual lanes over all unique arm-local samples."""

    def __init__(self, root: str | Path = "/workspace/runs/pi05_duo/prepared", action_horizon: int = 16):
        self.root = Path(os.environ.get("OPENPI_DUOBENCH_ROOT", root))
        if action_horizon != 16: raise ValueError("upstream pi0.5 DuoBench horizon is fixed at 16")
        self.action_horizon = action_horizon
        self._open_arrays()

    def _open_arrays(self) -> None:
        self.data, self.streams, self.cumulative = [], [[] for _ in TASKS], [[] for _ in TASKS]
        unique = episodes = 0
        for task_id, task in enumerate(TASKS):
            arrays = {name: np.load(self.root / task / f"{name}.npy", mmap_mode="r") for name in ("state", "action", "head", "left", "right", "episodes")}
            n = len(arrays["episodes"])
            if arrays["state"].shape != (n, 16) or arrays["action"].shape != (n, 16): raise RuntimeError(f"{task}: numeric schema drift")
            if any(arrays[name].shape != (n, 224, 224, 3) or arrays[name].dtype != np.uint8 for name in ("head", "left", "right")): raise RuntimeError(f"{task}: RGB schema drift")
            starts = np.flatnonzero(np.r_[True, arrays["episodes"][1:] != arrays["episodes"][:-1]])
            ends = np.r_[starts[1:], n]
            if len(starts) != 50: raise RuntimeError(f"{task}: expected all 50 episodes")
            self.data.append(arrays); episodes += len(starts)
            for start, end in zip(starts, ends, strict=True):
                if end - start < 2: raise RuntimeError(f"{task}: episode has no causal pair")
                for arm in (0, 1):
                    count = int(end - start - 1)
                    self.streams[task_id].append((int(start), int(end), arm))
                    self.cumulative[task_id].append((self.cumulative[task_id][-1] if self.cumulative[task_id] else 0) + count)
                    unique += count
        if episodes != 550 or unique != 570876: raise RuntimeError(f"DuoBench corpus drift episodes={episodes}, samples={unique}")
        self.unique_samples = unique
        self.lane_length = max(values[-1] for values in self.cumulative)

    def __getstate__(self) -> dict:
        """Keep spawned DataLoader workers from serializing 121 GB of mmap data."""
        return {"root": str(self.root), "action_horizon": self.action_horizon}

    def __setstate__(self, state: dict) -> None:
        self.root = Path(state["root"])
        self.action_horizon = int(state["action_horizon"])
        self._open_arrays()

    def __len__(self) -> int: return self.lane_length * len(TASKS)

    def __getitem__(self, index: SupportsIndex) -> dict:
        flat = index.__index__(); flat = flat + len(self) if flat < 0 else flat
        if flat < 0 or flat >= len(self): raise IndexError(flat)
        task_id = flat % len(TASKS)
        local = (flat // len(TASKS)) % self.cumulative[task_id][-1]
        stream_id = bisect.bisect_right(self.cumulative[task_id], local)
        lane_start = self.cumulative[task_id][stream_id - 1] if stream_id else 0
        episode_start, episode_end, arm = self.streams[task_id][stream_id]
        row = episode_start + local - lane_start
        positions = np.minimum(np.arange(row + 1, row + 17), episode_end - 1)
        arrays = self.data[task_id]
        state = arrays["state"].reshape(-1, 2, 8)
        action = arrays["action"].reshape(-1, 2, 8)
        return {
            "observation/head": np.asarray(arrays["head"][row], np.uint8),
            "observation/wrist": np.asarray(arrays["left" if arm == 0 else "right"][row], np.uint8),
            "observation/state": np.asarray(state[row, arm], np.float32),
            "actions": np.asarray(action[positions, arm], np.float32),
            "prompt": PROMPTS[TASKS[task_id]],
        }
