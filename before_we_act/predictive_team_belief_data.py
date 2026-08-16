"""Cached, paired data projection for predictive team-belief training."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from before_we_act.raw_team_signal_data import FUTURE_OFFSETS, TeamEpisode, TeamSampleRequest
from before_we_act.action_grounded_belief import ActionGroundedDataset
from before_we_act.temporal_history_data import (
    ACTION_HORIZON,
    EFFECTIVE_BATCH,
    HISTORY_STEPS,
    SAMPLES_PER_TASK,
    SIX_TASKS,
    TASK_TEXT,
    task_text_tensor,
)


TEAMMATE_ACTION_HORIZON = 16


class PredictiveTeamBeliefDataset(ActionGroundedDataset):
    """Legal runtime history plus privileged training-only N2 targets.

    The large frozen B0-H action context is read from a one-time cache.  Raw
    DINO vectors and robot targets continue to come from the N1 cache bound to
    the same 720 HDF5 hashes; no episode/frame identity enters the model path.
    """

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

    def __init__(self, cache_root: str | Path, action_context_root: str | Path) -> None:
        super().__init__(cache_root)
        self.action_context_root = Path(action_context_root)
        receipt_path = self.action_context_root / "cache_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("format_version") != "before-we-act.b3-n2-action-context-cache/1":
            raise ValueError("unsupported N2 action-context cache")
        if receipt.get("status") != "PASSED" or receipt.get("samples") != len(self):
            raise ValueError("N2 action-context cache is incomplete")
        self.action_context_receipt = receipt
        task_tokens = np.load(self.action_context_root / "task_tokens.npy")
        if task_tokens.shape != (len(SIX_TASKS), 384) or task_tokens.dtype != np.float32:
            raise ValueError("N2 cached task-token contract differs")
        self.task_tokens = torch.from_numpy(task_tokens.copy())
        self._contexts: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _task_contexts(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        if task not in self._contexts:
            decoded = np.load(
                self.action_context_root / f"{task}_decoded.npy", mmap_mode="r"
            )
            base = np.load(
                self.action_context_root / f"{task}_base_action.npy", mmap_mode="r"
            )
            expected_rows = sum(
                episode.length * 2 for episode in self.episodes if episode.task == task
            )
            if decoded.shape != (expected_rows, ACTION_HORIZON, 384):
                raise ValueError(f"N2 decoded cache shape differs for {task}: {decoded.shape}")
            if base.shape != (expected_rows, ACTION_HORIZON, 8):
                raise ValueError(f"N2 base-action cache shape differs for {task}: {base.shape}")
            if decoded.dtype != np.float16 or base.dtype != np.float16:
                raise ValueError("N2 action-context cache must be float16")
            self._contexts[task] = decoded, base
        return self._contexts[task]

    def __getitem__(self, request: TeamSampleRequest | tuple) -> dict:
        if not isinstance(request, TeamSampleRequest):
            request = TeamSampleRequest(*request)
        result = super().__getitem__(request)
        episode = self.episodes[request.episode_index]
        visual_np, qpos_np, action_np = self._task_arrays(episode.task)
        t = request.time_index
        absolute = episode.offset + t
        ego = request.arm
        teammate = 1 - ego

        first = max(0, t - (HISTORY_STEPS - 1))
        indices = np.arange(episode.offset + first, absolute + 1)
        offset = HISTORY_STEPS - len(indices)
        runtime_visual = torch.zeros(HISTORY_STEPS, 2, 1, 768, dtype=torch.float32)
        runtime_visual[offset:, :, 0] = torch.from_numpy(
            np.array(visual_np[indices][:, [0, ego + 1]], dtype=np.float32, copy=True)
        )
        runtime_mask = result["history_mask"][:, None, None].expand(-1, 2, 1).clone()
        reset_mask = torch.zeros(HISTORY_STEPS, dtype=torch.bool)
        reset_mask[offset] = True

        current_visual = torch.from_numpy(
            np.array(visual_np[absolute, [0, ego + 1, teammate + 1]], dtype=np.float32, copy=True)
        ).unsqueeze(1)
        future_visual = torch.zeros(len(FUTURE_OFFSETS), 3, 1, 768)
        future_visual_mask = torch.zeros(len(FUTURE_OFFSETS), 3, 1, dtype=torch.bool)
        teammate_delta = torch.zeros(len(FUTURE_OFFSETS), 9)
        teammate_now_raw = torch.from_numpy(
            np.array(qpos_np[absolute, teammate], dtype=np.float32, copy=True)
        )
        for anchor, future_offset in enumerate(FUTURE_OFFSETS):
            target_t = t + future_offset
            if target_t >= episode.length:
                continue
            target_absolute = episode.offset + target_t
            future_visual[anchor, :, 0] = torch.from_numpy(
                np.array(
                    visual_np[target_absolute, [0, ego + 1, teammate + 1]],
                    dtype=np.float32,
                    copy=True,
                )
            )
            future_visual_mask[anchor] = True
            future_teammate = torch.from_numpy(
                np.array(qpos_np[target_absolute, teammate], dtype=np.float32, copy=True)
            )
            teammate_delta[anchor] = (future_teammate - teammate_now_raw) / self.q_std

        agent_state = torch.stack(
            (
                (torch.from_numpy(np.array(qpos_np[absolute, ego], copy=True)) - self.q_mean)
                / self.q_std,
                (teammate_now_raw - self.q_mean) / self.q_std,
            )
        ).float()
        teammate_action = torch.zeros(TEAMMATE_ACTION_HORIZON, 8)
        teammate_action_mask = torch.zeros(TEAMMATE_ACTION_HORIZON, dtype=torch.bool)
        teammate_end = min(episode.offset + episode.length, absolute + TEAMMATE_ACTION_HORIZON)
        teammate_source = torch.from_numpy(
            np.array(action_np[absolute:teammate_end, teammate], dtype=np.float32, copy=True)
        )
        teammate_valid = len(teammate_source)
        teammate_action[:teammate_valid] = (teammate_source - self.a_mean) / self.a_std
        teammate_action_mask[:teammate_valid] = True

        ego_end = min(episode.offset + episode.length, absolute + ACTION_HORIZON)
        ego_source = torch.from_numpy(
            np.array(action_np[absolute:ego_end, ego], dtype=np.float32, copy=True)
        )
        ego_valid = len(ego_source)
        ego_action = torch.empty(ACTION_HORIZON, 8)
        normalized_ego = (ego_source - self.a_mean) / self.a_std
        ego_action[:ego_valid] = normalized_ego
        ego_action[ego_valid:] = normalized_ego[-1]
        ego_action_mask = torch.zeros(ACTION_HORIZON, dtype=torch.bool)
        ego_action_mask[:ego_valid] = True

        decoded_np, base_np = self._task_contexts(episode.task)
        context_index = 2 * absolute + ego
        task_bytes, task_mask = task_text_tensor(TASK_TEXT[episode.task])
        phase = float(t / max(episode.length - 1, 1))
        result.update(
            {
                "runtime_visual_tokens": runtime_visual,
                "runtime_visual_mask": runtime_mask,
                "episode_reset_mask": reset_mask,
                "task_bytes": task_bytes,
                "task_text_mask": task_mask,
                "task_token": self.task_tokens[episode.task_index].clone(),
                "decoded_action_hidden": torch.from_numpy(
                    np.array(decoded_np[context_index], dtype=np.float32, copy=True)
                ),
                "base_action": torch.from_numpy(
                    np.array(base_np[context_index], dtype=np.float32, copy=True)
                ),
                "teacher_current_visual_tokens": current_visual,
                "teacher_current_visual_mask": torch.ones(3, 1, dtype=torch.bool),
                "teacher_future_visual_tokens": future_visual,
                "teacher_future_visual_mask": future_visual_mask,
                "teacher_future_anchor_mask": result["future_mask"].clone(),
                "teacher_agent_state": agent_state,
                "teacher_agent_mask": torch.ones(2, dtype=torch.bool),
                "teacher_relative_agent_role": torch.tensor((0, 1), dtype=torch.long),
                "teammate_delta": teammate_delta,
                "teammate_action": teammate_action,
                "teammate_action_mask": teammate_action_mask,
                "action": ego_action,
                "action_mask": ego_action_mask,
                "pair_id": torch.tensor(request.episode_index * 1_000_000 + t),
                "phase_bin": torch.tensor(min(3, int(phase * 4)), dtype=torch.long),
            }
        )
        return result


class PairedSituationBatchSampler(Sampler[list[TeamSampleRequest]]):
    """Four paired ego/teammate situations per task and update."""

    def __init__(
        self,
        episodes: Sequence[TeamEpisode],
        split: Mapping[str, str],
        *,
        updates: int,
        data_seed: int,
        start_update: int = 0,
    ) -> None:
        if SAMPLES_PER_TASK != 8 or EFFECTIVE_BATCH != 48:
            raise ValueError("N2 paired sampler is frozen to 8 samples/task and batch 48")
        if not 0 <= start_update <= updates:
            raise ValueError("invalid N2 update interval")
        self.episodes = list(episodes)
        self.split = dict(split)
        self.updates = int(updates)
        self.data_seed = int(data_seed)
        self.start_update = int(start_update)
        self.by_task = {
            task: [
                index
                for index, episode in enumerate(self.episodes)
                if episode.task == task and self.split.get(episode.episode_key) == "train"
            ]
            for task in SIX_TASKS
        }
        if any(len(indices) != 96 for indices in self.by_task.values()):
            raise ValueError("N2 sampler expects 96 scenario-group train episodes/task")

    def __len__(self) -> int:
        return self.updates - self.start_update

    def requests_for_update(self, update: int) -> list[TeamSampleRequest]:
        if not 1 <= update <= self.updates:
            raise IndexError(update)
        rng = random.Random(self.data_seed + 1_000_003 * update)
        pairs: list[list[TeamSampleRequest]] = []
        for task in SIX_TASKS:
            used_situations: set[tuple[int, int]] = set()
            for _ in range(SAMPLES_PER_TASK // 2):
                # Sampling is with replacement across updates, but the four
                # situations inside one task/batch must be distinct.  If the
                # same episode/time is drawn twice, pair_id would occur four
                # times and the paired exchange target would be ambiguous.
                while True:
                    episode_index = self.by_task[task][
                        rng.randrange(len(self.by_task[task]))
                    ]
                    episode = self.episodes[episode_index]
                    time_index = rng.randrange(episode.length)
                    situation = (episode_index, time_index)
                    if situation not in used_situations:
                        used_situations.add(situation)
                        break
                pair = []
                for arm in (0, 1):
                    identity = f"{episode.episode_key}:{arm}:{time_index}:n2"
                    pair.append(
                        TeamSampleRequest(
                            episode_index,
                            arm,
                            time_index,
                            hashlib.sha256(identity.encode()).hexdigest(),
                            task,
                        )
                    )
                pairs.append(pair)
        rng.shuffle(pairs)
        requests = [row for pair in pairs for row in pair]
        counts = Counter(request.task for request in requests)
        if len(requests) != EFFECTIVE_BATCH or counts != Counter(
            {task: SAMPLES_PER_TASK for task in SIX_TASKS}
        ):
            raise AssertionError(f"N2 batch balance failure: {counts}")
        return requests

    def __iter__(self) -> Iterable[list[TeamSampleRequest]]:
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
            "format_version": "before-we-act.b3-n2-paired-cursor/1",
            "data_seed": self.data_seed,
            "completed_update": completed_update,
            "next_update": next_update if keys else None,
            "next_sample_keys": keys,
            "effective_batch": EFFECTIVE_BATCH,
            "samples_per_task": SAMPLES_PER_TASK,
            "paired_arms": True,
        }


__all__ = [
    "PredictiveTeamBeliefDataset",
    "PairedSituationBatchSampler",
    "TEAMMATE_ACTION_HORIZON",
]
