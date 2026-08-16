"""Frozen components for action-grounded belief experiments.

R1 deliberately keeps the Step-2 B0-H policy and the old N1 representation
read-only.  This module supplies the scenario-group split, exact sample cursor,
frozen feature extraction, and matched action heads used by the fair R1-1 gate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Sampler

from before_we_act.temporal_history_policy import TemporalHistoryPolicy
from before_we_act.raw_team_signal_data import (
    ACTION_PROBE_HORIZON,
    FUTURE_OFFSETS,
    TeamEpisode,
    TeamSampleRequest,
    RawTeamSignalDataset,
)
from before_we_act.raw_team_signal_model import MatchedActionProbe, RawTeamSignalEncoder
from before_we_act.temporal_history_data import (
    EFFECTIVE_BATCH,
    SAMPLES_PER_TASK,
    SIX_TASKS,
    TASK_TEXT,
    task_text_tensor,
)


BELIEF_SEEDS = (20260815, 20260816, 20260817)
BELIEF_DATA_SEED = 20260815
BELIEF_TOKEN_CAPACITY = 16
BELIEF_MAX_UPDATES = 80_000
BELIEF_MIN_UPDATES = 25_000
BELIEF_EARLIEST_PLATFORM = 35_000
BELIEF_EVAL_EVERY = 5_000
BELIEF_LR_DROP = 20_000
ACTION_GROUNDED_CONDITIONS = (
    "h",
    "b_only",
    "h_b",
    "h_b_shuffle",
    "h_matched_capacity",
    "h_b_row",
    "h_b_phase",
    "time",
)
PRIVILEGED_ORACLE_CONDITIONS = (
    "h",
    "h_oracle",
    "h_oracle_shuffle",
    "h_matched_capacity",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_split(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != "before-we-act.b3-n1-r1-scenario-split/1":
        raise ValueError("unsupported R1 scenario split")
    rows = payload.get("episodes", [])
    if len(rows) != 720:
        raise ValueError("R1 scenario split must bind all 720 episodes")
    counts = Counter((row["task"], row["split"]) for row in rows)
    expected = {
        (task, split): count
        for task in SIX_TASKS
        for split, count in (("train", 96), ("validation", 12), ("test", 12))
    }
    if counts != Counter(expected):
        raise ValueError(f"R1 split counts differ: {counts}")
    return payload


def split_by_episode_key(payload: Mapping) -> dict[str, str]:
    return {str(row["episode_key"]): str(row["split"]) for row in payload["episodes"]}


class ActionGroundedBatchSampler(Sampler[list[TeamSampleRequest]]):
    """Six-task-balanced cursor restricted to scenario-group train episodes."""

    def __init__(
        self,
        episodes: Sequence[TeamEpisode],
        split: Mapping[str, str],
        *,
        updates: int,
        data_seed: int,
        start_update: int = 0,
    ) -> None:
        if not 0 <= start_update <= updates:
            raise ValueError("invalid R1 update interval")
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
            raise ValueError("R1 sampler expects 96 scenario-group train episodes/task")

    def __len__(self) -> int:
        return self.updates - self.start_update

    def requests_for_update(self, update: int) -> list[TeamSampleRequest]:
        if not 1 <= update <= self.updates:
            raise IndexError(update)
        rng = random.Random(self.data_seed + 1_000_003 * update)
        requests: list[TeamSampleRequest] = []
        for task in SIX_TASKS:
            candidates = self.by_task[task]
            for _ in range(SAMPLES_PER_TASK):
                episode_index = candidates[rng.randrange(len(candidates))]
                episode = self.episodes[episode_index]
                arm = rng.randrange(2)
                time_index = rng.randrange(episode.length)
                identity = f"{episode.episode_key}:{arm}:{time_index}:r1"
                requests.append(
                    TeamSampleRequest(
                        episode_index=episode_index,
                        arm=arm,
                        time_index=time_index,
                        sample_key=hashlib.sha256(identity.encode()).hexdigest(),
                        task=task,
                    )
                )
        rng.shuffle(requests)
        counts = Counter(item.task for item in requests)
        if len(requests) != EFFECTIVE_BATCH or counts != Counter(
            {task: SAMPLES_PER_TASK for task in SIX_TASKS}
        ):
            raise AssertionError(f"R1 batch balance failure: {counts}")
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
            "format_version": "before-we-act.b3-n1-r1-cursor/1",
            "data_seed": self.data_seed,
            "completed_update": completed_update,
            "next_update": next_update if keys else None,
            "next_sample_keys": keys,
            "effective_batch": EFFECTIVE_BATCH,
            "samples_per_task": SAMPLES_PER_TASK,
        }


def fixed_requests(
    episodes: Sequence[TeamEpisode],
    split: Mapping[str, str],
    name: str,
) -> list[TeamSampleRequest]:
    if name not in {"validation", "test"}:
        raise ValueError(name)
    requests: list[TeamSampleRequest] = []
    for index, episode in enumerate(episodes):
        if split.get(episode.episode_key) != name:
            continue
        times = np.unique(
            np.linspace(
                0,
                episode.length - 1,
                num=min(16, episode.length),
                dtype=np.int64,
            )
        )
        for arm in (0, 1):
            for time_index in times.tolist():
                requests.append(
                    TeamSampleRequest(
                        episode_index=index,
                        arm=arm,
                        time_index=time_index,
                        sample_key=f"{episode.episode_key}:{arm}:{time_index}:r1-{name}",
                        task=episode.task,
                    )
                )
    expected = 6 * 12 * 2 * 16
    if len(requests) != expected:
        raise ValueError(f"R1 {name} requests differ: {len(requests)} != {expected}")
    return requests


class ActionGroundedDataset(RawTeamSignalDataset):
    """N1 cache projection augmented with explicitly privileged teammate actions."""

    ORACLE_ONLY_FIELDS = frozenset(
        {
            "teammate_qpos",
            "previous_teammate_qpos",
            "teammate_delta",
            "future_mask",
            "oracle_teammate_action",
            "oracle_teammate_action_mask",
        }
    )

    def __getitem__(self, request: TeamSampleRequest | tuple) -> dict:
        if not isinstance(request, TeamSampleRequest):
            request = TeamSampleRequest(*request)
        result = super().__getitem__(request)
        episode = self.episodes[request.episode_index]
        _, _, action_np = self._task_arrays(episode.task)
        absolute = episode.offset + request.time_index
        end = min(episode.offset + episode.length, absolute + ACTION_PROBE_HORIZON)
        teammate = 1 - request.arm
        source = torch.from_numpy(
            np.array(action_np[absolute:end, teammate], dtype=np.float32, copy=True)
        )
        valid = len(source)
        action = torch.zeros(ACTION_PROBE_HORIZON, 8, dtype=torch.float32)
        mask = torch.zeros(ACTION_PROBE_HORIZON, dtype=torch.bool)
        action[:valid] = (source - self.a_mean) / self.a_std
        mask[:valid] = True
        result["oracle_teammate_action"] = action
        result["oracle_teammate_action_mask"] = mask
        return result


def _constructor_config(payload: Mapping) -> dict:
    config = payload.get("config", {})
    return {
        "state_dim": int(config.get("state_dim", 9)),
        "action_dim": int(config.get("action_dim", 8)),
        "variant": "hidden_residual",
        "horizon": int(config.get("horizon", 100)),
        "d_model": int(config.get("d_model", 384)),
        "enc_layers": int(config.get("enc_layers", 4)),
        "dec_layers": int(config.get("dec_layers", 7)),
        "roles": int(config.get("roles", 4)),
        "role_rank": int(config.get("role_rank", 32)),
        "history_layers": int(config.get("history_layers", 2)),
        "dino_model": str(config["dino_model"]),
    }


@dataclass(frozen=True)
class FrozenBeliefFeatures:
    h: torch.Tensor
    belief: torch.Tensor
    history: torch.Tensor | None = None


class FrozenBeliefBackbones(nn.Module):
    """Read-only temporal history and full-token team-signal extractor."""

    def __init__(
        self,
        *,
        temporal_checkpoint: str | Path,
        signal_checkpoint: str | Path,
        visual_mean: torch.Tensor,
        visual_std: torch.Tensor,
    ) -> None:
        super().__init__()
        temporal_payload = torch.load(temporal_checkpoint, map_location="cpu", weights_only=False)
        self.temporal_policy = TemporalHistoryPolicy(**_constructor_config(temporal_payload))
        self.temporal_policy.load_state_dict(temporal_payload["model"], strict=True)
        signal_payload = torch.load(signal_checkpoint, map_location="cpu", weights_only=False)
        model_config = dict(signal_payload["model_config"])
        model_config.pop("capacities", None)
        self.signal_encoder = RawTeamSignalEncoder(**model_config)
        self.signal_encoder.load_state_dict(signal_payload["real_model"], strict=True)
        self.register_buffer("visual_mean", visual_mean.float().clone(), persistent=False)
        self.register_buffer("visual_std", visual_std.float().clone(), persistent=False)
        task_values = [task_text_tensor(TASK_TEXT[task]) for task in SIX_TASKS]
        self.register_buffer(
            "task_bytes", torch.stack([row[0] for row in task_values]), persistent=False
        )
        self.register_buffer(
            "task_masks", torch.stack([row[1] for row in task_values]), persistent=False
        )
        self.eval().requires_grad_(False)

    def _raw_visual(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        arm = batch["agent_slot"].long()
        view = torch.stack((torch.zeros_like(arm), arm + 1), dim=1)
        mean = self.visual_mean[view].unsqueeze(1)
        std = self.visual_std[view].unsqueeze(1)
        return batch["history_visual"] * std + mean

    @torch.no_grad()
    def forward(self, batch: Mapping[str, torch.Tensor]) -> FrozenBeliefFeatures:
        self.eval()
        signal_inputs = {key: batch[key] for key in RawTeamSignalDataset.RUNTIME_FIELDS}
        signal_output = self.signal_encoder(**signal_inputs)
        raw_visual = self._raw_visual(batch)
        task_index = batch["task_index"].long()
        history, h, _ = self.temporal_policy._encode_history(
            raw_visual,
            raw_visual[:, -1],
            batch["history_qpos"],
            batch["history_action"],
            batch["history_mask"],
            batch["action_history_mask"],
            self.task_bytes[task_index],
            self.task_masks[task_index],
            batch["time_index"].eq(0),
        )
        return FrozenBeliefFeatures(
            h=h,
            belief=signal_output.capacities[BELIEF_TOKEN_CAPACITY].tokens,
            history=history,
        )


class CrossAttentionActionHead(nn.Module):
    def __init__(self, d_model: int = 384, tokens: int = BELIEF_TOKEN_CAPACITY) -> None:
        super().__init__()
        self.tokens = int(tokens)
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(
            d_model, 8, dropout=0.1, batch_first=True
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.probe = MatchedActionProbe(d_model)

    def forward(self, h: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1:] != (self.tokens, h.shape[-1]):
            raise ValueError(f"R1 full-token contract differs: {tokens.shape}")
        query = self.query_norm(h).unsqueeze(1)
        memory = self.memory_norm(tokens)
        attended = self.cross(query, memory, memory, need_weights=False)[0].squeeze(1)
        return self.probe(self.output_norm(h + attended))


class MatchedCapacityActionHead(CrossAttentionActionHead):
    def __init__(self, d_model: int = 384, tokens: int = BELIEF_TOKEN_CAPACITY) -> None:
        super().__init__(d_model=d_model, tokens=tokens)
        # Fixed non-informative memory keeps the trainable parameter count
        # exactly equal to H+B while withholding B.
        self.register_buffer(
            "null_tokens", torch.zeros(1, tokens, d_model), persistent=True
        )

    def forward(self, h: torch.Tensor, ignored: torch.Tensor) -> torch.Tensor:
        tokens = self.null_tokens.expand(h.shape[0], -1, -1)
        return super().forward(h, tokens)


class BeliefOnlyActionHead(nn.Module):
    def __init__(self, d_model: int = 384, tokens: int = BELIEF_TOKEN_CAPACITY) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, 8, dropout=0.1, batch_first=True)
        self.probe = MatchedActionProbe(d_model)
        self.tokens = int(tokens)

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        if belief.shape[1] != self.tokens:
            raise ValueError("R1 belief-only token count differs")
        query = self.query.expand(belief.shape[0], -1, -1)
        value = self.cross(query, self.norm(belief), self.norm(belief), need_weights=False)[0]
        return self.probe(value.squeeze(1))


class ActionGroundedProbeSet(nn.Module):
    """Independent, matched heads for all pre-registered R1-1 conditions."""

    def __init__(self, d_model: int = 384) -> None:
        super().__init__()
        self.h = MatchedActionProbe(d_model)
        self.b_only = BeliefOnlyActionHead(d_model)
        self.h_b = CrossAttentionActionHead(d_model)
        self.h_b_shuffle = CrossAttentionActionHead(d_model)
        self.h_matched_capacity = MatchedCapacityActionHead(d_model)
        self.h_b_row = CrossAttentionActionHead(d_model)
        self.h_b_phase = CrossAttentionActionHead(d_model)
        self.time_projection = nn.Sequential(
            nn.Linear(11, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.time = MatchedActionProbe(d_model)

    def time_feature(self, task_index: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        task = F.one_hot(task_index, num_classes=6).to(phase.dtype)
        phase_feature = torch.stack(
            (
                phase,
                phase.square(),
                torch.sin(math.pi * phase),
                torch.cos(math.pi * phase),
                torch.ones_like(phase),
            ),
            dim=-1,
        )
        return self.time_projection(torch.cat((task, phase_feature), dim=-1))

    def parameter_counts(self) -> dict[str, int]:
        names = ACTION_GROUNDED_CONDITIONS
        result = {
            name: sum(parameter.numel() for parameter in getattr(self, name).parameters())
            for name in names
        }
        result["time"] += sum(
            parameter.numel() for parameter in self.time_projection.parameters()
        )
        return result


class OracleCrossActionHead(nn.Module):
    """Encode privileged teammate state/action tokens before matched cross-attention."""

    def __init__(self, d_model: int = 384) -> None:
        super().__init__()
        self.qpos = nn.Linear(9, d_model)
        self.delta = nn.Linear(9, d_model)
        self.action = nn.Linear(8, d_model)
        self.type_embedding = nn.Parameter(
            torch.randn(1, 2 + len(FUTURE_OFFSETS) + ACTION_PROBE_HORIZON, d_model)
            * 0.02
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, 8, dropout=0.1, batch_first=True)
        self.output_norm = nn.LayerNorm(d_model)
        self.probe = MatchedActionProbe(d_model)

    def encode(self, oracle: Mapping[str, torch.Tensor], *, zero: bool) -> tuple[torch.Tensor, torch.Tensor]:
        current = oracle["teammate_qpos"]
        previous = oracle["previous_teammate_qpos"]
        delta = oracle["teammate_delta"]
        action = oracle["oracle_teammate_action"]
        if zero:
            current = torch.zeros_like(current)
            previous = torch.zeros_like(previous)
            delta = torch.zeros_like(delta)
            action = torch.zeros_like(action)
        tokens = torch.cat(
            (
                self.qpos(current).unsqueeze(1),
                self.qpos(previous).unsqueeze(1),
                self.delta(delta),
                self.action(action),
            ),
            dim=1,
        )
        tokens = tokens + self.type_embedding.to(tokens.dtype)
        valid = torch.cat(
            (
                torch.ones(current.shape[0], 2, dtype=torch.bool, device=current.device),
                oracle["future_mask"].bool(),
                oracle["oracle_teammate_action_mask"].bool(),
            ),
            dim=1,
        )
        return tokens, valid

    def forward(
        self,
        h: torch.Tensor,
        oracle: Mapping[str, torch.Tensor],
        *,
        zero: bool = False,
    ) -> torch.Tensor:
        tokens, valid = self.encode(oracle, zero=zero)
        query = self.query_norm(h).unsqueeze(1)
        memory = self.memory_norm(tokens)
        attended = self.cross(
            query,
            memory,
            memory,
            key_padding_mask=~valid,
            need_weights=False,
        )[0].squeeze(1)
        return self.probe(self.output_norm(h + attended))


class PrivilegedOracleProbeSet(nn.Module):
    def __init__(self, d_model: int = 384) -> None:
        super().__init__()
        self.h = MatchedActionProbe(d_model)
        self.h_oracle = OracleCrossActionHead(d_model)
        self.h_oracle_shuffle = OracleCrossActionHead(d_model)
        self.h_matched_capacity = OracleCrossActionHead(d_model)

    def parameter_counts(self) -> dict[str, int]:
        return {
            name: sum(parameter.numel() for parameter in getattr(self, name).parameters())
            for name in PRIVILEGED_ORACLE_CONDITIONS
        }


def oracle_mapping(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: batch[key] for key in ActionGroundedDataset.ORACLE_ONLY_FIELDS}


def permute_mapping(
    value: Mapping[str, torch.Tensor], permutation: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {key: item[permutation] for key, item in value.items()}


def oracle_predictions(
    probes: PrivilegedOracleProbeSet,
    frozen: FrozenBeliefFeatures,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    oracle = oracle_mapping(batch)
    permutation = deterministic_permutations(batch)["shuffle"]
    return {
        "h": probes.h(frozen.h),
        "h_oracle": probes.h_oracle(frozen.h, oracle),
        "h_oracle_shuffle": probes.h_oracle_shuffle(
            frozen.h, permute_mapping(oracle, permutation)
        ),
        "h_matched_capacity": probes.h_matched_capacity(
            frozen.h, oracle, zero=True
        ),
    }


def all_oracle_conditions_platform(metrics: Sequence[Mapping]) -> bool:
    if len(metrics) < 4 or int(metrics[-1]["update"]) < BELIEF_EARLIEST_PLATFORM:
        return False
    for condition in PRIVILEGED_ORACLE_CONDITIONS:
        scores = [
            float(row["validation"]["macro"][condition]) for row in metrics[-4:]
        ]
        improvements = [
            (previous - current) / max(abs(previous), 1e-12)
            for previous, current in zip(scores, scores[1:])
        ]
        if any(value >= 0.01 for value in improvements):
            return False
    return True


def deterministic_permutations(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return no-fixed-point row, task/phase, and nearest-phase permutations."""

    count = int(batch["task_index"].shape[0])
    device = batch["task_index"].device
    row = torch.roll(torch.arange(count, device=device), 1)
    task = batch["task_index"].detach().cpu().tolist()
    phase_bin = batch["phase_bin"].detach().cpu().tolist()
    episode = batch["episode_label"].detach().cpu().tolist()
    phase_value = batch["phase"].detach().cpu().tolist()

    def choose(index: int, strict_bin: bool) -> int:
        candidates = [
            other
            for other in range(count)
            if other != index
            and task[other] == task[index]
            and episode[other] != episode[index]
            and (not strict_bin or phase_bin[other] == phase_bin[index])
        ]
        if not candidates and strict_bin:
            return choose(index, False)
        if not candidates:
            candidates = [other for other in range(count) if other != index]
        return min(candidates, key=lambda other: (abs(phase_value[other] - phase_value[index]), other))

    matched = torch.tensor([choose(index, True) for index in range(count)], device=device)
    phase = torch.tensor([choose(index, False) for index in range(count)], device=device)
    return {"row": row, "shuffle": matched, "phase": phase}


def predictions(
    probes: ActionGroundedProbeSet,
    frozen: FrozenBeliefFeatures,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    permutation = deterministic_permutations(batch)
    time = probes.time_feature(batch["task_index"], batch["phase"])
    return {
        "h": probes.h(frozen.h),
        "b_only": probes.b_only(frozen.belief),
        "h_b": probes.h_b(frozen.h, frozen.belief),
        "h_b_shuffle": probes.h_b_shuffle(
            frozen.h, frozen.belief[permutation["shuffle"]]
        ),
        "h_matched_capacity": probes.h_matched_capacity(frozen.h, frozen.belief),
        "h_b_row": probes.h_b_row(frozen.h, frozen.belief[permutation["row"]]),
        "h_b_phase": probes.h_b_phase(frozen.h, frozen.belief[permutation["phase"]]),
        "time": probes.time(time),
    }


def action_sample_mse(
    prediction: torch.Tensor, batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    squared = (prediction - batch["action"]).square().mean(-1)
    mask = batch["action_mask"].to(squared.dtype)
    return (squared * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def condition_platform(metrics: Sequence[Mapping], condition: str) -> bool:
    if len(metrics) < 4 or int(metrics[-1]["update"]) < BELIEF_EARLIEST_PLATFORM:
        return False
    scores = [float(row["validation"]["macro"][condition]) for row in metrics[-4:]]
    improvements = [
        (previous - current) / max(abs(previous), 1e-12)
        for previous, current in zip(scores, scores[1:])
    ]
    return all(value < 0.01 for value in improvements)


def all_conditions_platform(metrics: Sequence[Mapping]) -> bool:
    return all(condition_platform(metrics, condition) for condition in ACTION_GROUNDED_CONDITIONS)


__all__ = [
    "FrozenBeliefBackbones",
    "FrozenBeliefFeatures",
    "ActionGroundedBatchSampler",
    "ActionGroundedProbeSet",
    "ActionGroundedDataset",
    "PrivilegedOracleProbeSet",
    "ACTION_GROUNDED_CONDITIONS",
    "PRIVILEGED_ORACLE_CONDITIONS",
    "BELIEF_DATA_SEED",
    "BELIEF_EARLIEST_PLATFORM",
    "BELIEF_EVAL_EVERY",
    "BELIEF_LR_DROP",
    "BELIEF_MAX_UPDATES",
    "BELIEF_MIN_UPDATES",
    "BELIEF_SEEDS",
    "action_sample_mse",
    "all_conditions_platform",
    "all_oracle_conditions_platform",
    "canonical_sha256",
    "condition_platform",
    "deterministic_permutations",
    "fixed_requests",
    "load_split",
    "oracle_predictions",
    "predictions",
    "split_by_episode_key",
]
