"""Strict local-input runtime shared by B0-H/B-core/CARE evaluators.

The runtime intentionally has no peer-state argument.  A single model object
is called with a batch of focal-arm streams, which proves weight sharing while
keeping each arm's camera/state/action history independent.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from before_we_act.temporal_history_data import task_text_tensor
from .config import (
    ACTION_DIM,
    ACTION_ENCODING,
    ACTION_HORIZON,
    GRIPPER_ENCODING,
    GRIPPER_NATIVE_RANGE,
    HISTORY_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    STATE_DIM,
    TASK_TEXT,
    validate_native_gripper_vector,
)
from .data import project_local_observation
from .preprocessing import decode_bicoord_jpeg_rgb, resize_rgb_batch


@dataclass
class _HistoryRow:
    visual: np.ndarray  # [2,768], shared head + focal wrist
    state: np.ndarray  # normalized [7]
    action: np.ndarray | None  # normalized action emitted after this observation


class AbsoluteChunkEnsemble:
    """ACT-style absolute-time ensemble without action clipping/rebasing."""

    def __init__(self, decay: float = 0.01):
        if decay < 0:
            raise ValueError("ensemble decay must be non-negative")
        self.decay = float(decay)
        self.step = 0
        self.values: dict[int, list[tuple[int, np.ndarray]]] = {0: [], 1: []}

    def reset(self) -> None:
        self.step = 0
        for rows in self.values.values():
            rows.clear()

    def add_and_plan(self, chunks: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        for arm in (0, 1):
            chunk = np.asarray(chunks[arm], dtype=np.float32)
            if chunk.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(chunk).all():
                raise ValueError("absolute action chunk must be finite [100,7]")
            history = self.values[arm]
            history.append((self.step, chunk.copy()))
            history[:] = [(born, row) for born, row in history if self.step - born < ACTION_HORIZON]
            plan: list[np.ndarray] = []
            for offset in range(ACTION_HORIZON):
                absolute = self.step + offset
                available = [(born, row[absolute - born]) for born, row in history if 0 <= absolute - born < ACTION_HORIZON]
                age = np.asarray([absolute - born for born, _row in available], np.float64)
                weights = np.exp(-self.decay * age); weights /= weights.sum()
                plan.append(np.sum(np.stack([row for _born, row in available]) * weights[:, None], axis=0).astype(np.float32))
            result[arm] = np.stack(plan)
        self.step += 1
        return result


def _as_frame(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"expected one frame, got {array.shape}")
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB frame, got {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.min() >= 0 and array.max() <= 1:
            array = np.rint(array * 255).astype(np.uint8)
        else:
            raise ValueError("runtime frame must be uint8 or float [0,1]")
    return np.ascontiguousarray(array)


class B0HRuntime:
    """Load one frozen B0-H checkpoint and expose a two-arm local API."""

    def __init__(self, model: torch.nn.Module, stats: Mapping[str, Any], *, device: torch.device):
        self.model = model.to(device).eval()
        self.device = device
        self._validate_stats_contract(stats)
        self.q_mean = np.asarray(stats["q_mean"], dtype=np.float32).reshape(STATE_DIM)
        self.q_std = np.asarray(stats["q_std"], dtype=np.float32).reshape(STATE_DIM)
        self.a_mean = np.asarray(stats["a_mean"], dtype=np.float32).reshape(ACTION_DIM)
        self.a_std = np.asarray(stats["a_std"], dtype=np.float32).reshape(ACTION_DIM)
        if not np.isfinite(self.q_mean).all() or not np.isfinite(self.q_std).all() or np.any(self.q_std <= 0):
            raise ValueError("invalid B0-H qpos normalization")
        if not np.isfinite(self.a_mean).all() or not np.isfinite(self.a_std).all() or np.any(self.a_std <= 0):
            raise ValueError("invalid B0-H action normalization")
        self.rows: dict[int, deque[_HistoryRow]] = {0: deque(maxlen=HISTORY_STEPS), 1: deque(maxlen=HISTORY_STEPS)}
        self.pending_actions: dict[int, np.ndarray | None] = {0: None, 1: None}
        self.ensemble = AbsoluteChunkEnsemble()
        # Range telemetry is intentionally kept separate from the native
        # command gate.  A regression head is unbounded in normalized space;
        # future rows of a 100-step chunk may therefore fall outside the
        # simulator population range even though they are not executed at the
        # current tick.  We record that fact, but never transform the values.
        self.last_prediction_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _validate_stats_contract(stats: Mapping[str, Any]) -> None:
        """Require the audited continuous native gripper contract.

        The range fields describe the source population; they are not runtime
        clipping bounds.  Policy outputs and executed commands remain untouched.
        """

        expected_metadata = {
            "action_encoding": ACTION_ENCODING,
            "gripper_encoding": GRIPPER_ENCODING,
            "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        }
        for key, expected in expected_metadata.items():
            observed = stats.get(key)
            if key == "gripper_native_range":
                try:
                    observed = list(map(float, observed))
                except (TypeError, ValueError):
                    observed = None
            if observed != expected:
                raise ValueError(
                    f"B0-H normalization contract differs at {key}: "
                    f"{observed!r} != {expected!r}"
                )
        ranges: dict[str, np.ndarray] = {}
        for name in ("q_min", "q_max", "a_min", "a_max"):
            value = np.asarray(stats.get(name), dtype=np.float32)
            if value.shape != (STATE_DIM,) or not np.isfinite(value).all():
                raise ValueError(f"invalid B0-H source range {name}")
            ranges[name] = value
        if np.any(ranges["q_min"] > ranges["q_max"]) or np.any(
            ranges["a_min"] > ranges["a_max"]
        ):
            raise ValueError("B0-H source normalization ranges are inverted")
        low, high = GRIPPER_NATIVE_RANGE
        if not (
            float(ranges["q_min"][-1]) == low
            and float(ranges["a_min"][-1]) == low
            and float(ranges["q_max"][-1]) == high
            and float(ranges["a_max"][-1]) == high
        ):
            raise ValueError("B0-H source gripper range is not the audited [0,1]")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        dino_model: str | Path,
        device: str | torch.device = "cuda:0",
    ) -> "B0HRuntime":
        path = Path(checkpoint).expanduser().resolve(strict=True)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(saved, Mapping):
            raise ValueError("B0-H checkpoint is not a mapping")
        fmt = saved.get("format") or saved.get("format_version")
        if fmt != "before-we-act.bicoord.dino-b0h/1":
            raise ValueError(f"unsupported B0-H checkpoint format: {fmt!r}")
        config = saved.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("B0-H checkpoint has no frozen config")
        expected = {
            "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
            "horizon": ACTION_HORIZON, "history_steps": HISTORY_STEPS,
            "d_model": 384, "enc_layers": 4, "dec_layers": 7,
            "roles": 4, "role_rank": 32, "history_layers": 2,
            "action_encoding": ACTION_ENCODING,
            "gripper_encoding": GRIPPER_ENCODING,
            "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
            "strict_dino_contract": True,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"B0-H checkpoint contract differs at {key}: {config.get(key)!r} != {value!r}")
        state = saved.get("model")
        if not isinstance(state, Mapping):
            raise ValueError("B0-H checkpoint has no model state")
        stats = saved.get("stats")
        if not isinstance(stats, Mapping):
            raise ValueError("B0-H checkpoint has no normalization stats")
        from before_we_act.temporal_history_policy import TemporalHistoryPolicy

        model = TemporalHistoryPolicy(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, variant="hidden_residual",
            horizon=ACTION_HORIZON, d_model=384, enc_layers=4, dec_layers=7,
            roles=4, role_rank=32, history_layers=2, dino_model=str(dino_model),
            image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH,
            strict_dino_contract=True,
        )
        model.load_state_dict(state, strict=True)
        return cls(model, stats, device=torch.device(device))

    def reset(self) -> None:
        for rows in self.rows.values():
            rows.clear()
        self.pending_actions = {0: None, 1: None}
        self.ensemble.reset()
        self.last_prediction_diagnostics = {}

    def _features(self, head: np.ndarray, wrists: Sequence[np.ndarray]) -> np.ndarray:
        frames = np.stack([head, *wrists], axis=0)
        resized = resize_rgb_batch(frames, IMAGE_HEIGHT, IMAGE_WIDTH).float().div_(255).to(self.device)
        with torch.inference_mode():
            tokens = self.model._raw_vision_tokens(resized)
        # [3,patches,768] -> [head,left,right], pooled only for history.  The
        # model receives images separately and computes the same frozen path.
        return tokens.float().mean(1).cpu().numpy()

    def _append(self, arm: int, head: np.ndarray, wrist: np.ndarray, state: np.ndarray) -> None:
        pooled = self._features(head, [wrist])[0:2]
        previous = self.pending_actions[int(arm)]
        self.rows[int(arm)].append(
            _HistoryRow(
                visual=pooled.astype(np.float32),
                state=((state - self.q_mean) / self.q_std).astype(np.float32),
                action=None if previous is None else ((previous - self.a_mean) / self.a_std).astype(np.float32),
            )
        )
        self.pending_actions[int(arm)] = None

    def _batch_inputs(self, arm: int, head: np.ndarray, wrist: np.ndarray, state: np.ndarray, task: str) -> dict[str, torch.Tensor]:
        self._append(arm, head, wrist, state)
        rows = list(self.rows[int(arm)])
        n = len(rows); offset = HISTORY_STEPS - n
        visual = np.zeros((HISTORY_STEPS, 2, 768), dtype=np.float32)
        qpos = np.zeros((HISTORY_STEPS, STATE_DIM), dtype=np.float32)
        action = np.zeros((HISTORY_STEPS, ACTION_DIM), dtype=np.float32)
        hmask = np.zeros(HISTORY_STEPS, dtype=bool); amask = np.zeros(HISTORY_STEPS, dtype=bool)
        for i, row in enumerate(rows, start=offset):
            visual[i] = row.visual; qpos[i] = row.state; hmask[i] = True
            if row.action is not None:
                action[i] = row.action; amask[i] = True
        task_bytes, text_mask = task_text_tensor(TASK_TEXT[task])
        return {
            "global_rgb": torch.from_numpy(resize_rgb_batch(head, IMAGE_HEIGHT, IMAGE_WIDTH).numpy()).unsqueeze(0).to(self.device).float().div_(255),
            "local_rgb": torch.from_numpy(resize_rgb_batch(wrist, IMAGE_HEIGHT, IMAGE_WIDTH).numpy()).unsqueeze(0).to(self.device).float().div_(255),
            "history_visual_raw": torch.from_numpy(visual).unsqueeze(0).to(self.device),
            "history_qpos": torch.from_numpy(qpos).unsqueeze(0).to(self.device),
            "history_action": torch.from_numpy(action).unsqueeze(0).to(self.device),
            "history_mask": torch.from_numpy(hmask).unsqueeze(0).to(self.device),
            "action_history_mask": torch.from_numpy(amask).unsqueeze(0).to(self.device),
            "task_bytes": task_bytes.unsqueeze(0).to(self.device),
            "task_text_mask": text_mask.unsqueeze(0).to(self.device),
            "episode_reset": torch.tensor([n == 1], dtype=torch.bool, device=self.device),
        }

    @torch.inference_mode()
    def act(self, observation: Mapping[str, Any], task: str) -> dict[int, np.ndarray]:
        if task not in TASK_TEXT:
            raise ValueError(f"unknown BiCoord task: {task}")
        local = [project_local_observation(observation, arm) for arm in (0, 1)]
        inputs = [self._batch_inputs(arm, _as_frame(item["head_rgb"]), _as_frame(item["wrist_rgb"]), np.asarray(item["state"], np.float32), task) for arm, item in enumerate(local)]
        merged = {key: torch.cat([row[key] for row in inputs], dim=0) for key in inputs[0]}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            prediction, _mu, _logvar = self.model(**merged)
        values = prediction.float().cpu().numpy() * self.a_std[None, None, :] + self.a_mean[None, None, :]
        if values.shape != (2, ACTION_HORIZON, ACTION_DIM) or not np.isfinite(values).all():
            raise ValueError("B0-H predicted action chunk must be finite [2,100,7]")
        raw_gripper = values[..., -1]
        prediction_oob = int(np.count_nonzero(
            (raw_gripper < GRIPPER_NATIVE_RANGE[0])
            | (raw_gripper > GRIPPER_NATIVE_RANGE[1])
        ))
        actions = self.ensemble.add_and_plan({arm: values[arm].astype(np.float32) for arm in (0, 1)})
        executed = np.stack((actions[0][0], actions[1][0])).astype(np.float32)
        # Only the row sent to the native controller is a command boundary.
        # Keep this fail-closed check (and do not clip): an out-of-range
        # current action is a real adapter/model error, while an out-of-range
        # tail is telemetry for a future tick.
        validate_native_gripper_vector(executed, context="B0-H executed action")
        plan_gripper = np.stack((actions[0][..., -1], actions[1][..., -1]))
        plan_oob = int(np.count_nonzero(
            (plan_gripper < GRIPPER_NATIVE_RANGE[0])
            | (plan_gripper > GRIPPER_NATIVE_RANGE[1])
        ))
        self.last_prediction_diagnostics = {
            "prediction_gripper_oob_count": prediction_oob,
            "prediction_gripper_min": float(np.min(raw_gripper)),
            "prediction_gripper_max": float(np.max(raw_gripper)),
            "ensemble_plan_gripper_oob_count": plan_oob,
            "ensemble_plan_gripper_min": float(np.min(plan_gripper)),
            "ensemble_plan_gripper_max": float(np.max(plan_gripper)),
            "executed_gripper_oob_count": 0,
            "policy_output_clipping": False,
        }
        # The first row is the command consumed at this control tick.  Keep
        # the full chunk for temporal ensembling in the caller if desired.
        for arm in (0, 1):
            self.pending_actions[arm] = actions[arm][0].copy()
        return actions

    def action_trace(self, actions: Mapping[int, np.ndarray]) -> str:
        digest = hashlib.sha256()
        for arm in (0, 1):
            digest.update(np.asarray(actions[arm], dtype=np.float32).tobytes())
        return digest.hexdigest()


def checkpoint_candidates(run: str | Path, *, stages: Sequence[str]) -> list[Path]:
    root = Path(run)
    found: list[Path] = []
    for stage in stages:
        base = root / "artifacts" / stage
        for name in ("final.pt", "checkpoint_latest.pt", "latest.pt"):
            path = base / name
            if path.is_file():
                found.append(path)
        found.extend(sorted(base.glob("**/final.pt")))
        found.extend(sorted(base.glob("**/checkpoint_latest.pt")))
    # Preserve deterministic newest candidate order but never silently choose
    # between two completed checkpoints.
    unique = list(dict.fromkeys(path.resolve() for path in found))
    return unique


__all__ = ["AbsoluteChunkEnsemble", "B0HRuntime", "checkpoint_candidates"]
