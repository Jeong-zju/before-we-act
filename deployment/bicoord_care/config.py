"""Frozen configuration and data contracts for the BiCoord CARE adapter.

The upstream BiCoord files contain two robot streams in every episode.  CARE
uses one shared policy instance for both streams, with a seven dimensional
local command (six joints and one continuous absolute gripper drive target).
This module is deliberately boring: all values which affect tensor shapes,
sampling, image processing, or
the validation horizon live here so that training and rollout cannot silently
drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Final, Mapping


# Dataset identity.  Do not use a moving ``main``/``latest`` revision for a
# formal run: changing one demonstration would invalidate every checkpoint.
DATASET_REPO_ID: Final[str] = "GradiusTwinbee/BiCoord"
DATASET_REVISION: Final[str] = "df8bd41f21ed0da3a08c4d2bf6614e6cb56a5274"
DATASET_COMMIT: Final[str] = DATASET_REVISION


# The benchmark has 18 paired tasks.  Keep this order stable; task IDs are
# persisted in manifests, sampler receipts, and checkpoint metadata.
TASKS: Final[tuple[str, ...]] = (
    "balance_roller",
    "build_bridge",
    "build_tower_with_blocks",
    "clean_table",
    "collect_pens",
    "cook",
    "divide_block_tower",
    "exchange_mics",
    "exchange_pots",
    "extract_bottom_block_to_top",
    "fetch_block_with_roller",
    "handover_block_with_bowls",
    "jigsaw",
    "match_blocks_with_signs",
    "place_plate_and_cup",
    "put_objects_cabinet",
    "stack_bowls",
    "sweep_block",
)

# Short canonical prompts are intentionally below the 64-byte temporal text
# codec limit.  They identify the task but contain no stage labels, object
# state, or arm identity.
TASK_TEXT: Final[dict[str, str]] = {
    "balance_roller": "Balance the roller on the block",
    "build_bridge": "Build a bridge with the blocks",
    "build_tower_with_blocks": "Build a tower with the blocks",
    "clean_table": "Clean all objects from the table",
    "collect_pens": "Collect the pens into the container",
    "cook": "Heat the bread and place it on the plate",
    "divide_block_tower": "Divide the block tower",
    "exchange_mics": "Exchange the microphones",
    "exchange_pots": "Exchange the pots",
    "extract_bottom_block_to_top": "Move the bottom block to the top",
    "fetch_block_with_roller": "Fetch the block with the roller",
    "handover_block_with_bowls": "Handover the block using the bowls",
    "jigsaw": "Complete the jigsaw puzzle",
    "match_blocks_with_signs": "Match blocks with their signs",
    "place_plate_and_cup": "Place the plate and cup",
    "put_objects_cabinet": "Put the objects in the cabinet",
    "stack_bowls": "Stack the bowls",
    "sweep_block": "Sweep the block into position",
}

# Per-task validation limits from the benchmark's evaluation configuration.
VALIDATION_MAX_STEPS: Final[dict[str, int]] = {
    "balance_roller": 300,
    "build_bridge": 900,
    "build_tower_with_blocks": 800,
    "clean_table": 1000,
    "collect_pens": 900,
    "cook": 1000,
    "divide_block_tower": 1000,
    "exchange_mics": 900,
    "exchange_pots": 900,
    "extract_bottom_block_to_top": 400,
    "fetch_block_with_roller": 450,
    "handover_block_with_bowls": 350,
    "jigsaw": 800,
    "match_blocks_with_signs": 1300,
    "place_plate_and_cup": 700,
    "put_objects_cabinet": 1000,
    "stack_bowls": 900,
    "sweep_block": 500,
}

EPISODES_PER_TASK: Final[int] = 100
TOTAL_EPISODES: Final[int] = len(TASKS) * EPISODES_PER_TASK
VALIDATION_EPISODES: Final[int] = 20
VALIDATION_HORIZON_METHOD: Final[str] = "benchmark_fixed_per_task"
# A learned-policy smoke is an interface test, not a competence evaluation.
# Two controller ticks are the minimum which exercise reset/inference/act on
# the first tick and observation + executed-action history feedback on the
# second.  Formal probes and Validation20 continue to use the task-specific
# horizons above.
SMOKE_INTERFACE_STEPS: Final[int] = 2

# BiCoord's benchmark-owned RoboTwin-to-LeRobot converter registers the
# demonstration stream at 15 FPS.  This is the sequence clock used by the
# predictive belief teacher; the simulator's 250 Hz physics clock and the
# 30 FPS presentation-video clock are not policy-sample frequencies.
SOURCE_FREQUENCY_HZ: Final[int] = 15
FUTURE_OFFSETS_SECONDS: Final[tuple[float, ...]] = (0.2, 0.4, 0.8, 1.6)
FUTURE_OFFSETS_STEPS: Final[tuple[int, ...]] = tuple(
    int(round(seconds * SOURCE_FREQUENCY_HZ))
    for seconds in FUTURE_OFFSETS_SECONDS
)


# CARE/B0-H tensor contract.  Seven dimensions are native BiCoord I/O; the
# upstream hidden architecture remains untouched (384-wide model, 4/7 layers,
# four routing roles, etc.).
ARM_COUNT: Final[int] = 2
STATE_DIM: Final[int] = 7
ACTION_DIM: Final[int] = 7
JOINT_DIM: Final[int] = 6
GRIPPER_DIM: Final[int] = 1
HISTORY_STEPS: Final[int] = 16
ACTION_HORIZON: Final[int] = 100
# ``joint6`` denotes the six physical arm joints; the trailing gripper makes
# the policy vector seven dimensional.  The upstream HDF5 stream contains
# intermediate drive targets (not merely {0,1}), so thresholding, binarizing,
# clipping, or otherwise reparameterizing this coordinate is forbidden.
ACTION_ENCODING: Final[str] = "absolute_joint6_continuous_gripper1"
ACTION_ENCODING_DIMENSIONAL: Final[str] = (
    "absolute_7d_(6_joint+1_continuous_gripper)"
)
GRIPPER_ENCODING: Final[str] = "continuous_absolute_drive_target"
GRIPPER_NATIVE_RANGE: Final[tuple[float, float]] = (0.0, 1.0)
IMAGE_HEIGHT: Final[int] = 224
IMAGE_WIDTH: Final[int] = 224

# Frozen upstream model widths/depths.  These are metadata and constructor
# defaults, not an invitation to shrink or replace any model module.
D_MODEL: Final[int] = 384
ENCODER_LAYERS: Final[int] = 4
DECODER_LAYERS: Final[int] = 7
ROLES: Final[int] = 4
ROLE_RANK: Final[int] = 32
HISTORY_LAYERS: Final[int] = 2
VISION_BACKBONE: Final[str] = "dinov3_vitb16_frozen"
DINO_HIDDEN_SIZE: Final[int] = 768


# Matched-compute training protocol.  48 is not divisible by 18, so each
# update has two examples/task plus twelve rotating extras.  Over a three
# update cycle every task receives six base examples plus two rotating extras,
# while every update remains a full batch and each trajectory is sampled
# uniformly within its task bucket.
EFFECTIVE_BATCH: Final[int] = 48
BASE_SAMPLES_PER_TASK: Final[int] = 2
EXTRA_SAMPLES_PER_UPDATE: Final[int] = EFFECTIVE_BATCH - len(TASKS) * BASE_SAMPLES_PER_TASK
BALANCE_CYCLE_UPDATES: Final[int] = len(TASKS) // math.gcd(
    len(TASKS), EXTRA_SAMPLES_PER_UPDATE
)  # 3 updates; 12 extras/update
LOCAL_BATCH_4GPU: Final[int] = EFFECTIVE_BATCH // 4

FORMAL_B0H_UPDATES: Final[int] = 120_000
FORMAL_BCORE_UPDATES: Final[int] = 120_000
FORMAL_CARE_UPDATES: Final[int] = 4_000
FORMAL_SEEDS: Final[tuple[int, ...]] = (20260901, 20260902, 20260903)


@dataclass(frozen=True)
class ModelContract:
    """Serializable architecture contract shared by all CARE stages."""

    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    horizon: int = ACTION_HORIZON
    history_steps: int = HISTORY_STEPS
    d_model: int = D_MODEL
    enc_layers: int = ENCODER_LAYERS
    dec_layers: int = DECODER_LAYERS
    roles: int = ROLES
    role_rank: int = ROLE_RANK
    history_layers: int = HISTORY_LAYERS
    vision_backbone: str = VISION_BACKBONE
    action_encoding: str = ACTION_ENCODING
    gripper_encoding: str = GRIPPER_ENCODING
    gripper_native_range: tuple[float, float] = GRIPPER_NATIVE_RANGE
    strictly_decentralized: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "horizon": self.horizon,
            "history_steps": self.history_steps,
            "d_model": self.d_model,
            "enc_layers": self.enc_layers,
            "dec_layers": self.dec_layers,
            "roles": self.roles,
            "role_rank": self.role_rank,
            "history_layers": self.history_layers,
            "vision_backbone": self.vision_backbone,
            "action_encoding": self.action_encoding,
            "gripper_encoding": self.gripper_encoding,
            "gripper_native_range": list(self.gripper_native_range),
            "strictly_decentralized": self.strictly_decentralized,
        }


MODEL_CONTRACT: Final[ModelContract] = ModelContract()


def canonical_json_hash(value: object) -> str:
    """Hash JSON metadata deterministically for receipts."""

    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def validate_native_gripper_vector(value: object, *, context: str = "vector") -> None:
    """Validate, but never transform, native continuous gripper channels.

    This is intentionally a range check rather than a binarizer or clipper.
    Any out-of-contract value is surfaced to the caller so a simulator/policy
    mismatch cannot silently contaminate a run.  A single ``[7]`` vector or a
    batch/chunk with trailing dimension seven is accepted.
    """

    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != ACTION_DIM or not array.size:
        raise ValueError(
            f"{context} must have non-empty trailing native dimension {ACTION_DIM}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{context} must contain only finite native data")
    low, high = GRIPPER_NATIVE_RANGE
    gripper = array[..., -1]
    observed_min = float(np.min(gripper))
    observed_max = float(np.max(gripper))
    if observed_min < low or observed_max > high:
        raise ValueError(
            f"{context} continuous gripper range [{observed_min:.9g},"
            f"{observed_max:.9g}] is outside native range [{low},{high}]"
        )


def validate_task_constants() -> None:
    """Fail closed when a task table is incomplete or internally inconsistent."""

    if len(set(TASKS)) != 18:
        raise ValueError("BiCoord must contain exactly 18 unique tasks")
    if tuple(TASK_TEXT) != TASKS:
        raise ValueError("BiCoord task text order/coverage differs")
    if tuple(VALIDATION_MAX_STEPS) != TASKS:
        raise ValueError("BiCoord validation horizon order/coverage differs")
    if any(not text or len(text.encode("utf-8")) > 64 for text in TASK_TEXT.values()):
        raise ValueError("BiCoord task text exceeds the frozen 64-byte codec")
    if any(int(value) <= 0 for value in VALIDATION_MAX_STEPS.values()):
        raise ValueError("BiCoord validation max steps must be positive")
    if EFFECTIVE_BATCH != BASE_SAMPLES_PER_TASK * len(TASKS) + EXTRA_SAMPLES_PER_UPDATE:
        raise ValueError("BiCoord sampler constants do not sum to effective batch")
    if EFFECTIVE_BATCH % 4:
        raise ValueError("formal four-GPU batch must divide evenly")


validate_task_constants()


__all__ = [
    "ACTION_DIM",
    "ACTION_ENCODING",
    "ACTION_HORIZON",
    "ACTION_ENCODING_DIMENSIONAL",
    "ARM_COUNT",
    "BALANCE_CYCLE_UPDATES",
    "BASE_SAMPLES_PER_TASK",
    "DATASET_COMMIT",
    "DATASET_REPO_ID",
    "DATASET_REVISION",
    "DINO_HIDDEN_SIZE",
    "D_MODEL",
    "DECODER_LAYERS",
    "ENCODER_LAYERS",
    "EPISODES_PER_TASK",
    "EFFECTIVE_BATCH",
    "EXTRA_SAMPLES_PER_UPDATE",
    "FORMAL_B0H_UPDATES",
    "FORMAL_BCORE_UPDATES",
    "FORMAL_CARE_UPDATES",
    "FORMAL_SEEDS",
    "GRIPPER_DIM",
    "GRIPPER_ENCODING",
    "GRIPPER_NATIVE_RANGE",
    "HISTORY_LAYERS",
    "HISTORY_STEPS",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "JOINT_DIM",
    "LOCAL_BATCH_4GPU",
    "MODEL_CONTRACT",
    "ModelContract",
    "ROLES",
    "ROLE_RANK",
    "SOURCE_FREQUENCY_HZ",
    "SMOKE_INTERFACE_STEPS",
    "STATE_DIM",
    "TASKS",
    "TASK_TEXT",
    "TOTAL_EPISODES",
    "VALIDATION_EPISODES",
    "VALIDATION_HORIZON_METHOD",
    "VALIDATION_MAX_STEPS",
    "VISION_BACKBONE",
    "FUTURE_OFFSETS_SECONDS",
    "FUTURE_OFFSETS_STEPS",
    "canonical_json_hash",
    "validate_task_constants",
    "validate_native_gripper_vector",
]
