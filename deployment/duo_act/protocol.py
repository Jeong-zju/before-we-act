"""Frozen DuoBench data and Validation20 protocol constants.

The validation horizons are not hand-tuned rollout knobs.  They are the
ceiling of NumPy's default (linear) 0.99 quantile of the 50 successful
demonstration lengths for each task in the immutable DuoBench snapshot.  This
module is intentionally dependency-light so the converter, policy adapters,
and supervisor can all consume one source of truth.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np


FORMAL_DATASET_REVISION = "b741bc915d942ecadaefb4e3de6bbd716c1b8b1b"
DUOBENCH_CODE_REVISION = "082a57cdafea9db115029e6fe9e03691e755f93f"
RCS_LEROBOT_CONVERTER_REVISION = "4f78aeffae3bc4d0c02e7beab993e5406261dcf6"
FORMAL_IMAGE_SIZE = 224
# Frozen torchvision-v2 uint8 resize used by the pinned RCS converter and
# independently reproduced by evaluate.policy_image.
IMAGE_PREPROCESS_ID = "rcs_lerobot_v2_resize_uint8_bilinear_antialias_v1"
FORMAL_EPISODES_PER_TASK = 50
VALIDATION_HORIZON_QUANTILE = 0.99
VALIDATION_HORIZON_METHOD = "ceil_numpy_linear_quantile_episode_lengths"

# Byte identities of the immutable simulation parquet shards.  Video hashes
# are deliberately outside the action contract; these hashes pin the raw
# observation.state/action columns from which controller-equivalent targets
# and normalization are produced.
FORMAL_SIM_PARQUET_SHA256 = {
    "ball_maze": "7859fd3f0bd0eece2b14fe783c501e84fa6b4527ed8f744fe390bcd6bcda5ece",
    "bin_sort": "206e2266c00df6c6f2e073f5d2096c1bcb9ba54d7f1b9af5a11027f0537e9ff0",
    "block_balance": "dd110a3a81505b47c70a558032f3333d9e2c6faf49a8777c6a053201346ce0cf",
    "carry_pot": "54dce0ad6dcd613f632953211f75927417ec0b0a8d0ecb756c87da29a090c5e1",
    "hinge_chest": "20dbd768268f43f4e325068fa04b9657be68050b868369262aff72aa5325f493",
    "join_blocks": "78dea2c08c0c28854029a84d4e1d3212d2ecf9002b847b0533131426d3b391dc",
    "pour_marbles": "7f32df771c2f2b57a43e6ccc7973ad2f454cf03626ba35621cdca50f1d8b2b16",
    "spring_door": "abaaaa1f8083ba065d9103ecb5a6ee1f791e65bfbb3cb2d297171392dfca4830",
    "transfer_cube": "acceb001b6fdf2666a1a390015d6d5f2cac1757b7190b6417befc5d66e57bed3",
    "transfer_gate": "faaab8906c94a8e90e05045e187a0fad95cf6bf7f621d6d4f093066e1c224c8f",
    "transfer_reorient": "d3c0022de9034dc1031ce88df964608755a047c74f888035fe86bcda6a0bcf58",
}

# Counts independently audited at FORMAL_DATASET_REVISION.  They are entries,
# not frames: both arms and all seven joints are included.  Freezing these
# numbers prevents a converter or bound drift from being hidden by a freshly
# generated self-consistent receipt.
FORMAL_CONTROLLER_CORRECTIONS_BY_TASK = {
    "ball_maze": [0, 20, 0, 78, 10, 0, 8],
    "bin_sort": [0, 0, 0, 39, 307, 0, 0],
    "block_balance": [0, 0, 0, 22, 69, 17, 347],
    "carry_pot": [0, 178, 51, 146, 0, 0, 202],
    "hinge_chest": [118, 50, 6, 511, 825, 5, 8],
    "join_blocks": [0, 223, 0, 0, 7, 0, 0],
    "pour_marbles": [35, 0, 560, 0, 40, 0, 186],
    "spring_door": [0, 1, 0, 4142, 1433, 1, 0],
    "transfer_cube": [318, 0, 0, 197, 21, 0, 0],
    "transfer_gate": [134, 0, 3, 1868, 565, 0, 0],
    "transfer_reorient": [0, 0, 0, 0, 0, 0, 0],
}
FORMAL_CONTROLLER_CORRECTIONS_BY_JOINT = [605, 472, 620, 7003, 3277, 23, 751]
FORMAL_CONTROLLER_CORRECTION_ENTRIES = 12_751
FORMAL_RCS_API_OUT_OF_RANGE_ENTRIES_DIAGNOSTIC = 105_590

# Audited directly from the 50-demo parquet at FORMAL_DATASET_REVISION.  The
# q99 values below are included as evidence rather than recomputed constants;
# prepare.py recomputes them from every downloaded parquet and fails closed on
# any mismatch.
FORMAL_TASK_LENGTH_STATS = {
    "ball_maze": {"frames": 17180, "min": 249, "mean": 343.60, "max": 530, "q99": 525.10},
    "bin_sort": {"frames": 13292, "min": 145, "mean": 265.84, "max": 378, "q99": 364.28},
    "block_balance": {"frames": 37688, "min": 519, "mean": 753.76, "max": 1110, "q99": 1090.40},
    "carry_pot": {"frames": 17849, "min": 153, "mean": 356.98, "max": 840, "q99": 839.02},
    "hinge_chest": {"frames": 21954, "min": 369, "mean": 439.08, "max": 625, "q99": 609.32},
    "join_blocks": {"frames": 43431, "min": 600, "mean": 868.62, "max": 1401, "q99": 1313.78},
    "pour_marbles": {"frames": 22098, "min": 360, "mean": 441.96, "max": 567, "q99": 548.87},
    "spring_door": {"frames": 40326, "min": 643, "mean": 806.52, "max": 1112, "q99": 1069.86},
    "transfer_cube": {"frames": 21833, "min": 348, "mean": 436.66, "max": 627, "q99": 604.46},
    "transfer_gate": {"frames": 26261, "min": 429, "mean": 525.22, "max": 640, "q99": 629.22},
    "transfer_reorient": {"frames": 24076, "min": 322, "mean": 481.52, "max": 1053, "q99": 882.97},
}

VALIDATION_MAX_STEPS = {
    task: int(math.ceil(float(stats["q99"])))
    for task, stats in FORMAL_TASK_LENGTH_STATS.items()
}


def derive_validation_max_steps(lengths: Sequence[int] | np.ndarray) -> tuple[float, int]:
    """Return the frozen linear q99 value and its integer rollout ceiling."""

    value = np.asarray(lengths, dtype=np.int64)
    if value.shape != (FORMAL_EPISODES_PER_TASK,) or np.any(value <= 0):
        raise ValueError(
            "formal Duo horizon requires exactly 50 positive episode lengths"
        )
    q99 = float(np.quantile(value, VALIDATION_HORIZON_QUANTILE, method="linear"))
    return q99, int(math.ceil(q99))


def validate_task_length_contract(task: str, lengths: Sequence[int] | np.ndarray) -> dict[str, float | int]:
    """Validate one parquet's lengths against the immutable snapshot receipt."""

    if task not in FORMAL_TASK_LENGTH_STATS:
        raise ValueError(f"unknown DuoBench task: {task}")
    value = np.asarray(lengths, dtype=np.int64)
    q99, horizon = derive_validation_max_steps(value)
    expected = FORMAL_TASK_LENGTH_STATS[task]
    observed = {
        "frames": int(value.sum()),
        "min": int(value.min()),
        "mean": float(value.mean()),
        "max": int(value.max()),
        "q99": q99,
        "validation_max_steps": horizon,
    }
    checks = (
        observed["frames"] == expected["frames"],
        observed["min"] == expected["min"],
        abs(float(observed["mean"]) - float(expected["mean"])) <= 1e-9,
        observed["max"] == expected["max"],
        abs(float(observed["q99"]) - float(expected["q99"])) <= 1e-9,
        horizon == VALIDATION_MAX_STEPS[task],
    )
    if not all(checks):
        raise ValueError(
            f"{task}: demonstration-length contract drifted: {observed} != {expected}"
        )
    return observed


__all__ = [
    "DUOBENCH_CODE_REVISION",
    "FORMAL_DATASET_REVISION",
    "FORMAL_EPISODES_PER_TASK",
    "FORMAL_IMAGE_SIZE",
    "IMAGE_PREPROCESS_ID",
    "FORMAL_CONTROLLER_CORRECTION_ENTRIES",
    "FORMAL_CONTROLLER_CORRECTIONS_BY_JOINT",
    "FORMAL_CONTROLLER_CORRECTIONS_BY_TASK",
    "FORMAL_RCS_API_OUT_OF_RANGE_ENTRIES_DIAGNOSTIC",
    "FORMAL_SIM_PARQUET_SHA256",
    "FORMAL_TASK_LENGTH_STATS",
    "RCS_LEROBOT_CONVERTER_REVISION",
    "VALIDATION_HORIZON_METHOD",
    "VALIDATION_HORIZON_QUANTILE",
    "VALIDATION_MAX_STEPS",
    "derive_validation_max_steps",
    "validate_task_length_contract",
]
