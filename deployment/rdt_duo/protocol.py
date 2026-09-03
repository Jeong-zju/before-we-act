"""Immutable DuoBench/RDT deployment constants."""
from __future__ import annotations

from deployment.duo_act.protocol import (
    DUOBENCH_CODE_REVISION, FORMAL_DATASET_REVISION, FORMAL_EPISODES_PER_TASK,
    FORMAL_IMAGE_SIZE, IMAGE_PREPROCESS_ID, VALIDATION_MAX_STEPS,
)

TASKS = (
    "ball_maze", "bin_sort", "block_balance", "carry_pot", "hinge_chest",
    "join_blocks", "pour_marbles", "spring_door", "transfer_cube",
    "transfer_gate", "transfer_reorient",
)
RDT_UPSTREAM_REPO = "https://github.com/thu-ml/RoboticsDiffusionTransformer.git"
RDT_UPSTREAM_COMMIT = "cd79363a1387e8f81c7724d070ef7e45fd23150f"
RDT_MODEL = "robotics-diffusion-transformer/rdt-1b"
VISION_MODEL = "google/siglip-so400m-patch14-384"
T5_MODEL = "google/t5-v1_1-xxl"
CONTROL_FREQUENCY_HZ = 30
IMAGE_HISTORY_SIZE = 2
ACTION_CHUNK_SIZE = 64
STATE_DIM = 128
LOCAL_ACTION_DIM = 8
FORMAL_STEPS = 215_000
GLOBAL_BATCH_SIZE = 16
POLICY_CONTRACT = (
    "shared_weights_decentralized_head_rgb_own_wrist_rgb_own_qpos_to_own_absolute_action8"
)
TEMPORAL_ENSEMBLE_DECAY = 0.01
EVALUATOR_REVISION = "duobench-rdt1b-local-obs2-chunk64-ensemble-v1"

__all__ = [name for name in globals() if name.isupper()]
