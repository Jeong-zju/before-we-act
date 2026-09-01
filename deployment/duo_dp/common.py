from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


TASKS = (
    "ball_maze",
    "bin_sort",
    "block_balance",
    "carry_pot",
    "hinge_chest",
    "join_blocks",
    "pour_marbles",
    "spring_door",
    "transfer_cube",
    "transfer_gate",
    "transfer_reorient",
)

OBS_STEPS = 3
HORIZON = 8
EXECUTION_STEPS = HORIZON - OBS_STEPS + 1
ACTION_LAG_ROWS = 1
IMAGE_SIZE = 224
POLICY_CONTRACT = (
    "shared_weights_decentralized_head_rgb_local_wrist_rgb_own_qpos_to_own_absolute_action8"
)
TASK_CONDITIONED_POLICY_CONTRACT = (
    "shared_weights_decentralized_head_rgb_local_wrist_rgb_own_qpos_task_onehot11"
    "_to_own_absolute_action8"
)
TEMPORAL_CONTRACT = "post_action_obs3_lag1_horizon8_execute6"
EVALUATOR_REVISION = "duobench-dp-obs3-lag1-h8-exec6-direct-v1"


def policy_contract(task_conditioning: bool = False) -> str:
    return TASK_CONDITIONED_POLICY_CONTRACT if task_conditioning else POLICY_CONTRACT


def atomic_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
