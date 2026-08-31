from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


TASKS = (
    "ball_maze", "bin_sort", "block_balance", "carry_pot", "hinge_chest",
    "join_blocks", "pour_marbles", "spring_door", "transfer_cube",
    "transfer_gate", "transfer_reorient",
)

VALIDATION_MAX_STEPS = {
    "ball_maze": 526, "bin_sort": 365, "block_balance": 1091,
    "carry_pot": 840, "hinge_chest": 610, "join_blocks": 1314,
    "pour_marbles": 549, "spring_door": 1070, "transfer_cube": 605,
    "transfer_gate": 630, "transfer_reorient": 883,
}

POLICY_CONTRACT = (
    "shared_checkpoint_strict_local_shared_head_own_wrist_qpos8_task11_arm_id2_to_local_action8"
)
ACTION_ENCODING = "absolute_joint7_binary_gripper1"
UPSTREAM_COMMIT = "a51d929027799a53d54e7d7d2ba90e2703642b4a"
DUOBENCH_COMMIT = "082a57cdafea9db115029e6fe9e03691e755f93f"
DATASET_REVISION = "b741bc915d942ecadaefb4e3de6bbd716c1b8b1b"
RCS_COMMIT = "4f78aeffae3bc4d0c02e7beab993e5406261dcf6"
FROZEN_CONFIG = Path(__file__).with_name("duobench_latent_tom_v1.json")


def atomic_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path = FROZEN_CONFIG) -> dict:
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "duobench.latent-tom.frozen-config.v1":
        raise ValueError("unexpected DuoBench LatentToM config schema")
    if value.get("policy_contract") != POLICY_CONTRACT:
        raise ValueError("strict-local policy contract drift")
    if value.get("tasks") != list(TASKS):
        raise ValueError("DuoBench task order drift")
    if value["data"]["dataset_revision"] != DATASET_REVISION:
        raise ValueError("DuoBench dataset revision drift")
    if value["data"]["duobench_commit"] != DUOBENCH_COMMIT:
        raise ValueError("DuoBench source revision drift")
    return value


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_id: int
    max_steps: int


TASK_SPECS = tuple(TaskSpec(name, i, VALIDATION_MAX_STEPS[name]) for i, name in enumerate(TASKS))
