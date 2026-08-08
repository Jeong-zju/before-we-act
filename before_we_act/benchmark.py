from __future__ import annotations

from before_we_act.r13n import TASKS as ACTIVE_TASKS

# Frozen R12 compatibility registry.  New execution must use ACTIVE_TASKS;
# keeping this name avoids rewriting archived R12 receipts and tests.
TASKS = {
    "lift_barrier": {
        "env_id": "LiftBarrier-rf",
        "config": "robofactory/configs/table/lift_barrier.yaml",
        "agents": (0, 1),
    },
    "camera_alignment": {
        "env_id": "CameraAlignment-rf",
        "config": "robofactory/configs/table/camera_alignment.yaml",
        "agents": (0, 1, 2),
    },
    "three_robots_stack_cube": {
        "env_id": "ThreeRobotsStackCube-rf",
        "config": "robofactory/configs/table/three_robots_stack_cube.yaml",
        "agents": (0, 1, 2),
    },
    "long_pipeline_delivery": {
        "env_id": "LongPipelineDelivery-rf",
        "config": "robofactory/configs/table/long_pipeline_delivery.yaml",
        "agents": (0, 1, 2, 3),
    },
    "take_photo": {
        "env_id": "TakePhoto-rf",
        "config": "robofactory/configs/table/take_photo.yaml",
        "agents": (0, 1, 2, 3),
    },
}

ACTIVE_TASK_SPECS = {
    "lift_barrier": TASKS["lift_barrier"],
    "camera_alignment": TASKS["camera_alignment"],
    "long_pipeline_delivery": TASKS["long_pipeline_delivery"],
    "take_photo": TASKS["take_photo"],
    "pass_shoe": {
        "env_id": "PassShoe-rf",
        "config": "robofactory/configs/table/pass_shoe.yaml",
        "agents": (0, 1),
    },
    "place_food": {
        "env_id": "PlaceFood-rf",
        "config": "robofactory/configs/table/place_food.yaml",
        "agents": (0, 1),
    },
}

if tuple(ACTIVE_TASK_SPECS) != ACTIVE_TASKS:
    raise RuntimeError("active R13N benchmark order differs")

ALL_TASK_SPECS = {**TASKS, **ACTIVE_TASK_SPECS}


def get_task(name: str):
    try:
        return ALL_TASK_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"unknown active no-stack benchmark task {name!r}") from exc
