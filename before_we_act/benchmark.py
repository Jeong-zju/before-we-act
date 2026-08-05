from __future__ import annotations


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


def get_task(name: str):
    try:
        return TASKS[name]
    except KeyError as exc:
        raise KeyError(f"unknown R12 Gate20 task {name!r}") from exc
