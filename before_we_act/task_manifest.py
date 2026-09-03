"""Canonical task manifest for reusable 2/3/4-agent wrist-camera experiments."""

TASKS = {
    "lift_barrier": {
        "env_id": "LiftBarrier-rf", "config": "robofactory/configs/table/lift_barrier.yaml", "agents": (0, 1), "group": "two",
    },
    "pass_shoe": {
        "env_id": "PassShoe-rf", "config": "robofactory/configs/table/pass_shoe.yaml", "agents": (0, 1), "group": "two",
    },
    "place_food": {
        "env_id": "PlaceFood-rf", "config": "robofactory/configs/table/place_food.yaml", "agents": (0, 1), "group": "two",
    },
    "two_robots_stack_cube": {
        "env_id": "TwoRobotsStackCube-rf", "config": "robofactory/configs/table/two_robots_stack_cube.yaml", "agents": (0, 1), "group": "two",
    },
    "camera_alignment": {
        "env_id": "CameraAlignment-rf", "config": "robofactory/configs/table/camera_alignment.yaml", "agents": (0, 1, 2), "group": "three",
    },
    "three_robots_stack_cube": {
        "env_id": "ThreeRobotsStackCube-rf", "config": "robofactory/configs/table/three_robots_stack_cube.yaml", "agents": (0, 1, 2), "group": "three",
    },
    "long_pipeline_delivery": {
        "env_id": "LongPipelineDelivery-rf", "config": "robofactory/configs/table/long_pipeline_delivery.yaml", "agents": (0, 1, 2, 3), "group": "four",
    },
    "take_photo": {
        "env_id": "TakePhoto-rf", "config": "robofactory/configs/table/take_photo.yaml", "agents": (0, 1, 2, 3), "group": "four",
    },
}


def get_task(name):
    try:
        return TASKS[name]
    except KeyError as error:
        raise KeyError(f"unknown task {name}; choose one of {', '.join(TASKS)}") from error
