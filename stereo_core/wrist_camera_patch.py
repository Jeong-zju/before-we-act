"""Process-local wrist-camera retrofit for every 2/3/4-agent RoboFactory task.

RoboFactory's YAML files declare world-fixed cameras, even when their uid says
``head_camera_agent{i}``.  This module leaves those files untouched and changes
only the live CameraConfig objects: local camera i is mounted on Panda i's
``panda_hand`` link.  Collection, training rollout, and evaluation must import
this same module so the observation distribution is identical end-to-end.
"""
from __future__ import annotations

import os
import numpy as np
import sapien


# This is the simulated equivalent of adding a ``camera_link`` to the Panda
# URDF with a fixed joint whose parent is ``panda_hand``.  It is intentionally
# identical for every Panda in every task: the lower edge keeps the gripper
# fingertips in view while the camera follows that hand rigidly.
CAMERA_LINK_IN_HAND = sapien.Pose(
    p=[0.0465, -0.0200, 0.0360], q=[0.0, 0.70710678, 0.0, 0.70710678]
)
# Backwards-compatible alias used by preview utilities.
LOCAL_POSE = CAMERA_LINK_IN_HAND
# Match RoboFactory's native 320x240, 90-degree pinhole camera.  A prior
# 135-degree experiment made the peripheral perspective visibly stretched.
CAMERA_FOV = 1.5707963268
# The conventional ACT corpus stays at RoboFactory's 320×240 default.  The
# future frozen-DINO experiment sets these *before importing this module* to
# archive genuine 640×480 wrist frames instead of upsampling old RGB data.
CAMERA_WIDTH = int(os.environ.get("ROBOFACTORY_WRIST_WIDTH", "320"))
CAMERA_HEIGHT = int(os.environ.get("ROBOFACTORY_WRIST_HEIGHT", "240"))
CAMERA_NEAR = 0.01
CAMERA_FAR = 10.0
# PlaceFood starts with both wrists on the far side of a large pot.  Turning
# the physical hand camera toward the interior side of the wrist exposes the
# pot/meat workspace without making the camera world-fixed.
def _install(task_class, num_agents: int):
    stock_property = task_class._default_sensor_configs

    def wrist_sensor_configs(self):
        # A decentralized policy must never receive or archive a global camera
        # stream.  Human-render cameras remain separate from these observations.
        configs = [config for config in stock_property.fget(self)
                   if config.uid != "head_camera_global"]
        local = {f"head_camera_agent{index}": index for index in range(num_agents)}
        for config in configs:
            agent_index = local.get(config.uid)
            if agent_index is None:
                continue
            hand = next(
                link for link in self.agent.agents[agent_index].robot.links
                if link.name == "panda_hand"
            )
            config.mount = hand
            config.pose = CAMERA_LINK_IN_HAND
            # All policy cameras share exactly one pinhole projection model.
            config.width = CAMERA_WIDTH
            config.height = CAMERA_HEIGHT
            config.fov = CAMERA_FOV
            config.near = CAMERA_NEAR
            config.far = CAMERA_FAR
        return configs

    task_class._default_sensor_configs = property(wrist_sensor_configs)


def install_all():
    """Install once per Python process; idempotent for normal script imports."""
    from robofactory.tasks.camera_alignment import CameraAlignmentEnv
    from robofactory.tasks.lift_barrier import LiftBarrierEnv
    from robofactory.tasks.pass_shoe import PassShoeEnv
    from robofactory.tasks.place_food import PlaceFoodEnv
    from robofactory.tasks.three_robots_stack_cube import ThreeRobotsStackCubeEnv
    from robofactory.tasks.two_robots_stack_cube import TwoRobotsStackCubeEnv
    from robofactory.tasks.long_pipeline_delivery import LongPipelineDeliveryEnv
    from robofactory.tasks.take_photo import TakePhotoEnv

    for task_class, count in (
        (LiftBarrierEnv, 2),
        (PassShoeEnv, 2),
        (PlaceFoodEnv, 2),
        (TwoRobotsStackCubeEnv, 2),
        (CameraAlignmentEnv, 3),
        (ThreeRobotsStackCubeEnv, 3),
        (LongPipelineDeliveryEnv, 4),
        (TakePhotoEnv, 4),
    ):
        if not getattr(task_class, "_wrist_camera_patch_installed", False):
            _install(task_class, count)
            task_class._wrist_camera_patch_installed = True


install_all()
