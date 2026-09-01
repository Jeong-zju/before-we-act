from __future__ import annotations

from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from deployment.bicoord_care.bicoord_snapshot import (
    SNAPSHOT_TOLERANCE,
    SnapshotCapabilityError,
    capability_report,
    capture_state,
    restore_probe,
    restore_state,
    state_sha256,
)


class Pose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)):
        self.p = np.asarray(p, np.float64)
        self.q = np.asarray(q, np.float64)


def _pose(value) -> Pose:
    if isinstance(value, Pose):
        return Pose(value.p.copy(), value.q.copy())
    return Pose(value[0], value[1])


class PhysxRigidDynamicComponent:
    def __init__(self):
        self.linear_velocity = np.asarray([0.1, 0.2, 0.3])
        self.angular_velocity = np.asarray([0.4, 0.5, 0.6])
        self.kinematic = True
        self.kinematic_target = Pose([1.0, 2.0, 3.0])

    def get_linear_velocity(self):
        return self.linear_velocity

    def set_linear_velocity(self, value):
        self.linear_velocity = np.asarray(value, np.float64).copy()

    def get_angular_velocity(self):
        return self.angular_velocity

    def set_angular_velocity(self, value):
        self.angular_velocity = np.asarray(value, np.float64).copy()

    def get_kinematic(self):
        return self.kinematic

    def set_kinematic(self, value):
        self.kinematic = bool(value)

    def get_kinematic_target(self):
        return self.kinematic_target

    def set_kinematic_target(self, value):
        self.kinematic_target = _pose(value)


class Actor:
    def __init__(self):
        self.name = "movable"
        self.pose = Pose([0.1, 0.2, 0.3])
        self.component = PhysxRigidDynamicComponent()

    def get_name(self):
        return self.name

    def get_pose(self):
        return self.pose

    def set_pose(self, value):
        self.pose = _pose(value)

    def get_components(self):
        return [self.component]


class Joint:
    def __init__(self):
        self.drive_target = np.asarray([0.25])
        self.drive_velocity_target = np.asarray([-0.5])

    def get_drive_target(self):
        return self.drive_target

    def set_drive_target(self, value):
        self.drive_target = np.asarray(value, np.float64).copy()

    def get_drive_velocity_target(self):
        return self.drive_velocity_target

    def set_drive_velocity_target(self, value):
        self.drive_velocity_target = np.asarray(value, np.float64).copy()


class Articulation:
    def __init__(self):
        self.name = "robot"
        self.qpos = np.asarray([0.2, 0.3])
        self.qvel = np.asarray([0.4, 0.5])
        self.qacc = np.asarray([0.6, 0.7])
        self.qf = np.asarray([0.8, 0.9])
        self.root_pose = Pose([0.0, 0.0, 1.0])
        self.root_linear_velocity = np.asarray([0.1, 0.0, 0.0])
        self.root_angular_velocity = np.asarray([0.0, 0.1, 0.0])
        self.joints = [Joint(), Joint()]

    def get_name(self):
        return self.name

    def get_qpos(self):
        return self.qpos

    def set_qpos(self, value):
        self.qpos = np.asarray(value, np.float64).copy()

    def get_qvel(self):
        return self.qvel

    def set_qvel(self, value):
        self.qvel = np.asarray(value, np.float64).copy()

    def get_qacc(self):
        return self.qacc

    def set_qacc(self, value):
        self.qacc = np.asarray(value, np.float64).copy()

    def get_qf(self):
        return self.qf

    def set_qf(self, value):
        self.qf = np.asarray(value, np.float64).copy()

    def get_root_pose(self):
        return self.root_pose

    def set_root_pose(self, value):
        self.root_pose = _pose(value)

    def get_root_linear_velocity(self):
        return self.root_linear_velocity

    def set_root_linear_velocity(self, value):
        self.root_linear_velocity = np.asarray(value, np.float64).copy()

    def get_root_angular_velocity(self):
        return self.root_angular_velocity

    def set_root_angular_velocity(self, value):
        self.root_angular_velocity = np.asarray(value, np.float64).copy()

    def get_active_joints(self):
        return self.joints


class PhysxCpuSystem:
    def __init__(self, scene=None):
        self.scene = scene
        self.hidden_solver_state = np.asarray([0.125, -0.25], np.float64)

    def pack(self):
        return self.hidden_solver_state.tobytes()

    def unpack(self, value):
        restored = np.frombuffer(value, dtype=np.float64)
        if restored.shape != (2,):
            raise ValueError("malformed fake PhysX state")
        self.hidden_solver_state = restored.copy()
        # Model the SAPIEN side effect observed on contact-rich articulation
        # scenes: unpack may recompute qacc/qf caches.  The production
        # serializer must re-apply the captured values after this call.
        if self.scene is not None:
            self.scene.articulation.qacc[:] = 77
            self.scene.articulation.qf[:] = 78


class Scene:
    def __init__(self):
        self.actor = Actor()
        self.articulation = Articulation()
        self.physx_system = PhysxCpuSystem(self)
        self.packed = b"scene-poses-v1"

    def get_physx_system(self):
        return self.physx_system

    def pack_poses(self):
        return self.packed

    def unpack_poses(self, value):
        self.packed = bytes(value)

    def get_all_actors(self):
        return [self.actor]

    def get_all_articulations(self):
        return [self.articulation]


class Robot:
    def __init__(self):
        self.left_gripper_val = 0.1
        self.right_gripper_val = 0.2


class Env:
    def __init__(self):
        self.scene = Scene()
        self.robot = Robot()
        self.take_action_cnt = 3
        self.left_cnt = 4
        self.right_cnt = 5
        self.stage_eval_score = 0.125
        self.eval_success = False
        self._episode_rng = np.random.RandomState(41)
        self.generator_rng = np.random.default_rng(42)
        self.python_rng = random.Random(43)
        self.torch_rng = torch.Generator().manual_seed(44)

    def get_obs(self):
        return {
            "actor": self.scene.actor.pose.p.copy(),
            "qpos": self.scene.articulation.qpos.copy(),
            "progress": self.stage_eval_score,
        }

    def take_action(self, action):
        value = float(np.asarray(action).reshape(-1)[0])
        noise = (
            random.random()
            + float(np.random.random())
            + float(torch.rand(()))
            + float(self._episode_rng.random_sample())
            + float(self.generator_rng.random())
            + float(self.python_rng.random())
            + float(torch.rand((), generator=self.torch_rng))
        )
        solver_effect = float(self.scene.physx_system.hidden_solver_state[0])
        self.scene.actor.pose.p[0] += value + noise + solver_effect
        self.scene.physx_system.hidden_solver_state += np.asarray([0.01, -0.02])
        self.scene.articulation.qpos[0] += value
        self.take_action_cnt += 1
        self.stage_eval_score += 0.01

    def _update_render(self):
        return None


class Wrapper:
    def __init__(self, env):
        self.env = env
        self._elapsed_steps = 7


def _observable(wrapper: Wrapper) -> dict[str, object]:
    env = wrapper.env
    return {
        "observation": env.get_obs(),
        "actor_velocity": env.scene.actor.component.linear_velocity.copy(),
        "qacc": env.scene.articulation.qacc.copy(),
        "qf": env.scene.articulation.qf.copy(),
        "root_pose": env.scene.articulation.root_pose.p.copy(),
        "drive_target": env.scene.articulation.joints[0].drive_target.copy(),
        "wrapper_steps": wrapper._elapsed_steps,
        "counter": env.take_action_cnt,
    }


def test_reference_snapshot_restores_physics_controllers_wrappers_and_rng() -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    wrapper = Wrapper(Env())
    state = capture_state(wrapper)
    before = deepcopy(_observable(wrapper))
    digest = state_sha256(state)
    assert state["serializer"] == "reference_serializer"
    assert len(digest) == 64 and digest == state_sha256(state)

    env = wrapper.env
    env.scene.actor.pose.p[:] = 99
    env.scene.actor.component.linear_velocity[:] = 98
    env.scene.actor.component.kinematic = False
    env.scene.physx_system.hidden_solver_state[:] = 90
    env.scene.articulation.qacc[:] = 97
    env.scene.articulation.qf[:] = 96
    env.scene.articulation.root_pose = Pose([95, 95, 95])
    env.scene.articulation.joints[0].drive_target[:] = 94
    env.robot.left_gripper_val = 93
    wrapper._elapsed_steps = 92
    env.take_action_cnt = 91
    random.random()
    np.random.random()
    torch.rand(())

    restore_state(wrapper, state)
    assert state_sha256(state) == digest
    assert _observable(wrapper).keys() == before.keys()
    for key in before:
        left, right = before[key], _observable(wrapper)[key]
        if isinstance(left, dict):
            for child in left:
                assert np.allclose(left[child], right[child])
        else:
            assert np.allclose(left, right)
    assert env.scene.actor.component.kinematic is True
    assert np.array_equal(
        env.scene.physx_system.hidden_solver_state,
        np.asarray([0.125, -0.25]),
    )
    assert env.robot.left_gripper_val == 0.1


def test_reference_snapshot_requires_physx_pack_and_unpack() -> None:
    wrapper = Wrapper(Env())
    wrapper.env.scene.physx_system = object()
    report = capability_report(wrapper)
    assert report["exact"] is False
    assert "PhysX system lacks pack" in report["missing"]
    with pytest.raises(SnapshotCapabilityError, match="PhysX system lacks pack"):
        capture_state(wrapper)


def test_reference_restore_rejects_snapshot_without_physx_state() -> None:
    wrapper = Wrapper(Env())
    state = capture_state(wrapper)
    state.pop("physx_state")
    with pytest.raises(SnapshotCapabilityError, match="lacks PhysX system state"):
        restore_state(wrapper, state)


def test_restore_probe_requires_bitwise_repeatable_post_restore_rollouts() -> None:
    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)
    wrapper = Wrapper(Env())
    state = capture_state(wrapper)

    def rollout(current: Wrapper):
        current.env.take_action(np.asarray([0.5]))
        current._elapsed_steps += 1
        return _observable(current)

    report = restore_probe(wrapper, state, rollout, repeats=2)
    assert report["passed"] is True
    assert report["max_abs_error"] <= SNAPSHOT_TOLERANCE


def test_non_kinematic_sapien_actor_does_not_require_a_kinematic_target() -> None:
    wrapper = Wrapper(Env())
    component = wrapper.env.scene.actor.component
    component.kinematic = False

    def unavailable_target():
        raise RuntimeError("failed to get kinematic target: actor is not kinematic")

    component.get_kinematic_target = unavailable_target
    state = capture_state(wrapper)
    actor = next(iter(state["actors"].values()))
    assert actor["kinematic"] is False
    assert "kinematic_target" not in actor


def test_unknown_kinematic_target_error_remains_fail_closed() -> None:
    wrapper = Wrapper(Env())
    component = wrapper.env.scene.actor.component

    def broken_target():
        raise RuntimeError("PhysX scene state is unavailable")

    component.get_kinematic_target = broken_target
    with pytest.raises(RuntimeError, match="PhysX scene state is unavailable"):
        capture_state(wrapper)


class NativeEnv(Env):
    def get_state_dict(self):
        return {
            "actor": self.scene.actor.pose.p.copy(),
            "qpos": self.scene.articulation.qpos.copy(),
            "counter": self.take_action_cnt,
        }

    def set_state_dict(self, value):
        self.scene.actor.pose.p[:] = value["actor"]
        self.scene.articulation.qpos[:] = value["qpos"]
        self.take_action_cnt = int(value["counter"])


def test_native_snapshot_also_captures_rng_and_wrapper_state() -> None:
    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    wrapper = Wrapper(NativeEnv())
    state = capture_state(wrapper)
    assert state["serializer"] == "native"
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    wrapper._elapsed_steps = 100
    wrapper.env.take_action(np.asarray([2.0]))
    restore_state(wrapper, state)
    observed = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert observed == expected
    assert wrapper._elapsed_steps == 7
    assert wrapper.env.take_action_cnt == 3
