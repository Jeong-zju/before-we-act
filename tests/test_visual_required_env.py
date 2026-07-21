from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
import pytest

from envs.runtime import RenderRequest, RunnerConfig, SimulationRunner
from envs.two_robot_carry_env import TwoRobotCooperativeStopEnv
from envs.visual_required_env import (
    CAMERA_NAMES,
    EVENT_DECISION_STEP,
    VISUAL_REQUIRED_TASKS,
    VISUAL_REQUIRED_TASK_TEXTS,
    VisualRequiredEnv,
    VisualRequiredEnvConfig,
)
from policies.visual_required import (
    PrivilegedScriptedOraclePolicy,
    StateOnlyPolicy,
    VisionOraclePolicy,
)


def test_public_contract_and_config_fail_closed() -> None:
    assert VISUAL_REQUIRED_TASKS == (
        "visual_event_stop",
        "visual_target_select",
        "visual_obstacle_avoid",
    )
    with pytest.raises(ValueError, match="unknown visual-required task"):
        VisualRequiredEnvConfig(task_id="unknown")
    with pytest.raises(ValueError, match="render_cue_mode"):
        VisualRequiredEnvConfig(render_cue_mode="random")
    with pytest.raises(ValueError, match="randomization_template_id"):
        VisualRequiredEnvConfig(randomization_template_id="")


@pytest.mark.parametrize("task_id", VISUAL_REQUIRED_TASKS)
def test_paired_seed_hides_cue_from_state_and_task_but_changes_raw_rgb(
    task_id: str,
) -> None:
    config = VisualRequiredEnvConfig(
        task_id=task_id,
        randomization_template_id="train_scene_a",
    )
    even = VisualRequiredEnv(config)
    odd = VisualRequiredEnv(config)
    try:
        even_observation, even_info = even.reset(seed=2468, randomize=True)
        odd_observation, odd_info = odd.reset(seed=2469, randomize=True)
        np.testing.assert_array_equal(
            even_observation["proprioception"], odd_observation["proprioception"]
        )
        assert even.task_condition == odd.task_condition == {
            "id": task_id,
            "text": VISUAL_REQUIRED_TASK_TEXTS[task_id],
        }
        assert tuple(even_observation) == ("proprioception",)
        assert tuple(odd_observation) == ("proprioception",)
        assert even_info["physical_seed"] == odd_info["physical_seed"] == 1234
        assert (even_info["cue_variant"], odd_info["cue_variant"]) == (0, 1)
        assert even_info["scene_id"] == odd_info["scene_id"]
        assert (
            even_info["object_combination_id"]
            == odd_info["object_combination_id"]
        )
        assert even_info["randomization_template_id"] == "train_scene_a"
        initial_even = {camera: even.render(camera=camera) for camera in CAMERA_NAMES}
        initial_odd = {camera: odd.render(camera=camera) for camera in CAMERA_NAMES}
        if task_id == "visual_event_stop":
            assert not even_info["visual_signal_active"]
            assert even_info["visual_signal_kind"] == "stop_pass"
            assert all(
                np.array_equal(initial_even[camera], initial_odd[camera])
                for camera in CAMERA_NAMES
            )
        else:
            assert even_info["visual_signal_active"]
            assert all(
                not np.array_equal(initial_even[camera], initial_odd[camera])
                for camera in CAMERA_NAMES
            )

        # Identical actions preserve identical proprioception before an
        # irreversible task choice: the cue does not perturb dynamics.
        cruise = np.asarray([0, 0.78, 0, 1, 0, 0.78, 0, 1], dtype=np.float32)
        paired_steps = EVENT_DECISION_STEP if task_id == "visual_event_stop" else 8
        for _ in range(paired_steps):
            even_observation, *_, even_step_info = even.step(cruise)
            odd_observation, *_, odd_step_info = odd.step(cruise)
            np.testing.assert_array_equal(
                even_observation["proprioception"],
                odd_observation["proprioception"],
            )
            np.testing.assert_array_equal(
                even_step_info["executed_action"], odd_step_info["executed_action"]
            )
        if task_id == "visual_event_stop":
            assert even_step_info["visual_signal_active"]
            assert even_step_info["visual_signal_onset_step"] == EVENT_DECISION_STEP
            assert all(
                not np.array_equal(
                    even.render(camera=camera), odd.render(camera=camera)
                )
                for camera in CAMERA_NAMES
            )
    finally:
        even.close()
        odd.close()


def test_randomization_template_changes_real_cue_independent_nuisance() -> None:
    first = VisualRequiredEnv(
        VisualRequiredEnvConfig(
            task_id="visual_target_select",
            randomization_template_id="scene_train_a",
        )
    )
    second = VisualRequiredEnv(
        VisualRequiredEnvConfig(
            task_id="visual_target_select",
            randomization_template_id="scene_test_z",
        )
    )
    try:
        first_observation, first_info = first.reset(seed=400, randomize=True)
        second_observation, second_info = second.reset(seed=400, randomize=True)
        assert first_info["scene_id"] != second_info["scene_id"]
        assert (
            first_info["object_combination_id"]
            != second_info["object_combination_id"]
        )
        assert not np.array_equal(
            first_observation["proprioception"],
            second_observation["proprioception"],
        )
        assert not np.array_equal(first.render(), second.render())
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("task_id", VISUAL_REQUIRED_TASKS)
@pytest.mark.parametrize("cue_variant", (0, 1))
def test_privileged_truth_oracle_solves_every_task_and_cue(
    task_id: str, cue_variant: int
) -> None:
    success, reason, steps, diagnostics = _run_policy(
        task_id,
        cue_variant,
        policy_kind="scripted",
    )
    assert success, reason
    assert reason == "none"
    assert 20 <= steps < 64
    assert diagnostics["action_source"] == "privileged_scripted_oracle"
    assert diagnostics["privileged_state_seen"] is False


@pytest.mark.parametrize("task_id", VISUAL_REQUIRED_TASKS)
@pytest.mark.parametrize("cue_variant", (0, 1))
def test_clean_vision_succeeds_and_opposite_rgb_fails(
    task_id: str, cue_variant: int
) -> None:
    clean = _run_policy(task_id, cue_variant, policy_kind="vision")
    opposite = _run_policy(
        task_id,
        cue_variant,
        policy_kind="vision",
        render_cue_mode="opposite",
    )
    assert clean[0], clean[1]
    assert not opposite[0]
    assert clean[3]["decoded_cue_variant"] == cue_variant
    assert opposite[3]["decoded_cue_variant"] == 1 - cue_variant
    assert clean[3]["action_source"] == "vision_oracle"
    assert clean[3]["consumed_observation_paths"] == (
        "images.fixed",
        "past_executed_actions",
        "proprioception",
        "task.id",
        "task.text",
    )


@pytest.mark.parametrize("task_id", VISUAL_REQUIRED_TASKS)
def test_state_only_blind_branch_is_exactly_one_of_each_cue_pair(task_id: str) -> None:
    outcomes = [
        _run_policy(task_id, cue, policy_kind="state")[0] for cue in (0, 1)
    ]
    assert outcomes == [True, False]


def test_policies_validate_all_required_inputs_and_vision_fails_without_rgb() -> None:
    env = VisualRequiredEnv(
        VisualRequiredEnvConfig(task_id="visual_target_select")
    )
    try:
        observation, _ = env.reset(seed=20)
        policy_observation = _policy_observation(
            env, observation, history=np.zeros((0, 8))
        )
        state_policy = StateOnlyPolicy()
        vision_policy = VisionOraclePolicy()
        state_action = state_policy.act(policy_observation)
        vision_action = vision_policy.act(policy_observation)
        assert state_action.shape == vision_action.shape == (8,)
        assert state_policy.last_diagnostics["consumed_observation_paths"] == (
            "past_executed_actions",
            "proprioception",
            "task.id",
            "task.text",
        )

        missing_image = dict(policy_observation)
        missing_image.pop("images")
        with pytest.raises(KeyError, match="images"):
            VisionOraclePolicy().act(missing_image)
        wrong_text = {
            **policy_observation,
            "task": {"id": "visual_target_select", "text": "choose left"},
        }
        with pytest.raises(ValueError, match="task text"):
            StateOnlyPolicy().act(wrong_text)
        bad_history = {**policy_observation, "past_executed_actions": np.zeros((2, 7))}
        with pytest.raises(ValueError, match="past_executed_actions"):
            VisionOraclePolicy().act(bad_history)
    finally:
        env.close()


def test_action_history_is_consumed_by_the_controller() -> None:
    env = VisualRequiredEnv(
        VisualRequiredEnvConfig(task_id="visual_target_select")
    )
    try:
        observation, _ = env.reset(seed=20)
        empty = _policy_observation(env, observation, history=np.zeros((0, 8)))
        previous = np.zeros((1, 8), dtype=np.float32)
        previous[0, [0, 4]] = 1.0
        with_history = _policy_observation(env, observation, history=previous)
        first = StateOnlyPolicy(blind_cue_variant=0).act(empty)
        second = StateOnlyPolicy(blind_cue_variant=0).act(with_history)
        assert not np.array_equal(first, second)
    finally:
        env.close()


def test_camera_calibration_and_render_are_raw_deterministic_rgb() -> None:
    env = VisualRequiredEnv(
        VisualRequiredEnvConfig(
            task_id="visual_obstacle_avoid",
            image_width=80,
            image_height=64,
        )
    )
    try:
        env.reset(seed=100)
        calibrations = {}
        for camera in CAMERA_NAMES:
            frame = env.render(camera=camera, width=80, height=64)
            calibration = env.camera_calibration(
                camera=camera, width=80, height=64
            )
            calibrations[camera] = calibration
            assert frame.shape == (64, 80, 3)
            assert frame.dtype == np.uint8
            assert frame.std() > 1.0
            assert calibration["intrinsics"].shape == (3, 3)
            assert calibration["intrinsics"].dtype == np.float32
            assert calibration["extrinsics"].shape == (4, 4)
            assert calibration["resolution"].tolist() == [64, 80]
            assert calibration["camera_name"] == camera
            assert calibration["model"] == "pinhole"
            assert calibration["convention"] == (
                "opencv_optical_camera_pose_in_world"
            )
            assert calibration["optical_frame"] == "opencv"
        assert calibrations["fixed"]["parent_body_name"] == "world"
        assert calibrations["robot_0_camera"]["parent_body_name"] == "robot_a"
        assert calibrations["robot_1_camera"]["parent_body_name"] == "robot_b"

        fixed_before = calibrations["fixed"]["extrinsics"].copy()
        robot_before = calibrations["robot_0_camera"]["extrinsics"].copy()
        action = np.asarray([0.2, 0.8, 0.1, 1, 0, 0.8, 0, 1], dtype=np.float32)
        env.step(action)
        fixed_after = env.camera_calibration(
            camera="fixed", width=80, height=64
        )["extrinsics"]
        robot_after = env.camera_calibration(
            camera="robot_0_camera", width=80, height=64
        )["extrinsics"]
        np.testing.assert_allclose(fixed_before, fixed_after, atol=1e-7)
        assert not np.allclose(robot_before, robot_after)
        with pytest.raises(ValueError, match="exposes cameras"):
            env.render(camera="annotated")
    finally:
        env.close()


def test_visual_camera_override_does_not_mutate_standard_legacy_rig() -> None:
    standard = TwoRobotCooperativeStopEnv()
    visual = VisualRequiredEnv()
    try:
        for camera_name in ("robot_0_camera", "robot_1_camera"):
            standard_id = mujoco.mj_name2id(
                standard.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            visual_id = mujoco.mj_name2id(
                visual.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            np.testing.assert_allclose(
                standard.model.cam_pos[standard_id], [0.0, 0.08, 0.24]
            )
            assert standard.model.cam_fovy[standard_id] == pytest.approx(45.0)
            np.testing.assert_allclose(
                visual.model.cam_pos[visual_id], [0.0, -0.90, 0.75]
            )
            assert visual.model.cam_fovy[visual_id] == pytest.approx(100.0)

        xml_bytes = Path(visual.model_xml_path).read_bytes()
        assert hashlib.sha256(xml_bytes).hexdigest() == visual.model_xml_sha256
    finally:
        standard.close()
        visual.close()


def test_event_onset_cue_and_peer_are_visible_in_all_three_raw_cameras() -> None:
    stop = VisualRequiredEnv(
        VisualRequiredEnvConfig(
            task_id="visual_event_stop", image_width=160, image_height=120
        )
    )
    passed = VisualRequiredEnv(
        VisualRequiredEnvConfig(
            task_id="visual_event_stop", image_width=160, image_height=120
        )
    )
    try:
        stop.reset(seed=200, randomize=False)
        passed.reset(seed=201, randomize=False)
        cruise = np.asarray([0, 0.9, 0, 1, 0, 0.9, 0, 1], dtype=np.float32)
        pre_stop = {camera: stop.render(camera=camera) for camera in CAMERA_NAMES}
        pre_pass = {camera: passed.render(camera=camera) for camera in CAMERA_NAMES}
        assert all(
            np.array_equal(pre_stop[camera], pre_pass[camera])
            for camera in CAMERA_NAMES
        )
        for _ in range(EVENT_DECISION_STEP):
            stop.step(cruise)
            passed.step(cruise)

        for camera in CAMERA_NAMES:
            red_frame = stop.render(camera=camera)
            green_frame = passed.render(camera=camera)
            red_signal = (
                (red_frame[..., 0] >= 120)
                & (red_frame[..., 1] <= 40)
                & (red_frame[..., 2] <= 40)
            )
            green_signal = (
                (green_frame[..., 1] >= 120)
                & (green_frame[..., 0] <= 40)
                & (green_frame[..., 2] <= 50)
            )
            assert int(red_signal.sum()) >= 20, camera
            assert int(green_signal.sum()) >= 20, camera
            assert np.count_nonzero(np.any(red_frame != green_frame, axis=2)) >= 20

        for camera in ("robot_0_camera", "robot_1_camera"):
            frame = stop.render(camera=camera)
            red_robot = (
                (frame[..., 0] > 100)
                & (frame[..., 0] > 1.8 * frame[..., 1])
                & (frame[..., 0] > 1.8 * frame[..., 2])
            )
            blue_robot = (
                (frame[..., 2] > 80)
                & (frame[..., 2] > 1.3 * frame[..., 0])
                & (frame[..., 2] > 1.3 * frame[..., 1])
            )
            assert int(red_robot.sum()) >= 20, camera
            assert int(blue_robot.sum()) >= 20, camera
    finally:
        stop.close()
        passed.close()


def test_event_signal_is_isolated_from_action_causal_red_only_brake_lights() -> None:
    stop = VisualRequiredEnv(
        VisualRequiredEnvConfig(task_id="visual_event_stop", render_cue_mode="truth")
    )
    passed = VisualRequiredEnv(
        VisualRequiredEnvConfig(task_id="visual_event_stop", render_cue_mode="truth")
    )
    opposite = VisualRequiredEnv(
        VisualRequiredEnvConfig(
            task_id="visual_event_stop", render_cue_mode="opposite"
        )
    )
    environments = (stop, passed, opposite)
    brake_ids: tuple[int, int] | None = None
    signal_id: int | None = None
    off = np.asarray([0.025, 0.004, 0.004, 1.0], dtype=np.float32)
    red = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    cruise = np.asarray([0, 0.9, 0, 1, 0, 0.9, 0, 1], dtype=np.float32)
    brake_robot_0 = np.asarray([0, 0.0, 0, 1, 0, 0.9, 0, 1], dtype=np.float32)
    brake_both = np.asarray([0, 0.0, 0, 1, 0, 0.0, 0, 1], dtype=np.float32)
    try:
        stop.reset(seed=200, randomize=False)
        passed.reset(seed=201, randomize=False)
        opposite.reset(seed=200, randomize=False)
        brake_ids = tuple(
            mujoco.mj_name2id(stop.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("robot_0_brake_light", "robot_1_brake_light")
        )
        signal_id = mujoco.mj_name2id(
            stop.model, mujoco.mjtObj.mjOBJ_GEOM, "visual_event_signal"
        )
        for env in environments:
            np.testing.assert_allclose(
                env.model.geom_rgba[list(brake_ids)], [off, off]
            )

        for _ in range(EVENT_DECISION_STEP):
            stop.step(cruise)
            passed.step(cruise)
            opposite.step(cruise)

        # The first cue-bearing capture is produced by a cruise action.  Only
        # the dedicated signal may differ; neither robot has braked yet.
        for env in environments:
            np.testing.assert_allclose(
                env.model.geom_rgba[list(brake_ids)], [off, off]
            )
            assert env._info()["brake_light_active_agents"] == []
        changed_geoms = set(
            np.flatnonzero(
                np.any(
                    np.abs(stop.model.geom_rgba - passed.model.geom_rgba) > 1e-7,
                    axis=1,
                )
            ).tolist()
        )
        assert changed_geoms == {signal_id}

        # Truth/opposite changes only the dedicated signal.  Identical actions
        # produce identical per-agent lamp state regardless of cue pixels.
        stop.step(brake_robot_0)
        opposite.step(brake_robot_0)
        np.testing.assert_allclose(
            stop.model.geom_rgba[list(brake_ids)], [red, off]
        )
        np.testing.assert_allclose(
            opposite.model.geom_rgba[list(brake_ids)], [red, off]
        )
        assert stop._info()["brake_light_active_agents"] == [0]

        stop.step(brake_both)
        np.testing.assert_allclose(stop.model.geom_rgba[list(brake_ids)], [red, red])
        assert stop._info()["brake_light_active_agents"] == [0, 1]

        passed.step(cruise)
        np.testing.assert_allclose(
            passed.model.geom_rgba[list(brake_ids)], [off, off]
        )
        assert passed._info()["brake_light_active_agents"] == []
        for env in environments:
            assert np.all(env.model.geom_rgba[list(brake_ids), 1] <= 0.004 + 1e-7)
    finally:
        for env in environments:
            env.close()


def test_vision_oracle_cruises_without_decoding_neutral_event_signal() -> None:
    actions = []
    for seed in (200, 201):
        env = VisualRequiredEnv(VisualRequiredEnvConfig(task_id="visual_event_stop"))
        try:
            observation, info = env.reset(seed=seed, randomize=False)
            assert not info["visual_signal_active"]
            policy = VisionOraclePolicy()
            action = policy.act(
                _policy_observation(
                    env, observation, history=np.zeros((0, 8), dtype=np.float32)
                )
            )
            assert not policy.last_diagnostics["visual_signal_decoded"]
            assert policy.last_diagnostics["decoded_cue_variant"] is None
            actions.append(action)
        finally:
            env.close()
    np.testing.assert_array_equal(actions[0], actions[1])
    assert actions[0][1] > 0.0 and actions[0][5] > 0.0


def _run_policy(
    task_id: str,
    cue_variant: int,
    *,
    policy_kind: str,
    render_cue_mode: str = "truth",
) -> tuple[bool, str, int, Mapping[str, Any]]:
    env = VisualRequiredEnv(
        VisualRequiredEnvConfig(
            task_id=task_id,
            episode_len=64,
            render_cue_mode=render_cue_mode,
        )
    )
    if policy_kind == "scripted":
        policy: Any = PrivilegedScriptedOraclePolicy(env)
    elif policy_kind == "state":
        policy = StateOnlyPolicy(blind_cue_variant=0)
    elif policy_kind == "vision":
        policy = VisionOraclePolicy()
    else:  # pragma: no cover - test helper guard.
        raise ValueError(policy_kind)
    config = RunnerConfig(
        max_steps=64,
        render=(RenderRequest("fixed", "fixed", width=96, height=96, fps=10.0),),
        policy_observation_keys=("proprioception",),
        expose_rendered_images_to_policy=True,
        policy_image_streams=("fixed",),
        expose_task_to_policy=True,
        task_id=task_id,
        task=VISUAL_REQUIRED_TASK_TEXTS[task_id],
        policy_action_history=4,
    )
    try:
        summary = SimulationRunner(env, policy, config).run_episode(
            seed=4000 + cue_variant,
            randomize=True,
        )
        return (
            bool(summary.final_info["success"]),
            str(summary.final_info["failure_reason"]),
            summary.steps,
            dict(policy.last_diagnostics),
        )
    finally:
        env.close()


def _policy_observation(
    env: VisualRequiredEnv,
    observation: Mapping[str, Any],
    *,
    history: np.ndarray,
) -> dict[str, Any]:
    return {
        "proprioception": np.asarray(observation["proprioception"]).copy(),
        "images": {"fixed": env.render()},
        "task": dict(env.task_condition),
        "past_executed_actions": np.asarray(history, dtype=np.float32).copy(),
    }
