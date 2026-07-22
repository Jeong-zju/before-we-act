from __future__ import annotations

import argparse
from pathlib import Path
import struct

import numpy as np
import pytest

from robofactory_rpc import (
    FORMAL_LIFTBARRIER_M1_CONFIG_SHA256,
    PROTOCOL_VERSION,
    RoboFactoryRPCError,
    extract_liftbarrier_observation,
    receive_message,
    scalar_bool,
    send_message,
    split_liftbarrier_action,
    wilson_interval,
)
from scripts.run_robofactory_m1_inference import (
    _load_yaml,
    _policy_config,
    _validate_observation_message,
)
from scripts.serve_robofactory_m1_rollout import (
    FORMAL_LIFTBARRIER_ENV_CONFIG_SHA256,
    FORMAL_LIFTBARRIER_RF_SOURCE_SHA256,
    _build_summary,
    _environment_contract,
    _inspect_environment_limits,
    _prepare_output_directory,
    _seeded_reset,
    _validate_action_message,
    _validate_args,
    _validate_client_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


class _MemorySocket:
    """Minimal blocking socket surface for sandbox-independent wire tests."""

    def __init__(self) -> None:
        self.payload = bytearray()
        self.cursor = 0

    def sendall(self, value: bytes | bytearray | memoryview) -> None:
        self.payload.extend(value)

    def recv_into(self, target: memoryview) -> int:
        available = len(self.payload) - self.cursor
        if available <= 0:
            return 0
        count = min(len(target), available)
        target[:count] = self.payload[self.cursor : self.cursor + count]
        self.cursor += count
        return count


def test_rpc_round_trip_preserves_lossless_arrays_and_metadata() -> None:
    connection = _MemorySocket()
    state = np.linspace(-1.0, 1.0, 36, dtype=np.float32)
    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    send_message(
        connection,  # type: ignore[arg-type]
        {"type": "observation", "episode_index": 7, "step": 3},
        {"proprioception": state, "rgb_global": rgb},
    )
    message, arrays = receive_message(connection)  # type: ignore[arg-type]
    assert message == {
        "protocol": PROTOCOL_VERSION,
        "type": "observation",
        "episode_index": 7,
        "step": 3,
    }
    np.testing.assert_array_equal(arrays["proprioception"], state)
    np.testing.assert_array_equal(arrays["rgb_global"], rgb)
    assert arrays["proprioception"].flags.c_contiguous
    assert arrays["rgb_global"].flags.c_contiguous


def test_rpc_refuses_pickle_like_object_arrays() -> None:
    connection = _MemorySocket()
    with pytest.raises(ValueError, match="unsupported ndarray dtype"):
        send_message(
            connection,  # type: ignore[arg-type]
            {"type": "bad"},
            {"payload": np.asarray([{"untrusted": True}], dtype=object)},
        )


def test_rpc_receive_rejects_non_finite_json_constants() -> None:
    connection = _MemorySocket()
    header = (
        '{"arrays":[],"protocol":"'
        + PROTOCOL_VERSION
        + '","type":"bad","value":NaN}'
    ).encode("utf-8")
    frame = struct.pack("!I", len(header)) + header
    connection.payload.extend(struct.pack("!Q", len(frame)) + frame)
    with pytest.raises(RoboFactoryRPCError, match="invalid JSON"):
        receive_message(connection)  # type: ignore[arg-type]


def test_live_observation_mapping_matches_training_state_order() -> None:
    observation = {
        "agent": {
            "panda-0": {
                "qpos": np.arange(0, 9, dtype=np.float32)[None],
                "qvel": np.arange(9, 18, dtype=np.float32)[None],
            },
            "panda-1": {
                "qpos": np.arange(18, 27, dtype=np.float32)[None],
                "qvel": np.arange(27, 36, dtype=np.float32)[None],
            },
        },
        "sensor_data": {
            "head_camera_global": {
                "rgb": np.full((1, 240, 320, 3), 17, dtype=np.uint8)
            }
        },
    }
    state, rgb = extract_liftbarrier_observation(observation)

    np.testing.assert_array_equal(state, np.arange(36, dtype=np.float32))
    assert state.shape == (36,)
    assert state.dtype == np.float32
    assert rgb.shape == (240, 320, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb == 17)


def test_live_observation_rejects_lossy_or_wrong_camera_data() -> None:
    observation = {
        "agent": {
            name: {
                "qpos": np.zeros((1, 9), dtype=np.float32),
                "qvel": np.zeros((1, 9), dtype=np.float32),
            }
            for name in ("panda-0", "panda-1")
        },
        "sensor_data": {
            "head_camera_global": {
                "rgb": np.zeros((1, 240, 320, 3), dtype=np.float32)
            }
        },
    }
    with pytest.raises(ValueError, match="lossless uint8"):
        extract_liftbarrier_observation(observation)


def test_action_split_and_scalar_labels_are_fail_closed() -> None:
    action = np.arange(16, dtype=np.float32)
    split = split_liftbarrier_action(action)
    np.testing.assert_array_equal(split["panda-0"], action[:8])
    np.testing.assert_array_equal(split["panda-1"], action[8:])
    assert scalar_bool(np.asarray([True]), name="success") is True
    with pytest.raises(ValueError, match=r"float32\[16\]"):
        split_liftbarrier_action(np.zeros(15, dtype=np.float32))
    with pytest.raises(ValueError, match="exactly one"):
        scalar_bool(np.asarray([True, False]), name="success")
    with pytest.raises(ValueError, match="boolean"):
        scalar_bool(np.asarray([1], dtype=np.int64), name="success")


def test_policy_request_sequence_blocks_duplicate_state_mutation() -> None:
    arrays = {
        "proprioception": np.zeros(36, dtype=np.float32),
        "rgb_global": np.zeros((240, 320, 3), dtype=np.uint8),
    }
    reset = {
        "request_id": "0:0",
        "episode_index": 0,
        "step": 0,
        "reset": True,
        "image_frame_index": 0,
        "task": {"id": "lift_barrier", "text": "Lift the barrier together"},
    }
    assert _validate_observation_message(
        reset,
        arrays,
        current_episode=None,
        expected_episode=0,
        expected_step=0,
    ) == (0, 0)

    step_one = {
        **reset,
        "request_id": "0:1",
        "step": 1,
        "reset": False,
        "image_frame_index": 1,
    }
    assert _validate_observation_message(
        step_one,
        arrays,
        current_episode=0,
        expected_episode=0,
        expected_step=1,
    ) == (0, 1)
    with pytest.raises(RuntimeError, match="out of order"):
        _validate_observation_message(
            step_one,
            arrays,
            current_episode=0,
            expected_episode=0,
            expected_step=2,
        )
    with pytest.raises(RuntimeError, match="stale or out of order"):
        _validate_observation_message(
            reset,
            arrays,
            current_episode=None,
            expected_episode=1,
            expected_step=0,
        )


def test_action_response_rejects_fallback_or_out_of_order_policy_output() -> None:
    codec_sha256 = "a" * 64
    message = {
        "type": "action",
        "request_id": "0:1",
        "episode_index": 0,
        "step": 1,
        "diagnostics": {
            "action_source": "m1_scratch_latent_flow",
            "initialization_mode": "scratch",
            "action_anchor_mode": "none",
            "legacy_bypass_used": False,
            "fallback_used": False,
            "privileged_state_seen": False,
            "action_dim": 16,
            "action_codec_sha256": codec_sha256,
        },
    }
    _validate_action_message(
        message,
        request_id="0:1",
        episode_index=0,
        step=1,
        action_codec_sha256=codec_sha256,
    )
    with pytest.raises(RuntimeError, match="stale/out-of-order"):
        _validate_action_message(
            message,
            request_id="0:2",
            episode_index=0,
            step=2,
            action_codec_sha256=codec_sha256,
        )
    with pytest.raises(RuntimeError, match="diagnostics violate"):
        _validate_action_message(
            {
                **message,
                "diagnostics": {**message["diagnostics"], "fallback_used": True},
            },
            request_id="0:1",
            episode_index=0,
            step=1,
            action_codec_sha256=codec_sha256,
        )


def test_runtime_policy_config_is_derived_from_training_contract() -> None:
    config = _load_yaml(
        ROOT / "configs/wam_multimodal/m1_liftbarrier_scratch.yaml"
    )
    policy = _policy_config(config)
    assert policy.action_chunk.action_dim == 16
    assert policy.action_chunk.horizon == 8
    assert policy.action_chunk.execution_steps == 2
    assert policy.action_chunk.solver_steps == 4
    assert policy.camera_order == ("global",)
    assert policy.visual_history_frames == 2
    assert policy.replan_on_new_image is False
    assert policy.replan_warm_start_enabled is True


def test_formal_training_config_sha_is_bound_to_checked_in_yaml() -> None:
    import hashlib

    payload = (
        ROOT / "configs/wam_multimodal/m1_liftbarrier_scratch.yaml"
    ).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == FORMAL_LIFTBARRIER_M1_CONFIG_SHA256


def test_seeded_reset_reproduces_robofactory_global_numpy_randomization() -> None:
    class FakeEnvironment:
        def reset(self, *, seed: int) -> tuple[tuple[int, np.ndarray], dict[str, object]]:
            return (seed, np.random.rand(4)), {}

    env = FakeEnvironment()
    first, _ = _seeded_reset(env, 1000)
    np.random.seed(99)
    _ = np.random.rand(17)
    repeated, _ = _seeded_reset(env, 1000)
    different, _ = _seeded_reset(env, 1001)

    assert first[0] == repeated[0] == 1000
    np.testing.assert_array_equal(first[1], repeated[1])
    assert not np.array_equal(first[1], different[1])


def test_server_rejects_seed_schedules_outside_numpy_uint32() -> None:
    args = argparse.Namespace(
        port=8765,
        episodes=2,
        max_steps=500,
        seed_start=np.iinfo(np.uint32).max,
        video_fps=20,
        socket_timeout=600.0,
        allow_remote=False,
        host="127.0.0.1",
    )
    with pytest.raises(ValueError, match="uint32"):
        _validate_args(args)


def test_server_rejects_rollouts_longer_than_environment_time_limit() -> None:
    args = argparse.Namespace(
        port=8765,
        episodes=1,
        max_steps=501,
        seed_start=1000,
        video_fps=20,
        socket_timeout=600.0,
        allow_remote=False,
        host="127.0.0.1",
    )
    with pytest.raises(ValueError, match=r"\[1,500\]"):
        _validate_args(args)


def test_environment_limits_use_actual_control_frequency() -> None:
    class BaseEnvironment:
        control_freq = 20

    base = BaseEnvironment()

    class TimeLimitLike:
        def __init__(self) -> None:
            self.env = base
            self._max_episode_steps = 500

        @property
        def unwrapped(self) -> BaseEnvironment:
            return base

    class RecordEpisodeLike:
        def __init__(self) -> None:
            self.env = TimeLimitLike()
            self.max_episode_steps = 500

        @property
        def unwrapped(self) -> BaseEnvironment:
            return base

    env = RecordEpisodeLike()
    assert _inspect_environment_limits(env) == (20.0, 500)
    assert _inspect_environment_limits(TimeLimitLike()) == (20.0, 500)
    BaseEnvironment.control_freq = 10
    with pytest.raises(RuntimeError, match="control frequency"):
        _inspect_environment_limits(env)

    BaseEnvironment.control_freq = 20
    env.max_episode_steps = 499
    with pytest.raises(RuntimeError, match="TimeLimits disagree"):
        _inspect_environment_limits(env)

    with pytest.raises(RuntimeError, match="no episode TimeLimit"):
        _inspect_environment_limits(base)


def test_environment_contract_summary_and_fresh_output_policy(tmp_path: Path) -> None:
    contract = _environment_contract(
        np.zeros(36, dtype=np.float32),
        np.zeros((240, 320, 3), dtype=np.uint8),
        500,
    )
    assert contract["camera_shape"] == [240, 320, 3]
    assert contract["rgb_encoding"] == "raw_lossless"

    output = tmp_path / "run"
    _prepare_output_directory(output)
    _prepare_output_directory(output)
    (output / "evidence.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="fresh directory"):
        _prepare_output_directory(output)

    client = {
        "checkpoint": "/checkpoint",
        "checkpoint_tree_sha256": "a" * 64,
        "checkpoint_format": "wam.multimodal.m1.scratch_checkpoint/1",
        "config_sha256": (
            "c2627a5176c53e36b2cd34079a3dc9c058c819912763006c8f213a9f81e1fa89"
        ),
        "train_seed": 101,
        "vision_identity": {
            "family": "FrozenDINOv3Encoder",
            "output_dim": 1024,
            "frozen": True,
            "artifact_sha256": "c" * 64,
            "config_sha256": "d" * 64,
        },
        "vision_runtime": {
            "encoder_name": "dinov3_vitl16_lvd",
            "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "revision": "dd0a398fa8e84f2a37179332f6c561d20276300b",
            "expected_config_sha256": (
                "ce962b0c8ca4f2deb48c6fdfd6035257e3769f1d4d9154c92aba51991e46e290"
            ),
            "expected_weights_sha256": (
                "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
            ),
            "preprocess_id": "dinov3_imagenet_rgb_resize_square_antialias_v1",
            "input_size": 256,
            "inference_batch_size": 16,
            "frozen": True,
        },
        "action_codec_sha256": "b" * 64,
        "action_anchor_mode": "none",
        "device": "cuda:0",
        "provenance": {
            "repository_root": "/wam",
            "source_sha256": {
                name: "f" * 64
                for name in (
                    "robofactory_rpc.py",
                    "scripts/run_robofactory_m1_inference.py",
                    "models/wam/action_codec.py",
                    "models/wam/config.py",
                    "models/wam/heads.py",
                    "models/wam/stateful_action_flow.py",
                    "models/wam_multimodal/latent_wam.py",
                    "models/wam_multimodal/latent_world_head.py",
                    "models/wam_multimodal/token_resampler.py",
                    "models/wam_multimodal/vision_encoder.py",
                    "policies/scratch_m1.py",
                    "train/m1_scratch_builder.py",
                    "train/m1_scratch_checkpointing.py",
                )
            },
            "git": {
                "available": True,
                "commit": "1" * 40,
                "dirty": True,
                "status_porcelain": [" M source.py"],
                "tracked_diff_sha256": "2" * 64,
            },
            "runtime": {
                "python": "3.11.14",
                "numpy": "2.4.4",
                "torch": "2.11.0",
                "torch_cuda": "12.8",
                "cuda_available": True,
                "cuda_device_name": "GPU",
                "transformers": "5.12.1",
                "safetensors": "0.8.0",
                "pyyaml": "6.0.3",
            },
        },
        "policy": {
            "camera_order": ["global"],
            "visual_history_frames": 2,
            "action_horizon": 8,
            "execution_steps": 2,
            "solver_steps": 4,
            "solver": "euler",
            "normalized_action_clip": 10.0,
            "replan_on_new_image": False,
            "warm_start": True,
            "cold_start_history": "masked_zero_padding_no_action/1",
        },
    }
    assert _validate_client_metadata(client) == client
    with pytest.raises(RuntimeError, match="formal LiftBarrier M1 config"):
        _validate_client_metadata({**client, "config_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="policy runtime contract"):
        _validate_client_metadata(
            {**client, "policy": {**client["policy"], "warm_start": False}}
        )
    with pytest.raises(RuntimeError, match="source hashes are missing"):
        _validate_client_metadata(
            {
                **client,
                "provenance": {
                    **client["provenance"],
                    "source_sha256": {"policies/scratch_m1.py": "f" * 64},
                },
            }
        )
    args = argparse.Namespace(
        sim_backend="cpu",
        shader="default",
        max_steps=500,
        episodes=2,
        seed_start=1000,
        no_video=False,
        video_fps=20,
    )
    episodes = [
        {"success": True, "steps": 60},
        {"success": False, "steps": 500},
    ]
    summary = _build_summary(
        args=args,
        config_path=Path("lift_barrier.yaml"),
        output_dir=output,
        results=episodes,
        client_metadata=client,
        robofactory_provenance={"config_sha256": "e" * 64},
        environment_contract=contract,
        action_min=np.full(16, -0.5, dtype=np.float32),
        action_max=np.full(16, 0.5, dtype=np.float32),
        started_at="2026-01-01T00:00:00+00:00",
        elapsed_seconds=12.0,
        completed=True,
        fatal_error=None,
    )
    assert summary["successes"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["mean_episode_steps"] == 280.0
    assert summary["limitations"][0]["id"] == "reset_history_distribution_gap"
    assert summary["environment"]["provenance"]["config_sha256"] == "e" * 64
    assert summary["formal_benchmark"]["reportable"] is False
    assert "episodes_must_equal_100" in summary["formal_benchmark"]["violations"]
    lower, upper = summary["success_rate_wilson_95"]
    assert 0.0 < lower < 0.5 < upper < 1.0
    assert wilson_interval(1, 2) == pytest.approx((lower, upper))


def test_formal_summary_gate_accepts_only_complete_canonical_evidence(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        sim_backend="cpu",
        shader="default",
        max_steps=500,
        episodes=100,
        seed_start=1000,
        no_video=False,
        video_fps=20,
    )
    results = [
        {
            "format_version": "wam.robofactory.closed_loop_episode/1",
            "seed": seed,
            "success": seed % 2 == 0,
            "steps": 500,
            "video": f"videos/episode_{index:04d}_seed_{seed}.mp4",
        }
        for index, seed in enumerate(range(1000, 1100))
    ]
    summary = _build_summary(
        args=args,
        config_path=Path("lift_barrier.yaml"),
        output_dir=tmp_path,
        results=results,
        client_metadata={"validated": True},
        robofactory_provenance={
            "config_sha256": FORMAL_LIFTBARRIER_ENV_CONFIG_SHA256,
            "source_sha256": dict(FORMAL_LIFTBARRIER_RF_SOURCE_SHA256),
        },
        environment_contract=_environment_contract(
            np.zeros(36, dtype=np.float32),
            np.zeros((240, 320, 3), dtype=np.uint8),
            500,
        ),
        action_min=np.full(16, -0.5, dtype=np.float32),
        action_max=np.full(16, 0.5, dtype=np.float32),
        started_at="2026-01-01T00:00:00+00:00",
        elapsed_seconds=12.0,
        completed=True,
        fatal_error=None,
    )
    assert summary["formal_benchmark"] == {
        "protocol_id": "robofactory.lift_barrier.m1.seed1000_n100/1",
        "reportable": True,
        "violations": [],
        "requirements": summary["formal_benchmark"]["requirements"],
    }
