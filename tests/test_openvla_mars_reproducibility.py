import json
import hashlib
from pathlib import Path

import numpy as np

from deployment.vla_baselines.policy_rpc_server import _decode_openvla_chunk


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/openvla_oft_mars_control_lora_r32_formal_v1.json"
DEPLOY = ROOT / "deployment/openvla_mars"


def test_frozen_mars_contract():
    cfg = json.loads(CONFIG.read_text())
    assert cfg["benchmark"] == "MARS-Control"
    assert cfg["data"]["episodes"] == 600
    assert cfg["data"]["local_streams"] == 1650
    assert cfg["optimization"]["world_size"] == 4
    assert cfg["optimization"]["per_device_batch_size"] == 8
    assert cfg["optimization"]["global_microbatch_size"] == 32
    assert cfg["optimization"]["max_steps"] == 150000
    assert cfg["lora"]["enabled"] and cfg["lora"]["rank"] == 32
    assert cfg["lora"]["alpha"] == 16
    assert cfg["interface"]["action_encoding"] == "joint_residual_gripper_absolute"
    assert cfg["interface"]["action_chunk"] == 8
    assert cfg["validation"]["episodes_per_task"] == 20
    assert cfg["validation"]["max_steps"] == {
        "place_cube_in_cup": 500,
        "strike_cube_hard": 500,
        "three_robots_place_shoes": 1200,
        "four_robots_stack_cube": 800,
    }
    for path, expected in cfg["source"]["adaptation_source_hashes"].items():
        local = ROOT / path
        if local.is_file():
            assert hashlib.sha256(local.read_bytes()).hexdigest() == expected


def test_launchers_freeze_config_and_four_gpu_dispatch():
    train = (DEPLOY / "run_train.sh").read_text()
    supervisor = (DEPLOY / "supervisor.py").read_text()
    wrapper = (DEPLOY / "mars-openvla-supervisor.sh").read_text()
    assert "openvla_oft_mars_control_lora_r32_formal_v1.json" in train
    assert '"BWA_GPU_COUNT": "4"' in supervisor
    assert "MARS_OPENVLA_RUN_ROOT:-/workspace/bwa_mars_openvla_runs" in wrapper


def test_patch_has_expected_upstream_files():
    patch = (ROOT / "patches/openvla_oft_mars_control_formal.patch").read_text()
    for path in (
        "experiments/robot/openvla_utils.py",
        "prismatic/vla/constants.py",
        "prismatic/vla/datasets/datasets.py",
        "vla-scripts/finetune.py",
    ):
        assert f"diff --git a/{path} b/{path}" in patch


def test_residual_chunk_is_decoded_before_temporal_ensemble():
    qpos = np.asarray([0.1, -0.2, 0.3, -1.0, 0.5, 2.0, -0.4, 0.02, 0.02], np.float32)
    chunk = np.zeros((8, 8), np.float32)
    chunk[:, :7] = 0.05
    chunk[:, 7] = -1.0
    decoded = _decode_openvla_chunk(chunk, qpos, "joint_residual_gripper_absolute")
    np.testing.assert_allclose(decoded[:, :7], np.broadcast_to(qpos[None, :7] + 0.05, (8, 7)))
    np.testing.assert_allclose(decoded[:, 7], -1.0)
