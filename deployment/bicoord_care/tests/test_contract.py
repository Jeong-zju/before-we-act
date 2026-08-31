from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import torch

from deployment.bicoord_care.config import (
    ACTION_DIM,
    ACTION_HORIZON,
    BASE_SAMPLES_PER_TASK,
    EFFECTIVE_BATCH,
    EXTRA_SAMPLES_PER_UPDATE,
    MODEL_CONTRACT,
    STATE_DIM,
    TASKS,
    TASK_TEXT,
    VALIDATION_MAX_STEPS,
    ACTION_ENCODING,
    GRIPPER_ENCODING,
    GRIPPER_NATIVE_RANGE,
    SOURCE_FREQUENCY_HZ,
    FUTURE_OFFSETS_STEPS,
    validate_native_gripper_vector,
)
from deployment.bicoord_care.bcore_data import (
    BALANCE_CYCLE_UPDATES as BCORE_BALANCE_CYCLE_UPDATES,
    BiCoordPairedSituationBatchSampler,
)
from deployment.bicoord_care.data import (
    BALANCE_CYCLE_UPDATES as B0H_BALANCE_CYCLE_UPDATES,
    BiCoordBalancedDistributedBatchSampler,
    BiCoordEpisode,
    BiCoordTemporalDataset,
    BiCoordTemporalRequest,
    compute_normalization,
    project_local_observation,
    validate_local_sample,
    write_normalization_receipt,
)
from deployment.bicoord_care.hdf5_data import (
    BiCoordHDF5Reader,
    load_stage_segments,
    sha256_file,
    validate_hdf5_schema,
)
from deployment.bicoord_care.preprocessing import (
    decode_bicoord_jpeg_rgb,
    denormalize_vector,
    dino_normalize,
    encode_jpeg_rgb,
    normalize_vector,
    resize_rgb_batch,
    validate_dino_model_contract,
)


def _jpeg(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return bytes(encoded)


def _episode_file(root: Path, *, task: str = "cook", episode_id: int = 0, length: int = 8, peer_bias: float = 1000.0) -> Path:
    data_root = root / task / "demo_clean" / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    path = data_root / f"episode{episode_id}.hdf5"
    left = np.arange(length * 6, dtype=np.float64).reshape(length, 6)
    right = peer_bias + np.arange(length * 6, dtype=np.float64).reshape(length, 6)
    left_gripper = (np.arange(length) % 2).astype(np.float64)
    right_gripper = (1 - np.arange(length) % 2).astype(np.float64)
    frames = {
        "head_camera": np.full((12, 18, 3), (10, 20, 30), np.uint8),
        "left_camera": np.full((12, 18, 3), (40, 50, 60), np.uint8),
        "right_camera": np.full((12, 18, 3), (180, 170, 160), np.uint8),
        "front_camera": np.full((12, 18, 3), (70, 80, 90), np.uint8),
    }
    payload = {name: [_jpeg(frame + index) for index in range(length)] for name, frame in frames.items()}
    with h5py.File(path, "w") as handle:
        joint = handle.create_group("joint_action")
        joint.create_dataset("left_arm", data=left)
        joint.create_dataset("left_gripper", data=left_gripper)
        joint.create_dataset("right_arm", data=right)
        joint.create_dataset("right_gripper", data=right_gripper)
        joint.create_dataset("vector", data=np.concatenate((left, left_gripper[:, None], right, right_gripper[:, None]), axis=1))
        handle.create_dataset("endpose", data=np.zeros((length, 14)))
        observation = handle.create_group("observation")
        for name, rows in payload.items():
            camera = observation.create_group(name)
            width = max(map(len, rows))
            camera.create_dataset("rgb", data=np.asarray(rows, dtype=f"S{width}"))
    stage_root = root / task / "demo_clean" / "stages"
    stage_root.mkdir(parents=True, exist_ok=True)
    (stage_root / f"episode{episode_id}.json").write_text(
        json.dumps([[0, 3, "global a", "left a", "right a"], [3, length, "global b", "left b", "right b"]])
    )
    return path


def _record(path: Path, *, task: str = "cook", episode_id: int = 0) -> BiCoordEpisode:
    with h5py.File(path, "r") as handle:
        length = len(handle["joint_action/left_gripper"])
    return BiCoordEpisode(
        path=str(path),
        task=task,
        task_text=TASK_TEXT[task],
        episode_id=episode_id,
        length=length,
        hdf5_sha256=sha256_file(path),
        stage_path=str(path.parent.parent / "stages" / f"episode{episode_id}.json"),
    )


def _stats() -> dict[str, object]:
    return {
        "qpos_mean": np.zeros(7, np.float32),
        "qpos_std": np.ones(7, np.float32),
        "action_mean": np.zeros(7, np.float32),
        "action_std": np.ones(7, np.float32),
        "state_dim": 7,
        "action_dim": 7,
        "state_encoding": ACTION_ENCODING,
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
    }


def _cache(root: Path, episode: BiCoordEpisode, *, head: float = 1, left: float = 2, right: float = 3) -> Path:
    path = root / episode.task / f"{episode.hdf5_sha256}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        source_identity=np.asarray(episode.hdf5_sha256),
        view_head=np.full((episode.length, 768), head, np.float16),
        view_wrist_0=np.full((episode.length, 768), left, np.float16),
        view_wrist_1=np.full((episode.length, 768), right, np.float16),
    )
    return root


def test_frozen_benchmark_and_model_contract() -> None:
    assert len(TASKS) == len(set(TASKS)) == 18
    assert tuple(TASK_TEXT) == TASKS == tuple(VALIDATION_MAX_STEPS)
    assert STATE_DIM == ACTION_DIM == 7
    assert EFFECTIVE_BATCH == 48
    assert BASE_SAMPLES_PER_TASK * 18 + EXTRA_SAMPLES_PER_UPDATE == 48
    assert MODEL_CONTRACT.d_model == 384
    assert MODEL_CONTRACT.enc_layers == 4
    assert MODEL_CONTRACT.dec_layers == 7
    assert MODEL_CONTRACT.roles == 4
    assert MODEL_CONTRACT.role_rank == 32
    assert MODEL_CONTRACT.history_layers == 2
    assert SOURCE_FREQUENCY_HZ == 15
    assert FUTURE_OFFSETS_STEPS == (3, 6, 12, 24)


def test_continuous_gripper_is_preserved_and_checked_fail_closed() -> None:
    raw = {
        "observation": {
            "head_camera": {"rgb": np.zeros((2, 2, 3), np.uint8)},
            "left_camera": {"rgb": np.ones((2, 2, 3), np.uint8)},
            "right_camera": {"rgb": np.full((2, 2, 3), 2, np.uint8)},
        },
        "joint_action": {
            "left_arm": np.arange(6, dtype=np.float32),
            "left_gripper": np.float32(0.375),
            "right_arm": np.arange(6, dtype=np.float32) + 10,
            "right_gripper": np.float32(0.625),
        },
    }
    left = project_local_observation(raw, 0)
    right = project_local_observation(raw, 1)
    assert float(left["state"][-1]) == pytest.approx(0.375)
    assert float(right["state"][-1]) == pytest.approx(0.625)
    chunk = np.zeros((2, 100, ACTION_DIM), np.float32)
    chunk[..., -1] = 0.41
    validate_native_gripper_vector(chunk, context="test chunk")
    chunk[1, 4, -1] = 1.0001
    with pytest.raises(ValueError, match="outside native range"):
        validate_native_gripper_vector(chunk, context="test chunk")
    raw["joint_action"]["left_gripper"] = -1e-4
    with pytest.raises(ValueError, match="outside native range"):
        project_local_observation(raw, 0)


def test_hdf5_jpeg_schema_and_stage_alignment(tmp_path: Path) -> None:
    path = _episode_file(tmp_path)
    metadata = validate_hdf5_schema(path, check_images=True)
    assert metadata["length"] == 8
    reader = BiCoordHDF5Reader(path, task="cook")
    assert reader.state(2, 0).shape == (7,)
    assert reader.state(2, 0)[-1] == 0
    assert reader.state(2, 1)[0] >= 1000
    assert reader.frame("left_camera", 2).shape == (12, 18, 3)
    stages = load_stage_segments(path.parent.parent / "stages" / "episode0.json", length=8)
    assert [(row.start, row.end) for row in stages] == [(0, 3), (3, 8)]
    bad = path.parent.parent / "stages" / "bad.json"
    bad.write_text(json.dumps([[0, 2, "a", "a", "a"], [3, 8, "b", "b", "b"]]))
    with pytest.raises(ValueError):
        load_stage_segments(bad, length=8)


def test_jpeg_and_dino_preprocessing_contract() -> None:
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    payload = encode_jpeg_rgb(image, quality=100)
    observed = decode_bicoord_jpeg_rgb(payload + b"\0\0")
    expected = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(observed, expected)
    resized = resize_rgb_batch(observed, 224, 224)
    assert resized.shape == (3, 224, 224) and resized.dtype == torch.uint8
    normalized = dino_normalize(resized)
    assert normalized.shape == resized.shape and normalized.dtype == torch.float32

    class Config:
        model_type = "dinov3_vit"
        hidden_size = 768
        patch_size = 16
        num_register_tokens = 4
        image_size = 224

    class Model:
        config = Config()

    assert validate_dino_model_contract(Model())["hidden_size"] == 768
    Config.hidden_size = 1024
    with pytest.raises(ValueError):
        validate_dino_model_contract(Model())


def test_normalization_round_trip_and_receipt(tmp_path: Path) -> None:
    path = _episode_file(tmp_path)
    episode = _record(path)
    receipt = compute_normalization([episode])
    assert receipt["status"] == "SMOKE"
    assert receipt["recording_alignment"]["action_lag_rows"] == 1
    destination = write_normalization_receipt(tmp_path / "normalization.json", receipt)
    assert destination.is_file()
    value = torch.tensor([[-2.5, 0.0, 1.5, 4.0, 8.0, -1.0, 1.0]])
    mean = torch.tensor(receipt["qpos_mean"])
    std = torch.tensor(receipt["qpos_std"])
    assert torch.allclose(denormalize_vector(normalize_vector(value, mean, std), mean, std), value, atol=1e-6)
    # Outliers are preserved rather than clipped into a hand-chosen range.
    assert abs(float(normalize_vector(value, mean, std)[0, 0])) > 0


def test_causal_history_and_target_are_lag_one(tmp_path: Path) -> None:
    path = _episode_file(tmp_path, length=8)
    episode = _record(path)
    cache = _cache(tmp_path / "cache", episode)
    dataset = BiCoordTemporalDataset([episode], _stats(), cache)
    first = dataset[BiCoordTemporalRequest(0, 0, 0, "first", "cook")]
    assert int(first["history_mask"].sum()) == 1
    assert int(first["action_history_mask"].sum()) == 0
    assert first["episode_reset"]
    assert torch.equal(first["action"][0, :6], torch.arange(6, 12).float())
    assert int(first["action_mask"].sum()) == 7

    middle = dataset[BiCoordTemporalRequest(0, 0, 3, "middle", "cook")]
    assert int(middle["history_mask"].sum()) == 4
    assert int(middle["action_history_mask"].sum()) == 3
    # Latest executed action before observation 3 is source row 3; the new
    # target starts at source row 4.
    assert torch.equal(middle["history_action"][-1, :6], torch.arange(18, 24).float())
    assert torch.equal(middle["action"][0, :6], torch.arange(24, 30).float())
    assert middle["history_visual_raw"][-1, 1, 0] == 2

    final = dataset[BiCoordTemporalRequest(0, 0, 6, "final", "cook")]
    assert int(final["action_mask"].sum()) == 1
    assert torch.equal(final["action"][0], final["action"][-1])


def test_strict_decentralization_peer_mutation_has_no_effect(tmp_path: Path) -> None:
    path_a = _episode_file(tmp_path / "a", length=6, peer_bias=1000)
    path_b = _episode_file(tmp_path / "b", length=6, peer_bias=9000)
    episode_a = _record(path_a)
    episode_b = _record(path_b)
    cache_a = _cache(tmp_path / "cache_a", episode_a, right=7)
    cache_b = _cache(tmp_path / "cache_b", episode_b, right=99)
    dataset_a = BiCoordTemporalDataset([episode_a], _stats(), cache_a)
    dataset_b = BiCoordTemporalDataset([episode_b], _stats(), cache_b)
    sample_a = dataset_a[BiCoordTemporalRequest(0, 0, 2, "a", "cook")]
    sample_b = dataset_b[BiCoordTemporalRequest(0, 0, 2, "b", "cook")]
    validate_local_sample(sample_a, arm=0)
    for key in BiCoordTemporalDataset.MODEL_INPUT_FIELDS | BiCoordTemporalDataset.TARGET_FIELDS:
        assert torch.equal(sample_a[key], sample_b[key]), key
    assert not ({"peer_qpos", "peer_action", "peer_rgb", "stage", "task_id"} & set(sample_a))

    raw = {
        "observation": {
            "head_camera": {"rgb": np.zeros((2, 2, 3), np.uint8)},
            "left_camera": {"rgb": np.ones((2, 2, 3), np.uint8)},
            "right_camera": {"rgb": np.full((2, 2, 3), 200, np.uint8)},
        },
        "joint_action": {
            "left_arm": np.arange(6), "left_gripper": 1,
            "right_arm": np.arange(6) + 1000, "right_gripper": 0,
            "vector": np.arange(14),
        },
        "endpose": np.arange(14),
    }
    local = project_local_observation(raw, 0)
    assert set(local) == {"head_rgb", "wrist_rgb", "state", "action_history", "task_text", "reset"}
    assert np.array_equal(local["state"], np.r_[np.arange(6), 1])


def _sampler_episodes() -> list[BiCoordEpisode]:
    episodes: list[BiCoordEpisode] = []
    for task_id, task in enumerate(TASKS):
        for episode_id in range(3):
            identity = hashlib.sha256(f"{task}-{episode_id}".encode()).hexdigest()
            episodes.append(BiCoordEpisode(f"/{task}/episode{episode_id}.hdf5", task, TASK_TEXT[task], episode_id, 20 + episode_id, identity))
    return episodes


def test_sampler_task_balance_ddp_and_cursor() -> None:
    episodes = _sampler_episodes()
    sampler = BiCoordBalancedDistributedBatchSampler(episodes, updates=4, seed=17)
    totals: Counter[str] = Counter()
    for update in range(1, 4):
        rows = sampler.requests_for_update(update)
        counts = Counter(row.task for row in rows)
        assert len(rows) == EFFECTIVE_BATCH
        assert set(counts.values()) <= {2, 3}
        totals.update(counts)
    assert totals == Counter({task: 8 for task in TASKS})

    global_rows = sampler.requests_for_update(2)
    reconstructed: list[tuple[int, object]] = []
    for rank in range(4):
        local = BiCoordBalancedDistributedBatchSampler(episodes, updates=4, seed=17, rank=rank, world_size=4)
        direct = local.requests_for_update(2)[rank::4]
        assert len(direct) == 12
        reconstructed.extend((offset * 4 + rank, row) for offset, row in enumerate(direct))
    reconstructed.sort(key=lambda item: item[0])
    assert [row.sample_key for _, row in reconstructed] == [row.sample_key for row in global_rows]
    cursor = sampler.cursor_receipt(1)
    assert B0H_BALANCE_CYCLE_UPDATES == 3
    assert cursor["balance_cycle_updates"] == 3
    assert Counter(row.task for row in sampler.requests_for_update(1)) == Counter(
        row.task for row in sampler.requests_for_update(4)
    )
    assert sampler.validate_cursor(cursor) == 1
    cursor["next_sample_keys"][0] = "drift"
    with pytest.raises(ValueError):
        sampler.validate_cursor(cursor)


def test_bcore_sampler_rotation_cycle_and_cursor_receipt() -> None:
    sampler = BiCoordPairedSituationBatchSampler(
        _sampler_episodes(), updates=4, data_seed=17
    )
    totals: Counter[str] = Counter()
    for update in range(1, 4):
        rows = sampler.requests_for_update(update)
        counts = Counter(row.task for row in rows)
        assert len(rows) == EFFECTIVE_BATCH
        assert set(counts.values()) <= {2, 4}
        totals.update(counts)
    # One base pair per update plus one rotating extra pair per task over the
    # complete three-update cycle gives eight local-arm rows per task.
    assert totals == Counter({task: 8 for task in TASKS})
    assert Counter(row.task for row in sampler.requests_for_update(1)) == Counter(
        row.task for row in sampler.requests_for_update(4)
    )

    cursor = sampler.cursor_receipt(1)
    assert BCORE_BALANCE_CYCLE_UPDATES == 3
    assert cursor["balance_cycle_updates"] == 3
    assert sampler.validate_cursor(cursor) == 1
