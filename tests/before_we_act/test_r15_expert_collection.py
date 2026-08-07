from __future__ import annotations

import json

import h5py
import pytest

from before_we_act.collect_r15_stack_expert import (
    CAMERAS,
    validate_recorded_source,
)


def write_source(tmp_path, shape=(1, 480, 640, 3)):
    hdf5 = tmp_path / "expert.h5"
    metadata = tmp_path / "expert.json"
    with h5py.File(hdf5, "w") as handle:
        sensors = handle.create_group("traj_0/obs/sensor_data")
        for camera in CAMERAS:
            sensors.create_dataset(f"{camera}/rgb", shape=shape, dtype="u1")
    metadata.write_text(json.dumps({"episodes": [{"success": True}]}) + "\n")
    return hdf5, metadata


def test_native_expert_receipt_requires_all_480x640_cameras(tmp_path):
    hdf5, metadata = write_source(tmp_path)
    receipt = validate_recorded_source(hdf5, metadata, 1)
    assert receipt["episodes"] == 1
    assert receipt["rgb_shape"] == [480, 640, 3]
    assert len(receipt["recorded_shapes"]) == 4


def test_native_expert_receipt_rejects_old_240x320_capture(tmp_path):
    hdf5, metadata = write_source(tmp_path, shape=(1, 240, 320, 3))
    with pytest.raises(ValueError, match="RGB differs"):
        validate_recorded_source(hdf5, metadata, 1)
