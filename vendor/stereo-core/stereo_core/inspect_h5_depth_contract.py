"""Print HDF5 depth storage metadata without assuming its unit scale."""
from __future__ import annotations

import argparse
import json
import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()
with h5py.File(args.path, "r") as h5:
    sensor = h5["traj_0"]["obs"]["sensor_data"]["head_camera_agent0"]
    depth = sensor["depth"]
    sample = np.asarray(depth[: min(4, len(depth))])
    print(json.dumps({
        "file_attrs": {key: str(value) for key, value in h5.attrs.items()},
        "sensor_keys": sorted(sensor.keys()),
        "depth_shape": list(depth.shape), "depth_dtype": str(depth.dtype),
        "depth_attrs": {key: str(value) for key, value in depth.attrs.items()},
        "raw_min": float(sample.min()), "raw_max": float(sample.max()),
    }, indent=2))
