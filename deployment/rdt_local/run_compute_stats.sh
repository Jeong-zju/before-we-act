#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace/repos/rdt-1b
exec /workspace/venvs/rdt/bin/python -m data.compute_dataset_stat_hdf5 --save_path configs/dataset_stat.json
