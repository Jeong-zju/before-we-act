#!/usr/bin/env python3
import json, shutil
from pathlib import Path
repo=Path("/workspace/repos/rdt-1b"); here=Path(__file__).resolve().parent
shutil.copy2(here/"hdf5_vla_dataset.py",repo/"data/hdf5_vla_dataset.py")
# The upstream checkout intentionally leaves ``data`` as a namespace package.
# Our supervisor also exposes the benchmark package on PYTHONPATH; make the
# official RDT data package explicit so imports cannot resolve to the sibling
# before-we-act/data package.
(repo/"data/__init__.py").write_text('"""Official RDT data namespace."""\n')
(repo/"train/__init__.py").write_text('"""Official RDT training namespace."""\n')
for package in ("models","configs"): (repo/package/"__init__.py").write_text('"""Official RDT namespace."""\n')
control=json.loads((repo/"configs/dataset_control_freq.json").read_text()); control["mars_control"]=20
(repo/"configs/dataset_control_freq.json").write_text(json.dumps(control,indent=2)+"\n")
(repo/"configs/finetune_datasets.json").write_text(json.dumps(["mars_control"],indent=2)+"\n")
(repo/"configs/finetune_sample_weights.json").write_text(json.dumps([1.0],indent=2)+"\n")
print("configured RDT-1B for strict-local MARS-Control")
