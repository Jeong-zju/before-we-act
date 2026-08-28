#!/usr/bin/env python3
"""Install the audited RoboFactory adapter into a pinned official RDT checkout."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

repo = Path("/workspace/repos/rdt-1b")
deployment = Path("/workspace/repos/before-we-act/deployment/rdt_local")
shutil.copy2(deployment / "hdf5_vla_dataset.py", repo / "data/hdf5_vla_dataset.py")
# The official checkout uses implicit namespace packages.  before-we-act also
# contains regular top-level ``models``/``configs`` packages, which Python would
# otherwise prefer even when the RDT repo is first on sys.path.  Explicit empty
# package markers keep the RDT inference worker bound to the official modules.
for package in ("models", "configs"):
    marker = repo / package / "__init__.py"
    marker.write_text('"""Official RDT package namespace."""\n')

control_path = repo / "configs/dataset_control_freq.json"
control = json.loads(control_path.read_text())
control["robofactory"] = 20
control_path.write_text(json.dumps(control, indent=4) + "\n")
(repo / "configs/finetune_datasets.json").write_text(json.dumps(["robofactory"], indent=4) + "\n")
(repo / "configs/finetune_sample_weights.json").write_text(json.dumps([1.0], indent=4) + "\n")
print("configured official RDT checkout for decentralized RoboFactory", flush=True)
