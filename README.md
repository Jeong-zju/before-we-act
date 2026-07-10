# FE-PC-WAM

**FE-PC-WAM** stands for **Free-Energy-Guided Selective Plan Communication World Action Model**.

This repository implements a research prototype for multi-robot collaborative world-action modeling. The core objective is to enable decentralized robots to infer teammate intent from local observations, roll out candidate collaborative futures, and selectively communicate compact plan latents when communication is expected to reduce free energy.

For a detailed Chinese walkthrough of the model signal flow, codebook, parameters, losses, and evaluation metrics, see [docs/MODEL_SIGNAL_FLOW_ZH.md](docs/MODEL_SIGNAL_FLOW_ZH.md).

## Current Scope

The current implementation focuses on a minimal but extensible simulation pipeline:

- Two-robot collaborative carrying in a narrow passage
- Scripted, noisy, and recovery data collection
- HDF5-based trajectory storage
- Dataset validation, filtering, replay, and diagnostics
- Window-based PyTorch dataset interface

## Repository Structure

```text
fe_pc_wam/
├── data/
│   ├── collect.py
│   ├── dataset.py
│   ├── diagnostics.py
│   ├── filter_dataset.py
│   ├── policies.py
│   ├── replay.py
│   ├── schema.py
│   ├── split_dataset.py
│   ├── stats.py
│   └── validate_dataset.py
├── envs/
│   ├── two_robot_carry_env.py
│   ├── wrappers.py
│   └── assets/
├── models/
├── scripts/
│   ├── check_data_collection.sh
│   └── check_diagnostics.sh
├── tests/
└── README.md
