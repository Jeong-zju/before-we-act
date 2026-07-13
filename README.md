# FE-PC-WAM

**FE-PC-WAM** stands for **Free-Energy-Guided Selective Plan Communication World Action Model**.

This repository implements decentralized multi-robot collaborative carrying with ego-local sensing, local belief inference, selective plan-latent request/reply, and decentralized execution. Both robots share model weights but keep independent histories, message caches, random state, and cooldown state. Model inference never receives both robots' observations or simulator truth.

The information contract and VPI derivation are documented in [docs/DECENTRALIZED_DESIGN_ZH.md](docs/DECENTRALIZED_DESIGN_ZH.md).

## Workflow

Collect a dataset:

```bash
python scripts/collect_fe_pc_wam_dataset.py \
  --out-dir datasets/private_gates_v1
```

The default collection profile is the incompatible `private_gates_v1` contract:
100 pilot episodes are collected and audited first, followed by frozen
2400/400/400 train/validation/test splits. Old datasets and checkpoints cannot
be loaded under this schema. Use `--pilot-only` to stop after the quality gate.

Run the staged trainer:

```bash
python scripts/train_fe_pc_wam_pipeline.py \
  --dataset-root datasets/private_gates_v1 \
  --out-dir checkpoints/private_gates_v1
```

The stages are `plan → belief → wam → intention → wam_robust`. Use `--smoke` only to validate wiring; smoke losses are not model-quality evidence. Every checkpoint records the information contract, dataset schema, empirical `PlanCodeSupport`, and upstream SHA256 lineage.

Audit data and checkpoints before evaluation:

```bash
python scripts/audit_contract.py \
  --data datasets/carry/train \
  --checkpoint checkpoints/carry/plan.pt \
               checkpoints/carry/belief.pt \
               checkpoints/carry/wam.pt \
               checkpoints/carry/intention.pt \
               checkpoints/carry/wam_robust.pt \
  --output checkpoints/carry/audit.json
```

Runtime loading is provided by `policies.runtime.DecentralizedRuntime`. Detailed smoke, full-data, resume, GPU, and evaluation commands are in [docs/TRAINING_QUICKSTART_ZH.md](docs/TRAINING_QUICKSTART_ZH.md).

## Decision input

For the current base-only simulator (`J=0`), one robot receives:

```text
local_history[L,21]
  = base_twist(3) + local_force/contact/grasp(3)
  + ego-frame task goal(3) + private cue/valid/age/gate context(8)
  + previous ego action(4)
object_history[L,3] + valid/confidence/age + history_mask
ego_id
```

With `J` real joints, `q/dq/tau` add `3J` values. Missing perception is represented by `valid=0`; simulator object truth is never substituted at inference.

## Repository structure

```text
fe_pc_wam/
├── data/       # collection, local observation contract, HDF5 schema, dataset
├── envs/       # two-robot MuJoCo carrying environment
├── models/     # tokenizer, belief encoder, WAM, intention, communication
├── policies/   # decentralized planner and checkpoint runtime
├── train/      # staged training implementation
├── eval/       # component and paired rollout metrics
├── scripts/    # collection, training, audit, and evaluation entry points
├── tests/      # current implementation regression tests
└── docs/       # design, training, signal-flow, and acceptance notes
```
