# FE-PC-WAM

**FE-PC-WAM** stands for **Free-Energy-Guided Selective Plan Communication World Action Model**.

This repository implements decentralized multi-robot collaborative carrying with ego-local sensing, local belief inference, selective plan-latent request/reply, and decentralized execution. Both robots share model weights but keep independent histories, message caches, random state, and cooldown state. Model inference never receives both robots' observations or simulator truth.

The information contract and VPI derivation are documented in [docs/V1_DESIGN_DECENTRALIZED_ZH.md](docs/V1_DESIGN_DECENTRALIZED_ZH.md).

The version-isolated block-causal Research-v2 implementation is documented in [docs/V2_IMPLEMENTATION_RESEARCH_ZH.md](docs/V2_IMPLEMENTATION_RESEARCH_ZH.md).

## Documentation

Files in `docs/` follow `VERSION_TYPE_DESCRIPTION` naming. The version is one of
`V1`, `V1.4`, or `V2`; the type identifies the document's purpose.

| Version | Type | Document |
|---|---|---|
| V1 | Design | [Decentralized information contract](docs/V1_DESIGN_DECENTRALIZED_ZH.md) |
| V1 | Design | [Model signal flow](docs/V1_DESIGN_MODEL_SIGNAL_FLOW_ZH.md) |
| V1 | Guide | [Training quickstart](docs/V1_GUIDE_TRAINING_QUICKSTART_ZH.md) |
| V1 | Guide | [Private Gates collection and acceptance](docs/V1_GUIDE_PRIVATE_GATES_ZH.md) |
| V1 | Audit | [Manual acceptance audit](docs/V1_AUDIT_ACCEPTANCE_ZH.md) |
| V1 | Manifest | [Frozen baseline manifest](docs/V1_MANIFEST_BASELINE.json) |
| V1.4 | Plan | [Research upgrade plan](docs/V1.4_PLAN_RESEARCH_UPGRADE_ZH.md) |
| V2 | Implementation | [Research-v2 implementation](docs/V2_IMPLEMENTATION_RESEARCH_ZH.md) |
| V2 | Report | [Stage A report](docs/V2_REPORT_STAGE_A_ZH.md) |

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

Research-v2 uses separate data and artifact roots and never consumes the V1 checkpoints above:

```bash
# This workspace's Python environment
source /home/jeong/miniconda3/etc/profile.d/conda.sh
conda activate wam-py311

# Fast contract smoke test (CPU)
python scripts/collect_research_v2_dataset.py \
  --out-dir datasets/research_v2_smoke --smoke
python scripts/train_research_v2_pipeline.py \
  --dataset-root datasets/research_v2_smoke \
  --out-dir checkpoints/research_v2_smoke --smoke --device cpu

# Formal collection (6400/800/800 plus a 100-episode pilot)
python scripts/collect_research_v2_dataset.py \
  --out-dir datasets/research_v2 --workers 16

# Formal training; the default profile is tuned for one RTX 5090 and 64 GB RAM
python scripts/train_research_v2_pipeline.py \
  --dataset-root datasets/research_v2 \
  --out-dir checkpoints/research_v2 \
  --profile rtx5090 --device cuda
```

Both formal commands are restartable. Add `--resume` after an interruption; the
collector audits existing HDF5 files before filling holes, and the trainer
restores model, optimizer, scaler, early-stopping, and RNG state. The RTX 5090
profile automatically selects BF16, TF32, fused AdamW, pinned/persistent data
workers, stage-specific microbatches, and three independently seeded block-world
members. See [docs/V2_IMPLEMENTATION_RESEARCH_ZH.md](docs/V2_IMPLEMENTATION_RESEARCH_ZH.md)
for the exact profile and artifact layout.

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

Runtime loading is provided by `policies.runtime.DecentralizedRuntime`. Detailed smoke, full-data, resume, GPU, and evaluation commands are in [docs/V1_GUIDE_TRAINING_QUICKSTART_ZH.md](docs/V1_GUIDE_TRAINING_QUICKSTART_ZH.md).

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
