# DuoBench ACT reproduction

This directory is the reproducibility boundary for the completed DuoBench ACT
run.  The formal run is intentionally pinned to one shared, decentralized
ResNet-18 CVAE-ACT policy.  The policy sees the shared head camera, the acting
arm's wrist camera, its own 8-D proprioception, and a fixed task ID; it never
receives the other arm's observation or action.

## Source of truth

All training-affecting values are in
`configs/duobench_act_causal_lag1_prior_v1.json`.  Its SHA-256 is
`dd4b18826ca080a497db7c3facfc0dae99342215ea6f6d6f3e90cca3d58fdea7`.
`train.py --config` rejects byte-level config changes, CLI overrides that drift
from the freeze, stale dataset normalization/alignment, and model/dataset source
hash drift.  The pipeline contract is in `pipeline.json`.

The frozen data contract is DuoBench revision
`b741bc915d942ecadaefb4e3de6bbd716c1b8b1b`, DuoBench checkout
`082a57cdafea9db115029e6fe9e03691e755f93f`, and RCS converter revision
`4f78aeffae3bc4d0c02e7beab993e5406261dcf6`.  It uses all 50 demonstrations of
each of the 11 tasks (550 demonstrations, no held-out split), controller-
equivalent absolute joint targets, lag-1 observation/action alignment, and
population statistics computed over all causal pairs and both local arms.

## Canonical layout

The published commands use the same layout as the RTX 5090 run.  Other paths
may be used only by adding an explicit path-remapping layer outside the frozen
config; do not edit the frozen JSON when reporting the formal result.

```text
/workspace/repos/before-we-act       this repository
/workspace/repos/duobench            DuoBench checkout at the pinned revision
/workspace/repos/robot-control-stack RCS checkout at the pinned converter revision
/workspace/datasets/duobench         immutable Hugging Face dataset checkout
/workspace/datasets/duobench_assets  MuJoCo/XML assets used by the evaluator
/workspace/runs/duobench-act/data_unclipped
/workspace/runs/duobench-act/causal_lag1_prior
```

The dataset and simulator repositories are external dependencies and are not
copied into this repository.  Download them with a token supplied through the
environment; never commit a Hugging Face token:

```bash
git clone https://github.com/RobotControlStack/duobench.git /workspace/repos/duobench
git -C /workspace/repos/duobench checkout 082a57cdafea9db115029e6fe9e03691e755f93f
git clone https://github.com/RobotControlStack/robot-control-stack.git /workspace/repos/robot-control-stack
git -C /workspace/repos/robot-control-stack checkout 4f78aeffae3bc4d0c02e7beab993e5406261dcf6

HF_TOKEN="$HF_TOKEN" hf download RobotControlStack/duobench \
  --repo-type dataset \
  --revision b741bc915d942ecadaefb4e3de6bbd716c1b8b1b \
  --local-dir /workspace/datasets/duobench \
  --max-workers 8
```

Install the locked Python environment with the repository's `uv.lock` (the
formal run used Python 3.12, PyTorch 2.11.0+cu128, and one RTX 5090):

```bash
cd /workspace/repos/before-we-act
uv sync --frozen
source .venv/bin/activate
export PYTHONPATH="$PWD:/workspace/repos/duobench/src"
```

## Prepare, audit, smoke-test, train, validate

Preparation performs the controller-range canonicalization, decodes the
official already-224-square videos without a second resize, and writes the
normalization statistics and immutable receipts:

```bash
python -m deployment.duo_act.prepare \
  --dataset /workspace/datasets/duobench \
  --output /workspace/runs/duobench-act/data_unclipped \
  --image-size 224 --jobs 6
python -m deployment.duo_act.audit \
  --data /workspace/runs/duobench-act/data_unclipped \
  --output /workspace/runs/duobench-act/audit_unclipped.json
python -m deployment.duo_act.preflight \
  --data /workspace/runs/duobench-act/data_unclipped \
  --output /workspace/runs/duobench-act/preflight_state_binary.json
```

The mandatory smoke run is deliberately separate from the formal freeze:

```bash
python -m deployment.duo_act.train \
  --data /workspace/runs/duobench-act/data_unclipped \
  --output /workspace/runs/duobench-act/smoke \
  --updates 5 --batch-size 40 --workers 8 --horizon 100 --action-lag 1 \
  --save-every 5 --smoke
python -m deployment.duo_act.validation_launcher \
  --checkpoint /workspace/runs/duobench-act/smoke/final.pt \
  --data /workspace/runs/duobench-act/data_unclipped \
  --output /workspace/runs/duobench-act/smoke/validation \
  --episodes 1 --max-steps 2 --workers 1
```

After the smoke gates pass, the exact formal training command is only:

```bash
python -m deployment.duo_act.train \
  --config configs/duobench_act_causal_lag1_prior_v1.json
```

It resumes `latest.pt` with model/optimizer/scheduler state when present and
writes the fixed final checkpoint at update 50,000.  Validation20 is then run
with the task-specific maximum horizons from the prepared manifest:

```bash
python -m deployment.duo_act.validation_launcher \
  --checkpoint /workspace/runs/duobench-act/causal_lag1_prior/final.pt \
  --data /workspace/runs/duobench-act/data_unclipped \
  --output /workspace/runs/duobench-act/causal_lag1_prior/validation20_open30 \
  --episodes 20 --workers 3 --mode open30
```

The `supervisor.py` and the accompanying supervisor config execute these gates
in order.  They reserve GPU 0 exclusively for training and use up to three
validation simulator workers only after training exits.  A restarted supervisor
never falls back to the old 120k `formal/` experiment or the pre-causal clipped
dataset.

## Receipts and expected result

Small hash-bound receipts from the completed remote run are stored under
`receipts/` and verified by `receipts/SHA256SUMS`: data manifest/action audit, preflight, training progress, the
three-task action-execution ablation, and the all-task Validation20 summary.
The final checkpoint is intentionally not committed to Git; its expected
SHA-256 is
`04294c5b0aa42b8c53701c50c1ff1ae47542d167c468f809d9f22962ae27db99`.
The expected Validation20 summary SHA-256 is
`d98758a78829808d27548ba825dba421ac3a28e59b8b38519445284e8b6f444d` and the
run reports 17/220 successes (7.73% macro-average) under `open30`.

Run the focused contract tests before publishing a result:

```bash
python -m pytest -q \
  tests/deployment/test_duo_act_frozen_config.py \
  tests/deployment/test_duo_action_target.py
```
