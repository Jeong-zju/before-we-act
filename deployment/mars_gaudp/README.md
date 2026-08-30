# GauDP on MARS-Control

This directory freezes the exact shared-weight, decentralized GauDP adaptation
used for the reported MARS-Control result. Every arm runs the same policy and
receives only its own three-frame RGB, Gaussian, and 9-D qpos histories. It emits
only its own absolute 8-D `pd_joint_pos` action.

## Reproduction contract

`mars_control_gaudp_v1.json` is the source of truth for the data, model,
optimizer, EMA, cache, checkpoint, and Validation20 settings. Before a run, the
supervisor verifies this contract and the SHA-256 of all critical adaptation and
upstream GauDP sources. After training it also verifies the checkpoint's embedded
policy contract, update count, and core configuration.

The reported result is archived in `reported_validation20.json`: 0/20 on each of
the four tasks, or 0/80 overall. This negative result is retained intentionally.

## Critical fixes relative to the original adaptation

1. `precompute.py` divides uint8 RGB by 255 before the `[-1,1]` NoPoSplat
   affine transform. The old path operated on `[0,255]` values.
2. Gaussian cache generation and online validation both use FP32. The cache is
   stored as float32 and carries a versioned preprocessing schema.
3. `cache_parity.py` reconstructs Gaussian features from the same raw RGB and
   rejects cache/online drift. The formal run passed with MAE/RMSE/max error 0
   and correlation 1.0.
4. Validation reproduces the training codec exactly:
   `256x256 -> Gaussian 30x40 -> policy 120x160`. Direct `256 -> 120x160`
   resizing is not equivalent.
5. Closed-loop execution replans every simulator control step and temporally
   ensembles every predicted chunk covering the current step with decay 0.01.
   The earlier six-step open-loop executor was not the GauDP execution contract.
6. `compare_inference.py` runs one full episode per task at both 20 and 100
   denoising steps before Validation20. The reported confirmation uses 20 steps.
7. Task-specific episode limits are fixed at 500, 500, 1200, and 800 steps.

## Expected layout

The defaults match the original server, but every path can be overridden:

```text
/workspace/repos/before-we-act
/workspace/repos/Policy-Lightning
/workspace/repos/RoboFactory
/workspace/datasets/mars_control
/workspace/runs/mars_gaudp_fp32_v2
```

Required environment variables are accepted as overrides:

```bash
export MARS_GAUDP_REPO=/workspace/repos/before-we-act
export MARS_GAUDP_ROBOFACTORY=/workspace/repos/RoboFactory
export MARS_GAUDP_DATA_ROOT=/workspace/datasets/mars_control
export MARS_GAUDP_RUN_ROOT=/workspace/runs/mars_gaudp_fp32_v2
export MARS_GAUDP_CACHE_ROOT=$MARS_GAUDP_RUN_ROOT/cache
export MARS_GAUDP_WEIGHT=/workspace/repos/Policy-Lightning/weights/re10k.ckpt
export MARS_GAUDP_PYTHON=/venv/main/bin/python
```

`PYTHONPATH` must expose this repository, Policy-Lightning, and RoboFactory's
Diffusion-Policy package. The supervisor sets it for every child stage.

## One-command pipeline

Run from the repository root:

```bash
bash deployment/mars_gaudp/mars-gaudp-supervisor-v2.sh
```

The resumable supervisor owns GPU 0 and executes:

1. frozen-config and CUDA preflight checks;
2. FP32 Gaussian cache generation;
3. raw-RGB/cache/online parity gate;
4. data, normalization, and decentralization audit;
5. smoke training and four-task smoke validation;
6. 60,000-update formal training and checkpoint audit;
7. 20-vs-100-step full-episode comparison;
8. four-task Validation20;
9. hash-bound final report generation.

Each completed stage writes a receipt under `$MARS_GAUDP_RUN_ROOT/receipts`.
Logs are under `$MARS_GAUDP_RUN_ROOT/logs`, and the current supervisor state is
stored atomically in `$MARS_GAUDP_RUN_ROOT/state.json`. A failed stage is retried
without discarding completed receipts.

## Standalone audits

Static source/config audit:

```bash
python -m deployment.mars_gaudp.verify_frozen_config
```

Checkpoint-bound audit:

```bash
python -m deployment.mars_gaudp.verify_frozen_config \
  --checkpoint /workspace/runs/mars_gaudp_fp32_v2/formal/last.pt
```

The NoPoSplat weight must have SHA-256
`60d537c3d79554fe9954ac2bc277a6800bd4b56d682f6df90aa576f27dd50f07`.
The reported GauDP checkpoint has SHA-256
`b1bcdd11bce4f7aeb3d141bb9e1f1a4741608440b2320e354d29f539a7c00ec4`.
