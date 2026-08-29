# ManiFlow on MARS-Control: reproducible runbook

This directory contains the complete MARS-Control adaptation of ManiFlow.  The
pipeline is decentralized: one frozen checkpoint is shared by every arm, while
each arm supplies only its own RGB observation and 9-D proprioception and emits
its own 8-D absolute `pd_joint_pos` target.  No task, arm, peer observation, or
global action is passed to the policy.

The implementation is intentionally contract-driven.  The frozen contract is
[`mars_control_maniflow_v1.json`](mars_control_maniflow_v1.json); the code refuses
to train or evaluate if its task order, optimizer budget, normalization, action
contract, or Validation20 settings drift.

## Pinned sources and environment

The recorded run used:

| Component | Pin |
|---|---|
| ManiFlow upstream | `https://github.com/geyan21/ManiFlow_Policy`, commit `ef2f116f1f90163ed36e657b8c5503740bb468af` |
| RoboFactory | commit `2d34fb38c80cb06550a5dbf99abac2c89f4336ed` |
| Python | `/venv/main/bin/python` |
| GPU | one `cuda:0` device (`CUDA_VISIBLE_DEVICES=0`) |

The upstream workspace needs the image-policy patch in
[`../../patches/maniflow_image_workspace_no_pytorch3d.patch`](../../patches/maniflow_image_workspace_no_pytorch3d.patch).
It replaces the point-cloud workspace policy with
`ManiFlowTransformerImagePolicy`; no checkpoint is stored in Git.

On the reference server the repositories were checked out as:

```bash
git clone https://github.com/geyan21/ManiFlow_Policy /workspace/repos/ManiFlow_Policy
git -C /workspace/repos/ManiFlow_Policy checkout ef2f116f1f90163ed36e657b8c5503740bb468af
git clone https://github.com/MARS-EAI/RoboFactory.git /workspace/repos/RoboFactory
git -C /workspace/repos/RoboFactory checkout 2d34fb38c80cb06550a5dbf99abac2c89f4336ed
git -C /workspace/repos/ManiFlow_Policy/ManiFlow apply \
  /path/to/patches/maniflow_image_workspace_no_pytorch3d.patch
```

Install the project dependencies in the same environment as RoboFactory and
ManiFlow.  The runner sets `PYTHONPATH` to this repository, RoboFactory, and
`/workspace/repos/ManiFlow_Policy/ManiFlow`.

## Data download and audit

Set a Hugging Face token through the environment or a file; never commit it:

```bash
export HF_TOKEN=...                 # or write it to /workspace/.secrets/hf_token
source /venv/main/bin/activate
python -m deployment.mars_maniflow.download \
  --data-root /workspace/datasets/mars_control
python -m deployment.mars_maniflow.audit \
  --data-root /workspace/datasets/mars_control \
  --output /workspace/runs/mars_maniflow/audit.json \
  --stats /workspace/runs/mars_maniflow/normalization.json
python -m deployment.mars_maniflow.contract_test \
  --data-root /workspace/datasets/mars_control \
  --stats /workspace/runs/mars_maniflow/normalization.json \
  --output /workspace/runs/mars_maniflow/contract_test.json
```

The four pinned Hugging Face revisions and the expected ten HDF5 shards per task
are defined in `common.py`.  All 600 demonstrations (150 per task) are used;
there is no train/test split.  The resulting corpus has 1,650 arm-local streams
and 1,035,318 indexed local timesteps.  RGB is `uint8 HWC -> float32 / 255 ->
bilinear 224`; qpos and actions use global-corpus min/max statistics.  Actions
are clipped to RoboFactory's `pd_joint_pos` limits *before* statistics and again
after inverse decoding in the evaluator.  The exact recorded statistics are in
[`normalization.json`](../../docs/reproducibility/mars_maniflow/normalization.json).

## Smoke gate, training, and supervisor

Run the crash-resumable one-GPU supervisor after setting the paths appropriate to
the host:

```bash
export MARS_MANIFLOW_RUN_ROOT=/workspace/runs/mars_maniflow
export MARS_MANIFLOW_DATA_ROOT=/workspace/datasets/mars_control
export ROBOFACTORY_ROOT=/workspace/repos/RoboFactory
python -m deployment.mars_maniflow.supervisor
```

The supervisor executes preflight, resumable download, audit, contract test,
two-update smoke training, one-episode smoke validation, formal training, and
Validation20 in order.  Each stage writes a receipt and can be resumed safely;
the GPU is explicitly pinned to `cuda:0`.  For a system supervisor, install
`mars-maniflow-fixed-supervisor.conf` and use `supervisorctl` to start/stop
`mars_maniflow_fixed`.  The standalone Validation20 runner is
`mars-maniflow-validation20-v3.sh`.

The formal frozen optimization is 60,000 updates, batch 128, one accumulation,
16 workers, AdamW (`lr=1e-4`, betas `(0.9,0.95)`, weight decay `1e-3`), cosine
schedule with 500 warm-up updates, bfloat16 autocast, TF32, gradient clipping at
10, seed `20260822`, and a final EMA checkpoint (saved every 5,000 updates).
The complete parameter table is the `optimization`, `model`, `ema`, and
`validation20` sections of the frozen JSON, so the table can be copied directly
into an appendix without relying on shell defaults.

## Validation20 contract

Each task has an explicit maximum episode length:

| Task | Maximum steps |
|---|---:|
| place cube in cup | 500 |
| strike cube hard | 500 |
| three robots place shoes | 1,200 |
| four robots stack cube | 800 |

The evaluator observes two frames (`t-1,t`), predicts a 15-action chunk every 8
steps, and applies temporal ensemble aggregation with exponential decay `0.01`.
At a chunk boundary, all still-valid overlapping predictions are blended, with
the newest chunk receiving weight 1.  The selected normalized action is inverse
min/max decoded exactly once and clipped to the environment bounds.  These
settings are checked by the contract test and recorded in every task receipt.

## Recorded result and artifact verification

The reference formal run used checkpoint SHA-256
`69a53b432b90aeebd286d7ec445a025264676ab4b81f86a02d5379008d14781e`.  Its
Validation20 result is 12/80 (macro success rate 15.0%): place cube in cup 5/20,
strike cube hard 7/20, three robots place shoes 0/20, and four robots stack cube
0/20.  Per-episode receipts and the summary are in
[`docs/reproducibility/mars_maniflow/`](../../docs/reproducibility/mars_maniflow/).
The checkpoint itself is deliberately external; download it separately and
verify the SHA before evaluation.  `artifact_manifest.json` records hashes for
the frozen config, normalization, contract test, training receipt, and all
Validation20 receipts.

The JSON receipts are small, reviewable provenance artifacts; tokens, caches,
full logs, and large model weights are excluded from the repository.
