# Strict-local LatentToM baseline

This directory is a reproducible adaptation of
[LatentToM](https://github.com/StanfordMSL/LatentToM) for RoboFactory
multi-arm control. It uses one shared checkpoint for every arm. At deployment,
one actor consumes only its own RGB history, 9D joint position history, and a
six-way task identifier and emits only that actor's 8D joint-position action.
There is no peer observation, joint state/action, communication latent, planner
state, or privileged simulator input. Batching actors during inference is only
a throughput optimization: rows do not interact, and the runtime isolation
audit checks this property.

## Reported run

The paper reports the final EMA checkpoint trained from the official LatentToM
commit `a51d929027799a53d54e7d7d2ba90e2703642b4a`. Its SHA-256 is
`144956453881db766e3e92cece76a03f28db27210e0cd53f74059c240bd3ed8f`.
Weights are deliberately not included in this repository. The corrected
Validation20 summary SHA-256 is
`5f34bf465faa6ea12eb52f56e9a2b7a4db3af9fe7ad1ae201337c627c6733f93`.

| Task | Successes / 20 | SR |
|---|---:|---:|
| Lift Barrier | 20/20 | 100.0% |
| Camera Alignment | 7/20 | 35.0% |
| Long Pipeline Delivery | 19/20 | 95.0% |
| Take Photo | 7/20 | 35.0% |
| Pass Shoe | 19/20 | 95.0% |
| Place Food | 15/20 | 75.0% |
| Mean / total | 87/120 | 72.5% |

This is a fixed-seed closed-loop diagnostic, not a held-out evaluation: all
900 available demonstrations are used for training. The six simulator tasks
use seeds 20260820--20260839, one seed per episode index, and CPU simulation.

## Fixed protocol

[`fixed_params.env`](fixed_params.env) is the single source for launch-time
defaults. The reported values are seed 20260822; 300,000 updates; batch 512;
gradient accumulation 1; 16 data/cache workers; 320x240 RGB; two observation
frames; action horizon 40 and action dimension 8. Optimisation is bfloat16
autocast with AdamW (learning rate `1e-4`, betas `(0.95, 0.999)`, weight decay
`1e-6`), gradient-norm clip 1.0, a 500-step linear warm-up then cosine decay,
and EMA with `inv_gamma=1`, `power=.75`, and `max_value=.9999`.

Validation runs 20 episodes per task, uses a base seed of 20260820, DDIM with
20 denoising steps, and replans every eight control steps. Actions are
standardized by global dataset mean and standard deviation; therefore
`DDIMScheduler(clip_sample=False)` is mandatory. Setting it to `True` clips
valid standardized actions to `[-1, 1]` (including required gripper targets),
which produced the superseded 0/120 evaluator result. This is an evaluator
correction only; the reported checkpoint was not retrained.

## Dependencies and data

Use Python 3.11+ with PyTorch CUDA, `diffusers`, `torchvision`, `einops`,
`h5py`, `gymnasium`, ManiSkill/RoboFactory, and the official LatentToM source
on `PYTHONPATH`. Pin RoboFactory to the version used by your project and keep
the configurations under `robofactory/configs/table` available.

The downloader in `../vla_baselines/download_datasets.py` fetches and
byte-verifies 150 episodes from each pinned Hugging Face revision:

- `zeno-ai/robofactory-lift-barrier-multiview`
- `zeno-ai/robofactory-camera-alignment-multiview`
- `zeno-ai/robofactory-long-pipeline-delivery-multiview`
- `zeno-ai/robofactory-take-photo-multiview`
- `zeno-ai/robofactory-pass-shoe-multiview`
- `zeno-ai/robofactory-place-food-multiview`

Put a Hugging Face token in a permission-restricted file and set
`BWA_HF_TOKEN_FILE` to it. Never commit the token. The dataset layout expected
by the launch scripts is `$BWA_DATASET_ROOT/<task>/training_manifest.json` plus
the referenced HDF5 files.

## Reproduction

The supervisor is intentionally crash-resumable. It runs one pipeline stage at
a time, waits for required GPUs to be free, starts a process group, validates
declared artifacts before advancing, and records atomic receipts. It only
terminates its own process group on shutdown.

For the original A100-style layout, clone `before-we-act`, `LatentToM`, and
`RoboFactory` under `/workspace/repos`, then run:

```bash
export BWA_LATENT_TOM_ROOT=/workspace/repos/before-we-act/deployment/latent_tom_local
export BWA_LATENT_TOM_PIPELINE="$BWA_LATENT_TOM_ROOT/pipeline.json"
export BWA_LATENT_TOM_STATE_ROOT=/workspace/bwa_latent_tom_runs/supervisor
export BWA_HF_TOKEN_FILE=/workspace/.secrets/hf_token
bash "$BWA_LATENT_TOM_ROOT/bwa-latent-tom-supervisor.sh"
```

The checked-in pipeline preserves the reported 18-stage `/workspace` sequence,
including the corrected validation and strict-local runtime audit. For another
layout, either retain that layout in a container or copy the pipeline and
replace its absolute paths; the individual launch scripts accept
`BWA_LATENT_TOM_ROOT`, `BWA_LATENT_TOM_UPSTREAM`, `BWA_ROBOFACTORY_ROOT`,
`BWA_DATASET_ROOT`, `BWA_LATENT_TOM_OUTPUT`, and `BWA_PYTHON_BIN`. To run only
the corrected evaluator after training, set `BWA_LATENT_TOM_CHECKPOINT` and run
`bash run_validation20_fixed.sh` from this directory.
