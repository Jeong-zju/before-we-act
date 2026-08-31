# LatentToM on DuoBench

This pipeline trains one shared LatentToM checkpoint on all 550 official
DuoBench demonstrations and then runs fixed-seed Validation20 over all eleven
tasks. Each actor receives the public head stream, its own wrist stream, its
own 8-D proprioception, the fixed task id, and its own two-way arm id. The
deployed policy graph never receives the partner wrist, partner
proprioception/action, partner arm id, or simulator state.

The data boundary reuses `deployment.duo_act.prepare` so lag-1 recording
alignment, controller-equivalent absolute joint targets, binary grippers,
camera resizing, normalization population, source revisions, and per-task
rollout horizons stay identical to the completed ACT entry. LatentToM retains
its two-frame observation, 40-action horizon, ResNet-18/local-ToM/UNet model,
100-step diffusion training process, DDIM-100 inference with sample clipping
enabled, EMA, AdamW, 500-step warmup, and cosine schedule. Formal training
loads `duobench_latent_tom_v1.json`; the independently frozen, implementation-
level audit mirror (including source defaults and artifact provenance) is
`duobench_latent_tom_complete_config_v1.json` (SHA-256
`f451281863d3f04eaa83ed9f171c3adc733795f2ef2fba9a12b79573d35c04ff`).

The frozen budget is 60,000 updates at global batch 64: 3.84M samples and
6.7265 equivalent traversals of 570,876 causal arm-local samples. A fixed final
checkpoint is used; no validation result selects a checkpoint or hyperparameter.

`supervisor.py` owns only the process groups it starts. It waits for GPU 0,
downloads and verifies the pinned dataset, prepares and audits it, runs CUDA
training plus checkpoint/isolation smoke tests, performs a two-step rollout
smoke on every task, resumes full training, then launches Validation20 with
the eleven frozen task-specific horizons. Atomic stage receipts make the
entire sequence restartable.

## Reproduction

On the prepared server image, put the Hugging Face credential in
`/workspace/.secrets/hf_token` and install the repository plus its pinned
LatentToM, DuoBench, and RobotControlStack source revisions. Then run:

```bash
cd /workspace/repos/before-we-act
DUO_LATENT_TOM_REPO=$PWD \\
  DUO_LATENT_TOM_RUN=/workspace/runs/duobench-latent-tom \\
  DUO_LATENT_TOM_PREPARED=/workspace/runs/duobench-latent-tom/data \\
  bash deployment/duo_latent_tom/launch.sh
```

The launcher performs download, all-data preparation, audits, smoke tests,
60,000-update training, checkpoint checks, and 20-rollout validation for every
task. The formal run uses `duobench_latent_tom_v1.json`; the complete contract
and source-file hashes are in `duobench_latent_tom_complete_config_v1.json`.
The small JSON files under `receipts/` are hash-bound results from the
completed server run; the large checkpoint is intentionally not vendored.
