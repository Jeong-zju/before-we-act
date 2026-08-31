# LatentToM on DuoBench

This pipeline trains one shared LatentToM checkpoint on all 550 official
DuoBench demonstrations and then runs fixed-seed Validation20 over all eleven
tasks. Each actor receives the public head stream, its own wrist stream, its
own 8-D proprioception, and the fixed task id. The policy graph never receives
the partner wrist, partner proprioception/action, arm id, or simulator state.

The data boundary reuses `deployment.duo_act.prepare` so lag-1 recording
alignment, controller-equivalent absolute joint targets, binary grippers,
camera resizing, normalization population, source revisions, and per-task
rollout horizons stay identical to the completed ACT entry. LatentToM retains
its two-frame observation, 40-action horizon, ResNet-18/local-ToM/UNet model,
100-step diffusion training process, DDIM-20 inference with clipping disabled,
EMA, AdamW, 500-step warmup, and cosine schedule.

The frozen budget is 50,000 updates at batch 64: 3.2M samples and 5.6054
equivalent traversals of 570,876 causal arm-local samples, matching the
completed DuoBench ACT sample budget. A fixed final checkpoint is used; no
validation result selects a checkpoint or hyperparameter.

`supervisor.py` owns only the process groups it starts. It waits for GPU 0,
downloads and verifies the pinned dataset, prepares and audits it, runs CUDA
training plus checkpoint/isolation smoke tests, performs a two-step rollout
smoke on every task, resumes full training, then launches Validation20 with
the eleven frozen task-specific horizons. Atomic stage receipts make the
entire sequence restartable.
