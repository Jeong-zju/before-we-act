# LatentToM on MARS-Control

This pipeline trains one shared, strictly decentralized LatentToM policy on all
600 official MARS-Control demonstrations and automatically runs Validation20 on
all four tasks. Every actor receives only its own two-frame head-camera RGB and
9D qpos history and emits only its own 8D action. Task IDs, arm IDs, peer inputs,
global observations and joint actions are excluded from the policy graph.

The frozen comparison budget is 60,000 optimizer updates at global batch 128
(7.68M local samples), no train/test split, task-balanced sampling, 40-step
action chunks, AdamW 1e-4, 500-step warmup plus cosine decay, bfloat16, EMA and
20-step DDIM inference with clipping disabled. Validation uses 20 deterministic
seeds per task and the official task-specific maximum horizons.

`mars_control_latent_tom_v1.json` is the machine-readable single source of
truth for every policy, optimization, loader, checkpoint, runtime, and
Validation20 parameter. Both training and evaluation construct the policy from
this file and reject command-line or checkpoint configuration drift. The JSON
also binds the reported artifact by its SHA-256 checksum.

Run `python -m deployment.mars_latent_tom.verify_frozen_config` to reconstruct
the policy, check parameter counts and budget arithmetic, and strictly load and
hash-verify the reported checkpoint when it is present.

The `mars-latent-tom` system supervisor owns the pipeline process group, retries
failed stages, resumes formal training from `last.pt`, and advances to closed-loop
validation only after download, contract audit, training smoke and rollout smoke
have succeeded.
