# RDT-1B on MARS-Control

This directory is the reproducibility entry point for the completed
decentralized, full-parameter RDT-1B MARS-Control run. The formal run used RDT
commit `cd79363a1387e8f81c7724d070ef7e45fd23150f`, four RTX PRO 6000 GPUs,
600 successful demonstrations (150 per task), and no train--test split.

## 1. Prepare the checkouts

The scripts use the canonical paths below (the same paths used for the formal
run):

```bash
mkdir -p /workspace/repos
git clone https://github.com/thu-ml/RoboticsDiffusionTransformer.git \
  /workspace/repos/rdt-1b
git -C /workspace/repos/rdt-1b checkout \
  cd79363a1387e8f81c7724d070ef7e45fd23150f
git -C /workspace/repos/rdt-1b apply \
  /workspace/repos/before-we-act/patches/rdt1b_mars_control_formal.patch

git clone https://github.com/MARS-EAI/RoboFactory.git \
  /workspace/repos/RoboFactory
git -C /workspace/repos/RoboFactory checkout \
  2d34fb38c80cb06550a5dbf99abac2c89f4336ed
```

The patch is hash-bound in `patches/README.md`. It contains the upstream RDT
dataset adapter and generated binary language-embedding artifact; it does not
contain credentials or model weights.

## 2. Prepare data and environment

Create the Python environment expected by the launcher using
`requirements.rdt_mars.txt` (the formal versions are also recorded in
`configs/rdt/mars_control_rdt1b_full_data_v1.json`). Install the CUDA-enabled
PyTorch wheel first when your package index does not provide the `+cu128`
variant. Store the
Hugging Face credential only in a mode-0600 file:

```bash
install -d -m 700 /workspace/.secrets
umask 077
printf '%s\n' "$HUGGINGFACE_HUB_TOKEN" > /workspace/.secrets/hf_token
```

Do not commit that file. The supervisor downloads and verifies all four task
datasets, computes full-corpus statistics, prepares task language embeddings,
and audits the local observation/action tensor contract before training.

## 3. Run the crash-resumable pipeline

Run the supervisor from the `before-we-act` checkout:

```bash
cd /workspace/repos/before-we-act
python deployment/rdt_mars/supervisor.py
```

The pipeline stages are download, configure, audit, statistics, language,
assets, smoke training, smoke checkpoint audit, smoke Validation20, formal
training, formal checkpoint audit, and four-task Validation20. It assigns GPUs
explicitly, retries failed stages with immutable commands, resumes the latest
complete checkpoint, and writes receipts under
`/workspace/runs/rdt_mars/`. To stop safely, send SIGTERM to the supervisor;
the active process group is terminated and the next invocation resumes from
the last complete stage.

For a short preflight-only run, stop after `smoke_validation20` or set
`RDT_MAX_TRAIN_STEPS=2` for a local smoke invocation. The formal target is
300,000 optimizer steps with per-device batch 4 (effective global batch 16).

## 4. Reproduce or inspect outputs

The final checkpoint is `formal/checkpoints/checkpoint-300000`. The strict
local-only evaluator is `run_validation.py`; it handles task-specific episode
limits (500/500/1200/800), five DDPM denoising steps, one-control-step
replanning, temporal-ensemble decay 0.01, and the inverse action codec. The
machine-readable parameter and artifact contract is
`configs/rdt/mars_control_rdt1b_full_data_v1.json`; verify it with:

```bash
python deployment/rdt_mars/audit_frozen_config.py
```
