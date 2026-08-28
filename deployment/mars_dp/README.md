# MARS-Control Diffusion Policy (v3)

This directory contains the corrected, strictly decentralized Diffusion Policy
used for the MARS-Control Validation20 result. The frozen policy contract and
all training/evaluation values are machine-readable in
[`configs/dp/mars_control_full_data_v3.json`](../../configs/dp/mars_control_full_data_v3.json).

## Contract

- One shared checkpoint is used by every arm and all four tasks.
- Each arm receives only its own head-camera RGB history and its own commanded
  8-D state (seven joint targets plus gripper command). Peer/global RGB or
  state, task/arm identity, language, and joint actions are not inputs.
- The temporal window is `obs=[t-2,t-1,t]`, prediction horizon 8, and the
  policy executes `prediction.action == action_pred[:,2:]`, six controls before
  replanning.
- Absolute `pd_joint_pos` targets are clipped to RoboFactory bounds before
  corpus statistics and training. The per-dimension corpus min/max codec maps
  to `[-1,1]`; evaluation applies the inverse exactly once and then clips to
  environment bounds.

## Reproduce from a prepared environment

The original run used Python at `/venv/main/bin/python`, the RoboFactory
checkout at `/workspace/repos/RoboFactory`, and the four-task dataset at
`/workspace/datasets/mars_control`. The dataset must contain 150 successful
demonstrations per task. Install the upstream RoboFactory Diffusion-Policy
dependencies and make the three source roots importable, for example:

```bash
export MARS_DP_REPO="$PWD"
export MARS_DP_ROBOFACTORY=/workspace/repos/RoboFactory
export MARS_DP_DATA_ROOT=/workspace/datasets/mars_control
export MARS_DP_RUN_ROOT=/workspace/runs/mars_dp_v3
export MARS_DP_PYTHON=/venv/main/bin/python
export PYTHONPATH="$MARS_DP_REPO:$MARS_DP_ROBOFACTORY:$MARS_DP_ROBOFACTORY/robofactory/policy/Diffusion-Policy"
```

The pinned Hugging Face source repositories and revisions are listed in the
v3 JSON manifest. To download only the ten promoted formal shards per task,
use the repository's downloader (the downloader writes and verifies a receipt
for each task; keep the token out of the repository):

```bash
export MARS_ACT_DATA_ROOT="$MARS_DP_DATA_ROOT"
export HF_TOKEN_FILE=/path/to/huggingface-token
$MARS_DP_PYTHON deployment/mars_act/download.py
```

Do not use a broad repository glob: the revisions also contain failed/timeout
`.parts` fragments that are intentionally excluded from the formal corpus.

Run the crash-resumable supervisor:

```bash
bash deployment/mars_dp/reproduce_mars_dp_v3.sh
```

The wrapper verifies the checked-in manifest and v3 source hashes, then runs,
in order, a CUDA/dependency preflight, dataset audit,
temporal/normalization contract test, 10-update smoke training and smoke
rollout, a 5k diagnostic checkpoint, the formal 60,000-update training, all
four Validation20 task runs, and finalization. It writes stage logs and JSON
artifacts below `$MARS_DP_RUN_ROOT`; rerunning the command resumes completed
training and evaluation stages. GPU use is restricted to
`CUDA_VISIBLE_DEVICES=0` and is released when the process exits.

For an individual stage, the underlying entry points are:

```bash
$MARS_DP_PYTHON -m deployment.mars_dp.audit
$MARS_DP_PYTHON -m deployment.mars_dp.preflight_v2
$MARS_DP_PYTHON -m deployment.mars_dp.train \
  --data-root "$MARS_DP_DATA_ROOT" --output "$MARS_DP_RUN_ROOT/formal" \
  --steps 60000 --batch-size 64 --workers 8
$MARS_DP_PYTHON -m deployment.mars_dp.run_validation \
  --checkpoint "$MARS_DP_RUN_ROOT/formal/last.pt" \
  --output-root "$MARS_DP_RUN_ROOT/validation20" \
  --robofactory-root "$MARS_DP_ROBOFACTORY"
```

The formal run is intentionally fixed to the final update-60,000 checkpoint;
no test result is used for checkpoint selection. The published run artifact
was `/workspace/runs/mars_dp_v3/formal/last.pt` with SHA-256
`e8c65a0e434b4c36c749b8393320524098e24c13c456cd85d25c4a311f9bfe9a`.

## Expected result artifact

`final_report.json` and `validation20/summary.json` should report 600 training
demonstrations, 60,000 updates, four tasks × 20 episodes, and the evaluator
revision `dp-official-obs3-horizon8-exec6-command-state-topp-v6`. The completed
reference run obtained 0/20 on each task (0/80 total); this is a recorded
closed-loop outcome, not a skipped evaluation.

The older `mars_control_full_data_v1.json` and `supervisor.py` are retained as
historical v1 artifacts. Use the v3 manifest and `supervisor_v3.py` for the
reported result.

Small, non-secret receipts from the reference run are archived under
[`docs/reproducibility/mars_dp_v3/`](../../docs/reproducibility/mars_dp_v3/):
dataset audit, temporal/normalization preflight, training status, normalization
statistics, Validation20 summary, and the final report. The model checkpoint is
not vendored because it is large; verify a separately obtained checkpoint
against the SHA-256 in the manifest before evaluation.
