# DuoBench π0.5 LoRA reproduction

This adapter keeps the pinned upstream OpenPI π0.5 model and only adds a
DuoBench data/interface layer. The supervisor downloads the full Hugging Face
snapshot, prepares all 550 simulation demonstrations, computes exact corpus
statistics, runs interface and four-GPU collective smoke tests, trains the
25,000-update LoRA policy, then runs Validation20 (20 episodes × 11 tasks).

The decentralized request boundary is deliberately strict: one request has
only `head`, that arm's `wrist`, that arm's 8-D state, and the shared task
prompt. The peer wrist slot is masked. Actions are converted to seven-joint
delta plus absolute binary gripper for the upstream model and inverted exactly
once at inference; MuJoCo FR3 actuator ctrlrange is the only action saturation.

## Frozen result and contract

The formal policy uses the upstream OpenPI revision
`15a9616a00943ada6c20a0f158e3adb39df2ccac`, the released `pi05_base`
checkpoint, `gemma_2b_lora`, `gemma_300m_lora`, action dimension 32, and
horizon 16. It trains on all 550 simulation demonstrations without a split for
25,000 updates with global batch 128 and four-way JAX data parallelism. The
machine-readable source of truth is
`configs/pi05_duobench_lora_formal_v1.json`; it records every resolved model,
data, normalization, optimizer, scheduler, loader, runtime, and checkpoint
field together with source and artifact hashes.

The fixed step-24,999 checkpoint produced 85/220 successes (38.64% macro SR)
under Validation20. The final report binds that result to checkpoint tree SHA
`61c582c6...`, parameter identity SHA `77df0f56...`, normalization SHA
`e3dd6de3...`, and evaluator revision
`duobench-pi05-lora-decentralized-lag1-v1`.

## Repository layout

- `openpi_overlay/`: three new OpenPI modules plus the minimal registry/data
  loader patch. No upstream model module is replaced.
- `download.py`, `compute_norm.py`, `audit_contract.py`: pinned data download,
  exact normalization, and information-boundary/temporal audits.
- `train_stage.py`: smoke/formal OpenPI launcher and checkpoint receipt.
- `rpc_server.py`, `evaluate.py`, `validation_launcher.py`: split-environment
  policy RPC and four-GPU Validation20 waves.
- `supervisor.py`: crash-resumable orchestration and foreign-GPU-PID guard.
- `verify_release.py`: offline source/config audit, optionally checked against
  a copied `final_report.json`.

## Environment setup

Use separate environments. OpenPI is pinned to JAX 0.6.2 and NumPy 1.26.4;
DuoBench/RCS uses its own binary-compatible simulator environment. This split
is required because loading RCS source paths in the OpenPI process can shadow
the compiled `rcs._core` wheel, while pickled NumPy arrays are not ABI-stable
across the two NumPy versions. The RPC therefore transfers image, state, and
action arrays as raw bytes.

Clone the pinned upstream, apply the shared π0.5 RoboFactory/MARS base overlay,
then apply the DuoBench-only increment. Both installers verify the same pinned
OpenPI commit; this ordering is required because the DuoBench registry patch
extends the shared dataset registry:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git /workspace/repos/openpi
git -C /workspace/repos/openpi checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
deployment/pi05_reproduction/openpi_patch/apply.sh /workspace/repos/openpi
deployment/duo_pi05/openpi_overlay/apply.sh /workspace/repos/openpi
```

Install OpenPI and the benchmark according to their pinned upstream lockfiles.
The completed run used Python 3.11.16, JAX 0.6.2, NumPy 1.26.4, Flax 0.10.2,
Optax 0.2.4, and Orbax Checkpoint 0.11.13 on four RTX PRO 6000 Blackwell GPUs.
Set the paths accepted by `run_supervisor.sh` if your layout differs from
`/workspace`.

Store a Hugging Face token in a mode-600 file outside the repository; never
commit it:

```bash
install -m 600 /dev/null /workspace/.secrets/hf_token
# Write your token interactively to /workspace/.secrets/hf_token.
```

## End-to-end reproduction

The crash-resumable entrypoint is:

```bash
deployment/duo_pi05/run_supervisor.sh
```

For a managed server, install `duobench-pi05-supervisor.conf` under the local
Supervisor configuration directory and update Supervisor. The pipeline runs
download, preparation, normalization/audit, four-GPU and closed-loop smoke
tests, formal training, checkpoint isolation, Validation20, and finalization.
It writes state, receipts, logs, checkpoints, per-task journals, and reports
under `${DUO_PI05_RUN:-/workspace/runs/pi05_duo}`. It never kills processes it
did not create and never stops/recycles/destroys the instance.

Expected terminal artifacts:

```text
state.json                                      # stage/status = complete
formal/status.json                              # 25,000 updates
formal/checkpoints/.../24999/                   # fixed final checkpoint
formal/validation20/summary.json                # 220 episodes
final_report.json
final_report.sha256.json
```

Audit a checkout before release, and optionally bind it to a copied report:

```bash
python -m deployment.duo_pi05.verify_release
python -m deployment.duo_pi05.verify_release --final-report /path/to/final_report.json
```

## Fixes preserved by this adapter

- Post-action recording semantics use observation row `i` to predict action
  row `i+1`; chunks never cross episodes and repeat the last valid tail action.
- Exact normalization uses every causal sample from all tasks and both arms,
  after canonical FR3 actuator clipping and the upstream delta transform.
- The DataLoader dataset pickles only its root/horizon and reopens large mmap
  arrays in spawned workers, avoiding serialization of the 121 GB image set.
- Simulator and OpenPI dependencies remain isolated; RPC tensors use raw bytes
  to avoid NumPy-1/NumPy-2 pickle ABI failures.
- The temporal ensemble retains a `deque` but materializes retained entries
  before indexing, avoiding unsupported deque slicing.
- Validation uses each task's native maximum step count and resumable per-task
  JSONL journals, and pins the checkpoint/evaluator identity on every episode.
- Both arms share one checkpoint, but each request contains exactly head RGB,
  its own wrist RGB, its own state8, and task text; peer and privileged fields
  are rejected.
