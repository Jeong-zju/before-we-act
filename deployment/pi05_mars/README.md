# π0.5 on MARS-Control

This directory is the code-level reproduction record for the formal π0.5
LoRA run on the four-task MARS-Control benchmark. The code here is synchronized
with the server that produced the published result; the source hashes are
recorded in `configs/pi05_mars_control_lora_v1.json`.

## Frozen sources and protocol

- OpenPI: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- RoboFactory/MARS-Control: `2d34fb38c80cb06550a5dbf99abac2c89f4336ed`
- Data: four pinned Hugging Face dataset revisions, 150 demonstrations each
- Training: all 600 demonstrations, no held-out split, 30,000 updates,
  global batch 128, four-way data parallelism
- Policy: one shared decentralized policy; each arm receives only its own head
  RGB, its own 9-D qpos, and the task prompt, and emits its own 8-D action
- Selection: fixed final checkpoint at step 29,999; no validation selection
- Validation: 20 fixed seeds per task with task-specific horizons

The exact optimizer, schedule, normalization statistics, action bounds,
dataset revisions, task seeds, horizons, package versions, and result receipt
hashes are in `../../configs/pi05_mars_control_lora_v1.json`.

## What each file preserves

- `download.py`: pinned dataset revisions, ten shards per task, checksummed
  download receipts, and no train/test split.
- `audit_norm.py`: fail-closed corpus/schema audit and the exact quantile
  normalization transform used by training.
- `supervisor.py`: four-GPU collective preflight, training/validation smoke
  gates, crash-safe receipts, 30k-step formal training, and Validation20.
- `rpc_server.py`: strict arm-local inference contract.
- `task_validate.py`: inverse action transform exactly once, environment-bound
  clipping, temporal ensembling, fixed seeds, and per-task maximum steps.
- `validation_launcher.py`: one policy/simulator pair per task and GPU.
- `run_supervisor.sh` and `pi05-mars-supervisor.conf`: process environment and
  supervised lifecycle.

The OpenPI registration, dataset loader, transforms, and train configuration
are vendored in `../pi05_reproduction/openpi_patch`.

## Reproduce

Create the same directory layout or adjust the absolute `/workspace` paths in
the supervisor wrappers:

```text
/workspace/repos/before-we-act
/workspace/repos/openpi
/workspace/repos/RoboFactory
/workspace/datasets/mars_control
/workspace/runs/pi05_mars
/workspace/venvs/openpi
/workspace/venvs/robofactory
```

Then:

1. Check out OpenPI and RoboFactory at the commits above.
2. Apply `../pi05_reproduction/openpi_patch/apply.sh` to the OpenPI checkout.
3. Install the normal OpenPI dependencies and the pinned Blackwell overlay in
   `../pi05_reproduction/openpi_patch/requirements-blackwell-cu129.txt`.
4. Install the RoboFactory simulator environment.
5. Put the Hugging Face token in `/workspace/.secrets/hf_token` with mode 0600.
   The token is a runtime secret and is intentionally absent from this repo.
6. Run `./run_supervisor.sh`, or install the supplied Supervisor config.

Every stage is receipt-gated under `/workspace/runs/pi05_mars`; an interrupted
formal run resumes from an existing numeric checkpoint, while a completed
stage is not repeated. The smoke gates must pass before formal training starts.

## Recorded Validation20 result

| Task | Successes |
|---|---:|
| Place Cube in Cup | 3/20 |
| Strike Cube Hard | 11/20 |
| Three Robots Place Shoes | 3/20 |
| Four Robots Stack Cube | 0/20 |
| **Mean** | **17/80 (21.25%)** |

These values are descriptive evidence, not checkpoint-selection signals.
