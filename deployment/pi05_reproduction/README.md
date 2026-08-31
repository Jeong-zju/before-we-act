# π0.5 RoboFactory-MA and MARS-Control reproduction bundle

This bundle is the synchronized record of the π0.5 RoboFactory run on one
NVIDIA H200 and the four-task MARS-Control run on four RTX PRO 6000 Blackwell
GPUs. It keeps the upstream OpenPI checkout separate from the application code
and makes the previously implicit fixes reviewable.

## Contents

- `openpi_patch/`: the exact OpenPI integration patch and the five new
  RoboFactory/MARS-Control modules. Apply it to OpenPI commit
  `15a9616a00943ada6c20a0f158e3adb39df2ccac` with `openpi_patch/apply.sh`.
- `../../configs/pi05_robofactory_lora_formal_v1.json`: machine-readable
  frozen training contract.
- `../vla_baselines/pipeline.pi05.h200.json`: Supervisor DAG for preflight,
  six-dataset download, normalization, smoke tests, 120k-step LoRA training,
  Validation20, and final report.
- `../vla_baselines/run_pi05.sh`, `run_pi05_train_smoke.sh`,
  `run_pi05_closed_loop_smoke.sh`, `validation_launcher.py`, and
  `policy_rpc_server.py`: launch, smoke, inference, and validation code.
- `../pi05_mars/`: the MARS-Control download, corpus/normalization audit,
  smoke gates, four-GPU supervisor, arm-local inference server, and
  task-specific Validation20 evaluator.
- `../../configs/pi05_mars_control_lora_v1.json`: frozen, machine-readable
  MARS-Control training and validation contract.
- `openpi_patch/requirements-blackwell-cu129.txt`: exact runtime overlay used
  to make four-device JAX collectives work on the Blackwell server.

## Reproduction order

1. Checkout OpenPI at the pinned commit and run `openpi_patch/apply.sh`.
2. Checkout this repository at the pinned reproduction commit and install its
   declared RoboFactory environment.
3. Download the six `zeno-ai` datasets using `download_six_parallel.sh`.
4. Run the Supervisor pipeline in `pipeline.pi05.h200.json`; it fails closed
   if the six tasks or 900 episodes are incomplete.
5. Use the final checkpoint only after the smoke gates and the 120-rollout
   Validation20 stage complete.

The formal result is 71/120 (59.17%) over six tasks. Secrets are intentionally
not committed; provide the Hugging Face token through the runtime secret file
expected by `run_pi05.sh`.

## MARS-Control reproduction

The MARS-Control source is RoboFactory commit
`2d34fb38c80cb06550a5dbf99abac2c89f4336ed`. Follow
`../pi05_mars/README.md` after applying the same OpenPI patch. The run uses all
600 demonstrations, trains one shared decentralized policy for 30,000 updates
at global batch 128, and evaluates the fixed final checkpoint on 20 seeds per
task. The recorded result is 17/80 (21.25%).
