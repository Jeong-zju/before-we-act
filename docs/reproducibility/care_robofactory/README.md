# CARE on RoboFactory — pinned reproduction (85.83%)

This document pins the exact configuration behind the reported RoboFactory
six-task Validation20 result and is the single supported way to reproduce it.

## Pinned result

Paired Validation20, 6 tasks × 20 episodes per mode
(`before-we-act.a7r1-care-validation20-summary/1`):

| task | successes / 20 |
|---|---|
| lift_barrier | 20 |
| camera_alignment | 12 |
| long_pipeline_delivery | 18 |
| take_photo | 20 |
| pass_shoe | 20 |
| place_food | 13 |
| **total** | **103 / 120 = 0.8583** |

Both modes (`care`, `selector_off`) scored exactly 103/120 with zero
override steps: the calibrated selector's 90% lower confidence bound never
released a substitute action, so the number equals the frozen reference
policy's performance inside the CARE harness. Identical numbers across the
two modes are expected, not a bug.

## Checkpoint identity

| artifact | identity |
|---|---|
| Reference policy (frozen B-core) | seed **20260817**, 120 000 updates; deployment checkpoint sha256 `deb628b9cdee68a3243f91158e2b0165dd2e2fe03196c138c3a669904bc6792e`; verified byte-identical before and after the run |
| CARE scorer | variant `care`, seed **20260820**, selected at update **2000**; deployment checkpoint sha256 `77d37e84ba30651c0f47a659f66688ddf4814f3a0354ea1d7384dcc3beb91b93`; 1 643 408 parameters; calibration: nominal coverage 0.9, lower correction 0.02387, hard-safety max 0.25 |

## Pipeline

- supervisor entry: `scripts/before_we_act/bwa_care_robofactory.supervisor.conf`
  → `scripts/before_we_act/run_care_robofactory.sh`
- frozen settings: `configs/before_we_act/care_robofactory_reproduction.json`
  (verified up front by `scripts/before_we_act/verify_frozen_settings.py`)
- RoboFactory benchmark commit `5868242322414a91454e22f1dd9641f613ba1bcf`,
  640×480 RGB observations, `pd_joint_pos` control.

Required inputs (published alongside the code):

- `BWA_CARE_REFERENCE_CHECKPOINT` — the frozen B-core reference checkpoint.
- `BWA_CARE_FAMILY_ROOT` / `BWA_CARE_QUALITY_ROOT` — the branch-family
  corpus (30 families per task). To re-collect from scratch use
  `before_we_act.care_branch_collector` against the same reference
  checkpoint.

Stages, in order:

1. `prepare_care_training.py` — assemble the prepared scorer dataset from
   the family corpus (provenance-hashed).
2. `before_we_act.train_care_belief` — 4 variants × seeds
   20260818/20260819/20260820, 4000 updates, batch 48, lr 3e-4, wd 1e-4,
   eval every 200.
3. `select_calibrate_care.py` — select and calibrate the deployed scorer →
   `offline/care_deployment_checkpoint.pt` + `offline_report.json`.
4. `prepare_care_test_seeds.py` — freeze per-task closed-loop seed files
   before any rollout.
5. `before_we_act.evaluate_care_closed_loop` — per task and per mode:

```
python -m before_we_act.evaluate_care_closed_loop \
  --reference-checkpoint $BWA_CARE_REFERENCE_CHECKPOINT \
  --care-checkpoint <run>/offline/care_deployment_checkpoint.pt \
  --mode <care|selector_off> --task <task> \
  --seed-file <run>/tests/seeds/<task>.json \
  --settings configs/before_we_act/care_robofactory_reproduction.json \
  --episodes 20 --max-steps <steps> --device cuda:0 \
  --robofactory-root <RoboFactory checkout> --output <run>/tests/<mode>/<task>.json
```

`max-steps`: 500 for lift_barrier / pass_shoe / place_food, 1500 for
camera_alignment / long_pipeline_delivery / take_photo.

6. `summarize_care_tests.py` — the sha256-stamped summary.

## Difference from the MARS pipeline

RoboFactory consumes a frozen, published reference checkpoint and a
pre-collected family corpus; MARS trains its reference (B0-H + team belief)
from scratch inside its own pipeline. CARE-stage hyperparameters are
identical across the two benchmarks (candidate arity 6, horizons
8/16/32/64, scorer seeds 20260818-20, 4000 updates).
