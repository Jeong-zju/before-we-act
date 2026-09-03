# CARE on MARS-Control — pinned reproduction (35.00%)

This document pins the exact configuration that produced the reported
MARS-Control Validation20 result and is the single supported way to
reproduce it.

## Pinned result

`validation20/{care,selector_off}/summary.json`
(`before-we-act.care-mars-validation20-summary/1`, policy
`official_care_mars_bench_port`, strict local observation):

| task | successes | episodes | success rate |
|---|---|---|---|
| place_cube_in_cup | 3 | 20 | 0.15 |
| strike_cube_hard | 19 | 20 | 0.95 |
| three_robots_place_shoes | 2 | 20 | 0.10 |
| four_robots_stack_cube | 4 | 20 | 0.20 |
| **total** | **28** | **80** | **0.3500** |

In this run the calibrated CARE selector issued zero substitute actions
(`overrides = 0` on every episode), so `care` and `selector_off` report
identical numbers. Seeing both modes agree exactly is expected, not a bug.

## Checkpoint identity

| artifact | identity |
|---|---|
| Reference policy (B0-H + team belief) | selection `before-we-act.mars-belief-selection/1`; seed **20260815**, update **105000**, metric `diagnostic macro.b_core`; deployment checkpoint sha256 `af4fb12a769e81ec9c3fb09ae0e7a48c341c1d3cfc80e382b5dde2d118c16a10` |
| CARE scorer | variant `care`, seed **20260820**, trained 4000 updates (batch 48) on the 120-family branch corpus; exported as `care_offline/care_deployment_checkpoint.pt` |

## Pipeline

Everything runs through one resumable script (stage receipts in
`pipeline_status.json`):

- supervisor entry: `scripts/before_we_act/mars_care_official_pipeline.supervisor.conf`
  → `run_mars_care_autonomous.sh` → `run_mars_care_official_pipeline.sh`
- settings: `configs/before_we_act/care_mars_bench_port.json`
- action contract: `configs/before_we_act/mars_action_contract_v1.json`

Stages, in order:

1. `PREFLIGHT` — requires the smoke receipts produced by
   `run_mars_care_smoke.sh` and `run_mars_care_branch_smoke.sh` once per host.
2. `B0H_FORMAL` — `before_we_act.train_mars_temporal_policy`, 120 000 updates,
   4-GPU DDP, frozen DINOv3 ViT-B/16, `absolute_pd_joint_pos`.
3. `ACTION_CONTEXT_CACHE` — `scripts.before_we_act.build_mars_action_context_cache`
   over all 600 demonstrations.
4. `BELIEF_FORMAL` — `before_we_act.train_mars_predictive_team_belief`,
   seeds 20260815/20260816/20260817, 120 000 updates each.
5. `BELIEF_SELECT` — `scripts.before_we_act.select_mars_care_belief`
   (metric `macro.b_core`) → `belief_selected/deployment_checkpoint.pt`.
6. `CARE_BRANCHES` — `before_we_act.mars_care_branch_collector`,
   120 families × 24 branches, serial Vulkan rendering.
7. `CARE_PREPARE` — `scripts.before_we_act.prepare_mars_care_training`
   (`quality` then `prepare`).
8. `CARE_SCORERS` — `before_we_act.train_mars_care_belief`, 4 variants
   (`care`, `reactive_only`, `replay_only`, `capacity`) × seeds
   20260818/20260819/20260820, 4000 updates, batch 48.
9. `CARE_CALIBRATE` — `scripts.before_we_act.select_calibrate_mars_care`
   → `care_offline/care_deployment_checkpoint.pt`.
10. `VALIDATION20` — per task and per mode (`selector_off`, `care`):

```
python -m before_we_act.evaluate_mars_care_closed_loop \
  --reference-checkpoint <run>/belief_selected/deployment_checkpoint.pt \
  --care-checkpoint <run>/care_offline/care_deployment_checkpoint.pt \
  --task <task> --robofactory-root <RoboFactory checkout> \
  --output <run>/validation20/<mode>/<task>.json \
  --episodes 20 --seed-start 20260827 --max-steps <steps> \
  --mode <mode> --device cuda:0 --render-device cuda:0
```

`max-steps`: place_cube_in_cup 500, strike_cube_hard 500,
three_robots_place_shoes 1200, four_robots_stack_cube 800.
Summaries via `scripts.before_we_act.summarize_mars_care_validation`.

## Dataset

150 recorded motion-planning successes per task (600 episodes) collected by
`deployment/mars_care/expert_shard.py`; download/verify with
`scripts/before_we_act/download_mars_control.py`.

## Known deviation from the archived run

The archived 35.00% run cached B0-H hidden activations in float16
(`mars-action-context-cache/1`). Late in formal training those activations
can exceed float16's finite range, so the cache contract is now
`mars-action-context-cache/2`: float32 hidden states with an all-finite
check (`build_mars_action_context_cache.py`, enforced at load time by
`before_we_act/mars_team_belief_data.py`). A fresh reproduction therefore
trains the belief stage on a numerically cleaner cache; small deviations
from 28/80 are possible and the corrected cache is authoritative.
