#!/usr/bin/env bash
set -Eeuo pipefail

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_SCHEMA="${FE_ROOT}/checkpoints/phase_m2_liftbarrier_longpipeline_joint_seed101/schema.json"
TRAINING_REPORT="${FE_ROOT}/outputs/phase_m2_liftbarrier_longpipeline_joint/seed101_training.json"
CURRENT_TRAINING_LOG="${FE_ROOT}/outputs/phase_m2_liftbarrier_longpipeline_joint/seed101_training.log"
CURRENT_GATE="${FE_ROOT}/outputs/phase_m2_liftbarrier_longpipeline_joint_gate_20260724_093235"

jq -e '
  .format_version == "wam.robofactory.m2.checkpoint/4" and
  .action_space == "per_task_zscore_canonical_unit_action" and
  .task_vocabulary == ["lift_barrier", "long_pipeline_delivery"] and
  .model_config.action_horizon == 32
' "${CHECKPOINT_SCHEMA}" >/dev/null
jq -e '
  .format_version == "wam.robofactory.m2.training_report/4" and
  .passed == true and
  .dataset.task_action_horizons == {
    "lift_barrier": 16,
    "long_pipeline_delivery": 32
  }
' "${TRAINING_REPORT}" >/dev/null

for summary in \
  lift_train_seed3000 \
  lift_validation_seed3003 \
  lift_unseen_seed900_902 \
  long_train_seed3000 \
  long_validation_seed3003 \
  long_unseen_seed900_902
do
  jq -e '.format_version == "wam.robofactory.m2.rollout_summary/1"' \
    "${CURRENT_GATE}/${summary}/rollout_summary.json" >/dev/null
done

obsolete=(
  "${FE_ROOT}/checkpoints/phase_m2_lift_barrier_seed101"
  "${FE_ROOT}/checkpoints/phase_m2_liftbarrier_fixed_seed101"
  "${FE_ROOT}/checkpoints/phase_m2_liftbarrier_tailfixed_seed101"
  "${FE_ROOT}/checkpoints/phase_m2_robofactory_multitask_seed101"
  "${FE_ROOT}/checkpoints/smoke"
  "${FE_ROOT}/checkpoints/archive/before_lift_rebuild_20260723_140416"
  "${FE_ROOT}/outputs/smoke"
  "${FE_ROOT}/outputs/phase_m2_robofactory_multitask"
  "${FE_ROOT}/outputs/phase_m2_rollout_smoke_20260723_112633"
  "${FE_ROOT}/outputs/phase_m2_lift_seed_diagnostic_20260723_113809"
  "${FE_ROOT}/outputs/phase_m2_lift_no_warm_seed3000_20260723_115641"
  "${FE_ROOT}/outputs/phase_m2_lift_no_warm_exec1_seed3000_20260723_115932"
  "${FE_ROOT}/outputs/phase_m2_lift_fixed_smoke_20260723_134205"
  "${FE_ROOT}/outputs/phase_m2_lift_fixed_smoke_20260723_134409"
  "${FE_ROOT}/outputs/phase_m2_lift_barrier_single_20260723_140416"
  "${FE_ROOT}/outputs/phase_m2_liftbarrier_fixed"
  "${FE_ROOT}/outputs/phase_m2_liftbarrier_fixed_gate_20260723_181701"
  "${FE_ROOT}/outputs/phase_m2_liftbarrier_tailfixed"
  "${FE_ROOT}/outputs/phase_m2_liftbarrier_tailfixed_gate_20260723_203503"
  "${FE_ROOT}/outputs/phase_m2_lift_expert_replay_seed3000_20260723_120507"
  "${FE_ROOT}/outputs/phase_m2_seed_contract_20260723_133807"
)

printf 'Removing obsolete M2 artifacts:\n'
printf '  %s\n' "${obsolete[@]}"
rm -rf -- "${obsolete[@]}"
rm -f -- "${CURRENT_TRAINING_LOG}"
printf 'Cleanup complete. Preserved M1 artifacts, task datasets, the formal M2 v4 joint checkpoint/report, and the latest joint rollout evidence.\n'
