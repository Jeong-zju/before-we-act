#!/usr/bin/env bash
set -euo pipefail

# Reproduce the released six-task ACT baseline from the audited WAM HDF5 data.
# The adapter refuses to overwrite an existing Zarr store.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ACT_PYTHON:-python3}"
DATA_ROOT="${ACT_DATA_ROOT:-${ROOT}/data/robofactory_multitask}"
ROBOFACTORY_ROOT="${ACT_ROBOFACTORY_ROOT:-${ROOT}/RoboFactory}"
RUN_ROOT="${ACT_RUN_ROOT:-${ROOT}/runs/robofactory_act_formal_v1}"
ZARR_ROOT="${ACT_ZARR_ROOT:-${RUN_ROOT}/zarr}"
CHECKPOINT="${ACT_CHECKPOINT:-${RUN_ROOT}/train/last.pt}"
DEVICE="${ACT_DEVICE:-cuda:0}"
SIM_BACKEND="${ACT_SIM_BACKEND:-cpu}"
MODE="${1:-all}"

case "${MODE}" in
  prepare|train|eval|all) ;;
  *) printf 'usage: %s [prepare|train|eval|all]\n' "$0" >&2; exit 2 ;;
esac

[[ -d "${DATA_ROOT}" ]] || { printf 'missing ACT_DATA_ROOT: %s\n' "${DATA_ROOT}" >&2; exit 1; }
[[ -d "${ROBOFACTORY_ROOT}" ]] || { printf 'missing ACT_ROBOFACTORY_ROOT: %s\n' "${ROBOFACTORY_ROOT}" >&2; exit 1; }
mkdir -p "${ZARR_ROOT}" "${RUN_ROOT}/train" "${RUN_ROOT}/closed_loop"
export PYTHONPATH="${ROOT}:${ROBOFACTORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${MODE}" == prepare || "${MODE}" == all ]]; then
  for arm in 0 1 2 3; do
    output="${ZARR_ROOT}/agent${arm}.zarr"
    if [[ ! -e "${output}" ]]; then
      "${PYTHON_BIN}" "${ROOT}/scripts/adapt_wam_to_dp_zarr.py" \
        --data-root "${DATA_ROOT}" \
        --output "${output}" \
        --agent "${arm}" \
        --allow-missing \
        --width 320 \
        --height 240
    fi
  done
fi

if [[ "${MODE}" == train || "${MODE}" == all ]]; then
  "${PYTHON_BIN}" -u "${ROOT}/stereo_core/train_act.py" \
    --data unused \
    --shared --shared-arms 0,1,2,3 \
    --output "${RUN_ROOT}/train" \
    --zarr-agent 0="${ZARR_ROOT}/agent0.zarr" \
    --zarr-agent 1="${ZARR_ROOT}/agent1.zarr" \
    --zarr-agent 2="${ZARR_ROOT}/agent2.zarr" \
    --zarr-agent 3="${ZARR_ROOT}/agent3.zarr" \
    --horizon 100 \
    --enc-layers 4 \
    --dec-layers 7 \
    --d-model 384 \
    --batch-size 40 \
    --updates 120000 \
    --save-updates 20000,40000,60000,80000,100000,120000 \
    --workers 0 \
    --lazy-cache-episodes 4 \
    --episode-block-updates 64 \
    --task-balanced \
    --camera-width 320 \
    --camera-height 240 \
    --stats-root "${DATA_ROOT}" \
    --validation-updates 64 \
    --resume \
    --seed 20260819 \
    --formal-six-task
fi

if [[ "${MODE}" == eval || "${MODE}" == all ]]; then
  [[ -f "${CHECKPOINT}" ]] || { printf 'missing ACT_CHECKPOINT: %s\n' "${CHECKPOINT}" >&2; exit 1; }
  "${PYTHON_BIN}" -u "${ROOT}/scripts/evaluate_act_closed_loop.py" \
    --checkpoint "${CHECKPOINT}" \
    --stats-root "${DATA_ROOT}" \
    --config-root "${ROBOFACTORY_ROOT}/robofactory/configs/table" \
    --output-root "${RUN_ROOT}/closed_loop" \
    --episodes 20 \
    --max-steps-profile care \
    --seed 20260820 \
    --device "${DEVICE}" \
    --sim-backend "${SIM_BACKEND}" \
    --temporal-ensemble-decay 0.01 \
    --formal-six-task
fi
