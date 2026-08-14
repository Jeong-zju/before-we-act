#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${STEP2_REPO_ROOT:-/workspace/fe-pc-wam-step2}"
RUN_ROOT="${STEP2_RUN_ROOT:-/workspace/bwa_runs/p1-step2-b0h-v2}"
DATA_ROOT="${STEP2_DATA_ROOT:-/workspace/datasets/robofactory_multitask}"
CACHE_ROOT="${STEP2_CACHE_ROOT:-/workspace/bwa_runs/shared/p1-step2-dino-history-cache-v2}"
PYTHON_BIN="${STEP2_PYTHON:-/venv/robofactory-act/bin/python}"
TORCHRUN_BIN="${STEP2_TORCHRUN:-/venv/robofactory-act/bin/torchrun}"
DINO_MODEL="${STEP2_DINO_MODEL:-/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m}"
DINO_CARRIER="${STEP2_DINO_CARRIER:-/workspace/bwa_runs/w10-six-task-v1/train/formal/checkpoint_120000.pt}"
NORMALIZATION_SOURCE="${STEP2_NORMALIZATION_SOURCE:-/workspace/bwa_runs/w10-six-task-v1/train/formal/normalization.pt}"
LABEL_RECEIPT="${STEP2_LABEL_RECEIPT:-/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/m2_r3_formal/m2_r3_conclusion.json}"
PLACE_FOOD_RECEIPT="${STEP2_PLACE_FOOD_RECEIPT:-${DATA_ROOT}/place_food/place_food_hf_activation_receipt.json}"
STATUS="${RUN_ROOT}/pipeline_status.json"
CONTRACT_ROOT="${RUN_ROOT}/contract"
CONTRACT="${CONTRACT_ROOT}/step2_contract.json"
NORMALIZATION="${CONTRACT_ROOT}/normalization.pt"

fail() {
  printf >&2 'Step-2 B0-H pipeline: %s\n' "$*"
  exit 1
}

write_status() {
  local status="$1" stage="$2" detail="$3"
  "${PYTHON_BIN}" - "${STATUS}" "${status}" "${stage}" "${detail}" <<'PY'
import json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
v={"status":sys.argv[2],"stage":sys.argv[3],"detail":sys.argv[4],"updated_at_epoch":time.time()}
t=p.with_name(f".{p.name}.{os.getpid()}.tmp"); t.write_text(json.dumps(v,sort_keys=True)+"\n"); os.replace(t,p)
PY
}

on_error() {
  local code=$?
  write_status FAILED error "pipeline exited with code ${code}" || true
  exit "${code}"
}
trap on_error ERR
trap 'write_status STOPPED interrupted "signal"; exit 130' INT TERM

[[ -e "${ROOT}/.git" ]] || fail "repository is missing: ${ROOT}"
[[ -x "${PYTHON_BIN}" ]] || fail "Python is missing: ${PYTHON_BIN}"
[[ -x "${TORCHRUN_BIN}" ]] || fail "torchrun is missing: ${TORCHRUN_BIN}"
if [[ ! -f "${DINO_MODEL}/foundation_receipt.json" ]]; then
  write_status RUNNING foundation "recovering frozen DINO-only asset from lossless carrier"
  "${PYTHON_BIN}" -u "${ROOT}/scripts/before_we_act/recover_step2_dino_foundation.py" \
    --carrier "${DINO_CARRIER}" --output "${DINO_MODEL}"
fi
[[ -d "${DINO_MODEL}" ]] || fail "DINO model is missing"
[[ -f "${NORMALIZATION_SOURCE}" ]] || fail "normalization source is missing"
[[ -f "${LABEL_RECEIPT}" ]] || fail "Measurement label receipt is missing"
[[ -f "${PLACE_FOOD_RECEIPT}" ]] || fail "Place Food activation receipt is missing"
[[ -z "$(git -C "${ROOT}" status --short)" ]] || fail "Step-2 worktree must be clean"
[[ "$(git -C /workspace/RoboFactory rev-parse HEAD)" == 5868242322414a91454e22f1dd9641f613ba1bcf ]] || fail "RoboFactory commit drift"
[[ -z "$(git -C /workspace/RoboFactory status --short)" ]] || fail "RoboFactory worktree is dirty"

TASKS=(lift_barrier camera_alignment long_pipeline_delivery take_photo pass_shoe place_food)
MANIFESTS=()
for task in "${TASKS[@]}"; do
  manifest="${DATA_ROOT}/${task}/training_manifest.json"
  [[ -f "${manifest}" ]] || fail "training manifest is missing: ${task}"
  MANIFESTS+=("${manifest}")
done
BASE_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
export PYTHONPATH="${ROOT}/vendor/stereo-core/stereo_core:${ROOT}:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}/logs" "${CONTRACT_ROOT}"

write_status RUNNING contract "freezing original 720-episode contract"
if [[ ! -f "${CONTRACT}" ]]; then
  "${PYTHON_BIN}" -u "${ROOT}/scripts/before_we_act/prepare_step2_b0h.py" contract \
    --manifests "${MANIFESTS[@]}" \
    --output "${CONTRACT_ROOT}" \
    --normalization-source "${NORMALIZATION_SOURCE}" \
    --visual-cache "${CACHE_ROOT}" \
    --measurement-label-receipt "${LABEL_RECEIPT}" \
    --place-food-activation-receipt "${PLACE_FOOD_RECEIPT}" \
    --dino-model "${DINO_MODEL}" \
    --base-commit "${BASE_COMMIT}"
fi

write_status RUNNING visual_cache "encoding original 640x480 history on four GPUs"
if [[ "$(jq -r '.status // empty' "${CACHE_ROOT}/cache_receipt.json" 2>/dev/null || true)" != PASSED ]]; then
  "${TORCHRUN_BIN}" --standalone --nproc_per_node=4 \
    "${ROOT}/scripts/before_we_act/build_step2_visual_cache.py" \
    --manifests "${MANIFESTS[@]}" \
    --dino-model "${DINO_MODEL}" \
    --output "${CACHE_ROOT}" \
    --batch-size 16
fi

write_status RUNNING f0 "auditing begin/middle/end, masks, reset and agent slots"
"${PYTHON_BIN}" -u "${ROOT}/scripts/before_we_act/prepare_step2_b0h.py" f0 \
  --manifests "${MANIFESTS[@]}" --output "${CONTRACT_ROOT}" \
  --normalization-source "${NORMALIZATION_SOURCE}" --visual-cache "${CACHE_ROOT}" \
  --measurement-label-receipt "${LABEL_RECEIPT}" --dino-model "${DINO_MODEL}" \
  --place-food-activation-receipt "${PLACE_FOOD_RECEIPT}" \
  --base-commit "${BASE_COMMIT}"
"${PYTHON_BIN}" -u "${ROOT}/scripts/before_we_act/prepare_step2_b0h.py" cursor \
  --manifests "${MANIFESTS[@]}" --output "${CONTRACT_ROOT}" \
  --normalization-source "${NORMALIZATION_SOURCE}" --visual-cache "${CACHE_ROOT}" \
  --measurement-label-receipt "${LABEL_RECEIPT}" --dino-model "${DINO_MODEL}" \
  --place-food-activation-receipt "${PLACE_FOOD_RECEIPT}" \
  --base-commit "${BASE_COMMIT}"

COMMON=(
  --manifests "${MANIFESTS[@]}"
  --contract "${CONTRACT}"
  --normalization "${NORMALIZATION}"
  --visual-cache "${CACHE_ROOT}"
  --dino-model "${DINO_MODEL}"
  --workers 4
  --seed 20260814
)

write_status RUNNING f1 "four-update fresh/resume equivalence on four GPUs"
F1_REFERENCE="${RUN_ROOT}/f1/reference"
F1_RESUMED="${RUN_ROOT}/f1/resumed"
if [[ ! -f "${F1_REFERENCE}/checkpoint_000004.pt" ]]; then
  "${TORCHRUN_BIN}" --standalone --nproc_per_node=4 \
    "${ROOT}/before_we_act/train_step2_b0h.py" \
    --variant hidden_residual --stage f1 --updates 4 \
    --output "${F1_REFERENCE}" --save-every 2 "${COMMON[@]}"
fi
if [[ ! -f "${F1_RESUMED}/checkpoint_000002.pt" ]]; then
  "${TORCHRUN_BIN}" --standalone --nproc_per_node=4 \
    "${ROOT}/before_we_act/train_step2_b0h.py" \
    --variant hidden_residual --stage f1 --updates 2 \
    --output "${F1_RESUMED}" --save-every 2 "${COMMON[@]}"
fi
if [[ ! -f "${F1_RESUMED}/checkpoint_000004.pt" ]]; then
  "${TORCHRUN_BIN}" --standalone --nproc_per_node=4 \
    "${ROOT}/before_we_act/train_step2_b0h.py" \
    --variant hidden_residual --stage f1 --updates 4 \
    --resume "${F1_RESUMED}/checkpoint_000002.pt" \
    --output "${F1_RESUMED}" --save-every 2 "${COMMON[@]}"
fi
"${PYTHON_BIN}" -u "${ROOT}/scripts/before_we_act/verify_step2_f1.py" \
  --reference "${F1_REFERENCE}/checkpoint_000004.pt" \
  --resumed "${F1_RESUMED}/checkpoint_000004.pt" \
  --output "${CONTRACT_ROOT}/f1_receipt.json"

write_status RUNNING history_discovery "training history-only for 5000 updates"
HISTORY_ROOT="${RUN_ROOT}/history_only/discovery"
HISTORY_RESUME=()
if [[ -f "${HISTORY_ROOT}/checkpoint_latest.pt" ]]; then
  HISTORY_RESUME=(--resume "${HISTORY_ROOT}/checkpoint_latest.pt")
fi
if [[ "$(jq -r '.status // empty' "${HISTORY_ROOT}/status.json" 2>/dev/null || true)" != PASSED ]]; then
  "${TORCHRUN_BIN}" --standalone --nproc_per_node=4 \
    "${ROOT}/before_we_act/train_step2_b0h.py" \
    --variant history_only --stage discovery --updates 5000 \
    --output "${HISTORY_ROOT}" --save-every 1000 \
    "${HISTORY_RESUME[@]}" "${COMMON[@]}"
fi

write_status RUNNING history_validation5 "closed-loop diagnostic on 30 episodes"
env STEP2_RUN_ROOT="${RUN_ROOT}/history_only" \
  STEP2_CHECKPOINT="${HISTORY_ROOT}/checkpoint_005000.pt" \
  STEP2_VALIDATION_EPISODES=5 \
  STEP2_PYTHON="${PYTHON_BIN}" \
  bash "${ROOT}/scripts/before_we_act/validate_step2_b0h.sh"

write_status RUNNING hidden_formal "training hidden-residual for 120000 updates on four GPUs"
FORMAL_ROOT="${RUN_ROOT}/hidden_residual/formal"
FORMAL_RESUME=()
if [[ -f "${FORMAL_ROOT}/checkpoint_latest.pt" ]]; then
  FORMAL_RESUME=(--resume "${FORMAL_ROOT}/checkpoint_latest.pt")
fi
if [[ "$(jq -r '.status // empty' "${FORMAL_ROOT}/status.json" 2>/dev/null || true)" != PASSED ]]; then
  "${TORCHRUN_BIN}" --standalone --nproc_per_node=4 \
    "${ROOT}/before_we_act/train_step2_b0h.py" \
    --variant hidden_residual --stage formal --updates 120000 \
    --output "${FORMAL_ROOT}" --save-every 5000 \
    "${FORMAL_RESUME[@]}" "${COMMON[@]}"
fi

write_status RUNNING hidden_validation20 "closed-loop acceptance on 120 episodes"
env STEP2_RUN_ROOT="${RUN_ROOT}/hidden_residual" \
  STEP2_CHECKPOINT="${FORMAL_ROOT}/checkpoint_120000.pt" \
  STEP2_VALIDATION_EPISODES=20 \
  STEP2_PYTHON="${PYTHON_BIN}" \
  bash "${ROOT}/scripts/before_we_act/validate_step2_b0h.sh"

SUMMARY="${RUN_ROOT}/hidden_residual/evaluation/validation20/summary.json"
ACCEPTANCE="$(jq -r '.acceptance.status' "${SUMMARY}")"
write_status PASSED complete "hidden-residual Validation20 acceptance=${ACCEPTANCE}"
trap - ERR
printf 'STEP2_PIPELINE_COMPLETED acceptance=%s\n' "${ACCEPTANCE}"
