#!/usr/bin/env bash

FE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_ROOT="${S2_R4_SHARED_ROOT:-/workspace/fe-pc-wam}"
RUN_ID="${S2_R4_HYBRID_RUN_ID:-s2-r4-hybrid-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT=""
OWN_SOURCE="${S2_R4_HYBRID_OWN_SOURCE:-}"
TEAM_SOURCE="${S2_R4_HYBRID_TEAM_SOURCE:-}"
SESSION="${S2_R4_HYBRID_TMUX_SESSION:-}"
FOCUS_MONITOR=1
DRY_RUN=0

usage() {
  printf 'usage: %s [--run-id ID] [--shared-root PATH] [--own-source PT] [--team-source PT] [--session NAME] [--no-focus-monitor] [--dry-run]\n' "$0"
}

while (( $# )); do
  case "$1" in
    --run-id) RUN_ID="${2:?--run-id requires a value}"; shift 2 ;;
    --shared-root) SHARED_ROOT="${2:?--shared-root requires a value}"; shift 2 ;;
    --own-source) OWN_SOURCE="${2:?--own-source requires a value}"; shift 2 ;;
    --team-source) TEAM_SOURCE="${2:?--team-source requires a value}"; shift 2 ;;
    --session) SESSION="${2:?--session requires a value}"; shift 2 ;;
    --no-focus-monitor) FOCUS_MONITOR=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  printf >&2 'Invalid run id: %s\n' "${RUN_ID}"
  exit 2
fi
SHARED_ROOT="$(realpath -e "${SHARED_ROOT}")" || exit 3
RUN_ROOT="${SHARED_ROOT}/outputs/s2_r4_hybrid/${RUN_ID}"
CONFIG="${FE_ROOT}/configs/wam_flow/s2_r4_hybrid_diagnostic.yaml"
WINDOW_PREFIX="${S2_R4_HYBRID_WINDOW_PREFIX:-${RUN_ID}}"
PREPARE_WINDOW="${WINDOW_PREFIX}-prepare"
EVALUATE_WINDOW="${WINDOW_PREFIX}-evaluate"
MONITOR_WINDOW="${WINDOW_PREFIX}-monitor"

for command_name in git tmux jq nvidia-smi python3 realpath find sort tail uv; do
  if ! command -v "${command_name}" >/dev/null; then
    printf >&2 'Missing required command: %s\n' "${command_name}"
    exit 3
  fi
done

if [[ -z "${SESSION}" && -n "${TMUX:-}" ]]; then
  SESSION="$(tmux display-message -p '#S')"
fi
if [[ -z "${SESSION}" ]]; then
  mapfile -t SESSIONS < <(tmux list-sessions -F '#S' 2>/dev/null)
  if (( ${#SESSIONS[@]} != 1 )); then
    printf >&2 'Expected exactly one permanent tmux session; found %d.\n' \
      "${#SESSIONS[@]}"
    exit 3
  fi
  SESSION="${SESSIONS[0]}"
fi
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf >&2 'Permanent tmux session does not exist: %s\n' "${SESSION}"
  exit 3
fi

if [[ -z "${OWN_SOURCE}" ]]; then
  OWN_SOURCE="$(find "${SHARED_ROOT}/outputs/s2_r4_runs" -type f \
    -path '*/candidates/p0/checkpoints/predictor.pt' -printf '%T@ %p\n' \
    2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi
if [[ -z "${TEAM_SOURCE}" ]]; then
  TEAM_SOURCE="$(find "${SHARED_ROOT}/outputs/s2_r4_runs" -type f \
    -path '*/candidates/p1/checkpoints/predictor.pt' -printf '%T@ %p\n' \
    2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi
OWN_SOURCE="$(realpath -e "${OWN_SOURCE}")" || exit 3
TEAM_SOURCE="$(realpath -e "${TEAM_SOURCE}")" || exit 3
if [[ "${OWN_SOURCE}" == "${TEAM_SOURCE}" ]]; then
  printf >&2 'Own and team source paths must differ.\n'
  exit 3
fi

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( GPU_COUNT < 1 )); then
  printf >&2 'S2-R4 hybrid requires at least one GPU.\n'
  exit 3
fi
if [[ ! -f "${CONFIG}" ]]; then
  printf >&2 'Missing hybrid config: %s\n' "${CONFIG}"
  exit 3
fi
for window_name in "${PREPARE_WINDOW}" "${EVALUATE_WINDOW}" "${MONITOR_WINDOW}"; do
  if tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "${window_name}"; then
    printf >&2 'Tmux window already exists: %s\n' "${window_name}"
    exit 3
  fi
done
if [[ -e "${RUN_ROOT}" ]]; then
  printf >&2 'Refusing to overwrite existing run root: %s\n' "${RUN_ROOT}"
  exit 3
fi

printf 'S2-R4 protected hybrid plan\n'
printf '  repo: %s\n  shared: %s\n  run: %s\n' "${FE_ROOT}" "${SHARED_ROOT}" "${RUN_ROOT}"
printf '  own source: %s\n  team source: %s\n' "${OWN_SOURCE}" "${TEAM_SOURCE}"
printf '  tmux: %s (%s, %s, %s)\n' "${SESSION}" \
  "${PREPARE_WINDOW}" "${EVALUATE_WINDOW}" "${MONITOR_WINDOW}"
printf '  GPU: physical 0 only; no training, optimizer, or statistics fit\n'
if (( DRY_RUN )); then
  printf 'Dry run complete; no run directory/window/process created.\n'
  exit 0
fi

for shared_name in artifacts datasets; do
  if [[ ! -e "${FE_ROOT}/${shared_name}" ]]; then
    ln -s "${SHARED_ROOT}/${shared_name}" "${FE_ROOT}/${shared_name}" || exit 3
  fi
done
mkdir -p "${RUN_ROOT}" || exit 3
python3 scripts/s2_r4_hybrid_runtime.py init \
  --run-root "${RUN_ROOT}" --run-id "${RUN_ID}" --session "${SESSION}" \
  --window-prefix "${WINDOW_PREFIX}" --monitor-window "${MONITOR_WINDOW}" \
  --repo "${FE_ROOT}" --own-source "${OWN_SOURCE}" \
  --team-source "${TEAM_SOURCE}" || exit 3

COMMON_ENV="export S2_R4_HYBRID_RUN_ROOT=$(printf %q "${RUN_ROOT}"); export S2_R4_HYBRID_OWN_SOURCE=$(printf %q "${OWN_SOURCE}"); export S2_R4_HYBRID_TEAM_SOURCE=$(printf %q "${TEAM_SOURCE}"); export UV_PROJECT_ENVIRONMENT=$(printf %q "${SHARED_ROOT}/.venv"); export UV_CACHE_DIR=$(printf %q "${SHARED_ROOT}/.uv-cache");"
PREPARE_COMMAND="${COMMON_ENV} cd $(printf %q "${FE_ROOT}"); bash scripts/prepare_s2_r4_hybrid.sh"
EVALUATE_COMMAND="${COMMON_ENV} export CUDA_VISIBLE_DEVICES=0; cd $(printf %q "${FE_ROOT}"); bash scripts/run_s2_r4_hybrid_evaluation.sh"
MONITOR_COMMAND="cd $(printf %q "${FE_ROOT}"); python3 scripts/s2_r4_hybrid_runtime.py monitor --run-root $(printf %q "${RUN_ROOT}") --interval 5"

PREPARE_ID="$(tmux new-window -d -P -F '#{window_id}' -t "${SESSION}:" \
  -n "${PREPARE_WINDOW}" -c "${FE_ROOT}" "${PREPARE_COMMAND}")" || exit 3
EVALUATE_ID="$(tmux new-window -d -P -F '#{window_id}' -t "${SESSION}:" \
  -n "${EVALUATE_WINDOW}" -c "${FE_ROOT}" "${EVALUATE_COMMAND}")" || exit 3
MONITOR_ID="$(tmux new-window -d -P -F '#{window_id}' -t "${SESSION}:" \
  -n "${MONITOR_WINDOW}" -c "${FE_ROOT}" "${MONITOR_COMMAND}")" || exit 3
for window_id in "${PREPARE_ID}" "${EVALUATE_ID}" "${MONITOR_ID}"; do
  tmux set-option -w -t "${window_id}" remain-on-exit on
  tmux set-option -w -t "${window_id}" history-limit 200000
done
if (( FOCUS_MONITOR )) && [[ -n "${TMUX:-}" ]]; then
  tmux select-window -t "${SESSION}:${MONITOR_WINDOW}"
fi
printf 'S2-R4 hybrid started; permanent session remains alive.\n'
printf 'Monitor: python3 scripts/s2_r4_hybrid_runtime.py monitor --once --run-root %s\n' \
  "${RUN_ROOT}"
