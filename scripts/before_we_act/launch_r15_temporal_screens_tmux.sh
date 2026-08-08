#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="r15-temporal-$(date -u +%Y%m%dT%H%M%SZ)"; RUN_ROOT=""; SELECTION="p1,p2"; SPLIT=discovery20; REFERENCE_RUN=""; GPU_OVERRIDE=""; MODE_OVERRIDE=""; SESSION_OVERRIDE=""; DRY_RUN=0
PYTHON=/venv/robofactory-act/bin/python
PROTOCOL_ROOT=/workspace/bwa_runs/shared/r15_stack_protocol_v1
CHECKPOINT=/workspace/bwa_runs/r12e1-20260806-agent-slot-v4/candidates/p2/train/formal/checkpoints/checkpoint_130000.pt
while (($#)); do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --candidate|--candidates) SELECTION="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --reference-run-root) REFERENCE_RUN="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --gpu-index) GPU_OVERRIDE="$2"; shift 2 ;;
    --execution-mode) MODE_OVERRIDE="$2"; shift 2 ;;
    --session) SESSION_OVERRIDE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
RUN_ROOT="${RUN_ROOT:-/workspace/bwa_runs/$RUN_ID}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ && "$SPLIT" =~ ^(discovery20|validation20|reserve20|final20)$ && -n "$REFERENCE_RUN" ]] || { printf 'valid run/split and reference run root required\n' >&2; exit 2; }
IFS=',' read -r -a SELECTED <<<"$SELECTION"
for candidate in "${SELECTED[@]}"; do [[ "$candidate" =~ ^p[1-3]$ ]] || { printf 'temporal candidates are p1,p2,p3\n' >&2; exit 2; }; done
if [[ -n "$GPU_OVERRIDE" ]]; then
  [[ "${#SELECTED[@]}" -eq 1 && "$GPU_OVERRIDE" =~ ^[0-3]$ ]] || { printf -- '--gpu-index requires one candidate and GPU 0..3\n' >&2; exit 2; }
fi
if [[ -n "$SESSION_OVERRIDE" ]]; then
  [[ "${#SELECTED[@]}" -eq 1 && "$SESSION_OVERRIDE" =~ ^bwa-r15s-[A-Za-z0-9_.-]+$ ]] || { printf -- '--session requires one candidate and a validated bwa-r15s-* name\n' >&2; exit 2; }
fi
if [[ -n "$MODE_OVERRIDE" ]]; then
  [[ "${#SELECTED[@]}" -eq 1 && "$MODE_OVERRIDE" =~ ^(act_temporal_ensemble|mild_temporal_ensemble|balanced_temporal_ensemble|recent_temporal_ensemble|responsive_temporal_ensemble|cogact_adaptive_ensemble|aac_entropy_chunk|latest_chunk)$ ]] || { printf -- '--execution-mode requires one candidate and a registered mode\n' >&2; exit 2; }
fi
for command in git tmux nvidia-smi sha256sum jq; do command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 3; }; done
BRANCH="$(git -C "$ROOT" branch --show-current)"
[[ "$BRANCH" =~ ^bwa/r15-(closed-loop-evolution|aac-entropy-chunk|role-query-specialist|role-query-view-dedup)$ && -z "$(git -C "$ROOT" status --porcelain)" ]] || { printf 'launcher requires a clean R15 evolution branch\n' >&2; exit 3; }
git -C "$ROOT" fetch origin --prune
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"; [[ "$COMMIT" == "$(git -C "$ROOT" rev-parse "origin/$BRANCH")" ]] || { printf 'R15 branch differs from origin\n' >&2; exit 3; }
SEED_FILE="$PROTOCOL_ROOT/$SPLIT.json"; SEED_SHA="$(sha256sum "$SEED_FILE" | awk '{print $1}')"
REFERENCE_MANIFEST="$REFERENCE_RUN/run_manifest.json"
[[ -f "$REFERENCE_MANIFEST" && -f "$CHECKPOINT" ]] || { printf 'reference manifest or W12 checkpoint missing\n' >&2; exit 3; }
[[ "$(jq -r .split "$REFERENCE_MANIFEST")" == "$SPLIT" && "$(jq -r .seed_file_sha256 "$REFERENCE_MANIFEST")" == "$SEED_SHA" && "$(jq -r .candidates.p0.reference "$REFERENCE_MANIFEST")" == true ]] || { printf 'reference run is not the identical paired W12 control\n' >&2; exit 3; }
[[ ! -e "$RUN_ROOT" ]] || { printf 'temporal run root already exists: %s\n' "$RUN_ROOT" >&2; exit 3; }
declare -A LABEL GPU MODE
LABEL[p1]=w12_recent_decay_0p10; GPU[p1]=1; MODE[p1]=recent_temporal_ensemble
LABEL[p2]=w12_latest_chunk; GPU[p2]=3; MODE[p2]=latest_chunk
LABEL[p3]=w12_balanced_decay_0p05; GPU[p3]=2; MODE[p3]=balanced_temporal_ensemble
[[ -n "$GPU_OVERRIDE" ]] && GPU["${SELECTED[0]}"]="$GPU_OVERRIDE"
if [[ -n "$MODE_OVERRIDE" ]]; then
  selected="${SELECTED[0]}"; MODE["$selected"]="$MODE_OVERRIDE"
  case "$MODE_OVERRIDE" in
    mild_temporal_ensemble) LABEL["$selected"]=w12_mild_decay_0p02 ;;
    balanced_temporal_ensemble) LABEL["$selected"]=w12_balanced_decay_0p05 ;;
    recent_temporal_ensemble) LABEL["$selected"]=w12_recent_decay_0p10 ;;
    responsive_temporal_ensemble) LABEL["$selected"]=w12_responsive_decay_0p20 ;;
    cogact_adaptive_ensemble) LABEL["$selected"]=cogact_adaptive_alpha0p1_h2 ;;
    aac_entropy_chunk) LABEL["$selected"]=aac_entropy20_h16 ;;
    latest_chunk) LABEL["$selected"]=w12_latest_chunk ;;
    *) LABEL["$selected"]=checkpoint_act_temporal_ensemble ;;
  esac
fi
if [[ "$BRANCH" == bwa/r15-role-query-specialist ]]; then
  for candidate in "${SELECTED[@]}"; do
    [[ "${MODE[$candidate]}" == act_temporal_ensemble ]] && LABEL["$candidate"]=role_query_act_temporal_ensemble
  done
elif [[ "$BRANCH" == bwa/r15-role-query-view-dedup ]]; then
  for candidate in "${SELECTED[@]}"; do
    [[ "${MODE[$candidate]}" == act_temporal_ensemble ]] && LABEL["$candidate"]=role_query_view_dedup_act_temporal_ensemble
  done
fi
for candidate in "${SELECTED[@]}"; do
  session="bwa-r15s-$candidate"; gpu="${GPU[$candidate]}"
  [[ -n "$SESSION_OVERRIDE" ]] && session="$SESSION_OVERRIDE"
  tmux has-session -t "$session" 2>/dev/null && { printf 'session already exists: %s\n' "$session" >&2; exit 3; }
  nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]' && { printf 'GPU %s is in use\n' "$gpu" >&2; exit 3; } || true
  printf '%s label=%s mode=%s branch=%s commit=%s GPU=%s\n' "$candidate" "${LABEL[$candidate]}" "${MODE[$candidate]}" "$BRANCH" "$COMMIT" "$gpu"
done
printf 'R15 temporal preflight run=%s split=%s reference=%s root=%s\n' "$RUN_ID" "$SPLIT" "$REFERENCE_RUN" "$RUN_ROOT"
if ((DRY_RUN)); then printf 'dry-run passed; no output/tmux created\n'; exit 0; fi

P0_WORKTREE="$(jq -r .candidates.p0.worktree "$REFERENCE_MANIFEST")"
P0_BRANCH="$(jq -r .candidates.p0.branch "$REFERENCE_MANIFEST")"
P0_COMMIT="$(jq -r .candidates.p0.commit "$REFERENCE_MANIFEST")"
P0_CONFIG="$(jq -r .candidates.p0.config "$REFERENCE_MANIFEST")"
P0_CHECKPOINT="$(jq -r .candidates.p0.checkpoint "$REFERENCE_MANIFEST")"
"$PYTHON" "$ROOT/scripts/before_we_act/r15_runtime.py" register --run-root "$RUN_ROOT" --run-id "$RUN_ID" --split "$SPLIT" --seed-file "$SEED_FILE" --seed-file-sha256 "$SEED_SHA" --candidate p0 --label w12_control --gpu 0 --worktree "$P0_WORKTREE" --branch "$P0_BRANCH" --commit "$P0_COMMIT" --config "$P0_CONFIG" --checkpoint "$P0_CHECKPOINT" --reference
ln -s "$REFERENCE_RUN/candidates/p0/validation" "$RUN_ROOT/candidates/p0/validation"
"$PYTHON" - "$RUN_ROOT/candidates/p0/reference_source.json" "$REFERENCE_RUN" <<'PY'
import datetime,json,os,sys
path,source=sys.argv[1:]
p={"schema_version":1,"kind":"exact_paired_reference_symlink","source_run_root":source,"source_candidate":"p0","created_at":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z")}
t=path+".tmp"; open(t,"w").write(json.dumps(p,indent=2,sort_keys=True)+"\n"); os.replace(t,path)
PY
CONFIG="$ROOT/configs/before_we_act/r12_action/e1_p2.yaml"
for candidate in "${SELECTED[@]}"; do
  session="bwa-r15s-$candidate"; [[ -n "$SESSION_OVERRIDE" ]] && session="$SESSION_OVERRIDE"
  "$PYTHON" "$ROOT/scripts/before_we_act/r15_runtime.py" register --run-root "$RUN_ROOT" --run-id "$RUN_ID" --split "$SPLIT" --seed-file "$SEED_FILE" --seed-file-sha256 "$SEED_SHA" --candidate "$candidate" --label "${LABEL[$candidate]}" --gpu "${GPU[$candidate]}" --worktree "$ROOT" --branch "$BRANCH" --commit "$COMMIT" --config "$CONFIG" --checkpoint "$CHECKPOINT" --session "$session"
done
for candidate in "${SELECTED[@]}"; do
  session="bwa-r15s-$candidate"; [[ -n "$SESSION_OVERRIDE" ]] && session="$SESSION_OVERRIDE"
  tmux new-session -d -s "$session" -n screen \
    "cd '$ROOT' && exec env BWA_R15_RUN_ROOT='$RUN_ROOT' BWA_R15_CANDIDATE='$candidate' '$ROOT/scripts/before_we_act/run_r15_stack_screen.sh' --run-root '$RUN_ROOT' --candidate '$candidate' --gpu-index '${GPU[$candidate]}' --python '$PYTHON'"
done
printf 'started temporal candidates: %s\n' "${SELECTED[*]}"
printf 'monitor: %s/scripts/before_we_act/monitor_r15_stack_screens.sh --run-root %s --candidate all --interval 30\n' "$ROOT" "$RUN_ROOT"
printf 'safe stop: %s/scripts/before_we_act/stop_r15_stack_screens.sh --run-root %s --candidate all\n' "$ROOT" "$RUN_ROOT"
