#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="r15-expert-ft-$(date -u +%Y%m%dT%H%M%SZ)"; RUN_ROOT=""; EXPERT_INDEX=""; PHASE_MANIFEST=""; REFERENCE_RUN=""
CANDIDATE=p1; GPU_INDEX=""; UPDATES=10000; SPLIT=discovery20; SESSION=""; DRY_RUN=0
BATCH_SIZE=12; EXPERT_ROWS=6; LEARNING_RATE=2e-5; WARMUP=500
PYTHON=/venv/robofactory-act/bin/python; PROTOCOL_ROOT=/workspace/bwa_runs/shared/r15_stack_protocol_v1
while (($#)); do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --expert-index) EXPERT_INDEX="$2"; shift 2 ;;
    --phase-manifest) PHASE_MANIFEST="$2"; shift 2 ;;
    --reference-run-root) REFERENCE_RUN="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --updates) UPDATES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --expert-rows) EXPERT_ROWS="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
RUN_ROOT="${RUN_ROOT:-/workspace/bwa_runs/$RUN_ID}"
SESSION="${SESSION:-bwa-r15s-$CANDIDATE}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ && "$CANDIDATE" =~ ^p[1-3]$ && "$GPU_INDEX" =~ ^[0-3]$ && "$UPDATES" =~ ^[1-9][0-9]*$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$EXPERT_ROWS" =~ ^[1-9][0-9]*$ && "$WARMUP" =~ ^[1-9][0-9]*$ && "$LEARNING_RATE" =~ ^[0-9]+([.][0-9]+)?([eE]-?[0-9]+)?$ && "$SPLIT" =~ ^(discovery20|validation20|reserve20|final20)$ && "$SESSION" =~ ^bwa-r15s-[A-Za-z0-9_.-]+$ && -n "$EXPERT_INDEX" && -n "$REFERENCE_RUN" ]] || { printf 'valid run/candidate/GPU/training/expert index/reference/session required\n' >&2; exit 2; }
((BATCH_SIZE >= 2 && EXPERT_ROWS < BATCH_SIZE && WARMUP <= UPDATES)) || { printf 'expert rows/warmup exceed training budget\n' >&2; exit 2; }
for command in git tmux nvidia-smi sha256sum jq; do command -v "$command" >/dev/null || { printf 'missing command: %s\n' "$command" >&2; exit 3; }; done
BRANCH="$(git -C "$ROOT" branch --show-current)"
[[ "$BRANCH" =~ ^bwa/r15-(closed-loop-evolution|expert-evolution|phase-balanced-expert)$ && -z "$(git -C "$ROOT" status --porcelain)" ]] || { printf 'launcher requires a clean R15 expert branch\n' >&2; exit 3; }
git -C "$ROOT" fetch origin --prune
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"; [[ "$COMMIT" == "$(git -C "$ROOT" rev-parse "origin/$BRANCH")" ]] || { printf 'R15 branch differs from origin\n' >&2; exit 3; }
SEED_FILE="$PROTOCOL_ROOT/$SPLIT.json"; SEED_SHA="$(sha256sum "$SEED_FILE" | awk '{print $1}')"
REFERENCE_MANIFEST="$REFERENCE_RUN/run_manifest.json"; [[ -f "$REFERENCE_MANIFEST" && -f "$EXPERT_INDEX" ]] || { printf 'reference manifest or expert index missing\n' >&2; exit 3; }
[[ "$(jq -r .split "$REFERENCE_MANIFEST")" == "$SPLIT" && "$(jq -r .seed_file_sha256 "$REFERENCE_MANIFEST")" == "$SEED_SHA" && "$(jq -r .candidates.p0.reference "$REFERENCE_MANIFEST")" == true ]] || { printf 'reference is not identical paired W12 control\n' >&2; exit 3; }
[[ "$(jq -r .extension.protocol "$EXPERT_INDEX")" == r15_raw_success_expert_physical_pd_joint_pos_direct_dinov3_v2 && "$(jq -r .extension.expert_episodes "$EXPERT_INDEX")" -ge 1 ]] || { printf 'expert cache identity differs\n' >&2; exit 3; }
PHASE_ARGS=()
LABEL="w12_expert_ft_b${BATCH_SIZE}_e${EXPERT_ROWS}"
if [[ "$BRANCH" == bwa/r15-phase-balanced-expert || -n "$PHASE_MANIFEST" ]]; then
  [[ -f "$PHASE_MANIFEST" && $((EXPERT_ROWS % 3)) -eq 0 ]] || { printf 'phase-balanced branch requires manifest and expert rows divisible by three\n' >&2; exit 3; }
  [[ "$(jq -r .protocol "$PHASE_MANIFEST")" == r15_stack_expert_monotone_three_phase_v1 && "$(jq -r .expert_index_sha256 "$PHASE_MANIFEST")" == "$(sha256sum "$EXPERT_INDEX" | awk '{print $1}')" ]] || { printf 'phase manifest identity differs\n' >&2; exit 3; }
  PHASE_ARGS=(--phase-manifest "$PHASE_MANIFEST")
  LABEL="w12_expert_ft_phase3_b${BATCH_SIZE}_e${EXPERT_ROWS}"
fi
[[ ! -e "$RUN_ROOT" ]] || { printf 'run root already exists: %s\n' "$RUN_ROOT" >&2; exit 3; }
tmux has-session -t "$SESSION" 2>/dev/null && { printf 'session already exists: %s\n' "$SESSION" >&2; exit 3; }
nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]' && { printf 'GPU %s is in use\n' "$GPU_INDEX" >&2; exit 3; } || true
CHECKPOINT="$RUN_ROOT/candidates/$CANDIDATE/train/stack_expert/checkpoints/checkpoint_$(printf '%06d' "$UPDATES").pt"
printf 'R15 expert fine-tune preflight run=%s candidate=%s GPU=%s updates=%s batch=%s expert_rows=%s lr=%s warmup=%s expert=%s commit=%s\n' "$RUN_ID" "$CANDIDATE" "$GPU_INDEX" "$UPDATES" "$BATCH_SIZE" "$EXPERT_ROWS" "$LEARNING_RATE" "$WARMUP" "$EXPERT_INDEX" "$COMMIT"
if ((DRY_RUN)); then printf 'dry-run passed; no output/tmux created\n'; exit 0; fi

P0_WORKTREE="$(jq -r .candidates.p0.worktree "$REFERENCE_MANIFEST")"; P0_BRANCH="$(jq -r .candidates.p0.branch "$REFERENCE_MANIFEST")"
P0_COMMIT="$(jq -r .candidates.p0.commit "$REFERENCE_MANIFEST")"; P0_CONFIG="$(jq -r .candidates.p0.config "$REFERENCE_MANIFEST")"
P0_CHECKPOINT="$(jq -r .candidates.p0.checkpoint "$REFERENCE_MANIFEST")"
"$PYTHON" "$ROOT/scripts/before_we_act/r15_runtime.py" register --run-root "$RUN_ROOT" --run-id "$RUN_ID" --split "$SPLIT" --seed-file "$SEED_FILE" --seed-file-sha256 "$SEED_SHA" --candidate p0 --label w12_control --gpu 0 --worktree "$P0_WORKTREE" --branch "$P0_BRANCH" --commit "$P0_COMMIT" --config "$P0_CONFIG" --checkpoint "$P0_CHECKPOINT" --reference
ln -s "$REFERENCE_RUN/candidates/p0/validation" "$RUN_ROOT/candidates/p0/validation"
CONFIG="$ROOT/configs/before_we_act/r12_action/e1_p2.yaml"
"$PYTHON" "$ROOT/scripts/before_we_act/r15_runtime.py" register --run-root "$RUN_ROOT" --run-id "$RUN_ID" --split "$SPLIT" --seed-file "$SEED_FILE" --seed-file-sha256 "$SEED_SHA" --candidate "$CANDIDATE" --label "$LABEL" --gpu "$GPU_INDEX" --worktree "$ROOT" --branch "$BRANCH" --commit "$COMMIT" --config "$CONFIG" --checkpoint "$CHECKPOINT" --session "$SESSION"
"$PYTHON" - "$RUN_ROOT/candidates/$CANDIDATE/expert_finetune.json" "$EXPERT_INDEX" "$PHASE_MANIFEST" "$REFERENCE_RUN" "$UPDATES" "$BATCH_SIZE" "$EXPERT_ROWS" "$LEARNING_RATE" "$WARMUP" <<'PY'
import datetime,hashlib,json,os,sys
target,index,phase_manifest,reference,updates,batch,expert_rows,learning_rate,warmup=sys.argv[1:]
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()
d={"schema_version":1,"protocol":"r15_stack_phase_balanced_expert_source_aware_v1" if phase_manifest else "r15_stack_original_plus_raw_success_expert_source_aware_v2","expert_index":os.path.realpath(index),"expert_index_sha256":sha(index),"phase_manifest":os.path.realpath(phase_manifest) if phase_manifest else None,"phase_manifest_sha256":sha(phase_manifest) if phase_manifest else None,"reference_run_root":os.path.realpath(reference),"updates":int(updates),"batch_size":int(batch),"expert_rows":int(expert_rows),"learning_rate":float(learning_rate),"warmup":int(warmup),"created_at":datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z')}
tmp=target+f'.{os.getpid()}.tmp'; open(tmp,'w').write(json.dumps(d,indent=2,sort_keys=True)+'\n'); os.replace(tmp,target)
PY
tmux new-session -d -s "$SESSION" -n expert-ft \
  "cd '$ROOT' && exec env BWA_R15_RUN_ROOT='$RUN_ROOT' BWA_R15_CANDIDATE='$CANDIDATE' '$ROOT/scripts/before_we_act/run_r15_expert_finetune_candidate.sh' --run-root '$RUN_ROOT' --candidate '$CANDIDATE' --gpu-index '$GPU_INDEX' --expert-index '$EXPERT_INDEX' ${PHASE_ARGS[*]} --updates '$UPDATES' --batch-size '$BATCH_SIZE' --expert-rows '$EXPERT_ROWS' --learning-rate '$LEARNING_RATE' --warmup '$WARMUP' --python '$PYTHON'"
printf 'started session=%s output=%s\n' "$SESSION" "$RUN_ROOT"
printf 'monitor: %s/scripts/before_we_act/monitor_r15_stack_screens.sh --run-root %s --candidate all --interval 30\n' "$ROOT" "$RUN_ROOT"
printf 'safe stop: %s/scripts/before_we_act/stop_r15_stack_screens.sh --run-root %s --candidate %s\n' "$ROOT" "$RUN_ROOT" "$CANDIDATE"
