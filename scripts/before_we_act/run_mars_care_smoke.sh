#!/usr/bin/env bash
set -Eeuo pipefail

repo=${MARS_CARE_REPO:-/workspace/repos/care-official}
py=${MARS_PYTHON:-/workspace/venvs/mars/bin/python}
raw=${MARS_RAW_ROOT:-/workspace/datasets/mars_control/raw}
run=${MARS_CARE_RUN_ROOT:-/workspace/runs/care_official_mars_v1}
rf=${MARS_ROBOFACTORY_ROOT:-/workspace/repos/RoboFactory}
dino=${MARS_DINO_MODEL:-/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m}
norm=${run}/contract/mars_norm_absolute.json
cache=${MARS_VISUAL_CACHE:-/workspace/runs/care_dino_mars/dino_cache}
manifest=${run}/contract/care_family_manifest.json
logs=${run}/logs/smoke

export PYTHONPATH=${repo}:${repo}/stereo_core${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${run}/contract" "${logs}"
cd "${repo}"

pass_json() {
  "${py}" - "${1}" "${2}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
raise SystemExit(0 if x.get("status")==sys.argv[2] else 1)
PY
}

if ! pass_json "${cache}/cache_receipt.json" PASSED 2>/dev/null; then
  CUDA_VISIBLE_DEVICES=0,1,2,3 "${py}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m before_we_act.cache_mars_dino --raw-root "${raw}" --dino-model "${dino}" \
    --output "${cache}" --batch-size 64 >"${logs}/dino_cache.log" 2>&1
fi
if [[ ! -f "${norm}" ]]; then
  "${py}" - "${raw}" "${norm}" "${run}/contract/raw_audit.json" <<'PY'
import json,sys
from pathlib import Path
from before_we_act.mars_temporal_data import compute_normalization, load_mars_episodes
raw,norm,audit=map(Path,sys.argv[1:])
episodes=load_mars_episodes(raw)
stats=compute_normalization(episodes,norm)
audit.parent.mkdir(parents=True,exist_ok=True)
audit.write_text(json.dumps({"status":"PASSED","episodes":len(episodes),"action_encoding":stats["action_encoding"],"normalization":stats},indent=2)+"\n")
PY
fi
if [[ ! -f "${manifest}" ]]; then
  "${py}" -m scripts.before_we_act.prepare_mars_care_training manifest \
    --raw-root "${raw}" --output "${manifest}" --families-per-task 30 \
    >"${logs}/family_manifest.log" 2>&1
fi

f1fresh=${run}/smoke/b0h_f1_fresh
f1resume=${run}/smoke/b0h_f1_resume
if [[ ! -f "${run}/contract/b0h_f1_receipt.json" ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3 "${py}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m before_we_act.train_mars_temporal_policy --stage f1 --raw-root "${raw}" \
    --normalization "${norm}" --visual-cache "${cache}" --dino-model "${dino}" \
    --output "${f1fresh}" --updates 4 --workers 0 --save-every 2 --log-every 1 \
    >"${logs}/b0h_f1_fresh.log" 2>&1
  mkdir -p "${f1resume}"
  cp "${f1fresh}/checkpoint_000002.pt" "${f1resume}/checkpoint_latest.pt"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "${py}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m before_we_act.train_mars_temporal_policy --stage f1 --raw-root "${raw}" \
    --normalization "${norm}" --visual-cache "${cache}" --dino-model "${dino}" \
    --output "${f1resume}" --updates 4 --workers 0 --save-every 2 --log-every 1 \
    --resume "${f1resume}/checkpoint_latest.pt" >"${logs}/b0h_f1_resume.log" 2>&1
  "${py}" scripts/before_we_act/verify_mars_b0h_smoke.py \
    --reference "${f1fresh}/checkpoint_000004.pt" --resumed "${f1resume}/checkpoint_000004.pt" \
    --output "${run}/contract/b0h_f1_receipt.json" >"${logs}/b0h_f1_verify.log" 2>&1
fi

b0h=${run}/smoke/b0h_f1_fresh/checkpoint_000004.pt
for task in place_cube_in_cup strike_cube_hard three_robots_place_shoes four_robots_stack_cube; do
  out=${run}/smoke/b0h_closed_loop/${task}.json
  [[ -f "${out}" ]] && continue
  CUDA_VISIBLE_DEVICES=0 "${py}" -m before_we_act.evaluate_mars_temporal_policy \
    --checkpoint "${b0h}" --dino-model "${dino}" --task "${task}" --robofactory-root "${rf}" \
    --output "${out}" --episodes 1 --seed-start 20269900 --max-steps 2 --device cuda:0 \
    >"${logs}/b0h_closed_loop_${task}.log" 2>&1
done
"${py}" - "${run}/smoke/b0h_closed_loop" "${run}/contract/b0h_closed_loop_smoke_receipt.json" <<'PY'
import json,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:])
tasks=("place_cube_in_cup","strike_cube_hard","three_robots_place_shoes","four_robots_stack_cube")
rows={t:json.loads((root/f"{t}.json").read_text()) for t in tasks}
assert all(x.get("status")=="complete" and x.get("episodes")==1 and x["rows"][0]["steps"]==2 for x in rows.values())
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({"status":"PASSED","tasks":tasks,"episodes":4,"max_steps_each":2},indent=2)+"\n")
PY

action_smoke=${run}/smoke/action_context
belief1=${run}/smoke/belief_f1_fresh
belief2=${run}/smoke/belief_f1_resume
if ! pass_json "${run}/contract/belief_closed_loop_smoke_receipt.json" PASSED 2>/dev/null; then
  if ! pass_json "${action_smoke}/cache_receipt.json" PASSED 2>/dev/null; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 "${py}" -m torch.distributed.run --standalone --nproc_per_node=4 \
      -m scripts.before_we_act.build_mars_action_context_cache --raw-root "${raw}" --normalization "${norm}" \
      --visual-cache "${cache}" --temporal-checkpoint "${b0h}" --dino-model "${dino}" \
      --output "${action_smoke}" --batch-size 16 --episodes-per-task 1 >"${logs}/action_context_smoke.log" 2>&1
  fi
  CUDA_VISIBLE_DEVICES=0 "${py}" -m before_we_act.train_mars_predictive_team_belief \
    --raw-root "${raw}" --normalization "${norm}" --visual-cache "${cache}" \
    --action-context-cache "${action_smoke}" --b0h-checkpoint "${b0h}" --output "${belief1}" \
    --seed 20269901 --updates 2 --workers 0 --save-every 2 --log-every 1 --episodes-per-task 1 \
    >"${logs}/belief_f1_fresh.log" 2>&1
  mkdir -p "${belief2}"
  cp "${belief1}/checkpoint_latest.pt" "${belief2}/checkpoint_latest.pt"
  CUDA_VISIBLE_DEVICES=0 "${py}" -m before_we_act.train_mars_predictive_team_belief \
    --raw-root "${raw}" --normalization "${norm}" --visual-cache "${cache}" \
    --action-context-cache "${action_smoke}" --b0h-checkpoint "${b0h}" --output "${belief2}" \
    --seed 20269901 --updates 4 --workers 0 --save-every 2 --log-every 1 --episodes-per-task 1 \
    >"${logs}/belief_f1_resume.log" 2>&1
  "${py}" scripts/before_we_act/build_mars_belief_smoke_deployment.py \
    --training-checkpoint "${belief2}/checkpoint_latest.pt" --b0h-checkpoint "${b0h}" \
    --output "${run}/smoke/belief_smoke_deployment.pt"
  for task in place_cube_in_cup strike_cube_hard three_robots_place_shoes four_robots_stack_cube; do
    out=${run}/smoke/belief_closed_loop/${task}.json
    CUDA_VISIBLE_DEVICES=0 "${py}" -m before_we_act.evaluate_mars_predictive_team_belief \
      --checkpoint "${run}/smoke/belief_smoke_deployment.pt" --task "${task}" --robofactory-root "${rf}" \
      --output "${out}" --episodes 1 --seed-start 20269910 --max-steps 2 --device cuda:0 \
      >"${logs}/belief_closed_loop_${task}.log" 2>&1
  done
  "${py}" - "${run}/smoke/belief_closed_loop" "${run}/contract/belief_closed_loop_smoke_receipt.json" <<'PY'
import json,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:])
tasks=("place_cube_in_cup","strike_cube_hard","three_robots_place_shoes","four_robots_stack_cube")
rows={t:json.loads((root/f"{t}.json").read_text()) for t in tasks}
assert all(x.get("status")=="complete" and x.get("episodes")==1 and x["rows"][0]["steps"]==2 for x in rows.values())
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({"status":"PASSED","tasks":tasks,"episodes":4,"max_steps_each":2},indent=2)+"\n")
PY
fi

if [[ ! -f "${run}/contract/care_end_to_end_smoke_receipt.json" ]]; then
  MARS_CARE_BELIEF_CHECKPOINT="${run}/smoke/belief_smoke_deployment.pt" \
    bash scripts/before_we_act/run_mars_care_branch_smoke.sh >"${logs}/care_end_to_end.log" 2>&1
fi
echo MARS_CARE_SMOKE_PASSED
