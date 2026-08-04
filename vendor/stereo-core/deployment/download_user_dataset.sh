#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 1 ]]; then
  printf 'usage: %s <task>\n' "$0" >&2
  exit 2
fi

task="$1"
case "$task" in
  lift_barrier)
    repo='zeno-ai/robofactory-lift-barrier-multiview'
    revision='6ab620091677e69370412f08cd7adecacc28c146'
    ;;
  long_pipeline_delivery)
    repo='zeno-ai/robofactory-long-pipeline-delivery-multiview'
    revision='fee628311ff52a3ae0ddfddf82379c63d28f7533'
    ;;
  take_photo)
    repo='zeno-ai/robofactory-take-photo-multiview'
    revision='3966385a4c688a5610d4b6cde044150f6b73d320'
    ;;
  three_robots_stack_cube)
    repo='zeno-ai/robofactory-three-robots-stack-cube-multiview'
    revision='d0ae346bf2ce63ec801af1f036c08a4a91faf366'
    ;;
  camera_alignment)
    repo='zeno-ai/robofactory-camera-alignment-multiview'
    revision='e204af13f7191dfd86dab3da529316a51558f479'
    ;;
  *)
    printf 'unknown task: %s\n' "$task" >&2
    exit 2
    ;;
esac

destination="/workspace/datasets/robofactory_multitask/$task"
mkdir -p "$destination"
attempt=1
delay=15
while (( attempt <= 8 )); do
  printf '[%s] attempt %d/8 repo=%s revision=%s\n' \
    "$(date -u +%FT%TZ)" "$attempt" "$repo" "$revision"
  if HF_HOME=/workspace/.hf_home \
      HF_HUB_DISABLE_XET=0 \
      HF_XET_HIGH_PERFORMANCE=1 \
      /venv/main/bin/hf download "$repo" \
        --repo-type dataset \
        --revision "$revision" \
        --local-dir "$destination"; then
    printf '[%s] download complete: %s\n' "$(date -u +%FT%TZ)" "$task"
    touch "$destination/.download-complete"
    exit 0
  fi
  if (( attempt == 8 )); then
    break
  fi
  printf '[%s] download failed; retrying in %d seconds\n' \
    "$(date -u +%FT%TZ)" "$delay" >&2
  sleep "$delay"
  attempt=$((attempt + 1))
  delay=$((delay * 2))
  (( delay > 300 )) && delay=300
done

printf '[%s] download failed permanently: %s\n' "$(date -u +%FT%TZ)" "$task" >&2
exit 1

