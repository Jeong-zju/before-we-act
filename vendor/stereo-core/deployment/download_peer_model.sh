#!/usr/bin/env bash
set -Eeuo pipefail

destination=/workspace/artifacts/Stereo-CoRE
mkdir -p "$destination"
attempt=1
delay=15
while (( attempt <= 8 )); do
  printf '[%s] attempt %d/8: B111ue/Stereo-CoRE\n' "$(date -u +%FT%TZ)" "$attempt"
  if HF_HOME=/workspace/.hf_home \
      HF_HUB_DISABLE_XET=0 \
      HF_XET_HIGH_PERFORMANCE=1 \
      /venv/main/bin/hf download B111ue/Stereo-CoRE \
        --revision main \
        --local-dir "$destination"; then
    printf '[%s] peer model download complete\n' "$(date -u +%FT%TZ)"
    touch "$destination/.download-complete"
    exit 0
  fi
  if (( attempt == 8 )); then
    break
  fi
  printf '[%s] peer model download failed; retrying in %d seconds\n' \
    "$(date -u +%FT%TZ)" "$delay" >&2
  sleep "$delay"
  attempt=$((attempt + 1))
  delay=$((delay * 2))
  (( delay > 300 )) && delay=300
done
exit 1
