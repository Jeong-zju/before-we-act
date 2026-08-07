#!/usr/bin/env bash
set -Euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTERVAL=30; ONCE=0
declare -a SCREEN_ROOTS=() SCREEN_CANDIDATES=() COLLECTION_ROOTS=() CACHE_ROOTS=()
while (($#)); do
  case "$1" in
    --screen)
      spec="$2"; candidate="${spec##*:}"; run_root="${spec%:*}"
      [[ "$candidate" =~ ^p[0-3]$ && -n "$run_root" && "$run_root" != "$spec" ]] || { printf 'screen must be RUN_ROOT:p0..p3\n' >&2; exit 2; }
      SCREEN_ROOTS+=("$run_root"); SCREEN_CANDIDATES+=("$candidate"); shift 2 ;;
    --expert-collection) COLLECTION_ROOTS+=("$2"); shift 2 ;;
    --expert-cache) CACHE_ROOTS+=("$2"); shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --once) ONCE=1; shift ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || { printf 'positive interval required\n' >&2; exit 2; }
(( ${#SCREEN_ROOTS[@]} + ${#COLLECTION_ROOTS[@]} + ${#CACHE_ROOTS[@]} > 0 )) || { printf 'at least one screen, collection, or cache target is required\n' >&2; exit 2; }

snapshot() {
  printf 'R15 evolution portfolio | %s | screens=%s collections=%s caches=%s\n\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#SCREEN_ROOTS[@]}" "${#COLLECTION_ROOTS[@]}" "${#CACHE_ROOTS[@]}"
  local index
  for index in "${!SCREEN_ROOTS[@]}"; do
    "$ROOT/scripts/before_we_act/monitor_r15_stack_screens.sh" \
      --run-root "${SCREEN_ROOTS[$index]}" \
      --candidate "${SCREEN_CANDIDATES[$index]}" --once || \
      printf 'screen monitor failed root=%s candidate=%s\n' "${SCREEN_ROOTS[$index]}" "${SCREEN_CANDIDATES[$index]}"
  done
  for index in "${!COLLECTION_ROOTS[@]}"; do
    "$ROOT/scripts/before_we_act/monitor_r15_expert_collection.sh" \
      --output-root "${COLLECTION_ROOTS[$index]}" --once || \
      printf 'expert collection monitor failed root=%s\n' "${COLLECTION_ROOTS[$index]}"
  done
  for index in "${!CACHE_ROOTS[@]}"; do
    "$ROOT/scripts/before_we_act/monitor_r15_expert_cache.sh" \
      --cache-root "${CACHE_ROOTS[$index]}" --once || \
      printf 'expert cache monitor failed root=%s\n' "${CACHE_ROOTS[$index]}"
  done
}

while :; do
  ((ONCE)) || printf '\033[2J\033[H'
  snapshot
  ((ONCE)) && break
  sleep "$INTERVAL"
done
