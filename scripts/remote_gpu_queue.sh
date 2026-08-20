#!/usr/bin/env bash
set -u

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU QUEUE_NAME COMMAND [COMMAND ...]" >&2
  exit 2
fi

gpu=$1
queue_name=$2
shift 2
root=${BWA_RUN_ROOT:-/workspace/bwa-baselines-runs}
queue_dir="${root}/queues/${queue_name}"
mkdir -p "${queue_dir}"

for command in "$@"; do
  job=$(printf '%s' "${command}" | sha256sum | cut -c1-12)
  status="${queue_dir}/${job}.json"
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '{"gpu":%s,"command":%s,"status":"running","started_at":"%s"}\n' \
    "${gpu}" "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${command}")" "${started}" > "${status}"

  CUDA_VISIBLE_DEVICES="${gpu}" bash -lc "${command}"
  rc=$?
  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [[ ${rc} -eq 0 ]]; then
    state=completed
  else
    state=failed
  fi
  printf '{"gpu":%s,"command":%s,"status":"%s","exit_code":%s,"started_at":"%s","finished_at":"%s"}\n' \
    "${gpu}" "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${command}")" \
    "${state}" "${rc}" "${started}" "${finished}" > "${status}"
done
