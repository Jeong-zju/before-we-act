#!/usr/bin/env bash
set -Eeuo pipefail

# This wrapper is intentionally the only entry point enabled by supervisor.
# It does not start automatically from the repository; deployment tooling may
# opt in by installing the accompanying .conf.  Credentials are inherited
# from the host environment and never appear in argv or receipts.
if [[ -f /workspace/.env ]]; then
  # shellcheck disable=SC1091
  set -a
  . /workspace/.env
  set +a
fi

export DUO_DINO_REPO="${DUO_DINO_REPO:-/workspace/repos/care-official}"
export DUO_DINO_DUOBENCH_REPO="${DUO_DINO_DUOBENCH_REPO:-/workspace/repos/duobench}"
export DUO_DINO_DATASET="${DUO_DINO_DATASET:-/workspace/datasets/duobench}"
export DUO_DINO_RUN="${DUO_DINO_RUN:-/workspace/runs/duobench-care-dino-v1}"
export DUO_DINO_MODEL="${DUO_DINO_MODEL:-/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m}"
export DUO_DINO_PYTHON="${DUO_DINO_PYTHON:-/venv/main/bin/python}"

exec "${DUO_DINO_PYTHON}" -u -m deployment.duo_dino_reference.supervisor run
