#!/usr/bin/env bash
set -Eeuo pipefail
OPENPI_DIR=${1:?usage: apply.sh /path/to/openpi}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
EXPECTED=15a9616a00943ada6c20a0f158e3adb39df2ccac
[[ "$(git -C "$OPENPI_DIR" rev-parse HEAD)" == "$EXPECTED" ]]
git -C "$OPENPI_DIR" apply --check "$ROOT/openpi_duobench.patch"
git -C "$OPENPI_DIR" apply "$ROOT/openpi_duobench.patch"
install -D -m 0644 "$ROOT/src/openpi/policies/duobench_policy.py" "$OPENPI_DIR/src/openpi/policies/duobench_policy.py"
install -D -m 0644 "$ROOT/src/openpi/training/duobench_dataset.py" "$OPENPI_DIR/src/openpi/training/duobench_dataset.py"
install -D -m 0644 "$ROOT/src/openpi/training/duobench_config.py" "$OPENPI_DIR/src/openpi/training/duobench_config.py"
echo "Applied DuoBench pi0.5 overlay"
