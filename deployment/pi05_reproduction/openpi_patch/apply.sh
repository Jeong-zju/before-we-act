#!/usr/bin/env bash
set -Eeuo pipefail

OPENPI_DIR=${1:?usage: apply.sh /path/to/openpi}
PATCH_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
EXPECTED_COMMIT=15a9616a00943ada6c20a0f158e3adb39df2ccac

[[ -d "$OPENPI_DIR/.git" ]] || { echo "Not an OpenPI git checkout: $OPENPI_DIR" >&2; exit 2; }
actual=$(git -C "$OPENPI_DIR" rev-parse HEAD)
[[ "$actual" == "$EXPECTED_COMMIT" ]] || {
  echo "OpenPI commit mismatch: expected $EXPECTED_COMMIT, got $actual" >&2
  exit 3
}

git -C "$OPENPI_DIR" apply --check "$PATCH_DIR/openpi_tracked.patch"
git -C "$OPENPI_DIR" apply "$PATCH_DIR/openpi_tracked.patch"
install -D -m 0644 "$PATCH_DIR/src/openpi/policies/robofactory_policy.py" "$OPENPI_DIR/src/openpi/policies/robofactory_policy.py"
install -D -m 0644 "$PATCH_DIR/src/openpi/training/robofactory_config.py" "$OPENPI_DIR/src/openpi/training/robofactory_config.py"
install -D -m 0644 "$PATCH_DIR/src/openpi/training/robofactory_dataset.py" "$OPENPI_DIR/src/openpi/training/robofactory_dataset.py"
echo "Applied π0.5 RoboFactory patch to $OPENPI_DIR"
