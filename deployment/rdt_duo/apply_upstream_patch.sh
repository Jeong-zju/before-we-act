#!/usr/bin/env bash
set -Eeuo pipefail

# Replay the exact upstream files used by the formal DuoBench run.  The patch
# is generated against RDT commit cd79363a; it intentionally includes the
# computed dataset_stat.json so a public rerun does not depend on a hidden
# server-side file.
repo="${1:-${RDT_DUO_UPSTREAM:-/workspace/repos/rdt-1b}}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
expected="cd79363a1387e8f81c7724d070ef7e45fd23150f"
test -d "$repo/.git" || { echo "not a git checkout: $repo" >&2; exit 2; }
actual="$(git -C "$repo" rev-parse HEAD)"
test "$actual" = "$expected" || { echo "expected RDT $expected, got $actual" >&2; exit 3; }
gzip -dc "$here/upstream_rdt1b_duobench.patch.gz" | git -C "$repo" apply --3way --whitespace=nowarn -
for package in configs data models train; do
  if [[ ! -e "$repo/$package/__init__.py" ]]; then
    printf '%s\n' '"""Official RDT namespace."""' > "$repo/$package/__init__.py"
  fi
done
echo "applied DuoBench RDT-1B upstream overlay to $repo"
