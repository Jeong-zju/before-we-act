#!/usr/bin/env python3
"""Fail closed when the CARE/RoboFactory reproduction contract drifts."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from before_we_act.frozen_settings import (
    DEFAULT_SETTINGS_PATH,
    load_frozen_settings,
    verify_dependency_lock,
    verify_robofactory_checkout,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--robofactory-root", type=Path, required=True)
    args = parser.parse_args()
    settings = load_frozen_settings(args.settings)
    commit = verify_robofactory_checkout(args.robofactory_root, settings)
    lock_hash = verify_dependency_lock(args.repo_root, settings)
    print(f"FROZEN_SETTINGS_OK commit={commit} uv_lock_sha256={lock_hash}")


if __name__ == "__main__":
    main()
