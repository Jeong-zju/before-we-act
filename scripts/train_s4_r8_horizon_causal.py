#!/usr/bin/env python3
"""Train one fail-closed S4-R8 horizon-causal candidate."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_s4_r7_world_utility import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
