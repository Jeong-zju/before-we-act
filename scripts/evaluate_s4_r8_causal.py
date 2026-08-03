#!/usr/bin/env python3
"""Run common Gate20 and offline causal evaluation for one S4-R8 candidate."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_s4_r7_causal import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
