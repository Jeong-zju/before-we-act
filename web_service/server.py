#!/usr/bin/env python3
"""Compatibility launcher for the workspace RoboFactory control room."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    canonical = Path(__file__).resolve().parents[2] / "web_service" / "server.py"
    runpy.run_path(str(canonical), run_name="__main__")
