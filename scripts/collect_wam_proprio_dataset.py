"""Convenience entry point for image-free Phase 0 WAM collection."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_modular_dataset import main as collect_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    user_args = list(sys.argv[1:] if argv is None else argv)
    return collect_main(
        [
            "--format",
            "hdf5",
            "--profile",
            "wam_proprio",
            "--behavior-profile",
            "phase0_mixed_v1",
            *user_args,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
