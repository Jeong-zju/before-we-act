#!/usr/bin/env python3
"""Download the public Stereo-CoRE model package and All-5 RGB-D dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="artifacts/Stereo-CoRE")
    parser.add_argument("--data-dir", default="data/RoboFactory-5Task-RGBD-Decentralized")
    parser.add_argument("--models-only", action="store_true")
    args = parser.parse_args()

    snapshot_download(
        "B111ue/Stereo-CoRE",
        repo_type="model",
        local_dir=Path(args.model_dir),
    )
    if not args.models_only:
        snapshot_download(
            "B111ue/RoboFactory-5Task-RGBD-Decentralized",
            repo_type="dataset",
            local_dir=Path(args.data_dir),
        )


if __name__ == "__main__":
    main()
