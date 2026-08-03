#!/usr/bin/env python3
"""Create or validate the shared five-task HDF5 verification receipt."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.shared_hdf5_receipt import (  # noqa: E402
    create_shared_hdf5_receipt,
    file_sha256,
    validate_shared_hdf5_receipt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--proof-checkpoint", type=Path)
    parser.add_argument("--expected-proof-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--expected-receipt-sha256")
    parser.add_argument("--verify-imported-content-if-newer", action="store_true")
    parser.add_argument("--progress-log", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.verify:
        payload = validate_shared_hdf5_receipt(
            args.output,
            args.manifest,
            expected_proof_sha256=args.expected_proof_sha256,
            expected_receipt_sha256=args.expected_receipt_sha256,
        )
    else:
        if args.proof_checkpoint is None:
            raise ValueError("--proof-checkpoint is required when creating a receipt")
        if args.progress_log is not None:
            args.progress_log.parent.mkdir(parents=True, exist_ok=True)

        def progress(value: Mapping[str, object]) -> None:
            if args.progress_log is None:
                return
            with args.progress_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(value), sort_keys=True) + "\n")

        payload = create_shared_hdf5_receipt(
            args.manifest,
            proof_checkpoint=args.proof_checkpoint,
            expected_proof_sha256=args.expected_proof_sha256,
            output=args.output,
            verify_imported_content_if_newer=args.verify_imported_content_if_newer,
            progress=progress,
        )
    print(
        json.dumps(
            {
                "receipt": str(args.output.resolve(strict=True)),
                "sha256": file_sha256(args.output),
                "manifests": len(payload["manifests"]),
                "files": len(payload["files"]),
                "verified": args.verify,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
