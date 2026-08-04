#!/usr/bin/env python3
"""Fail-closed audit for the R9 direct Stereo-CoRE source import."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MUTABLE_R9 = {
    "no_wrist_pair_model.py",
    "evaluate_no_wrist_pair.py",
    "train_no_wrist_pair.py",
    "stereo_decoder_variants.py",
    "train_stereo_act.py",
    "train_act.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "UPSTREAM_CORE_MANIFEST.json"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = ROOT / manifest["upstream_release"]["source_root"]
    import_root = ROOT / manifest["import_root"]
    rows = []
    failures = []
    declared = {row["path"]: row["sha256"] for row in manifest["files"]}
    actual_source = {
        path.name
        for path in source_root.iterdir()
        if path.is_file()
    } | {"LICENSE"}
    if set(declared) != actual_source:
        failures.append(
            f"manifest file set mismatch missing={sorted(actual_source-set(declared))} "
            f"extra={sorted(set(declared)-actual_source)}"
        )
    for relative, expected in sorted(declared.items()):
        source = (
            ROOT / manifest["upstream_release"]["license_source"]
            if relative == "LICENSE"
            else source_root / relative
        )
        imported = import_root / relative
        source_hash = sha256(source) if source.is_file() else None
        imported_hash = sha256(imported) if imported.is_file() else None
        source_ok = source_hash == expected
        imported_exact = imported_hash == expected
        allowed_modified = relative in MUTABLE_R9
        if not source_ok:
            failures.append(f"upstream hash mismatch: {relative}")
        if imported_hash is None:
            failures.append(f"missing imported file: {relative}")
        elif not imported_exact and not allowed_modified:
            failures.append(f"unexpected local source change: {relative}")
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "source_sha256": source_hash,
                "imported_sha256": imported_hash,
                "source_ok": source_ok,
                "imported_exact": imported_exact,
                "allowed_r9_modified": allowed_modified,
            }
        )
    result = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "files": rows,
        "failures": failures,
        "passed": not failures,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
