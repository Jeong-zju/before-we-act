#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--patch-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()
    lock = yaml.safe_load(Path(args.lock).read_text(encoding="utf-8"))
    project, upstream = Path(args.project_root), Path(args.upstream)
    destination = project / lock["local_destination"]
    whitelist = set(lock["copied_algorithm_files_edit_whitelist"])
    diffs, edited = [], []
    for relative in lock["copied_upstream_files"]:
        before = (upstream / relative).read_text(encoding="utf-8").splitlines(keepends=True)
        after = (destination / relative).read_text(encoding="utf-8").splitlines(keepends=True)
        if before == after:
            continue
        edited.append(relative)
        diffs.extend(
            difflib.unified_diff(before, after, fromfile=f"upstream/{relative}", tofile=f"local/{relative}")
        )
    unexpected = sorted(set(edited) - whitelist)
    patch = "".join(diffs)
    patch_path = Path(args.patch_output)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch, encoding="utf-8")
    algorithmic_lines_changed = int(lock["algorithmic_lines_changed"])
    passed = not unexpected and (not edited or algorithmic_lines_changed == 0)
    result = {
        "schema_version": 1,
        "candidate_id": lock["candidate_id"],
        "passed": passed,
        "edited_upstream_files": edited,
        "unexpected_edited_files": unexpected,
        "algorithmic_lines_changed": algorithmic_lines_changed,
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "adaptation_only_assertion": algorithmic_lines_changed == 0,
    }
    report = Path(args.report_output)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
