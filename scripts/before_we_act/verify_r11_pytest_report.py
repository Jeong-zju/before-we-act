#!/usr/bin/env python3
"""Fail closed on any skipped or incomplete remote R11 F0 test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from before_we_act.train_r11_candidate import atomic_json, sha256_file


def summarize(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    if not suites:
        raise ValueError("JUnit report has no testsuite")
    totals = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    if totals["tests"] <= 0:
        raise ValueError("JUnit report executed no tests")
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    junit = args.junit.resolve(strict=True)
    totals = summarize(junit)
    passed = all(totals[name] == 0 for name in ("failures", "errors", "skipped"))
    result = {
        "format_version": "before-we-act.r11.f0_pytest_receipt/1",
        "status": "PASSED" if passed else "FAILED",
        "junit": str(junit),
        "junit_sha256": sha256_file(junit),
        **totals,
        "completed_at_epoch": time.time(),
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit(10)


if __name__ == "__main__":
    main()
