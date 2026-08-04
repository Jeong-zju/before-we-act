#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


FORBIDDEN_PREFIXES = ("vjepa2", "lpwm", "dino_wm", "lerobot")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    violations = []
    scanned = 0
    for path in sorted((root / "before_we_act").rglob("*.py")):
        relative = path.relative_to(root)
        if "upstream_components" in relative.parts and "tests" in relative.parts:
            continue
        scanned += 1
        source = path.read_text(encoding="utf-8")
        if "/workspace/bwa_upstream" in source or "/tmp/bwa-r11-upstreams" in source:
            violations.append({"path": str(relative), "reason": "absolute upstream cache path"})
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.startswith("before_we_act.upstream_components"):
                    continue
                if name.split(".", 1)[0] in FORBIDDEN_PREFIXES:
                    violations.append({"path": str(relative), "reason": f"full upstream import {name}"})
    result = {
        "schema_version": 1,
        "passed": not violations,
        "full_repo_runtime_dependency": bool(violations),
        "scanned_python_files": scanned,
        "violations": violations,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
