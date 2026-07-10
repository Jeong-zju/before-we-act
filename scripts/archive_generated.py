from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_TARGETS = ["datasets", "checkpoints", "artifacts", "outputs", "logs"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def path_size_bytes(path: Path) -> int:
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size

    total = 0
    for child in path.rglob("*"):
        try:
            total += child.lstat().st_size
        except FileNotFoundError:
            continue
    return total


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{idx:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not choose a unique destination for {path}")


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def collect_cache_dirs(root: Path, archive_root: Path) -> list[Path]:
    caches = []
    pytest_cache = root / ".pytest_cache"
    if pytest_cache.exists():
        caches.append(pytest_cache)
    for path in root.rglob("__pycache__"):
        if is_inside(path, archive_root):
            continue
        caches.append(path)
    return sorted(set(caches))


def make_archive_plan(root: Path, archive_dir: Path, targets: Iterable[str], clean_caches: bool) -> dict:
    moves = []
    for target in targets:
        src = (root / target).resolve()
        if not src.exists() and not src.is_symlink():
            moves.append(
                {
                    "action": "move",
                    "source": str(src),
                    "destination": None,
                    "exists": False,
                    "size_bytes": 0,
                    "size": "0.0B",
                }
            )
            continue
        if src == archive_dir or is_inside(src, archive_dir):
            raise ValueError(f"Refusing to archive the archive directory itself: {src}")
        dst = unique_destination(archive_dir / target)
        size = path_size_bytes(src)
        moves.append(
            {
                "action": "move",
                "source": str(src),
                "destination": str(dst),
                "exists": True,
                "size_bytes": size,
                "size": human_size(size),
            }
        )

    cache_removals = []
    if clean_caches:
        for cache in collect_cache_dirs(root, archive_dir.parent):
            size = path_size_bytes(cache)
            cache_removals.append(
                {
                    "action": "remove_cache",
                    "source": str(cache),
                    "destination": None,
                    "exists": True,
                    "size_bytes": size,
                    "size": human_size(size),
                }
            )

    return {
        "created_at_utc": utc_stamp(),
        "root": str(root),
        "archive_dir": str(archive_dir),
        "moves": moves,
        "cache_removals": cache_removals,
        "total_move_bytes": sum(item["size_bytes"] for item in moves if item["exists"]),
        "total_cache_bytes": sum(item["size_bytes"] for item in cache_removals if item["exists"]),
    }


def execute_plan(plan: dict):
    archive_dir = Path(plan["archive_dir"])
    archive_dir.mkdir(parents=True, exist_ok=True)

    for item in plan["moves"]:
        if not item["exists"]:
            continue
        src = Path(item["source"])
        dst = Path(item["destination"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"archive: {src} -> {dst}")
        shutil.move(str(src), str(dst))

    for item in plan["cache_removals"]:
        src = Path(item["source"])
        if src.exists():
            print(f"remove cache: {src}")
            shutil.rmtree(src)

    manifest_path = archive_dir / "archive_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(plan, f, indent=2)
    print("wrote:", manifest_path)


def main():
    parser = argparse.ArgumentParser(description="Archive generated FE-PC-WAM files without deleting model/data outputs.")
    parser.add_argument("--root", type=str, default=str(repo_root()))
    parser.add_argument("--archive_root", type=str, default="archive")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--execute", action="store_true", help="Move files. Without this flag, only prints the plan.")
    parser.add_argument("--clean-caches", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    archive_root = Path(args.archive_root)
    if not archive_root.is_absolute():
        archive_root = root / archive_root
    archive_dir = archive_root / utc_stamp()

    plan = make_archive_plan(root=root, archive_dir=archive_dir, targets=args.targets, clean_caches=bool(args.clean_caches))

    print(json.dumps(plan, indent=2))
    print("total archive size:", human_size(plan["total_move_bytes"]))
    print("total cache size:", human_size(plan["total_cache_bytes"]))

    if args.execute:
        execute_plan(plan)
    else:
        print("dry-run only. Re-run with --execute to move generated files.")


if __name__ == "__main__":
    main()
