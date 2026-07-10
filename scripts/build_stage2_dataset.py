from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CollectionSpec:
    name: str
    mode: str
    episodes: int
    seed_start: int
    noise_std: float
    randomize: int = 1


FULL_RECIPE = [
    CollectionSpec(name="scripted", mode="scripted", episodes=1000, seed_start=0, noise_std=0.0),
    CollectionSpec(name="noisy", mode="noisy", episodes=500, seed_start=100000, noise_std=10.0),
    CollectionSpec(name="recovery", mode="recovery", episodes=100, seed_start=200000, noise_std=0.0),
]

SMOKE_RECIPE = [
    CollectionSpec(name="scripted", mode="scripted", episodes=10, seed_start=900000, noise_std=0.0),
    CollectionSpec(name="noisy", mode="noisy", episodes=6, seed_start=910000, noise_std=10.0),
    CollectionSpec(name="recovery", mode="recovery", episodes=4, seed_start=920000, noise_std=0.0),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def command_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) if not current else f"{root}{os.pathsep}{current}"
    return env


def run_python(root: Path, args: list[str]):
    cmd = [sys.executable, *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=root, env=command_env(root), check=True)


def archive_path(path: Path, archive_root: Path, label: str, run_id: str, keep_name: bool = True) -> Path | None:
    if not path.exists() and not path.is_symlink():
        return None
    destination = archive_root / label / run_id
    if keep_name:
        destination = destination / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Archive destination already exists: {destination}")
    shutil.move(str(path), str(destination))
    return destination


def list_episode_files(path: Path) -> list[Path]:
    return sorted(path.glob("episode_*.hdf5"))


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def validate_episode_files_are_real(stage2_dir: Path):
    for split in ["train", "val", "test"]:
        split_dir = stage2_dir / split
        files = list_episode_files(split_dir)
        if not files:
            raise RuntimeError(f"No episode files found in final split: {split_dir}")
        symlinks = [p for p in files if p.is_symlink()]
        if symlinks:
            raise RuntimeError(f"Final split contains symlinks, expected copied HDF5 files: {symlinks[:5]}")


def build_dataset_readme(manifest: dict[str, Any]) -> str:
    lines = [
        "# Stage 2 数据集",
        "",
        "本目录由 `scripts/build_stage2_dataset.py` 自动生成。",
        "",
        "## 数据采集配方",
        "",
    ]
    for item in manifest["recipe"]:
        lines.append(
            f"- `{item['name']}`: mode=`{item['mode']}`, episodes={item['episodes']}, "
            f"seed_start={item['seed_start']}, noise_std={item['noise_std']}"
        )

    split = manifest["split_manifest"]
    lines.extend(
        [
            "",
            "## 数据划分",
            "",
            f"- train: {split['num_train']} episodes",
            f"- val: {split['num_val']} episodes",
            f"- test: {split['num_test']} episodes",
            f"- total: {split['num_total']} episodes",
            "",
            "最终 `train/`、`val/` 和 `test/` 内的文件是真实复制出的 HDF5 文件，不是符号链接。",
            "",
            "## HDF5 Schema",
            "",
            "- `obs/robot_0/proprio`, `obs/robot_1/proprio`",
            "- `actions/joint`, `actions/robot_0`, `actions/robot_1`",
            "- `global/global_state`, `global/object_pose`, `global/force_proxy`, `global/contacts`, `global/robot_distance`",
            "- `labels/phase`, `labels/success`, `labels/failure`, `labels/failure_reason`, `labels/communication_dummy`",
            "- `rewards/reward`, `meta/time`, `meta/episode_index`, `meta/seed`",
            "",
            "## 数据诊断",
            "",
            "数据诊断结果已写入 `diagnostics/`。",
        ]
    )
    for split_name, summary in manifest["split_summaries"].items():
        lines.append(
            f"- {split_name}: success_rate={summary.get('success_rate', 0.0):.3f}, "
            f"mean_T={summary.get('mean_T', 0.0):.1f}, max_force_proxy={summary.get('max_force_proxy', 0.0):.3f}"
        )

    archived_build_root = manifest.get("archived_build_root")
    if archived_build_root:
        lines.extend(
            [
                "",
                "## 过程文件",
                "",
                f"最终复制数据集生成后，中间采集文件已移动到 `{archived_build_root}`。",
            ]
        )

    return "\n".join(lines) + "\n"


def recipe_from_name(name: str) -> list[CollectionSpec]:
    if name == "full":
        return FULL_RECIPE
    if name == "smoke":
        return SMOKE_RECIPE
    raise ValueError(f"Unknown recipe: {name}")


def main():
    parser = argparse.ArgumentParser(description="Collect, split, validate, document, and clean the Stage 2 dataset.")
    parser.add_argument("--recipe", choices=["full", "smoke"], default="full")
    parser.add_argument("--stage2_dir", type=str, default="datasets/stage2")
    parser.add_argument("--build_root", type=str, default="datasets/.stage2_builds")
    parser.add_argument("--archive_root", type=str, default="archive")
    parser.add_argument("--archive-existing", dest="archive_existing", action="store_true", default=True)
    parser.add_argument("--no-archive-existing", dest="archive_existing", action="store_false")
    parser.add_argument("--cleanup-build", dest="cleanup_build", action="store_true", default=False)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--num_diagnostic_examples", type=int, default=3)
    args = parser.parse_args()

    root = repo_root()
    run_id = utc_stamp()
    stage2_dir = (root / args.stage2_dir).resolve()
    build_root = (root / args.build_root / run_id).resolve()
    archive_root = (root / args.archive_root).resolve()

    archived_existing = None
    if stage2_dir.exists():
        if not args.archive_existing:
            raise FileExistsError(f"{stage2_dir} exists. Use --archive-existing or --stage2_dir to avoid clobbering it.")
        archived_existing = archive_path(stage2_dir, archive_root, "datasets_stage2_existing", run_id)
        print("archived existing dataset:", archived_existing)

    build_root.mkdir(parents=True, exist_ok=False)
    raw_root = build_root / "raw"
    recipe = recipe_from_name(args.recipe)

    collection_summaries = {}
    for spec in recipe:
        out_dir = raw_root / spec.name
        run_python(
            root,
            [
                "data/collect.py",
                "--mode",
                spec.mode,
                "--num_episodes",
                str(spec.episodes),
                "--out_dir",
                str(out_dir.relative_to(root)),
                "--seed_start",
                str(spec.seed_start),
                "--noise_std",
                str(spec.noise_std),
                "--randomize",
                str(spec.randomize),
            ],
        )
        run_python(root, ["data/validate_dataset.py", "--data_dir", str(out_dir.relative_to(root))])
        collection_summaries[spec.name] = load_json(out_dir / "summary.json")

    sources = [str((raw_root / spec.name).relative_to(root)) for spec in recipe]
    run_python(
        root,
        [
            "data/split_dataset.py",
            "--sources",
            *sources,
            "--out_root",
            str(stage2_dir.relative_to(root)),
            "--train_ratio",
            str(args.train_ratio),
            "--val_ratio",
            str(args.val_ratio),
            "--seed",
            str(args.split_seed),
            "--copy",
            "1",
        ],
    )

    for split in ["train", "val", "test"]:
        run_python(root, ["data/validate_dataset.py", "--data_dir", str((stage2_dir / split).relative_to(root))])

    validate_episode_files_are_real(stage2_dir)

    diagnostics_dir = stage2_dir / "diagnostics"
    split_summaries = {}
    for split in ["train", "val", "test"]:
        run_python(
            root,
            [
                "data/diagnostics.py",
                "--data_dir",
                str((stage2_dir / split).relative_to(root)),
                "--out_dir",
                str(diagnostics_dir.relative_to(root)),
                "--num_examples",
                str(args.num_diagnostic_examples),
            ],
        )
        split_summaries[split] = load_json(diagnostics_dir / f"{split}_summary.json")

    archived_build_root = None
    if args.cleanup_build:
        archived_build_root = archive_path(build_root, archive_root, "data_builds", run_id, keep_name=False)
        print("archived build root:", archived_build_root)

    manifest = {
        "created_at_utc": run_id,
        "recipe_name": args.recipe,
        "recipe": [asdict(spec) for spec in recipe],
        "stage2_dir": str(stage2_dir),
        "archived_existing_dataset": str(archived_existing) if archived_existing else None,
        "archived_build_root": str(archived_build_root) if archived_build_root else None,
        "collection_summaries": collection_summaries,
        "split_manifest": load_json(stage2_dir / "split_manifest.json"),
        "split_summaries": split_summaries,
        "final_files_are_copies": True,
    }

    with open(stage2_dir / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    (stage2_dir / "README.md").write_text(build_dataset_readme(manifest))

    print(json.dumps(manifest, indent=2))
    print("wrote:", stage2_dir / "dataset_manifest.json")
    print("wrote:", stage2_dir / "README.md")


if __name__ == "__main__":
    main()
