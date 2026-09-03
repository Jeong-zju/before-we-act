from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
from .protocol import CONTROL_FREQUENCY_HZ

def configure_repo(repo: Path) -> None:
    """Install the DuoBench data overlay into a pinned upstream RDT checkout."""
    here = Path(__file__).resolve().parent
    required = (repo / "configs", repo / "data", repo / "models", repo / "train")
    if not all(path.is_dir() for path in required):
        raise FileNotFoundError(f"not an RDT-1B checkout: {repo}")
    (repo / "data").mkdir(exist_ok=True); shutil.copy2(here / "hdf5_vla_dataset.py", repo / "data/hdf5_vla_dataset.py")
    (repo / "data/__init__.py").write_text('"""RDT data package with the DuoBench adapter."""\n')
    # Explicit package markers prevent a sibling benchmark's namespace package
    # from shadowing the official RDT modules during distributed launch.
    for package in ("models", "configs", "train"): (repo / package / "__init__.py").write_text('"""Official RDT namespace."""\n')
    control = json.loads((repo / "configs/dataset_control_freq.json").read_text()); control["duobench"] = CONTROL_FREQUENCY_HZ; (repo / "configs/dataset_control_freq.json").write_text(json.dumps(control, indent=2)+"\n")
    (repo / "configs/finetune_datasets.json").write_text(json.dumps(["duobench"], indent=2)+"\n")
    (repo / "configs/finetune_sample_weights.json").write_text(json.dumps([1.0], indent=2)+"\n")
    print("configured upstream RDT-1B for DuoBench")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("RDT_DUO_UPSTREAM", "/workspace/repos/rdt-1b")),
        help="Pinned RoboticsDiffusionTransformer checkout to configure.",
    )
    args = parser.parse_args()
    configure_repo(args.repo.resolve())

if __name__ == "__main__": main()
