"""Install (optionally) and audit the pinned CARE/BiCoord runtime.

No source checkout is edited and no package is upgraded opportunistically.
Set ``BICOORD_INSTALL=1`` (or pass ``--install``) to run the benchmark's
explicit pinned requirements file before the audit.  Missing dependencies and
an unusable simulator are fatal; this stage never emits a synthetic PASS.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

from .config import (
    ACTION_DIM,
    ACTION_HORIZON,
    D_MODEL,
    ENCODER_LAYERS,
    DECODER_LAYERS,
    HISTORY_LAYERS,
    HISTORY_STEPS,
    ROLE_RANK,
    ROLES,
    STATE_DIM,
)
from .stage_common import artifact, assert_common_paths, atomic_json, common_parser, publish_result


# Import names, not distribution names.  ``yaml`` comes from PyYAML and is
# used by the RoboTwin task configs; ``sapien`` is the simulator backend.
REQUIRED_IMPORTS: tuple[str, ...] = (
    "numpy",
    "torch",
    "torchvision",
    "transformers",
    "h5py",
    "cv2",
    "gymnasium",
    "yaml",
    "sapien",
    "warp",
)

# BiCoord/SAPIEN on the rented RTX 5090 image is only validated with these
# exact ABI-compatible builds.  A nominally newer NumPy or Warp is not an
# acceptable substitute for a formal run.
REQUIRED_DISTRIBUTION_VERSIONS: dict[str, str] = {
    "numpy": "1.26.4",
    "warp-lang": "1.4.0",
}


def _module_versions() -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "available"))
        except Exception as error:  # pragma: no cover - host dependent
            versions[name] = f"MISSING:{type(error).__name__}:{error}"
            missing.append(name)
    return versions, missing


def _distribution_contract() -> dict[str, str]:
    observed: dict[str, str] = {}
    for distribution, expected in REQUIRED_DISTRIBUTION_VERSIONS.items():
        try:
            value = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required distribution is missing: {distribution}") from error
        observed[distribution] = value
        if value != expected:
            raise RuntimeError(
                f"BiCoord runtime requires {distribution}=={expected}, observed {value}"
            )
    return observed


def _requirements_path(repo: Path, benchmark_repo: Path) -> Path:
    configured = os.environ.get("BICOORD_REQUIREMENTS")
    if configured:
        return Path(configured).expanduser().resolve()
    # Keep the benchmark's own pinned dependency list as the default.  CARE's
    # existing repository requirements are intentionally not replaced.
    candidate = benchmark_repo / "script" / "requirements.txt"
    if candidate.is_file():
        return candidate.resolve()
    candidate = repo / "requirements" / "r11" / "d-lawam.txt"
    return candidate.resolve()


def _install(requirements: Path) -> dict[str, Any]:
    if not requirements.is_file():
        raise FileNotFoundError(requirements)
    # Restrict installation to the two checked-out workspaces or /workspace;
    # an accidental path from a user shell cannot become a pip target.
    approved = (Path.cwd().resolve(), Path("/workspace").resolve())
    if not any(requirements == root or root in requirements.parents for root in approved):
        raise ValueError(f"requirements path is outside approved roots: {requirements}")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--requirement",
        str(requirements),
        *(f"{name}=={version}" for name, version in REQUIRED_DISTRIBUTION_VERSIONS.items()),
    ]
    try:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "")
        raise RuntimeError(f"pinned dependency installation failed: {detail[-3000:]}") from error
    return {
        "requirements": str(requirements),
        "command": command,
        "stdout_tail": completed.stdout[-2000:],
    }


def _torch_contract() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or int(torch.cuda.device_count()) != 4:
        raise RuntimeError(
            f"environment requires four CUDA devices, available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
        )
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": int(torch.cuda.device_count()),
        "device_names": [str(torch.cuda.get_device_name(i)) for i in range(4)],
        "model_contract": {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "horizon": ACTION_HORIZON,
            "history_steps": HISTORY_STEPS,
            "d_model": D_MODEL,
            "enc_layers": ENCODER_LAYERS,
            "dec_layers": DECODER_LAYERS,
            "roles": ROLES,
            "role_rank": ROLE_RANK,
            "history_layers": HISTORY_LAYERS,
        },
        "model_substitution": False,
        "model_dimension_override": False,
        "normalization_override": False,
    }


def _simulator_contract(benchmark_repo: Path) -> dict[str, Any]:
    required = {
        "envs": benchmark_repo / "envs",
        "task_config": benchmark_repo / "task_config",
        "assets": benchmark_repo / "assets",
        "asset_downloader": benchmark_repo / "assets" / "_download.py",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"BiCoord simulator checkout lacks {missing}")
    return {name: str(path.resolve()) for name, path in required.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args)
    install_requested = bool(
        getattr(args, "install", False) or os.environ.get("BICOORD_INSTALL") == "1"
    )
    installation: dict[str, Any] | None = None
    if install_requested:
        installation = _install(_requirements_path(args.repo, args.benchmark_repo))
    versions, missing = _module_versions()
    if missing:
        raise RuntimeError(f"required runtime modules are missing: {missing}")
    distributions = _distribution_contract()
    torch_contract = _torch_contract()
    simulator = _simulator_contract(args.benchmark_repo)
    report_value = {
        "schema": "before-we-act.bicoord-care-environment/1",
        "status": "PASSED",
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "modules": versions,
        "required_distribution_versions": dict(REQUIRED_DISTRIBUTION_VERSIONS),
        "distribution_versions": distributions,
        "torch_contract": torch_contract,
        "simulator": simulator,
        "installation": installation,
        "install_requested": install_requested,
        "source_tree_mutated": False,
        "destructive_instance_operations": False,
    }
    report = args.run / "artifacts" / "environment" / "environment.json"
    atomic_json(report, report_value)
    return publish_result(
        args,
        stage="environment",
        artifacts=[artifact(report, kind="environment")],
        environment=report_value,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = common_parser(__doc__, ("install-and-audit", "audit"))
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["REQUIRED_DISTRIBUTION_VERSIONS", "REQUIRED_IMPORTS", "main", "run"]
