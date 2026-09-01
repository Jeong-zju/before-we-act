"""Resumable four-GPU supervisor for the formal BiCoord CARE pipeline.

The supervisor is an orchestrator, not a replacement implementation.  Every
pipeline adapter must live below :mod:`deployment.bicoord_care`, emit the
frozen result contract defined here, and preserve the upstream CARE and
B-core/TUNE model.  Missing simulator or benchmark-specific adapters block the
run before any training starts.

Only child process groups created by this process are ever signalled.  There
is intentionally no host shutdown, cloud-instance stop/destroy, repository
reset, or service-control operation in this module.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

from .asset_contract import (
    CONTACT_KEY,
    DEFAULT_SHOVEL_SCALE,
    canonical_json_sha256,
    LEGACY_CONTACT_KEY,
    LEGACY_TRANSFORM_KEY,
    PRISTINE_SHOVEL_METADATA_SHA256,
    SHOVEL_CONTACT_POINTS_POSE_SHA256,
    SHOVEL_OVERLAY_METADATA_CANONICAL_SHA256,
    SHOVEL_COLLISION_BYTES,
    SHOVEL_COLLISION_SHA256,
    SHOVEL_METADATA_NAME,
    SHOVEL_MODEL_ID,
    SHOVEL_OBJECT_NAME,
    SHOVEL_VISUAL_BYTES,
    SHOVEL_VISUAL_SHA256,
)
from .config import (
    ACTION_DIM,
    ACTION_ENCODING,
    ACTION_HORIZON,
    DATASET_REPO_ID,
    DATASET_REVISION,
    EFFECTIVE_BATCH,
    EPISODES_PER_TASK,
    FORMAL_B0H_UPDATES,
    FORMAL_BCORE_UPDATES,
    FORMAL_CARE_UPDATES,
    FORMAL_SEEDS,
    HISTORY_STEPS,
    MODEL_CONTRACT as FROZEN_MODEL_CONTRACT,
    SMOKE_INTERFACE_STEPS,
    STATE_DIM,
    SOURCE_FREQUENCY_HZ as FROZEN_SOURCE_FREQUENCY_HZ,
    TASK_ASSET_DYNAMIC_INVENTORY_SHA256,
    TASK_ASSET_DYNAMIC_ITEM_COUNT,
    TASK_ASSET_UNRESOLVED_INTERACTION_COUNT,
    TASK_ASSET_UNRESOLVED_INTERACTION_INVENTORY_SHA256,
    TASKS,
    TOTAL_EPISODES,
    VALIDATION_EPISODES,
    VALIDATION_MAX_STEPS,
)


MAX_STEPS = VALIDATION_MAX_STEPS
SMOKE_MAX_STEPS = {task: SMOKE_INTERFACE_STEPS for task in TASKS}
FORMAL_DATASET_REPO = DATASET_REPO_ID
FORMAL_DATASET_REVISION = DATASET_REVISION
BICOORD_CODE_REVISION = "c4577b8808e45c15836945ee23f01f89c8a056c3"
FORMAL_EPISODES_PER_TASK = EPISODES_PER_TASK
FORMAL_EPISODES = TOTAL_EPISODES
SOURCE_FREQUENCY_HZ = FROZEN_SOURCE_FREQUENCY_HZ
GLOBAL_BATCH = EFFECTIVE_BATCH
B0H_UPDATES = FORMAL_B0H_UPDATES
BCORE_UPDATES = FORMAL_BCORE_UPDATES
CARE_UPDATES = FORMAL_CARE_UPDATES
BRANCH_FAMILIES_PER_TASK = 30
BRANCHES_PER_FAMILY = 24
BCORE_SEEDS = FORMAL_SEEDS
CARE_SEEDS = (20260904, 20260905, 20260906)
CARE_VARIANTS = ("care", "reactive_only", "replay_only", "capacity")
CARE_OOF_FOLDS = (0, 1, 2)
CARE_OOF_VARIANT = "care"
CARE_OOF_SEED = 20260904
CARE_OOF_TRAINING_SEED_OFFSET = 1000
GPU_IDS = (0, 1, 2, 3)
HEARTBEAT_INTERVAL_SECONDS = 15.0
CHILD_TERMINATION_GRACE_SECONDS = 10.0
CHILD_WAIT_POLL_SECONDS = 1.0

SUPERVISOR_SCHEMA = "before-we-act.bicoord-care-supervisor/1"
RESULT_SCHEMA = "before-we-act.bicoord-care-stage-result/1"
RECEIPT_SCHEMA = "before-we-act.bicoord-care-stage-receipt/1"

# This is an I/O adapter contract, not a smaller model.  Hidden widths and
# upstream CARE capacity remain identical to the official implementation.
MODEL_CONTRACT: dict[str, Any] = {
    **FROZEN_MODEL_CONTRACT.as_dict(),
    "benchmark_adapter": "BiCoord",
    "method_family": "CARE",
    "reference_policy": "B-core/TUNE",
    "reference_policy_family": "PredictiveTeamBeliefPolicy",
    "policy_family": "PredictiveTeamBeliefPolicy",
    "strictly_decentralized": True,
    "shared_checkpoint_for_both_arms": True,
    "global_view": "shared_head_camera",
    "local_view": "own_wrist_camera_only",
    "peer_qpos_action_wrist_allowed": False,
    "state_dim": STATE_DIM,
    "action_dim": ACTION_DIM,
    "action_encoding": ACTION_ENCODING,
    "state_source": "joint_action_drive_target",
    "temporal_alignment": "observation_row_t_to_action_row_t_plus_1",
    "training_pairs_per_episode": "source_length_minus_1",
    "source_frequency_hz": SOURCE_FREQUENCY_HZ,
    "history_steps": HISTORY_STEPS,
    "horizon": ACTION_HORIZON,
    "state_clipping": False,
    "action_clipping": False,
    "gripper_reparameterization": False,
    "normalization_population": "all_1800_demos_both_local_arms",
    "vision_width": 768,
}


# All production commands are benchmark-owned.  Environment overrides may
# change a module only within this prefix and are recorded in the run config.
DEFAULT_MODULES = {
    "source_preflight": "deployment.bicoord_care.preflight",
    "environment": "deployment.bicoord_care.environment",
    "dataset_download": "deployment.bicoord_care.download",
    "asset_contract": "deployment.bicoord_care.asset_stage",
    "dataset_audit": "deployment.bicoord_care.audit",
    "dino_cache": "deployment.bicoord_care.cache_dino",
    "b0h_train": "deployment.bicoord_care.train_b0h",
    "b0h_evaluate": "deployment.bicoord_care.evaluate_b0h",
    "bcore_cache": "deployment.bicoord_care.cache_bcore",
    "bcore_train": "deployment.bicoord_care.train_bcore",
    "bcore_select": "deployment.bicoord_care.select_bcore",
    "seed_discovery": "deployment.bicoord_care.seed_discovery",
    "bcore_evaluate": "deployment.bicoord_care.evaluate_bcore",
    "branch_collect": "deployment.bicoord_care.branch_collection",
    "branch_prepare": "deployment.bicoord_care.prepare_branches",
    "branch_audit": "deployment.bicoord_care.audit_branch_signal",
    "belief_train": "deployment.bicoord_care.train_belief",
    "care_select": "deployment.bicoord_care.select_calibrate",
    "paired_evaluate": "deployment.bicoord_care.paired_evaluate",
}


@dataclass(frozen=True)
class StageSpec:
    name: str
    dependencies: tuple[str, ...]
    module_key: str | None
    operation: str
    gpu_plan: str
    description: str
    result_kind: str = "generic"


STAGES: "OrderedDict[str, StageSpec]" = OrderedDict(
    (stage.name, stage)
    for stage in (
        StageSpec("source_preflight", (), "source_preflight", "source-preflight", "all", "pin CARE/BiCoord source and prove four RTX 5090 GPUs"),
        StageSpec("environment", ("source_preflight",), "environment", "install-and-audit", "all", "install the pinned runtime and simulator dependencies"),
        StageSpec("dataset_download", ("environment",), "dataset_download", "download", "cpu", "download the immutable 18-task Hugging Face snapshot"),
        StageSpec("asset_contract", ("dataset_download",), "asset_contract", "verify-and-overlay", "cpu", "bind both official object archives and build the plate and legacy-shovel contact runtime overlays", "asset"),
        StageSpec("dataset_audit", ("asset_contract",), "dataset_audit", "formal-audit", "cpu", "audit 1800 HDF5 demos, native ranges, alignment, instructions, and stages", "dataset"),
        StageSpec("dino_cache", ("dataset_audit",), "dino_cache", "cache-all", "sharded4", "cache frozen DINOv3 ViT-B/16 head and local-wrist features"),
        StageSpec("b0h_smoke_train", ("dino_cache",), "b0h_train", "smoke-train", "all", "five-update four-GPU B0-H training smoke", "training"),
        StageSpec("b0h_smoke_closed_loop", ("b0h_smoke_train",), "b0h_evaluate", "smoke-closed-loop", "task_queue4", "one closed-loop interface smoke per task", "smoke_validation"),
        # The complete downstream smoke chain deliberately precedes every
        # formal optimizer step.  Its B-core cache is a one-episode-per-task
        # interface fixture derived from the real smoke B0-H checkpoint; it
        # never becomes input to a formal stage.
        StageSpec("bcore_smoke_cache", ("b0h_smoke_closed_loop",), "bcore_cache", "cache-all", "sharded4", "cache one real episode per task for B-core smoke", "cache"),
        StageSpec("bcore_smoke_train", ("bcore_smoke_cache",), "bcore_train", "smoke-train", "all", "B-core/TUNE training smoke without model substitution", "training"),
        StageSpec("bcore_smoke_closed_loop", ("bcore_smoke_train",), "bcore_evaluate", "smoke-closed-loop", "task_queue4", "B-core/TUNE 18-task closed-loop smoke", "smoke_validation"),
        StageSpec("seed_discovery_smoke", ("bcore_smoke_closed_loop",), "seed_discovery", "smoke-discover", "seed_task_queue4", "one official expert-valid seed per task for paired smoke", "seed_manifest"),
        StageSpec("branch_smoke", ("bcore_smoke_closed_loop", "seed_discovery_smoke"), "branch_collect", "smoke", "sharded4", "same-snapshot physical branch collection smoke from B-core/TUNE", "branch"),
        StageSpec("branch_prepare_smoke", ("branch_smoke",), "branch_prepare", "prepare-smoke", "cpu", "pack only smoke physical families", "branch"),
        StageSpec("branch_signal_gate_smoke", ("branch_prepare_smoke",), "branch_audit", "signal-gate-smoke", "cpu", "reject degenerate smoke counterfactual/event supervision", "gate"),
        StageSpec("belief_smoke_train", ("branch_signal_gate_smoke",), "belief_train", "smoke-train", "all", "CARE belief-head training smoke", "training"),
        StageSpec("paired_validation_smoke", ("belief_smoke_train", "seed_discovery_smoke"), "paired_evaluate", "smoke-paired", "task_queue4", "paired selector-off/CARE 18-task smoke", "smoke_validation"),
        StageSpec("b0h_formal", ("paired_validation_smoke",), "b0h_train", "formal-train", "all", "formal B0-H initialization on every demonstration", "training"),
        StageSpec("b0h_probe", ("b0h_formal",), "b0h_evaluate", "probe", "task_queue4", "closed-loop B0-H reference probe before B-core", "validation"),
        StageSpec("bcore_cache", ("b0h_probe",), "bcore_cache", "cache-all", "sharded4", "build formal B-core/TUNE belief contexts from B0-H", "cache"),
        StageSpec("bcore_train_3seeds", ("bcore_cache",), "bcore_train", "formal-train", "seed_wave", "three independent formal B-core/TUNE seeds", "training_grid"),
        StageSpec("bcore_select", ("bcore_train_3seeds",), "bcore_select", "offline-select", "cpu", "select B-core/TUNE using offline metrics only", "selection"),
        StageSpec("seed_discovery", ("bcore_select",), "seed_discovery", "discover-seeds", "seed_task_queue4", "freeze official expert-valid validation seeds before learned rollout", "seed_manifest"),
        StageSpec("bcore_validation20", ("bcore_select", "seed_discovery"), "bcore_evaluate", "validation20", "task_queue4", "selected B-core/TUNE Validation20 reference", "validation20"),
        StageSpec("branch_collection", ("bcore_validation20", "seed_discovery"), "branch_collect", "formal", "sharded4", "collect formal counterfactual branch families on four GPUs", "branch"),
        StageSpec("branch_prepare", ("branch_collection",), "branch_prepare", "prepare", "cpu", "prepare belief-head data without train/test splitting", "branch"),
        StageSpec("branch_signal_gate", ("branch_prepare",), "branch_audit", "signal-gate", "cpu", "reject degenerate counterfactual/event supervision", "gate"),
        StageSpec("belief_train", ("branch_signal_gate",), "belief_train", "formal-train", "care_grid", "four CARE variants by three seeds in four-GPU waves", "training_grid"),
        StageSpec("offline_selection_calibration", ("belief_train",), "care_select", "select-calibrate", "cpu", "offline selection and calibration with no closed-loop leakage", "selection"),
        StageSpec("paired_validation20", ("offline_selection_calibration",), "paired_evaluate", "validation20-paired", "task_queue4", "paired selector-off/CARE Validation20 on all 18 tasks", "validation20"),
    )
)

STAGE_DEPENDENCIES = OrderedDict(
    (name, spec.dependencies) for name, spec in STAGES.items()
)


class SupervisorError(RuntimeError):
    pass


class Blocked(SupervisorError):
    pass


class InvalidArtifact(SupervisorError):
    pass


class Interrupted(SupervisorError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_stage_path(
    value: object, expected: Path, *, label: str
) -> Path:
    """Return an exact, non-symbolic path for a released stage artifact.

    Stage results are child-process input and therefore must not be allowed to
    redirect the runtime to an out-of-run (or symlinked) receipt.  Comparing
    the lexical absolute spelling first deliberately rejects aliases such as
    ``..`` and symlink paths; every existing component is then checked before
    the final file is opened.
    """

    if not isinstance(value, str) or not value:
        raise InvalidArtifact(f"{label} path is missing")
    observed = Path(value).expanduser()
    expected_absolute = expected.expanduser().absolute()
    if not observed.is_absolute() or observed != expected_absolute:
        raise InvalidArtifact(
            f"{label} path is not the canonical run artifact: "
            f"{observed} != {expected_absolute}"
        )
    cursor = observed
    while True:
        if cursor.is_symlink():
            raise InvalidArtifact(f"{label} path contains a symbolic component: {cursor}")
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    try:
        resolved = observed.resolve(strict=False)
    except OSError as error:
        raise InvalidArtifact(f"{label} path cannot be resolved: {observed}") from error
    if resolved != expected_absolute:
        raise InvalidArtifact(
            f"{label} path resolves outside the canonical run artifact: {observed}"
        )
    return observed


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise InvalidArtifact(f"missing JSON artifact: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidArtifact(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise InvalidArtifact(f"expected JSON object: {path}")
    return value


def _validate_paired_progress(
    path: Path,
    *,
    mode: str,
    task: str,
    max_steps: int,
    seed_steps: Sequence[tuple[int, int]],
    context: str,
) -> None:
    """Validate a paired progress stream using seed-local step counters.

    Per-seed progress files start at step one.  The combined task stream is a
    concatenation of those files, so its counter restarts at every seed
    boundary rather than increasing across episodes.  ``seed_steps`` binds the
    boundary order and exact rollout length to the paired receipt.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise InvalidArtifact(f"{context}: progress cannot be read") from error
    if not lines:
        raise InvalidArtifact(f"{context}: progress is empty")
    if not seed_steps or len({seed for seed, _steps in seed_steps}) != len(seed_steps):
        raise InvalidArtifact(f"{context}: expected seed boundaries are invalid")

    expected_rows = sum(steps for _seed, steps in seed_steps)
    if any(steps < 1 or steps > max_steps for _seed, steps in seed_steps):
        raise InvalidArtifact(f"{context}: expected rollout length is invalid")
    if len(lines) != expected_rows:
        raise InvalidArtifact(f"{context}: progress row count differs")

    line_index = 0
    for seed, steps in seed_steps:
        for expected_step in range(1, steps + 1):
            line = lines[line_index]
            line_index += 1
            try:
                row = json.loads(line)
            except (TypeError, json.JSONDecodeError) as error:
                raise InvalidArtifact(f"{context}: progress has invalid JSON") from error
            if not isinstance(row, Mapping):
                raise InvalidArtifact(f"{context}: progress row is not an object")
            expected = {
                "task": task,
                "seed": seed,
                "mode": mode,
                "max_steps": max_steps,
                "action_clipped": False,
            }
            for key, value in expected.items():
                if row.get(key) != value:
                    raise InvalidArtifact(
                        f"{context} progress row differs at {key}: "
                        f"{row.get(key)!r} != {value!r}"
                    )
            if row.get("step") != expected_step:
                raise InvalidArtifact(
                    f"{context}: progress seed/step sequence differs"
                )


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def _git_revision(path: Path) -> str:
    """Read a checkout's HEAD without mutating it."""

    try:
        value = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--verify", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise Blocked(f"cannot inspect git revision: {path}") from error
    if len(value) != 40:
        raise Blocked(f"invalid git revision from {path}: {value!r}")
    try:
        int(value, 16)
    except ValueError as error:
        raise Blocked(f"git revision from {path} is not hexadecimal: {value!r}") from error
    return value


def _dino_preflight(path: Path) -> dict[str, Any]:
    """Check the frozen local DINOv3 artifact without loading model weights."""

    if not path.is_dir():
        raise Blocked(f"DINO model directory is absent: {path}")
    required = ("config.json", "preprocessor_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    weights = sorted(
        item
        for item in path.iterdir()
        if item.is_file()
        and (
            item.name in {"model.safetensors", "pytorch_model.bin"}
            or item.name.startswith("model-")
            or item.name.startswith("pytorch_model-")
        )
    )
    if missing or not weights:
        raise Blocked(
            f"DINO model artifact is incomplete: missing={missing}, weights={len(weights)}"
        )
    # Use the same Transformers interpretation as the real cache adapter.
    # This accepts metadata defaults supplied by the official class while
    # remaining local/read-only and never allocating the weight tensors.
    try:
        from types import SimpleNamespace

        from transformers import AutoConfig, AutoImageProcessor

        from .preprocessing import (
            validate_dino_model_contract,
            validate_dino_processor_contract,
        )

        config = AutoConfig.from_pretrained(str(path), local_files_only=True)
        processor = AutoImageProcessor.from_pretrained(
            str(path), local_files_only=True
        )
        model_contract = validate_dino_model_contract(
            SimpleNamespace(config=config)
        )
        processor_contract = validate_dino_processor_contract(processor)
    except Exception as error:
        raise Blocked(f"DINO model/processor contract is invalid: {path}") from error
    return {
        "path": str(path.resolve()),
        "weights": [
            {"name": item.name, "size": item.stat().st_size} for item in weights
        ],
        "model_contract": model_contract,
        "processor_contract": processor_contract,
    }


def _gpu_preflight() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise Blocked("nvidia-smi hardware probe failed") from error
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    parsed = [[item.strip() for item in row.split(",", 4)] for row in rows]
    if len(parsed) != 4 or any(len(row) != 5 for row in parsed):
        raise Blocked(f"formal run requires exactly four physical GPUs, got {len(parsed)}")
    names = [row[1] for row in parsed]
    if not all("5090" in name for name in names):
        raise Blocked(f"formal run requires RTX 5090 devices, observed {names!r}")
    return {
        "device_count": 4,
        "device_names": names,
        "devices": [
            {
                "index": int(row[0]),
                "name": row[1],
                "uuid": row[2],
                "memory_total_mib": int(row[3]),
                "driver_version": row[4],
            }
            for row in parsed
        ],
    }


def _assert_dag() -> None:
    seen: set[str] = set()
    for name, spec in STAGES.items():
        unknown = set(spec.dependencies) - seen
        if unknown:
            raise RuntimeError(f"stage {name} has forward/unknown dependencies: {unknown}")
        seen.add(name)


_assert_dag()


@dataclass(frozen=True)
class Settings:
    repo: Path
    benchmark_repo: Path
    dataset: Path
    run: Path
    dino_model: Path
    python: str
    # The formal run must always bind the exact CARE checkout.  A default is
    # kept solely so dataclass construction remains ergonomic in small unit
    # fixtures; ``validate`` intentionally rejects the empty value.
    care_source_revision: str = ""
    modules: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_MODULES))
    b0h_updates: int = B0H_UPDATES
    bcore_updates: int = BCORE_UPDATES
    care_updates: int = CARE_UPDATES
    families_per_task: int = BRANCH_FAMILIES_PER_TASK

    @classmethod
    def from_environment(cls) -> "Settings":
        repo = Path(
            os.environ.get("BICOORD_CARE_REPO", "/workspace/repos/before-we-act")
        ).resolve()
        modules = {
            key: os.environ.get(f"BICOORD_MODULE_{key.upper()}", value)
            for key, value in DEFAULT_MODULES.items()
        }
        return cls(
            repo=repo,
            benchmark_repo=Path(
                os.environ.get(
                    "BICOORD_BENCH_REPO", "/workspace/repos/bicoord-bench"
                )
            ).resolve(),
            dataset=Path(
                os.environ.get(
                    "BICOORD_DATASET", "/workspace/repos/bicoord-bench/data"
                )
            ).resolve(),
            run=Path(
                os.environ.get("BICOORD_CARE_RUN", "/workspace/runs/bicoord-care-v3")
            ).resolve(),
            dino_model=Path(
                os.environ.get(
                    "BICOORD_DINO_MODEL",
                    "/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m",
                )
            ).resolve(),
            python=os.environ.get("BICOORD_CARE_PYTHON", sys.executable),
            care_source_revision=os.environ.get(
                "BICOORD_CARE_SOURCE_REVISION", ""
            ),
            modules=modules,
            b0h_updates=int(os.environ.get("BICOORD_B0H_UPDATES", B0H_UPDATES)),
            bcore_updates=int(
                os.environ.get("BICOORD_BCORE_UPDATES", BCORE_UPDATES)
            ),
            care_updates=int(os.environ.get("BICOORD_CARE_UPDATES", CARE_UPDATES)),
            families_per_task=int(
                os.environ.get(
                    "BICOORD_FAMILIES_PER_TASK", BRANCH_FAMILIES_PER_TASK
                )
            ),
        )

    def validate(self) -> None:
        if len(self.care_source_revision) != 40:
            raise ValueError(
                "BICOORD_CARE_SOURCE_REVISION must be a pinned 40-character Git commit"
            )
        try:
            int(self.care_source_revision, 16)
        except ValueError as error:
            raise ValueError(
                "BICOORD_CARE_SOURCE_REVISION must contain only hexadecimal characters"
            ) from error
        frozen_budget = {
            "b0h_updates": B0H_UPDATES,
            "bcore_updates": BCORE_UPDATES,
            "care_updates": CARE_UPDATES,
            "families_per_task": BRANCH_FAMILIES_PER_TASK,
        }
        for name, expected in frozen_budget.items():
            observed = int(getattr(self, name))
            if observed != expected:
                raise ValueError(
                    f"formal BiCoord protocol freezes {name}={expected}, "
                    f"observed {observed}"
                )
        for key, module in self.modules.items():
            if key not in DEFAULT_MODULES:
                raise ValueError(f"unknown module role: {key}")
            if not module.startswith("deployment.bicoord_care."):
                raise ValueError(
                    f"{key} module is outside the formal BiCoord adapter: {module}"
                )

    def frozen_config(self) -> dict[str, Any]:
        return {
            "schema": SUPERVISOR_SCHEMA,
            "tasks": list(TASKS),
            "max_steps": dict(MAX_STEPS),
            "dataset_repo": FORMAL_DATASET_REPO,
            "dataset_revision": FORMAL_DATASET_REVISION,
            "bicoord_code_revision": BICOORD_CODE_REVISION,
            "care_source_revision": self.care_source_revision,
            "episodes_per_task": FORMAL_EPISODES_PER_TASK,
            "validation_episodes": VALIDATION_EPISODES,
            "smoke_interface_steps": SMOKE_INTERFACE_STEPS,
            "global_batch": GLOBAL_BATCH,
            "b0h_updates": self.b0h_updates,
            "bcore_updates": self.bcore_updates,
            "care_updates": self.care_updates,
            "families_per_task": self.families_per_task,
            "branches_per_family": BRANCHES_PER_FAMILY,
            "bcore_seeds": list(BCORE_SEEDS),
            "care_seeds": list(CARE_SEEDS),
            "care_variants": list(CARE_VARIANTS),
            "care_oof": {
                "variant": CARE_OOF_VARIANT,
                "public_seed": CARE_OOF_SEED,
                "folds": list(CARE_OOF_FOLDS),
                "training_seeds": [
                    CARE_OOF_SEED + CARE_OOF_TRAINING_SEED_OFFSET + fold
                    for fold in CARE_OOF_FOLDS
                ],
                "deployment_candidate": False,
            },
            "model_contract": MODEL_CONTRACT,
            "asset_contract": {
                "small_object": "003_plate",
                "contact_donor": "003_plate_large",
                "copied_fields": ["contact_points_pose"],
                "legacy_shovel": {
                    "task": "sweep_block",
                    "object": SHOVEL_OBJECT_NAME,
                    "model_id": SHOVEL_MODEL_ID,
                    "metadata": SHOVEL_METADATA_NAME,
                    "pristine_metadata_sha256": (
                        PRISTINE_SHOVEL_METADATA_SHA256
                    ),
                    "legacy_fields": [LEGACY_CONTACT_KEY, LEGACY_TRANSFORM_KEY],
                    "derived_fields": [CONTACT_KEY],
                    "derived_contact_points_pose_sha256": (
                        SHOVEL_CONTACT_POINTS_POSE_SHA256
                    ),
                    "conversion": (
                        "scale(contact_pose) @ trans_matrix -> "
                        "scale(contact_points_pose)"
                    ),
                    "scale": list(DEFAULT_SHOVEL_SCALE),
                    "collision_mesh": {
                        "relative_path": "collision/base3.glb",
                        "bytes": SHOVEL_COLLISION_BYTES,
                        "sha256": SHOVEL_COLLISION_SHA256,
                    },
                    "visual_mesh": {
                        "relative_path": "visual/base3.glb",
                        "bytes": SHOVEL_VISUAL_BYTES,
                        "sha256": SHOVEL_VISUAL_SHA256,
                    },
                    "model_variant_replaced": False,
                },
                "runtime_scope": "actor_config_in_memory_only",
                "supplemental_assets_installed": True,
                "task_source_modified": False,
                "benchmark_tracked_source_modified": False,
                "benchmark_asset_source_modified": False,
            },
            "modules": dict(self.modules),
            "paths": {
                "repo": str(self.repo),
                "benchmark_repo": str(self.benchmark_repo),
                "dataset": str(self.dataset),
                "run": str(self.run),
                "dino_model": str(self.dino_model),
                "python": self.python,
            },
            "gpu_scheduler": {
                "physical_gpus": [0, 1, 2, 3],
                "ddp": "B0-H/B-core/CARE smoke and B0-H formal claim all four",
                "seed_wave": "one B-core seed per GPU; GPU3 remains available",
                "care_grid": (
                    "12 deployment-main jobs plus three independent OOF-shadow "
                    "jobs, at most four jobs per wave"
                ),
                "task_queue": "18 deterministic task jobs dynamically consume four GPUs",
                "seed_task_queue": (
                    "18 expert-planner jobs dynamically consume four GPUs and "
                    "publish one immutable aggregate seed manifest"
                ),
                "sharded": "four rank-sharded cache/branch workers",
            },
        }


@dataclass
class ActiveProcess:
    name: str
    process: subprocess.Popen
    gpus: tuple[int, ...]
    log_path: Path
    started_at: float


class GpuScheduler:
    """Exclusive physical-GPU leases for supervisor-owned subprocesses."""

    def __init__(self, settings: Settings, status_callback: Callable[[], None]):
        self.settings = settings
        self._status_callback = status_callback
        self._lock = threading.RLock()
        self.active: dict[int, ActiveProcess] = {}
        self._interrupted = threading.Event()

    def environment(self, gpus: Sequence[int]) -> dict[str, str]:
        environment = os.environ.copy()
        pythonpath = [str(self.settings.repo), str(self.settings.benchmark_repo)]
        if environment.get("PYTHONPATH"):
            pythonpath.append(environment["PYTHONPATH"])
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join(pythonpath),
                "CUDA_VISIBLE_DEVICES": ",".join(str(value) for value in gpus),
                "MUJOCO_GL": environment.get("MUJOCO_GL", "egl"),
                "HF_HOME": environment.get("HF_HOME", "/workspace/.hf_home"),
                "WANDB_MODE": environment.get("WANDB_MODE", "disabled"),
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": environment.get("OMP_NUM_THREADS", "8"),
                "MKL_NUM_THREADS": environment.get("MKL_NUM_THREADS", "8"),
                # Bind children to the same checkout identity that is part of
                # the frozen supervisor config, even when the parent was
                # instantiated directly rather than through the environment.
                "BICOORD_CARE_SOURCE_REVISION": self.settings.care_source_revision,
                # Both compatibility files are hashed outputs of the
                # ``asset_contract`` DAG stage.  Simulator adapters apply each
                # one only to the corresponding official actor config in
                # memory.
                "BICOORD_PLATE_ASSET_OVERLAY": str(
                    self.settings.run
                    / "artifacts"
                    / "asset_contract"
                    / "overlay"
                    / "003_plate"
                    / "model_data0.json"
                ),
                "BICOORD_SHOVEL_ASSET_OVERLAY": str(
                    self.settings.run
                    / "artifacts"
                    / "asset_contract"
                    / "overlay"
                    / SHOVEL_OBJECT_NAME
                    / SHOVEL_METADATA_NAME
                ),
                "BICOORD_REQUIRE_ASSET_OVERLAY": "1",
            }
        )
        return environment

    def _spawn(
        self,
        name: str,
        command: Sequence[str],
        gpus: Sequence[int],
        log_path: Path,
    ) -> ActiveProcess:
        # A service stop is terminal for this supervisor process.  In
        # particular, do not let a stage-level retry launch a fresh worker
        # after SIGTERM has already interrupted the previous wave.
        if self._interrupted.is_set():
            raise Interrupted("supervisor interrupted")
        gpu_tuple = tuple(int(value) for value in gpus)
        if any(value not in GPU_IDS for value in gpu_tuple):
            raise ValueError(f"invalid GPU lease for {name}: {gpu_tuple}")
        with self._lock:
            # ``interrupt`` sets the event before acquiring this lock.  This
            # second check closes the race between the cheap check above and
            # entering the spawn critical section.
            if self._interrupted.is_set():
                raise Interrupted("supervisor interrupted")
            occupied = {gpu for active in self.active.values() for gpu in active.gpus}
            overlap = occupied.intersection(gpu_tuple)
            if overlap:
                raise RuntimeError(f"GPU lease collision for {name}: {sorted(overlap)}")
            child_environment = self.environment(gpu_tuple)
            # A same-thread signal handler can run while building the child
            # environment even though this is an RLock.  Recheck before any
            # log or process is created.
            if self._interrupted.is_set():
                raise Interrupted("supervisor interrupted")
            stream = None
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                stream = log_path.open("a", buffering=1)
                stream.write(
                    json.dumps(
                        {
                            "event": "spawn",
                            "at": _utc_now(),
                            "name": name,
                            "gpus": list(gpu_tuple),
                            # Credentials never enter command arguments.
                            "command": list(command),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                # This check handles an interrupt delivered while preparing
                # the append-only spawn record.  If a signal lands in the
                # irreducible Popen window, the post-registration check below
                # terminates and reaps the newly owned group before returning.
                if self._interrupted.is_set():
                    raise Interrupted("supervisor interrupted")
                process = subprocess.Popen(
                    list(command),
                    # RoboTwin/BiCoord task code resolves assets and controller
                    # metadata relative to its benchmark checkout (not the CARE
                    # adapter checkout).  Keep both trees on PYTHONPATH, but run
                    # every child from the benchmark root so simulator imports
                    # and relative asset paths are identical to the official
                    # evaluator.
                    cwd=self.settings.benchmark_repo,
                    env=child_environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                if stream is not None:
                    stream.close()
            active = ActiveProcess(name, process, gpu_tuple, log_path, time.time())
            self.active[process.pid] = active
        # ``interrupt`` may have run synchronously inside Popen, before the
        # child could be entered in ``active``.  Registration first gives the
        # group an ownership identity; this check then closes that window.
        if self._interrupted.is_set():
            cleanup_errors = self._terminate_and_reap((active,))
            detail = self._format_cleanup_errors(cleanup_errors)
            raise Interrupted(f"supervisor interrupted{detail}")
        try:
            self._status_callback()
        except BaseException as error:
            cleanup_errors = self._terminate_and_reap((active,))
            if cleanup_errors:
                raise SupervisorError(
                    f"failed to publish {name} spawn status"
                    f"{self._format_cleanup_errors(cleanup_errors)}"
                ) from error
            raise
        return active

    def _remove_owned(self, active: ActiveProcess) -> bool:
        """Remove exactly the process object registered by this scheduler."""

        with self._lock:
            if self.active.get(active.process.pid) is not active:
                return False
            self.active.pop(active.process.pid)
            return True

    def _signal_owned_process_group(
        self, active: ActiveProcess, signum: int
    ) -> None:
        """Signal a group only while its exact Popen remains registered."""

        with self._lock:
            if self.active.get(active.process.pid) is not active:
                return
            # start_new_session=True makes the direct child's PID the PGID.
            # poll() prevents signalling a recycled PID after the child was
            # already reaped by another path.  The getattr fallback keeps the
            # ownership guard usable with narrow Popen test doubles.
            poll = getattr(active.process, "poll", None)
            if poll is not None:
                try:
                    if poll() is not None:
                        return
                except ChildProcessError:
                    return
            try:
                os.killpg(active.process.pid, signum)
            except ProcessLookupError:
                pass

    @staticmethod
    def _wait_process(process: subprocess.Popen, timeout: float | None = None) -> int:
        """Wait with a timeout while tolerating minimal Popen test doubles."""

        if timeout is None:
            return process.wait()
        try:
            return process.wait(timeout=timeout)
        except TypeError as error:
            # Some contract tests use a tiny ``wait()`` double without the
            # optional Popen timeout parameter.  Retry only when the fallback
            # itself accepts no timeout; preserve unrelated TypeErrors.
            try:
                return process.wait()
            except TypeError:
                raise error

    @staticmethod
    def _format_cleanup_errors(errors: Sequence[BaseException]) -> str:
        if not errors:
            return ""
        return "; cleanup errors: " + "; ".join(repr(error) for error in errors)

    def _terminate_and_reap(
        self, processes: Sequence[ActiveProcess]
    ) -> list[BaseException]:
        """Terminate, wait for, and unregister a set of owned child groups."""

        errors: list[BaseException] = []
        for active in processes:
            try:
                self._signal_owned_process_group(active, signal.SIGTERM)
            except BaseException as error:
                errors.append(error)
        registry_changed = False
        for active in processes:
            try:
                self._wait_process(
                    active.process, timeout=CHILD_TERMINATION_GRACE_SECONDS
                )
            except subprocess.TimeoutExpired:
                try:
                    self._signal_owned_process_group(active, signal.SIGKILL)
                    self._wait_process(active.process)
                except BaseException as error:
                    errors.append(error)
            except BaseException as error:
                errors.append(error)
            finally:
                registry_changed = self._remove_owned(active) or registry_changed
        if registry_changed:
            try:
                self._status_callback()
            except BaseException as error:
                errors.append(error)
        return errors

    def _wait(self, active: ActiveProcess) -> None:
        try:
            while True:
                try:
                    code = self._wait_process(
                        active.process, timeout=CHILD_WAIT_POLL_SECONDS
                    )
                    break
                except subprocess.TimeoutExpired:
                    if not self._interrupted.is_set():
                        continue
                    cleanup_errors = self._terminate_and_reap((active,))
                    raise Interrupted(
                        "supervisor interrupted"
                        f"{self._format_cleanup_errors(cleanup_errors)}"
                    )
        finally:
            removed = self._remove_owned(active)
            if removed:
                self._status_callback()
        if self._interrupted.is_set():
            raise Interrupted("supervisor interrupted")
        if code != 0:
            raise SupervisorError(
                f"{active.name} exited {code}; see {active.log_path}"
            )

    def run(
        self,
        name: str,
        command: Sequence[str],
        gpus: Sequence[int],
        log_path: Path,
    ) -> None:
        self._wait(self._spawn(name, command, gpus, log_path))

    def run_wave(
        self,
        jobs: Sequence[tuple[str, Sequence[str], int, Path]],
    ) -> None:
        if len(jobs) > 4 or len({gpu for _, _, gpu, _ in jobs}) != len(jobs):
            raise ValueError("a GPU wave must contain at most one job per GPU")
        active: list[ActiveProcess] = []
        try:
            for name, command, gpu, log_path in jobs:
                active.append(self._spawn(name, command, (gpu,), log_path))
        except BaseException as error:
            # A later Popen can fail, or SIGTERM can land while it is being
            # created.  Either way, a partially launched wave is not allowed
            # to leave earlier workers running or registered.
            cleanup_errors = self._terminate_and_reap(active)
            if isinstance(error, Interrupted):
                raise Interrupted(
                    f"{error}{self._format_cleanup_errors(cleanup_errors)}"
                ) from error
            if cleanup_errors:
                raise SupervisorError(
                    f"wave spawn failed: {error!r}"
                    f"{self._format_cleanup_errors(cleanup_errors)}"
                ) from error
            raise
        errors: list[BaseException] = []
        for item in active:
            try:
                self._wait(item)
            except BaseException as error:  # finish accounting for every child
                errors.append(error)
        if errors:
            # Preserve the terminal interruption type.  Wrapping it in the
            # generic SupervisorError makes ``run_stage`` treat a deliberate
            # service stop as a retryable worker failure and can start a new
            # GPU wave while supervisord is waiting for shutdown.
            interrupted = [error for error in errors if isinstance(error, Interrupted)]
            if interrupted:
                raise Interrupted("; ".join(str(error) for error in interrupted))
            raise SupervisorError("; ".join(str(error) for error in errors))

    def interrupt(self) -> None:
        """Terminate only child process groups launched by this scheduler."""

        self._interrupted.set()
        with self._lock:
            active = list(self.active.values())
        for item in active:
            self._signal_owned_process_group(item, signal.SIGTERM)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": item.name,
                    "pid": item.process.pid,
                    "gpus": list(item.gpus),
                    "log": str(item.log_path),
                    "elapsed_seconds": round(time.time() - item.started_at, 3),
                }
                for item in self.active.values()
            ]


class Supervisor:
    def __init__(self, settings: Settings):
        settings.validate()
        self.s = settings
        self.config = settings.frozen_config()
        self.config_hash = _canonical_hash(self.config)
        self.current_stage: str | None = None
        self.state = "initializing"
        self.detail: dict[str, Any] = {}
        self.scheduler = GpuScheduler(settings, self._write_status)
        self._old_handlers: dict[int, Any] = {}
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    @property
    def receipts(self) -> Path:
        return self.s.run / "receipts"

    @property
    def results(self) -> Path:
        return self.s.run / "stage_results"

    def receipt_path(self, stage: str) -> Path:
        index = tuple(STAGES).index(stage) + 1
        return self.receipts / f"{index:02d}-{stage}.json"

    def result_path(self, stage: str) -> Path:
        return self.results / f"{stage}.json"

    def _write_status(self) -> None:
        if not self.s.run.exists() and self.state == "initializing":
            return
        value = {
            "schema": SUPERVISOR_SCHEMA,
            "state": self.state,
            "stage": self.current_stage,
            "updated_at": _utc_now(),
            "pid": os.getpid(),
            "config_sha256": self.config_hash,
            "active_children": self.scheduler.snapshot(),
            **self.detail,
        }
        _atomic_json(self.s.run / "status.json", value)
        # Keep a separate liveness receipt so an external monitor can
        # distinguish a healthy long-running child from a stale status file.
        _atomic_json(
            self.s.run / "heartbeat.json",
            {
                "schema": "before-we-act.bicoord-care-heartbeat/1",
                "pid": os.getpid(),
                "stage": self.current_stage,
                "state": self.state,
                "updated_at": _utc_now(),
                "active_children": self.scheduler.snapshot(),
                "config_sha256": self.config_hash,
            },
        )

    def _heartbeat_loop(self) -> None:
        interval = float(
            os.environ.get(
                "BICOORD_HEARTBEAT_SECONDS", HEARTBEAT_INTERVAL_SECONDS
            )
        )
        if not interval > 0:
            interval = HEARTBEAT_INTERVAL_SECONDS
        while not self._heartbeat_stop.wait(interval):
            try:
                self._write_status()
            except OSError:
                # A transient filesystem error must not kill a running child;
                # the next tick will retry and the stage itself remains
                # fail-closed when its final receipt is checked.
                continue

    def _start_heartbeat(self) -> None:
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="bicoord-care-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._heartbeat_thread = None

    def _set_status(self, state: str, stage: str | None, **detail: Any) -> None:
        self.state = state
        self.current_stage = stage
        self.detail = detail
        self._write_status()

    def _install_signals(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            # Set the terminal spawn barrier before status I/O.  A slow or
            # failing filesystem must never leave a window for another child.
            self.scheduler.interrupt()
            self._set_status("interrupting", self.current_stage, signal=signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            self._old_handlers[signum] = signal.signal(signum, handler)

    def _restore_signals(self) -> None:
        for signum, previous in self._old_handlers.items():
            signal.signal(signum, previous)

    def preflight(self) -> dict[str, Any]:
        missing_paths = [
            str(path)
            for path in (self.s.repo, self.s.benchmark_repo)
            if not path.is_dir()
        ]
        missing_modules = sorted(
            {
                module
                for module in self.s.modules.values()
                if not _module_available(module)
            }
        )
        if not Path(self.s.python).is_file() and shutil.which(self.s.python) is None:
            missing_paths.append(self.s.python)
        checks: dict[str, Any] = {}
        failures: list[str] = []
        if not missing_paths and not missing_modules:
            try:
                care_revision = _git_revision(self.s.repo)
                benchmark_revision = _git_revision(self.s.benchmark_repo)
                if care_revision != self.s.care_source_revision:
                    raise Blocked(
                        f"CARE source revision drift: {care_revision} != "
                        f"{self.s.care_source_revision}"
                    )
                if benchmark_revision != BICOORD_CODE_REVISION:
                    raise Blocked(
                        f"BiCoord benchmark revision drift: {benchmark_revision} != "
                        f"{BICOORD_CODE_REVISION}"
                    )
                checks["source"] = {
                    "care_revision": care_revision,
                    "benchmark_revision": benchmark_revision,
                    "care_source_revision": self.s.care_source_revision,
                    "bicoord_code_revision": BICOORD_CODE_REVISION,
                }
            except Blocked as error:
                failures.append(str(error))
            try:
                checks["gpu"] = _gpu_preflight()
            except Blocked as error:
                failures.append(str(error))
            try:
                checks["dino"] = _dino_preflight(self.s.dino_model)
            except Blocked as error:
                failures.append(str(error))
            # Import the simulator in the benchmark working directory.  This
            # catches ABI/import failures (SAPIEN/Warp/Curobo) before any
            # stage can acquire a lease or write a training artifact.
            try:
                completed = subprocess.run(
                    [
                        self.s.python,
                        "-c",
                        "import sapien, warp; print('sapien/warp import ok')",
                    ],
                    cwd=self.s.benchmark_repo,
                    env=self.scheduler.environment(()),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                checks["simulator_import"] = {
                    "status": "PASSED",
                    "stdout": completed.stdout.strip()[-500:],
                }
            except (OSError, subprocess.SubprocessError) as error:
                failures.append(f"SAPIEN/Warp import preflight failed: {error}")
        report = {
            "status": (
                "PASSED"
                if not missing_paths and not missing_modules and not failures
                else "BLOCKED"
            ),
            "missing_paths": missing_paths,
            "missing_modules": missing_modules,
            "failures": failures,
            "checks": checks,
            "required_modules": dict(self.s.modules),
            "config_sha256": self.config_hash,
            "hf_token_source": "environment_only",
            "destructive_instance_operations": False,
        }
        if report["status"] != "PASSED":
            raise Blocked(
                "formal BiCoord pipeline adapters/environment are incomplete: "
                f"paths={missing_paths}, modules={missing_modules}, failures={failures}"
            )
        return report

    def _dependency_hashes(self, stage: StageSpec) -> dict[str, str]:
        values: dict[str, str] = {}
        for dependency in stage.dependencies:
            path = self.receipt_path(dependency)
            self._validate_receipt(dependency)
            values[dependency] = _sha256(path)
        return values

    def _validate_artifacts(self, result: Mapping[str, Any]) -> None:
        artifacts = result.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise InvalidArtifact("stage result artifacts must be a list")
        for row in artifacts:
            if not isinstance(row, Mapping):
                raise InvalidArtifact("stage artifact entry must be an object")
            path = Path(str(row.get("path", "")))
            if not path.is_absolute():
                path = self.s.run / path
            if not path.is_file() or path.stat().st_size <= 0:
                raise InvalidArtifact(f"missing/empty stage artifact: {path}")
            expected = row.get("sha256")
            if not isinstance(expected, str) or _sha256(path) != expected:
                raise InvalidArtifact(f"stage artifact hash differs: {path}")

    def _artifact_path(self, row: Mapping[str, Any]) -> Path:
        path = Path(str(row.get("path", "")))
        return path if path.is_absolute() else self.s.run / path

    def _validated_worker_result(
        self, spec: StageSpec, path: Path
    ) -> dict[str, Any]:
        row = _read_json(path)
        expected = {
            "schema": RESULT_SCHEMA,
            "stage": spec.name,
            "status": "PASSED",
            "benchmark_adapter": "BiCoord",
            "config_sha256": self.config_hash,
        }
        self._require_mapping_values(row, expected, f"{spec.name} worker")
        self._validate_model_contract(row)
        self._validate_artifacts(row)
        return row

    @staticmethod
    def _deduplicate_artifacts(
        rows: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise InvalidArtifact("worker artifact entry must be an object")
            key = (str(row.get("path", "")), str(row.get("sha256", "")))
            if key in seen:
                continue
            seen.add(key)
            result.append(dict(row))
        return result

    @staticmethod
    def _validated_seed_exception_diagnostics(
        type_counts_value: Any,
        signature_counts_value: Any,
        *,
        context: str,
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        """Validate one task's exception-count pair without coercing types."""

        if not isinstance(type_counts_value, Mapping):
            raise InvalidArtifact(f"{context}: exception type counts are not an object")
        if not isinstance(signature_counts_value, list):
            raise InvalidArtifact(f"{context}: exception signature counts are not a list")
        type_counts: dict[str, int] = {}
        for error_type, count in type_counts_value.items():
            if (
                not isinstance(error_type, str)
                or not error_type
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
            ):
                raise InvalidArtifact(f"{context}: malformed exception type count")
            type_counts[error_type] = count

        rows: list[dict[str, Any]] = []
        derived_type_counts: Counter[str] = Counter()
        identities: set[tuple[str, str]] = set()
        for row in signature_counts_value:
            if not isinstance(row, Mapping) or set(row) != {
                "error_type",
                "error_signature",
                "count",
            }:
                raise InvalidArtifact(f"{context}: malformed exception signature row")
            error_type = row.get("error_type")
            signature = row.get("error_signature")
            count = row.get("count")
            if (
                not isinstance(error_type, str)
                or not error_type
                or not isinstance(signature, str)
                or len(signature) != 64
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
            ):
                raise InvalidArtifact(f"{context}: malformed exception signature count")
            try:
                int(signature, 16)
            except ValueError as error:
                raise InvalidArtifact(
                    f"{context}: exception signature is not hexadecimal"
                ) from error
            identity = (error_type, signature)
            if identity in identities:
                raise InvalidArtifact(f"{context}: duplicate exception signature row")
            identities.add(identity)
            derived_type_counts[error_type] += count
            rows.append(
                {
                    "error_type": error_type,
                    "error_signature": signature,
                    "count": count,
                }
            )
        if rows != sorted(
            rows, key=lambda row: (row["error_type"], row["error_signature"])
        ):
            raise InvalidArtifact(f"{context}: exception signature rows are not sorted")
        if dict(sorted(derived_type_counts.items())) != dict(sorted(type_counts.items())):
            raise InvalidArtifact(f"{context}: exception type/signature totals differ")
        return dict(sorted(type_counts.items())), rows

    @classmethod
    def _seed_attempt_exception_diagnostics(
        cls,
        attempts: Sequence[Any],
        *,
        structural_only: bool,
        context: str,
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        """Recompute ordinary or structural diagnostics from immutable rows."""

        signature_counts: Counter[tuple[str, str]] = Counter()
        for index, row in enumerate(attempts, start=1):
            if not isinstance(row, Mapping):
                raise InvalidArtifact(f"{context}: attempt {index} is not an object")
            structural = row.get("structural_error")
            if not isinstance(structural, bool):
                raise InvalidArtifact(
                    f"{context}: attempt {index} lacks a boolean structural_error"
                )
            has_exception = "error_type" in row or "error_signature" in row
            if structural and not has_exception:
                raise InvalidArtifact(
                    f"{context}: structural attempt {index} lacks exception evidence"
                )
            if not has_exception:
                continue
            error_type = row.get("error_type")
            signature = row.get("error_signature")
            if (
                not isinstance(error_type, str)
                or not error_type
                or not isinstance(signature, str)
                or len(signature) != 64
            ):
                raise InvalidArtifact(
                    f"{context}: attempt {index} has malformed exception evidence"
                )
            try:
                int(signature, 16)
            except ValueError as error:
                raise InvalidArtifact(
                    f"{context}: attempt {index} exception signature is not hexadecimal"
                ) from error
            if not structural_only or structural:
                signature_counts[(error_type, signature)] += 1

        rows = [
            {
                "error_type": error_type,
                "error_signature": signature,
                "count": count,
            }
            for (error_type, signature), count in sorted(signature_counts.items())
        ]
        type_totals: Counter[str] = Counter()
        for (error_type, _signature), count in signature_counts.items():
            type_totals[error_type] += count
        type_counts = dict(sorted(type_totals.items()))
        return cls._validated_seed_exception_diagnostics(
            type_counts,
            rows,
            context=context,
        )

    def _validate_cache_workers(
        self, spec: StageSpec, workers: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        if len(workers) != 4:
            raise InvalidArtifact(f"{spec.name} requires exactly four cache workers")
        ranks = [int(row.get("rank", -1)) for row in workers]
        if sorted(ranks) != list(range(4)):
            raise InvalidArtifact(f"{spec.name} cache ranks differ: {ranks}")
        if any(int(row.get("world_size", -1)) != 4 for row in workers):
            raise InvalidArtifact(f"{spec.name} cache world size differs")
        receipt_paths = {
            Path(str(row.get("cache_receipt", ""))).resolve() for row in workers
        }
        if len(receipt_paths) != 1:
            raise InvalidArtifact(f"{spec.name} workers disagree on cache receipt")
        receipt_path = next(iter(receipt_paths))
        receipt = _read_json(receipt_path)
        receipt_sha = _sha256(receipt_path)
        artifact_rows = [
            artifact
            for worker in workers
            for artifact in worker.get("artifacts", [])
            if isinstance(artifact, Mapping)
        ]
        if not any(
            self._artifact_path(row).resolve() == receipt_path
            and row.get("sha256") == receipt_sha
            for row in artifact_rows
        ):
            raise InvalidArtifact(f"{spec.name} aggregate cache receipt is not hashed")

        if spec.name == "dino_cache":
            expected = {
                "schema": "before-we-act.bicoord.dino-cache/1",
                "status": "PASSED",
                "episodes": FORMAL_EPISODES,
                "episodes_per_task": {
                    task: FORMAL_EPISODES_PER_TASK for task in TASKS
                },
                "strict_dino_contract": True,
                "encoder": "dinov3_vitb16_frozen",
                "feature_width": 768,
                "image_height": 224,
                "image_width": 224,
                "patch_size": 16,
            }
            self._require_mapping_values(receipt, expected, "DINO cache receipt")
            if sum(int(row.get("episodes", -1)) for row in workers) != FORMAL_EPISODES:
                raise InvalidArtifact("DINO cache worker episode coverage differs")
            if any(row.get("strict_dino_contract") is not True for row in workers):
                raise InvalidArtifact("DINO cache worker relaxed the model contract")
            # Rank zero publishes the global receipt; non-zero ranks publish
            # only their immutable shard manifest.  Do not require every
            # worker to hash the global receipt (that would reject the
            # intended rank-sharded protocol), but do require every shard and
            # every file row to be hash-verified and bound to that receipt.
            global_files = receipt.get("files")
            if not isinstance(global_files, list) or len(global_files) != FORMAL_EPISODES:
                raise InvalidArtifact("DINO cache receipt file coverage differs")
            global_keys: set[tuple[str, int, str, str]] = set()
            for file_row in global_files:
                if not isinstance(file_row, Mapping):
                    raise InvalidArtifact("DINO cache receipt contains an invalid file row")
                file_path = self._artifact_path(file_row).resolve()
                digest = file_row.get("sha256")
                key = (
                    str(file_row.get("task", "")),
                    int(file_row.get("episode_id", -1)),
                    str(file_row.get("source_identity", "")),
                    str(digest),
                )
                if key in global_keys or not file_path.is_file() or not isinstance(digest, str):
                    raise InvalidArtifact("DINO cache receipt has duplicate/missing file row")
                if _sha256(file_path) != digest:
                    raise InvalidArtifact(f"DINO cache file hash differs: {file_path}")
                global_keys.add(key)
            shard_keys: set[tuple[str, int, str, str]] = set()
            for worker in workers:
                rank = int(worker.get("rank", -1))
                shard_candidates = [
                    self._artifact_path(artifact).resolve()
                    for artifact in worker.get("artifacts", [])
                    if isinstance(artifact, Mapping)
                    and artifact.get("kind") == "dino_cache"
                ]
                if len(shard_candidates) != 1:
                    raise InvalidArtifact(
                        f"DINO rank {rank} must publish exactly one shard/aggregate artifact"
                    )
                evidence = shard_candidates[0]
                expected_evidence = receipt_path if rank == 0 else (
                    receipt_path.parent / f"shard_{rank}.json"
                )
                if evidence != expected_evidence.resolve():
                    raise InvalidArtifact(
                        f"DINO rank {rank} published the wrong evidence: {evidence}"
                    )
                if not evidence.is_file():
                    raise InvalidArtifact(f"DINO rank {rank} evidence is missing: {evidence}")
                # Rank zero publishes the aggregate as its worker artifact,
                # but its shard manifest must still exist and participate in
                # exact union coverage just like ranks 1..3.
                shard_path = receipt_path.parent / f"shard_{rank}.json"
                shard = _read_json(shard_path)
                dino_source = receipt.get("dino_source")
                dino_source_sha = (
                    dino_source.get("sha256")
                    if isinstance(dino_source, Mapping)
                    else None
                )
                self._require_mapping_values(
                    shard,
                    {
                        "schema": "before-we-act.bicoord.dino-cache-shard/1",
                        "rank": rank,
                        "world_size": 4,
                        "smoke": False,
                        "config_sha256": self.config_hash,
                        "dataset_revision": FORMAL_DATASET_REVISION,
                        "dino_source_sha256": dino_source_sha,
                    },
                    f"DINO shard {rank}",
                )
                rows = shard.get("files")
                if not isinstance(rows, list):
                    raise InvalidArtifact(f"DINO shard {rank} lacks file rows")
                for file_row in rows:
                    if not isinstance(file_row, Mapping):
                        raise InvalidArtifact(f"DINO shard {rank} has an invalid file row")
                    file_path = self._artifact_path(file_row).resolve()
                    digest = file_row.get("sha256")
                    key = (
                        str(file_row.get("task", "")),
                        int(file_row.get("episode_id", -1)),
                        str(file_row.get("source_identity", "")),
                        str(digest),
                    )
                    if key in shard_keys or not file_path.is_file() or not isinstance(digest, str):
                        raise InvalidArtifact(f"DINO shard {rank} has duplicate/missing file row")
                    if _sha256(file_path) != digest:
                        raise InvalidArtifact(f"DINO shard {rank} file hash differs: {file_path}")
                    shard_keys.add(key)
            if shard_keys != global_keys:
                raise InvalidArtifact(
                    "DINO shard union is not exactly the global receipt coverage"
                )
            return {
                "cache_complete": True,
                "episodes": FORMAL_EPISODES,
                "episodes_per_task": expected["episodes_per_task"],
                "cache_receipt": str(receipt_path),
                "cache_receipt_sha256": receipt_sha,
                "rank_coverage": list(range(4)),
            }

        formal = spec.name == "bcore_cache"
        expected_episodes = FORMAL_EPISODES if formal else len(TASKS)
        per_task = {
            task: FORMAL_EPISODES_PER_TASK if formal else 1 for task in TASKS
        }
        expected = {
            "schema": "before-we-act.bicoord.bcore-cache/1",
            "status": "PASSED",
            "cache_complete": True,
            "formal": formal,
            "episodes": expected_episodes,
            "episodes_per_task": per_task,
            "tasks": list(TASKS),
            "world_size": 4,
            "dataset_revision": FORMAL_DATASET_REVISION,
            "config_sha256": self.config_hash,
            "action_lag_rows": 1,
            "source_frequency_hz": SOURCE_FREQUENCY_HZ,
            "strictly_decentralized": True,
        }
        self._require_mapping_values(receipt, expected, f"{spec.name} receipt")
        for row in workers:
            self._require_mapping_values(
                row,
                {
                    "cache_complete": True,
                    "episodes": expected_episodes,
                    "episodes_per_task": per_task,
                },
                f"{spec.name} worker",
            )
            if not any(
                self._artifact_path(artifact).resolve() == receipt_path
                and artifact.get("sha256") == receipt_sha
                for artifact in row.get("artifacts", [])
                if isinstance(artifact, Mapping)
            ):
                raise InvalidArtifact(
                    f"{spec.name} rank {row.get('rank')} did not bind the global receipt"
                )
        rank_hashes = receipt.get("rank_receipt_sha256")
        if not isinstance(rank_hashes, Mapping) or len(rank_hashes) != 4:
            raise InvalidArtifact(f"{spec.name} lacks four shard receipt hashes")
        return {
            "cache_complete": True,
            "formal": formal,
            "episodes": expected_episodes,
            "episodes_per_task": per_task,
            "samples": int(receipt.get("samples", -1)),
            "cache_receipt": str(receipt_path),
            "cache_receipt_sha256": receipt_sha,
            "rank_coverage": list(range(4)),
        }

    def _validate_training_workers(
        self, spec: StageSpec, workers: Sequence[Mapping[str, Any]]
    ) -> None:
        if spec.name == "bcore_train_3seeds":
            observed = [int(row.get("seed", -1)) for row in workers]
            if sorted(observed) != sorted(BCORE_SEEDS) or len(workers) != len(BCORE_SEEDS):
                raise InvalidArtifact(f"B-core seed coverage differs: {observed}")
            for row in workers:
                self._require_mapping_values(
                    row,
                    {
                        "update": self.s.bcore_updates,
                        "effective_batch": GLOBAL_BATCH,
                        "all_1800_demonstrations": True,
                        "closed_loop_results_used_for_selection": False,
                        "teacher_present": False,
                    },
                    "B-core training worker",
                )
            return
        if spec.name == "belief_train":
            # The aggregate contains the twelve deployment-main jobs and
            # three explicitly separate OOF shadows.  Keep these namespaces
            # disjoint: shadows are calibration-only and may never silently
            # replace a deployment candidate.
            main_workers = [
                row for row in workers if row.get("oof_shadow") is not True
            ]
            shadow_workers = [
                row for row in workers if row.get("oof_shadow") is True
            ]
            observed = [
                (str(row.get("variant", "")), int(row.get("seed", -1)))
                for row in main_workers
            ]
            expected = [
                (variant, seed)
                for variant in CARE_VARIANTS
                for seed in CARE_SEEDS
            ]
            if (
                len(main_workers) != len(expected)
                or sorted(observed) != sorted(expected)
                or len(observed) != len(set(observed))
            ):
                raise InvalidArtifact(
                    "CARE deployment-main variant/seed grid coverage differs"
                )
            for row in main_workers:
                self._require_mapping_values(
                    row,
                    {
                        "updates": self.s.care_updates,
                        "all_families_for_training": True,
                        "held_out_families": 0,
                        "oof_shadow": False,
                        "oof_calibration_role": "deployment_main_only",
                    },
                    "CARE deployment-main worker",
                )
            observed_folds: list[int] = []
            for row in shadow_workers:
                fold = int(row.get("oof_shadow_fold", -1))
                observed_folds.append(fold)
                self._require_mapping_values(
                    row,
                    {
                        "variant": CARE_OOF_VARIANT,
                        "seed": CARE_OOF_SEED,
                        "updates": self.s.care_updates,
                        "training_seed": CARE_OOF_SEED
                        + CARE_OOF_TRAINING_SEED_OFFSET
                        + fold,
                        "oof_shadow": True,
                        "oof_shadow_fold": fold,
                        "oof_calibration_role": "shadow_only",
                        "all_families_for_training": False,
                        "oof_shadow_complete_partition": True,
                    },
                    "CARE OOF-shadow worker",
                )
                held_out = int(row.get("oof_shadow_held_out_families", -1))
                train_count = int(row.get("oof_shadow_train_families", -1))
                total_count = int(row.get("oof_shadow_total_families", -1))
                if held_out <= 0 or train_count <= 0 or total_count != held_out + train_count:
                    raise InvalidArtifact(
                        f"CARE OOF fold {fold} family partition is invalid"
                    )
                if int(row.get("held_out_families", -1)) != held_out:
                    raise InvalidArtifact(
                        f"CARE OOF fold {fold} held-out family count differs"
                    )
            if len(shadow_workers) != len(CARE_OOF_FOLDS) or sorted(observed_folds) != list(CARE_OOF_FOLDS):
                raise InvalidArtifact(
                    f"CARE OOF fold coverage differs: {observed_folds}"
                )
            if len(set(observed_folds)) != len(observed_folds):
                raise InvalidArtifact("CARE OOF fold coverage contains duplicates")

    def _validate_branch_result(
        self, spec: StageSpec, result: Mapping[str, Any]
    ) -> None:
        smoke = spec.name == "branch_smoke"
        families_per_task = 1 if smoke else self.s.families_per_task
        expected_families = families_per_task * len(TASKS)
        self._require_mapping_values(
            result,
            {
                "provider_policy": "B-core/TUNE",
                "families": expected_families,
                "branches_per_family": BRANCHES_PER_FAMILY,
                "physical_simulator_outcomes": True,
                "offline_demonstration_error_used": False,
            },
            f"{spec.name} result",
        )
        manifests = [
            self._artifact_path(row).resolve()
            for row in result.get("artifacts", [])
            if isinstance(row, Mapping) and row.get("kind") == "branch_manifest"
        ]
        if len(manifests) != 4:
            raise InvalidArtifact(f"{spec.name} must contain four shard manifests")
        family_ids: set[int] = set()
        snapshot_ids: set[str] = set()
        task_counts: Counter[str] = Counter()
        ranks: set[int] = set()
        for manifest_path in manifests:
            manifest = _read_json(manifest_path)
            self._require_mapping_values(
                manifest,
                {
                    "status": "PASSED",
                    "world_size": 4,
                    "families_per_task": families_per_task,
                    "branches_per_family": BRANCHES_PER_FAMILY,
                    "provider_policy": "B-core/TUNE",
                    "physical_simulator_outcomes": True,
                    "offline_demonstration_error_used": False,
                },
                f"branch shard {manifest_path}",
            )
            rank = int(manifest.get("rank", -1))
            if rank in ranks or rank not in range(4):
                raise InvalidArtifact(f"duplicate/invalid branch rank: {rank}")
            ranks.add(rank)
            records = manifest.get("records")
            if not isinstance(records, list) or int(manifest.get("families", -1)) != len(records):
                raise InvalidArtifact(f"branch shard record coverage differs: {manifest_path}")
            for record in records:
                if not isinstance(record, Mapping):
                    raise InvalidArtifact("branch record is not an object")
                npz_path = Path(str(record.get("npz", ""))).resolve()
                family_path = Path(str(record.get("manifest", ""))).resolve()
                if (
                    not npz_path.is_file()
                    or record.get("npz_sha256") != _sha256(npz_path)
                    or not family_path.is_file()
                    or record.get("manifest_sha256") != _sha256(family_path)
                ):
                    raise InvalidArtifact("branch family tensor/manifest hash differs")
                family = _read_json(family_path)
                self._require_mapping_values(
                    family,
                    {
                        "status": "PASSED",
                        "branches_per_family": BRANCHES_PER_FAMILY,
                        "provider_policy": "B-core/TUNE",
                        "physical_simulator_outcomes": True,
                        "offline_demonstration_error_used": False,
                        "pseudo_labels_used": False,
                        "action_clipping": False,
                        "candidate_transform_clipping": False,
                        "strict_lag_one": True,
                    },
                    f"branch family {family_path}",
                )
                family_id = int(family.get("family_id", -1))
                snapshot_id = str(family.get("snapshot_id", ""))
                task = str(family.get("task", ""))
                if family_id in family_ids or snapshot_id in snapshot_ids or task not in TASKS:
                    raise InvalidArtifact("branch family identity is duplicate/invalid")
                if family_id % 4 != rank:
                    raise InvalidArtifact("branch family was published by the wrong rank")
                family_ids.add(family_id)
                snapshot_ids.add(snapshot_id)
                task_counts[task] += 1
                probe = family.get("restore_probe")
                if (
                    not isinstance(probe, Mapping)
                    or probe.get("passed") is not True
                    or float(probe.get("max_abs_error", float("inf"))) > 1e-6
                ):
                    raise InvalidArtifact("physical branch restore probe failed")
                fidelity = family.get("reference_reactive_replay_fidelity")
                if (
                    not isinstance(fidelity, list)
                    or len(fidelity) != 2
                    or any(
                        float(row.get("utility_max_abs_error", float("inf"))) > 1e-6
                        for row in fidelity
                        if isinstance(row, Mapping)
                    )
                    or any(not isinstance(row, Mapping) for row in fidelity)
                ):
                    raise InvalidArtifact("reference reactive/replay fidelity failed")
                branches = family.get("branches")
                if not isinstance(branches, list) or len(branches) != BRANCHES_PER_FAMILY:
                    raise InvalidArtifact("branch family width differs")
                keys = {
                    (
                        int(row.get("candidate_id", -1)),
                        str(row.get("regime", "")),
                        int(row.get("repeat_id", -1)),
                    )
                    for row in branches
                    if isinstance(row, Mapping)
                    and row.get("physical_simulator_outcome") is True
                }
                expected_keys = {
                    (candidate, regime, repeat)
                    for candidate in range(6)
                    for regime in ("reactive", "replay")
                    for repeat in (0, 1)
                }
                if keys != expected_keys:
                    raise InvalidArtifact("branch family lacks 24 unique physical outcomes")
        if ranks != set(range(4)):
            raise InvalidArtifact("branch rank coverage differs")
        if family_ids != set(range(expected_families)):
            raise InvalidArtifact("branch family ID coverage differs")
        if task_counts != Counter({task: families_per_task for task in TASKS}):
            raise InvalidArtifact(f"branch task coverage differs: {task_counts}")

    def _validate_paired_result(
        self,
        result: Mapping[str, Any],
        *,
        episodes: int,
        max_steps_by_task: Mapping[str, int] | None = None,
    ) -> None:
        horizon_map = MAX_STEPS if max_steps_by_task is None else max_steps_by_task
        tasks = result.get("tasks")
        if not isinstance(tasks, Mapping) or tuple(tasks) != TASKS:
            raise InvalidArtifact("paired validation task rows/order differ")

        def _sha(value: Any, context: str) -> str:
            if not isinstance(value, str) or len(value) != 64:
                raise InvalidArtifact(f"{context}: expected a SHA-256 digest")
            try:
                int(value, 16)
            except ValueError as error:
                raise InvalidArtifact(f"{context}: digest is not hexadecimal") from error
            return value

        def _hashed_path(value: Any, digest: Any, context: str) -> Path:
            path = self._artifact_path({"path": value}).resolve()
            expected = _sha(digest, f"{context} hash")
            if not path.is_file() or _sha256(path) != expected:
                raise InvalidArtifact(f"{context}: path/hash differs: {path}")
            return path

        def _telemetry_count(
            row: Mapping[str, Any], key: str, context: str
        ) -> int:
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidArtifact(
                    f"{context}: {key} must be a non-negative integer"
                )
            return value

        def _progress(
            path: Path,
            mode: str,
            task: str,
            seed: int,
            max_steps: int,
            context: str,
        ) -> dict[str, Any]:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as error:
                raise InvalidArtifact(f"{context}: progress cannot be read") from error
            if not lines:
                raise InvalidArtifact(f"{context}: progress is empty")
            previous = 0
            final_success: bool | None = None
            prediction_oob = 0
            plan_oob = 0
            executed_oob = 0
            for line in lines:
                try:
                    row = json.loads(line)
                except (TypeError, json.JSONDecodeError) as error:
                    raise InvalidArtifact(f"{context}: progress has invalid JSON") from error
                if not isinstance(row, Mapping):
                    raise InvalidArtifact(f"{context}: progress row is not an object")
                self._require_mapping_values(
                    row,
                    {
                        "task": task,
                        "seed": seed,
                        "mode": mode,
                        "max_steps": max_steps,
                        "action_clipped": False,
                        "policy_output_clipping": False,
                        "gripper_reparameterization": False,
                        "executed_gripper_oob_count": 0,
                    },
                    f"{context} progress row",
                )
                step = int(row.get("step", -1))
                if step != previous + 1 or step > max_steps:
                    raise InvalidArtifact(f"{context}: progress step sequence differs")
                success = row.get("success")
                if not isinstance(success, bool):
                    raise InvalidArtifact(f"{context}: progress success is not boolean")
                prediction_oob += _telemetry_count(
                    row, "prediction_gripper_oob_count", context
                )
                plan_oob += _telemetry_count(
                    row, "ensemble_plan_gripper_oob_count", context
                )
                executed_oob += _telemetry_count(
                    row, "executed_gripper_oob_count", context
                )
                previous = step
                final_success = success
            return {
                "steps": previous,
                "success": final_success,
                "prediction_gripper_oob_count": prediction_oob,
                "ensemble_plan_gripper_oob_count": plan_oob,
                "executed_gripper_oob_count": executed_oob,
            }

        total_control = 0
        total_care = 0
        reference_hashes: set[str] = set()
        care_hashes: set[str] = set()
        normalization_hashes: set[str] = set()
        reference_paths: set[str] = set()
        care_paths: set[str] = set()
        all_pair_files: set[str] = set()
        expected_operation = (
            "smoke-paired" if episodes == 1 else "validation20-paired"
        )
        for task in TASKS:
            expected_steps = int(horizon_map[task])
            task_row = tasks[task]
            if not isinstance(task_row, Mapping):
                raise InvalidArtifact(f"paired validation lacks {task}")
            self._require_mapping_values(
                task_row,
                {
                    "task": task,
                    "paired": True,
                    "selector_off_control": True,
                    "execution_order": ["selector_off", "care"],
                    "episodes": episodes,
                    "completed": episodes,
                    "rollouts": episodes * 2,
                    "max_steps": expected_steps,
                    "same_initial_state_verified": True,
                    "per_arm_independent_selector": True,
                    "cross_arm_lower_bound_arbitration": False,
                    "action_encoding": ACTION_ENCODING,
                    "action_horizon": ACTION_HORIZON,
                    "action_clipping": False,
                    "state_clipping": False,
                    "policy_output_clipping": False,
                    "gripper_reparameterization": False,
                    "executed_gripper_oob_count": 0,
                    "selector_off_progress": task_row.get("selector_off_progress"),
                    "care_progress": task_row.get("care_progress"),
                },
                f"paired task {task}",
            )
            combined_progress_paths: dict[str, Path] = {}
            for mode in ("selector_off", "care"):
                combined_progress_paths[mode] = _hashed_path(
                    task_row.get(f"{mode}_progress"),
                    task_row.get(f"{mode}_progress_sha256"),
                    f"{task} combined {mode} progress",
                )
            seeds = task_row.get("seeds")
            if (
                not isinstance(seeds, list)
                or len(seeds) != episodes
                or len(set(seeds)) != episodes
                or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
            ):
                raise InvalidArtifact(f"{task}: paired seed coverage differs")
            receipt_path = _hashed_path(
                task_row.get("progress_receipt"),
                task_row.get("progress_receipt_sha256"),
                f"{task} paired progress receipt",
            )
            receipt = _read_json(receipt_path)
            self._require_mapping_values(
                receipt,
                {
                    "schema": "before-we-act.bicoord.care-paired-progress/1",
                    "status": "PASSED",
                    "task": task,
                    "episodes": episodes,
                    "completed": episodes,
                    "rollouts": episodes * 2,
                    "max_steps": expected_steps,
                    "seeds": seeds,
                    "paired": True,
                    "selector_off_control": True,
                    "execution_order": ["selector_off", "care"],
                    "all_pairs_same_initial_simulator_state": True,
                    "all_pairs_same_initial_observation": True,
                    "per_arm_independent_selector": True,
                    "cross_arm_lower_bound_arbitration": False,
                    "operation": expected_operation,
                    "smoke_interface_steps": (
                        SMOKE_INTERFACE_STEPS if episodes == 1 else None
                    ),
                    "policy_output_clipping": False,
                    "action_clipping": False,
                    "state_clipping": False,
                    "gripper_reparameterization": False,
                    "executed_gripper_oob_count": 0,
                    "selector_off_progress": task_row.get(
                        "selector_off_progress"
                    ),
                    "selector_off_progress_sha256": task_row.get(
                        "selector_off_progress_sha256"
                    ),
                    "care_progress": task_row.get("care_progress"),
                    "care_progress_sha256": task_row.get(
                        "care_progress_sha256"
                    ),
                },
                f"paired receipt {task}",
            )
            receipt_reference_hash = _sha(
                receipt.get("reference_checkpoint_sha256"),
                f"{task} reference checkpoint",
            )
            receipt_care_hash = _sha(
                receipt.get("care_checkpoint_sha256"),
                f"{task} CARE checkpoint",
            )
            _hashed_path(
                receipt.get("reference_checkpoint"),
                receipt_reference_hash,
                f"{task} reference checkpoint",
            )
            _hashed_path(
                receipt.get("care_checkpoint"),
                receipt_care_hash,
                f"{task} CARE checkpoint",
            )
            normalization = receipt.get("normalization")
            if not isinstance(normalization, Mapping):
                raise InvalidArtifact(f"{task}: normalization provenance is missing")
            self._require_mapping_values(
                normalization,
                {"state_clipping": False, "action_clipping": False},
                f"{task} paired normalization",
            )
            normalization_hash = _sha(
                normalization.get("sha256"), f"{task} normalization"
            )
            _hashed_path(
                normalization.get("path"),
                normalization_hash,
                f"{task} normalization",
            )
            seed_manifest_hash = _sha(
                receipt.get("seed_manifest_sha256"), f"{task} seed manifest"
            )
            _hashed_path(
                receipt.get("seed_manifest"),
                seed_manifest_hash,
                f"{task} seed manifest",
            )
            self._require_mapping_values(
                task_row,
                {
                    "reference_checkpoint": receipt.get("reference_checkpoint"),
                    "reference_checkpoint_sha256": receipt_reference_hash,
                    "care_checkpoint": receipt.get("care_checkpoint"),
                    "care_checkpoint_sha256": receipt_care_hash,
                    "seed_manifest": receipt.get("seed_manifest"),
                    "seed_manifest_sha256": seed_manifest_hash,
                    "normalization": normalization,
                },
                f"paired task {task} provenance",
            )
            reference_hashes.add(receipt_reference_hash)
            care_hashes.add(receipt_care_hash)
            normalization_hashes.add(normalization_hash)
            reference_paths.add(str(Path(str(receipt["reference_checkpoint"])).resolve()))
            care_paths.add(str(Path(str(receipt["care_checkpoint"])).resolve()))
            rows = receipt.get("rows")
            if not isinstance(rows, list) or len(rows) != episodes:
                raise InvalidArtifact(f"{task}: paired receipt rows differ")
            if task_row.get("paired_rows") != rows:
                raise InvalidArtifact(f"{task}: aggregate dropped/changed paired rows")
            pair_manifest_path = _hashed_path(
                task_row.get("pair_manifest"),
                task_row.get("pair_manifest_sha256"),
                f"{task} pair manifest",
            )
            if receipt.get("pair_manifest") != str(pair_manifest_path):
                raise InvalidArtifact(f"{task}: receipt pair manifest path differs")
            if receipt.get("pair_manifest_sha256") != _sha256(pair_manifest_path):
                raise InvalidArtifact(f"{task}: receipt pair manifest hash differs")
            pair_manifest = _read_json(pair_manifest_path)
            self._require_mapping_values(
                pair_manifest,
                {
                    "schema": "before-we-act.bicoord.care-pair-manifest/1",
                    "status": "PASSED",
                    "task": task,
                    "episodes": episodes,
                    "operation": expected_operation,
                },
                f"{task} pair manifest",
            )
            manifest_pairs = pair_manifest.get("pairs")
            if not isinstance(manifest_pairs, list) or len(manifest_pairs) != episodes:
                raise InvalidArtifact(f"{task}: pair manifest coverage differs")
            manifest_identity = pair_manifest.get("identity")
            if not isinstance(manifest_identity, Mapping):
                raise InvalidArtifact(f"{task}: pair manifest identity is missing")
            self._require_mapping_values(
                manifest_identity,
                {
                    "task": task,
                    "max_steps": expected_steps,
                    "reference_checkpoint_sha256": receipt_reference_hash,
                    "care_checkpoint_sha256": receipt_care_hash,
                    "seed_manifest_sha256": seed_manifest_hash,
                    "normalization_sha256": normalization_hash,
                    "operation": expected_operation,
                },
                f"{task} pair manifest identity",
            )
            task_control = 0
            task_care = 0
            task_prediction_oob = 0
            task_plan_oob = 0
            task_executed_oob = 0
            seed_steps_by_mode: dict[str, list[tuple[int, int]]] = {
                "selector_off": [],
                "care": [],
            }
            for seed, pair, manifest_pair in zip(seeds, rows, manifest_pairs, strict=True):
                if not isinstance(pair, Mapping):
                    raise InvalidArtifact(f"{task}: paired row is invalid")
                self._require_mapping_values(
                    pair,
                    {
                        "task": task,
                        "seed": int(seed),
                        "paired": True,
                        "execution_order": ["selector_off", "care"],
                        "same_initial_simulator_state": True,
                        "same_initial_observation": True,
                    },
                    f"paired row {task}/{seed}",
                )
                for state_key in (
                    "initial_state_sha256",
                    "restored_state_sha256",
                    "initial_observation_sha256",
                    "restored_observation_sha256",
                ):
                    _sha(pair.get(state_key), f"{task}/{seed} {state_key}")
                if pair.get("initial_state_sha256") != pair.get("restored_state_sha256"):
                    raise InvalidArtifact(f"{task}/{seed}: paired state hashes differ")
                if pair.get("initial_observation_sha256") != pair.get("restored_observation_sha256"):
                    raise InvalidArtifact(f"{task}/{seed}: paired observation hashes differ")
                pair_path = _hashed_path(
                    manifest_pair.get("path"),
                    manifest_pair.get("sha256"),
                    f"{task}/{seed} pair file",
                )
                if str(pair_path) in all_pair_files:
                    raise InvalidArtifact(f"{task}/{seed}: duplicate pair file")
                all_pair_files.add(str(pair_path))
                if int(manifest_pair.get("seed", -1)) != int(seed):
                    raise InvalidArtifact(f"{task}/{seed}: pair manifest seed differs")
                pair_file = _read_json(pair_path)
                self._require_mapping_values(
                    pair_file,
                    {
                        "task": task,
                        "seed": int(seed),
                        "paired": True,
                        "execution_order": ["selector_off", "care"],
                        "same_initial_simulator_state": True,
                        "same_initial_observation": True,
                    },
                    f"pair file {task}/{seed}",
                )
                payload_sha = pair_file.get("pair_payload_sha256")
                unsigned_pair = dict(pair_file)
                unsigned_pair.pop("pair_payload_sha256", None)
                if not isinstance(payload_sha, str) or payload_sha != _canonical_hash(unsigned_pair):
                    raise InvalidArtifact(f"{task}/{seed}: pair payload hash differs")
                if manifest_pair.get("pair_identity_sha256") != pair_file.get("pair_identity_sha256"):
                    raise InvalidArtifact(f"{task}/{seed}: pair identity hash differs")
                pair_identity = pair_file.get("pair_identity")
                pair_identity_sha = _sha(
                    pair_file.get("pair_identity_sha256"),
                    f"{task}/{seed} pair identity",
                )
                if not isinstance(pair_identity, Mapping) or pair_identity_sha != _canonical_hash(pair_identity):
                    raise InvalidArtifact(f"{task}/{seed}: pair identity digest differs")
                if pair_identity != {
                    "task": task,
                    "seed": int(seed),
                    "max_steps": expected_steps,
                    "reference_checkpoint_sha256": receipt_reference_hash,
                    "care_checkpoint_sha256": receipt_care_hash,
                    "seed_manifest_sha256": seed_manifest_hash,
                    "normalization_sha256": normalization_hash,
                    "operation": expected_operation,
                }:
                    raise InvalidArtifact(f"{task}/{seed}: pair identity differs")
                expected_receipt_pair = dict(pair_file)
                expected_receipt_pair.update(
                    {
                        "pair_file": str(pair_path),
                        "pair_file_sha256": _sha256(pair_path),
                    }
                )
                if dict(pair) != expected_receipt_pair:
                    raise InvalidArtifact(
                        f"{task}/{seed}: progress receipt pair differs from pair file"
                    )
                if pair_file.get("reference_checkpoint_sha256") not in (None, receipt_reference_hash):
                    raise InvalidArtifact(f"{task}/{seed}: reference checkpoint hash differs")
                if pair_file.get("care_checkpoint_sha256") not in (None, receipt_care_hash):
                    raise InvalidArtifact(f"{task}/{seed}: CARE checkpoint hash differs")
                for mode in ("selector_off", "care"):
                    rollout = pair_file.get(mode)
                    if not isinstance(rollout, Mapping):
                        raise InvalidArtifact(f"{task}/{seed}: missing {mode} rollout")
                    self._require_mapping_values(
                        rollout,
                        {
                            "task": task,
                            "seed": int(seed),
                            "mode": mode,
                            "max_steps": expected_steps,
                            "action_clipping": False,
                            "state_clipping": False,
                            "policy_output_clipping": False,
                            "gripper_reparameterization": False,
                            "executed_gripper_oob_count": 0,
                            "per_arm_independent_selector": True,
                            "cross_arm_lower_bound_arbitration": False,
                            "strictly_decentralized": True,
                        },
                        f"{task}/{seed}/{mode}",
                    )
                    steps = int(rollout.get("steps", -1))
                    if steps < 1 or steps > expected_steps:
                        raise InvalidArtifact(f"{task}/{seed}/{mode}: invalid step count")
                    if not isinstance(rollout.get("success"), bool):
                        raise InvalidArtifact(f"{task}/{seed}/{mode}: success is not boolean")
                    if steps < expected_steps and not bool(rollout["success"]):
                        raise InvalidArtifact(
                            f"{task}/{seed}/{mode}: early termination without success"
                        )
                    for trace_key in ("action_trace_sha256", "reference_action_trace_sha256"):
                        _sha(rollout.get(trace_key), f"{task}/{seed}/{mode} {trace_key}")
                    for progress_key in (f"{mode}_progress", f"{mode}_progress_sha256"):
                        if progress_key not in pair_file:
                            raise InvalidArtifact(f"{task}/{seed}: missing {progress_key}")
                    progress_path = _hashed_path(
                        pair_file.get(f"{mode}_progress"),
                        pair_file.get(f"{mode}_progress_sha256"),
                        f"{task}/{seed}/{mode} progress",
                    )
                    progress_summary = _progress(
                        progress_path,
                        mode,
                        task,
                        int(seed),
                        expected_steps,
                        f"{task}/{seed}/{mode}",
                    )
                    if progress_summary["steps"] != steps:
                        raise InvalidArtifact(f"{task}/{seed}/{mode}: progress/trace length differs")
                    if progress_summary["success"] is not rollout["success"]:
                        raise InvalidArtifact(
                            f"{task}/{seed}/{mode}: progress final success differs"
                        )
                    seed_steps_by_mode[mode].append((int(seed), steps))
                    rollout_telemetry = {
                        key: _telemetry_count(rollout, key, f"{task}/{seed}/{mode}")
                        for key in (
                            "prediction_gripper_oob_count",
                            "ensemble_plan_gripper_oob_count",
                            "executed_gripper_oob_count",
                        )
                    }
                    if any(
                        rollout_telemetry[key] != progress_summary[key]
                        for key in rollout_telemetry
                    ):
                        raise InvalidArtifact(
                            f"{task}/{seed}/{mode}: progress telemetry differs from rollout"
                        )
                    task_prediction_oob += rollout_telemetry[
                        "prediction_gripper_oob_count"
                    ]
                    task_plan_oob += rollout_telemetry[
                        "ensemble_plan_gripper_oob_count"
                    ]
                    task_executed_oob += rollout_telemetry[
                        "executed_gripper_oob_count"
                    ]
                if bool(pair_file["selector_off"]["success"]):
                    task_control += 1
                if bool(pair_file["care"]["success"]):
                    task_care += 1
            for mode in ("selector_off", "care"):
                _validate_paired_progress(
                    combined_progress_paths[mode],
                    mode=mode,
                    task=task,
                    max_steps=expected_steps,
                    seed_steps=seed_steps_by_mode[mode],
                    context=f"{task} combined {mode}",
                )
            control = int(receipt.get("selector_off_successes", -1))
            care = int(receipt.get("care_successes", -1))
            if control != task_control or care != task_care:
                raise InvalidArtifact(f"{task}: paired receipt successes differ from rows")
            if (
                control != int(task_row.get("selector_off_successes", -2))
                or care != int(task_row.get("care_successes", -2))
                or int(task_row.get("successes", -3)) != care
            ):
                raise InvalidArtifact(f"{task}: paired successes were not preserved")
            task_telemetry = {
                "prediction_gripper_oob_count": task_prediction_oob,
                "ensemble_plan_gripper_oob_count": task_plan_oob,
                "executed_gripper_oob_count": task_executed_oob,
            }
            self._require_mapping_values(
                receipt, task_telemetry, f"{task} paired receipt telemetry"
            )
            self._require_mapping_values(
                task_row, task_telemetry, f"{task} paired worker telemetry"
            )
            artifact_rows = task_row.get("artifacts")
            if not isinstance(artifact_rows, list):
                raise InvalidArtifact(f"{task}: paired worker artifacts are missing")
            by_kind: dict[str, list[Path]] = {}
            for artifact_row in artifact_rows:
                if not isinstance(artifact_row, Mapping):
                    raise InvalidArtifact(f"{task}: invalid paired worker artifact")
                by_kind.setdefault(str(artifact_row.get("kind", "")), []).append(
                    self._artifact_path(artifact_row).resolve()
                )
            required_counts = {
                "selector_off_progress": 1,
                "care_progress": 1,
                "paired_progress_receipt": 1,
                "paired_seed_manifest": 1,
                "paired_seed_result": episodes,
            }
            for kind, count in required_counts.items():
                if len(by_kind.get(kind, [])) != count:
                    raise InvalidArtifact(
                        f"{task}: expected {count} {kind} artifacts"
                    )
            if by_kind["paired_progress_receipt"] != [receipt_path]:
                raise InvalidArtifact(f"{task}: paired receipt artifact path differs")
            if by_kind["paired_seed_manifest"] != [pair_manifest_path]:
                raise InvalidArtifact(f"{task}: pair manifest artifact path differs")
            expected_pair_paths = {
                Path(str(item["path"])).resolve() for item in manifest_pairs
            }
            if set(by_kind["paired_seed_result"]) != expected_pair_paths:
                raise InvalidArtifact(f"{task}: paired seed artifacts differ")
            total_control += control
            total_care += care
        if len(reference_hashes) != 1 or len(care_hashes) != 1 or len(normalization_hashes) != 1:
            raise InvalidArtifact("paired workers used different checkpoints/normalization")
        if len(reference_paths) != 1 or len(care_paths) != 1:
            raise InvalidArtifact("paired workers used different checkpoint paths")
        aggregate_expected = {
            "paired": True,
            "selector_off_control": True,
            "total_episodes": len(TASKS) * episodes,
            "total_rollouts": len(TASKS) * episodes * 2,
            "selector_off_successes": total_control,
            "care_successes": total_care,
            "total_successes": total_care,
            "same_initial_state_verified": True,
            "per_arm_independent_selector": True,
            "cross_arm_lower_bound_arbitration": False,
            "execution_order": ["selector_off", "care"],
        }
        self._require_mapping_values(result, aggregate_expected, "paired aggregate")
        # If the aggregator exposes provenance at the top level, bind it to
        # the values proved independently by every task receipt.
        if "reference_checkpoint_sha256" in result and result["reference_checkpoint_sha256"] != next(iter(reference_hashes)):
            raise InvalidArtifact("paired aggregate reference checkpoint hash differs")
        if "care_checkpoint_sha256" in result and result["care_checkpoint_sha256"] != next(iter(care_hashes)):
            raise InvalidArtifact("paired aggregate CARE checkpoint hash differs")
        if "normalization_sha256" in result and result["normalization_sha256"] != next(iter(normalization_hashes)):
            raise InvalidArtifact("paired aggregate normalization hash differs")

    @staticmethod
    def _require_mapping_values(
        observed: Mapping[str, Any], expected: Mapping[str, Any], context: str
    ) -> None:
        for key, value in expected.items():
            if observed.get(key) != value:
                raise InvalidArtifact(
                    f"{context} differs at {key}: {observed.get(key)!r} != {value!r}"
                )

    def _validate_model_contract(self, result: Mapping[str, Any]) -> None:
        contract = result.get("model_contract")
        if not isinstance(contract, Mapping):
            raise InvalidArtifact("model stage omitted model_contract")
        self._require_mapping_values(contract, MODEL_CONTRACT, "model_contract")

    def _validate_task_results(
        self,
        result: Mapping[str, Any],
        episodes: int,
        *,
        require_success: bool,
        max_steps_by_task: Mapping[str, int] | None = None,
    ) -> None:
        horizon_map = MAX_STEPS if max_steps_by_task is None else max_steps_by_task
        tasks = result.get("tasks")
        if not isinstance(tasks, Mapping) or tuple(tasks) != TASKS:
            raise InvalidArtifact("validation task coverage/order differs")
        successes = 0
        for task in TASKS:
            row = tasks[task]
            if not isinstance(row, Mapping):
                raise InvalidArtifact(f"invalid validation row: {task}")
            if int(row.get("episodes", -1)) != episodes:
                raise InvalidArtifact(f"{task}: validation episode count differs")
            expected_steps = int(horizon_map[task])
            if int(row.get("max_steps", -1)) != expected_steps:
                raise InvalidArtifact(f"{task}: validation max_steps differs")
            completed = int(row.get("completed", -1))
            if completed != episodes:
                raise InvalidArtifact(f"{task}: validation is incomplete")
            task_successes = int(row.get("successes", -1))
            if not 0 <= task_successes <= episodes:
                raise InvalidArtifact(f"{task}: invalid validation successes")
            if expected_steps == SMOKE_INTERFACE_STEPS and row.get("paired") is not True:
                # Smoke workers publish a hashed receipt containing the
                # per-episode rows and the JSONL trace.  Verify both rather
                # than trusting aggregate counters supplied by a worker.
                self._require_mapping_values(
                    row,
                    {
                        "smoke_interface_steps": SMOKE_INTERFACE_STEPS,
                        "policy_output_clipping": False,
                        "action_clipping": False,
                        "state_clipping": False,
                        "gripper_reparameterization": False,
                        "executed_gripper_oob_count": 0,
                    },
                    f"{task} smoke worker",
                )
                receipt_value = row.get("progress_receipt")
                receipt_digest = row.get("progress_receipt_sha256")
                if not isinstance(receipt_value, str) or not isinstance(receipt_digest, str):
                    raise InvalidArtifact(
                        f"{task}: smoke progress receipt provenance is missing"
                    )
                receipt_path = Path(receipt_value).expanduser().resolve()
                if (
                    not receipt_path.is_file()
                    or len(receipt_digest) != 64
                    or _sha256(receipt_path) != receipt_digest
                ):
                    raise InvalidArtifact(f"{task}: smoke progress receipt hash differs")
                artifacts = row.get("artifacts")
                if not isinstance(artifacts, list) or not any(
                    isinstance(item, Mapping)
                    and item.get("kind") == "progress_receipt"
                    and self._artifact_path(item).resolve() == receipt_path
                    and item.get("sha256") == receipt_digest
                    for item in artifacts
                ):
                    raise InvalidArtifact(
                        f"{task}: smoke result did not bind its progress receipt artifact"
                    )
                receipt = _read_json(receipt_path)
                self._require_mapping_values(
                    receipt,
                    {
                        "status": "PASSED",
                        "task": task,
                        "episodes": episodes,
                        "completed": episodes,
                        "max_steps": expected_steps,
                        "smoke_interface_steps": SMOKE_INTERFACE_STEPS,
                        "policy_output_clipping": False,
                        "action_clipping": False,
                        "state_clipping": False,
                        "gripper_reparameterization": False,
                        "executed_gripper_oob_count": 0,
                    },
                    f"{task} smoke progress receipt",
                )
                receipt_rows = receipt.get("rows")
                if not isinstance(receipt_rows, list) or len(receipt_rows) != episodes:
                    raise InvalidArtifact(f"{task}: smoke receipt row coverage differs")
                progress_value = receipt.get("rows_path")
                progress_digest = receipt.get("rows_sha256")
                if not isinstance(progress_value, str) or not isinstance(progress_digest, str):
                    raise InvalidArtifact(f"{task}: smoke progress path/hash is missing")
                progress_path = Path(progress_value).expanduser().resolve()
                if (
                    not progress_path.is_file()
                    or len(progress_digest) != 64
                    or _sha256(progress_path) != progress_digest
                ):
                    raise InvalidArtifact(f"{task}: smoke progress hash differs")
                if not any(
                    isinstance(item, Mapping)
                    and item.get("kind") == "validation_progress"
                    and self._artifact_path(item).resolve() == progress_path
                    and item.get("sha256") == progress_digest
                    for item in artifacts
                ):
                    raise InvalidArtifact(
                        f"{task}: smoke result did not bind its progress artifact"
                    )
                expected_rollout_steps: list[int] = []
                expected_seeds: list[int] = []
                receipt_successes = 0
                prediction_oob = 0
                plan_oob = 0
                for episode_row in receipt_rows:
                    if not isinstance(episode_row, Mapping):
                        raise InvalidArtifact(f"{task}: smoke receipt episode row is invalid")
                    if episode_row.get("task") != task:
                        raise InvalidArtifact(f"{task}: smoke receipt task differs")
                    if int(episode_row.get("max_steps", -1)) != expected_steps:
                        raise InvalidArtifact(f"{task}: smoke receipt max_steps differs")
                    steps = int(episode_row.get("steps", -1))
                    success = episode_row.get("success")
                    if steps < 1 or steps > expected_steps or not isinstance(success, bool):
                        raise InvalidArtifact(f"{task}: smoke receipt episode bounds differ")
                    # Environments may report success before the configured
                    # smoke horizon; only an unsuccessful early stop is
                    # invalid.
                    if steps < expected_steps and not success:
                        raise InvalidArtifact(
                            f"{task}: smoke episode terminated before horizon without success"
                        )
                    expected_rollout_steps.append(steps)
                    seed = int(episode_row.get("seed", -1))
                    if seed < 0:
                        raise InvalidArtifact(f"{task}: smoke receipt seed is invalid")
                    expected_seeds.append(seed)
                    receipt_successes += int(success)
                    prediction_oob += int(
                        episode_row.get("prediction_gripper_oob_count", 0)
                    )
                    plan_oob += int(
                        episode_row.get("ensemble_plan_gripper_oob_count", 0)
                    )
                    trace_digest = episode_row.get("action_trace_sha256")
                    if not isinstance(trace_digest, str) or len(trace_digest) != 64:
                        raise InvalidArtifact(f"{task}: smoke action trace hash is invalid")
                    try:
                        int(trace_digest, 16)
                    except ValueError as error:
                        raise InvalidArtifact(
                            f"{task}: smoke action trace hash is not hexadecimal"
                        ) from error
                if len(set(expected_seeds)) != episodes:
                    raise InvalidArtifact(f"{task}: smoke receipt seeds are not unique")
                if row.get("rollout_steps") != expected_rollout_steps:
                    raise InvalidArtifact(f"{task}: smoke rollout step summary differs")
                if receipt.get("rollout_steps") != expected_rollout_steps:
                    raise InvalidArtifact(f"{task}: smoke receipt rollout steps differ")
                if task_successes != receipt_successes:
                    raise InvalidArtifact(f"{task}: smoke success summary differs")
                telemetry = {
                    "prediction_gripper_oob_count": prediction_oob,
                    "ensemble_plan_gripper_oob_count": plan_oob,
                    "executed_gripper_oob_count": 0,
                }
                self._require_mapping_values(receipt, telemetry, f"{task} smoke receipt")
                self._require_mapping_values(row, telemetry, f"{task} smoke worker")
                lines = progress_path.read_text(encoding="utf-8").splitlines()
                if not lines:
                    raise InvalidArtifact(f"{task}: smoke progress is empty")
                by_seed: dict[int, list[Mapping[str, Any]]] = {}
                for line in lines:
                    try:
                        progress_row = json.loads(line)
                    except (TypeError, json.JSONDecodeError) as error:
                        raise InvalidArtifact(
                            f"{task}: smoke progress has invalid JSON"
                        ) from error
                    if not isinstance(progress_row, Mapping):
                        raise InvalidArtifact(f"{task}: smoke progress row is not an object")
                    if progress_row.get("task") != task:
                        raise InvalidArtifact(f"{task}: smoke progress task differs")
                    if int(progress_row.get("max_steps", -1)) != expected_steps:
                        raise InvalidArtifact(f"{task}: smoke progress max_steps differs")
                    self._require_mapping_values(
                        progress_row,
                        {
                            "action_clipped": False,
                            "policy_output_clipping": False,
                            "executed_gripper_oob_count": 0,
                        },
                        f"{task} smoke progress row",
                    )
                    seed = int(progress_row.get("seed", -1))
                    by_seed.setdefault(seed, []).append(progress_row)
                if set(by_seed) != set(expected_seeds):
                    raise InvalidArtifact(f"{task}: smoke progress seed coverage differs")
                for episode_row in receipt_rows:
                    seed = int(episode_row["seed"])
                    trace = by_seed.get(seed, [])
                    steps = int(episode_row["steps"])
                    sequence = [int(item.get("step", -1)) for item in trace]
                    if len(trace) != steps or sequence != list(range(1, steps + 1)):
                        raise InvalidArtifact(f"{task}: smoke progress step sequence differs")
                    if trace[-1].get("success") is not episode_row["success"]:
                        raise InvalidArtifact(
                            f"{task}: smoke progress final success differs"
                        )
            successes += task_successes
        if require_success and successes <= 0:
            raise Blocked("reference policy has zero successes; downstream CARE is unsafe")

    def _asset_runtime_expectations(self) -> dict[str, dict[str, Any]]:
        """Return the two runtime overlay identities proved by asset_contract.

        Seed discovery runs in separate child processes and records the
        selected asset metadata on *every* attempt.  Re-read and validate the
        asset stage here instead of trusting an environment variable or a
        worker-supplied path; this keeps a resumed supervisor fail-closed when
        either overlay or its receipt has drifted.
        """

        asset_result_path = self.result_path("asset_contract")
        if not asset_result_path.is_file():
            raise InvalidArtifact("asset contract result is missing for runtime overlay evidence")
        # This is intentionally a full validation.  It binds both source
        # revisions, receipt hashes, metadata overrides, and mesh provenance
        # before any seed evidence is accepted.
        self._validate_result(STAGES["asset_contract"], asset_result_path)
        asset_result = _read_json(asset_result_path)
        receipt_path = _canonical_stage_path(
            asset_result.get("asset_contract"),
            self.s.run / "artifacts" / "asset_contract" / "asset_contract.json",
            label="asset contract receipt",
        )
        if (
            not receipt_path.is_file()
            or asset_result.get("asset_contract_sha256") != _sha256(receipt_path)
        ):
            raise InvalidArtifact("asset contract receipt/hash differs for runtime overlay evidence")
        receipt = _read_json(receipt_path)
        plate = receipt.get("plate_overlay")
        shovel = receipt.get("shovel_overlay")
        if not isinstance(plate, Mapping) or not isinstance(shovel, Mapping):
            raise InvalidArtifact("asset contract does not contain both runtime overlays")
        plate_path = _canonical_stage_path(
            plate.get("overlay_metadata"),
            self.s.run
            / "artifacts"
            / "asset_contract"
            / "overlay"
            / "003_plate"
            / "model_data0.json",
            label="plate runtime overlay",
        )
        shovel_path = _canonical_stage_path(
            shovel.get("overlay_metadata"),
            self.s.run
            / "artifacts"
            / "asset_contract"
            / "overlay"
            / SHOVEL_OBJECT_NAME
            / SHOVEL_METADATA_NAME,
            label="shovel runtime overlay",
        )
        for label, path, row in (
            ("plate", plate_path, plate),
            ("shovel", shovel_path, shovel),
        ):
            if not path.is_file() or row.get("target_metadata_sha256") != _sha256(path):
                raise InvalidArtifact(f"{label} runtime overlay/hash differs")
        plate_contacts = plate.get("target_contact_points_pose_sha256")
        shovel_contacts = shovel.get("contact_points_pose_sha256")
        if (
            not isinstance(plate_contacts, str)
            or len(plate_contacts) != 64
            or not isinstance(shovel_contacts, str)
            or shovel_contacts != SHOVEL_CONTACT_POINTS_POSE_SHA256
        ):
            raise InvalidArtifact("runtime overlay contact-pose identity is invalid")
        try:
            int(plate_contacts, 16)
            int(shovel_contacts, 16)
        except ValueError as error:
            raise InvalidArtifact("runtime overlay contact-pose hash is not hexadecimal") from error
        if shovel.get("contact_points_pose_count") != 1:
            raise InvalidArtifact("shovel runtime overlay contact count differs")
        return {
            "place_plate_and_cup": {
                "overlay": str(plate_path),
                "contact_points_pose_sha256": plate_contacts,
                "contact_points_pose_count": 4,
                "receipt": str(receipt_path),
                "receipt_sha256": _sha256(receipt_path),
            },
            "sweep_block": {
                "overlay": str(shovel_path),
                "contact_points_pose_sha256": shovel_contacts,
                "contact_points_pose_count": 1,
                "receipt": str(receipt_path),
                "receipt_sha256": _sha256(receipt_path),
            },
        }

    def _validate_seed_asset_overlay(
        self,
        task: str,
        attempt: Mapping[str, Any],
        expectations: Mapping[str, Mapping[str, Any]],
        *,
        context: str,
    ) -> None:
        """Validate per-attempt runtime overlay evidence for known defects."""

        if task not in {"place_plate_and_cup", "sweep_block"}:
            return
        overlay = attempt.get("asset_overlay")
        if not isinstance(overlay, Mapping):
            raise InvalidArtifact(f"{context}: {task} attempt omitted asset overlay evidence")
        expected = expectations.get(task)
        if not isinstance(expected, Mapping):
            raise InvalidArtifact(f"{context}: no expected overlay identity for {task}")
        applied = overlay.get("applied")
        if not isinstance(applied, bool):
            raise InvalidArtifact(f"{context}: {task} overlay applied flag is invalid")
        observed_path = overlay.get("overlay")
        if applied and observed_path != expected.get("overlay"):
            raise InvalidArtifact(
                f"{context}: {task} overlay path differs: "
                f"{observed_path!r} != {expected.get('overlay')!r}"
            )
        if not applied and observed_path not in (None, expected.get("overlay")):
            raise InvalidArtifact(
                f"{context}: {task} overlay path differs: "
                f"{observed_path!r} != {expected.get('overlay')!r}"
            )
        if overlay.get("task") != task:
            raise InvalidArtifact(f"{context}: {task} overlay task provenance differs")
        if not applied:
            # A construction failure must remain explicit and may not be
            # converted into a nominal successful rollout.  It can be either
            # structural or an ordinary seed rejection (for example an
            # UnStableError during setup), but it must carry complete
            # exception evidence and may claim no runtime receipt or actor
            # mutation.
            if (
                attempt.get("valid") is not False
                or attempt.get("plan_success") is True
                or attempt.get("expert_success") is True
                or not (
                    attempt.get("structural_error") is True
                    or attempt.get("expected_seed_rejection") is True
                )
            ):
                raise InvalidArtifact(
                    f"{context}: {task} unapplied overlay attempt outcome differs"
                )
            if overlay.get("contact_points_pose_sha256") is not None:
                raise InvalidArtifact(
                    f"{context}: {task} failed overlay must not claim a contact hash"
                )
            if overlay.get("receipt") is not None or overlay.get(
                "receipt_sha256"
            ) is not None:
                raise InvalidArtifact(
                    f"{context}: {task} failed overlay must not claim a receipt"
                )
            actors = overlay.get("actors", {})
            if not isinstance(actors, Mapping) or actors:
                raise InvalidArtifact(
                    f"{context}: {task} failed overlay must not claim actor mutation"
                )
            reason = overlay.get("reason")
            if not isinstance(reason, str) or not reason:
                raise InvalidArtifact(f"{context}: {task} failed overlay lacks reason")
            error_type = attempt.get("error_type")
            error_signature = attempt.get("error_signature")
            if (
                not isinstance(error_type, str)
                or not error_type
                or not isinstance(error_signature, str)
                or len(error_signature) != 64
            ):
                raise InvalidArtifact(
                    f"{context}: {task} failed overlay lacks exception identity"
                )
            try:
                int(error_signature, 16)
            except ValueError as error:
                raise InvalidArtifact(
                    f"{context}: {task} failed overlay exception hash is invalid"
                ) from error
            return
        if overlay.get("task_source_modified") is not False:
            raise InvalidArtifact(f"{context}: {task} overlay source provenance differs")
        if (
            overlay.get("receipt") != expected.get("receipt")
            or overlay.get("receipt_sha256") != expected.get("receipt_sha256")
        ):
            raise InvalidArtifact(f"{context}: {task} overlay receipt provenance differs")
        if overlay.get("contact_points_pose_sha256") != expected.get(
            "contact_points_pose_sha256"
        ):
            raise InvalidArtifact(f"{context}: {task} converted contact hash differs")
        if task == "place_plate_and_cup":
            self._require_mapping_values(
                overlay,
                {
                    "copied_fields": [CONTACT_KEY],
                },
                f"{context}: plate runtime overlay",
            )
            if any(key in overlay for key in ("legacy_conversion", "derived_fields", "source_fields")):
                raise InvalidArtifact(f"{context}: plate overlay carries legacy shovel provenance")
            actors = overlay.get("actors")
            if not isinstance(actors, Mapping) or set(actors) != {"plate", "plate_2"}:
                raise InvalidArtifact(f"{context}: plate actor overlay coverage differs")
            for actor_name in ("plate", "plate_2"):
                actor = actors[actor_name]
                if not isinstance(actor, Mapping):
                    raise InvalidArtifact(f"{context}: plate actor evidence is invalid")
                self._require_mapping_values(
                    actor,
                    {
                        "after_sha256": expected["contact_points_pose_sha256"],
                        "contact_points_pose_count": 4,
                        "scale_preserved": True,
                        "changed_fields": [CONTACT_KEY],
                    },
                    f"{context}: {actor_name} overlay evidence",
                )
        else:
            self._require_mapping_values(
                overlay,
                {
                    "copied_fields": [CONTACT_KEY],
                    "derived_fields": [CONTACT_KEY],
                    "source_fields": [LEGACY_CONTACT_KEY, LEGACY_TRANSFORM_KEY],
                    "legacy_conversion": True,
                },
                f"{context}: shovel runtime overlay",
            )
            actors = overlay.get("actors")
            if not isinstance(actors, Mapping) or set(actors) != {"shovel"}:
                raise InvalidArtifact(f"{context}: shovel actor overlay coverage differs")
            actor = actors["shovel"]
            if not isinstance(actor, Mapping):
                raise InvalidArtifact(f"{context}: shovel actor evidence is invalid")
            self._require_mapping_values(
                actor,
                {
                    "after_sha256": expected["contact_points_pose_sha256"],
                    "contact_points_pose_count": 1,
                    "scale_preserved": True,
                    "changed_fields": [CONTACT_KEY],
                },
                f"{context}: shovel actor overlay evidence",
            )

    def _validate_result(self, spec: StageSpec, path: Path) -> dict[str, Any]:
        result = _read_json(path)
        expected_head = {
            "schema": RESULT_SCHEMA,
            "stage": spec.name,
            "status": "PASSED",
            "benchmark_adapter": "BiCoord",
            "config_sha256": self.config_hash,
        }
        self._require_mapping_values(result, expected_head, f"{spec.name} result")
        self._validate_artifacts(result)

        if spec.name == "source_preflight":
            preflight = result.get("preflight")
            source = preflight.get("source") if isinstance(preflight, Mapping) else None
            if not isinstance(source, Mapping):
                raise InvalidArtifact("source preflight omitted source provenance")
            self._require_mapping_values(
                source,
                {
                    "care_revision": self.s.care_source_revision,
                    "expected_care_revision": self.s.care_source_revision,
                    "care_tracked_tree_clean": True,
                    "benchmark_revision": BICOORD_CODE_REVISION,
                    "expected_benchmark_revision": BICOORD_CODE_REVISION,
                    "benchmark_tracked_tree_clean": True,
                    "dataset_revision": FORMAL_DATASET_REVISION,
                },
                "source preflight provenance",
            )
            tracked = source.get("tracked_source_contract")
            if not isinstance(tracked, Mapping):
                raise InvalidArtifact("source preflight omitted tracked-source contract")
            self._require_mapping_values(
                tracked,
                {
                    "status": "PASSED",
                    "scope": "tracked_files_only",
                    "untracked_supplemental_assets_allowed": True,
                },
                "tracked source contract",
            )
            for label in ("care", "benchmark"):
                row = tracked.get(label)
                if not isinstance(row, Mapping):
                    raise InvalidArtifact(f"tracked source contract lacks {label}")
                self._require_mapping_values(
                    row,
                    {
                        "status": "CLEAN",
                        "tracked_tree_clean": True,
                        "scope": "git_index_and_worktree_tracked_files",
                        "untracked_files_ignored": True,
                        "tracked_changes": [],
                    },
                    f"tracked {label} source contract",
                )

        if spec.name == "asset_contract":
            from .asset_stage import (
                BICOORD_OBJECTS_SHA256,
                DONOR_METADATA_SHA256,
                PLATE_COLLISION_SHA256,
                PLATE_VISUAL_SHA256,
                PRISTINE_SMALL_METADATA_SHA256,
                ROBOTWIN_OBJECTS_SHA256,
            )

            self._require_mapping_values(
                result,
                {
                    "dataset_archive_sha256": BICOORD_OBJECTS_SHA256,
                    "base_archive_sha256": ROBOTWIN_OBJECTS_SHA256,
                    "contact_points_pose_count": 4,
                    "shovel_contact_points_pose_count": 1,
                    "copied_fields": ["contact_points_pose"],
                    "task_source_modified": False,
                    "upstream_model_modified": False,
                    "normalization_modified": False,
                    "task_asset_references_checked": {
                        "tasks": len(TASKS),
                        "actors": 21,
                        "interactions": 95,
                    },
                    "task_asset_task_count": len(TASKS),
                    "task_asset_actor_reference_count": 21,
                    "task_asset_interaction_reference_count": 95,
                    "task_asset_dynamic_inventory_sha256": TASK_ASSET_DYNAMIC_INVENTORY_SHA256,
                    "task_asset_unresolved_inventory_sha256": TASK_ASSET_UNRESOLVED_INTERACTION_INVENTORY_SHA256,
                },
                "asset contract result",
            )
            if (
                not isinstance(result.get("shovel_metadata_sha256"), str)
                or result.get("shovel_contact_points_pose_sha256")
                != SHOVEL_CONTACT_POINTS_POSE_SHA256
                or result.get("shovel_contact_points_pose_count") != 1
            ):
                raise InvalidArtifact("asset contract result shovel overlay/hash differs")
            receipt_path = _canonical_stage_path(
                result.get("asset_contract"),
                self.s.run
                / "artifacts"
                / "asset_contract"
                / "asset_contract.json",
                label="asset contract receipt",
            )
            if (
                not receipt_path.is_file()
                or result.get("asset_contract_sha256") != _sha256(receipt_path)
            ):
                raise InvalidArtifact("asset contract receipt/hash differs")
            receipt = _read_json(receipt_path)
            self._require_mapping_values(
                receipt,
                {
                    "schema": "before-we-act.bicoord.asset-contract/1",
                    "status": "PASSED",
                    "dataset_repo_id": FORMAL_DATASET_REPO,
                    "dataset_revision": FORMAL_DATASET_REVISION,
                    "benchmark_revision": BICOORD_CODE_REVISION,
                    "tasks": list(TASKS),
                    "supplemental_assets_installed": True,
                    "benchmark_tracked_source_modified": False,
                    "task_source_modified": False,
                    "upstream_model_modified": False,
                    "normalization_modified": False,
                },
                "asset contract receipt",
            )
            plate = receipt.get("plate_overlay")
            if not isinstance(plate, Mapping):
                raise InvalidArtifact("asset contract omitted plate overlay")
            overlay_path = _canonical_stage_path(
                plate.get("overlay_metadata"),
                self.s.run
                / "artifacts"
                / "asset_contract"
                / "overlay"
                / "003_plate"
                / "model_data0.json",
                label="plate runtime overlay",
            )
            self._require_mapping_values(
                plate,
                {
                    "copied_fields": ["contact_points_pose"],
                    "small_scale_preserved": True,
                    "contact_points_pose_count": 4,
                    "source_small_metadata_sha256": (
                        PRISTINE_SMALL_METADATA_SHA256
                    ),
                    "pristine_small_metadata_sha256": (
                        PRISTINE_SMALL_METADATA_SHA256
                    ),
                    "donor_metadata_expected_sha256": DONOR_METADATA_SHA256,
                    "donor_metadata_sha256": DONOR_METADATA_SHA256,
                    "task_source_modified": False,
                    "planner_modified": False,
                    "model_modified": False,
                    "normalization_modified": False,
                    "benchmark_asset_source_modified": False,
                    "mutation_scope": "run_artifact_and_actor_config_in_memory_only",
                },
                "plate overlay receipt",
            )
            plate_source_path = _canonical_stage_path(
                plate.get("source_small_metadata"),
                self.s.benchmark_repo
                / "assets"
                / "objects"
                / "003_plate"
                / "model_data0.json",
                label="pristine plate source metadata",
            )
            plate_donor_path = _canonical_stage_path(
                plate.get("large_metadata"),
                self.s.benchmark_repo
                / "assets"
                / "objects"
                / "003_plate_large"
                / "model_data0.json",
                label="plate contact donor metadata",
            )
            if (
                not plate_source_path.is_file()
                or _sha256(plate_source_path) != PRISTINE_SMALL_METADATA_SHA256
                or not plate_donor_path.is_file()
                or _sha256(plate_donor_path) != DONOR_METADATA_SHA256
                or plate.get("target_contact_points_pose_sha256")
                != plate.get("large_contact_points_pose_sha256")
            ):
                raise InvalidArtifact("plate source/donor contact provenance differs")
            pristine_plate_value = _read_json(plate_source_path)
            effective_plate_value = _read_json(overlay_path)
            if (
                pristine_plate_value.get(CONTACT_KEY) != []
                or {
                    key: value
                    for key, value in effective_plate_value.items()
                    if key != CONTACT_KEY
                }
                != {
                    key: value
                    for key, value in pristine_plate_value.items()
                    if key != CONTACT_KEY
                }
                or canonical_json_sha256(effective_plate_value.get(CONTACT_KEY))
                != plate.get("target_contact_points_pose_sha256")
            ):
                raise InvalidArtifact("plate overlay did not preserve pristine metadata")
            plate_meshes = plate.get("small_meshes")
            donor_meshes = plate.get("large_meshes")
            if not isinstance(plate_meshes, list) or not isinstance(donor_meshes, list):
                raise InvalidArtifact("plate mesh provenance is missing")
            plate_mesh_hashes = {
                str(row.get("relative_path")): str(row.get("sha256"))
                for row in plate_meshes
                if isinstance(row, Mapping)
            }
            donor_mesh_hashes = {
                str(row.get("relative_path")): str(row.get("sha256"))
                for row in donor_meshes
                if isinstance(row, Mapping)
            }
            if plate_mesh_hashes != {
                "collision/base0.glb": PLATE_COLLISION_SHA256,
                "visual/base0.glb": PLATE_VISUAL_SHA256,
            } or donor_mesh_hashes != plate_mesh_hashes:
                raise InvalidArtifact("plate/donor mesh provenance differs")
            if (
                not overlay_path.is_file()
                or plate.get("target_metadata_sha256") != _sha256(overlay_path)
                or result.get("plate_metadata_sha256") != _sha256(overlay_path)
            ):
                raise InvalidArtifact("audited plate runtime overlay/hash differs")
            shovel = receipt.get("shovel_overlay")
            if not isinstance(shovel, Mapping):
                raise InvalidArtifact("asset contract omitted shovel overlay")
            self._require_mapping_values(
                shovel,
                {
                    "status": "PASSED",
                    "modelname": SHOVEL_OBJECT_NAME,
                    "model_id": SHOVEL_MODEL_ID,
                    "pristine_source_metadata_sha256": (
                        PRISTINE_SHOVEL_METADATA_SHA256
                    ),
                    "source_metadata_sha256": PRISTINE_SHOVEL_METADATA_SHA256,
                    "conversion": (
                        "scale(contact_pose) @ trans_matrix -> "
                        "scale(contact_points_pose)"
                    ),
                    "legacy_contact_pose_count": 1,
                    "contact_points_pose_count": 1,
                    "contact_points_pose_sha256": (
                        SHOVEL_CONTACT_POINTS_POSE_SHA256
                    ),
                    "copied_fields": [CONTACT_KEY],
                    "added_fields": [CONTACT_KEY],
                    "derived_fields": [CONTACT_KEY],
                    "source_fields": [LEGACY_CONTACT_KEY, LEGACY_TRANSFORM_KEY],
                    "preserved_fields": "all_except_contact_points_pose",
                    "changed_fields": [CONTACT_KEY],
                    "scale": list(DEFAULT_SHOVEL_SCALE),
                    "scale_preserved": True,
                    "task_source_modified": False,
                    "planner_modified": False,
                    "model_modified": False,
                    "normalization_modified": False,
                    "benchmark_asset_source_modified": False,
                    "mutation_scope": (
                        "run_artifact_and_actor_config_in_memory_only"
                    ),
                },
                "shovel overlay receipt",
            )
            equivalence_error = shovel.get("max_scale_equivalence_error")
            if (
                isinstance(equivalence_error, bool)
                or not isinstance(equivalence_error, (int, float))
                or not 0.0 <= float(equivalence_error) <= 1e-12
            ):
                raise InvalidArtifact(
                    "shovel overlay scale-equivalence proof differs"
                )
            shovel_overlay_path = _canonical_stage_path(
                shovel.get("overlay_metadata"),
                self.s.run
                / "artifacts"
                / "asset_contract"
                / "overlay"
                / SHOVEL_OBJECT_NAME
                / SHOVEL_METADATA_NAME,
                label="shovel runtime overlay",
            )
            if (
                not shovel_overlay_path.is_file()
                or shovel.get("target_metadata_sha256")
                != _sha256(shovel_overlay_path)
                or result.get("shovel_metadata_sha256")
                != _sha256(shovel_overlay_path)
                or result.get("shovel_contact_points_pose_sha256")
                != SHOVEL_CONTACT_POINTS_POSE_SHA256
            ):
                raise InvalidArtifact("audited shovel runtime overlay/hash differs")
            shovel_source_path = _canonical_stage_path(
                shovel.get("source_metadata"),
                self.s.benchmark_repo
                / "assets"
                / "objects"
                / SHOVEL_OBJECT_NAME
                / SHOVEL_METADATA_NAME,
                label="pristine shovel source metadata",
            )
            if (
                not shovel_source_path.is_file()
                or _sha256(shovel_source_path)
                != PRISTINE_SHOVEL_METADATA_SHA256
            ):
                raise InvalidArtifact("pristine shovel source metadata/hash differs")
            pristine_shovel = _read_json(shovel_source_path)
            effective_shovel = _read_json(shovel_overlay_path)
            if CONTACT_KEY in pristine_shovel:
                raise InvalidArtifact(
                    "pristine shovel source unexpectedly has current contact metadata"
                )
            if (
                effective_shovel.get(LEGACY_CONTACT_KEY)
                != pristine_shovel.get(LEGACY_CONTACT_KEY)
                or effective_shovel.get(LEGACY_TRANSFORM_KEY)
                != pristine_shovel.get(LEGACY_TRANSFORM_KEY)
                or {
                    key: value
                    for key, value in effective_shovel.items()
                    if key != CONTACT_KEY
                }
                != pristine_shovel
            ):
                raise InvalidArtifact(
                    "shovel overlay did not preserve the pristine legacy record"
                )
            effective_contacts = effective_shovel.get(CONTACT_KEY)
            if (
                not isinstance(effective_contacts, list)
                or len(effective_contacts) != 1
                or canonical_json_sha256(effective_contacts)
                != SHOVEL_CONTACT_POINTS_POSE_SHA256
            ):
                raise InvalidArtifact("shovel overlay converted pose differs")
            identity = shovel.get("asset_identity")
            if not isinstance(identity, Mapping):
                raise InvalidArtifact("shovel overlay omitted asset identity")
            self._require_mapping_values(
                identity,
                {
                    "object": SHOVEL_OBJECT_NAME,
                    "model_id": SHOVEL_MODEL_ID,
                    "source_metadata": str(shovel_source_path),
                    "source_metadata_sha256": PRISTINE_SHOVEL_METADATA_SHA256,
                    "mesh_and_metadata_identity": "PASSED",
                },
                "shovel asset identity",
            )
            mesh_rows = identity.get("meshes")
            expected_mesh_rows = (
                (
                    "collision/base3.glb",
                    SHOVEL_COLLISION_BYTES,
                    SHOVEL_COLLISION_SHA256,
                ),
                (
                    "visual/base3.glb",
                    SHOVEL_VISUAL_BYTES,
                    SHOVEL_VISUAL_SHA256,
                ),
            )
            if not isinstance(mesh_rows, list) or len(mesh_rows) != 2:
                raise InvalidArtifact("shovel mesh identity coverage differs")
            for row, (relative, expected_bytes, expected_sha) in zip(
                mesh_rows, expected_mesh_rows, strict=True
            ):
                if not isinstance(row, Mapping):
                    raise InvalidArtifact("shovel mesh identity row is invalid")
                expected_mesh_path = (
                    self.s.benchmark_repo
                    / "assets"
                    / "objects"
                    / SHOVEL_OBJECT_NAME
                    / relative
                )
                mesh_path = _canonical_stage_path(
                    row.get("path"),
                    expected_mesh_path,
                    label=f"shovel {relative} mesh",
                )
                self._require_mapping_values(
                    row,
                    {
                        "relative_path": relative,
                        "path": str(mesh_path),
                        "bytes": expected_bytes,
                        "sha256": expected_sha,
                    },
                    f"shovel {relative} mesh identity",
                )
                if (
                    not mesh_path.is_file()
                    or mesh_path.stat().st_size != expected_bytes
                    or _sha256(mesh_path) != expected_sha
                ):
                    raise InvalidArtifact(f"shovel {relative} mesh/hash differs")
            task_audit = receipt.get("task_asset_audit")
            if not isinstance(task_audit, Mapping):
                raise InvalidArtifact("18-task object point-index audit did not pass")
            self._require_mapping_values(
                task_audit,
                {
                    "schema": "before-we-act.bicoord.task-asset-audit/1",
                    "status": "PASSED",
                    "tasks": list(TASKS),
                    "task_count": len(TASKS),
                    "actor_reference_count": 21,
                    "interaction_reference_count": 95,
                    "references_checked": {
                        "tasks": len(TASKS),
                        "actors": 21,
                        "interactions": 95,
                    },
                    "dynamic_item_count": TASK_ASSET_DYNAMIC_ITEM_COUNT,
                    "dynamic_inventory_sha256": TASK_ASSET_DYNAMIC_INVENTORY_SHA256,
                    "unresolved_interaction_count": TASK_ASSET_UNRESOLVED_INTERACTION_COUNT,
                    "unresolved_interaction_inventory_sha256": TASK_ASSET_UNRESOLVED_INTERACTION_INVENTORY_SHA256,
                    "violations": [],
                    "expected_pristine_defect_count": 0,
                    "unexpected_violation_count": 0,
                    "metadata_override_count": 2,
                    "metadata_override_keys": [
                        {"modelname": "003_plate", "model_id": 0},
                        {
                            "modelname": SHOVEL_OBJECT_NAME,
                            "model_id": SHOVEL_MODEL_ID,
                        },
                    ],
                    "read_only_benchmark": True,
                    "benchmark_files_written": False,
                },
                "18-task object point-index audit",
            )
            reports = task_audit.get("task_reports")
            if (
                not isinstance(reports, list)
                or len(reports) != len(TASKS)
                or [row.get("task") for row in reports if isinstance(row, Mapping)]
                != list(TASKS)
                or any(
                    not isinstance(row, Mapping)
                    or row.get("status") != "PASSED"
                    or row.get("violations") != []
                    for row in reports
                )
            ):
                raise InvalidArtifact("18-task object point-index report coverage differs")
            overrides = task_audit.get("metadata_overrides")
            if not isinstance(overrides, list) or len(overrides) != 2:
                raise InvalidArtifact("plate/shovel metadata override provenance is missing")
            override_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
            for item in overrides:
                if not isinstance(item, Mapping) or not isinstance(item.get("key"), Mapping):
                    raise InvalidArtifact("metadata override provenance is invalid")
                key = item["key"]
                modelname = key.get("modelname")
                model_id = key.get("model_id")
                if (
                    not isinstance(modelname, str)
                    or isinstance(model_id, bool)
                    or not isinstance(model_id, int)
                ):
                    raise InvalidArtifact("metadata override key is invalid")
                identity = (modelname, model_id)
                if identity in override_by_key:
                    raise InvalidArtifact("duplicate metadata override provenance key")
                override_by_key[identity] = item
            if set(override_by_key) != {
                ("003_plate", 0),
                (SHOVEL_OBJECT_NAME, SHOVEL_MODEL_ID),
            }:
                raise InvalidArtifact("metadata override provenance keys differ")
            override = override_by_key[("003_plate", 0)]
            expected_source = overlay_path.resolve()
            self._require_mapping_values(
                override,
                {
                    "key": {"modelname": "003_plate", "model_id": 0},
                    "status": "USED",
                    "source_type": "file",
                    "source_path": str(expected_source),
                    "source_sha256": _sha256(expected_source),
                    "pristine_source_sha256": plate.get("source_small_metadata_sha256"),
                    "contract_status": "PASSED",
                    "contract_type": "contact_points_pose_only",
                    "error": None,
                    "used_by_actor_count": 2,
                    "used_by_interaction_count": 4,
                },
                "plate metadata override provenance",
            )
            for field_name in ("used_by_actors", "used_by_interactions"):
                rows = override.get(field_name)
                if not isinstance(rows, list):
                    raise InvalidArtifact(
                        f"plate metadata override {field_name} is missing"
                    )
            actor_rows = override["used_by_actors"]
            if {
                (row.get("task"), row.get("actor"))
                for row in actor_rows
                if isinstance(row, Mapping)
            } != {
                ("place_plate_and_cup", "plate"),
                ("place_plate_and_cup", "plate_2"),
            }:
                raise InvalidArtifact("plate metadata override actor provenance differs")
            interaction_rows = override["used_by_interactions"]
            observed_interactions = {
                (
                    row.get("task"),
                    row.get("actor"),
                    row.get("kind"),
                    tuple(row.get("index_values") or ()),
                )
                for row in interaction_rows
                if isinstance(row, Mapping)
            }
            if observed_interactions != {
                ("place_plate_and_cup", "plate", "grasp_actor", (2,)),
                ("place_plate_and_cup", "plate", "place_actor", (0,)),
                ("place_plate_and_cup", "plate_2", "grasp_actor", (2,)),
                ("place_plate_and_cup", "plate_2", "place_actor", (0,)),
            }:
                raise InvalidArtifact(
                    "plate metadata override interaction provenance differs"
                )
            shovel_override = override_by_key[(SHOVEL_OBJECT_NAME, SHOVEL_MODEL_ID)]
            self._require_mapping_values(
                shovel_override,
                {
                    "key": {
                        "modelname": SHOVEL_OBJECT_NAME,
                        "model_id": SHOVEL_MODEL_ID,
                    },
                    "status": "USED",
                    "source_type": "file",
                    "source_path": str(shovel_overlay_path.resolve()),
                    "source_sha256": _sha256(shovel_overlay_path),
                    "pristine_source_sha256": PRISTINE_SHOVEL_METADATA_SHA256,
                    "contract_status": "PASSED",
                    "error": None,
                    "used_by_actor_count": 1,
                    "used_by_interaction_count": 1,
                    "contract_type": "derived_legacy_contact",
                },
                "shovel metadata override provenance",
            )
            shovel_actor_rows = shovel_override.get("used_by_actors")
            if not isinstance(shovel_actor_rows, list) or {
                (row.get("task"), row.get("actor"))
                for row in shovel_actor_rows
                if isinstance(row, Mapping)
            } != {("sweep_block", "shovel")}:
                raise InvalidArtifact("shovel metadata override actor provenance differs")
            shovel_interaction_rows = shovel_override.get("used_by_interactions")
            if not isinstance(shovel_interaction_rows, list) or {
                (
                    row.get("task"),
                    row.get("actor"),
                    row.get("kind"),
                    tuple(row.get("index_values") or ()),
                )
                for row in shovel_interaction_rows
                if isinstance(row, Mapping)
            } != {("sweep_block", "shovel", "grasp_actor", (0,))}:
                raise InvalidArtifact(
                    "shovel metadata override interaction provenance differs"
                )
            provenance = shovel_override.get("contract_provenance")
            if not isinstance(provenance, Mapping):
                raise InvalidArtifact("shovel legacy conversion provenance is missing")
            self._require_mapping_values(
                provenance,
                {
                    "contact_points_pose_sha256": SHOVEL_CONTACT_POINTS_POSE_SHA256,
                    "derived_fields": [CONTACT_KEY],
                    "source_fields": [LEGACY_CONTACT_KEY, LEGACY_TRANSFORM_KEY],
                    "scale_preserved": True,
                    "max_scale_equivalence_error": 0.0,
                    "legacy_fields_preserved": True,
                },
                "shovel legacy conversion provenance",
            )

        if spec.result_kind == "dataset":
            if int(result.get("episodes", -1)) != FORMAL_EPISODES:
                raise InvalidArtifact("dataset audit must cover all 1800 demonstrations")
            if result.get("episodes_per_task") != {
                task: FORMAL_EPISODES_PER_TASK for task in TASKS
            }:
                raise InvalidArtifact("dataset per-task episode counts differ")
            if result.get("held_out_episodes") != 0:
                raise InvalidArtifact("formal training must use all demonstrations")
            if result.get("dataset_revision") != FORMAL_DATASET_REVISION:
                raise InvalidArtifact("dataset revision differs")
            numeric = result.get("numeric_contract")
            expected_numeric = {
                "state_dim": 7,
                "action_dim": 7,
                "source_frequency_hz": SOURCE_FREQUENCY_HZ,
                "alignment": "observation_row_t_to_action_row_t_plus_1",
                "action_encoding": ACTION_ENCODING,
                "state_clipping": False,
                "action_clipping": False,
                "normalization_from_all_data": True,
            }
            if not isinstance(numeric, Mapping):
                raise InvalidArtifact("dataset audit lacks numeric_contract")
            self._require_mapping_values(numeric, expected_numeric, "numeric_contract")

        if spec.result_kind == "seed_manifest":
            overlay_expectations = self._asset_runtime_expectations()
            if result.get("policy_independent") is not True:
                raise InvalidArtifact("expert seed manifest is not policy-independent")
            if result.get("learned_policy_used") is not False:
                raise InvalidArtifact("expert seed discovery imported a learned policy")
            if result.get("closed_loop_policy_results_used") is not False:
                raise InvalidArtifact("expert seed discovery used policy results")
            if result.get("structural_error_streak_limit") != 3:
                raise InvalidArtifact("expert seed structural-error limit differs")
            if int(result.get("episodes_per_task", -1)) not in {1, VALIDATION_EPISODES}:
                raise InvalidArtifact("expert seed manifest episode count differs")
            valid = result.get("valid_seeds")
            if not isinstance(valid, Mapping) or tuple(valid) != TASKS:
                raise InvalidArtifact("expert seed manifest task coverage/order differs")
            required_count = int(result["episodes_per_task"])
            for task in TASKS:
                values = valid[task]
                if not isinstance(values, list) or len(values) != required_count:
                    raise InvalidArtifact(f"{task}: expert seed count differs")
                if len(set(values)) != len(values) or any(int(seed) < 0 for seed in values):
                    raise InvalidArtifact(f"{task}: expert seed list is malformed")
            manifest = Path(str(result.get("seed_manifest", "")))
            expected_sha = result.get("seed_manifest_sha256")
            if not manifest.is_file() or not isinstance(expected_sha, str) or _sha256(manifest) != expected_sha:
                raise InvalidArtifact("expert seed manifest artifact/hash differs")
            manifest_value = _read_json(manifest)
            self._require_mapping_values(
                manifest_value,
                {
                    "valid_seeds": valid,
                    "structural_error_streak_limit": 3,
                    "learned_policy_used": False,
                    "closed_loop_policy_results_used": False,
                },
                "expert seed manifest diagnostics",
            )
            diagnostic_maps: dict[str, Mapping[str, Any]] = {}
            for field_name in (
                "exception_type_counts",
                "exception_counts",
                "structural_exception_type_counts",
                "structural_exception_counts",
            ):
                value = manifest_value.get(field_name)
                if not isinstance(value, Mapping) or tuple(value) != TASKS:
                    raise InvalidArtifact(
                        f"expert seed manifest {field_name} task coverage differs"
                    )
                if result.get(field_name) != value:
                    raise InvalidArtifact(
                        f"expert seed result differs from manifest at {field_name}"
                    )
                diagnostic_maps[field_name] = value
            progress_paths = manifest_value.get("progress_receipts")
            progress_hashes = manifest_value.get("progress_receipt_sha256")
            seed_receipts = manifest_value.get("seed_receipts")
            seed_hashes = manifest_value.get("seed_receipts_sha256")
            attempts = manifest_value.get("attempts")
            for mapping, label in (
                (progress_paths, "progress receipts"),
                (progress_hashes, "progress hashes"),
                (seed_receipts, "attempt receipts"),
                (seed_hashes, "attempt receipt hashes"),
                (attempts, "attempt rows"),
            ):
                if not isinstance(mapping, Mapping) or tuple(mapping) != TASKS:
                    raise InvalidArtifact(
                        f"expert seed manifest {label} task coverage differs"
                    )
            for task in TASKS:
                progress_path = Path(str(progress_paths[task]))
                if (
                    not progress_path.is_file()
                    or progress_hashes[task] != _sha256(progress_path)
                ):
                    raise InvalidArtifact(f"{task}: expert seed progress changed")
                progress = _read_json(progress_path)
                self._require_mapping_values(
                    progress,
                    {
                        "schema": "before-we-act.bicoord.expert-seed-progress/1",
                        "status": "PASSED",
                        "task": task,
                        "valid_seeds": valid[task],
                        "structural_error_streak_limit": 3,
                        "policy_independent": True,
                    },
                    f"{task} expert seed progress",
                )
                task_receipts = seed_receipts[task]
                task_hashes = seed_hashes[task]
                task_attempts = attempts[task]
                if (
                    not isinstance(task_receipts, list)
                    or not isinstance(task_hashes, list)
                    or not isinstance(task_attempts, list)
                    or len(task_receipts) != len(task_attempts)
                    or len(task_hashes) != len(task_attempts)
                    or not task_attempts
                ):
                    raise InvalidArtifact(f"{task}: expert attempt evidence differs")
                if task in {"place_plate_and_cup", "sweep_block"}:
                    for attempt_index, attempt_row in enumerate(
                        task_attempts, start=1
                    ):
                        if not isinstance(attempt_row, Mapping):
                            raise InvalidArtifact(
                                f"{task}: aggregate attempt {attempt_index} is invalid"
                            )
                        self._validate_seed_asset_overlay(
                            task,
                            attempt_row,
                            overlay_expectations,
                            context=f"{task} aggregate attempt {attempt_index}",
                        )
                task_type_counts, task_error_counts = (
                    self._validated_seed_exception_diagnostics(
                        diagnostic_maps["exception_type_counts"][task],
                        diagnostic_maps["exception_counts"][task],
                        context=f"{task} aggregate exception diagnostics",
                    )
                )
                task_structural_type_counts, task_structural_error_counts = (
                    self._validated_seed_exception_diagnostics(
                        diagnostic_maps["structural_exception_type_counts"][task],
                        diagnostic_maps["structural_exception_counts"][task],
                        context=f"{task} aggregate structural exception diagnostics",
                    )
                )
                derived_type_counts, derived_error_counts = (
                    self._seed_attempt_exception_diagnostics(
                        task_attempts,
                        structural_only=False,
                        context=f"{task} aggregate exception attempt evidence",
                    )
                )
                derived_structural_type_counts, derived_structural_error_counts = (
                    self._seed_attempt_exception_diagnostics(
                        task_attempts,
                        structural_only=True,
                        context=f"{task} aggregate structural attempt evidence",
                    )
                )
                if (
                    task_type_counts != derived_type_counts
                    or task_error_counts != derived_error_counts
                    or task_structural_type_counts
                    != derived_structural_type_counts
                    or task_structural_error_counts
                    != derived_structural_error_counts
                ):
                    raise InvalidArtifact(
                        f"{task}: aggregate exception diagnostics differ from attempts"
                    )
                self._require_mapping_values(
                    progress,
                    {
                        "exception_type_counts": task_type_counts,
                        "exception_counts": task_error_counts,
                        "structural_exception_type_counts": (
                            task_structural_type_counts
                        ),
                        "structural_exception_counts": (
                            task_structural_error_counts
                        ),
                    },
                    f"{task} aggregate progress diagnostics",
                )
                for index, (attempt_path_raw, attempt_hash, attempt_row) in enumerate(
                    zip(task_receipts, task_hashes, task_attempts, strict=True), start=1
                ):
                    attempt_path = Path(str(attempt_path_raw))
                    if (
                        not attempt_path.is_file()
                        or not isinstance(attempt_hash, str)
                        or _sha256(attempt_path) != attempt_hash
                    ):
                        raise InvalidArtifact(
                            f"{task}: expert attempt receipt {index} changed"
                        )
                    attempt = _read_json(attempt_path)
                    if (
                        attempt.get("schema")
                        != "before-we-act.bicoord.expert-seed-attempt/1"
                        or int(attempt.get("attempt_index", -1)) != index
                        or attempt.get("task") != task
                        or attempt.get("row") != attempt_row
                    ):
                        raise InvalidArtifact(
                            f"{task}: expert attempt receipt {index} differs"
                        )
                    prefix_type_counts, prefix_error_counts = (
                        self._seed_attempt_exception_diagnostics(
                            task_attempts[:index],
                            structural_only=False,
                            context=f"{task} aggregate attempt receipt {index}",
                        )
                    )
                    prefix_structural_type_counts, prefix_structural_error_counts = (
                        self._seed_attempt_exception_diagnostics(
                            task_attempts[:index],
                            structural_only=True,
                            context=(
                                f"{task} aggregate structural attempt receipt {index}"
                            ),
                        )
                    )
                    self._require_mapping_values(
                        attempt,
                        {
                            "exception_type_counts": prefix_type_counts,
                            "exception_counts": prefix_error_counts,
                            "structural_exception_type_counts": (
                                prefix_structural_type_counts
                            ),
                            "structural_exception_counts": (
                                prefix_structural_error_counts
                            ),
                            "structural_error_streak_limit": 3,
                        },
                        f"{task} aggregate attempt receipt {index} diagnostics",
                    )

        model_kinds = {
            "training",
            "training_grid",
            "selection",
            "validation",
            "validation20",
            "smoke_validation",
            "cache",
            "branch",
            "gate",
        }
        if spec.result_kind in model_kinds:
            self._validate_model_contract(result)
        if spec.result_kind == "smoke_validation":
            self._validate_task_results(
                result,
                1,
                require_success=False,
                max_steps_by_task=SMOKE_MAX_STEPS,
            )
        elif spec.result_kind == "validation20":
            self._validate_task_results(result, VALIDATION_EPISODES, require_success=False)
        elif spec.result_kind == "validation":
            probe_episodes = int(result.get("episodes_per_task", 1))
            if probe_episodes < 1:
                raise InvalidArtifact("probe episodes must be positive")
            self._validate_task_results(result, probe_episodes, require_success=True)
        if spec.name == "bcore_validation20":
            self._validate_task_results(result, VALIDATION_EPISODES, require_success=True)
        if spec.name == "paired_validation_smoke":
            self._validate_paired_result(
                result, episodes=1, max_steps_by_task=SMOKE_MAX_STEPS
            )
        elif spec.name == "paired_validation20":
            self._validate_paired_result(result, episodes=VALIDATION_EPISODES)
        if spec.name == "bcore_select":
            if result.get("closed_loop_results_used_for_selection") is not False:
                raise InvalidArtifact("B-core selection must be offline")
        if spec.name == "offline_selection_calibration":
            if result.get("closed_loop_results_used_for_selection") is not False:
                raise InvalidArtifact("CARE selection/calibration must be offline")
        if spec.name in {"branch_smoke", "branch_collection"}:
            self._validate_branch_result(spec, result)
        if spec.name in {"branch_signal_gate_smoke", "branch_signal_gate"} and result.get("downstream_authorized") is not True:
            raise Blocked("counterfactual/event signal audit did not authorize belief training")
        if spec.name == "b0h_formal":
            self._require_mapping_values(
                result,
                {
                    "update": self.s.b0h_updates,
                    "target_updates": self.s.b0h_updates,
                    "effective_batch": GLOBAL_BATCH,
                    "world_size": 4,
                    "local_batch": GLOBAL_BATCH // 4,
                    "all_1800_demonstrations": True,
                    "held_out_demonstrations": 0,
                    "teacher_present": False,
                    "strictly_decentralized": True,
                    "shared_weights": True,
                },
                "formal B0-H result",
            )
        if spec.name in {"branch_prepare_smoke", "branch_prepare"}:
            expected_families = (
                len(TASKS)
                if spec.name == "branch_prepare_smoke"
                else self.s.families_per_task * len(TASKS)
            )
            self._require_mapping_values(
                result,
                {
                    "families": expected_families,
                    "held_out_families": 0,
                    "provider_policy": "B-core/TUNE",
                    "all_families_for_training": True,
                },
                f"{spec.name} result",
            )
        if spec.name in {"bcore_smoke_cache", "bcore_cache", "dino_cache"}:
            if result.get("cache_complete") is not True:
                raise InvalidArtifact(f"{spec.name} aggregate cache is incomplete")
        if spec.name == "bcore_train_3seeds":
            if result.get("seeds") != list(BCORE_SEEDS):
                raise InvalidArtifact("formal B-core seed list differs")
            if int(result.get("updates_per_seed", -1)) != self.s.bcore_updates:
                raise InvalidArtifact("formal B-core update count differs")
        if spec.name == "belief_train":
            if result.get("variants") != list(CARE_VARIANTS):
                raise InvalidArtifact("formal CARE variant list differs")
            if result.get("seeds") != list(CARE_SEEDS):
                raise InvalidArtifact("formal CARE seed list differs")
            if int(result.get("updates_per_run", -1)) != self.s.care_updates:
                raise InvalidArtifact("formal CARE update count differs")
            self._require_mapping_values(
                result,
                {
                    "deployment_main_workers": len(CARE_VARIANTS)
                    * len(CARE_SEEDS),
                    "oof_shadow_workers": len(CARE_OOF_FOLDS),
                    "oof_shadow_variant": CARE_OOF_VARIANT,
                    "oof_shadow_seed": CARE_OOF_SEED,
                    "oof_shadow_folds": list(CARE_OOF_FOLDS),
                    "oof_shadow_deployment_candidates": False,
                },
                "formal CARE aggregate",
            )
        return result

    def _validate_receipt(self, stage_name: str) -> dict[str, Any]:
        spec = STAGES[stage_name]
        path = self.receipt_path(stage_name)
        receipt = _read_json(path)
        expected = {
            "schema": RECEIPT_SCHEMA,
            "stage": stage_name,
            "status": "COMPLETED",
            "config_sha256": self.config_hash,
        }
        self._require_mapping_values(receipt, expected, f"{stage_name} receipt")
        dependency_hashes = {
            name: _sha256(self.receipt_path(name)) for name in spec.dependencies
        }
        if receipt.get("dependency_receipt_sha256") != dependency_hashes:
            raise InvalidArtifact(f"{stage_name} dependency receipts changed")
        result_path = self.result_path(stage_name)
        if not result_path.is_file():
            raise InvalidArtifact(f"{stage_name} result is missing")
        if receipt.get("result_sha256") != _sha256(result_path):
            raise InvalidArtifact(f"{stage_name} result receipt hash differs")
        self._validate_result(spec, result_path)
        return receipt

    def _base_command(
        self,
        spec: StageSpec,
        candidate: Path,
        extra: Sequence[str] = (),
    ) -> list[str]:
        if spec.module_key is None:
            raise ValueError(f"stage {spec.name} has no adapter")
        module = self.s.modules[spec.module_key]
        if not _module_available(module):
            raise Blocked(f"{spec.name} requires unavailable adapter {module}")
        # B0-H is the only adapter here whose upstream trainer is genuinely
        # data-parallel.  Merely exposing four devices would still execute a
        # one-rank job, so launch its unchanged trainer through torchrun.  The
        # adapter accepts the same generic stage CLI and only rank zero emits
        # the result receipt.
        launcher = [self.s.python, "-u"]
        if spec.module_key == "b0h_train":
            launcher += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                "4",
                "-m",
                module,
            ]
        else:
            launcher += ["-m", module]
        stage_extra: list[str] = []
        if spec.name == "bcore_smoke_cache":
            # Do not rely on cache_bcore's path-name compatibility fallback.
            # The supervisor must explicitly request the isolated 18-episode
            # smoke cache and bind the emitted worker stage identity.
            stage_extra += ["--smoke", "--stage-name", "bcore_smoke_cache"]
        elif spec.name == "bcore_cache":
            stage_extra += ["--stage-name", "bcore_cache"]
        return [
            *launcher,
            spec.operation,
            "--repo",
            str(self.s.repo),
            "--benchmark-repo",
            str(self.s.benchmark_repo),
            "--dataset",
            str(self.s.dataset),
            "--run",
            str(self.s.run),
            "--dino-model",
            str(self.s.dino_model),
            "--result",
            str(candidate),
            "--config-sha256",
            self.config_hash,
            "--auto-resume",
            *stage_extra,
            *extra,
        ]

    def _single_action(self, spec: StageSpec, candidate: Path) -> None:
        extras: list[str] = []
        if spec.name in {"b0h_smoke_train", "bcore_smoke_train", "belief_smoke_train"}:
            extras += ["--updates", "5", "--global-batch", str(GLOBAL_BATCH), "--smoke"]
        elif spec.name == "b0h_formal":
            extras += ["--updates", str(self.s.b0h_updates), "--global-batch", str(GLOBAL_BATCH)]
        elif spec.name in {"branch_prepare_smoke", "branch_prepare"}:
            extras += ["--use-all-families", "--held-out-families", "0"]
        elif spec.name == "offline_selection_calibration":
            extras += ["--offline-only", "--closed-loop-selection", "false"]
        if spec.gpu_plan == "cpu":
            # Empty CUDA visibility prevents a metadata/download process from
            # claiming a GPU behind the scheduler's back.  It is still
            # scheduler-owned so signals never leave an orphan process.
            self.scheduler.run(
                spec.name,
                self._base_command(spec, candidate, extras),
                (),
                self.s.run / "logs" / f"{spec.name}.log",
            )
            return
        self.scheduler.run(
            spec.name,
            self._base_command(spec, candidate, extras),
            (0, 1, 2, 3),
            self.s.run / "logs" / f"{spec.name}.log",
        )

    def _aggregate_worker_results(
        self,
        spec: StageSpec,
        candidate: Path,
        worker_paths: Sequence[Path],
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        workers = []
        artifacts: list[dict[str, str]] = []
        for path in worker_paths:
            row = self._validated_worker_result(spec, path)
            workers.append(row)
            artifacts.extend(row.get("artifacts", []))
        if spec.name in {"dino_cache", "bcore_smoke_cache", "bcore_cache"}:
            cache_extra = self._validate_cache_workers(spec, workers)
        else:
            cache_extra = {}
        if spec.result_kind == "training_grid" or spec.name in {
            "bcore_train_3seeds",
            "belief_train",
        }:
            self._validate_training_workers(spec, workers)
        value: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "stage": spec.name,
            "status": "PASSED",
            "benchmark_adapter": "BiCoord",
            "config_sha256": self.config_hash,
            "model_contract": dict(MODEL_CONTRACT),
            "workers": workers,
            "artifacts": self._deduplicate_artifacts(artifacts),
            "completed_at": _utc_now(),
        }
        value.update(cache_extra)
        if extra:
            value.update(extra)
        if spec.name in {"branch_smoke", "branch_collection"}:
            # Branch validation is deliberately performed on the aggregate
            # view so a missing family cannot be hidden in an individual
            # worker receipt.
            self._validate_branch_result(spec, value)
        _atomic_json(candidate, value)

    def _seed_wave_action(self, spec: StageSpec, candidate: Path) -> None:
        worker_root = self.s.run / "worker_results" / spec.name
        worker_paths: list[Path] = []
        jobs = []
        for gpu, seed in enumerate(BCORE_SEEDS):
            path = worker_root / f"seed_{seed}.json"
            worker_paths.append(path)
            jobs.append(
                (
                    f"{spec.name}_seed_{seed}",
                    self._base_command(
                        spec,
                        path,
                        (
                            "--seed",
                            str(seed),
                            "--updates",
                            str(self.s.bcore_updates),
                            "--global-batch",
                            str(GLOBAL_BATCH),
                        ),
                    ),
                    gpu,
                    self.s.run / "logs" / f"{spec.name}_seed_{seed}.log",
                )
            )
        self.scheduler.run_wave(jobs)
        self._aggregate_worker_results(
            spec,
            candidate,
            worker_paths,
            extra={
                "seeds": list(BCORE_SEEDS),
                "updates_per_seed": self.s.bcore_updates,
            },
        )

    def _care_grid_action(self, spec: StageSpec, candidate: Path) -> None:
        work: list[tuple[str, int, int | None, Path, list[str]]] = []
        worker_root = self.s.run / "worker_results" / spec.name
        for variant in CARE_VARIANTS:
            for seed in CARE_SEEDS:
                path = worker_root / "deployment_main" / variant / f"seed_{seed}.json"
                command = self._base_command(
                    spec,
                    path,
                    (
                        "--variant",
                        variant,
                        "--seed",
                        str(seed),
                        "--updates",
                        str(self.s.care_updates),
                        "--global-batch",
                        str(GLOBAL_BATCH),
                    ),
                )
                work.append((variant, seed, None, path, command))
        for fold in CARE_OOF_FOLDS:
            path = worker_root / "oof_shadow" / f"fold_{fold}.json"
            command = self._base_command(
                spec,
                path,
                (
                    "--variant",
                    CARE_OOF_VARIANT,
                    "--seed",
                    str(CARE_OOF_SEED),
                    "--updates",
                    str(self.s.care_updates),
                    "--global-batch",
                    str(GLOBAL_BATCH),
                    "--oof-shadow-fold",
                    str(fold),
                ),
            )
            work.append((CARE_OOF_VARIANT, CARE_OOF_SEED, fold, path, command))
        for first in range(0, len(work), 4):
            wave = work[first : first + 4]
            self.scheduler.run_wave(
                [
                    (
                        (
                            f"belief_{variant}_{seed}"
                            if fold is None
                            else f"belief_oof_shadow_fold_{fold}"
                        ),
                        command,
                        gpu,
                        self.s.run
                        / "logs"
                        / (
                            f"belief_{variant}_{seed}.log"
                            if fold is None
                            else f"belief_oof_shadow_fold_{fold}.log"
                        ),
                    )
                    for gpu, (variant, seed, fold, _path, command) in enumerate(wave)
                ]
            )
        self._aggregate_worker_results(
            spec,
            candidate,
            [path for _variant, _seed, _fold, path, _command in work],
            extra={
                "variants": list(CARE_VARIANTS),
                "seeds": list(CARE_SEEDS),
                "updates_per_run": self.s.care_updates,
                "deployment_main_workers": len(CARE_VARIANTS) * len(CARE_SEEDS),
                "oof_shadow_workers": len(CARE_OOF_FOLDS),
                "oof_shadow_variant": CARE_OOF_VARIANT,
                "oof_shadow_seed": CARE_OOF_SEED,
                "oof_shadow_folds": list(CARE_OOF_FOLDS),
                "oof_shadow_deployment_candidates": False,
            },
        )

    def _sharded_action(self, spec: StageSpec, candidate: Path) -> None:
        worker_root = self.s.run / "worker_results" / spec.name
        paths = [worker_root / f"rank_{rank}.json" for rank in range(4)]
        jobs = []
        for rank in range(4):
            extras = ["--rank", str(rank), "--world-size", "4"]
            if spec.name in {"branch_smoke", "branch_collection"}:
                families = (
                    1 if spec.name == "branch_smoke" else self.s.families_per_task
                )
                extras += [
                    "--families-per-task",
                    str(families),
                    "--branches-per-family",
                    str(BRANCHES_PER_FAMILY),
                ]
                if spec.name == "branch_smoke":
                    extras += ["--smoke"]
            jobs.append(
                (
                    f"{spec.name}_rank_{rank}",
                    self._base_command(spec, paths[rank], extras),
                    rank,
                    self.s.run / "logs" / f"{spec.name}_rank_{rank}.log",
                )
            )
        self.scheduler.run_wave(jobs)
        extra: dict[str, Any] = {}
        if spec.name in {"branch_smoke", "branch_collection"}:
            families = 1 if spec.name == "branch_smoke" else self.s.families_per_task
            extra = {
                "provider_policy": "B-core/TUNE",
                "families": families * len(TASKS),
                "families_per_task": families,
                "branches_per_family": BRANCHES_PER_FAMILY,
                "physical_simulator_outcomes": True,
                "offline_demonstration_error_used": False,
            }
        self._aggregate_worker_results(spec, candidate, paths, extra=extra)

    def _seed_task_queue_action(self, spec: StageSpec, candidate: Path) -> None:
        """Discover expert-valid seeds in four-GPU task waves.

        BiCoord's expert planner uses the simulator/Curobo CUDA path.  Running
        this stage with an empty CUDA lease fails even before policy rollout,
        while running all 18 tasks in one process strands three GPUs.  Each
        worker therefore owns one task and one GPU; this method then builds a
        single immutable all-task manifest consumed by every later rollout.
        """

        worker_root = self.s.run / "worker_results" / spec.name
        results: dict[str, Path] = {}
        episodes = 1 if spec.name == "seed_discovery_smoke" else VALIDATION_EPISODES
        for first in range(0, len(TASKS), 4):
            wave = TASKS[first : first + 4]
            jobs = []
            for gpu, task in enumerate(wave):
                path = worker_root / f"{task}.json"
                results[task] = path
                jobs.append(
                    (
                        f"{spec.name}_{task}",
                        self._base_command(
                            spec,
                            path,
                            ("--task", task, "--episodes", str(episodes)),
                        ),
                        gpu,
                        self.s.run / "logs" / f"{spec.name}_{task}.log",
                    )
                )
            self.scheduler.run_wave(jobs)

        workers: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        valid_seeds: dict[str, list[int]] = {}
        attempts: dict[str, list[dict[str, Any]]] = {}
        exception_type_counts: dict[str, dict[str, int]] = {}
        exception_counts: dict[str, list[dict[str, Any]]] = {}
        structural_exception_type_counts: dict[str, dict[str, int]] = {}
        structural_exception_counts: dict[str, list[dict[str, Any]]] = {}
        progress_receipts: dict[str, str] = {}
        progress_receipt_sha256: dict[str, str] = {}
        seed_receipts: dict[str, list[str]] = {}
        seed_receipts_sha256: dict[str, list[str]] = {}
        overlay_expectations = self._asset_runtime_expectations()
        for task in TASKS:
            row = _read_json(results[task])
            self._require_mapping_values(
                row,
                {
                    "schema": RESULT_SCHEMA,
                    "stage": "seed_discovery_worker",
                    "status": "PASSED",
                    "benchmark_adapter": "BiCoord",
                    "config_sha256": self.config_hash,
                    "task": task,
                    "episodes": episodes,
                    "completed": episodes,
                    "episodes_per_task": episodes,
                    "tasks": [task],
                    "policy_independent": True,
                    "learned_policy_used": False,
                    "closed_loop_policy_results_used": False,
                    "structural_error_streak_limit": 3,
                },
                f"{spec.name} worker {task}",
            )
            self._validate_artifacts(row)
            worker_seeds = row.get("valid_seeds")
            seeds = worker_seeds.get(task) if isinstance(worker_seeds, Mapping) else None
            if (
                not isinstance(seeds, list)
                or len(seeds) != episodes
                or len(set(seeds)) != episodes
                or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
            ):
                raise InvalidArtifact(f"{task}: expert seed worker coverage differs")
            manifest_path = self._artifact_path(
                {"path": row.get("seed_manifest", "")}
            ).resolve()
            digest = row.get("seed_manifest_sha256")
            if not manifest_path.is_file() or digest != _sha256(manifest_path):
                raise InvalidArtifact(f"{task}: expert seed worker manifest differs")
            manifest = _read_json(manifest_path)
            self._require_mapping_values(
                manifest,
                {
                    "schema": "before-we-act.bicoord.expert-seed-manifest/1",
                    "status": "PASSED",
                    "policy_independent": True,
                    "episodes_per_task": episodes,
                    "tasks": [task],
                    "valid_seeds": {task: seeds},
                    "learned_policy_used": False,
                    "closed_loop_policy_results_used": False,
                },
                f"{task} expert seed manifest",
            )
            manifest_attempts = manifest.get("attempts")
            task_attempts = (
                manifest_attempts.get(task)
                if isinstance(manifest_attempts, Mapping)
                else None
            )
            if not isinstance(task_attempts, list) or not task_attempts:
                raise InvalidArtifact(f"{task}: expert seed attempt evidence is missing")
            valid_attempts = [
                int(item.get("seed", -1))
                for item in task_attempts
                if isinstance(item, Mapping) and item.get("valid") is True
            ]
            if valid_attempts[:episodes] != seeds:
                raise InvalidArtifact(f"{task}: selected seeds differ from expert evidence")
            diagnostic_maps: dict[str, Mapping[str, Any]] = {}
            for field_name in (
                "exception_type_counts",
                "exception_counts",
                "structural_exception_type_counts",
                "structural_exception_counts",
            ):
                value = manifest.get(field_name)
                if not isinstance(value, Mapping) or tuple(value) != (task,):
                    raise InvalidArtifact(
                        f"{task}: {field_name} task coverage differs"
                    )
                diagnostic_maps[field_name] = value
            task_type_counts, task_error_counts = (
                self._validated_seed_exception_diagnostics(
                    diagnostic_maps["exception_type_counts"][task],
                    diagnostic_maps["exception_counts"][task],
                    context=f"{task} exception diagnostics",
                )
            )
            task_structural_type_counts, task_structural_error_counts = (
                self._validated_seed_exception_diagnostics(
                    diagnostic_maps["structural_exception_type_counts"][task],
                    diagnostic_maps["structural_exception_counts"][task],
                    context=f"{task} structural exception diagnostics",
                )
            )
            derived_type_counts, derived_error_counts = (
                self._seed_attempt_exception_diagnostics(
                    task_attempts,
                    structural_only=False,
                    context=f"{task} exception attempt evidence",
                )
            )
            derived_structural_type_counts, derived_structural_error_counts = (
                self._seed_attempt_exception_diagnostics(
                    task_attempts,
                    structural_only=True,
                    context=f"{task} structural exception attempt evidence",
                )
            )
            if (
                task_type_counts != derived_type_counts
                or task_error_counts != derived_error_counts
                or task_structural_type_counts != derived_structural_type_counts
                or task_structural_error_counts != derived_structural_error_counts
            ):
                raise InvalidArtifact(
                    f"{task}: exception diagnostics differ from attempt evidence"
                )
            self._require_mapping_values(
                row,
                {
                    "exception_type_counts": {task: task_type_counts},
                    "exception_counts": {task: task_error_counts},
                    "structural_exception_type_counts": {
                        task: task_structural_type_counts
                    },
                    "structural_exception_counts": {
                        task: task_structural_error_counts
                    },
                },
                f"{task} seed worker diagnostics",
            )
            if task in {"place_plate_and_cup", "sweep_block"}:
                for attempt_index, attempt_row in enumerate(task_attempts, start=1):
                    if not isinstance(attempt_row, Mapping):
                        raise InvalidArtifact(
                            f"{task}: attempt {attempt_index} is not an object"
                        )
                    self._validate_seed_asset_overlay(
                        task,
                        attempt_row,
                        overlay_expectations,
                        context=f"{task} attempt {attempt_index}",
                    )
            task_progress = row.get("progress_receipt")
            task_progress_sha = row.get("progress_receipt_sha256")
            progress_path = Path(str(task_progress))
            if (
                not progress_path.is_file()
                or not isinstance(task_progress_sha, str)
                or _sha256(progress_path) != task_progress_sha
                or manifest.get("progress_receipts") != {task: str(progress_path)}
                or manifest.get("progress_receipt_sha256")
                != {task: task_progress_sha}
            ):
                raise InvalidArtifact(f"{task}: expert progress receipt differs")
            progress = _read_json(progress_path)
            self._require_mapping_values(
                progress,
                {
                    "schema": "before-we-act.bicoord.expert-seed-progress/1",
                    "status": "PASSED",
                    "task": task,
                    "valid_seeds": seeds,
                    "attempts_completed": len(task_attempts),
                    "structural_error_streak_limit": 3,
                    "exception_type_counts": task_type_counts,
                    "exception_counts": task_error_counts,
                    "structural_exception_type_counts": (
                        task_structural_type_counts
                    ),
                    "structural_exception_counts": task_structural_error_counts,
                    "policy_independent": True,
                    "learned_policy_used": False,
                },
                f"{task} expert progress",
            )
            task_seed_receipts = manifest.get("seed_receipts", {}).get(task)
            task_seed_hashes = manifest.get("seed_receipts_sha256", {}).get(task)
            if (
                not isinstance(task_seed_receipts, list)
                or not isinstance(task_seed_hashes, list)
                or len(task_seed_receipts) != len(task_attempts)
                or len(task_seed_hashes) != len(task_attempts)
            ):
                raise InvalidArtifact(f"{task}: per-seed receipt coverage differs")
            for index, (attempt_path_raw, attempt_hash, attempt_row) in enumerate(
                zip(
                    task_seed_receipts,
                    task_seed_hashes,
                    task_attempts,
                    strict=True,
                ),
                start=1,
            ):
                attempt_path = Path(str(attempt_path_raw))
                if (
                    not attempt_path.is_file()
                    or _sha256(attempt_path) != attempt_hash
                ):
                    raise InvalidArtifact(
                        f"{task}: per-seed receipt {index} changed"
                    )
                attempt = _read_json(attempt_path)
                if (
                    attempt.get("schema")
                    != "before-we-act.bicoord.expert-seed-attempt/1"
                    or attempt.get("task") != task
                    or int(attempt.get("attempt_index", -1)) != index
                    or attempt.get("row") != attempt_row
                ):
                    raise InvalidArtifact(
                        f"{task}: per-seed receipt {index} differs"
                    )
                prefix_type_counts, prefix_error_counts = (
                    self._seed_attempt_exception_diagnostics(
                        task_attempts[:index],
                        structural_only=False,
                        context=f"{task} attempt receipt {index}",
                    )
                )
                prefix_structural_type_counts, prefix_structural_error_counts = (
                    self._seed_attempt_exception_diagnostics(
                        task_attempts[:index],
                        structural_only=True,
                        context=f"{task} structural attempt receipt {index}",
                    )
                )
                self._require_mapping_values(
                    attempt,
                    {
                        "exception_type_counts": prefix_type_counts,
                        "exception_counts": prefix_error_counts,
                        "structural_exception_type_counts": (
                            prefix_structural_type_counts
                        ),
                        "structural_exception_counts": (
                            prefix_structural_error_counts
                        ),
                        "structural_error_streak_limit": 3,
                    },
                    f"{task} per-seed receipt {index} diagnostics",
                )
            valid_seeds[task] = [int(seed) for seed in seeds]
            attempts[task] = [dict(item) for item in task_attempts]
            exception_type_counts[task] = task_type_counts
            exception_counts[task] = task_error_counts
            structural_exception_type_counts[task] = task_structural_type_counts
            structural_exception_counts[task] = task_structural_error_counts
            progress_receipts[task] = str(progress_path)
            progress_receipt_sha256[task] = task_progress_sha
            seed_receipts[task] = [str(value) for value in task_seed_receipts]
            seed_receipts_sha256[task] = [str(value) for value in task_seed_hashes]
            workers.append(row)
            artifacts.extend(row.get("artifacts", []))

        aggregate_manifest = self.s.run / "artifacts" / spec.name / "seed_manifest.json"
        aggregate_value = {
            "schema": "before-we-act.bicoord.expert-seed-manifest/1",
            "status": "PASSED",
            "policy_independent": True,
            "selection_policy": (
                "official_expert_play_once_then_plan_success_and_check_success"
            ),
            "seed_protocol": "100000*(1+seed_bucket), increment by one",
            "seed_bucket": 0,
            "seed_multiplier": 100_000,
            "episodes_per_task": episodes,
            "tasks": list(TASKS),
            "max_steps": dict(MAX_STEPS),
            "valid_seeds": valid_seeds,
            "attempts": attempts,
            "exception_type_counts": exception_type_counts,
            "exception_counts": exception_counts,
            "structural_exception_type_counts": structural_exception_type_counts,
            "structural_exception_counts": structural_exception_counts,
            "structural_error_streak_limit": 3,
            "progress_receipts": progress_receipts,
            "progress_receipt_sha256": progress_receipt_sha256,
            "seed_receipts": seed_receipts,
            "seed_receipts_sha256": seed_receipts_sha256,
            "learned_policy_used": False,
            "closed_loop_policy_results_used": False,
        }
        _atomic_json(aggregate_manifest, aggregate_value)
        aggregate_status = aggregate_manifest.with_name("status.json")
        _atomic_json(
            aggregate_status,
            {
                "schema": "before-we-act.bicoord.expert-seed-manifest/1",
                "status": "PASSED",
                "manifest": str(aggregate_manifest.resolve()),
                "manifest_sha256": _sha256(aggregate_manifest),
                "episodes_per_task": episodes,
                "task_counts": {task: len(valid_seeds[task]) for task in TASKS},
                "exception_type_counts": exception_type_counts,
                "exception_counts": exception_counts,
                "structural_exception_type_counts": (
                    structural_exception_type_counts
                ),
                "structural_exception_counts": structural_exception_counts,
                "structural_error_streak_limit": 3,
                "policy_independent": True,
                "learned_policy_used": False,
            },
        )
        artifacts.extend(
            [
                {
                    "path": str(aggregate_manifest.resolve()),
                    "sha256": _sha256(aggregate_manifest),
                    "kind": "expert_seed_manifest",
                },
                {
                    "path": str(aggregate_status.resolve()),
                    "sha256": _sha256(aggregate_status),
                    "kind": "expert_seed_status",
                },
            ]
        )
        _atomic_json(
            candidate,
            {
                "schema": RESULT_SCHEMA,
                "stage": spec.name,
                "status": "PASSED",
                "benchmark_adapter": "BiCoord",
                "config_sha256": self.config_hash,
                "workers": workers,
                "artifacts": self._deduplicate_artifacts(artifacts),
                "episodes_per_task": episodes,
                "tasks": list(TASKS),
                "valid_seeds": valid_seeds,
                "seed_manifest": str(aggregate_manifest.resolve()),
                "seed_manifest_sha256": _sha256(aggregate_manifest),
                "progress_receipts": progress_receipts,
                "progress_receipts_sha256": progress_receipt_sha256,
                "exception_type_counts": exception_type_counts,
                "exception_counts": exception_counts,
                "structural_exception_type_counts": (
                    structural_exception_type_counts
                ),
                "structural_exception_counts": structural_exception_counts,
                "structural_error_streak_limit": 3,
                "seed_receipts": seed_receipts,
                "seed_receipts_sha256": seed_receipts_sha256,
                "policy_independent": True,
                "learned_policy_used": False,
                "closed_loop_policy_results_used": False,
                "gpu_task_queue": True,
                "completed_at": _utc_now(),
            },
        )

    def _task_queue_action(self, spec: StageSpec, candidate: Path) -> None:
        worker_root = self.s.run / "worker_results" / spec.name
        remaining = list(enumerate(TASKS))
        results: dict[str, Path] = {}
        # Deterministic waves preserve reproducible assignment and never
        # oversubscribe a GPU.  Short tasks free no hidden lease mid-wave.
        for first in range(0, len(remaining), 4):
            wave = remaining[first : first + 4]
            jobs = []
            for gpu, (_task_index, task) in enumerate(wave):
                path = worker_root / f"{task}.json"
                results[task] = path
                episodes = (
                    VALIDATION_EPISODES
                    if spec.result_kind == "validation20"
                    else 1
                )
                command = self._base_command(
                    spec,
                    path,
                    (
                        "--task",
                        task,
                        "--episodes",
                        str(episodes),
                        "--max-steps",
                        str(
                            SMOKE_INTERFACE_STEPS
                            if spec.result_kind == "smoke_validation"
                            else MAX_STEPS[task]
                        ),
                        "--record-progress",
                    ),
                )
                jobs.append(
                    (
                        f"{spec.name}_{task}",
                        command,
                        gpu,
                        self.s.run / "logs" / f"{spec.name}_{task}.log",
                    )
                )
            self.scheduler.run_wave(jobs)

        task_rows: dict[str, Any] = {}
        artifacts: list[dict[str, str]] = []
        successes = 0
        selector_off_successes = 0
        care_successes = 0
        for task in TASKS:
            # Validate the worker contract before aggregation.  In
            # particular, paired workers carry a cryptographically bound
            # progress receipt and seed-pair manifest; retaining only a few
            # counters here would make those proofs unverifiable later.
            row = self._validated_worker_result(spec, results[task])
            if row.get("task") != task:
                raise InvalidArtifact(f"invalid task worker result: {task}")
            episodes = (
                VALIDATION_EPISODES if spec.result_kind == "validation20" else 1
            )
            if int(row.get("episodes", -1)) != episodes:
                raise InvalidArtifact(f"{task}: worker episode count differs")
            # Preserve every worker field (seed lists, pair manifest, state /
            # observation hashes, action traces, checkpoint hashes, and
            # normalization provenance) while making a detached mapping so a
            # caller cannot mutate the source object after publication.
            task_row = dict(row)
            if spec.name in {"paired_validation_smoke", "paired_validation20"}:
                progress_path = self._artifact_path(
                    {"path": row.get("progress_receipt", "")}
                ).resolve()
                progress = _read_json(progress_path)
                # These are the evidence-bearing fields that live inside the
                # worker's hashed receipt rather than its shallow result.
                # Keep them explicitly in the aggregate so a resumed final
                # validation can inspect both complete rollouts per seed.
                receipt_fields = {
                    "pair_manifest": progress.get("pair_manifest"),
                    "pair_manifest_sha256": progress.get("pair_manifest_sha256"),
                    "paired_rows": progress.get("rows"),
                    "normalization": progress.get("normalization"),
                }
                for key, value in receipt_fields.items():
                    if key in task_row and task_row[key] != value:
                        raise InvalidArtifact(
                            f"{task}: paired worker differs from receipt at {key}"
                        )
                    task_row[key] = value
            task_rows[task] = task_row
            successes += int(row.get("successes", 0))
            selector_off_successes += int(row.get("selector_off_successes", 0))
            care_successes += int(row.get("care_successes", 0))
            artifacts.extend(row.get("artifacts", []))
        value: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "stage": spec.name,
            "status": "PASSED",
            "benchmark_adapter": "BiCoord",
            "config_sha256": self.config_hash,
            "model_contract": dict(MODEL_CONTRACT),
            "tasks": task_rows,
            "total_successes": successes,
            "artifacts": artifacts,
            "completed_at": _utc_now(),
        }
        if spec.result_kind == "validation":
            value["episodes_per_task"] = 1
        if spec.name == "paired_validation20":
            value.update(
                {
                    "paired": True,
                    "selector_off_control": True,
                    "total_episodes": len(TASKS) * VALIDATION_EPISODES,
                    "total_rollouts": len(TASKS) * VALIDATION_EPISODES * 2,
                    "selector_off_successes": selector_off_successes,
                    "care_successes": care_successes,
                    "total_successes": care_successes,
                    "same_initial_state_verified": True,
                    "per_arm_independent_selector": True,
                    "cross_arm_lower_bound_arbitration": False,
                    "execution_order": ["selector_off", "care"],
                }
            )
        elif spec.name == "paired_validation_smoke":
            value.update(
                {
                    "paired": True,
                    "selector_off_control": True,
                    "total_episodes": len(TASKS),
                    "total_rollouts": len(TASKS) * 2,
                    "selector_off_successes": selector_off_successes,
                    "care_successes": care_successes,
                    "total_successes": care_successes,
                    "same_initial_state_verified": True,
                    "per_arm_independent_selector": True,
                    "cross_arm_lower_bound_arbitration": False,
                    "execution_order": ["selector_off", "care"],
                }
            )
        _atomic_json(candidate, value)

    def _run_action(self, spec: StageSpec, candidate: Path) -> None:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if spec.gpu_plan == "seed_wave":
            self._seed_wave_action(spec, candidate)
        elif spec.gpu_plan == "care_grid":
            self._care_grid_action(spec, candidate)
        elif spec.gpu_plan == "seed_task_queue4":
            self._seed_task_queue_action(spec, candidate)
        elif spec.gpu_plan == "sharded4":
            self._sharded_action(spec, candidate)
        elif spec.gpu_plan == "task_queue4":
            self._task_queue_action(spec, candidate)
        else:
            self._single_action(spec, candidate)

    def run_stage(self, spec: StageSpec) -> str:
        try:
            self._validate_receipt(spec.name)
            return "resumed"
        except InvalidArtifact:
            pass
        dependency_hashes = self._dependency_hashes(spec)
        candidate = self.results / f".{spec.name}.{os.getpid()}.candidate.json"
        attempts = 3 if spec.operation not in {"source-preflight", "formal-audit"} else 2
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            self._set_status(
                "running",
                spec.name,
                description=spec.description,
                gpu_plan=spec.gpu_plan,
                attempt=attempt,
                attempts=attempts,
            )
            try:
                self._run_action(spec, candidate)
                last_error = None
                break
            except (Interrupted, Blocked):
                raise
            except BaseException as error:
                last_error = error
                if attempt == attempts:
                    break
                self._set_status(
                    "retrying",
                    spec.name,
                    attempt=attempt,
                    attempts=attempts,
                    error=repr(error),
                )
                time.sleep(min(5 * attempt, 15))
        if last_error is not None:
            raise SupervisorError(
                f"{spec.name} failed after {attempts} attempts: {last_error}"
            ) from last_error
        self._validate_result(spec, candidate)
        final_result = self.result_path(spec.name)
        final_result.parent.mkdir(parents=True, exist_ok=True)
        os.replace(candidate, final_result)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "stage": spec.name,
            "status": "COMPLETED",
            "completed_at": _utc_now(),
            "config_sha256": self.config_hash,
            "dependency_receipt_sha256": dependency_hashes,
            "result": str(final_result),
            "result_sha256": _sha256(final_result),
            "gpu_plan": spec.gpu_plan,
            "module": self.s.modules.get(spec.module_key or ""),
        }
        _atomic_json(self.receipt_path(spec.name), receipt)
        self._validate_receipt(spec.name)
        return "completed"

    def run(
        self,
        *,
        start_at: str | None = None,
        stop_after: str | None = None,
    ) -> int:
        if start_at is not None and start_at not in STAGES:
            raise ValueError(f"unknown start stage: {start_at}")
        if stop_after is not None and stop_after not in STAGES:
            raise ValueError(f"unknown stop stage: {stop_after}")
        self.s.run.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.s.run / "frozen_config.json", self.config)
        self._install_signals()
        self._start_heartbeat()
        try:
            self.preflight()
            started = start_at is None
            outcomes: dict[str, str] = {}
            for name, spec in STAGES.items():
                if name == start_at:
                    started = True
                if not started:
                    # An explicit start point still validates all dependency
                    # receipts; it never bypasses the DAG.
                    continue
                outcomes[name] = self.run_stage(spec)
                if name == stop_after:
                    self._set_status("paused_at_requested_boundary", name, outcomes=outcomes)
                    return 0
            self._set_status(
                "complete",
                "paired_validation20",
                outcomes=outcomes,
                final_result=str(self.result_path("paired_validation20")),
            )
            return 0
        except Blocked as error:
            self._set_status(
                "blocked",
                self.current_stage,
                error=str(error),
                traceback=traceback.format_exc(),
            )
            return 3
        except Interrupted as error:
            self._set_status("interrupted", self.current_stage, error=str(error))
            return 130
        except BaseException as error:
            self._set_status(
                "failed",
                self.current_stage,
                error=repr(error),
                traceback=traceback.format_exc(),
            )
            raise
        finally:
            self._stop_heartbeat()
            self._restore_signals()


def plan_payload(settings: Settings) -> dict[str, Any]:
    return {
        "schema": SUPERVISOR_SCHEMA,
        "config_sha256": _canonical_hash(settings.frozen_config()),
        "stages": [asdict(stage) for stage in STAGES.values()],
        "model_contract": MODEL_CONTRACT,
        "tasks": list(TASKS),
        "max_steps": MAX_STEPS,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    subparsers.add_parser("preflight")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--start-at", choices=tuple(STAGES))
    run_parser.add_argument("--stop-after", choices=tuple(STAGES))
    subparsers.add_parser("status")
    args = parser.parse_args(argv)
    settings = Settings.from_environment()
    settings.validate()
    if args.command == "plan":
        print(json.dumps(plan_payload(settings), indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        path = settings.run / "status.json"
        print(path.read_text() if path.is_file() else json.dumps({"state": "not_started"}))
        return 0
    supervisor = Supervisor(settings)
    if args.command == "preflight":
        try:
            report = supervisor.preflight()
        except Blocked as error:
            print(json.dumps({"status": "BLOCKED", "error": str(error)}, indent=2))
            return 3
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    return supervisor.run(start_at=args.start_at, stop_after=args.stop_after)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_HORIZON",
    "BCORE_SEEDS",
    "BICOORD_CODE_REVISION",
    "CARE_OOF_FOLDS",
    "CARE_OOF_SEED",
    "CARE_OOF_VARIANT",
    "CARE_OOF_TRAINING_SEED_OFFSET",
    "CARE_SEEDS",
    "CARE_VARIANTS",
    "FORMAL_DATASET_REVISION",
    "FORMAL_DATASET_REPO",
    "FORMAL_EPISODES",
    "FORMAL_EPISODES_PER_TASK",
    "GLOBAL_BATCH",
    "GPU_IDS",
    "HEARTBEAT_INTERVAL_SECONDS",
    "HISTORY_STEPS",
    "MAX_STEPS",
    "MODEL_CONTRACT",
    "STAGES",
    "STAGE_DEPENDENCIES",
    "Settings",
    "StageSpec",
    "Supervisor",
    "SupervisorError",
    "TASKS",
    "VALIDATION_EPISODES",
    "plan_payload",
]
