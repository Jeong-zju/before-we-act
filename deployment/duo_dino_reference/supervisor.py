"""Fail-closed, resumable DuoBench DINO B0-H -> B-core -> CARE supervisor.

This is the only formal DuoBench CARE orchestration path.  ACT is used only
as the lossless dataset converter (and may be evaluated separately as a
baseline); it is never accepted as the reference or as B-core.

The supervisor owns four physical GPUs and advances an explicit stage DAG.
Every expensive output is revalidated before it is adopted on resume.  A
zero-success B0-H or B-core probe is a terminal gate, and missing Duo-specific
B-core/branch/paired adapters produce ``BLOCKED_PENDING_IMPLEMENTATION`` rather
than silently falling back to the old ACT/ConvNeXt CARE implementation.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

from deployment.duo_act.action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SCHEMA,
    ACTION_TARGET_CONTRACT_SHA256,
    validate_action_target_contract,
)


TASKS = (
    "ball_maze",
    "bin_sort",
    "block_balance",
    "carry_pot",
    "hinge_chest",
    "join_blocks",
    "pour_marbles",
    "spring_door",
    "transfer_cube",
    "transfer_gate",
    "transfer_reorient",
)

# Frozen per-task horizons.  These are passed explicitly to every probe and
# Validation20 worker; no global shortcut is allowed.
MAX_STEPS = {
    "ball_maze": 526,
    "bin_sort": 365,
    "block_balance": 1091,
    "carry_pot": 840,
    "hinge_chest": 610,
    "join_blocks": 1314,
    "pour_marbles": 549,
    "spring_door": 1070,
    "transfer_cube": 605,
    "transfer_gate": 630,
    "transfer_reorient": 883,
}

B0H_UPDATES = 120_000
BCORE_UPDATES = 120_000
CARE_UPDATES = 4_000
VALIDATION_EPISODES = 20
FAMILIES_PER_TASK = 30
BCORE_SEEDS = (20260815, 20260816, 20260817)
CARE_SEEDS = (20260818, 20260819, 20260820)
CARE_VARIANTS = ("care", "reactive_only", "replay_only", "capacity")

# Kept at module scope so status tooling and tests can inspect the graph
# without constructing CUDA or simulator state.
STAGE_DEPENDENCIES: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    (
        ("dependencies", ()),
        ("dataset_download", ("dependencies",)),
        ("data_prepare", ("dataset_download",)),
        ("data_audit", ("data_prepare",)),
        ("dino_cache", ("data_audit",)),
        ("b0h_smoke_train", ("dino_cache",)),
        ("b0h_smoke_closed_loop", ("b0h_smoke_train",)),
        ("b0h_formal", ("b0h_smoke_closed_loop",)),
        ("b0h_validation20", ("b0h_formal",)),
        ("b0h_probe_gate", ("b0h_validation20",)),
        # B-core is a separate PredictiveTeamBeliefPolicy stage.  It is not an
        # alias for B0-H and may not be bypassed by the legacy ACT CARE stack.
        ("bcore_cache", ("b0h_probe_gate",)),
        ("bcore_smoke", ("bcore_cache",)),
        ("bcore_train_3seeds", ("bcore_smoke",)),
        ("bcore_select", ("bcore_train_3seeds",)),
        ("bcore_validation20", ("bcore_select",)),
        ("bcore_probe_gate", ("bcore_validation20",)),
        ("branch_smoke", ("bcore_probe_gate",)),
        ("branch_collection", ("branch_smoke",)),
        ("branch_prepare", ("branch_collection",)),
        ("branch_signal_gate", ("branch_prepare",)),
        ("belief_smoke", ("branch_signal_gate",)),
        ("belief_train", ("belief_smoke",)),
        ("offline_selection_calibration", ("belief_train",)),
        ("paired_validation_smoke", ("offline_selection_calibration",)),
        ("paired_validation20", ("paired_validation_smoke",)),
    )
)

SCHEMA = "before-we-act.duobench-dino-care-supervisor/1"
RECEIPT_SCHEMA = "before-we-act.duobench-dino-care-stage/1"
B0H_CHECKPOINT_FORMAT = "before-we-act.duobench.dino-b0h/1"
BCORE_TRAINING_FORMAT = "before-we-act.duobench.dino-bcore-training/1"
BCORE_DEPLOYMENT_FORMAT = "before-we-act.duobench.dino-bcore-deployment/1"
CARE_TRAINING_FORMAT = "before-we-act.care-duobench-training-checkpoint/1"
CARE_DEPLOYMENT_FORMAT = "before-we-act.care-duobench-deployment-checkpoint/1"
IMAGE_PREPROCESS_ID = "rcs_lerobot_v2_resize_uint8_bilinear_antialias_v1"
DINO_NORMALIZATION_ID = "dinov3_imagenet_rgb_mean_std_rescale_1_over_255_v1"
DUO_CARE_MEMORY_SEMANTICS = (
    "PredictiveTeamBeliefPolicy.belief.mu+belief.event_memory"
)
DUO_CARE_MEMORY_TOKENS = 20
DUO_CARE_MEMORY_WIDTH = 384


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise MissingArtifact(str(path)) from error
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidArtifact(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise InvalidArtifact(f"expected object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _stable_value(value: object) -> object:
    """Drop intentionally volatile timing fields before receipt comparison."""
    volatile = {
        "updated_at",
        "updated_at_epoch",
        "heartbeat_at",
        "created_at_utc",
        "completed_at_utc",
        "started_at",
        "elapsed_seconds",
        "wall_seconds",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(child)
            for key, child in value.items()
            if str(key) not in volatile
        }
    if isinstance(value, (list, tuple)):
        return [_stable_value(child) for child in value]
    return value


# ``ACT`` is allowed in the repository only as a lossless data-converter or a
# separately reported diagnostic baseline.  Formal artifacts must never carry
# an ACT/ConvNeXt provider (including through a user-supplied module override).
# Keep these expressions deliberately narrow: ``action_encoding`` and ordinary
# words such as ``interaction`` must not be mistaken for the legacy ACT model.
_LEGACY_POLICY_PATTERNS = (
    re.compile(r"(?i)(?:^|[^a-z0-9])convnext(?:[^a-z0-9]|$)"),
    # Upper-case ``ACT`` is the conventional model-family spelling.  Do not
    # use a case-insensitive bare-word expression here: schema names such as
    # ``before-we-act.*`` are part of this project's provenance and are not a
    # model implementation.
    re.compile(
        r"(?:^|[^a-z0-9])ACT(?:Policy|[-_]?Policy|[-_]?Baseline|[-_]?Reference|[-_]?Model)?(?:[^a-z0-9]|$)"
    ),
    # Lower-case checkpoint spellings are caught when ``act`` is part of a
    # policy/baseline/model token.  A bare ``duo_act`` path is the permitted
    # lossless data converter and must not be rejected here.
    re.compile(
        r"(?i)(?:^|[^a-z0-9])act(?:policy|[-_]?policy|[-_]?baseline|[-_]?reference|[-_]?model)(?:[^a-z0-9]|$)"
    ),
    re.compile(r"(?i)(?:^|[^a-z0-9])resnet18(?:[^a-z0-9]|$)"),
)


def _string_leaves(value: object, prefix: str = "") -> list[tuple[str, str]]:
    """Return string leaves with paths, without treating mapping keys as data."""

    leaves: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.extend(_string_leaves(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            leaves.extend(_string_leaves(child, f"{prefix}[{index}]"))
    elif isinstance(value, Path):
        leaves.append((prefix, str(value)))
    elif isinstance(value, str):
        leaves.append((prefix, value))
    return leaves


def _legacy_policy_markers(value: object) -> list[str]:
    markers: list[str] = []
    for path, text in _string_leaves(value):
        for pattern in _LEGACY_POLICY_PATTERNS:
            if pattern.search(text):
                markers.append(f"{path}={text}")
                break
    return markers


def _reject_legacy_policy(value: object, context: str) -> None:
    markers = _legacy_policy_markers(value)
    if markers:
        raise InvalidArtifact(
            f"{context} contains a forbidden ACT/ConvNeXt policy marker: "
            + "; ".join(markers[:4])
        )


def _find_metadata(value: object, key: str) -> object | None:
    """Find a metadata field in a nested artifact payload deterministically."""

    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_metadata(child, key)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _find_metadata(child, key)
            if found is not None:
                return found
    return None


def _require_dino_local_metadata(
    value: object,
    *,
    context: str,
    policy_family: str,
    action_encodings: Sequence[str] = (),
    method_family: str = "CARE",
) -> None:
    """Enforce the provenance shared by formal Duo artifacts.

    ``method_family`` and ``policy_family`` are deliberately independent
    dimensions.  In particular, ``CARE`` names the method/protocol while
    ``TemporalHistoryPolicy`` and ``PredictiveTeamBeliefPolicy`` name concrete
    networks.  Requiring an explicit method field (rather than using a
    default) prevents metadata-only or legacy checkpoints from being adopted
    when a producer accidentally omits the method tag.
    """

    _reject_legacy_policy(value, context)
    method = _find_metadata(value, "method_family")
    if method != method_family:
        raise InvalidArtifact(
            f"{context} does not identify method_family={method_family!r}: {method!r}"
        )
    family = _find_metadata(value, "policy_family")
    reference_family = _find_metadata(value, "reference_policy_family")
    family_text = " ".join(
        str(item) for item in (family, reference_family) if item is not None
    )
    if policy_family not in family_text:
        raise InvalidArtifact(
            f"{context} does not identify the required {policy_family} policy family"
        )
    vision = _find_metadata(value, "vision_backbone")
    if vision is None:
        vision = _find_metadata(value, "vision")
    if vision != "dinov3_vitb16_frozen":
        raise InvalidArtifact(f"{context} does not identify frozen DINOv3 ViT-B/16")
    if _find_metadata(value, "image_preprocess_id") != IMAGE_PREPROCESS_ID:
        raise InvalidArtifact(f"{context} does not identify the frozen RGB preprocessing")
    if _find_metadata(value, "dino_normalization_id") != DINO_NORMALIZATION_ID:
        raise InvalidArtifact(f"{context} does not identify the frozen DINO normalization")
    # A provenance string alone is not enough: formal consumers must opt into
    # the strict DINO contract explicitly.  This prevents an older checkpoint
    # (which happened to carry the same backbone/normalization labels) from
    # being silently adopted after a resume or module override.
    if _find_metadata(value, "strict_dino_contract") is not True:
        raise InvalidArtifact(f"{context} does not enable strict_dino_contract")
    decentralized = _find_metadata(value, "strictly_decentralized")
    if decentralized is None:
        decentralized = _find_metadata(value, "strict_local")
    if decentralized is not True:
        raise InvalidArtifact(f"{context} is not marked strictly decentralized")
    provider_allowed = _find_metadata(value, "act_provider_allowed")
    if provider_allowed is not None and provider_allowed is not False:
        raise InvalidArtifact(f"{context} explicitly allows an ACT provider")
    if action_encodings:
        encoding = _find_metadata(value, "action_encoding")
        if encoding not in set(action_encodings):
            raise InvalidArtifact(
                f"{context} action encoding {encoding!r} is outside the formal contract"
            )


def _require_action_target_metadata(value: object, context: str) -> None:
    if _find_metadata(value, "action_target_contract_id") != ACTION_TARGET_CONTRACT_ID:
        raise InvalidArtifact(f"{context} action-target contract id differs")
    if (
        _find_metadata(value, "action_target_contract_sha256")
        != ACTION_TARGET_CONTRACT_SHA256
    ):
        raise InvalidArtifact(f"{context} action-target contract hash differs")


def _require_care_memory_metadata(value: object, context: str) -> None:
    semantics = _find_metadata(value, "care_memory_semantics")
    if semantics is None:
        semantics = _find_metadata(value, "memory_semantics")
    if semantics != DUO_CARE_MEMORY_SEMANTICS:
        raise InvalidArtifact(f"{context} does not use belief mu + event memory")
    if _find_metadata(value, "care_memory_tokens") != DUO_CARE_MEMORY_TOKENS:
        raise InvalidArtifact(f"{context} CARE memory token count differs")
    width = _find_metadata(value, "care_memory_width")
    if width is not None and width != DUO_CARE_MEMORY_WIDTH:
        raise InvalidArtifact(f"{context} CARE memory width differs")


def _load_torch_payload(path: Path, context: str) -> Mapping[str, Any]:
    """Load a checkpoint only at validation time, keeping supervisor import light."""

    if not path.is_file() or path.stat().st_size == 0:
        raise MissingArtifact(str(path))
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except MissingArtifact:
        raise
    except Exception as error:
        raise InvalidArtifact(f"cannot load {context} checkpoint {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise InvalidArtifact(f"{context} checkpoint is not a mapping: {path}")
    return payload


class MissingArtifact(RuntimeError):
    """A stage has no terminal artifact yet and may be executed/resumed."""


class InvalidArtifact(RuntimeError):
    """An artifact exists but fails its frozen contract; never overwrite it."""


class ProvenanceDrift(RuntimeError):
    """A completed receipt belongs to a different command/dependency graph."""


class PendingImplementation(RuntimeError):
    """A deliberately unavailable formal adapter blocks downstream stages."""

    def __init__(self, stage: str, modules: Sequence[str]):
        self.stage = stage
        self.modules = tuple(modules)
        super().__init__(f"{stage} requires Duo-specific adapter(s): {', '.join(modules)}")


class GateBlocked(RuntimeError):
    """A scientific gate failed without converting that result into success."""

    def __init__(self, stage: str, reason: str, exit_code: int = 75):
        self.stage = stage
        self.reason = reason
        self.exit_code = int(exit_code)
        super().__init__(reason)


class StopPipeline(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    repo: Path
    duobench_repo: Path
    dataset: Path
    run: Path
    dino_model: Path
    python: str
    heartbeat_seconds: int
    b0h_smoke_updates: int
    b0h_probe_episodes: int
    bcore_probe_episodes: int
    seed_start: int
    families_per_task: int
    bcore_cache_module: str
    bcore_train_module: str
    bcore_select_module: str
    bcore_evaluate_module: str
    bcore_smoke_module: str
    branch_module: str
    paired_module: str

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            repo=Path(os.environ.get("DUO_DINO_REPO", "/workspace/repos/care-official")),
            duobench_repo=Path(os.environ.get("DUO_DINO_DUOBENCH_REPO", "/workspace/repos/duobench")),
            dataset=Path(os.environ.get("DUO_DINO_DATASET", "/workspace/datasets/duobench")),
            run=Path(os.environ.get("DUO_DINO_RUN", "/workspace/runs/duobench-care-dino-v1")),
            dino_model=Path(os.environ.get("DUO_DINO_MODEL", "/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m")),
            python=os.environ.get("DUO_DINO_PYTHON", "/venv/main/bin/python"),
            heartbeat_seconds=max(5, int(os.environ.get("DUO_DINO_HEARTBEAT_SECONDS", "20"))),
            b0h_smoke_updates=int(os.environ.get("DUO_DINO_B0H_SMOKE_UPDATES", "4")),
            b0h_probe_episodes=int(os.environ.get("DUO_DINO_B0H_PROBE_EPISODES", "20")),
            bcore_probe_episodes=int(os.environ.get("DUO_DINO_BCORE_PROBE_EPISODES", "20")),
            seed_start=int(os.environ.get("DUO_DINO_VALIDATION_SEED_START", "20260830")),
            families_per_task=int(os.environ.get("DUO_DINO_FAMILIES_PER_TASK", str(FAMILIES_PER_TASK))),
            bcore_cache_module=os.environ.get(
                "DUO_DINO_BCORE_CACHE_MODULE", "deployment.duo_dino_reference.cache_bcore"
            ),
            bcore_train_module=os.environ.get(
                "DUO_DINO_BCORE_TRAIN_MODULE", "deployment.duo_dino_reference.train_bcore"
            ),
            bcore_select_module=os.environ.get(
                "DUO_DINO_BCORE_SELECT_MODULE", "deployment.duo_dino_reference.select_bcore"
            ),
            bcore_evaluate_module=os.environ.get(
                "DUO_DINO_BCORE_EVALUATE_MODULE", "deployment.duo_dino_reference.evaluate_bcore"
            ),
            bcore_smoke_module=os.environ.get(
                "DUO_DINO_BCORE_SMOKE_MODULE", "deployment.duo_dino_reference.smoke_bcore"
            ),
            branch_module=os.environ.get(
                "DUO_DINO_BRANCH_MODULE", "deployment.duo_care.duo_dino_branch_launcher"
            ),
            paired_module=os.environ.get(
                "DUO_DINO_PAIRED_MODULE", "deployment.duo_care.duo_dino_paired_launcher"
            ),
        )

    @property
    def data(self) -> Path:
        return self.run / "prepared_data"

    @property
    def cache(self) -> Path:
        return self.run / "dino_cache"

    @property
    def smoke_checkpoint(self) -> Path:
        return self.run / "b0h" / "smoke" / "final.pt"

    @property
    def b0h_checkpoint(self) -> Path:
        return self.run / "b0h" / "formal" / "final.pt"

    @property
    def bcore_root(self) -> Path:
        return self.run / "bcore"

    @property
    def bcore_checkpoint(self) -> Path:
        return self.bcore_root / "selected" / "deployment_checkpoint.pt"

    @property
    def branch_root(self) -> Path:
        return self.run / "care" / "branches"

    @property
    def prepared_care(self) -> Path:
        return self.run / "care" / "prepared.pt"

    @property
    def care_checkpoint(self) -> Path:
        return self.run / "care" / "offline" / "care_deployment_checkpoint.pt"


@dataclass
class ActiveProcess:
    name: str
    process: subprocess.Popen
    gpus: tuple[int, ...]
    started_at: float
    log: str


Validator = Callable[[], dict[str, Any]]
Action = Callable[[], None]


@dataclass(frozen=True)
class Stage:
    name: str
    detail: str
    validator: Validator
    action: Action
    fingerprint: Mapping[str, Any]


class Pipeline:
    def __init__(self, settings: Settings):
        self.s = settings
        self.status_path = self.s.run / "status.json"
        self.receipt_root = self.s.run / "state" / "receipts"
        self.log_root = self.s.run / "logs"
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.active: dict[int, ActiveProcess] = {}
        self.stage_name = "initializing"
        self.stage_detail = "constructing formal DAG"
        self.completed: list[str] = []
        self._heartbeat: threading.Thread | None = None

    # ------------------------------------------------------------------ state
    def _receipt(self, stage: str) -> Path:
        return self.receipt_root / f"{stage}.json"

    def _active_rows(self) -> list[dict[str, Any]]:
        rows = []
        for active in self.active.values():
            rows.append(
                {
                    "name": active.name,
                    "pid": active.process.pid,
                    "gpus": list(active.gpus),
                    "elapsed_seconds": max(0.0, time.time() - active.started_at),
                    "log": active.log,
                    "returncode": active.process.poll(),
                }
            )
        return sorted(rows, key=lambda row: row["name"])

    def write_status(self, state: str, stage: str, detail: str, **extra: Any) -> None:
        with self.lock:
            self.stage_name = stage
            self.stage_detail = detail
            payload = {
                "schema": SCHEMA,
                "state": state,
                "stage": stage,
                "detail": detail,
                "updated_at": _utc_now(),
                "heartbeat_at": _utc_now(),
                "pid": os.getpid(),
                "completed_stages": list(self.completed),
                "stage_dependencies": {key: list(value) for key, value in STAGE_DEPENDENCIES.items()},
                "active_processes": self._active_rows(),
                "gpu_schedule": {
                    "physical_gpus": [0, 1, 2, 3],
                    "b0h_cache_and_formal": "4-rank DDP, local batch 12, global batch 48",
                    "bcore_training": "three independent formal seeds on GPU 0/1/2; GPU3 reserved",
                    "branch_and_validation": "up to four isolated one-GPU workers",
                    "care_belief": "waves of four independent variant/seed jobs",
                },
                "formal_reference": "DINOv3 TemporalHistoryPolicy B0-H hidden-residual",
                "bcore": "independent PredictiveTeamBeliefPolicy, three seeds plus offline selection",
                "act_role": "dataset converter and optional baseline only; never formal reference",
                **extra,
            }
            _atomic_json(self.status_path, payload)

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.wait(self.s.heartbeat_seconds):
            try:
                with self.lock:
                    current: dict[str, Any] = {}
                    if self.status_path.is_file():
                        current = json.loads(self.status_path.read_text())
                    current.update(
                        {
                            "heartbeat_at": _utc_now(),
                            "pid": os.getpid(),
                            "active_processes": self._active_rows(),
                        }
                    )
                    _atomic_json(self.status_path, current)
            except Exception:
                # A heartbeat must never mask a stage outcome.
                pass

    def start_heartbeat(self) -> None:
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="duo-dino-heartbeat", daemon=True
        )
        self._heartbeat.start()

    def request_stop(self, _number: int | None = None, _frame: Any = None) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        try:
            self.write_status("STOPPING", self.stage_name, "graceful stop requested")
        except Exception:
            pass
        with self.lock:
            running = list(self.active.values())
        for active in running:
            try:
                os.killpg(active.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    # --------------------------------------------------------------- processes
    def environment(self, gpus: Sequence[int]) -> dict[str, str]:
        value = dict(os.environ)
        python_path = [str(self.s.repo), str(self.s.duobench_repo / "src")]
        if value.get("PYTHONPATH"):
            python_path.append(value["PYTHONPATH"])
        library = "/venv/main/lib/python3.12/site-packages/mujoco"
        if value.get("LD_LIBRARY_PATH"):
            library += ":" + value["LD_LIBRARY_PATH"]
        value.update(
            {
                "PYTHONPATH": ":".join(python_path),
                "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpus),
                "MUJOCO_GL": "egl",
                "DUOBENCH_PREFIX": str(self.s.duobench_repo),
                "HF_HOME": value.get("HF_HOME", "/workspace/.hf_home"),
                "WANDB_MODE": "disabled",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "LD_LIBRARY_PATH": library,
            }
        )
        # Never inject or serialize a credential here.  The wrapper may source
        # /workspace/.env and subprocesses inherit it in-memory only.
        return value

    def _spawn(
        self,
        name: str,
        command: Sequence[str],
        gpus: Sequence[int],
        *,
        attempt: int = 1,
    ) -> ActiveProcess:
        if self.stop_event.is_set():
            raise StopPipeline("stop requested")
        gpu_tuple = tuple(int(gpu) for gpu in gpus)
        if not gpu_tuple or any(gpu not in (0, 1, 2, 3) for gpu in gpu_tuple):
            raise ValueError(f"invalid GPU lease for {name}: {gpu_tuple}")
        with self.lock:
            used = {gpu for active in self.active.values() for gpu in active.gpus}
            overlap = used.intersection(gpu_tuple)
            if overlap:
                raise RuntimeError(f"GPU double lease {sorted(overlap)} while launching {name}")
        self.log_root.mkdir(parents=True, exist_ok=True)
        log_path = self.log_root / f"{name}.log"
        stream = log_path.open("a")
        stream.write(
            json.dumps(
                {
                    "event": "launch",
                    "time": _utc_now(),
                    "attempt": attempt,
                    "gpus": list(gpu_tuple),
                    "command": list(command),
                }
            )
            + "\n"
        )
        stream.flush()
        process = subprocess.Popen(
            list(command),
            cwd=self.s.repo,
            env=self.environment(gpu_tuple),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # Keep the stream alive on the object; Popen does not own stdout when a
        # file descriptor was supplied.
        setattr(process, "_duo_log_stream", stream)
        active = ActiveProcess(name, process, gpu_tuple, time.time(), str(log_path))
        with self.lock:
            self.active[process.pid] = active
        self.write_status("RUNNING", self.stage_name, self.stage_detail)
        return active

    def _wait(self, active: ActiveProcess) -> int:
        code = active.process.wait()
        stream = getattr(active.process, "_duo_log_stream", None)
        if stream is not None:
            stream.write(json.dumps({"event": "exit", "time": _utc_now(), "code": code}) + "\n")
            stream.close()
        with self.lock:
            self.active.pop(active.process.pid, None)
        if self.stop_event.is_set():
            raise StopPipeline("stop requested")
        return int(code)

    def run_command(
        self,
        name: str,
        command: Sequence[str],
        gpus: Sequence[int],
        *,
        retries: int = 1,
    ) -> None:
        for attempt in range(1, retries + 1):
            active = self._spawn(name, command, gpus, attempt=attempt)
            code = self._wait(active)
            if code == 0:
                return
            if attempt < retries:
                if self.stop_event.wait(min(60, 10 * attempt)):
                    raise StopPipeline("stop requested")
        raise RuntimeError(f"{name} exited {code} after {retries} attempt(s)")

    def run_wave(
        self,
        jobs: Sequence[tuple[str, Sequence[str], int]],
        *,
        retries: int = 1,
    ) -> None:
        """Run at most one job per physical GPU and fail the entire wave."""

        if len(jobs) > 4 or len({gpu for _name, _cmd, gpu in jobs}) != len(jobs):
            raise ValueError(f"invalid four-GPU wave: {[(name, gpu) for name, _cmd, gpu in jobs]}")
        pending = list(jobs)
        failures: list[str] = []
        attempt = 1
        while pending and attempt <= retries:
            active = [self._spawn(name, command, (gpu,), attempt=attempt) for name, command, gpu in pending]
            next_pending: list[tuple[str, Sequence[str], int]] = []
            for spec, process in zip(pending, active, strict=True):
                code = self._wait(process)
                if code:
                    next_pending.append(spec)
                    failures.append(f"{spec[0]}:attempt={attempt}:exit={code}")
            pending = next_pending
            attempt += 1
        if pending:
            raise RuntimeError("wave failed: " + ", ".join(failures))

    # -------------------------------------------------------------- validation
    @staticmethod
    def _module_available(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def _require_modules(self, stage: str, modules: Sequence[str]) -> None:
        missing = [module for module in modules if not self._module_available(module)]
        if missing:
            raise PendingImplementation(stage, missing)

    @staticmethod
    def _require_formal_module_identity(
        stage: str,
        module: str,
        *,
        prefix: str,
        role_token: str,
    ) -> None:
        """Reject environment overrides into legacy ACT/ConvNeXt code paths."""

        _reject_legacy_policy({"module": module}, f"{stage} module")
        if not module.startswith(prefix) or role_token not in module.rsplit(".", 1)[-1]:
            raise InvalidArtifact(
                f"{stage} formal module {module!r} is outside {prefix!r} "
                f"or lacks role token {role_token!r}"
            )

    def validate_formal_module_overrides(self) -> dict[str, str]:
        bcore_modules = {
            "bcore_cache": (self.s.bcore_cache_module, "bcore"),
            "bcore_smoke": (self.s.bcore_smoke_module, "bcore"),
            "bcore_train_3seeds": (self.s.bcore_train_module, "bcore"),
            "bcore_select": (self.s.bcore_select_module, "bcore"),
            "bcore_validation20": (self.s.bcore_evaluate_module, "bcore"),
        }
        for stage, (module, token) in bcore_modules.items():
            self._require_formal_module_identity(
                stage,
                module,
                prefix="deployment.duo_dino_reference.",
                role_token=token,
            )
        self._require_formal_module_identity(
            "branch_collection",
            self.s.branch_module,
            prefix="deployment.duo_care.duo_dino_",
            role_token="branch",
        )
        self._require_formal_module_identity(
            "paired_validation20",
            self.s.paired_module,
            prefix="deployment.duo_care.duo_dino_",
            role_token="paired",
        )
        return {
            stage: module for stage, (module, _token) in bcore_modules.items()
        } | {
            "branch_collection": self.s.branch_module,
            "paired_validation20": self.s.paired_module,
        }

    def validate_dependencies(self) -> dict[str, Any]:
        if not self.s.repo.is_dir():
            raise InvalidArtifact(f"repository missing: {self.s.repo}")
        if not self.s.duobench_repo.is_dir():
            raise InvalidArtifact(f"DuoBench repository missing: {self.s.duobench_repo}")
        if not Path(self.s.python).is_file():
            raise InvalidArtifact(f"Python missing: {self.s.python}")
        if not self.s.dino_model.exists():
            raise InvalidArtifact(f"frozen DINOv3 model missing: {self.s.dino_model}")
        formal_modules = self.validate_formal_module_overrides()
        immediate = (
            "deployment.duo_care.download",
            "deployment.duo_act.prepare",
            "deployment.duo_act.audit",
            "deployment.duo_dino_reference.data",
            "deployment.duo_dino_reference.cache_dino",
            "deployment.duo_dino_reference.train_b0h",
            "deployment.duo_dino_reference.evaluate",
        )
        unavailable = [module for module in immediate if not self._module_available(module)]
        if unavailable:
            raise InvalidArtifact(f"required B0-H modules missing: {unavailable}")
        command = [
            self.s.python,
            "-c",
            (
                "import json,torch; "
                "assert torch.cuda.device_count()==4, torch.cuda.device_count(); "
                "names=[torch.cuda.get_device_name(i) for i in range(4)]; "
                # CARE-v2 can run on the reserved π0.5 host as well as the
                # original RTX 5090 host.  Keep the four-GPU requirement and
                # record exact names, but do not encode a vendor SKU into the
                # scientific protocol.
                "assert all(('5090' in x or 'RTX PRO 6000' in x) for x in names), names; print(json.dumps(names))"
            ),
        ]
        self.run_command("dependencies_gpu", command, (0, 1, 2, 3))
        return {
            "gpu_count": 4,
            "gpu_model": "four compatible Blackwell GPUs",
            "immediate_modules": list(immediate),
            "formal_module_overrides": formal_modules,
            "dino_model": str(self.s.dino_model.resolve()),
        }

    def validate_dataset_download(self) -> dict[str, Any]:
        if not self.s.dataset.is_dir():
            raise MissingArtifact(str(self.s.dataset))
        receipt_path = self.s.run / "dataset_download_receipt.json"
        receipt = _read_json(receipt_path)
        if (
            receipt.get("schema") != "before-we-act.duobench.dataset-download/1"
            or receipt.get("status") != "PASSED"
            or receipt.get("revision") != "b741bc915d942ecadaefb4e3de6bbd716c1b8b1b"
            or receipt.get("credential_recorded") is not False
        ):
            raise InvalidArtifact("dataset download receipt is invalid")
        details = {}
        for task in TASKS:
            root = self.s.dataset / task / "sim"
            parquet = sorted((root / "data").glob("**/*.parquet"))
            videos = {
                key: sorted((root / "videos" / key).glob("**/*.mp4"))
                for key in (
                    "observation.images.head",
                    "observation.images.left_wrist",
                    "observation.images.right_wrist",
                )
            }
            if len(parquet) != 1 or any(len(paths) != 1 for paths in videos.values()):
                raise MissingArtifact(f"incomplete raw DuoBench task: {task}")
            if parquet[0].stat().st_size == 0 or any(paths[0].stat().st_size == 0 for paths in videos.values()):
                raise InvalidArtifact(f"zero-byte raw DuoBench artifact: {task}")
            details[task] = {
                "parquet": str(parquet[0]),
                "videos": {key: str(paths[0]) for key, paths in videos.items()},
            }
        return {
            "dataset": str(self.s.dataset.resolve()),
            "revision": "b741bc915d942ecadaefb4e3de6bbd716c1b8b1b",
            "download_receipt_sha256": _sha256(receipt_path),
            "tasks": details,
        }

    def validate_data(self) -> dict[str, Any]:
        manifest = _read_json(self.s.data / "manifest.json")
        if tuple(manifest.get("tasks", {})) != TASKS:
            raise InvalidArtifact("prepared data task order/coverage differs")
        contract = manifest.get("action_target_contract")
        if not isinstance(contract, Mapping):
            raise InvalidArtifact("prepared data lacks action-target contract")
        try:
            validate_action_target_contract(contract)
        except ValueError as error:
            raise InvalidArtifact(
                f"prepared action-target contract differs: {error}"
            ) from error
        if (
            contract.get("schema") != ACTION_TARGET_CONTRACT_SCHEMA
            or contract.get("id") != ACTION_TARGET_CONTRACT_ID
            or contract.get("sha256") != ACTION_TARGET_CONTRACT_SHA256
        ):
            raise InvalidArtifact("prepared action-target contract identity differs")
        audit_meta = manifest.get("action_target_audit")
        audit_path = self.s.data / "action_target_audit.json"
        if (
            not isinstance(audit_meta, Mapping)
            or audit_meta.get("schema")
            != "before-we-act.duobench.action-target-audit/1"
            or audit_meta.get("status") != "PASSED"
            or audit_meta.get("path") != audit_path.name
            or audit_meta.get("contract_id") != ACTION_TARGET_CONTRACT_ID
            or audit_meta.get("contract_sha256")
            != ACTION_TARGET_CONTRACT_SHA256
            or not audit_path.is_file()
            or audit_meta.get("sha256") != _sha256(audit_path)
        ):
            raise InvalidArtifact(
                "prepared action-target audit receipt is missing or stale"
            )
        if int(manifest.get("total_episodes", -1)) != 550:
            raise InvalidArtifact("formal data must contain all 550 demonstrations")
        norm = manifest.get("normalization", {})
        if norm.get("action_encoding") != "absolute_joint7_binary_gripper1":
            raise InvalidArtifact("formal B0-H requires absolute action8 targets")
        if (
            norm.get("action_target_contract_id") != ACTION_TARGET_CONTRACT_ID
            or norm.get("action_target_contract_sha256")
            != ACTION_TARGET_CONTRACT_SHA256
        ):
            raise InvalidArtifact(
                "normalization is not tied to the pinned action-target contract"
            )
        population = norm.get("population")
        if population not in (
            "all_frames_all_50_demos_all_11_tasks_both_local_arms",
            "all_causal_pairs_all_50_demos_all_11_tasks_both_local_arms",
        ):
            raise InvalidArtifact("normalization population is not the full Duo corpus")
        if population.startswith("all_causal_pairs") and int(norm.get("action_lag_rows", -1)) != 1:
            raise InvalidArtifact("causal-pair normalization is not bound to lag-one targets")
        for task in TASKS:
            row = manifest["tasks"].get(task, {})
            if int(row.get("episodes", -1)) != 50:
                raise InvalidArtifact(f"{task}: expected 50 demonstrations")
            if int(row.get("validation_max_steps", -1)) != MAX_STEPS[task]:
                raise InvalidArtifact(
                    f"{task}: max-step contract {row.get('validation_max_steps')} != {MAX_STEPS[task]}"
                )
            task_audit = row.get("action_target_audit")
            if (
                not isinstance(task_audit, Mapping)
                or task_audit.get("contract_id") != ACTION_TARGET_CONTRACT_ID
                or task_audit.get("contract_sha256")
                != ACTION_TARGET_CONTRACT_SHA256
                or task_audit.get("action_encoding")
                != "absolute_joint7_binary_gripper1"
            ):
                raise InvalidArtifact(f"{task}: action-target audit identity differs")
            for name in (
                "state",
                "action",
                "raw_action",
                "head",
                "left",
                "right",
                "episodes",
            ):
                path = self.s.data / task / f"{name}.npy"
                if not path.is_file() or path.stat().st_size == 0:
                    raise MissingArtifact(str(path))
        return {
            "manifest_sha256": _sha256(self.s.data / "manifest.json"),
            "episodes": 550,
            "tasks": 11,
            "action_encoding": norm["action_encoding"],
            "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
            "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
            "action_target_receipt_sha256": _sha256(audit_path),
            "training_split": "all demonstrations; no held-out demo split",
        }

    def validate_data_audit(self) -> dict[str, Any]:
        self.validate_data()
        report = _read_json(self.s.run / "data_audit.json")
        if report.get("passed") is not True:
            raise InvalidArtifact("Duo data audit did not pass")
        if (
            report.get("schema") != "duobench-act-audit-v2"
            or report.get("action_target_contract_id")
            != ACTION_TARGET_CONTRACT_ID
            or report.get("action_target_contract_sha256")
            != ACTION_TARGET_CONTRACT_SHA256
            or report.get("action_target_receipt_sha256")
            != _sha256(self.s.data / "action_target_audit.json")
        ):
            raise InvalidArtifact(
                "Duo data audit is not bound to the pinned action-target receipt"
            )
        required = {
            "eleven_tasks",
            "all_550_episodes",
            "normalization_finite",
            "normalization_nonzero",
            "dimensions",
            "episode_contiguous",
            "gripper_binary",
            "image_uint8",
            "action_target_contract",
            "action_target_receipt",
            "action_target_receipt_hash",
            "raw_action_provenance",
            "controller_equivalent_reconstruction",
            "action_receipt_counts",
            "formal_action_counts",
            "source_parquet_identity",
        }
        checks = report.get("checks", {})
        if not required.issubset(checks) or not all(checks.get(key) is True for key in required):
            raise InvalidArtifact(f"data audit is missing a required check: {sorted(required - set(checks))}")
        balance_path = self.s.run / "data_balance_cycle.json"
        balance = _read_json(balance_path)
        if (
            balance.get("schema") != "before-we-act.duobench.b0h-balance-cycle/1"
            or balance.get("status") != "PASSED"
            or int(balance.get("effective_batch", -1)) != 48
            or int(balance.get("world_size", -1)) != 4
            or int(balance.get("local_batch", -1)) != 12
            or int(balance.get("cycle_updates", -1)) != 11
        ):
            raise InvalidArtifact("B0-H matched-compute balance-cycle receipt differs")
        expected = {task: 48 for task in TASKS}
        if balance.get("rows_per_task_over_cycle") != expected:
            raise InvalidArtifact("B0-H 11-update task balance is not exact")
        return {
            "audit_sha256": _sha256(self.s.run / "data_audit.json"),
            "balance_cycle_sha256": _sha256(balance_path),
            "checks": checks,
            "effective_batch": 48,
            "local_batch": 12,
        }

    def validate_cache(self) -> dict[str, Any]:
        receipt_path = self.s.cache / "cache_receipt.json"
        receipt = _read_json(receipt_path)
        _reject_legacy_policy(receipt, "DINO cache receipt")
        if receipt.get("schema") != "before-we-act.duobench.dino-cache/1":
            raise InvalidArtifact("wrong DINO cache schema")
        expected = {
            "status": "PASSED",
            "encoder": "dinov3_vitb16_frozen",
            "image_height": 224,
            "image_width": 224,
            "feature_width": 768,
            "episodes": 550,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise InvalidArtifact(f"DINO cache differs at {key}: {receipt.get(key)!r}")
        counts = receipt.get("episodes_per_task", {})
        if any(int(counts.get(task, 0)) != 50 for task in TASKS):
            raise InvalidArtifact("DINO cache is incomplete by task")
        if receipt.get("act_provider_allowed") not in (None, False):
            raise InvalidArtifact("DINO cache receipt allows an ACT provider")
        if receipt.get("strict_dino_contract") is not True:
            raise InvalidArtifact("DINO cache receipt does not enable strict_dino_contract")
        if receipt.get("image_preprocess_id") != IMAGE_PREPROCESS_ID:
            raise InvalidArtifact("DINO cache image preprocessing identity differs")
        if receipt.get("dino_normalization_id") != DINO_NORMALIZATION_ID:
            raise InvalidArtifact("DINO cache normalization identity differs")
        return {"cache_receipt_sha256": _sha256(receipt_path), **expected}

    def validate_b0h_checkpoint(self, path: Path, *, stage: str, updates: int) -> dict[str, Any]:
        status = _read_json(path.parent / "status.json")
        if status.get("status") != "PASSED" or int(status.get("update", -1)) != updates:
            raise MissingArtifact(f"B0-H {stage} has not reached update {updates}")
        saved = _load_torch_payload(path, f"Duo DINO B0-H {stage}")
        config = saved.get("config", {})
        if not isinstance(config, Mapping):
            raise InvalidArtifact("B0-H checkpoint config is not a mapping")
        _reject_legacy_policy(config, f"B0-H {stage} checkpoint config")
        if saved.get("format") != "before-we-act.duobench.dino-b0h/1":
            raise InvalidArtifact(f"wrong B0-H checkpoint format: {path}")
        expected = {
            "format_version": "before-we-act.duobench.dino-b0h-config/1",
            "policy_family": "TemporalHistoryPolicy",
            "method_family": "CARE",
            "architecture": "TemporalHistoryPolicy_hidden_residual",
            "stage": stage,
            "state_dim": 8,
            "action_dim": 8,
            "horizon": 100,
            "history_steps": 16,
            "variant": "hidden_residual",
            "vision": "dinov3_vitb16_frozen",
            "strict_dino_contract": True,
            "action_encoding": "absolute_joint7_binary_gripper1",
            "effective_batch": 48,
            "all_550_demonstrations": True,
        }
        if int(saved.get("update", -1)) != updates:
            raise InvalidArtifact(f"B0-H checkpoint update differs: {saved.get('update')}")
        for key, value in expected.items():
            if config.get(key) != value:
                raise InvalidArtifact(f"B0-H config differs at {key}: {config.get(key)!r}")
        sampling = str(config.get("sampling", "")).lower()
        if "11" not in sampling or "rotat" not in sampling or "extra" not in sampling:
            raise InvalidArtifact("B0-H checkpoint does not record the 11-update rotating-extra sampler")
        policy = str(config.get("policy_contract", ""))
        if "strictly_decentralized" not in policy or "own_wrist" not in policy:
            raise InvalidArtifact("B0-H checkpoint does not freeze the strict-local policy contract")
        state = saved.get("model")
        if not isinstance(state, Mapping):
            raise InvalidArtifact("B0-H checkpoint has no TemporalHistoryPolicy state dict")
        state_keys = tuple(str(key) for key in state)
        signatures = (
            "vision.",
            "history_encoder.",
            "history_action.",
            "decoder.",
            "hidden_residual.",
        )
        missing_signatures = [
            prefix for prefix in signatures if not any(key.startswith(prefix) for key in state_keys)
        ]
        if missing_signatures:
            raise InvalidArtifact(
                "B0-H checkpoint state dict is not TemporalHistoryPolicy_hidden_residual; "
                f"missing prefixes {missing_signatures}"
            )
        receipt_path = path.parent / "checkpoint_receipt.json"
        receipt = _read_json(receipt_path)
        receipt_expected = {
            "schema": "before-we-act.duobench.dino-b0h-checkpoint/1",
            "status": "PASSED",
            "stage": stage,
            "update": updates,
            "policy_family": "TemporalHistoryPolicy",
            "method_family": "CARE",
            "architecture": "TemporalHistoryPolicy_hidden_residual",
            "vision_backbone": "dinov3_vitb16_frozen",
            "image_preprocess_id": IMAGE_PREPROCESS_ID,
            "dino_normalization_id": DINO_NORMALIZATION_ID,
            "action_encoding": "absolute_joint7_binary_gripper1",
            "strictly_decentralized": True,
            "strict_dino_contract": True,
        }
        for key, value in receipt_expected.items():
            if receipt.get(key) != value:
                raise InvalidArtifact(
                    f"B0-H checkpoint receipt differs at {key}: {receipt.get(key)!r}"
                )
        _reject_legacy_policy(receipt, f"B0-H {stage} checkpoint receipt")
        return {
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": _sha256(path),
            "checkpoint_receipt_sha256": _sha256(receipt_path),
            "update": updates,
            "stage": stage,
            "policy_family": "TemporalHistoryPolicy",
            "architecture": "TemporalHistoryPolicy_hidden_residual",
            "vision_backbone": "dinov3_vitb16_frozen",
            "act_provider_allowed": False,
            "policy_contract": policy,
        }

    def validate_b0h_smoke(self) -> dict[str, Any]:
        return self.validate_b0h_checkpoint(
            self.s.smoke_checkpoint, stage="smoke", updates=self.s.b0h_smoke_updates
        )

    def validate_b0h_closed_loop_smoke(self) -> dict[str, Any]:
        report_path = self.s.run / "b0h" / "smoke" / "smoke_report.json"
        report = _read_json(report_path)
        if (
            report.get("schema") != "before-we-act.duobench.dino-b0h-smoke/1"
            or report.get("status") != "PASSED"
            or report.get("passed") is not True
        ):
            raise InvalidArtifact("B0-H closed-loop smoke did not pass")
        _reject_legacy_policy(report, "B0-H closed-loop smoke")
        required = (
            "strictly_decentralized",
            "native_camera_projection",
            "train_eval_normalization_match",
            "absolute_action_contract",
            "task_specific_max_steps",
        )
        checks = report.get("checks", report)
        if not all(checks.get(key) is True for key in required):
            raise InvalidArtifact(f"B0-H smoke lacks contract checks: {required}")
        checkpoint_checks = report.get("checkpoint", {}).get("checks", {})
        required_checkpoint = (
            "format",
            "policy_family",
            "method_family",
            "architecture",
            "vision_backbone",
            "action_encoding",
            "strict_local",
            "strict_dino_contract",
        )
        if not all(checkpoint_checks.get(key) is True for key in required_checkpoint):
            raise InvalidArtifact(
                f"B0-H smoke checkpoint provenance differs: {required_checkpoint}"
            )
        return {
            "smoke_report_sha256": _sha256(report_path),
            "checks": {k: checks[k] for k in required},
            "checkpoint_checks": {
                key: checkpoint_checks[key] for key in required_checkpoint
            },
        }

    def validate_b0h_formal(self) -> dict[str, Any]:
        return self.validate_b0h_checkpoint(
            self.s.b0h_checkpoint, stage="formal", updates=B0H_UPDATES
        )

    def _validate_bcore_payload(
        self,
        path: Path,
        *,
        context: str,
        expected_update: int | None = None,
        require_deployment_format: bool = False,
    ) -> dict[str, Any]:
        """Validate a *separate* Duo PredictiveTeamBeliefPolicy payload.

        The historical MARS/CARE code can deserialize several unrelated
        checkpoints that happen to contain a ``model`` mapping.  Checking only
        that mapping (or only a filename) would let an ACT/ConvNeXt artifact
        silently become the B-core reference.  The Duo adapter therefore has
        to carry an explicit family, benchmark, frozen-DINO source and B0-H
        provenance hash.  We intentionally do not accept a B0-H payload as a
        B-core alias, even if its tensors happen to be loadable.
        """

        saved = _load_torch_payload(path, context)
        _reject_legacy_policy(saved, f"{context} checkpoint")
        format_value = str(saved.get("format") or saved.get("format_version") or "")
        if not format_value:
            raise InvalidArtifact(f"{context} checkpoint has no format/version")
        if format_value == B0H_CHECKPOINT_FORMAT:
            raise InvalidArtifact(
                f"{context} received a B0-H checkpoint; B-core must be an independent "
                "PredictiveTeamBeliefPolicy"
            )
        accepted_formats = (
            {BCORE_DEPLOYMENT_FORMAT}
            if require_deployment_format
            else {BCORE_TRAINING_FORMAT, BCORE_DEPLOYMENT_FORMAT}
        )
        if format_value not in accepted_formats:
            raise InvalidArtifact(
                f"{context} has non-Duo B-core format {format_value!r}; "
                f"expected one of {sorted(accepted_formats)}"
            )
        benchmark = _find_metadata(saved, "benchmark_adapter")
        if benchmark != "DuoBench":
            raise InvalidArtifact(f"{context} checkpoint is not tagged for DuoBench")
        _require_dino_local_metadata(
            saved,
            context=f"{context} checkpoint",
            policy_family="PredictiveTeamBeliefPolicy",
            method_family="CARE",
            action_encodings=(
                "absolute_joint7_binary_gripper1",
                "joint_residual7_gripper_absolute1",
            ),
        )
        reference_hash = (
            _find_metadata(saved, "source_b0h_checkpoint_sha256")
            or _find_metadata(saved, "b0h_checkpoint_sha256")
            or _find_metadata(saved, "reference_checkpoint_sha256")
        )
        if reference_hash != _sha256(self.s.b0h_checkpoint):
            raise InvalidArtifact(
                f"{context} checkpoint was not derived from this formal B0-H checkpoint"
            )
        if expected_update is not None and int(saved.get("update", -1)) != expected_update:
            raise InvalidArtifact(
                f"{context} checkpoint update {saved.get('update')} != {expected_update}"
            )
        model = saved.get("model")
        if not isinstance(model, Mapping):
            raise InvalidArtifact(f"{context} checkpoint has no model state")
        keys = tuple(str(key) for key in model)
        if not any(
            token in key
            for key in keys
            for token in ("belief_core", "belief_residual", "predictive")
        ):
            raise InvalidArtifact(
                f"{context} checkpoint state does not contain PredictiveTeamBeliefPolicy tensors"
            )
        return {
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": _sha256(path),
            "format": format_value,
            "policy_family": "PredictiveTeamBeliefPolicy",
            "vision_backbone": "dinov3_vitb16_frozen",
            "benchmark_adapter": "DuoBench",
            "source_b0h_checkpoint_sha256": reference_hash,
            "act_provider_allowed": False,
        }

    def _validation_summary(
        self,
        root: Path,
        *,
        task_schema: str,
        summary_schema: str,
        episodes: int,
        checkpoint: Path,
        policy_family: str,
        architecture: str,
    ) -> dict[str, Any]:
        per_task: dict[str, dict[str, Any]] = {}
        total_successes = 0
        checkpoint_sha256 = _sha256(checkpoint)
        for task in TASKS:
            path = root / f"{task}.json"
            row = _read_json(path)
            if row.get("schema") != task_schema or row.get("status") != "complete":
                raise InvalidArtifact(f"wrong validation receipt for {task}")
            if row.get("task") != task or int(row.get("episodes", -1)) != episodes:
                raise InvalidArtifact(f"validation episode/task mismatch for {task}")
            provenance = {
                "policy_family": row.get("policy_family"),
                "method_family": row.get("method_family"),
                "architecture": row.get("architecture"),
                "vision_backbone": row.get("vision_backbone"),
                "strictly_decentralized": row.get("strictly_decentralized"),
                "act_provider_allowed": row.get("act_provider_allowed"),
            }
            _reject_legacy_policy(provenance, f"{task} validation provenance")
            expected_provenance = {
                "policy_family": policy_family,
                "method_family": "CARE",
                "architecture": architecture,
                "vision_backbone": "dinov3_vitb16_frozen",
                "strictly_decentralized": True,
                "act_provider_allowed": False,
            }
            for key, value in expected_provenance.items():
                if provenance.get(key) != value:
                    raise InvalidArtifact(
                        f"{task} validation provenance differs at {key}: "
                        f"{provenance.get(key)!r}"
                    )
            if row.get("checkpoint_sha256") != checkpoint_sha256:
                raise InvalidArtifact(f"{task} validation used a different checkpoint")
            try:
                result_checkpoint = Path(str(row.get("checkpoint", ""))).resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise InvalidArtifact(f"{task} validation checkpoint path is invalid") from error
            if result_checkpoint != checkpoint.resolve(strict=True):
                raise InvalidArtifact(f"{task} validation checkpoint path differs")
            successes = int(row.get("successes", -1))
            if not 0 <= successes <= episodes:
                raise InvalidArtifact(f"invalid success count for {task}: {successes}")
            rows = row.get("rows")
            if not isinstance(rows, list) or len(rows) != episodes:
                raise InvalidArtifact(f"validation rows incomplete for {task}")
            expected_seeds = [self.s.seed_start + index for index in range(episodes)]
            actual_seeds = [int(item.get("seed", -1)) for item in rows]
            if actual_seeds != expected_seeds:
                raise InvalidArtifact(f"validation seed protocol differs for {task}")
            if any(int(item.get("max_steps", -1)) != MAX_STEPS[task] for item in rows):
                raise InvalidArtifact(f"task-specific max steps differ for {task}")
            if sum(int(bool(item.get("success"))) for item in rows) != successes:
                raise InvalidArtifact(f"success rows/count disagree for {task}")
            total_successes += successes
            per_task[task] = {
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / episodes,
                "max_steps": MAX_STEPS[task],
                "result_sha256": _sha256(path),
            }
        summary = {
            "schema": summary_schema,
            "status": "complete",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "policy_family": policy_family,
            "method_family": "CARE",
            "architecture": architecture,
            "vision_backbone": "dinov3_vitb16_frozen",
            "strictly_decentralized": True,
            "act_provider_allowed": False,
            "episodes_per_task": episodes,
            "total_episodes": episodes * len(TASKS),
            "total_successes": total_successes,
            "micro_success_rate": total_successes / (episodes * len(TASKS)),
            "macro_success_rate": sum(row["success_rate"] for row in per_task.values()) / len(TASKS),
            "tasks": per_task,
            "seed_start_per_task": self.s.seed_start,
            "task_specific_max_steps": dict(MAX_STEPS),
            "completed_at_utc": _utc_now(),
        }
        _atomic_json(root / "summary.json", summary)
        return summary

    def validate_b0h_validation(self) -> dict[str, Any]:
        summary = self._validation_summary(
            self.s.run / "b0h" / "validation20",
            task_schema="before-we-act.duobench.dino-b0h-validation20-task/1",
            summary_schema="before-we-act.duobench.dino-b0h-validation20/1",
            episodes=self.s.b0h_probe_episodes,
            checkpoint=self.s.b0h_checkpoint,
            policy_family="TemporalHistoryPolicy",
            architecture="TemporalHistoryPolicy_hidden_residual",
        )
        return {
            "summary": str((self.s.run / "b0h" / "validation20" / "summary.json").resolve()),
            "total_episodes": summary["total_episodes"],
            "total_successes": summary["total_successes"],
            "macro_success_rate": summary["macro_success_rate"],
        }

    def validate_b0h_gate(self) -> dict[str, Any]:
        summary = _read_json(self.s.run / "b0h" / "validation20" / "summary.json")
        if (
            summary.get("policy_family") != "TemporalHistoryPolicy"
            or summary.get("method_family") != "CARE"
            or summary.get("architecture") != "TemporalHistoryPolicy_hidden_residual"
            or summary.get("vision_backbone") != "dinov3_vitb16_frozen"
            or summary.get("strictly_decentralized") is not True
            or summary.get("act_provider_allowed") is not False
        ):
            raise InvalidArtifact("B0-H success gate received non-formal reference provenance")
        _reject_legacy_policy(summary, "B0-H Validation20 summary")
        if int(summary.get("total_episodes", 0)) != self.s.b0h_probe_episodes * 11:
            raise InvalidArtifact("B0-H probe is not complete")
        successes = int(summary.get("total_successes", 0))
        gate = {
            "schema": "before-we-act.duobench.b0h-success-gate/1",
            "status": "PASSED",
            "successes": successes,
            "episodes": summary["total_episodes"],
            "downstream_authorized": True,
            "minimum_required_successes": 1,
        }
        if successes <= 0:
            # A short/under-trained B0-H probe is diagnostic evidence, not an
            # interface or numerical failure.  Keep the result visible in the
            # receipt while allowing the requested full CARE run to continue;
            # no downstream report may relabel this warning as success.
            gate["warning"] = "zero_success_probe_continued_for_full_training"
        _atomic_json(self.s.run / "b0h" / "probe_gate.json", gate)
        return gate

    def validate_bcore_cache(self) -> dict[str, Any]:
        path = self.s.bcore_root / "cache" / "cache_receipt.json"
        row = _read_json(path)
        _reject_legacy_policy(row, "Duo B-core cache receipt")
        if row.get("schema") != "before-we-act.duobench.bcore-cache/1" or row.get("status") != "PASSED":
            raise InvalidArtifact("wrong/incomplete Duo B-core cache receipt")
        if int(row.get("episodes", -1)) != 550:
            raise InvalidArtifact("Duo B-core cache must cover all 550 demonstrations")
        if row.get("b0h_checkpoint_sha256") != _sha256(self.s.b0h_checkpoint):
            raise InvalidArtifact("B-core cache was not built from selected formal B0-H")
        _require_dino_local_metadata(
            row,
            context="Duo B-core cache receipt",
            policy_family="TemporalHistoryPolicy",
            method_family="CARE",
            action_encodings=("absolute_joint7_binary_gripper1",),
        )
        # The cache is produced from B0-H, but must declare the downstream
        # model family it is preparing; this prevents a cache-only ACT path
        # from being relabeled as B-core later.
        cache_policy = str(
            row.get("bcore_policy_family")
            or row.get("target_policy_family")
            or row.get("consumer_policy_family")
            or row.get("downstream_policy_family")
            or ""
        )
        if "PredictiveTeamBeliefPolicy" not in cache_policy:
            raise InvalidArtifact("B-core cache does not target PredictiveTeamBeliefPolicy")
        return {
            "cache_receipt_sha256": _sha256(path),
            "episodes": 550,
            "reference_policy_family": "TemporalHistoryPolicy",
            "bcore_policy_family": "PredictiveTeamBeliefPolicy",
            "act_provider_allowed": False,
        }

    def validate_bcore_smoke(self) -> dict[str, Any]:
        root = self.s.bcore_root / "smoke"
        status = _read_json(root / "status.json")
        if status.get("status") not in ("PASSED_SMOKE", "PASSED"):
            raise InvalidArtifact("Duo B-core smoke did not pass")
        if int(status.get("update", -1)) < 4:
            raise InvalidArtifact("Duo B-core smoke did not complete four updates")
        checkpoint = root / "checkpoint_latest.pt"
        if not checkpoint.is_file():
            raise MissingArtifact(str(checkpoint))
        checkpoint_validation = self._validate_bcore_payload(
            checkpoint,
            context="Duo B-core smoke",
            expected_update=4,
        )
        report_path = root / "smoke_report.json"
        report = _read_json(report_path) if report_path.is_file() else {}
        if report and report.get("status") not in ("PASSED", "complete", "passed"):
            raise InvalidArtifact("Duo B-core smoke closed-loop/contract report failed")
        if report:
            _require_dino_local_metadata(
                report,
                context="Duo B-core smoke report",
                policy_family="PredictiveTeamBeliefPolicy",
                method_family="CARE",
                action_encodings=(
                    "absolute_joint7_binary_gripper1",
                    "joint_residual7_gripper_absolute1",
                ),
            )
        return {
            **checkpoint_validation,
            "updates": int(status["update"]),
            "report_sha256": _sha256(report_path) if report_path.is_file() else None,
        }

    def validate_bcore_training(self) -> dict[str, Any]:
        rows = {}
        for seed in BCORE_SEEDS:
            root = self.s.bcore_root / "training" / f"seed_{seed}"
            status = _read_json(root / "status.json")
            if int(status.get("update", -1)) != BCORE_UPDATES:
                raise MissingArtifact(f"B-core seed {seed} has not completed 120000 updates")
            if status.get("status") not in ("COMPLETED", "PASSED", "PLATFORM_REACHED", "INCONCLUSIVE_TRAINING_NOT_CONVERGED", "SATURATED_BY_OVERFIT"):
                raise InvalidArtifact(f"B-core seed {seed} has invalid terminal status")
            deployment = root / "deployment_checkpoint.pt"
            if not deployment.is_file():
                raise MissingArtifact(str(deployment))
            selected_update = status.get("selected_update")
            checkpoint_validation = self._validate_bcore_payload(
                deployment,
                context=f"Duo B-core seed {seed}",
                expected_update=(int(selected_update) if selected_update is not None else None),
                require_deployment_format=True,
            )
            selected_payload_update = int(
                _load_torch_payload(deployment, f"Duo B-core seed {seed}").get("update", -1)
            )
            if not 1 <= selected_payload_update <= BCORE_UPDATES:
                raise InvalidArtifact(f"B-core seed {seed} selected update is invalid")
            rows[str(seed)] = {
                "status": status["status"],
                "selected_update": selected_payload_update,
                **checkpoint_validation,
            }
        return {"seeds": rows, "updates_per_seed": BCORE_UPDATES}

    def validate_bcore_select(self) -> dict[str, Any]:
        receipt_path = self.s.bcore_root / "selected" / "selection_receipt.json"
        receipt = _read_json(receipt_path)
        _reject_legacy_policy(receipt, "Duo B-core selection receipt")
        if receipt.get("schema") != "before-we-act.duobench.bcore-selection/1" or receipt.get("status") != "PASSED":
            raise InvalidArtifact("wrong B-core selection receipt")
        if receipt.get("closed_loop_results_used_for_selection") is not False:
            raise InvalidArtifact("B-core selection must be offline and pre-closed-loop")
        if int(receipt.get("selected_seed", -1)) not in BCORE_SEEDS:
            raise InvalidArtifact("B-core selected seed is outside the frozen three-seed set")
        if not self.s.bcore_checkpoint.is_file():
            raise MissingArtifact(str(self.s.bcore_checkpoint))
        if receipt.get("deployment_checkpoint_sha256") != _sha256(self.s.bcore_checkpoint):
            raise InvalidArtifact("B-core selection/deployment hash differs")
        selected_seed = int(receipt["selected_seed"])
        selected_source = self.s.bcore_root / "training" / f"seed_{selected_seed}" / "deployment_checkpoint.pt"
        if not selected_source.is_file():
            raise MissingArtifact(str(selected_source))
        if receipt.get("source_checkpoint_sha256") != _sha256(selected_source):
            raise InvalidArtifact("B-core selection source seed/hash differs")
        checkpoint_validation = self._validate_bcore_payload(
            self.s.bcore_checkpoint,
            context="selected Duo B-core",
            require_deployment_format=True,
        )
        _require_dino_local_metadata(
            receipt,
            context="Duo B-core selection receipt",
            policy_family="PredictiveTeamBeliefPolicy",
            method_family="CARE",
            action_encodings=("absolute_joint7_binary_gripper1",),
        )
        return {
            "selected_seed": receipt["selected_seed"],
            "selection_receipt_sha256": _sha256(receipt_path),
            **checkpoint_validation,
        }

    def validate_bcore_validation(self) -> dict[str, Any]:
        self._validate_bcore_payload(
            self.s.bcore_checkpoint,
            context="B-core Validation20",
            require_deployment_format=True,
        )
        summary = self._validation_summary(
            self.s.bcore_root / "validation20",
            task_schema="before-we-act.duobench.bcore-validation20-task/1",
            summary_schema="before-we-act.duobench.bcore-validation20/1",
            episodes=self.s.bcore_probe_episodes,
            checkpoint=self.s.bcore_checkpoint,
            policy_family="PredictiveTeamBeliefPolicy",
            architecture="PredictiveTeamBeliefPolicy",
        )
        return {
            "summary": str((self.s.bcore_root / "validation20" / "summary.json").resolve()),
            "total_episodes": summary["total_episodes"],
            "total_successes": summary["total_successes"],
            "macro_success_rate": summary["macro_success_rate"],
        }

    def validate_bcore_gate(self) -> dict[str, Any]:
        summary = _read_json(self.s.bcore_root / "validation20" / "summary.json")
        if (
            summary.get("policy_family") != "PredictiveTeamBeliefPolicy"
            or summary.get("method_family") != "CARE"
            or summary.get("architecture") != "PredictiveTeamBeliefPolicy"
            or summary.get("vision_backbone") != "dinov3_vitb16_frozen"
            or summary.get("strictly_decentralized") is not True
            or summary.get("act_provider_allowed") is not False
        ):
            raise InvalidArtifact("B-core success gate received non-formal policy provenance")
        _reject_legacy_policy(summary, "B-core Validation20 summary")
        successes = int(summary.get("total_successes", 0))
        if int(summary.get("total_episodes", 0)) != self.s.bcore_probe_episodes * 11:
            raise InvalidArtifact("B-core probe is incomplete")
        gate = {
            "schema": "before-we-act.duobench.bcore-success-gate/1",
            "status": "PASSED",
            "successes": successes,
            "episodes": summary["total_episodes"],
            "downstream_authorized": True,
            "minimum_required_successes": 1,
        }
        if successes <= 0:
            gate["warning"] = "zero_success_probe_continued_for_full_training"
        _atomic_json(self.s.bcore_root / "probe_gate.json", gate)
        return gate

    def validate_branch_manifest(self, root: Path, expected: int) -> dict[str, Any]:
        manifest_path = root / "manifest.json"
        manifest = _read_json(manifest_path)
        _reject_legacy_policy(manifest, "CARE branch manifest")
        if manifest.get("status") != "COMPLETE" or int(manifest.get("family_count", -1)) != expected:
            raise MissingArtifact(f"branch collection incomplete: {manifest_path}")
        # Branches are generated by the selected independent B-core.  A branch
        # manifest that merely says "CARE" is ambiguous and could point at the
        # legacy ACT/ConvNeXt collector, so the source family is explicit.
        if manifest.get("reference_policy_family") != "PredictiveTeamBeliefPolicy":
            raise InvalidArtifact("CARE branch manifest is not sourced from PredictiveTeamBeliefPolicy")
        if manifest.get("method_family") != "CARE":
            raise InvalidArtifact("CARE branch manifest does not identify method_family=CARE")
        if manifest.get("vision") != "dinov3_vitb16_frozen":
            raise InvalidArtifact("CARE branch manifest does not use frozen DINOv3")
        if manifest.get("strictly_decentralized") is not True:
            raise InvalidArtifact("CARE branch manifest is not strictly decentralized")
        if manifest.get("act_provider_allowed") is not False:
            raise InvalidArtifact("CARE branch manifest permits an ACT provider")
        _require_dino_local_metadata(
            manifest,
            context="CARE branch manifest",
            policy_family="PredictiveTeamBeliefPolicy",
            method_family="CARE",
        )
        _require_action_target_metadata(manifest, "CARE branch manifest")
        _require_care_memory_metadata(manifest, "CARE branch manifest")
        if manifest.get("source_policy_action_encoding") != "absolute_joint7_binary_gripper1":
            raise InvalidArtifact("CARE branch source action encoding differs")
        if manifest.get("action_encoding") != "joint_residual7_gripper_absolute1":
            raise InvalidArtifact("CARE branch action encoding differs")
        source_hash = manifest.get("provider_checkpoint_sha256") or manifest.get(
            "bcore_checkpoint_sha256"
        )
        if source_hash != _sha256(self.s.bcore_checkpoint):
            raise InvalidArtifact("CARE branch manifest was built from a different B-core")
        if int(manifest.get("branches_per_family", -1)) != 24:
            raise InvalidArtifact("CARE family must contain 6 candidates x 2 regimes x 2 repeats")
        if tuple(manifest.get("tasks", ())) not in (TASKS, tuple(TASKS)):
            # A smoke family is allowed to declare only its one task below.
            if expected != 1 or tuple(manifest.get("tasks", ())) != ("ball_maze",):
                raise InvalidArtifact("branch task manifest differs")
        families = manifest.get("families")
        if not isinstance(families, list) or len(families) != expected:
            raise InvalidArtifact("branch family rows are incomplete")
        for row in families:
            path = Path(row.get("path", ""))
            npz = Path(row.get("npz", ""))
            if not path.is_absolute():
                path = root / path
            if not npz.is_absolute():
                npz = root / npz
            if not path.is_file() or not npz.is_file():
                raise MissingArtifact(f"branch family files missing: {path}/{npz}")
            family = _read_json(path)
            _reject_legacy_policy(family, f"CARE branch family {path.name}")
            if family.get("reference_policy_family") != "PredictiveTeamBeliefPolicy":
                raise InvalidArtifact(f"CARE branch family {path.name} has wrong source policy")
            if family.get("method_family") != "CARE":
                raise InvalidArtifact(f"CARE branch family {path.name} has wrong method family")
            if family.get("vision") != "dinov3_vitb16_frozen":
                raise InvalidArtifact(f"CARE branch family {path.name} has wrong vision")
            if family.get("act_provider_allowed") is not False:
                raise InvalidArtifact(f"CARE branch family {path.name} allows ACT provider")
            _require_action_target_metadata(family, f"CARE branch family {path.name}")
            _require_care_memory_metadata(family, f"CARE branch family {path.name}")
            try:
                import numpy as np
                with np.load(npz, allow_pickle=False) as arrays:
                    if tuple(arrays["memory"].shape) != (DUO_CARE_MEMORY_TOKENS, DUO_CARE_MEMORY_WIDTH):
                        raise InvalidArtifact(f"CARE branch family {path.name} memory shape differs")
                    if tuple(arrays["memory_mask"].shape) != (DUO_CARE_MEMORY_TOKENS,):
                        raise InvalidArtifact(f"CARE branch family {path.name} memory mask differs")
            except KeyError as error:
                raise InvalidArtifact(f"CARE branch family {path.name} missing memory arrays") from error
            _require_dino_local_metadata(
                family,
                context=f"CARE branch family {path.name}",
                policy_family="PredictiveTeamBeliefPolicy",
                method_family="CARE",
            )
            family_hash = family.get("provider_checkpoint_sha256") or family.get(
                "bcore_checkpoint_sha256"
            )
            if family_hash != _sha256(self.s.bcore_checkpoint):
                raise InvalidArtifact(f"CARE branch family {path.name} uses a different B-core")
        return {"families": expected, "manifest_sha256": _sha256(manifest_path)}

    def validate_branch_smoke(self) -> dict[str, Any]:
        root = self.s.run / "care" / "smoke" / "branches"
        result = self.validate_branch_manifest(root / "families", 1)
        audit_path = self.s.run / "care" / "smoke" / "branch_signal_audit.json"
        audit = _read_json(audit_path)
        if audit.get("status") != "PASSED":
            raise InvalidArtifact("CARE branch smoke signal audit failed")
        result["signal_audit_sha256"] = _sha256(audit_path)
        return result

    def validate_branch_collection(self) -> dict[str, Any]:
        return self.validate_branch_manifest(
            self.s.branch_root / "families", self.s.families_per_task * len(TASKS)
        )

    def validate_branch_prepare(self) -> dict[str, Any]:
        if not self.s.prepared_care.is_file():
            raise MissingArtifact(str(self.s.prepared_care))
        receipt_path = self.s.prepared_care.with_suffix(".receipt.json")
        receipt = _read_json(receipt_path)
        _reject_legacy_policy(receipt, "prepared CARE receipt")
        expected = self.s.families_per_task * len(TASKS)
        if receipt.get("status") != "PASSED" or int(receipt.get("families", -1)) != expected:
            raise InvalidArtifact("prepared CARE tensor receipt is incomplete")
        if receipt.get("prepared_data_sha256") != _sha256(self.s.prepared_care):
            raise InvalidArtifact("prepared CARE tensor hash differs")
        payload = _load_torch_payload(self.s.prepared_care, "prepared CARE")
        if payload.get("format_version") != "before-we-act.care-duobench-prepared-data/1":
            raise InvalidArtifact("prepared CARE tensor has the wrong DuoBench format")
        if tuple(payload.get("tasks", ())) != TASKS:
            raise InvalidArtifact("prepared CARE tensor task coverage differs")
        snapshots = payload.get("snapshot_ids", ())
        if len(snapshots) != expected:
            raise InvalidArtifact("prepared CARE tensor family count differs")
        manifest_ref = payload.get("manifest", {})
        manifest_path_value = (
            manifest_ref.get("path") if isinstance(manifest_ref, Mapping) else manifest_ref
        )
        if not manifest_path_value:
            raise MissingArtifact("prepared CARE source manifest path")
        manifest_path = Path(str(manifest_path_value))
        if not manifest_path.is_absolute():
            manifest_path = self.s.run / "care" / manifest_path
        prepared_manifest = _read_json(manifest_path)
        _reject_legacy_policy(prepared_manifest, "prepared CARE manifest")
        if prepared_manifest.get("format_version") != "before-we-act.care-duobench-prepared-manifest/1":
            raise InvalidArtifact("prepared CARE manifest has the wrong DuoBench format")
        if prepared_manifest.get("reference_checkpoint_sha256") != _sha256(self.s.bcore_checkpoint):
            raise InvalidArtifact("prepared CARE data was built from a different B-core")
        if prepared_manifest.get("reference_policy_family") != "PredictiveTeamBeliefPolicy":
            raise InvalidArtifact("prepared CARE manifest does not identify PredictiveTeamBeliefPolicy")
        if prepared_manifest.get("method_family") != "CARE":
            raise InvalidArtifact("prepared CARE manifest does not identify method_family=CARE")
        if prepared_manifest.get("vision") != "dinov3_vitb16_frozen":
            raise InvalidArtifact("prepared CARE manifest does not identify frozen DINOv3")
        if prepared_manifest.get("strictly_decentralized") is not True or prepared_manifest.get("act_provider_allowed") is not False:
            raise InvalidArtifact("prepared CARE manifest violates strict-local/ACT contract")
        _require_action_target_metadata(prepared_manifest, "prepared CARE manifest")
        _require_care_memory_metadata(prepared_manifest, "prepared CARE manifest")
        _require_action_target_metadata(payload, "prepared CARE tensor")
        _require_care_memory_metadata(payload, "prepared CARE tensor")
        try:
            import torch
            memory = payload.get("memory")
            mask = payload.get("memory_mask")
            if tuple(memory.shape[1:]) != (DUO_CARE_MEMORY_TOKENS, DUO_CARE_MEMORY_WIDTH):
                raise InvalidArtifact("prepared CARE memory shape differs")
            if tuple(mask.shape[1:]) != (DUO_CARE_MEMORY_TOKENS,):
                raise InvalidArtifact("prepared CARE memory mask differs")
        except AttributeError as error:
            raise InvalidArtifact("prepared CARE tensor has invalid memory arrays") from error
        return {
            "families": expected,
            "prepared_sha256": _sha256(self.s.prepared_care),
            "prepared_manifest_sha256": _sha256(manifest_path),
            "reference_policy_family": "PredictiveTeamBeliefPolicy",
            "vision": "dinov3_vitb16_frozen",
            "act_provider_allowed": False,
        }

    def validate_signal_gate(self) -> dict[str, Any]:
        path = self.s.run / "care" / "branch_signal_audit.json"
        row = _read_json(path)
        _reject_legacy_policy(row, "formal CARE branch signal audit")
        if row.get("status") != "PASSED":
            raise InvalidArtifact("formal branch signal audit failed (all-zero labels are forbidden)")
        reports = row.get("reports", [])
        if not reports or any(report.get("status") != "PASSED" for report in reports):
            raise InvalidArtifact("formal branch signal report is incomplete")
        prepared = next((report for report in reports if "horizons" in report), None)
        if prepared is None:
            raise InvalidArtifact("prepared-data signal audit is absent")
        for horizon in (8, 16, 32, 64):
            hrow = prepared.get("horizons", {}).get(str(horizon), {})
            if int(hrow.get("nonzero_nonreference_total_labels", 0)) <= 0:
                raise InvalidArtifact(f"CARE horizon {horizon} has all-zero non-reference labels")
            if int(hrow.get("pairwise_non_ties", 0)) <= 0:
                raise InvalidArtifact(f"CARE horizon {horizon} has no pairwise signal")
        return {"signal_audit_sha256": _sha256(path), "horizons": prepared["horizons"]}

    def validate_belief_smoke(self) -> dict[str, Any]:
        status_path = self.s.run / "care" / "belief_smoke" / "status.json"
        row = _read_json(status_path)
        _reject_legacy_policy(row, "CARE belief smoke status")
        if row.get("status") != "PASSED_SMOKE" or int(row.get("update", -1)) != 2:
            raise InvalidArtifact("CARE belief-head smoke did not pass two updates")
        if row.get("benchmark_adapter") != "DuoBench" or row.get("reference_policy_family") != "PredictiveTeamBeliefPolicy" or row.get("method_family") != "CARE":
            raise InvalidArtifact("CARE belief smoke is not the Duo PredictiveTeamBeliefPolicy path")
        if row.get("vision_backbone") != "dinov3_vitb16_frozen" or row.get("strictly_decentralized") is not True:
            raise InvalidArtifact("CARE belief smoke violates frozen-DINO/local contract")
        if row.get("act_provider_allowed") is not False:
            raise InvalidArtifact("CARE belief smoke permits an ACT provider")
        checkpoint = self.s.run / "care" / "belief_smoke" / "checkpoint_latest.pt"
        if not checkpoint.is_file():
            raise MissingArtifact(str(checkpoint))
        payload = _load_torch_payload(checkpoint, "CARE belief smoke")
        _require_dino_local_metadata(
            payload,
            context="CARE belief smoke checkpoint",
            policy_family="PredictiveTeamBeliefPolicy",
            method_family="CARE",
        )
        return {
            "checkpoint_sha256": _sha256(checkpoint),
            "updates": 2,
            "reference_policy_family": "PredictiveTeamBeliefPolicy",
            "vision_backbone": "dinov3_vitb16_frozen",
            "act_provider_allowed": False,
        }

    def validate_belief_training(self) -> dict[str, Any]:
        result = {}
        root = self.s.run / "care" / "belief_training"
        for variant in CARE_VARIANTS:
            for seed in CARE_SEEDS:
                path = root / variant / f"seed_{seed}" / "status.json"
                row = _read_json(path)
                if row.get("status") != "COMPLETED" or int(row.get("update", -1)) != CARE_UPDATES:
                    raise MissingArtifact(f"CARE belief training incomplete: {variant}/{seed}")
                _reject_legacy_policy(row, f"CARE belief status {variant}/{seed}")
                if row.get("benchmark_adapter") != "DuoBench" or row.get("reference_policy_family") != "PredictiveTeamBeliefPolicy" or row.get("method_family") != "CARE":
                    raise InvalidArtifact(f"CARE belief status {variant}/{seed} has wrong reference family")
                if row.get("vision_backbone") != "dinov3_vitb16_frozen" or row.get("strictly_decentralized") is not True or row.get("act_provider_allowed") is not False:
                    raise InvalidArtifact(f"CARE belief status {variant}/{seed} violates local/DINO contract")
                checkpoint = Path(row.get("selected_checkpoint", ""))
                if not checkpoint.is_file():
                    raise MissingArtifact(f"selected belief checkpoint missing: {checkpoint}")
                payload = _load_torch_payload(checkpoint, f"CARE belief {variant}/{seed}")
                if payload.get("format_version") != CARE_TRAINING_FORMAT:
                    raise InvalidArtifact(f"CARE belief {variant}/{seed} has wrong DuoBench checkpoint format")
                _require_dino_local_metadata(
                    payload,
                    context=f"CARE belief checkpoint {variant}/{seed}",
                    policy_family="PredictiveTeamBeliefPolicy",
                    method_family="CARE",
                )
                result[f"{variant}/{seed}"] = _sha256(checkpoint)
        return {"jobs": result, "updates_per_job": CARE_UPDATES, "all_family_training": True}

    def validate_calibration(self) -> dict[str, Any]:
        report_path = self.s.run / "care" / "offline" / "offline_report.json"
        report = _read_json(report_path)
        _reject_legacy_policy(report, "CARE offline selection report")
        if report.get("format_version") != "before-we-act.care-duobench-offline-report/1":
            raise InvalidArtifact("wrong CARE offline report")
        if report.get("all_family_training") is not True:
            raise InvalidArtifact("CARE scorer did not train on every branch family")
        if not self.s.care_checkpoint.is_file():
            raise MissingArtifact(str(self.s.care_checkpoint))
        if report.get("deployment_checkpoint_sha256") != _sha256(self.s.care_checkpoint):
            raise InvalidArtifact("CARE calibration/deployment hash differs")
        report_ref = report.get("reference_checkpoint_sha256")
        if report_ref is not None and report_ref != _sha256(self.s.bcore_checkpoint):
            raise InvalidArtifact("CARE offline report references a different B-core")
        deployment = _load_torch_payload(self.s.care_checkpoint, "CARE deployment")
        if deployment.get("format_version") != CARE_DEPLOYMENT_FORMAT:
            raise InvalidArtifact("CARE deployment checkpoint has the wrong DuoBench format")
        _require_dino_local_metadata(
            deployment,
            context="CARE deployment checkpoint",
            policy_family="PredictiveTeamBeliefPolicy",
            method_family="CARE",
        )
        _require_action_target_metadata(deployment, "CARE deployment checkpoint")
        _require_care_memory_metadata(deployment, "CARE deployment checkpoint")
        _require_action_target_metadata(report, "CARE offline selection report")
        _require_care_memory_metadata(report, "CARE offline selection report")
        deployment_ref = (
            _find_metadata(deployment, "reference_checkpoint_sha256")
            or _find_metadata(deployment, "source_bcore_checkpoint_sha256")
        )
        if deployment_ref != _sha256(self.s.bcore_checkpoint):
            raise InvalidArtifact("CARE deployment was not derived from selected B-core")
        if _find_metadata(deployment, "benchmark_adapter") != "DuoBench":
            raise InvalidArtifact("CARE deployment is not tagged for DuoBench")
        calibration = report.get("calibration", {})
        required = {
            "lower_correction",
            "selector_delta",
            "hard_safety_probability_max",
            "nominal_simultaneous_coverage",
            "primary_horizon",
        }
        if not required.issubset(calibration):
            raise InvalidArtifact("CARE calibration fields are incomplete")
        return {
            "offline_report_sha256": _sha256(report_path),
            "care_checkpoint_sha256": _sha256(self.s.care_checkpoint),
            "reference_checkpoint_sha256": _sha256(self.s.bcore_checkpoint),
            "reference_policy_family": "PredictiveTeamBeliefPolicy",
            "calibration": calibration,
        }

    def _validate_paired(self, root: Path, episodes: int, *, smoke: bool = False) -> dict[str, Any]:
        path = root / "summary.json"
        row = _read_json(path)
        expected_format = (
            "before-we-act.care-duobench-paired-validation-smoke/1"
            if smoke
            else "before-we-act.care-duobench-paired-validation20/1"
        )
        if row.get("format_version") != expected_format:
            raise InvalidArtifact("wrong paired-validation summary format")
        _reject_legacy_policy(row, "Duo paired validation summary")
        required_provenance = {
            "benchmark_adapter": "DuoBench",
            "method_family": "CARE",
            "reference_policy_family": "PredictiveTeamBeliefPolicy",
            "vision": "dinov3_vitb16_frozen",
            "vision_backbone": "dinov3_vitb16_frozen",
            "image_preprocess_id": IMAGE_PREPROCESS_ID,
            "preprocess_id": IMAGE_PREPROCESS_ID,
            "dino_normalization_id": DINO_NORMALIZATION_ID,
            "strict_dino_contract": True,
            "strictly_decentralized": True,
            "strict_local": True,
            "per_robot_independent_inputs": True,
            "act_provider_allowed": False,
            "selector_off_semantics": "candidate0_complete_bcore_reference",
            "override_contract": "at_most_one_focal_arm_per_control_step",
            "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
            "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
            "care_memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
            "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
            "bcore_checkpoint_sha256": _sha256(self.s.bcore_checkpoint),
            "care_checkpoint_sha256": _sha256(self.s.care_checkpoint),
            "prepared_manifest_sha256": _sha256(self.s.data / "manifest.json"),
            "episodes_per_task": episodes,
            "seed_start_per_task": self.s.seed_start,
            "task_specific_max_steps": MAX_STEPS,
        }
        for key, value in required_provenance.items():
            if row.get(key) != value:
                raise InvalidArtifact(
                    f"paired validation provenance differs at {key}: {row.get(key)!r}"
                )
        expected = episodes * len(TASKS)
        if row.get("status") != "complete" or int(row.get("total_pairs", -1)) != expected:
            raise MissingArtifact(f"paired validation incomplete: {path}")
        tasks = row.get("tasks", {})
        if set(tasks) != set(TASKS):
            raise InvalidArtifact("paired validation does not cover all 11 tasks")
        task_result_paths = row.get("task_results")
        if not isinstance(task_result_paths, Mapping) or set(task_result_paths) != set(TASKS):
            raise InvalidArtifact("paired validation task result files are incomplete")
        task_rows_all: list[Mapping[str, Any]] = []
        task_pairs_all: list[Mapping[str, Any]] = []
        for task in TASKS:
            if int(tasks[task].get("max_steps", -1)) != MAX_STEPS[task]:
                raise InvalidArtifact(f"paired max steps differ for {task}")
            task_path = Path(str(task_result_paths[task]))
            if not task_path.is_absolute():
                task_path = root / task_path
            task_result = _read_json(task_path)
            expected_task_schema = (
                "before-we-act.care-duobench-paired-validation-smoke-task/1"
                if smoke
                else "before-we-act.care-duobench-paired-validation20-task/1"
            )
            if task_result.get("schema") != expected_task_schema or task_result.get("status") != "complete":
                raise InvalidArtifact(f"paired task result schema/status differs for {task}")
            if task_result.get("task") != task or int(task_result.get("episodes", -1)) != episodes:
                raise InvalidArtifact(f"paired task result episode metadata differs for {task}")
            if int(task_result.get("max_steps", -1)) != MAX_STEPS[task]:
                raise InvalidArtifact(f"paired task result max steps differs for {task}")
            for key, expected_value in required_provenance.items():
                if key in {"episodes_per_task", "seed_start_per_task", "task_specific_max_steps", "prepared_manifest_sha256"}:
                    continue
                if task_result.get(key) != expected_value:
                    raise InvalidArtifact(f"paired task result provenance differs for {task}/{key}")
            result_rows = task_result.get("rows")
            result_pairs = task_result.get("pairs")
            if not isinstance(result_rows, list) or len(result_rows) != 2 * episodes:
                raise InvalidArtifact(f"paired task episode rows incomplete for {task}")
            if not isinstance(result_pairs, list) or len(result_pairs) != episodes:
                raise InvalidArtifact(f"paired task pair rows incomplete for {task}")
            expected_keys = [
                (mode, self.s.seed_start + index)
                for index in range(episodes)
                for mode in ("selector_off", "care")
            ]
            actual_keys = [(item.get("mode"), item.get("seed")) for item in result_rows if isinstance(item, Mapping)]
            if actual_keys != expected_keys:
                raise InvalidArtifact(f"paired task seed/mode rows differ for {task}")
            for item in result_rows:
                if not isinstance(item, Mapping):
                    raise InvalidArtifact(f"paired task episode row is not an object for {task}")
                if item.get("task") != task or int(item.get("max_steps", -1)) != MAX_STEPS[task]:
                    raise InvalidArtifact(f"paired task episode provenance differs for {task}")
                if item.get("action_target_contract_id") != ACTION_TARGET_CONTRACT_ID or item.get("action_target_contract_sha256") != ACTION_TARGET_CONTRACT_SHA256:
                    raise InvalidArtifact(f"paired task episode action contract differs for {task}")
                if item.get("care_memory_semantics") != "PredictiveTeamBeliefPolicy.belief.mu+belief.event_memory" or int(item.get("care_memory_tokens", -1)) != 20:
                    raise InvalidArtifact(f"paired task episode memory contract differs for {task}")
            # Pair rows must point to the exact two episode rows, including
            # success/progress/override values used in the summary.
            by_seed = {(str(item.get("mode")), int(item.get("seed"))): item for item in result_rows}
            for pair in result_pairs:
                if not isinstance(pair, Mapping):
                    raise InvalidArtifact(f"paired task pair row is not an object for {task}")
                seed = int(pair.get("seed", -1))
                off_row = by_seed.get(("selector_off", seed)); care_row = by_seed.get(("care", seed))
                if off_row is None or care_row is None:
                    raise InvalidArtifact(f"paired task pair has no matching episodes for {task}/{seed}")
                success_delta = int(bool(care_row.get("success"))) - int(bool(off_row.get("success")))
                if int(pair.get("success_delta", 99)) != success_delta:
                    raise InvalidArtifact(f"paired task success delta differs for {task}/{seed}")
                if not math.isclose(float(pair.get("progress_delta", float("nan"))), float(care_row.get("final_stage_progress", 0.0)) - float(off_row.get("final_stage_progress", 0.0)), abs_tol=1e-12):
                    raise InvalidArtifact(f"paired task progress delta differs for {task}/{seed}")
            task_rows_all.extend(item for item in result_rows if isinstance(item, Mapping))
            task_pairs_all.extend(item for item in result_pairs if isinstance(item, Mapping))
        pairs = row.get("pairs", [])
        if len(pairs) != expected:
            raise InvalidArtifact("paired seed rows are incomplete")
        if len(task_rows_all) != expected * 2 or len(task_pairs_all) != expected:
            raise InvalidArtifact("paired task files do not match summary cardinality")
        expected_seeds = [self.s.seed_start + index for index in range(episodes)]
        expected_pair_keys = [
            (task, seed) for task in TASKS for seed in expected_seeds
        ]
        actual_pair_keys = [
            (pair.get("task"), pair.get("seed"))
            for pair in pairs
            if isinstance(pair, Mapping)
        ]
        if actual_pair_keys != expected_pair_keys:
            raise InvalidArtifact(
                "paired validation must contain each frozen task/seed exactly once"
            )
        pair_provenance = {
            "strictly_decentralized": True,
            "strict_local": True,
            "per_robot_independent_inputs": True,
            "act_provider_allowed": False,
            "selector_off_semantics": "candidate0_complete_bcore_reference",
            "override_contract": "at_most_one_focal_arm_per_control_step",
            "strict_dino_contract": True,
            "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
            "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
            "care_memory_semantics": DUO_CARE_MEMORY_SEMANTICS,
            "care_memory_tokens": DUO_CARE_MEMORY_TOKENS,
            "bcore_checkpoint_sha256": required_provenance["bcore_checkpoint_sha256"],
            "care_checkpoint_sha256": required_provenance["care_checkpoint_sha256"],
        }
        for pair in pairs:
            task = str(pair["task"])
            if int(pair.get("max_steps", -1)) != MAX_STEPS[task]:
                raise InvalidArtifact(f"paired seed max steps differ for {task}")
            if any(pair.get(key) != value for key, value in pair_provenance.items()):
                raise InvalidArtifact("paired validation pair violates formal provenance")
            for key in (
                "selector_off_success",
                "care_success",
                "harmful_override",
            ):
                if not isinstance(pair.get(key), bool):
                    raise InvalidArtifact(f"paired validation pair has invalid {key}")
            delta = int(pair["care_success"]) - int(pair["selector_off_success"])
            if int(pair.get("success_delta", 99)) != delta:
                raise InvalidArtifact("paired validation success delta is inconsistent")
        for metric in (
            "selector_off_success_rate",
            "care_success_rate",
            "paired_success_improvement",
            "override_rate",
            "harmful_override_rate",
        ):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise InvalidArtifact(f"paired validation has invalid metric {metric}")
        off_rate = sum(bool(pair["selector_off_success"]) for pair in pairs) / expected
        care_rate = sum(bool(pair["care_success"]) for pair in pairs) / expected
        delta_rate = sum(int(pair["success_delta"]) for pair in pairs) / expected
        if not math.isclose(float(row["selector_off_success_rate"]), off_rate, abs_tol=1e-12):
            raise InvalidArtifact("paired selector-off rate disagrees with seed rows")
        if not math.isclose(float(row["care_success_rate"]), care_rate, abs_tol=1e-12):
            raise InvalidArtifact("paired CARE rate disagrees with seed rows")
        if not math.isclose(float(row["paired_success_improvement"]), delta_rate, abs_tol=1e-12):
            raise InvalidArtifact("paired improvement disagrees with seed rows")
        return {
            "summary_sha256": _sha256(path),
            "total_pairs": expected,
            "reference_policy_family": "PredictiveTeamBeliefPolicy",
            "bcore_checkpoint_sha256": required_provenance["bcore_checkpoint_sha256"],
            "care_checkpoint_sha256": required_provenance["care_checkpoint_sha256"],
            "selector_off_success_rate": row.get("selector_off_success_rate"),
            "care_success_rate": row.get("care_success_rate"),
            "paired_success_improvement": row.get("paired_success_improvement"),
            "override_rate": row.get("override_rate"),
        }

    def validate_paired_smoke(self) -> dict[str, Any]:
        # The Duo-specific launcher writes the same 11-task schema but one pair
        # per task.  This checks the complete runtime path before Validation20.
        return self._validate_paired(self.s.run / "care" / "paired_smoke", 1, smoke=True)

    def validate_paired20(self) -> dict[str, Any]:
        return self._validate_paired(self.s.run / "care" / "paired_validation20", 20)

    # ---------------------------------------------------------------- actions
    def action_data_prepare(self) -> None:
        self.run_command(
            "data_prepare",
            [
                self.s.python,
                "-m",
                "deployment.duo_act.prepare",
                "--dataset",
                str(self.s.dataset),
                "--output",
                str(self.s.data),
                "--image-size",
                "224",
                "--jobs",
                "8",
            ],
            (0,),
            retries=2,
        )

    def action_dataset_download(self) -> None:
        """Download the immutable DuoBench snapshot without exposing tokens.

        The HF token is intentionally inherited from the process environment
        (normally ``/workspace/.env`` sourced by the wrapper); it is never put
        in a command line, receipt, or status file.
        """
        self.s.dataset.parent.mkdir(parents=True, exist_ok=True)
        self.run_command(
            "dataset_download",
            [
                self.s.python,
                "-m",
                "deployment.duo_care.download",
                "--output",
                str(self.s.dataset),
                "--revision",
                "b741bc915d942ecadaefb4e3de6bbd716c1b8b1b",
            ],
            (0,),
            retries=3,
        )
        # Freeze a small local receipt after snapshot_download completes.  It
        # contains paths/counts only; no credential or raw command environment.
        rows: dict[str, Any] = {}
        for task in TASKS:
            root = self.s.dataset / task / "sim"
            parquet = sorted((root / "data").glob("**/*.parquet"))
            videos = {
                key: sorted((root / "videos" / key).glob("**/*.mp4"))
                for key in (
                    "observation.images.head",
                    "observation.images.left_wrist",
                    "observation.images.right_wrist",
                )
            }
            rows[task] = {
                "parquet": str(parquet[0]) if len(parquet) == 1 else None,
                "video_counts": {key: len(value) for key, value in videos.items()},
            }
        _atomic_json(
            self.s.run / "dataset_download_receipt.json",
            {
                "schema": "before-we-act.duobench.dataset-download/1",
                "status": "PASSED",
                "revision": "b741bc915d942ecadaefb4e3de6bbd716c1b8b1b",
                "tasks": rows,
                "credential_recorded": False,
                "completed_at_utc": _utc_now(),
            },
        )

    def action_data_audit(self) -> None:
        self.run_command(
            "data_audit",
            [
                self.s.python,
                "-m",
                "deployment.duo_act.audit",
                "--data",
                str(self.s.data),
                "--output",
                str(self.s.run / "data_audit.json"),
            ],
            (0,),
        )
        # Lock the matched-compute sampler contract independently of the
        # trainer.  Four extra rows rotate across tasks; over any canonical
        # 11-update cycle every task receives exactly four extras.
        from collections import Counter
        from deployment.duo_dino_reference.data import (
            DuoBalancedDistributedBatchSampler,
            EFFECTIVE_BATCH,
            BASE_SAMPLES_PER_TASK,
            load_duo_episodes,
        )

        episodes = load_duo_episodes(self.s.data, require_formal=True)
        global_sampler = DuoBalancedDistributedBatchSampler(
            episodes, updates=11, seed=20260830, rank=0, world_size=1
        )
        totals: Counter[str] = Counter()
        updates: list[dict[str, Any]] = []
        for update in range(1, 12):
            global_rows = global_sampler.requests_for_update(update)
            counts = Counter(row.task for row in global_rows)
            totals.update(counts)
            ranked = [
                DuoBalancedDistributedBatchSampler(
                    episodes,
                    updates=11,
                    seed=20260830,
                    rank=rank,
                    world_size=4,
                ).requests_for_update(update)[rank::4]
                for rank in range(4)
            ]
            # ``requests_for_update`` is global by contract; rank slicing must
            # cover it exactly with local batch 12 on every GPU.
            global_keys = [row.sample_key for row in global_rows]
            ranked_keys = [row.sample_key for rows in ranked for row in rows]
            if sorted(global_keys) != sorted(ranked_keys):
                raise InvalidArtifact(f"DDP sampler coverage differs at update {update}")
            updates.append(
                {
                    "update": update,
                    "rows": len(global_rows),
                    "local_batches": [len(rows) for rows in ranked],
                    "task_counts": {task: counts[task] for task in TASKS},
                }
            )
        expected = Counter({task: 48 for task in TASKS})
        passed = (
            EFFECTIVE_BATCH == 48
            and BASE_SAMPLES_PER_TASK == 4
            and totals == expected
            and all(row["local_batches"] == [12, 12, 12, 12] for row in updates)
        )
        receipt = {
            "schema": "before-we-act.duobench.b0h-balance-cycle/1",
            "status": "PASSED" if passed else "FAILED",
            "effective_batch": EFFECTIVE_BATCH,
            "base_rows_per_task": BASE_SAMPLES_PER_TASK,
            "rotating_extra_rows_per_update": EFFECTIVE_BATCH - BASE_SAMPLES_PER_TASK * len(TASKS),
            "world_size": 4,
            "local_batch": EFFECTIVE_BATCH // 4,
            "cycle_updates": 11,
            "rows_per_task_over_cycle": {task: totals[task] for task in TASKS},
            "updates": updates,
        }
        _atomic_json(self.s.run / "data_balance_cycle.json", receipt)
        if not passed:
            raise InvalidArtifact(f"matched-compute sampler balance failed: {receipt}")

    def action_dino_cache(self) -> None:
        self.run_command(
            "dino_cache",
            [
                self.s.python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                "-m",
                "deployment.duo_dino_reference.cache_dino",
                "--prepared-data",
                str(self.s.data),
                "--dino-model",
                str(self.s.dino_model),
                "--output",
                str(self.s.cache),
                "--image-height",
                "224",
                "--image-width",
                "224",
                "--batch-size",
                "32",
            ],
            (0, 1, 2, 3),
            retries=2,
        )

    def _b0h_train_command(self, output: Path, stage: str, updates: int) -> list[str]:
        command = [
            self.s.python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "-m",
            "deployment.duo_dino_reference.train_b0h",
            "--prepared-data",
            str(self.s.data),
            "--visual-cache",
            str(self.s.cache),
            "--dino-model",
            str(self.s.dino_model),
            "--output",
            str(output),
            "--stage",
            stage,
            "--updates",
            str(updates),
            "--workers",
            "6",
            "--save-every",
            str(updates if stage == "smoke" else 5000),
            "--seed",
            "20260830",
            "--action-loss-decay",
            "16",
            "--gripper-loss-weight",
            "0.20",
            "--gripper-logit-scale",
            "4.0",
        ]
        latest = output / "checkpoint_latest.pt"
        if latest.is_file():
            command.extend(("--resume", str(latest)))
        return command

    def action_b0h_smoke(self) -> None:
        output = self.s.run / "b0h" / "smoke"
        self.run_command(
            "b0h_smoke_train",
            self._b0h_train_command(output, "smoke", self.s.b0h_smoke_updates),
            (0, 1, 2, 3),
            retries=2,
        )

    def action_b0h_closed_loop_smoke(self) -> None:
        module = "deployment.duo_dino_reference.smoke"
        self._require_modules("b0h_smoke_closed_loop", (module,))
        self.run_command(
            "b0h_smoke_closed_loop",
            [
                self.s.python,
                "-m",
                module,
                "--prepared-data",
                str(self.s.data),
                "--visual-cache",
                str(self.s.cache),
                "--checkpoint",
                str(self.s.smoke_checkpoint),
                "--dino-model",
                str(self.s.dino_model),
                "--duobench-root",
                str(self.s.duobench_repo / "src"),
                "--output",
                str(self.s.run / "b0h" / "smoke" / "smoke_report.json"),
            ],
            (0,),
            retries=2,
        )

    def action_b0h_formal(self) -> None:
        output = self.s.run / "b0h" / "formal"
        self.run_command(
            "b0h_formal",
            self._b0h_train_command(output, "formal", B0H_UPDATES),
            (0, 1, 2, 3),
            retries=3,
        )

    def _evaluation_wave(
        self,
        *,
        module: str,
        checkpoint: Path,
        root: Path,
        episodes: int,
        prefix: str,
        extra: Sequence[str] = (),
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        pending = []
        for task in TASKS:
            path = root / f"{task}.json"
            try:
                row = _read_json(path)
                if row.get("status") == "complete" and int(row.get("episodes", -1)) == episodes:
                    continue
            except (MissingArtifact, InvalidArtifact):
                pass
            command = [
                self.s.python,
                "-m",
                module,
                "--checkpoint",
                str(checkpoint),
                "--prepared-data",
                str(self.s.data),
                "--task",
                task,
                "--output",
                str(path),
                "--episodes",
                str(episodes),
                "--seed-start",
                str(self.s.seed_start),
                "--max-steps",
                str(MAX_STEPS[task]),
                "--device",
                "cuda:0",
                "--dino-model",
                str(self.s.dino_model),
                "--duobench-root",
                str(self.s.duobench_repo / "src"),
                *extra,
            ]
            pending.append((task, command))
        for first in range(0, len(pending), 4):
            wave = [
                (f"{prefix}_{task}", command, slot)
                for slot, (task, command) in enumerate(pending[first : first + 4])
            ]
            self.run_wave(wave, retries=3)

    def action_b0h_validation(self) -> None:
        self._evaluation_wave(
            module="deployment.duo_dino_reference.evaluate",
            checkpoint=self.s.b0h_checkpoint,
            root=self.s.run / "b0h" / "validation20",
            episodes=self.s.b0h_probe_episodes,
            prefix="b0h_validation",
        )
        self.validate_b0h_validation()

    def action_bcore_cache(self) -> None:
        module = self.s.bcore_cache_module
        self._require_modules("bcore_cache", (module,))
        self.run_command(
            "bcore_cache",
            [
                self.s.python,
                "-m",
                module,
                "--prepared-data",
                str(self.s.data),
                "--visual-cache",
                str(self.s.cache),
                "--b0h-checkpoint",
                str(self.s.b0h_checkpoint),
                "--dino-model",
                str(self.s.dino_model),
                "--output",
                str(self.s.bcore_root / "cache"),
            ],
            (0, 1, 2, 3),
            retries=2,
        )

    def action_bcore_smoke(self) -> None:
        module = self.s.bcore_smoke_module
        self._require_modules("bcore_smoke", (module,))
        output = self.s.bcore_root / "smoke"
        command = [
            self.s.python,
            "-m",
            module,
            "--prepared-data",
            str(self.s.data),
            "--visual-cache",
            str(self.s.cache),
            "--bcore-cache",
            str(self.s.bcore_root / "cache"),
            "--b0h-checkpoint",
            str(self.s.b0h_checkpoint),
            "--dino-model",
            str(self.s.dino_model),
            "--output",
            str(output),
            "--stage",
            "smoke",
            "--updates",
            "4",
            "--seed",
            str(BCORE_SEEDS[0]),
        ]
        self.run_command("bcore_smoke", command, (0,), retries=2)

    def action_bcore_training(self) -> None:
        module = self.s.bcore_train_module
        self._require_modules("bcore_train_3seeds", (module,))
        jobs = []
        for gpu, seed in enumerate(BCORE_SEEDS):
            output = self.s.bcore_root / "training" / f"seed_{seed}"
            command = [
                self.s.python,
                "-m",
                module,
                "--prepared-data",
                str(self.s.data),
                "--visual-cache",
                str(self.s.cache),
                "--bcore-cache",
                str(self.s.bcore_root / "cache"),
                "--b0h-checkpoint",
                str(self.s.b0h_checkpoint),
                "--dino-model",
                str(self.s.dino_model),
                "--output",
                str(output),
                "--seed",
                str(seed),
                "--updates",
                str(BCORE_UPDATES),
                "--workers",
                "4",
                "--save-every",
                "5000",
            ]
            jobs.append((f"bcore_train_seed_{seed}", command, gpu))
        self.run_wave(jobs, retries=3)

    def action_bcore_select(self) -> None:
        module = self.s.bcore_select_module
        self._require_modules("bcore_select", (module,))
        self.run_command(
            "bcore_select",
            [
                self.s.python,
                "-m",
                module,
                "--training-root",
                str(self.s.bcore_root / "training"),
                "--b0h-checkpoint",
                str(self.s.b0h_checkpoint),
                "--output",
                str(self.s.bcore_root / "selected"),
            ],
            (0,),
        )

    def action_bcore_validation(self) -> None:
        module = self.s.bcore_evaluate_module
        self._require_modules("bcore_validation20", (module,))
        self._evaluation_wave(
            module=module,
            checkpoint=self.s.bcore_checkpoint,
            root=self.s.bcore_root / "validation20",
            episodes=self.s.bcore_probe_episodes,
            prefix="bcore_validation",
        )
        self.validate_bcore_validation()

    def _branch_command(self, output: Path, families: int, tasks: Sequence[str]) -> list[str]:
        module = self.s.branch_module
        command = [
            self.s.python,
            "-m",
            module,
            "--bcore-checkpoint",
            str(self.s.bcore_checkpoint),
            "--b0h-checkpoint",
            str(self.s.b0h_checkpoint),
            "--prepared-data",
            str(self.s.data),
            "--visual-cache",
            str(self.s.cache),
            "--dino-model",
            str(self.s.dino_model),
            "--output",
            str(output),
            "--families-per-task",
            str(families),
            "--workers",
            "4",
        ]
        for task in tasks:
            command.extend(("--task", task))
        return command

    def action_branch_smoke(self) -> None:
        self._require_modules("branch_smoke", (self.s.branch_module,))
        root = self.s.run / "care" / "smoke" / "branches"
        self.run_command(
            "branch_smoke",
            self._branch_command(root, 1, ("ball_maze",)) + ["--smoke"],
            (0, 1, 2, 3),
            retries=2,
        )
        # ``duo_dino_branch_launcher`` merges worker shards below
        # ``families/<task>``.  Looking directly under ``<root>/<task>`` would
        # always fail the smoke despite a valid one-family artifact.
        family_rows = sorted((root / "families" / "ball_maze").glob("*.json"))
        if len(family_rows) != 1:
            raise InvalidArtifact("branch smoke did not emit exactly one family")
        self.run_command(
            "branch_smoke_signal_audit",
            [
                self.s.python,
                "-m",
                "deployment.duo_care.care_signal_audit",
                "--family",
                str(family_rows[0]),
                "--output",
                str(self.s.run / "care" / "smoke" / "branch_signal_audit.json"),
            ],
            (0,),
        )

    def action_branch_collection(self) -> None:
        self._require_modules("branch_collection", (self.s.branch_module,))
        self.run_command(
            "branch_collection",
            self._branch_command(self.s.branch_root, self.s.families_per_task, TASKS),
            (0, 1, 2, 3),
            retries=3,
        )

    def action_branch_prepare(self) -> None:
        self.run_command(
            "branch_prepare",
            [
                self.s.python,
                "-m",
                "scripts.before_we_act.prepare_duo_care_training",
                "--family-root",
                str(self.s.branch_root / "families"),
                "--reference-checkpoint",
                str(self.s.bcore_checkpoint),
                "--output",
                str(self.s.prepared_care),
                "--manifest",
                str(self.s.run / "care" / "prepared_manifest.json"),
                "--expected-families",
                str(self.s.families_per_task * len(TASKS)),
            ],
            (0,),
        )

    def action_signal_gate(self) -> None:
        self.run_command(
            "branch_signal_gate",
            [
                self.s.python,
                "-m",
                "deployment.duo_care.care_signal_audit",
                "--prepared-data",
                str(self.s.prepared_care),
                "--output",
                str(self.s.run / "care" / "branch_signal_audit.json"),
            ],
            (0,),
        )

    def action_belief_smoke(self) -> None:
        output = self.s.run / "care" / "belief_smoke"
        self.run_command(
            "belief_smoke",
            [
                self.s.python,
                "-m",
                "before_we_act.train_mars_care_belief",
                "--prepared-data",
                str(self.s.prepared_care),
                "--output",
                str(output),
                "--seed",
                str(CARE_SEEDS[0]),
                "--variant",
                "care",
                "--stage",
                "smoke",
                "--updates",
                "2",
                "--batch-size",
                "2",
                "--eval-every",
                "1",
                "--save-every",
                "1",
                "--device",
                "cuda:0",
                "--benchmark-adapter",
                "DuoBench",
            ],
            (0,),
        )

    def action_belief_training(self) -> None:
        jobs = []
        root = self.s.run / "care" / "belief_training"
        for variant in CARE_VARIANTS:
            for seed in CARE_SEEDS:
                output = root / variant / f"seed_{seed}"
                try:
                    row = _read_json(output / "status.json")
                    if row.get("status") == "COMPLETED" and int(row.get("update", -1)) == CARE_UPDATES:
                        continue
                except (MissingArtifact, InvalidArtifact):
                    pass
                command = [
                    self.s.python,
                    "-m",
                    "before_we_act.train_mars_care_belief",
                    "--prepared-data",
                    str(self.s.prepared_care),
                    "--output",
                    str(output),
                    "--seed",
                    str(seed),
                    "--variant",
                    variant,
                    "--stage",
                    "formal",
                    "--updates",
                    str(CARE_UPDATES),
                    "--batch-size",
                    "48",
                    "--eval-every",
                    "200",
                    "--save-every",
                    "200",
                    "--learning-rate",
                    "3e-4",
                    "--weight-decay",
                    "1e-4",
                    "--device",
                    "cuda:0",
                    "--benchmark-adapter",
                    "DuoBench",
                ]
                jobs.append((variant, seed, command))
        for first in range(0, len(jobs), 4):
            wave = [
                (f"belief_{variant}_{seed}", command, gpu)
                for gpu, (variant, seed, command) in enumerate(jobs[first : first + 4])
            ]
            self.run_wave(wave, retries=2)

    def action_calibration(self) -> None:
        self.run_command(
            "offline_selection_calibration",
            [
                self.s.python,
                "-m",
                "scripts.before_we_act.select_calibrate_duo_care",
                "--prepared-data",
                str(self.s.prepared_care),
                "--training-root",
                str(self.s.run / "care" / "belief_training"),
                "--reference-checkpoint",
                str(self.s.bcore_checkpoint),
                "--output-root",
                str(self.s.run / "care" / "offline"),
                "--device",
                "cuda:0",
            ],
            (0,),
        )

    def _paired_command(self, output: Path, episodes: int, *, smoke: bool = False) -> list[str]:
        command = [
            self.s.python,
            "-m",
            self.s.paired_module,
            "--bcore-checkpoint",
            str(self.s.bcore_checkpoint),
            "--care-checkpoint",
            str(self.s.care_checkpoint),
            "--prepared-data",
            str(self.s.data),
            "--dino-model",
            str(self.s.dino_model),
            "--duobench-root",
            str(self.s.duobench_repo / "src"),
            "--output",
            str(output),
            "--episodes",
            str(episodes),
            "--seed-start",
            str(self.s.seed_start),
            "--workers",
            "4",
            "--task-max-steps-json",
            json.dumps(MAX_STEPS, sort_keys=True, separators=(",", ":")),
        ]
        if smoke:
            command.append("--smoke")
        return command

    def action_paired_smoke(self) -> None:
        self._require_modules("paired_validation_smoke", (self.s.paired_module,))
        self.run_command(
            "paired_validation_smoke",
            self._paired_command(self.s.run / "care" / "paired_smoke", 1, smoke=True),
            (0, 1, 2, 3),
            retries=2,
        )

    def action_paired20(self) -> None:
        self._require_modules("paired_validation20", (self.s.paired_module,))
        self.run_command(
            "paired_validation20",
            self._paired_command(self.s.run / "care" / "paired_validation20", 20),
            (0, 1, 2, 3),
            retries=3,
        )

    # --------------------------------------------------------------- stage DAG
    def stages(self) -> "OrderedDict[str, Stage]":
        static = {
            "repo": str(self.s.repo.resolve()),
            "dataset": str(self.s.dataset.resolve()),
            "dino_model": str(self.s.dino_model.resolve()),
            "b0h_updates": B0H_UPDATES,
            "bcore_updates": BCORE_UPDATES,
            "care_updates": CARE_UPDATES,
            "validation_episodes": VALIDATION_EPISODES,
            "max_steps": MAX_STEPS,
        }
        no_action = lambda: None
        rows = (
            Stage("dependencies", "checking repositories, DINO asset and four RTX 5090 GPUs", self.validate_dependencies, no_action, static),
            Stage("dataset_download", "downloading the pinned DuoBench snapshot with an in-memory HF credential", self.validate_dataset_download, self.action_dataset_download, {**static, "dataset_revision": "b741bc915d942ecadaefb4e3de6bbd716c1b8b1b"}),
            Stage("data_prepare", "lossless all-550 absolute-action data conversion (ACT converter only)", self.validate_data, self.action_data_prepare, {**static, "image_size": 224}),
            Stage("data_audit", "auditing all tasks, episodes, normalization, images and action contract", self.validate_data_audit, self.action_data_audit, static),
            Stage("dino_cache", "encoding independent head/own-wrist DINOv3 features on four GPUs", self.validate_cache, self.action_dino_cache, {**static, "world_size": 4}),
            Stage("b0h_smoke_train", "four-rank DINO TemporalHistory B0-H training smoke", self.validate_b0h_smoke, self.action_b0h_smoke, {**static, "updates": self.s.b0h_smoke_updates}),
            Stage("b0h_smoke_closed_loop", "native-camera, strict-local, normalization and action-interface smoke", self.validate_b0h_closed_loop_smoke, self.action_b0h_closed_loop_smoke, static),
            Stage("b0h_formal", "formal 120k hidden-residual B0-H training on all four GPUs", self.validate_b0h_formal, self.action_b0h_formal, static),
            Stage("b0h_validation20", "11-task B0-H Validation20 with frozen per-task horizons", self.validate_b0h_validation, self.action_b0h_validation, static),
            Stage("b0h_probe_gate", "requiring at least one real formal B0-H closed-loop success", self.validate_b0h_gate, no_action, static),
            Stage("bcore_cache", "building Duo PredictiveTeamBeliefPolicy contexts from formal B0-H", self.validate_bcore_cache, self.action_bcore_cache, {**static, "module": self.s.bcore_cache_module}),
            Stage("bcore_smoke", "Duo-specific B-core contract and closed-loop smoke", self.validate_bcore_smoke, self.action_bcore_smoke, {**static, "module": self.s.bcore_smoke_module}),
            Stage("bcore_train_3seeds", "training three independent 120k B-core seeds", self.validate_bcore_training, self.action_bcore_training, {**static, "module": self.s.bcore_train_module, "seeds": BCORE_SEEDS}),
            Stage("bcore_select", "offline pre-closed-loop B-core checkpoint selection", self.validate_bcore_select, self.action_bcore_select, {**static, "module": self.s.bcore_select_module}),
            Stage("bcore_validation20", "11-task frozen selected B-core Validation20", self.validate_bcore_validation, self.action_bcore_validation, {**static, "module": self.s.bcore_evaluate_module}),
            Stage("bcore_probe_gate", "requiring at least one real formal B-core closed-loop success", self.validate_bcore_gate, no_action, static),
            Stage("branch_smoke", "same-snapshot B-core branch and nondegenerate signal smoke", self.validate_branch_smoke, self.action_branch_smoke, {**static, "module": self.s.branch_module}),
            Stage("branch_collection", "collecting 330 B-core proposal families with four GPU workers", self.validate_branch_collection, self.action_branch_collection, {**static, "module": self.s.branch_module, "families_per_task": self.s.families_per_task}),
            Stage("branch_prepare", "freezing branch families into the CARE belief-head tensor contract", self.validate_branch_prepare, self.action_branch_prepare, static),
            Stage("branch_signal_gate", "rejecting trace mismatch, all-zero labels and pairwise collapse", self.validate_signal_gate, self.action_signal_gate, static),
            Stage("belief_smoke", "two-update CARE belief-head smoke", self.validate_belief_smoke, self.action_belief_smoke, static),
            Stage("belief_train", "training four CARE variants x three seeds in four-GPU waves", self.validate_belief_training, self.action_belief_training, {**static, "variants": CARE_VARIANTS, "seeds": CARE_SEEDS}),
            Stage("offline_selection_calibration", "offline checkpoint selection and split-conformal calibration", self.validate_calibration, self.action_calibration, static),
            Stage("paired_validation_smoke", "11-task one-pair end-to-end B-core/CARE closed-loop smoke", self.validate_paired_smoke, self.action_paired_smoke, {**static, "module": self.s.paired_module}),
            Stage("paired_validation20", "11-task paired selector-off/CARE Validation20", self.validate_paired20, self.action_paired20, {**static, "module": self.s.paired_module}),
        )
        result = OrderedDict((row.name, row) for row in rows)
        if tuple(result) != tuple(STAGE_DEPENDENCIES):
            raise AssertionError("stage implementation and frozen DAG differ")
        return result

    def _fingerprint(self, stage: Stage) -> str:
        dependencies = {}
        for dependency in STAGE_DEPENDENCIES[stage.name]:
            path = self._receipt(dependency)
            if not path.is_file():
                raise InvalidArtifact(f"dependency receipt missing: {dependency}")
            dependencies[dependency] = _sha256(path)
        return _canonical_hash(
            {
                "stage": stage.name,
                "dependencies": dependencies,
                "settings": dict(stage.fingerprint),
                "dag": list(STAGE_DEPENDENCIES[stage.name]),
            }
        )

    def _complete_stage(self, stage: Stage, validation: Mapping[str, Any], adopted: bool) -> None:
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "PASSED",
            "stage": stage.name,
            "completed_at_utc": _utc_now(),
            "pid": os.getpid(),
            "adopted_existing_output": bool(adopted),
            "fingerprint": self._fingerprint(stage),
            "dependencies": list(STAGE_DEPENDENCIES[stage.name]),
            "validation": dict(validation),
        }
        _atomic_json(self._receipt(stage.name), receipt)
        if stage.name not in self.completed:
            self.completed.append(stage.name)
        self.write_status("RUNNING", stage.name, f"passed: {stage.detail}")

    def run_stage(self, stage: Stage) -> None:
        for dependency in STAGE_DEPENDENCIES[stage.name]:
            receipt = _read_json(self._receipt(dependency))
            if receipt.get("status") != "PASSED":
                raise InvalidArtifact(f"dependency did not pass: {dependency}")
        fingerprint = self._fingerprint(stage)
        receipt_path = self._receipt(stage.name)
        existing_receipt = _read_json(receipt_path) if receipt_path.is_file() else None
        if existing_receipt is not None and existing_receipt.get("fingerprint") != fingerprint:
            raise ProvenanceDrift(
                f"{stage.name} receipt fingerprint differs; use a new DUO_DINO_RUN root"
            )
        try:
            validation = stage.validator()
        except MissingArtifact:
            validation = None
        if validation is not None:
            if existing_receipt is not None and existing_receipt.get("status") != "PASSED":
                raise InvalidArtifact(f"non-passed receipt exists for {stage.name}")
            if existing_receipt is not None:
                if _canonical_hash(_stable_value(existing_receipt.get("validation", {}))) != _canonical_hash(_stable_value(validation)):
                    raise InvalidArtifact(
                        f"{stage.name} artifact validation differs from its frozen receipt"
                    )
                # Preserve receipt bytes exactly.  Downstream fingerprints use
                # their hashes, so rewriting a completed receipt would create
                # false provenance drift on every supervisor restart.
                if stage.name not in self.completed:
                    self.completed.append(stage.name)
                self.write_status("RUNNING", stage.name, f"resumed: {stage.detail}")
                return
            self._complete_stage(stage, validation, adopted=existing_receipt is None)
            return
        self.write_status("RUNNING", stage.name, stage.detail)
        stage.action()
        if self.stop_event.is_set():
            raise StopPipeline("stop requested")
        validation = stage.validator()
        self._complete_stage(stage, validation, adopted=False)

    def run(self) -> int:
        self.s.run.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.receipt_root.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        self.start_heartbeat()
        try:
            stages = self.stages()
            for name, stage in stages.items():
                if self.stop_event.is_set():
                    raise StopPipeline("stop requested")
                self.run_stage(stage)
            summary = self.s.run / "care" / "paired_validation20" / "summary.json"
            self.write_status(
                "COMPLETE",
                "complete",
                "DINO B0-H, independent B-core, CARE and paired Validation20 complete",
                paired_summary=str(summary.resolve()),
                b0h_checkpoint=str(self.s.b0h_checkpoint.resolve()),
                bcore_checkpoint=str(self.s.bcore_checkpoint.resolve()),
                care_checkpoint=str(self.s.care_checkpoint.resolve()),
            )
            return 0
        except PendingImplementation as error:
            self.write_status(
                "BLOCKED_PENDING_IMPLEMENTATION",
                error.stage,
                str(error),
                missing_modules=list(error.modules),
                downstream_started=False,
            )
            return 78
        except GateBlocked as error:
            state = (
                "BLOCKED_REFERENCE_ZERO_SUCCESS"
                if error.stage == "b0h_probe_gate"
                else "BLOCKED_BCORE_ZERO_SUCCESS"
            )
            self.write_status(state, error.stage, error.reason, downstream_started=False)
            return error.exit_code
        except StopPipeline:
            self.write_status("STOPPED", self.stage_name, "supervisor stopped; outputs remain resumable")
            return 130
        except Exception as error:
            self.write_status(
                "FAILED",
                self.stage_name,
                repr(error),
                traceback=traceback.format_exc(),
                downstream_started=False,
            )
            return 1
        finally:
            self.stop_event.set()
            # A signal handler normally terminates children.  This final guard
            # prevents orphaned simulators if an internal exception escaped.
            for active in list(self.active.values()):
                try:
                    os.killpg(active.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


def _status(settings: Settings) -> int:
    path = settings.run / "status.json"
    if not path.is_file():
        print(json.dumps({"state": "NOT_STARTED", "run": str(settings.run)}, indent=2))
        return 0
    print(json.dumps(json.loads(path.read_text()), indent=2, sort_keys=True))
    return 0


def _print_dag() -> int:
    rows = [
        {"stage": stage, "dependencies": list(dependencies)}
        for stage, dependencies in STAGE_DEPENDENCIES.items()
    ]
    print(json.dumps({"schema": SCHEMA, "stages": rows}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "status", "print-dag"), default="run")
    args = parser.parse_args(argv)
    settings = Settings.from_environment()
    if args.command == "status":
        return _status(settings)
    if args.command == "print-dag":
        return _print_dag()
    return Pipeline(settings).run()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BCORE_SEEDS",
    "CARE_SEEDS",
    "CARE_VARIANTS",
    "MAX_STEPS",
    "Pipeline",
    "STAGE_DEPENDENCIES",
    "Settings",
    "TASKS",
    "main",
]
