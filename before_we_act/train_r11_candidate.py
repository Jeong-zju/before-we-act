"""Common, fail-closed trainer for one independent R11 candidate branch.

The candidate model itself is supplied by the branch-local ``r11_registry``.
This module owns only the frozen six-task sampler, optimizer-update boundary,
checkpoint provenance, deterministic resume cursor, and real worker heartbeat.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from before_we_act.r11_data import (
    EFFECTIVE_BATCH,
    ExactSixTaskAccumulationSampler,
    R11EpisodeDataset,
    SIX_TASKS,
    episode_receipt,
    load_r11_episodes,
)


FORMAL_UPDATES = 120_000
ALLOWED_STAGES = ("f1", "discovery", "selection", "formal")
INFERENCE_MODES = ("normal", "prediction_off", "prediction_shuffled")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    os.replace(temporary, destination)


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(
            descriptor,
            (json.dumps(payload, sort_keys=True, default=str) + "\n").encode(),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_alias(source: Path, destination: Path) -> None:
    """Atomically create a space-efficient immutable milestone hard link."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, destination)


def process_start_time_ticks(pid: int | None = None) -> int:
    pid = os.getpid() if pid is None else int(pid)
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    return int(fields[21])


def git_identity(root: Path) -> dict[str, str]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    dirty = run("status", "--porcelain")
    if dirty:
        raise ValueError("candidate worktree is dirty; refusing non-reproducible training")
    return {"branch": run("branch", "--show-current"), "commit": run("rev-parse", "HEAD")}


def _nested_get(config: Mapping[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        value: Any = config
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            return value
    raise KeyError(" or ".join(".".join(path) for path in paths))


def training_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    micro_batch = int(
        _nested_get(
            config,
            (("data", "micro_batch"), ("training", "micro_batch_size")),
        )
    )
    accumulation = int(
        _nested_get(
            config,
            (("data", "accumulation"), ("training", "gradient_accumulation")),
        )
    )
    if micro_batch * accumulation != EFFECTIVE_BATCH:
        raise ValueError(
            f"candidate config violates exact effective batch: {micro_batch}*"
            f"{accumulation}!={EFFECTIVE_BATCH}"
        )
    learning_rate = float(
        _nested_get(
            config,
            (("optimization", "learning_rate"), ("training", "learning_rate")),
        )
    )
    weight_decay = float(
        _nested_get(
            config,
            (("optimization", "weight_decay"), ("training", "weight_decay")),
        )
    )
    optimizer = str(
        config.get("model_config", {}).get(
            "optimizer", config.get("training", {}).get("optimizer", "AdamW")
        )
    )
    precision = str(
        config.get("optimization", {}).get(
            "dtype", config.get("training", {}).get("precision", "bfloat16")
        )
    )
    if precision.lower() not in {"bfloat16", "bf16"}:
        raise ValueError(f"R11 formal precision must be bfloat16, got {precision}")
    return {
        "micro_batch_size": micro_batch,
        "gradient_accumulation": accumulation,
        "effective_batch": EFFECTIVE_BATCH,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": optimizer,
        "precision": "bfloat16",
    }


def _run_manifest_contract(
    run_manifest: Mapping[str, Any], config: Mapping[str, Any], git: Mapping[str, str]
) -> None:
    candidate = config["candidate"]
    if run_manifest.get("stage") != "R11" or not run_manifest.get("immutable"):
        raise ValueError("R11 immutable run manifest identity is invalid")
    expected = run_manifest["candidates"].get(candidate)
    if not expected or expected.get("model") != config.get("model"):
        raise ValueError("candidate/model differs from immutable run manifest")
    if run_manifest["branches"].get(candidate) != git["branch"]:
        raise ValueError("candidate branch differs from immutable run manifest")
    if config.get("base_commit") != run_manifest["base"].get("commit"):
        raise ValueError("candidate base commit differs from immutable run manifest")
    if config.get("upstream_commit", expected.get("upstream_commit")) != expected.get(
        "upstream_commit"
    ):
        raise ValueError("candidate upstream commit differs from immutable run manifest")


def _validate_dataset_projection(
    run_manifest: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    expected = run_manifest["dataset"]["tasks"]
    actual = receipt["training_manifests"]
    if set(expected) != set(SIX_TASKS):
        raise ValueError("run manifest does not freeze all six task identities")
    for task, row in expected.items():
        matches = [
            digest
            for path, digest in actual.items()
            if Path(path).parent.name == task
        ]
        if matches != [row["manifest_sha256"]]:
            raise ValueError(f"{task} training manifest differs from R11 freeze")


def load_frozen_normalization(
    checkpoint_path: Path,
    expected_sha256: str,
    dataset: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    observed = sha256_file(checkpoint_path)
    if observed != expected_sha256:
        raise ValueError("baseline checkpoint SHA256 differs from immutable run manifest")
    saved = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if int(saved.get("update", -1)) != FORMAL_UPDATES:
        raise ValueError("baseline checkpoint is not the frozen 120000-update artifact")
    training_manifests = saved.get("config", {}).get("training_manifests", {})
    if training_manifests != dataset["training_manifests"]:
        raise ValueError("current dataset projection differs from baseline checkpoint")
    stats = saved.get("stats", {})
    expected_shapes = {"q_mean": (9,), "q_std": (9,), "a_mean": (8,), "a_std": (8,)}
    result: dict[str, np.ndarray] = {}
    for name, shape in expected_shapes.items():
        value = np.asarray(stats.get(name), dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid frozen baseline normalization field {name}")
        result[name] = value
    if np.any(result["q_std"] <= 0) or np.any(result["a_std"] <= 0):
        raise ValueError("frozen normalization standard deviation is not positive")
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and value.get("cuda"):
        torch.cuda.set_rng_state_all(value["cuda"])


def move_batch(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_batch(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_batch(item, device) for item in value)
    if isinstance(value, list):
        return [move_batch(item, device) for item in value]
    return value


def build_optimizer(
    parameters: list[torch.nn.Parameter], contract: Mapping[str, Any], *, device: torch.device
):
    name = str(contract["optimizer"]).lower().replace("_", "")
    kwargs = {
        "lr": float(contract["learning_rate"]),
        "weight_decay": float(contract["weight_decay"]),
        "betas": (0.9, 0.95),
    }
    if name in {"adamw8bit", "8bitadamw"}:
        if device.type != "cuda":
            raise RuntimeError("AdamW8bit formal optimizer requires CUDA")
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise RuntimeError("candidate requires bitsandbytes AdamW8bit") from error
        return bnb.optim.AdamW8bit(parameters, **kwargs)
    if name != "adamw":
        raise ValueError(f"unsupported frozen optimizer {contract['optimizer']}")
    return torch.optim.AdamW(parameters, **kwargs)


def scheduler_multiplier(step: int, *, warmup: int, total: int) -> float:
    warmup_value = min(1.0, (step + 1) / max(warmup, 1))
    progress = min(1.0, (step + 1) / max(total, 1))
    return warmup_value * 0.5 * (1.0 + math.cos(math.pi * progress))


def _optimizer_to(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


class WorkerHeartbeat:
    """Heartbeat emitted by a thread inside the actual training worker."""

    def __init__(
        self,
        path: Path,
        *,
        candidate: str,
        stage: str,
        interval: float = 20.0,
    ) -> None:
        self.path = path
        self.candidate = candidate
        self.stage = stage
        self.interval = interval
        self.pid = os.getpid()
        self.start_ticks = process_start_time_ticks(self.pid)
        self.started_at_epoch = time.time()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._progress: dict[str, Any] = {"update": 0, "micro_step": 0}
        self._thread = threading.Thread(target=self._loop, name="r11-heartbeat", daemon=True)

    def update(self, **values: Any) -> None:
        with self._lock:
            self._progress.update(values)
        self.write()

    def write(self) -> None:
        with self._lock:
            progress = dict(self._progress)
        alive = False
        try:
            alive = process_start_time_ticks(self.pid) == self.start_ticks
        except (FileNotFoundError, ProcessLookupError):
            pass
        atomic_json(
            self.path,
            {
                "format_version": "before-we-act.r11.worker_heartbeat/1",
                "candidate": self.candidate,
                "stage": self.stage,
                "pid": self.pid,
                "pid_start_time_ticks": self.start_ticks,
                "started_at_epoch": self.started_at_epoch,
                "worker_identity_alive": alive,
                "updated_at_epoch": time.time(),
                **progress,
            },
        )

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.write()

    def __enter__(self):
        self.write()
        self._thread.start()
        return self

    def __exit__(self, *_arguments):
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval))
        self.write()


def _resume_contract(
    saved: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    sampler: ExactSixTaskAccumulationSampler,
) -> int:
    provenance = saved.get("provenance", {})
    for field in (
        "candidate",
        "model",
        "base_commit",
        "config_sha256",
        "source_receipt_sha256",
        "dataset_receipt_sha256",
        "baseline_checkpoint_sha256",
        "micro_batch_size",
        "gradient_accumulation",
    ):
        if provenance.get(field) != identity.get(field):
            raise ValueError(f"resume provenance mismatch at {field}")
    completed = sampler.validate_resume_receipt(saved["sample_cursor"])
    if int(saved.get("update", -1)) != completed:
        raise ValueError("resume update differs from deterministic sample cursor")
    return completed


def _smoke_inference(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    destination: Path,
) -> None:
    keep = {"current_rgb", "qpos", "task", "task_text", "agent", "objective_slot"}
    inference = {key: value for key, value in batch.items() if key in keep}
    rows = []
    model.eval()
    with torch.no_grad():
        for mode in INFERENCE_MODES:
            output = model(inference, mode=mode)
            action = output.get("action")
            if not isinstance(action, torch.Tensor) or action.shape[-2:] != (100, 8):
                raise ValueError(f"{mode} inference did not return [B,100,8] action")
            if not torch.isfinite(action).all():
                raise FloatingPointError(f"{mode} inference returned non-finite actions")
            rows.append(
                {
                    "mode": mode,
                    "batch": action.shape[0],
                    "action_shape": list(action.shape),
                    "execution_cadence": int(output.get("execution_cadence", 100)),
                    "future_prediction_present": isinstance(
                        output.get("future_prediction"), torch.Tensor
                    ),
                }
            )
    atomic_json(
        destination,
        {
            "format_version": "before-we-act.r11.f1_inference/1",
            "status": "PASSED",
            "modes": rows,
            "completed_at_epoch": time.time(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--baseline-provenance", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=ALLOWED_STAGES, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--protocol-updates", type=int, default=FORMAL_UPDATES)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heartbeat", default="")
    parser.add_argument("--status", default="")
    parser.add_argument("--smoke-modes", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.protocol_updates != FORMAL_UPDATES:
        raise ValueError("R11 sample cursor protocol must remain 120000 updates")
    if not 1 <= args.updates <= args.protocol_updates:
        raise ValueError("stage target updates are outside the frozen protocol")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve(strict=True)
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    contract = training_contract(config)
    git = git_identity(project_root)
    run_manifest_path = Path(args.run_manifest).resolve(strict=True)
    run_manifest_raw = run_manifest_path.read_bytes()
    run_manifest = json.loads(run_manifest_raw)
    _run_manifest_contract(run_manifest, config, git)
    candidate = config["candidate"]

    expected_gpu = str(run_manifest["candidates"][candidate]["gpu"])
    device = torch.device(args.device)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device.type == "cuda":
        if visible != expected_gpu:
            raise ValueError(
                f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical GPU {expected_gpu}"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("R11 worker must see exactly its one assigned CUDA device")

    baseline_provenance_path = Path(args.baseline_provenance).resolve(strict=True)
    baseline_provenance = json.loads(baseline_provenance_path.read_text())
    baseline_expected = run_manifest["baseline"]
    if (
        baseline_provenance.get("status") != "PASSED"
        or baseline_provenance.get("checkpoint", {}).get("sha256")
        != baseline_expected["checkpoint_sha256"]
    ):
        raise ValueError("full baseline provenance gate is not PASSED")

    manifests = [Path(path).resolve(strict=True) for path in args.manifests]
    episodes = load_r11_episodes(manifests)
    dataset_projection = episode_receipt(episodes)
    _validate_dataset_projection(run_manifest, dataset_projection)
    stats = load_frozen_normalization(
        Path(args.baseline_checkpoint).resolve(strict=True),
        baseline_expected["checkpoint_sha256"],
        dataset_projection,
    )

    source_receipt = (project_root / config["source_receipt"]).resolve(strict=True)
    identity = {
        "candidate": candidate,
        "model": config["model"],
        "branch": git["branch"],
        "commit": git["commit"],
        "base_commit": config["base_commit"],
        "upstream_commit": run_manifest["candidates"][candidate]["upstream_commit"],
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "source_receipt": str(source_receipt),
        "source_receipt_sha256": sha256_file(source_receipt),
        "dataset_receipt_sha256": canonical_sha256(dataset_projection),
        "baseline_checkpoint": str(Path(args.baseline_checkpoint).resolve()),
        "baseline_checkpoint_sha256": baseline_expected["checkpoint_sha256"],
        "run_manifest": str(run_manifest_path),
        "run_manifest_sha256": hashlib.sha256(run_manifest_raw).hexdigest(),
        "baseline_provenance": str(baseline_provenance_path),
        "baseline_provenance_sha256": sha256_file(baseline_provenance_path),
        **contract,
    }

    seed_everything(args.seed)
    dataset = R11EpisodeDataset(episodes, stats)
    preliminary_sampler = ExactSixTaskAccumulationSampler(
        episodes,
        updates=args.protocol_updates,
        seed=args.seed,
        micro_batch_size=contract["micro_batch_size"],
        start_update=0,
    )

    from before_we_act.r11_registry import build_r11_model

    model = build_r11_model(config["model"], str(config_path), project_root)
    model = model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("candidate exposes no trainable parameters")
    optimizer = build_optimizer(trainable, contract, device=device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_multiplier(
            step, warmup=args.warmup, total=args.protocol_updates
        ),
    )

    saved = None
    start_update = 0
    resume_path = Path(args.resume).resolve(strict=True) if args.resume else None
    if resume_path is not None:
        saved = torch.load(resume_path, map_location="cpu", weights_only=False, mmap=True)
        start_update = _resume_contract(
            saved, identity=identity, sampler=preliminary_sampler
        )
        if start_update >= args.updates:
            raise ValueError("resume checkpoint already reached this stage target")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        _optimizer_to(optimizer, device)
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])

    sampler = ExactSixTaskAccumulationSampler(
        episodes,
        updates=args.protocol_updates,
        seed=args.seed,
        micro_batch_size=contract["micro_batch_size"],
        start_update=start_update,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        # Worker base-seed generation must not advance the model RNG restored
        # from a checkpoint; HDF5 decoding itself is deterministic.
        generator=torch.Generator().manual_seed(args.seed + 97_531),
    )
    output = Path(args.output).resolve()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    heartbeat_path = Path(args.heartbeat).resolve() if args.heartbeat else output / "heartbeat.json"
    status_path = Path(args.status).resolve() if args.status else output / "status.json"
    atomic_json(output / "config.json", config)
    atomic_json(output / "dataset_receipt.json", dataset_projection)
    atomic_json(output / "training_identity.json", identity)

    stop_requested = threading.Event()
    stop_signal: dict[str, int | None] = {"number": None}

    def request_stop(number, _frame):
        stop_signal["number"] = int(number)
        stop_requested.set()

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    started = time.time()
    last_metrics: dict[str, float] = dict(saved.get("last_metrics", {})) if saved else {}
    last_batch: Mapping[str, Any] | None = None
    completed_update = start_update
    micro_index = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()

    def save_checkpoint(update: int) -> tuple[Path, str]:
        payload = {
            "format_version": "before-we-act.r11.checkpoint/1",
            "candidate": candidate,
            "model_name": config["model"],
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "stats": stats,
            "config": config,
            "provenance": identity | {
                "current_commit": git["commit"],
                "model_provenance": getattr(model, "provenance", {}),
                "dataset_projection": dataset_projection,
            },
            "stage": args.stage,
            "stage_target_updates": args.updates,
            "protocol_updates": args.protocol_updates,
            "update": update,
            "sample_cursor": sampler.cursor_receipt(update),
            "last_metrics": last_metrics,
            "rng": capture_rng(),
            "saved_at_epoch": time.time(),
        }
        latest = checkpoints / "checkpoint_latest.pt"
        atomic_torch_save(payload, latest)
        digest = sha256_file(latest)
        if update == args.updates or update % args.save_every == 0:
            checkpoint_alias(latest, checkpoints / f"checkpoint_{update:06d}.pt")
        atomic_json(
            checkpoints / "checkpoint_latest.receipt.json",
            {
                "format_version": "before-we-act.r11.checkpoint_receipt/1",
                "candidate": candidate,
                "update": update,
                "path": str(latest),
                "sha256": digest,
                "sample_cursor_sha256": canonical_sha256(payload["sample_cursor"]),
                "saved_at_epoch": payload["saved_at_epoch"],
            },
        )
        return latest, digest

    atomic_json(
        status_path,
        {
            "status": "TRAINING",
            "stage": args.stage,
            "candidate": candidate,
            "model": config["model"],
            "pid": os.getpid(),
            "pid_start_time_ticks": process_start_time_ticks(),
            "update": start_update,
            "target_updates": args.updates,
            "protocol_updates": args.protocol_updates,
            "started_at_epoch": started,
            "updated_at_epoch": started,
        },
    )

    with WorkerHeartbeat(
        heartbeat_path, candidate=candidate, stage=args.stage
    ) as heartbeat:
        for raw_batch in loader:
            update = start_update + micro_index // sampler.accumulation_steps + 1
            if update > args.updates:
                break
            within_update = micro_index % sampler.accumulation_steps
            batch = move_batch(raw_batch, device)
            slot_start = within_update * contract["micro_batch_size"]
            batch["objective_slot"] = torch.arange(
                slot_start,
                slot_start + contract["micro_batch_size"],
                device=device,
            )
            last_batch = batch
            heartbeat.update(update=update, micro_step=within_update + 1)
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                metrics = model.training_step(batch, update)
                loss = metrics.get("loss")
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise TypeError("candidate training_step must return scalar tensor loss")
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at update {update}")
                scaled = loss / sampler.accumulation_steps
            scaled.backward()
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    scalar = float(value.detach().float().cpu())
                    if not math.isfinite(scalar):
                        raise FloatingPointError(f"non-finite metric {name} at update {update}")
                    last_metrics[name] = last_metrics.get(f"_{name}_sum", 0.0) + scalar
                    last_metrics[f"_{name}_sum"] = last_metrics[name]

            micro_index += 1
            if micro_index % sampler.accumulation_steps:
                continue

            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient at update {update}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_update = update
            completed_since_resume = update - start_update
            aggregated: dict[str, float] = {}
            for key in list(last_metrics):
                if key.startswith("_") and key.endswith("_sum"):
                    name = key[1:-4]
                    aggregated[name] = last_metrics.pop(key) / sampler.accumulation_steps
            last_metrics.update(aggregated)
            elapsed = time.time() - started
            progress = {
                "event": "optimizer_update",
                "candidate": candidate,
                "stage": args.stage,
                "update": update,
                "target_updates": args.updates,
                "protocol_updates": args.protocol_updates,
                "micro_batch_size": contract["micro_batch_size"],
                "gradient_accumulation": contract["gradient_accumulation"],
                "effective_batch": EFFECTIVE_BATCH,
                "gradient_norm": float(gradient_norm.detach().float().cpu()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "updates_per_hour": completed_since_resume / max(elapsed, 1e-9) * 3600,
                "eta_hours": (args.updates - update)
                * elapsed
                / max(completed_since_resume, 1)
                / 3600,
                "gpu_memory_gb": (
                    round(torch.cuda.max_memory_allocated() / 2**30, 3)
                    if device.type == "cuda"
                    else 0.0
                ),
                "updated_at_epoch": time.time(),
                **last_metrics,
            }
            append_jsonl(progress_path, progress)
            heartbeat.update(update=update, micro_step=0, **last_metrics)
            if update == start_update + 1 or update % args.log_every == 0:
                print(json.dumps(progress, sort_keys=True), flush=True)
            should_save = (
                update == args.updates
                or update % args.save_every == 0
                or stop_requested.is_set()
            )
            checkpoint = None
            checkpoint_sha256 = None
            if should_save:
                checkpoint, checkpoint_sha256 = save_checkpoint(update)
            atomic_json(
                status_path,
                {
                    "status": "STOPPED" if stop_requested.is_set() else "TRAINING",
                    "stage": args.stage,
                    "candidate": candidate,
                    "model": config["model"],
                    "pid": os.getpid(),
                    "pid_start_time_ticks": process_start_time_ticks(),
                    "update": update,
                    "target_updates": args.updates,
                    "protocol_updates": args.protocol_updates,
                    "last_metrics": last_metrics,
                    "checkpoint": str(checkpoint) if checkpoint else None,
                    "checkpoint_sha256": checkpoint_sha256,
                    "signal": stop_signal["number"],
                    "started_at_epoch": started,
                    "updated_at_epoch": time.time(),
                },
            )
            if stop_requested.is_set():
                break

    if completed_update == start_update:
        raise RuntimeError("trainer completed no optimizer update")
    latest = checkpoints / "checkpoint_latest.pt"
    if completed_update == args.updates and not latest.is_file():
        latest, _ = save_checkpoint(completed_update)
    if args.smoke_modes and not stop_requested.is_set():
        if last_batch is None:
            raise RuntimeError("no batch is available for inference smoke")
        _smoke_inference(model, last_batch, output / "f1_inference.json")

    terminal = "STOPPED" if stop_requested.is_set() else "PASSED"
    result = {
        "status": terminal,
        "stage": args.stage,
        "candidate": candidate,
        "model": config["model"],
        "update": completed_update,
        "target_updates": args.updates,
        "protocol_updates": args.protocol_updates,
        "checkpoint": str(latest),
        "checkpoint_sha256": sha256_file(latest),
        "sample_cursor": sampler.cursor_receipt(completed_update),
        "last_metrics": last_metrics,
        "stop_signal": stop_signal["number"],
        "started_at_epoch": started,
        "completed_at_epoch": time.time(),
    }
    atomic_json(status_path, result)
    print(json.dumps(result | {"sample_cursor": "saved"}, sort_keys=True), flush=True)
    return result


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
