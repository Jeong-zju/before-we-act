#!/usr/bin/env python3
"""Crash-resumable stage supervisor for the three formal VLA baselines.

System supervisor owns this foreground process.  This process in turn owns one
GPU stage process-group at a time and advances only after an exit-code-zero
stage plus explicit artifact checks.  It never uses name-pattern killing.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

PIPELINE = Path(os.environ.get("BWA_VLA_PIPELINE", "/workspace/bwa_vla_pipeline/pipeline.json"))
STATE_ROOT = Path(os.environ.get("BWA_VLA_STATE_ROOT", "/workspace/bwa_vla_runs/supervisor"))
POLL_SECONDS = 5
_stop = False
_active: subprocess.Popen | None = None
_active_identity: tuple[int, int] | None = None


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def proc_start_ticks(pid: int) -> int:
    # Field 22 of /proc/PID/stat.  Account for spaces inside '(comm)'.
    raw = Path(f"/proc/{pid}/stat").read_text()
    return int(raw[raw.rfind(")") + 2 :].split()[19])


def same_process(identity: tuple[int, int] | None) -> bool:
    if identity is None:
        return False
    pid, ticks = identity
    try:
        return proc_start_ticks(pid) == ticks
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return False


def terminate_active() -> None:
    global _active
    identity = _active_identity
    if not same_process(identity):
        return
    pid, _ = identity
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 45
    while same_process(identity) and time.monotonic() < deadline:
        # Reap an exited child promptly.  A zombie still has the same
        # /proc start ticks, so checking identity alone otherwise burns the
        # entire shutdown timeout on every safe supervisor restart.
        if _active is not None and _active.poll() is not None:
            _active = None
            return
        time.sleep(0.25)
    if same_process(identity):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if _active is not None:
        try:
            _active.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        _active = None


def on_signal(signum: int, _frame: Any) -> None:
    global _stop
    _stop = True
    print(json.dumps({"event": "signal", "signal": signum, "at": utcnow()}), flush=True)
    terminate_active()


def command_digest(stage: dict) -> str:
    canonical = json.dumps({"argv": stage["argv"], "cwd": stage.get("cwd"),
                            "env": stage.get("env", {}), "gpus": stage.get("gpus", [])},
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_artifacts(stage: dict) -> list[dict]:
    found: list[dict] = []
    for item in stage.get("artifacts", []):
        path = Path(item["path"])
        kind = item.get("kind", "file")
        if kind == "dir":
            if not path.is_dir():
                raise RuntimeError(f"missing artifact directory: {path}")
            found.append({"path": str(path), "kind": kind})
            continue
        if not path.is_file():
            raise RuntimeError(f"missing artifact file: {path}")
        row: dict[str, Any] = {"path": str(path), "kind": kind, "size_bytes": path.stat().st_size}
        if kind == "json":
            payload = json.loads(path.read_text())
            for key, value in item.get("equals", {}).items():
                current: Any = payload
                for part in key.split("."):
                    current = current[int(part)] if isinstance(current, list) else current[part]
                if current != value:
                    raise RuntimeError(f"artifact {path}: {key}={current!r}, expected {value!r}")
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif item.get("sha256"):
            h = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != item["sha256"]:
                raise RuntimeError(f"artifact hash mismatch: {path}")
            row["sha256"] = h.hexdigest()
        found.append(row)
    return found


def gpu_pids(gpus: list[int]) -> dict[int, list[int]]:
    if not gpus:
        return {}
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    ).stdout
    index_to_uuid = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    ).stdout
    mapping = {int(line.split(",", 1)[0].strip()): line.split(",", 1)[1].strip()
               for line in index_to_uuid.splitlines() if line.strip()}
    wanted = {mapping[g]: g for g in gpus}
    result = {g: [] for g in gpus}
    for line in query.splitlines():
        if not line.strip():
            continue
        uuid, pid = (part.strip() for part in line.split(",", 1))
        if uuid in wanted:
            result[wanted[uuid]].append(int(pid))
    return {gpu: pids for gpu, pids in result.items() if pids}


def run_stage(stage: dict) -> bool:
    global _active, _active_identity
    name = stage["name"]
    digest = command_digest(stage)
    receipt_path = STATE_ROOT / "receipts" / f"{name}.json"
    if receipt_path.is_file():
        try:
            old = json.loads(receipt_path.read_text())
            if old.get("status") == "complete" and old.get("command_sha256") == digest:
                validate_artifacts(stage)
                print(json.dumps({"event": "skip_complete", "stage": name}), flush=True)
                return True
        except Exception as exc:
            print(json.dumps({"event": "receipt_invalid", "stage": name, "error": str(exc)}), flush=True)

    conflicts = gpu_pids([int(g) for g in stage.get("gpus", [])])
    if conflicts:
        atomic_json(STATE_ROOT / "state.json", {"status": "waiting_for_gpus", "stage": name,
                                                "conflicts": conflicts, "updated_at": utcnow()})
        print(json.dumps({"event": "gpu_conflict", "stage": name, "conflicts": conflicts}), flush=True)
        return False

    logs = STATE_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    attempt = int(json.loads((STATE_ROOT / "attempts.json").read_text()).get(name, 0)) + 1 \
        if (STATE_ROOT / "attempts.json").is_file() else 1
    attempts = json.loads((STATE_ROOT / "attempts.json").read_text()) if (STATE_ROOT / "attempts.json").is_file() else {}
    attempts[name] = attempt
    atomic_json(STATE_ROOT / "attempts.json", attempts)
    log_path = logs / f"{name}.attempt-{attempt:03d}.log"
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in stage.get("env", {}).items()})
    if stage.get("gpus"):
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in stage["gpus"])
    started = utcnow()
    with log_path.open("ab", buffering=0) as log:
        _active = subprocess.Popen(stage["argv"], cwd=stage.get("cwd") or None, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        _active_identity = (_active.pid, proc_start_ticks(_active.pid))
        atomic_json(STATE_ROOT / "active.json", {"stage": name, "pid": _active_identity[0],
                    "proc_start_ticks": _active_identity[1], "command_sha256": digest,
                    "attempt": attempt, "log": str(log_path), "started_at": started})
        atomic_json(STATE_ROOT / "state.json", {"status": "running", "stage": name,
                    "attempt": attempt, "pid": _active.pid, "proc_start_ticks": _active_identity[1],
                    "log": str(log_path), "started_at": started, "heartbeat_at": utcnow(),
                    "runtime_seconds": 0, "updated_at": utcnow()})
        print(json.dumps({"event": "stage_start", "stage": name, "attempt": attempt,
                          "pid": _active.pid, "log": str(log_path)}), flush=True)
        next_heartbeat = time.monotonic() + 30
        stage_started_mono = time.monotonic()
        # The SIGTERM handler may synchronously terminate and clear _active
        # while this loop is sleeping.  Check the object before dereferencing
        # it so supervisor shutdown remains a clean, restartable transition.
        while _active is not None and _active.poll() is None and not _stop:
            time.sleep(POLL_SECONDS)
            if _active is not None and time.monotonic() >= next_heartbeat and same_process(_active_identity):
                heartbeat = utcnow()
                atomic_json(STATE_ROOT / "state.json", {"status": "running", "stage": name,
                            "attempt": attempt, "pid": _active.pid,
                            "proc_start_ticks": _active_identity[1], "log": str(log_path),
                            "started_at": started, "heartbeat_at": heartbeat,
                            "runtime_seconds": round(time.monotonic() - stage_started_mono, 1),
                            "updated_at": heartbeat})
                next_heartbeat = time.monotonic() + 30
        if _stop:
            terminate_active()
            return False
        returncode = _active.wait()
    _active = None
    _active_identity = None
    (STATE_ROOT / "active.json").unlink(missing_ok=True)
    if returncode != 0:
        atomic_json(STATE_ROOT / "state.json", {"status": "retrying", "stage": name,
                    "attempt": attempt, "returncode": returncode, "log": str(log_path), "updated_at": utcnow()})
        print(json.dumps({"event": "stage_failed", "stage": name, "attempt": attempt,
                          "returncode": returncode}), flush=True)
        return False
    try:
        artifacts = validate_artifacts(stage)
    except Exception as exc:
        atomic_json(STATE_ROOT / "state.json", {"status": "retrying", "stage": name,
                    "attempt": attempt, "returncode": 0, "artifact_error": str(exc),
                    "log": str(log_path), "updated_at": utcnow()})
        print(json.dumps({"event": "artifact_failed", "stage": name, "error": str(exc)}), flush=True)
        return False
    atomic_json(receipt_path, {"schema": "bwa.vla.stage_receipt.v1", "stage": name, "status": "complete",
                "command_sha256": digest, "attempt": attempt, "started_at": started,
                "completed_at": utcnow(), "log": str(log_path), "artifacts": artifacts})
    print(json.dumps({"event": "stage_complete", "stage": name, "attempt": attempt}), flush=True)
    return True


def main() -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    backoff = 15
    while not _stop:
        try:
            config = json.loads(PIPELINE.read_text())
            stages = config["stages"]
            pending = False
            for stage in stages:
                if _stop:
                    break
                if not stage.get("enabled", True):
                    continue
                if not run_stage(stage):
                    pending = True
                    break
            if _stop:
                break
            if pending:
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
            else:
                backoff = 15
                atomic_json(STATE_ROOT / "state.json", {"status": "complete", "stage_count": len(stages),
                            "pipeline_sha256": hashlib.sha256(PIPELINE.read_bytes()).hexdigest(),
                            "updated_at": utcnow()})
                # Keep foreground ownership and re-read the pipeline so newly
                # deployed stages can be appended without a loose process.
                time.sleep(30)
        except Exception as exc:
            atomic_json(STATE_ROOT / "state.json", {"status": "supervisor_error", "error": str(exc),
                        "updated_at": utcnow()})
            print(json.dumps({"event": "supervisor_error", "error": repr(exc)}), flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
    atomic_json(STATE_ROOT / "state.json", {"status": "stopped", "updated_at": utcnow()})


if __name__ == "__main__":
    main()
