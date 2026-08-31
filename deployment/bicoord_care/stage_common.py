"""Shared, fail-closed plumbing for BiCoord CARE supervisor stages.

The supervisor treats a JSON result as a cryptographic receipt, not as a log
message.  Helpers in this module therefore write evidence files first, hash
them, and publish the result atomically only after the stage-specific checks
have succeeded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


RESULT_SCHEMA = "before-we-act.bicoord-care-stage-result/1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb", buffering=0) as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_json(path: str | Path, value: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {source}")
    return value


def artifact(path: str | Path, *, kind: str | None = None) -> dict[str, str]:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError(f"stage evidence must be a non-empty file: {source}")
    result = {"path": str(source), "sha256": sha256_file(source)}
    if kind:
        result["kind"] = kind
    return result


# Compatibility spelling used by the production cache/audit adapters.  Keep
# the canonical ``artifact`` helper above for existing stage modules.
def artifact_record(
    path: str | Path, *, root: str | Path | None = None
) -> dict[str, str]:
    row = artifact(path)
    if root is not None:
        try:
            row["relative_path"] = str(
                Path(row["path"]).relative_to(Path(root).resolve())
            )
        except ValueError:
            pass
    return row


def common_parser(
    description: str,
    operations: Sequence[str],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("operation", choices=tuple(operations))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument(
        "--auto-resume", "--resume", dest="auto_resume", action="store_true"
    )
    parser.set_defaults(auto_resume=False)
    return parser


def parse_args(
    argv: Sequence[str] | None,
    *,
    description: str,
    operations: Sequence[str],
    extra: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    """Parse the exact generic command emitted by the supervisor."""

    parser = common_parser(description, operations)
    if extra is not None:
        extra(parser)
    return parser.parse_args(argv)


def assert_common_paths(args: argparse.Namespace, *, need_dataset: bool = False) -> None:
    if not args.repo.is_dir():
        raise FileNotFoundError(f"CARE repository is absent: {args.repo}")
    if not args.benchmark_repo.is_dir():
        raise FileNotFoundError(f"BiCoord repository is absent: {args.benchmark_repo}")
    if need_dataset and not args.dataset.is_dir():
        raise FileNotFoundError(f"BiCoord dataset is absent: {args.dataset}")
    if len(str(args.config_sha256)) != 64:
        raise ValueError("supervisor config SHA-256 must contain 64 hex characters")
    try:
        int(str(args.config_sha256), 16)
    except ValueError as error:
        raise ValueError("supervisor config SHA-256 is not hexadecimal") from error


def model_contract() -> dict[str, Any]:
    # Delayed import avoids constructing the supervisor when a lightweight
    # data utility imports this module.
    from .supervisor import MODEL_CONTRACT

    return dict(MODEL_CONTRACT)


def publish_result(
    args: argparse.Namespace,
    *,
    stage: str,
    artifacts: Iterable[Mapping[str, str]] = (),
    include_model_contract: bool = False,
    **fields: Any,
) -> dict[str, Any]:
    rows = [dict(row) for row in artifacts]
    # Validate every evidence hash again immediately before publication.
    for row in rows:
        source = Path(row["path"])
        if not source.is_file() or sha256_file(source) != row.get("sha256"):
            raise ValueError(f"stage evidence changed before publication: {source}")
    protected = {
        "schema": RESULT_SCHEMA,
        "stage": stage,
        "status": "PASSED",
        "benchmark_adapter": "BiCoord",
        "config_sha256": str(args.config_sha256),
    }
    overlap = set(fields).intersection(protected)
    if overlap:
        bad = {
            key: (fields[key], protected[key])
            for key in overlap
            if fields[key] != protected[key]
        }
        if bad:
            raise ValueError(f"stage result attempted to override protected identity: {bad}")
        fields = {key: item for key, item in fields.items() if key not in overlap}
    value: dict[str, Any] = {
        **protected,
        "artifacts": rows,
        "completed_at": utc_now(),
    }
    if include_model_contract:
        value["model_contract"] = model_contract()
    value.update(fields)
    atomic_json(args.result, value)
    return value


def emit_result(
    result_path: str | Path,
    *,
    stage: str,
    config_sha256: str,
    artifacts: Iterable[str | Path] = (),
    fields: Mapping[str, Any] | None = None,
    model: bool = False,
) -> dict[str, Any]:
    """Publish a result when a caller has paths rather than artifact rows."""

    namespace = argparse.Namespace(
        result=Path(result_path), config_sha256=str(config_sha256)
    )
    return publish_result(
        namespace,
        stage=stage,
        artifacts=(artifact(path) for path in artifacts),
        include_model_contract=model,
        **dict(fields or {}),
    )


def stage_result_path(run: str | Path, stage: str) -> Path:
    return Path(run) / "stage_results" / f"{stage}.json"


def require_stage_result(
    run: str | Path,
    stage: str,
    *,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    path = stage_result_path(run, stage)
    value = read_json(path)
    expected = {
        "schema": RESULT_SCHEMA,
        "stage": stage,
        "status": "PASSED",
        "benchmark_adapter": "BiCoord",
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"invalid {stage} dependency result at {key}")
    if config_sha256 is not None and value.get("config_sha256") != config_sha256:
        raise ValueError(f"{stage} dependency belongs to a different frozen config")
    return value


def require_hashed_artifact(
    result: Mapping[str, Any],
    *,
    kind: str | None = None,
    suffix: str | None = None,
) -> Path:
    matches: list[Path] = []
    for row in result.get("artifacts", []):
        if not isinstance(row, Mapping):
            continue
        if kind is not None and row.get("kind") != kind:
            continue
        path = Path(str(row.get("path", "")))
        if suffix is not None and not str(path).endswith(suffix):
            continue
        if path.is_file() and sha256_file(path) == row.get("sha256"):
            matches.append(path.resolve())
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one dependency artifact kind={kind!r} suffix={suffix!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


__all__ = [
    "RESULT_SCHEMA",
    "artifact",
    "artifact_record",
    "assert_common_paths",
    "atomic_json",
    "canonical_sha256",
    "common_parser",
    "emit_result",
    "model_contract",
    "publish_result",
    "parse_args",
    "read_json",
    "require_hashed_artifact",
    "require_stage_result",
    "sha256_file",
    "stage_result_path",
    "utc_now",
]
