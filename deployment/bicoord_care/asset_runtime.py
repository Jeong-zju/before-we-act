"""Runtime loader for the audited BiCoord plate metadata overlay.

The upstream benchmark checkout remains byte-for-byte at its pinned Git
revision.  Simulator adapters call :func:`apply_configured_task_overlay` after
``setup_demo``; for ``place_plate_and_cup`` it replaces only the two actor
configs' ``contact_points_pose`` value with the separately audited overlay.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .asset_contract import (
    CONTACT_KEY,
    DEFAULT_SMALL_SCALE,
    canonical_json_sha256,
    sha256_file,
)
from .config import DATASET_REPO_ID, DATASET_REVISION, TASKS
from .preflight import EXPECTED_BENCHMARK_COMMIT


OVERLAY_ENV = "BICOORD_PLATE_ASSET_OVERLAY"
REQUIRED_ENV = "BICOORD_REQUIRE_ASSET_OVERLAY"
PLATE_TASK = "place_plate_and_cup"
PLATE_ACTOR_ATTRIBUTES = ("plate", "plate_2")


class RuntimeAssetError(RuntimeError):
    """Raised when a configured runtime overlay is missing or inconsistent."""


def _load_overlay(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise RuntimeAssetError(f"plate overlay must not be a symlink: {source}")
    try:
        source = source.resolve(strict=True)
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeAssetError(f"cannot load audited plate overlay: {source}") from error
    if not isinstance(value, dict):
        raise RuntimeAssetError("audited plate overlay is not a JSON object")
    try:
        receipt_path = source.parents[2] / "asset_contract.json"
    except IndexError as error:
        raise RuntimeAssetError("audited plate overlay path has no stage receipt") from error
    if receipt_path.is_symlink():
        raise RuntimeAssetError(
            f"plate overlay stage receipt must not be a symlink: {receipt_path}"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeAssetError(
            f"cannot load plate overlay stage receipt: {receipt_path}"
        ) from error
    if not isinstance(receipt, Mapping):
        raise RuntimeAssetError("plate overlay stage receipt is not a JSON object")
    plate = receipt.get("plate_overlay")
    if (
        receipt.get("schema") != "before-we-act.bicoord.asset-contract/1"
        or receipt.get("status") != "PASSED"
        or receipt.get("dataset_repo_id") != DATASET_REPO_ID
        or receipt.get("dataset_revision") != DATASET_REVISION
        or receipt.get("benchmark_revision") != EXPECTED_BENCHMARK_COMMIT
        or receipt.get("tasks") != list(TASKS)
        or receipt.get("supplemental_assets_installed") is not True
        or receipt.get("benchmark_tracked_source_modified") is not False
        or receipt.get("task_source_modified") is not False
        or receipt.get("upstream_model_modified") is not False
        or receipt.get("normalization_modified") is not False
        or not isinstance(plate, Mapping)
        or Path(str(plate.get("overlay_metadata", ""))).resolve() != source
        or plate.get("target_metadata_sha256") != sha256_file(source)
        or plate.get("copied_fields") != [CONTACT_KEY]
        or plate.get("benchmark_asset_source_modified") is not False
    ):
        raise RuntimeAssetError("audited plate overlay stage receipt/hash differs")
    scale = value.get("scale")
    if scale != list(DEFAULT_SMALL_SCALE):
        raise RuntimeAssetError(
            f"audited plate overlay changed the small scale: {scale!r}"
        )
    contacts = value.get(CONTACT_KEY)
    if not isinstance(contacts, list) or len(contacts) < 3:
        raise RuntimeAssetError("audited plate overlay lacks contact point two")
    return source, value


def _apply_actor_overlay(actor: Any, overlay: Mapping[str, Any]) -> dict[str, Any]:
    config = getattr(actor, "config", None)
    if not isinstance(config, Mapping):
        raise RuntimeAssetError("small plate actor has no metadata mapping")
    if config.get("scale") != list(DEFAULT_SMALL_SCALE):
        raise RuntimeAssetError("small plate actor scale differs from official 0.025")
    for key in set(config) | set(overlay):
        if key == CONTACT_KEY:
            continue
        if config.get(key) != overlay.get(key):
            raise RuntimeAssetError(
                f"small plate actor metadata differs from overlay at {key}"
            )
    before = config.get(CONTACT_KEY)
    contacts = overlay[CONTACT_KEY]
    if before not in ([], contacts):
        raise RuntimeAssetError("small plate actor has conflicting contact metadata")
    updated = copy.deepcopy(dict(config))
    updated[CONTACT_KEY] = copy.deepcopy(contacts)
    actor.config = updated
    return {
        "before_sha256": canonical_json_sha256(before),
        "after_sha256": canonical_json_sha256(updated[CONTACT_KEY]),
        "contact_points_pose_count": len(updated[CONTACT_KEY]),
        "scale_preserved": updated.get("scale") == config.get("scale"),
    }


def apply_task_overlay(
    env: Any,
    task: str,
    overlay_path: str | Path,
) -> dict[str, Any]:
    """Apply the audited overlay to one newly-created official task env."""

    if task != PLATE_TASK:
        return {
            "task": task,
            "applied": False,
            "reason": "task_does_not_reference_003_plate",
        }
    source, overlay = _load_overlay(overlay_path)
    actors: dict[str, Any] = {}
    for attribute in PLATE_ACTOR_ATTRIBUTES:
        actor = getattr(env, attribute, None)
        if actor is None:
            raise RuntimeAssetError(f"official plate task lacks self.{attribute}")
        actors[attribute] = _apply_actor_overlay(actor, overlay)
    hashes = {row["after_sha256"] for row in actors.values()}
    if len(hashes) != 1:
        raise RuntimeAssetError("plate actors received different contact metadata")
    return {
        "task": task,
        "applied": True,
        "overlay": str(source),
        "contact_points_pose_sha256": next(iter(hashes)),
        "actors": actors,
        "copied_fields": [CONTACT_KEY],
        "task_source_modified": False,
    }


def apply_configured_task_overlay(env: Any, task: str) -> dict[str, Any]:
    """Apply the supervisor-configured overlay, failing closed when required."""

    configured = os.environ.get(OVERLAY_ENV)
    required = os.environ.get(REQUIRED_ENV) == "1"
    if task != PLATE_TASK:
        return {"task": task, "applied": False, "reason": "not_required"}
    if not configured:
        if required:
            raise RuntimeAssetError(
                f"{OVERLAY_ENV} is required for the released {PLATE_TASK} task"
            )
        return {"task": task, "applied": False, "reason": "not_configured"}
    return apply_task_overlay(env, task, configured)


__all__ = [
    "OVERLAY_ENV",
    "PLATE_TASK",
    "REQUIRED_ENV",
    "RuntimeAssetError",
    "apply_configured_task_overlay",
    "apply_task_overlay",
]
