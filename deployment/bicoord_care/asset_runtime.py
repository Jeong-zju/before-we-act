"""Runtime loader for the audited BiCoord object-metadata overlays.

The upstream benchmark checkout remains byte-for-byte at its pinned Git
revision.  Simulator adapters call :func:`apply_configured_task_overlay` after
``setup_demo``.  It changes only in-memory ``contact_points_pose`` fields on
the actors whose released metadata is incompatible with the current loader:
the two plates in ``place_plate_and_cup`` and the official model-3 shovel in
``sweep_block``.  Both run-local files must be outputs of one passed asset
contract receipt.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .asset_contract import (
    AssetContractError,
    CONTACT_KEY,
    DEFAULT_SHOVEL_SCALE,
    DEFAULT_SMALL_SCALE,
    LEGACY_CONTACT_KEY,
    LEGACY_TRANSFORM_KEY,
    PRISTINE_SHOVEL_METADATA_SHA256,
    SHOVEL_COLLISION_BYTES,
    SHOVEL_COLLISION_SHA256,
    SHOVEL_METADATA_NAME,
    SHOVEL_MODEL_ID,
    SHOVEL_OBJECT_NAME,
    SHOVEL_VISUAL_BYTES,
    SHOVEL_VISUAL_SHA256,
    canonical_json_sha256,
    overlay_legacy_contact_metadata,
    sha256_file,
)
from .config import DATASET_REPO_ID, DATASET_REVISION, TASKS
from .preflight import EXPECTED_BENCHMARK_COMMIT


ASSET_RECEIPT_SCHEMA = "before-we-act.bicoord.asset-contract/1"
OVERLAY_ENV = "BICOORD_PLATE_ASSET_OVERLAY"  # Backward-compatible name.
PLATE_OVERLAY_ENV = OVERLAY_ENV
SHOVEL_OVERLAY_ENV = "BICOORD_SHOVEL_ASSET_OVERLAY"
REQUIRED_ENV = "BICOORD_REQUIRE_ASSET_OVERLAY"
PLATE_TASK = "place_plate_and_cup"
SHOVEL_TASK = "sweep_block"
PLATE_ACTOR_ATTRIBUTES = ("plate", "plate_2")
SHOVEL_ACTOR_ATTRIBUTES = ("shovel",)
_OVERLAY_TASKS = (PLATE_TASK, SHOVEL_TASK)
_TASK_ENV = {
    PLATE_TASK: PLATE_OVERLAY_ENV,
    SHOVEL_TASK: SHOVEL_OVERLAY_ENV,
}
_TASK_RECEIPT_KEY = {
    PLATE_TASK: "plate_overlay",
    SHOVEL_TASK: "shovel_overlay",
}


class RuntimeAssetError(RuntimeError):
    """Raised when a configured runtime overlay is missing or inconsistent."""


def _read_json_object(path: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise RuntimeAssetError(f"{label} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise RuntimeAssetError(f"cannot resolve {label}: {candidate}") from error
    if not resolved.is_file():
        raise RuntimeAssetError(f"{label} is not a regular file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeAssetError(f"cannot load {label}: {resolved}") from error
    if not isinstance(value, dict):
        raise RuntimeAssetError(f"{label} is not a JSON object")
    return resolved, value


def _receipt_path(source: Path) -> Path:
    try:
        return source.parents[2] / "asset_contract.json"
    except IndexError as error:
        raise RuntimeAssetError("audited overlay path has no stage receipt") from error


def _common_receipt(source: Path) -> tuple[Path, Mapping[str, Any]]:
    receipt_candidate = _receipt_path(source)
    if receipt_candidate.is_symlink():
        raise RuntimeAssetError(
            f"overlay stage receipt must not be a symlink: {receipt_candidate}"
        )
    receipt_path, receipt = _read_json_object(
        receipt_candidate, label="overlay stage receipt"
    )
    if (
        receipt.get("schema") != ASSET_RECEIPT_SCHEMA
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
        or not isinstance(receipt.get("plate_overlay"), Mapping)
        or not isinstance(receipt.get("shovel_overlay"), Mapping)
    ):
        raise RuntimeAssetError("audited overlay stage receipt/hash differs")

    # Both compatibility files are one atomic run contract.  Re-hash each
    # target and prove that both paths lead back to this exact receipt before
    # an actor config can be touched.
    expected_layout = {
        "plate_overlay": ("003_plate", "model_data0.json"),
        "shovel_overlay": (SHOVEL_OBJECT_NAME, SHOVEL_METADATA_NAME),
    }
    for receipt_key, (object_name, metadata_name) in expected_layout.items():
        row = receipt[receipt_key]
        assert isinstance(row, Mapping)
        raw_target = str(row.get("overlay_metadata", ""))
        target = Path(raw_target).expanduser()
        if target.is_symlink():
            raise RuntimeAssetError(
                f"audited {receipt_key} target must not be a symlink: {target}"
            )
        try:
            target = target.resolve(strict=True)
        except OSError as error:
            raise RuntimeAssetError(
                f"audited {receipt_key} target is unavailable: {raw_target}"
            ) from error
        if (
            not target.is_file()
            or target.name != metadata_name
            or target.parent.name != object_name
            or target.parent.parent.name != "overlay"
            or _receipt_path(target).resolve() != receipt_path
            or row.get("target_metadata_sha256") != sha256_file(target)
        ):
            raise RuntimeAssetError(
                f"audited {receipt_key} target/receipt hash differs"
            )
    return receipt_path, receipt


def _load_source_metadata(
    row: Mapping[str, Any],
    *,
    path_field: str,
    hash_field: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    raw_path = row.get(path_field)
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeAssetError(f"audited {label} lacks {path_field}")
    source, value = _read_json_object(Path(raw_path), label=label)
    digest = sha256_file(source)
    if row.get(hash_field) != digest:
        raise RuntimeAssetError(f"audited {label} source hash differs")
    return source, value, digest


def _load_plate_overlay(
    source: Path,
    value: dict[str, Any],
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plate = receipt["plate_overlay"]
    assert isinstance(plate, Mapping)
    if (
        Path(str(plate.get("overlay_metadata", ""))).resolve() != source
        or plate.get("target_metadata_sha256") != sha256_file(source)
        or plate.get("copied_fields") != [CONTACT_KEY]
        or plate.get("benchmark_asset_source_modified") is not False
        or plate.get("mutation_scope")
        != "run_artifact_and_actor_config_in_memory_only"
    ):
        raise RuntimeAssetError("audited plate overlay stage receipt/hash differs")
    pristine_path, pristine, pristine_digest = _load_source_metadata(
        plate,
        path_field="source_small_metadata",
        hash_field="source_small_metadata_sha256",
        label="pristine small plate metadata",
    )
    if (
        plate.get("pristine_small_metadata_sha256") != pristine_digest
        or pristine.get(CONTACT_KEY) != []
        or pristine_path.name != "model_data0.json"
        or pristine_path.parent.name != "003_plate"
    ):
        raise RuntimeAssetError("audited plate pristine source identity differs")
    for key in set(pristine) | set(value):
        if key != CONTACT_KEY and pristine.get(key) != value.get(key):
            raise RuntimeAssetError(
                f"audited plate source differs from overlay at {key}"
            )
    if value.get("scale") != list(DEFAULT_SMALL_SCALE):
        raise RuntimeAssetError(
            f"audited plate overlay changed the small scale: {value.get('scale')!r}"
        )
    contacts = value.get(CONTACT_KEY)
    if not isinstance(contacts, list) or len(contacts) < 3:
        raise RuntimeAssetError("audited plate overlay lacks contact point two")
    contact_digest = canonical_json_sha256(contacts)
    if plate.get("target_contact_points_pose_sha256") != contact_digest:
        raise RuntimeAssetError("audited plate contact hash differs")
    return value, {
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "contact_points_pose_sha256": contact_digest,
        "copied_fields": [CONTACT_KEY],
    }


def _validate_shovel_meshes(
    row: Mapping[str, Any], source_metadata: Path
) -> None:
    identity = row.get("asset_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeAssetError("audited shovel overlay lacks asset identity")
    if (
        identity.get("object") != SHOVEL_OBJECT_NAME
        or identity.get("model_id") != SHOVEL_MODEL_ID
        or Path(str(identity.get("source_metadata", ""))).resolve()
        != source_metadata
        or identity.get("source_metadata_sha256")
        != PRISTINE_SHOVEL_METADATA_SHA256
        or identity.get("mesh_and_metadata_identity") != "PASSED"
    ):
        raise RuntimeAssetError("audited shovel asset identity differs")
    rows = identity.get("meshes")
    if not isinstance(rows, list):
        raise RuntimeAssetError("audited shovel mesh inventory is absent")
    expected = {
        "collision/base3.glb": (SHOVEL_COLLISION_BYTES, SHOVEL_COLLISION_SHA256),
        "visual/base3.glb": (SHOVEL_VISUAL_BYTES, SHOVEL_VISUAL_SHA256),
    }
    observed: dict[str, tuple[int, str]] = {}
    for mesh in rows:
        if not isinstance(mesh, Mapping):
            raise RuntimeAssetError("audited shovel mesh row is not an object")
        relative = str(mesh.get("relative_path", ""))
        if relative not in expected or relative in observed:
            raise RuntimeAssetError("audited shovel mesh inventory differs")
        expected_bytes, expected_sha = expected[relative]
        path = Path(str(mesh.get("path", ""))).expanduser()
        if path.is_symlink():
            raise RuntimeAssetError(f"audited shovel mesh must not be a symlink: {path}")
        try:
            path = path.resolve(strict=True)
        except OSError as error:
            raise RuntimeAssetError(f"audited shovel mesh is missing: {path}") from error
        if (
            path != (source_metadata.parent / relative).resolve()
            or not path.is_file()
            or path.stat().st_size != expected_bytes
            or mesh.get("bytes") != expected_bytes
            or mesh.get("sha256") != expected_sha
            or sha256_file(path) != expected_sha
        ):
            raise RuntimeAssetError(f"audited shovel mesh identity differs: {relative}")
        observed[relative] = (expected_bytes, expected_sha)
    if observed != expected:
        raise RuntimeAssetError("audited shovel mesh inventory differs")


def _load_shovel_overlay(
    source: Path,
    value: dict[str, Any],
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    shovel = receipt["shovel_overlay"]
    assert isinstance(shovel, Mapping)
    source_metadata, pristine, pristine_digest = _load_source_metadata(
        shovel,
        path_field="source_metadata",
        hash_field="source_metadata_sha256",
        label="pristine model-3 shovel metadata",
    )
    if (
        Path(str(shovel.get("overlay_metadata", ""))).resolve() != source
        or shovel.get("target_metadata_sha256") != sha256_file(source)
        or source_metadata.name != SHOVEL_METADATA_NAME
        or source_metadata.parent.name != SHOVEL_OBJECT_NAME
        or pristine_digest != PRISTINE_SHOVEL_METADATA_SHA256
        or shovel.get("pristine_source_metadata_sha256")
        != PRISTINE_SHOVEL_METADATA_SHA256
        or shovel.get("contact_points_pose_count") != 1
        or shovel.get("added_fields") != [CONTACT_KEY]
        or shovel.get("derived_fields") != [CONTACT_KEY]
        or shovel.get("source_fields")
        != [LEGACY_CONTACT_KEY, LEGACY_TRANSFORM_KEY]
        or shovel.get("scale_preserved") is not True
        or shovel.get("benchmark_asset_source_modified") is not False
        or shovel.get("mutation_scope")
        != "run_artifact_and_actor_config_in_memory_only"
    ):
        raise RuntimeAssetError("audited shovel overlay stage receipt/hash differs")
    equivalence_error = shovel.get("max_scale_equivalence_error")
    if (
        isinstance(equivalence_error, bool)
        or not isinstance(equivalence_error, (int, float))
        or not (0 <= equivalence_error <= 1e-12)
    ):
        raise RuntimeAssetError("audited shovel scale-equivalence proof differs")
    try:
        expected_value, proof = overlay_legacy_contact_metadata(pristine)
    except AssetContractError as contract_error:
        raise RuntimeAssetError("cannot reproduce audited shovel conversion") from contract_error
    if value != expected_value:
        differing = sorted(
            key
            for key in set(value) | set(expected_value)
            if value.get(key) != expected_value.get(key)
        )
        raise RuntimeAssetError(
            f"audited shovel overlay differs from deterministic conversion: {differing}"
        )
    contacts = value.get(CONTACT_KEY)
    contact_digest = canonical_json_sha256(contacts)
    if (
        not isinstance(contacts, list)
        or len(contacts) != 1
        or shovel.get("contact_points_pose_sha256") != contact_digest
        or proof.get("contact_points_pose_sha256") != contact_digest
        or float(proof.get("max_scale_equivalence_error", -1.0))
        != float(equivalence_error)
        or value.get("scale") != list(DEFAULT_SHOVEL_SCALE)
    ):
        raise RuntimeAssetError("audited shovel converted contact metadata differs")
    _validate_shovel_meshes(shovel, source_metadata)
    return value, {
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "contact_points_pose_sha256": contact_digest,
        "copied_fields": [CONTACT_KEY],
        "derived_fields": [CONTACT_KEY],
        "source_fields": [LEGACY_CONTACT_KEY, LEGACY_TRANSFORM_KEY],
        "legacy_conversion": True,
    }


def _load_overlay(
    path: str | Path, task: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if task not in _OVERLAY_TASKS:
        raise RuntimeAssetError(f"task has no audited runtime overlay: {task}")
    source, value = _read_json_object(Path(path), label=f"audited {task} overlay")
    receipt_path, receipt = _common_receipt(source)
    expected_entry = receipt[_TASK_RECEIPT_KEY[task]]
    assert isinstance(expected_entry, Mapping)
    if Path(str(expected_entry.get("overlay_metadata", ""))).resolve() != source:
        raise RuntimeAssetError(f"audited {task} overlay path differs from receipt")
    if task == PLATE_TASK:
        overlay, provenance = _load_plate_overlay(
            source, value, receipt_path, receipt
        )
    else:
        overlay, provenance = _load_shovel_overlay(
            source, value, receipt_path, receipt
        )
    return source, overlay, provenance


def _apply_actor_overlay(
    actor: Any,
    overlay: Mapping[str, Any],
    *,
    actor_label: str,
    expected_scale: tuple[float, float, float],
    missing_contact_allowed: bool,
) -> dict[str, Any]:
    config = getattr(actor, "config", None)
    if not isinstance(config, Mapping):
        raise RuntimeAssetError(f"{actor_label} actor has no metadata mapping")
    if config.get("scale") != list(expected_scale):
        raise RuntimeAssetError(
            f"{actor_label} actor scale differs from official {list(expected_scale)}"
        )
    for key in set(config) | set(overlay):
        if key == CONTACT_KEY:
            continue
        if config.get(key) != overlay.get(key):
            raise RuntimeAssetError(
                f"{actor_label} actor metadata differs from overlay at {key}"
            )
    before = config.get(CONTACT_KEY)
    contacts = overlay[CONTACT_KEY]
    permitted = [[], contacts]
    if missing_contact_allowed:
        permitted.append(None)
    if before not in permitted:
        raise RuntimeAssetError(f"{actor_label} actor has conflicting contact metadata")
    updated = copy.deepcopy(dict(config))
    updated[CONTACT_KEY] = copy.deepcopy(contacts)
    actor.config = updated
    return {
        "before_sha256": canonical_json_sha256(before),
        "after_sha256": canonical_json_sha256(updated[CONTACT_KEY]),
        "contact_points_pose_count": len(updated[CONTACT_KEY]),
        "scale_preserved": updated.get("scale") == config.get("scale"),
        "changed_fields": [CONTACT_KEY] if before != contacts else [],
    }


def apply_task_overlay(
    env: Any,
    task: str,
    overlay_path: str | Path,
) -> dict[str, Any]:
    """Apply one audited overlay to a newly-created official task env."""

    if task not in _OVERLAY_TASKS:
        return {"task": task, "applied": False, "reason": "not_required"}
    source, overlay, provenance = _load_overlay(overlay_path, task)
    if task == PLATE_TASK:
        attributes = PLATE_ACTOR_ATTRIBUTES
        actor_label = "small plate"
        expected_scale = DEFAULT_SMALL_SCALE
        missing_contact_allowed = False
    else:
        attributes = SHOVEL_ACTOR_ATTRIBUTES
        actor_label = "model-3 small shovel"
        expected_scale = DEFAULT_SHOVEL_SCALE
        missing_contact_allowed = True
    actors: dict[str, Any] = {}
    actor_objects: dict[str, Any] = {}
    original_configs: dict[str, Any] = {}
    for attribute in attributes:
        actor = getattr(env, attribute, None)
        if actor is None:
            raise RuntimeAssetError(f"official {task} task lacks self.{attribute}")
        actor_objects[attribute] = actor
        original_configs[attribute] = copy.deepcopy(getattr(actor, "config", None))
    try:
        for attribute, actor in actor_objects.items():
            actors[attribute] = _apply_actor_overlay(
                actor,
                overlay,
                actor_label=actor_label,
                expected_scale=expected_scale,
                missing_contact_allowed=missing_contact_allowed,
            )
    except BaseException:
        # Applying two plate actors is a single compatibility operation.  If
        # the second actor has drifted, restore the first actor's original
        # in-memory metadata before propagating the failure; otherwise a
        # caller could accidentally continue with a half-overlaid scene.
        for attribute, actor in actor_objects.items():
            try:
                actor.config = original_configs[attribute]
            except Exception:
                pass
        raise
    hashes = {row["after_sha256"] for row in actors.values()}
    if hashes != {provenance["contact_points_pose_sha256"]}:
        raise RuntimeAssetError(f"{task} actors received unaudited contact metadata")
    return {
        "task": task,
        "applied": True,
        "overlay": str(source),
        "contact_points_pose_sha256": next(iter(hashes)),
        "actors": actors,
        **provenance,
        "task_source_modified": False,
    }


def apply_configured_task_overlay(env: Any, task: str) -> dict[str, Any]:
    """Apply the supervisor-configured overlay, failing closed when required."""

    required = os.environ.get(REQUIRED_ENV) == "1"
    if task not in _OVERLAY_TASKS:
        return {"task": task, "applied": False, "reason": "not_required"}
    environment_name = _TASK_ENV[task]
    configured = os.environ.get(environment_name)
    if not configured:
        if required:
            raise RuntimeAssetError(
                f"{environment_name} is required for the released {task} task"
            )
        return {"task": task, "applied": False, "reason": "not_configured"}
    # Validate the complete two-file binding *before* touching an actor.  This
    # matters when a launcher accidentally carries one overlay from a prior
    # run: fail closed without leaving a partially modified simulator state.
    if required:
        configured_candidate = Path(configured).expanduser()
        if configured_candidate.is_symlink():
            raise RuntimeAssetError(
                f"configured {environment_name} overlay must not be a symlink"
            )
        try:
            configured_source = configured_candidate.resolve(strict=True)
        except OSError as error:
            raise RuntimeAssetError(
                f"configured {environment_name} overlay is unavailable"
            ) from error
        receipt_path, receipt = _common_receipt(configured_source)
        for other_task in _OVERLAY_TASKS:
            other_env = _TASK_ENV[other_task]
            other_configured = os.environ.get(other_env)
            if not other_configured:
                raise RuntimeAssetError(
                    f"{other_env} is required by the shared asset receipt"
                )
            expected = receipt[_TASK_RECEIPT_KEY[other_task]]
            assert isinstance(expected, Mapping)
            try:
                observed_path = Path(other_configured).expanduser().resolve(strict=True)
                expected_path = Path(str(expected["overlay_metadata"])).resolve(strict=True)
            except (OSError, KeyError) as error:
                raise RuntimeAssetError(
                    "configured overlays do not resolve through the shared asset receipt"
                ) from error
            if (
                observed_path != expected_path
                or _receipt_path(observed_path).resolve() != receipt_path.resolve()
            ):
                raise RuntimeAssetError(
                    "configured plate/shovel overlays do not share one asset receipt"
                )
    return apply_task_overlay(env, task, configured)


__all__ = [
    "OVERLAY_ENV",
    "PLATE_OVERLAY_ENV",
    "PLATE_TASK",
    "REQUIRED_ENV",
    "RuntimeAssetError",
    "SHOVEL_OVERLAY_ENV",
    "SHOVEL_ACTOR_ATTRIBUTES",
    "SHOVEL_TASK",
    "apply_configured_task_overlay",
    "apply_task_overlay",
]
