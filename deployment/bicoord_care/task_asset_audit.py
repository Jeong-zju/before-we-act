"""Static, fail-closed audit of assets referenced by BiCoord task sources.

The benchmark task files are deliberately not imported here: importing them
would initialise SAPIEN and would make a preflight dependent on a renderer.
Instead, a small AST evaluator extracts only literal actor/model/index
contracts.  Expressions which depend on RNG, a runtime lookup, a list index,
or another unsupported construct are reported as ``dynamic_items`` and are
never guessed.  This keeps the audit useful without turning a static check
into a source of false positives.

The audit also performs an in-memory compatibility check for
``place_plate_and_cup``.  It proves that the pristine small plate rejects
contact index 2, then overlays the contact poses from the mesh-identical
``003_plate_large`` metadata and proves that the same index is valid.  No
benchmark file is written.
"""

from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence


SCHEMA: Final[str] = "before-we-act.bicoord.task-asset-audit/1"
DEFAULT_MODEL_ID: Final[int] = 0
ACTOR_BUILDERS: Final[frozenset[str]] = frozenset({"create_actor"})
INTERACTION_METHODS: Final[frozenset[str]] = frozenset({"grasp_actor", "place_actor"})
CONTACT_KEY: Final[str] = "contact_points_pose"
FUNCTIONAL_KEY: Final[str] = "functional_matrix"
# A few historical RoboTwin object records use this spelling.  It is not
# interchangeable with the current simulator contract (we cannot safely
# infer its pose convention), so those records are surfaced as
# ``unsupported/dynamic`` rather than guessed or turned into a hard index
# failure.  The same policy is used for other plausible, list-valued legacy
# fields below.
LEGACY_CONTACT_KEYS: Final[frozenset[str]] = frozenset({"contact_pose"})
LEGACY_FUNCTIONAL_KEYS: Final[frozenset[str]] = frozenset(
    {"functional_points", "functional_point_pose", "functional_pose"}
)
DEFAULT_TASKS: Final[tuple[str, ...]] = (
    "balance_roller",
    "build_bridge",
    "build_tower_with_blocks",
    "clean_table",
    "collect_pens",
    "cook",
    "divide_block_tower",
    "exchange_mics",
    "exchange_pots",
    "extract_bottom_block_to_top",
    "fetch_block_with_roller",
    "handover_block_with_bowls",
    "jigsaw",
    "match_blocks_with_signs",
    "place_plate_and_cup",
    "put_objects_cabinet",
    "stack_bowls",
    "sweep_block",
)


class TaskAssetAuditError(RuntimeError):
    """Raised for malformed audit inputs or an explicitly requested gate."""


# An override is deliberately restricted to a JSON object (already decoded)
# or a path to one.  Keeping this type narrow prevents callers from passing an
# arbitrary callback/executable object into the static audit process.
MetadataOverride = str | Path | Mapping[str, Any]


def _override_key_dict(modelname: str, model_id: int | None) -> dict[str, Any]:
    return {"modelname": modelname, "model_id": model_id}


def _override_sort_key(
    key: tuple[str, int | None],
) -> tuple[str, int, int]:
    return key[0], int(key[1] is not None), int(key[1] or 0)


@dataclass
class OverrideRecord:
    """Decoded override plus auditable use/provenance information."""

    modelname: str
    model_id: int | None
    source_type: str
    source_path: Path | None = None
    source_sha256: str | None = None
    metadata: dict[str, Any] | None = None
    metadata_sha256: str | None = None
    error: str | None = None
    pristine_metadata_path: Path | None = None
    pristine_source_sha256: str | None = None
    pristine_metadata_sha256: str | None = None
    contract_status: str | None = None
    contract_reason: str | None = None
    actor_uses: list[dict[str, Any]] = field(default_factory=list)
    interaction_uses: list[dict[str, Any]] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int | None]:
        return self.modelname, self.model_id

    def bind_pristine(
        self,
        path: Path,
        metadata: Mapping[str, Any] | None,
        *,
        source_sha256: str | None,
        metadata_sha256: str | None,
    ) -> None:
        """Bind the source record the override replaces, without ambiguity."""

        if self.pristine_metadata_path is not None and self.pristine_metadata_path != path:
            self.error = "one override key resolved to multiple pristine metadata paths"
            return
        self.pristine_metadata_path = path
        self.pristine_source_sha256 = source_sha256
        self.pristine_metadata_sha256 = metadata_sha256

    def as_dict(self) -> dict[str, Any]:
        if self.error is not None:
            status = "FAILED"
        elif self.interaction_uses:
            status = "USED"
        elif self.actor_uses:
            status = "ACTOR_ONLY"
        else:
            status = "UNUSED"
        return {
            "key": _override_key_dict(self.modelname, self.model_id),
            "status": status,
            "source_type": self.source_type,
            "source_path": str(self.source_path) if self.source_path is not None else None,
            # For a file override this is the hash of its exact bytes.  A
            # mapping has no source byte stream, so this is intentionally null.
            "source_sha256": self.source_sha256,
            "override_metadata_sha256": self.metadata_sha256,
            # ``overlay_metadata_sha256`` is an explicit alias used by the
            # plate compatibility receipt and keeps its provenance readable.
            "overlay_metadata_sha256": self.metadata_sha256,
            "pristine_metadata_path": (
                str(self.pristine_metadata_path)
                if self.pristine_metadata_path is not None
                else None
            ),
            "pristine_source_sha256": self.pristine_source_sha256,
            "pristine_metadata_sha256": self.pristine_metadata_sha256,
            "contract_status": self.contract_status,
            "contract_reason": self.contract_reason,
            "error": self.error,
            "used_by_actor_count": len(self.actor_uses),
            "used_by_interaction_count": len(self.interaction_uses),
            "used_by_actors": list(self.actor_uses),
            "used_by_interactions": list(self.interaction_uses),
        }


@dataclass(frozen=True)
class DynamicItem:
    task: str
    source: str
    line: int
    kind: str
    expression: str
    reason: str
    actor: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value = {
            "task": self.task,
            "source": self.source,
            "line": self.line,
            "kind": self.kind,
            "expression": self.expression,
            "reason": self.reason,
        }
        if self.actor is not None:
            value["actor"] = self.actor
        return value


@dataclass
class ActorReference:
    attr: str
    source: str
    line: int
    modelname: str | None
    model_id: int | None
    model_id_defaulted: bool
    modelname_expression: str
    model_id_expression: str
    # ``metadata_path`` always names the pristine benchmark record.  An
    # override path (if any) is tracked separately; a mapping override has no
    # filesystem path at all.
    metadata_path: Path | None = None
    pristine_metadata_sha256: str | None = None
    pristine_source_sha256: str | None = None
    metadata: dict[str, Any] | None = None
    metadata_source: str = "unresolved"
    effective_metadata_path: Path | None = None
    effective_metadata_sha256: str | None = None
    effective_source_sha256: str | None = None
    override_selected: bool = False
    override_used: bool = False
    override_key: tuple[str, int | None] | None = None

    def as_dict(self, assets_root: Path) -> dict[str, Any]:
        return {
            "attribute": self.attr,
            "source": self.source,
            "line": self.line,
            "modelname": self.modelname,
            "model_id": self.model_id,
            "model_id_defaulted": self.model_id_defaulted,
            "modelname_expression": self.modelname_expression,
            "model_id_expression": self.model_id_expression,
            "metadata_path": (
                str(self.metadata_path.resolve()) if self.metadata_path is not None else None
            ),
            "metadata_present": self.metadata is not None,
            "pristine_metadata_path": (
                str(self.metadata_path.resolve()) if self.metadata_path is not None else None
            ),
            "pristine_source_sha256": self.pristine_source_sha256,
            "pristine_metadata_sha256": self.pristine_metadata_sha256,
            "effective_metadata_source": self.metadata_source,
            "effective_metadata_path": (
                str(self.effective_metadata_path.resolve())
                if self.effective_metadata_path is not None
                else None
            ),
            "effective_source_sha256": self.effective_source_sha256,
            "effective_metadata_sha256": self.effective_metadata_sha256,
            "override_selected": self.override_selected,
            "override_used": self.override_used,
            "override_key": (
                _override_key_dict(*self.override_key) if self.override_key is not None else None
            ),
        }


@dataclass
class InteractionReference:
    task: str
    source: str
    line: int
    kind: str
    actor_expression: str
    actor: str | None
    index_expression: str
    index_values: list[int] | None
    index_defaulted: bool
    field: str
    modelname: str | None = None
    model_id: int | None = None
    metadata_path: Path | None = None
    metadata_source: str = "unresolved"
    effective_metadata_path: Path | None = None
    pristine_metadata_sha256: str | None = None
    effective_metadata_sha256: str | None = None
    override_selected: bool = False
    override_used: bool = False
    override_key: tuple[str, int | None] | None = None
    available_count: int | None = None
    violations: list[str] = field(default_factory=list)
    unresolved_reasons: list[str] = field(default_factory=list)
    expected_pristine_defect: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "source": self.source,
            "line": self.line,
            "kind": self.kind,
            "actor_expression": self.actor_expression,
            "actor": self.actor,
            "index_expression": self.index_expression,
            "index_values": self.index_values,
            "index_defaulted": self.index_defaulted,
            "field": self.field,
            "metadata_path": (
                str(self.metadata_path.resolve()) if self.metadata_path is not None else None
            ),
            "pristine_metadata_path": (
                str(self.metadata_path.resolve()) if self.metadata_path is not None else None
            ),
            "pristine_metadata_sha256": self.pristine_metadata_sha256,
            "effective_metadata_source": self.metadata_source,
            "effective_metadata_path": (
                str(self.effective_metadata_path.resolve())
                if self.effective_metadata_path is not None
                else None
            ),
            "effective_metadata_sha256": self.effective_metadata_sha256,
            "override_selected": self.override_selected,
            "override_used": self.override_used,
            "override_key": (
                _override_key_dict(*self.override_key) if self.override_key is not None else None
            ),
            "available_count": self.available_count,
            "violations": list(self.violations),
            "unresolved_reasons": list(self.unresolved_reasons),
            "expected_pristine_defect": self.expected_pristine_defect,
            "status": (
                "FAILED"
                if self.violations
                else "UNRESOLVED"
                if self.unresolved_reasons
                else "PASSED"
            ),
        }


@dataclass
class TaskReport:
    task: str
    source: str
    status: str = "PASSED"
    actors: list[ActorReference] = field(default_factory=list)
    interactions: list[InteractionReference] = field(default_factory=list)
    dynamic_items: list[DynamicItem] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    expected_pristine_defects: list[str] = field(default_factory=list)
    unexpected_violations: list[str] = field(default_factory=list)

    def as_dict(self, assets_root: Path) -> dict[str, Any]:
        return {
            "task": self.task,
            "source": self.source,
            "status": self.status,
            "actors": [row.as_dict(assets_root) for row in self.actors],
            "interactions": [row.as_dict() for row in self.interactions],
            "dynamic_items": [row.as_dict() for row in self.dynamic_items],
            "violations": list(self.violations),
            "expected_pristine_defects": list(self.expected_pristine_defects),
            "unexpected_violations": list(self.unexpected_violations),
        }


_UNKNOWN = object()


def _source_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - Python AST compatibility fallback
        return type(node).__name__


def _literal(node: ast.AST | None, constants: Mapping[str, object]) -> tuple[bool, object, str]:
    """Evaluate a deliberately tiny literal language.

    ``ast.literal_eval`` alone cannot resolve ``self.foo`` or a local constant,
    while evaluating arbitrary AST would execute benchmark code.  This helper
    accepts only scalar/list/tuple literals, unary numeric signs, and names
    previously proven to hold one of those values.
    """

    expression = _source_text(node) if node is not None else "<default>"
    if node is None:
        return False, _UNKNOWN, expression
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or isinstance(value, (str, int, float, bool)):
            return True, value, expression
        return False, _UNKNOWN, expression
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        ok, value, _ = _literal(node.operand, constants)
        if ok and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True, -value if isinstance(node.op, ast.USub) else value, expression
        return False, _UNKNOWN, expression
    if isinstance(node, (ast.List, ast.Tuple)):
        values: list[object] = []
        for element in node.elts:
            ok, value, _ = _literal(element, constants)
            if not ok:
                return False, _UNKNOWN, expression
            values.append(value)
        return True, values, expression
    if isinstance(node, ast.Name) and node.id in constants:
        value = constants[node.id]
        if value is _UNKNOWN:
            return False, _UNKNOWN, expression
        return True, copy.deepcopy(value), expression
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        key = f"{node.value.id}.{node.attr}"
        if key in constants:
            value = constants[key]
            if value is _UNKNOWN:
                return False, _UNKNOWN, expression
            return True, copy.deepcopy(value), expression
    return False, _UNKNOWN, expression


def _assign_key(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
        if target.value.id == "self":
            return f"self.{target.attr}"
        return f"{target.value.id}.{target.attr}"
    return None


def _collect_constants(tree: ast.AST) -> dict[str, object]:
    constants: dict[str, object] = {}
    assignments: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            assignments.append(node)
    assignments.sort(key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))
    for node in assignments:
        if isinstance(node, ast.Assign):
            targets: Iterable[ast.AST] = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
            value_node = node.value
        else:
            targets = (node.target,)
            value_node = None
        ok, value, _ = _literal(value_node, constants)
        for target in targets:
            key = _assign_key(target)
            if key is not None:
                constants[key] = copy.deepcopy(value) if ok else _UNKNOWN
    return constants


def _resolve_attr(node: ast.AST | None, aliases: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id if node.id else None)
    return None


def _resolve_model_call(
    call: ast.Call,
    constants: Mapping[str, object],
) -> tuple[str | None, int | None, bool, str, str, str | None]:
    """Return modelname/id and expressions for a create_actor call."""

    keywords = {item.arg: item.value for item in call.keywords if item.arg is not None}
    # create_actor(scene, pose, modelname, scale, convex, is_static, model_id)
    modelname_node = keywords.get("modelname")
    model_id_node = keywords.get("model_id")
    if modelname_node is None and len(call.args) >= 3:
        modelname_node = call.args[2]
    if model_id_node is None and len(call.args) >= 7:
        model_id_node = call.args[6]
    modelname_ok, modelname, modelname_expr = _literal(modelname_node, constants)
    model_id_defaulted = model_id_node is None
    if model_id_defaulted:
        model_id_ok, model_id, model_id_expr = True, DEFAULT_MODEL_ID, "<default:0>"
    else:
        model_id_ok, model_id, model_id_expr = _literal(model_id_node, constants)
    reasons: list[str] = []
    if not modelname_ok or not isinstance(modelname, str):
        reasons.append("modelname is not a static string")
        modelname = None
    if not model_id_ok or (model_id is not None and (not isinstance(model_id, int) or isinstance(model_id, bool))):
        reasons.append("model_id is not a static integer or None")
        model_id = None
        model_id_ok = False
    return (
        modelname,
        model_id,
        model_id_defaulted,
        modelname_expr,
        model_id_expr,
        "; ".join(reasons) if reasons else None,
    )


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _find_direct_actor_assignments(tree: ast.AST) -> dict[int, str]:
    result: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if _call_name(node.value) not in ACTOR_BUILDERS:
            continue
        for target in node.targets:
            attr = _resolve_attr(target, {})
            if attr is not None:
                result[id(node.value)] = attr
    return result


def _index_values(
    node: ast.AST | None,
    constants: Mapping[str, object],
    *,
    default: int = 0,
) -> tuple[list[int] | None, bool, str]:
    if node is None:
        return [default], True, f"<default:{default}>"
    ok, value, expression = _literal(node, constants)
    if not ok:
        return None, False, expression
    values = value if isinstance(value, list) else [value]
    if not values or any(
        not isinstance(item, int) or isinstance(item, bool) for item in values
    ):
        return None, False, expression
    return [int(item) for item in values], False, expression


def _metadata_file(assets_root: Path, modelname: str, model_id: int | None) -> Path:
    suffix = "" if model_id is None else str(model_id)
    return assets_root / "objects" / modelname / f"model_data{suffix}.json"


def _load_metadata(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _canonical_sha256(value: object) -> str | None:
    """Hash decoded JSON deterministically, returning ``None`` for errors."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha256(payload).hexdigest()


def _safe_file_sha256(path: Path) -> str | None:
    """Hash a metadata file without turning absent files into an exception."""

    try:
        if path.is_symlink() or not path.is_file():
            return None
        return _sha256(path)
    except OSError:
        return None


def _metadata_provenance(path: Path, metadata: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    return _safe_file_sha256(path), _canonical_sha256(metadata) if metadata is not None else None


def _normalise_override_key(raw_key: object) -> tuple[str, int | None]:
    """Validate the public ``(modelname, model_id)`` override key."""

    if not isinstance(raw_key, tuple) or len(raw_key) != 2:
        raise TaskAssetAuditError(
            "metadata_overrides keys must be (modelname, model_id) tuples"
        )
    modelname, model_id = raw_key
    if not isinstance(modelname, str) or not modelname:
        raise TaskAssetAuditError("metadata override modelname must be a non-empty string")
    if model_id is not None and (
        not isinstance(model_id, int) or isinstance(model_id, bool)
    ):
        raise TaskAssetAuditError("metadata override model_id must be an integer or None")
    return modelname, model_id


def _load_override_source(
    key: tuple[str, int | None],
    source: MetadataOverride,
) -> OverrideRecord:
    """Decode one override source and retain immutable provenance."""

    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        try:
            if path.is_symlink():
                raise OSError("symbolic links are not accepted")
            resolved = path.resolve(strict=True)
            if not resolved.is_file():
                raise OSError("not a regular file")
            source_hash = _sha256(resolved)
            metadata = _load_metadata(resolved)
        except (OSError, UnicodeError) as error:
            return OverrideRecord(
                modelname=key[0],
                model_id=key[1],
                source_type="file",
                source_path=path.resolve(),
                error=f"override file is unavailable: {path}: {error}",
            )
        record = OverrideRecord(
            modelname=key[0],
            model_id=key[1],
            source_type="file",
            source_path=resolved,
            source_sha256=source_hash,
            metadata=copy.deepcopy(metadata) if metadata is not None else None,
            metadata_sha256=_canonical_sha256(metadata) if metadata is not None else None,
        )
        if metadata is None:
            record.error = f"override file is missing or invalid JSON object: {resolved}"
        return record
    if isinstance(source, Mapping):
        metadata = copy.deepcopy(dict(source))
        record = OverrideRecord(
            modelname=key[0],
            model_id=key[1],
            source_type="mapping",
            metadata=metadata,
            metadata_sha256=_canonical_sha256(metadata),
        )
        if record.metadata_sha256 is None:
            record.error = "override mapping is not JSON serialisable"
        return record
    return OverrideRecord(
        modelname=key[0],
        model_id=key[1],
        source_type=type(source).__name__,
        error=(
            "override value must be a metadata path or a JSON-object mapping"
        ),
    )


def _prepare_overrides(
    metadata_overrides: Mapping[object, MetadataOverride] | None,
) -> dict[tuple[str, int | None], OverrideRecord]:
    if metadata_overrides is None:
        return {}
    if not isinstance(metadata_overrides, Mapping):
        raise TaskAssetAuditError("metadata_overrides must be a mapping")
    records: dict[tuple[str, int | None], OverrideRecord] = {}
    for raw_key, source in metadata_overrides.items():
        key = _normalise_override_key(raw_key)
        if key in records:
            raise TaskAssetAuditError(f"duplicate metadata override key: {key!r}")
        records[key] = _load_override_source(key, source)
    return records


def _bind_override(
    record: OverrideRecord,
    *,
    task: str,
    actor: str,
    line: int,
    interaction: bool = False,
    kind: str | None = None,
    index_values: Sequence[int] | None = None,
) -> None:
    usage = {
        "task": task,
        "actor": actor,
        "line": line,
    }
    if interaction:
        usage.update(
            {
                "kind": kind,
                "index_values": list(index_values) if index_values is not None else None,
            }
        )
        record.interaction_uses.append(usage)
    else:
        record.actor_uses.append(usage)


def _validate_override_against_pristine(
    record: OverrideRecord,
    pristine: Mapping[str, Any] | None,
) -> None:
    """Fail closed on a plate override that changes anything but contacts."""

    if record.error is not None:
        return
    if record.key != ("003_plate", 0):
        record.contract_status = "NOT_APPLICABLE"
        record.contract_reason = "no specialised compatibility contract for this key"
        return
    if pristine is None or record.metadata is None:
        record.error = "plate override cannot be compared with pristine metadata"
        record.contract_status = "FAILED"
        record.contract_reason = record.error
        return
    changed = sorted(
        key
        for key in set(pristine) | set(record.metadata)
        if pristine.get(key) != record.metadata.get(key)
    )
    before = _index_check(pristine, CONTACT_KEY, 2)
    after = _index_check(record.metadata, CONTACT_KEY, 2)
    if changed != [CONTACT_KEY]:
        record.error = (
            "plate override must change only contact_points_pose; "
            f"changed fields: {changed}"
        )
    elif pristine.get("scale") != record.metadata.get("scale"):
        record.error = "plate override changed pristine scale"
    elif before.get("status") != "FAILED":
        record.error = "pristine plate unexpectedly already accepts contact index 2"
    elif after.get("status") != "PASSED":
        record.error = "plate override does not make contact index 2 valid"
    if record.error is not None:
        record.contract_status = "FAILED"
        record.contract_reason = record.error
    else:
        record.contract_status = "PASSED"
        record.contract_reason = "contact_points_pose-only overlay; index 2 now valid"


def _prime_override_records(
    assets_root: Path,
    overrides: Mapping[tuple[str, int | None], OverrideRecord],
) -> None:
    """Bind every supplied override to its pristine benchmark metadata.

    Doing this before task extraction means a run receipt remains truthful even
    when the caller audits a task subset that does not happen to instantiate a
    supplied asset.
    """

    for key, record in overrides.items():
        pristine_path = _metadata_file(assets_root, key[0], key[1])
        pristine = _load_metadata(pristine_path)
        source_hash, metadata_hash = _metadata_provenance(pristine_path, pristine)
        record.bind_pristine(
            pristine_path,
            pristine,
            source_sha256=source_hash,
            metadata_sha256=metadata_hash,
        )
        _validate_override_against_pristine(record, pristine)


def _attach_interaction_metadata(
    interaction: InteractionReference,
    actor_ref: ActorReference,
    overrides: Mapping[tuple[str, int | None], OverrideRecord],
) -> None:
    interaction.modelname = actor_ref.modelname
    interaction.model_id = actor_ref.model_id
    interaction.metadata_path = actor_ref.metadata_path
    interaction.metadata_source = actor_ref.metadata_source
    interaction.effective_metadata_path = actor_ref.effective_metadata_path
    interaction.pristine_metadata_sha256 = actor_ref.pristine_metadata_sha256
    interaction.effective_metadata_sha256 = actor_ref.effective_metadata_sha256
    interaction.override_selected = actor_ref.override_selected
    interaction.override_used = bool(
        actor_ref.override_used and interaction.index_values is not None
    )
    interaction.override_key = actor_ref.override_key
    if interaction.override_used and actor_ref.override_key is not None:
        record = overrides[actor_ref.override_key]
        _bind_override(
            record,
            task=interaction.task,
            actor=actor_ref.attr,
            line=interaction.line,
            interaction=True,
            kind=interaction.kind,
            index_values=interaction.index_values,
        )


def _record_interaction_result(report: TaskReport, interaction: InteractionReference) -> None:
    """Classify a proven failure without hiding the pristine plate defect."""

    if not interaction.violations:
        return
    interaction.expected_pristine_defect = bool(
        interaction.task == "place_plate_and_cup"
        and interaction.kind == "grasp_actor"
        and interaction.modelname == "003_plate"
        and interaction.model_id == 0
        and interaction.override_used is False
        and interaction.index_values == [2]
        and interaction.violations
        == [f"{CONTACT_KEY} index 2 out of range (length 0)"]
    )
    report.violations.extend(interaction.violations)
    if interaction.expected_pristine_defect:
        report.expected_pristine_defects.extend(interaction.violations)
    else:
        report.unexpected_violations.extend(interaction.violations)


def _validate_indices(
    interaction: InteractionReference,
    metadata: Mapping[str, Any] | None,
) -> None:
    if interaction.index_values is None:
        return
    if metadata is None:
        interaction.violations.append("metadata file is missing or invalid")
        return
    raw = metadata.get(interaction.field)
    if not isinstance(raw, list):
        legacy_keys = (
            LEGACY_CONTACT_KEYS
            if interaction.field == CONTACT_KEY
            else LEGACY_FUNCTIONAL_KEYS
        )
        present_legacy = sorted(key for key in legacy_keys if key in metadata)
        if present_legacy:
            interaction.unresolved_reasons.append(
                f"metadata uses unsupported legacy field(s) {present_legacy}; "
                f"cannot prove {interaction.field} index convention"
            )
        else:
            interaction.violations.append(
                f"metadata field {interaction.field} is missing or not a list"
            )
        return
    interaction.available_count = len(raw)
    for index in interaction.index_values:
        if index < 0 or index >= len(raw):
            interaction.violations.append(
                f"{interaction.field} index {index} out of range (length {len(raw)})"
            )


def _extract_task(
    benchmark_root: Path,
    assets_root: Path,
    task: str,
    overrides: Mapping[tuple[str, int | None], OverrideRecord],
) -> tuple[TaskReport, list[ActorReference]]:
    source_path = benchmark_root / "envs" / f"{task}.py"
    report = TaskReport(task=task, source=str(source_path.resolve()))
    if not source_path.is_file():
        report.status = "BLOCKED"
        report.violations.append(f"task source is missing: {source_path}")
        return report, []
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, UnicodeError, SyntaxError) as error:
        report.status = "BLOCKED"
        report.violations.append(f"task source cannot be parsed: {type(error).__name__}: {error}")
        return report, []

    constants = _collect_constants(tree)
    aliases: dict[str, str] = {}
    # Prove simple aliases (``actor = self.foo``) before resolving interactions.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            key = node.targets[0]
            if isinstance(key, ast.Name):
                resolved = _resolve_attr(node.value, aliases)
                if resolved is not None:
                    aliases[key.id] = resolved
    direct_calls = _find_direct_actor_assignments(tree)
    actors_by_attr: dict[str, ActorReference] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) not in ACTOR_BUILDERS:
            continue
        attr = direct_calls.get(id(node))
        modelname, model_id, model_id_defaulted, modelname_expr, model_id_expr, dynamic_reason = _resolve_model_call(node, constants)
        if attr is None:
            report.dynamic_items.append(
                DynamicItem(
                    task,
                    str(source_path.resolve()),
                    getattr(node, "lineno", 0),
                    "actor_creation",
                    _source_text(node),
                    "create_actor result is not assigned directly to self.<attribute>",
                )
            )
            continue
        actor = ActorReference(
            attr=attr,
            source=str(source_path.resolve()),
            line=getattr(node, "lineno", 0),
            modelname=modelname,
            model_id=model_id,
            model_id_defaulted=model_id_defaulted,
            modelname_expression=modelname_expr,
            model_id_expression=model_id_expr,
        )
        if dynamic_reason is not None:
            report.dynamic_items.append(
                DynamicItem(
                    task,
                    str(source_path.resolve()),
                    getattr(node, "lineno", 0),
                    "actor_model_reference",
                    _source_text(node),
                    dynamic_reason,
                    attr,
                )
            )
        elif modelname is not None:
            actor.metadata_path = _metadata_file(assets_root, modelname, model_id)
            pristine = _load_metadata(actor.metadata_path)
            (
                actor.pristine_source_sha256,
                actor.pristine_metadata_sha256,
            ) = _metadata_provenance(actor.metadata_path, pristine)
            actor.metadata = copy.deepcopy(pristine) if pristine is not None else None
            actor.metadata_source = "pristine"
            actor.effective_metadata_path = actor.metadata_path
            actor.effective_source_sha256 = actor.pristine_source_sha256
            actor.effective_metadata_sha256 = actor.pristine_metadata_sha256
            override_key = (modelname, model_id)
            override = overrides.get(override_key)
            if override is not None:
                actor.override_selected = True
                actor.override_key = override_key
                override.bind_pristine(
                    actor.metadata_path,
                    pristine,
                    source_sha256=actor.pristine_source_sha256,
                    metadata_sha256=actor.pristine_metadata_sha256,
                )
                if override.error is None and override.metadata is not None:
                    actor.metadata = copy.deepcopy(override.metadata)
                    actor.metadata_source = f"override:{override.source_type}"
                    actor.effective_metadata_path = override.source_path
                    actor.effective_source_sha256 = override.source_sha256
                    actor.effective_metadata_sha256 = override.metadata_sha256
                    actor.override_used = True
                    _bind_override(
                        override,
                        task=task,
                        actor=attr,
                        line=actor.line,
                    )
                else:
                    violation = (
                        f"{attr}: metadata override {override_key!r} invalid: "
                        f"{override.error or 'empty metadata'}"
                    )
                    report.violations.append(violation)
                    report.unexpected_violations.append(violation)
            if pristine is None:
                violation = f"{attr}: metadata file missing or invalid: {actor.metadata_path}"
                report.violations.append(violation)
                report.unexpected_violations.append(violation)
        actors_by_attr[attr] = actor
    report.actors = list(sorted(actors_by_attr.values(), key=lambda item: (item.line, item.attr)))

    # Scan every interaction call.  This intentionally includes calls nested
    # under ``move(...)`` and branch expressions, matching the source-level
    # contract rather than relying on execution coverage.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) not in INTERACTION_METHODS:
            continue
        kind = _call_name(node) or "interaction"
        keywords = {item.arg: item.value for item in node.keywords if item.arg is not None}
        actor_node = keywords.get("actor")
        if actor_node is None and node.args:
            actor_node = node.args[0]
        actor_expr = _source_text(actor_node) if actor_node is not None else "<missing>"
        actor_attr = _resolve_attr(actor_node, aliases)
        if actor_attr is None:
            actor_unresolved_reason = (
                "actor expression is not a direct self.<attribute> or proven alias"
            )
            report.dynamic_items.append(
                DynamicItem(
                    task,
                    str(source_path.resolve()),
                    getattr(node, "lineno", 0),
                    "interaction_actor",
                    actor_expr,
                    actor_unresolved_reason,
                )
            )
        else:
            actor_unresolved_reason = None
        field_name = CONTACT_KEY if kind == "grasp_actor" else FUNCTIONAL_KEY
        index_key = "contact_point_id" if kind == "grasp_actor" else "functional_point_id"
        values, defaulted, index_expr = _index_values(keywords.get(index_key), constants)
        if values is None:
            index_unresolved_reason = (
                f"{index_key} is not a static integer/list of integers"
            )
            report.dynamic_items.append(
                DynamicItem(
                    task,
                    str(source_path.resolve()),
                    getattr(node, "lineno", 0),
                    "interaction_index",
                    index_expr,
                    index_unresolved_reason,
                    actor_attr,
                )
            )
        else:
            index_unresolved_reason = None
        actor_ref = actors_by_attr.get(actor_attr or "")
        interaction = InteractionReference(
            task=task,
            source=str(source_path.resolve()),
            line=getattr(node, "lineno", 0),
            kind=kind,
            actor_expression=actor_expr,
            actor=actor_attr,
            index_expression=index_expr,
            index_values=values,
            index_defaulted=defaulted,
            field=field_name,
        )
        if actor_unresolved_reason is not None:
            interaction.unresolved_reasons.append(actor_unresolved_reason)
        if index_unresolved_reason is not None:
            interaction.unresolved_reasons.append(index_unresolved_reason)
        if actor_attr is not None and actor_ref is None:
            interaction.unresolved_reasons.append(
                "actor is not a directly-created create_actor asset"
            )
            report.dynamic_items.append(
                DynamicItem(
                    task,
                    str(source_path.resolve()),
                    getattr(node, "lineno", 0),
                    "interaction_asset_mapping",
                    actor_expr,
                    "actor is not a directly-created create_actor asset",
                    actor_attr,
                )
            )
        if actor_ref is not None and actor_ref.metadata_path is None:
            interaction.unresolved_reasons.append(
                "actor modelname/model_id is dynamic; metadata index cannot be proven statically"
            )
            report.dynamic_items.append(
                DynamicItem(
                    task,
                    str(source_path.resolve()),
                    getattr(node, "lineno", 0),
                    "interaction_asset_metadata",
                    actor_expr,
                    "actor modelname/model_id is dynamic; metadata index cannot be proven statically",
                    actor_attr,
                )
            )
        elif actor_ref is not None:
            _attach_interaction_metadata(interaction, actor_ref, overrides)
            _validate_indices(interaction, actor_ref.metadata)
            for reason in interaction.unresolved_reasons:
                report.dynamic_items.append(
                    DynamicItem(
                        task,
                        str(source_path.resolve()),
                        getattr(node, "lineno", 0),
                        "unsupported_metadata_field",
                        actor_expr,
                        reason,
                        actor_attr,
                    )
                )
        report.interactions.append(interaction)
        _record_interaction_result(report, interaction)

    # get_functional_point is an asset/config access even when it is not nested
    # in place_actor's target_pose.  Validate it separately.
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_functional_point"
        ):
            continue
        actor_expr = _source_text(node.func.value)
        actor_attr = _resolve_attr(node.func.value, aliases)
        actor_unresolved_reason = None
        if actor_attr is None:
            actor_unresolved_reason = (
                "actor expression is not a direct self.<attribute> or proven alias"
            )
        values, defaulted, index_expr = _index_values(
            node.args[0] if node.args else None,
            constants,
        )
        if values is None:
            index_unresolved_reason = "functional point index is not static"
            report.dynamic_items.append(
                DynamicItem(
                    task,
                    str(source_path.resolve()),
                    getattr(node, "lineno", 0),
                    "functional_point_index",
                    index_expr,
                    "functional point index is not static",
                    actor_attr,
                )
            )
        else:
            index_unresolved_reason = None
        actor_ref = actors_by_attr.get(actor_attr or "")
        interaction = InteractionReference(
            task=task,
            source=str(source_path.resolve()),
            line=getattr(node, "lineno", 0),
            kind="get_functional_point",
            actor_expression=actor_expr,
            actor=actor_attr,
            index_expression=index_expr,
            index_values=values,
            index_defaulted=defaulted,
            field=FUNCTIONAL_KEY,
        )
        if actor_unresolved_reason is not None:
            interaction.unresolved_reasons.append(actor_unresolved_reason)
        if index_unresolved_reason is not None:
            interaction.unresolved_reasons.append(index_unresolved_reason)
        if actor_attr is not None and actor_ref is None:
            interaction.unresolved_reasons.append(
                "actor is not a directly-created create_actor asset"
            )
            report.dynamic_items.append(
                DynamicItem(
                    task,
                    str(source_path.resolve()),
                    getattr(node, "lineno", 0),
                    "functional_point_asset_mapping",
                    actor_expr,
                    "actor is not a directly-created create_actor asset",
                    actor_attr,
                )
            )
        if actor_ref is not None and actor_ref.metadata_path is None:
            interaction.unresolved_reasons.append(
                "actor modelname/model_id is dynamic; metadata index cannot be proven statically"
            )
            report.dynamic_items.append(
                DynamicItem(
                    task,
                    str(source_path.resolve()),
                    getattr(node, "lineno", 0),
                    "functional_point_asset_metadata",
                    actor_expr,
                    "actor modelname/model_id is dynamic; metadata index cannot be proven statically",
                    actor_attr,
                )
            )
        elif actor_ref is not None:
            _attach_interaction_metadata(interaction, actor_ref, overrides)
            _validate_indices(interaction, actor_ref.metadata)
            for reason in interaction.unresolved_reasons:
                report.dynamic_items.append(
                    DynamicItem(
                        task,
                        str(source_path.resolve()),
                        getattr(node, "lineno", 0),
                        "unsupported_metadata_field",
                        actor_expr,
                        reason,
                        actor_attr,
                    )
                )
        report.interactions.append(interaction)
        _record_interaction_result(report, interaction)

    # A source-level parseable contract violation is a hard failure.  Dynamic
    # references remain visible but do not alter status.
    if report.violations:
        report.status = "FAILED"
    return report, list(actors_by_attr.values())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mesh_hash_inventory(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*.glb")):
        if path.is_file() and not path.is_symlink():
            rows.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return rows


def _read_mapping(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _index_check(mapping: Mapping[str, Any] | None, field_name: str, index: int) -> dict[str, Any]:
    if mapping is None:
        return {
            "status": "FAILED",
            "field": field_name,
            "index": index,
            "reason": "metadata is missing or invalid",
        }
    value = mapping.get(field_name)
    if not isinstance(value, list):
        return {
            "status": "FAILED",
            "field": field_name,
            "index": index,
            "reason": "field is missing or not a list",
            "available_count": None,
        }
    if index < 0 or index >= len(value):
        return {
            "status": "FAILED",
            "field": field_name,
            "index": index,
            "available_count": len(value),
            "reason": f"index {index} out of range (length {len(value)})",
        }
    return {
        "status": "PASSED",
        "field": field_name,
        "index": index,
        "available_count": len(value),
    }


def _overlay_check(assets_root: Path) -> dict[str, Any]:
    small_dir = assets_root / "objects" / "003_plate"
    large_dir = assets_root / "objects" / "003_plate_large"
    small_path = small_dir / "model_data0.json"
    large_path = large_dir / "model_data0.json"
    small = _read_mapping(small_path)
    large = _read_mapping(large_path)
    before = _index_check(small, CONTACT_KEY, 2)
    result: dict[str, Any] = {
        "task": "place_plate_and_cup",
        "actor": "plate",
        "modelname": "003_plate",
        "model_id": 0,
        "contact_point_id": 2,
        "small_metadata": str(small_path.resolve()),
        "large_metadata": str(large_path.resolve()),
        "pristine_metadata_sha256": _safe_file_sha256(small_path),
        "reference_metadata_sha256": _safe_file_sha256(large_path),
        "pristine_metadata_canonical_sha256": _canonical_sha256(small),
        "reference_metadata_canonical_sha256": _canonical_sha256(large),
        "before": before,
        "mesh_hashes_equal": False,
        "small_scale_before": small.get("scale") if small is not None else None,
        "small_scale_after": None,
        "copied_fields": [CONTACT_KEY],
        "mutation_scope": "in_memory_only",
        "benchmark_files_written": False,
    }
    small_meshes = _mesh_hash_inventory(small_dir)
    large_meshes = _mesh_hash_inventory(large_dir)
    small_by_path = {row["relative_path"]: row for row in small_meshes}
    large_by_path = {row["relative_path"]: row for row in large_meshes}
    result["small_meshes"] = small_meshes
    result["large_meshes"] = large_meshes
    result["mesh_hashes_equal"] = bool(
        small_by_path
        and set(small_by_path) == set(large_by_path)
        and all(small_by_path[key]["sha256"] == large_by_path[key]["sha256"] for key in small_by_path)
    )
    if small is None or large is None:
        result["after"] = _index_check(None, CONTACT_KEY, 2)
        result["status"] = "FAILED"
        result["reason"] = "small or large metadata is unavailable"
        return result
    overlaid = copy.deepcopy(small)
    source_contacts = large.get(CONTACT_KEY)
    if not isinstance(source_contacts, list):
        result["after"] = _index_check(None, CONTACT_KEY, 2)
        result["status"] = "FAILED"
        result["reason"] = "large contact_points_pose is missing or not a list"
        return result
    overlaid[CONTACT_KEY] = copy.deepcopy(source_contacts)
    result["after"] = _index_check(overlaid, CONTACT_KEY, 2)
    result["small_scale_after"] = overlaid.get("scale")
    result["source_contact_points_pose_count"] = len(source_contacts)
    result["source_contact_points_pose_sha256"] = hashlib.sha256(
        json.dumps(source_contacts, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    result["overlay_metadata_canonical_sha256"] = _canonical_sha256(overlaid)
    result["only_contact_field_changed"] = all(
        overlaid.get(key) == small.get(key) for key in set(small) | set(overlaid) if key != CONTACT_KEY
    )
    result["target_scale_preserved"] = overlaid.get("scale") == small.get("scale")
    # The required invariant is an expected before-failure followed by an
    # after-success.  An already-mutated checkout is surfaced as unexpected,
    # never silently reported as a pristine proof.
    before_expected = before.get("status") == "FAILED"
    after_ok = result["after"].get("status") == "PASSED"
    result["before_expected_failure"] = before_expected
    result["status"] = (
        "PASSED"
        if before_expected
        and after_ok
        and result["mesh_hashes_equal"]
        and result["only_contact_field_changed"]
        and result["target_scale_preserved"]
        else "FAILED"
    )
    if not before_expected:
        result["reason"] = "pristine small metadata unexpectedly already accepts contact index 2"
    elif not after_ok:
        result["reason"] = "in-memory contact overlay did not make contact index 2 valid"
    elif not result["mesh_hashes_equal"]:
        result["reason"] = "small/large mesh inventories or hashes differ"
    elif not result["only_contact_field_changed"] or not result["target_scale_preserved"]:
        result["reason"] = "overlay changed a field other than contact_points_pose"
    return result


def audit_task_assets(
    benchmark_root: str | Path,
    assets_root: str | Path,
    *,
    tasks: Sequence[str] | None = None,
    metadata_overrides: Mapping[object, MetadataOverride] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serialisable static task/asset audit report.

    ``metadata_overrides`` is keyed by ``(modelname, model_id)``.  Values may
    be a JSON metadata path or an in-memory mapping.  Overrides are applied
    only to the effective metadata used for static index checks; the pristine
    benchmark record is always retained and hashed in the report.
    """

    benchmark = Path(benchmark_root).expanduser().resolve()
    assets = Path(assets_root).expanduser().resolve()
    selected = tuple(DEFAULT_TASKS if tasks is None else (str(task) for task in tasks))
    if not selected or len(selected) != len(set(selected)):
        raise TaskAssetAuditError("task list must be non-empty and unique")
    overrides = _prepare_overrides(metadata_overrides)
    _prime_override_records(assets, overrides)
    task_reports: list[TaskReport] = []
    dynamic: list[DynamicItem] = []
    violations: list[str] = []
    for task in selected:
        report, _ = _extract_task(benchmark, assets, task, overrides)
        task_reports.append(report)
        dynamic.extend(report.dynamic_items)
        violations.extend(f"{task}: {value}" for value in report.violations)
    overlay = _overlay_check(assets)
    overlay_violation = None
    if overlay.get("status") != "PASSED":
        overlay_violation = (
            "place_plate_and_cup overlay contract: "
            f"{overlay.get('reason', 'failed')}"
        )
        violations.append(overlay_violation)
    # The known upstream plate mismatch is deliberately visible, but does not
    # make a pristine audit look like an unknown failure.  Any other violation
    # remains fail-closed.  A caller that requires a clean effective contract
    # should supply the recorded plate override.
    expected_defect_count = sum(
        len(row.expected_pristine_defects) for row in task_reports
    )
    unexpected_violation_count = sum(
        len(row.unexpected_violations) for row in task_reports
    )
    # A pristine run may contain only the explicitly documented plate defect.
    # Everything else (including a missing/blocked task) is a hard failure.
    # Keep this classification based on structured task reports rather than
    # string-prefix heuristics: an overlay error is itself prefixed with the
    # task name and could otherwise be mistaken for an expected defect.
    task_has_unexpected_failure = any(
        row.status == "BLOCKED"
        or any(item not in row.expected_pristine_defects for item in row.violations)
        for row in task_reports
    )
    if (
        overlay_violation is None
        and not task_has_unexpected_failure
        and unexpected_violation_count == 0
        and expected_defect_count > 0
    ):
        status = "PASSED_WITH_EXPECTED_PRISTINE_DEFECT"
    elif (
        overlay_violation is None
        and not task_has_unexpected_failure
        and unexpected_violation_count == 0
        and not violations
    ):
        status = "PASSED"
    else:
        status = "FAILED"
    # Invalid override sources are hard failures; an unused but valid override
    # is retained as a warning/provenance record because callers may audit a
    # task subset and intentionally provide a superset of overrides.
    for key, record in overrides.items():
        if record.error is not None:
            value = f"metadata override {key!r}: {record.error}"
            if value not in violations:
                violations.append(value)
            status = "FAILED"
    actor_reference_count = sum(len(row.actors) for row in task_reports)
    interaction_reference_count = sum(
        len(row.interactions) for row in task_reports
    )

    def _relative_source(value: str) -> str:
        try:
            return Path(value).resolve().relative_to(benchmark).as_posix()
        except (OSError, ValueError):
            # Keep an out-of-tree source visible and hash-stable; callers may
            # reject it as a provenance error rather than silently dropping it.
            return str(Path(value).resolve())

    # Dynamic references are intentionally not guessed.  Their exact ordered
    # inventory is nevertheless part of the pinned source contract, so a
    # future task edit cannot silently expand the set of unverified IDs while
    # retaining a nominal ``PASSED`` status.
    dynamic_inventory = []
    for item in dynamic:
        row = item.as_dict()
        row["source"] = _relative_source(str(row["source"]))
        dynamic_inventory.append(row)
    unresolved_interactions = []
    for task_report in task_reports:
        for interaction in task_report.interactions:
            if interaction.unresolved_reasons:
                unresolved_interactions.append(
                    {
                        "task": interaction.task,
                        "source": _relative_source(interaction.source),
                        "line": interaction.line,
                        "kind": interaction.kind,
                        "actor_expression": interaction.actor_expression,
                        "actor": interaction.actor,
                        "index_expression": interaction.index_expression,
                        "index_values": interaction.index_values,
                        "index_defaulted": interaction.index_defaulted,
                        "field": interaction.field,
                        "unresolved_reasons": list(interaction.unresolved_reasons),
                    }
                )
    dynamic_inventory_sha256 = _canonical_sha256(dynamic_inventory)
    unresolved_interaction_inventory_sha256 = _canonical_sha256(
        unresolved_interactions
    )

    return {
        "schema": SCHEMA,
        "status": status,
        "benchmark_root": str(benchmark),
        "assets_root": str(assets),
        "tasks": list(selected),
        "task_count": len(selected),
        "actor_reference_count": actor_reference_count,
        "interaction_reference_count": interaction_reference_count,
        "references_checked": {
            "tasks": len(selected),
            "actors": actor_reference_count,
            "interactions": interaction_reference_count,
        },
        "task_reports": [row.as_dict(assets) for row in task_reports],
        "dynamic_items": [row.as_dict() for row in dynamic],
        "dynamic_item_count": len(dynamic),
        "dynamic_inventory_sha256": dynamic_inventory_sha256,
        "unresolved_interaction_count": len(unresolved_interactions),
        "unresolved_interaction_inventory_sha256": (
            unresolved_interaction_inventory_sha256
        ),
        "violations": violations,
        "expected_pristine_defect_count": expected_defect_count,
        "unexpected_violation_count": unexpected_violation_count,
        "expected_pristine_defects": [
            {
                "task": row.task,
                "items": list(row.expected_pristine_defects),
            }
            for row in task_reports
            if row.expected_pristine_defects
        ],
        "metadata_overrides": [
            record.as_dict()
            for _, record in sorted(
                overrides.items(), key=lambda item: _override_sort_key(item[0])
            )
        ],
        "metadata_override_count": len(overrides),
        "metadata_override_keys": [
            _override_key_dict(*key) for key in sorted(overrides, key=_override_sort_key)
        ],
        "overlay_check": overlay,
        "read_only_benchmark": True,
        "benchmark_files_written": False,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument(
        "--metadata-override",
        action="append",
        nargs=3,
        metavar=("MODELNAME", "MODEL_ID", "JSON_PATH"),
        default=[],
        help=(
            "effective metadata override; repeat as needed. MODEL_ID may be "
            "'none' for model_data.json"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.tasks is None:
        # Keep the CLI lazy so importing this module remains independent of the
        # heavyweight CARE package and simulator.
        config_path = args.benchmark_root.parent / "before-we-act" / "deployment" / "bicoord_care" / "config.py"
        if config_path.is_file():
            tree = ast.parse(config_path.read_text(encoding="utf-8"))
            tasks: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "TASKS" for target in node.targets
                ):
                    try:
                        tasks = [str(value) for value in ast.literal_eval(node.value)]
                    except Exception:
                        tasks = []
                    break
            if not tasks:
                raise TaskAssetAuditError("could not recover TASKS from CARE config; pass --tasks")
        else:
            raise TaskAssetAuditError("pass --tasks when CARE config is outside the benchmark parent")
    else:
        tasks = list(args.tasks)
    overrides: dict[tuple[str, int | None], Path] = {}
    for modelname, raw_model_id, raw_path in args.metadata_override:
        try:
            model_id = (
                None if raw_model_id.strip().lower() == "none" else int(raw_model_id)
            )
        except ValueError as error:
            raise TaskAssetAuditError(
                f"invalid metadata override model id: {raw_model_id!r}"
            ) from error
        key = (modelname, model_id)
        if key in overrides:
            raise TaskAssetAuditError(f"duplicate metadata override key: {key!r}")
        overrides[key] = Path(raw_path)
    report = audit_task_assets(
        args.benchmark_root,
        args.assets_root,
        tasks=tasks,
        metadata_overrides=overrides,
    )
    if args.output:
        _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASSED" else 1


__all__ = [
    "SCHEMA",
    "TaskAssetAuditError",
    "audit_task_assets",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
