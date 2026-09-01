"""Exact BiCoord simulator snapshot/restore contract.

BiCoord's upstream ``Base_Task`` deliberately exposes no state serialization
API.  CARE's counterfactual branches cannot be collected from a replayed
observation or from a copied action cache: every sibling must start from the
same physics state and random stream.  This module is therefore intentionally
strict.  It can validate a benchmark-owned adapter which implements
``get_state_dict``/``set_state_dict``; otherwise it raises a descriptive
error before any branch is labelled as physical.

The serializer below is also a reference implementation for the adapter.  It
captures SAPIEN scene poses, rigid-body velocities, articulation qpos/qvel/
qacc/root state, drive targets, the PhysX CPU-system state blob (including
solver/contact state not exposed by entity accessors), controller/task scalar
state, wrapper clocks, and Python/NumPy/Torch RNG.  A caller must still run
:func:`restore_probe` twice and require the <=1e-6 result before using formal
CARE data.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import hashlib
import random
from typing import Any, Callable, Mapping

import numpy as np
import torch


SNAPSHOT_SCHEMA = "before-we-act.bicoord.exact-snapshot/2"
SNAPSHOT_TOLERANCE = 1e-6


class SnapshotCapabilityError(RuntimeError):
    """Raised when an environment cannot provide exact branch restoration."""


def _base_env(env: Any) -> Any:
    current = env
    # Gym wrappers are intentionally unwrapped one layer at a time; do not
    # silently pick a nested object which is not the simulator owner.
    seen: set[int] = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return getattr(current, "unwrapped", current)


def _numeric(value: Any) -> Any:
    """Copy a value that is safe to put in a state dictionary.

    ``None`` is used as an omission marker.  Physics handles and planners are
    not copied; their numerical state is captured explicitly below.
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return deepcopy(value)
    # Episode-local RNG objects are common in Gym/SAPIEN wrappers.  Treat
    # them as numerical controller state instead of either dropping them or
    # deep-copying an opaque Python object.  The tagged representation is
    # also understood by ``_restore_numeric`` and ``state_sha256``.
    if isinstance(value, np.random.RandomState):
        return {
            "__bicoord_rng__": "numpy_random_state",
            "state": _numeric(value.get_state()),
        }
    if isinstance(value, np.random.Generator):
        return {
            "__bicoord_rng__": "numpy_generator",
            "bit_generator": type(value.bit_generator).__name__,
            "state": _numeric(value.bit_generator.state),
        }
    if isinstance(value, random.Random):
        return {
            "__bicoord_rng__": "python_random",
            "state": _numeric(value.getstate()),
        }
    if isinstance(value, torch.Generator):
        return {
            "__bicoord_rng__": "torch_generator",
            "device": str(value.device),
            "state": value.get_state().detach().cpu().clone(),
        }
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            item = _numeric(child)
            if item is not None or child is None:
                result[deepcopy(key)] = item
            else:
                return None
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for child in value:
            item = _numeric(child)
            if item is None and child is not None:
                return None
            result.append(item)
        return tuple(result) if isinstance(value, tuple) else result
    # SAPIEN handles have numerical accessors and are deliberately omitted.
    if any(hasattr(value, name) for name in ("get_pose", "get_qpos", "get_scene")):
        return None
    return None


def _restore_numeric(value: Any) -> Any:
    if isinstance(value, Mapping):
        rng_kind = value.get("__bicoord_rng__")
        if rng_kind == "numpy_random_state":
            result = np.random.RandomState()
            result.set_state(_restore_numeric(value["state"]))
            return result
        if rng_kind == "numpy_generator":
            name = str(value.get("bit_generator", ""))
            bit_generator_type = getattr(np.random, name, None)
            if bit_generator_type is None:
                raise SnapshotCapabilityError(
                    f"snapshot requires unavailable NumPy bit generator {name!r}"
                )
            bit_generator = bit_generator_type()
            bit_generator.state = _restore_numeric(value["state"])
            return np.random.Generator(bit_generator)
        if rng_kind == "python_random":
            result = random.Random()
            result.setstate(_restore_numeric(value["state"]))
            return result
        if rng_kind == "torch_generator":
            device = str(value.get("device", "cpu"))
            if device != "cpu" and not torch.cuda.is_available():
                raise SnapshotCapabilityError(
                    f"snapshot requires unavailable Torch generator device {device}"
                )
            result = torch.Generator(device=device)
            result.set_state(torch.as_tensor(value["state"]).detach().cpu())
            return result
        return {deepcopy(k): _restore_numeric(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_restore_numeric(v) for v in value)
    if isinstance(value, list):
        return [_restore_numeric(v) for v in value]
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if torch.is_tensor(value):
        return value.clone()
    return deepcopy(value)


def _pose(value: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(value.p, dtype=np.float64).copy(), np.asarray(value.q, dtype=np.float64).copy()


def _sapien_pose(value: tuple[np.ndarray, np.ndarray]) -> Any:
    """Construct a SAPIEN pose without making the fake-env tests depend on SAPIEN."""

    try:
        import sapien

        pose_type = getattr(sapien, "Pose", None)
        if pose_type is None:
            from sapien import core as sapien_core

            pose_type = sapien_core.Pose
        return pose_type(value[0], value[1])
    except ImportError:
        # Pure-Python doubles accept the canonical (p, q) tuple.  Do not catch
        # setter errors below: a real SAPIEN restore failure must fail closed.
        # Make the fallback explicitly owning.  A test double (and a few
        # lightweight adapters) may retain the tuple arrays by reference;
        # retaining the arrays from the snapshot would let a subsequent
        # rollout mutate the supposedly immutable snapshot and make the
        # second restore non-deterministic.
        return (
            np.array(value[0], dtype=np.float64, copy=True),
            np.array(value[1], dtype=np.float64, copy=True),
        )


def _set_pose_value(setter: Callable[[Any], Any], value: tuple[np.ndarray, np.ndarray]) -> None:
    """Call a pose setter across real SAPIEN and lightweight test doubles.

    The production adapter receives a :class:`sapien.Pose`, while the small
    Python doubles used by the contract tests intentionally accept the
    canonical ``(p, q)`` tuple.  Importing SAPIEN is not enough to select the
    representation: the remote runtime has SAPIEN installed while the test
    double is still active.  Try the native object first, then use an owning
    tuple only when the setter explicitly rejects that type.  Any second
    failure is propagated so a real simulator restore error remains
    fail-closed.
    """

    native = _sapien_pose(value)
    try:
        setter(native)
    except TypeError as first_error:
        fallback = (
            np.array(value[0], dtype=np.float64, copy=True),
            np.array(value[1], dtype=np.float64, copy=True),
        )
        try:
            setter(fallback)
        except Exception:
            # Preserve the original type error, which identifies the
            # unsupported native representation at the adapter boundary.
            raise first_error


def _set_pose(obj: Any, value: tuple[np.ndarray, np.ndarray]) -> None:
    setter = getattr(obj, "set_pose", None)
    if not callable(setter):
        raise SnapshotCapabilityError("actor lacks set_pose")
    _set_pose_value(setter, value)


def _set_root_pose(obj: Any, value: tuple[np.ndarray, np.ndarray]) -> None:
    setter = getattr(obj, "set_root_pose", None)
    if not callable(setter):
        raise SnapshotCapabilityError("articulation lacks set_root_pose")
    _set_pose_value(setter, value)


def _get_attr(obj: Any, *names: str) -> Any:
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                continue
        if fn is not None:
            return fn
    return None


def _set_attr(obj: Any, names: tuple[str, ...], value: Any) -> bool:
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            fn(value)
            return True
        if hasattr(obj, name):
            setattr(obj, name, value)
            return True
    return False


def _object_key(obj: Any, index: int) -> str:
    name = _get_attr(obj, "get_name", "name")
    return f"{index}:{name or ''}"


def _component(entity: Any, predicate: str) -> Any:
    for component in getattr(entity, "get_components", lambda: ())():
        if predicate in type(component).__name__.lower():
            return component
    return None


def _physx_system(scene: Any) -> Any:
    """Resolve the scene-owned PhysX system across supported SAPIEN 3 APIs."""

    getter = getattr(scene, "get_physx_system", None)
    system = getter() if callable(getter) else getattr(scene, "physx_system", None)
    if system is None:
        raise SnapshotCapabilityError("scene lacks PhysX system access")
    if not callable(getattr(system, "pack", None)):
        raise SnapshotCapabilityError("PhysX system lacks pack")
    if not callable(getattr(system, "unpack", None)):
        raise SnapshotCapabilityError("PhysX system lacks unpack")
    return system


def _is_dynamic_component(component: Any) -> bool:
    if component is None:
        return False
    name = type(component).__name__.lower()
    if "static" in name:
        return False
    if "dynamic" in name:
        return True
    # Test doubles and compatible SAPIEN builds may use a less specific type
    # name.  A writable velocity API is sufficient evidence of dynamics.
    return any(
        callable(getattr(component, accessor, None))
        for accessor in ("set_linear_velocity", "set_angular_velocity")
    )


def _actor_state(actor: Any) -> dict[str, Any]:
    pose = _get_attr(actor, "get_pose", "pose")
    if pose is None:
        raise SnapshotCapabilityError("actor lacks get_pose")
    row: dict[str, Any] = {"pose": _pose(pose)}
    # Entity-level velocity accessors are not present in SAPIEN 3; the PhysX
    # rigid component owns them.
    component = _component(actor, "rigid") or _component(actor, "physx")
    if component is not None:
        linear = _get_attr(component, "get_linear_velocity", "linear_velocity")
        angular = _get_attr(component, "get_angular_velocity", "angular_velocity")
        if linear is not None:
            row["linear_velocity"] = np.asarray(linear, dtype=np.float64).copy()
        if angular is not None:
            row["angular_velocity"] = np.asarray(angular, dtype=np.float64).copy()
        # SAPIEN raises ``RuntimeError: ... actor is not kinematic`` when
        # ``get_kinematic_target`` is queried on an ordinary dynamic actor.
        # Probe the mode first and only ask for a target when it is meaningful.
        # This is not treated as a blanket optional accessor: an unknown
        # runtime failure remains fatal so a partially captured physics state
        # can never be accepted as an exact CARE branch snapshot.
        kinematic = _get_attr(component, "get_kinematic", "kinematic")
        if kinematic is not None:
            row["kinematic"] = bool(kinematic)
        if kinematic is None or bool(kinematic):
            getter = getattr(component, "get_kinematic_target", None)
            if callable(getter):
                try:
                    target = getter()
                except RuntimeError as exc:
                    message = str(exc).lower()
                    if "actor is not kinematic" in message:
                        # Dynamic bodies have no target to serialize.
                        target = None
                    else:
                        raise
                if target is not None:
                    row["kinematic_target"] = _pose(target)
    if _is_dynamic_component(component) and (
        "linear_velocity" not in row or "angular_velocity" not in row
    ):
        # A kinematic rigid body can still carry a velocity/target that affects
        # its next pose.  Every dynamic component therefore needs both fields.
        raise SnapshotCapabilityError("dynamic actor lacks velocity accessors")
    return row


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
        "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _wrapper_chain(env: Any) -> list[Any]:
    result: list[Any] = []
    current = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        result.append(current)
        child = getattr(current, "env", None)
        if child is None:
            break
        current = child
    return result


def _wrapper_state(env: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for depth, wrapper in enumerate(_wrapper_chain(env)):
        values: dict[str, Any] = {}
        for key in (
            "_elapsed_steps",
            "elapsed_steps",
            "take_action_cnt",
            "left_cnt",
            "right_cnt",
        ):
            if hasattr(wrapper, key):
                value = getattr(wrapper, key)
                item = _numeric(value)
                if item is None and value is not None:
                    raise SnapshotCapabilityError(
                        f"wrapper counter is not serializable: depth={depth} {key}"
                    )
                values[key] = item
        rows.append(
            {
                "depth": depth,
                "type": f"{type(wrapper).__module__}.{type(wrapper).__qualname__}",
                "values": values,
            }
        )
    return rows


def _articulation_state(articulation: Any) -> dict[str, Any]:
    required = ("get_qpos", "set_qpos", "get_qvel", "set_qvel", "get_active_joints")
    if any(not callable(getattr(articulation, name, None)) for name in required):
        raise SnapshotCapabilityError("articulation lacks qpos/qvel/active-joint API")
    row: dict[str, Any] = {
        "qpos": np.asarray(articulation.get_qpos(), dtype=np.float64).copy(),
        "qvel": np.asarray(articulation.get_qvel(), dtype=np.float64).copy(),
        "joints": [],
    }
    qacc = _get_attr(articulation, "get_qacc")
    if qacc is not None:
        row["qacc"] = np.asarray(qacc, dtype=np.float64).copy()
    for key, getter in (("root_pose", "get_root_pose"), ("root_linear_velocity", "get_root_linear_velocity"), ("root_angular_velocity", "get_root_angular_velocity"), ("qf", "get_qf")):
        value = _get_attr(articulation, getter)
        if value is not None:
            row[key] = _pose(value) if key == "root_pose" else np.asarray(value, dtype=np.float64).copy()
    for joint in articulation.get_active_joints():
        target = _get_attr(joint, "get_drive_target", "drive_target")
        velocity = _get_attr(joint, "get_drive_velocity_target", "drive_velocity_target")
        if target is None or velocity is None:
            raise SnapshotCapabilityError("active joint lacks drive target/velocity API")
        row["joints"].append({
            "drive_target": np.asarray(target, dtype=np.float64).copy(),
            "drive_velocity_target": np.asarray(velocity, dtype=np.float64).copy(),
        })
    return row


def capability_report(env: Any) -> dict[str, Any]:
    base = _base_env(env)
    scene = getattr(base, "scene", None)
    missing: list[str] = []
    if scene is None:
        missing.append("scene")
    else:
        for name in ("pack_poses", "unpack_poses", "get_all_actors", "get_all_articulations"):
            if not callable(getattr(scene, name, None)):
                missing.append(f"scene.{name}")
        try:
            _physx_system(scene)
        except SnapshotCapabilityError as error:
            missing.append(str(error))
    if not callable(getattr(base, "get_obs", None)):
        missing.append("get_obs")
    if not callable(getattr(base, "take_action", None)):
        missing.append("take_action")
    native = callable(getattr(base, "get_state_dict", None)) and callable(getattr(base, "set_state_dict", None))
    return {
        "schema": SNAPSHOT_SCHEMA,
        "native_state_api": native,
        "missing": missing,
        "reference_implementation_available": not bool(missing),
        "exact": not bool(missing),
    }


def require_capability(env: Any) -> dict[str, Any]:
    report = capability_report(env)
    if report["missing"]:
        raise SnapshotCapabilityError(
            "BiCoord exact snapshot/restore unavailable; missing " + ", ".join(report["missing"])
        )
    return report


def capture_state(env: Any) -> dict[str, Any]:
    """Capture the complete numerical simulator/controller state.

    Prefer a benchmark-owned native state API when present.  The fallback
    reference serializer is deliberately marked ``reference_serializer`` and
    still requires a deterministic restore probe before formal use.
    """
    base = _base_env(env)
    require_capability(base)
    native_get = getattr(base, "get_state_dict", None)
    if callable(native_get) and callable(getattr(base, "set_state_dict", None)):
        native = _numeric(native_get())
        if native is None:
            raise SnapshotCapabilityError("native get_state_dict returned non-serializable state")
        return {
            "schema": SNAPSHOT_SCHEMA,
            "serializer": "native",
            "native": native,
            "wrapper_attrs": _wrapper_state(env),
            "rng": _rng_state(),
        }
    scene = base.scene
    actors = scene.get_all_actors()
    arts = scene.get_all_articulations()
    state: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "serializer": "reference_serializer",
        "scene_poses": bytes(scene.pack_poses()),
        "physx_state": bytes(_physx_system(scene).pack()),
        "actors": {_object_key(obj, i): _actor_state(obj) for i, obj in enumerate(actors)},
        "articulations": {_object_key(obj, i): _articulation_state(obj) for i, obj in enumerate(arts)},
        "env_attrs": {},
        "wrapper_attrs": _wrapper_state(env),
        "rng": _rng_state(),
    }
    timestep = _get_attr(scene, "get_timestep", "timestep")
    if timestep is not None:
        state["scene_timestep"] = float(timestep)
    # Capture every plain task/controller value except known engine handles.
    skip = {"scene", "engine", "renderer", "robot", "cameras", "viewer", "left_entity", "right_entity"}
    for key, value in vars(base).items():
        if key in skip or key.startswith("_snapshot_"):
            continue
        item = _numeric(value)
        if item is not None or value is None:
            state["env_attrs"][key] = item
    robot = getattr(base, "robot", None)
    if robot is not None:
        state["robot_attrs"] = {}
        for key, value in vars(robot).items():
            if any(token in key for token in ("entity", "planner", "conn", "camera", "joint", "gripper")):
                # gripper scalar values are captured explicitly below.
                if key not in {"left_gripper_val", "right_gripper_val"}:
                    continue
            item = _numeric(value)
            if item is not None or value is None:
                state["robot_attrs"][key] = item
    return state


def _restore_wrapper_state(env: Any, rows: Any) -> None:
    if not isinstance(rows, list):
        raise SnapshotCapabilityError("snapshot wrapper state is malformed")
    chain = _wrapper_chain(env)
    if len(chain) != len(rows):
        raise SnapshotCapabilityError("wrapper depth drift during restore")
    for depth, (wrapper, saved) in enumerate(zip(chain, rows)):
        expected_type = f"{type(wrapper).__module__}.{type(wrapper).__qualname__}"
        if (
            not isinstance(saved, Mapping)
            or int(saved.get("depth", -1)) != depth
            or saved.get("type") != expected_type
            or not isinstance(saved.get("values"), Mapping)
        ):
            raise SnapshotCapabilityError("wrapper identity drift during restore")
        for key, value in saved["values"].items():
            if not hasattr(wrapper, key):
                raise SnapshotCapabilityError(
                    f"wrapper counter disappeared during restore: depth={depth} {key}"
                )
            setattr(wrapper, key, _restore_numeric(value))


def restore_state(env: Any, state: Mapping[str, Any]) -> None:
    base = _base_env(env)
    if state.get("schema") != SNAPSHOT_SCHEMA:
        raise SnapshotCapabilityError("snapshot schema mismatch")
    if state.get("serializer") == "native":
        setter = getattr(base, "set_state_dict", None)
        if not callable(setter):
            raise SnapshotCapabilityError("snapshot requires native set_state_dict")
        setter(deepcopy(state["native"]))
    else:
        require_capability(base)
        scene = base.scene
        if "scene_timestep" in state and not _set_attr(
            scene,
            ("set_timestep", "timestep"),
            float(state["scene_timestep"]),
        ):
            raise SnapshotCapabilityError("cannot restore scene timestep")
        scene.unpack_poses(bytes(state["scene_poses"]))
        actors = scene.get_all_actors()
        by_key = {_object_key(obj, i): obj for i, obj in enumerate(actors)}
        for key, row in state["actors"].items():
            obj = by_key.get(key)
            if obj is None:
                raise SnapshotCapabilityError(f"actor identity drift during restore: {key}")
            _set_pose(obj, row["pose"])
            component = _component(obj, "rigid") or _component(obj, "physx")
            if component is not None:
                if "kinematic" in row and not _set_attr(
                    component, ("set_kinematic", "kinematic"), bool(row["kinematic"])
                ):
                    raise SnapshotCapabilityError("cannot restore actor kinematic mode")
                if "kinematic_target" in row:
                    target_setter = getattr(component, "set_kinematic_target", None)
                    if callable(target_setter):
                        _set_pose_value(target_setter, row["kinematic_target"])
                    elif hasattr(component, "kinematic_target"):
                        setattr(
                            component,
                            "kinematic_target",
                            _sapien_pose(row["kinematic_target"]),
                        )
                    else:
                        raise SnapshotCapabilityError("cannot restore actor kinematic target")
                if "linear_velocity" in row and not _set_attr(component, ("set_linear_velocity", "linear_velocity"), np.array(row["linear_velocity"], copy=True)):
                    raise SnapshotCapabilityError("cannot restore actor linear velocity")
                if "angular_velocity" in row and not _set_attr(component, ("set_angular_velocity", "angular_velocity"), np.array(row["angular_velocity"], copy=True)):
                    raise SnapshotCapabilityError("cannot restore actor angular velocity")
        arts = scene.get_all_articulations()
        by_key = {_object_key(obj, i): obj for i, obj in enumerate(arts)}
        for key, row in state["articulations"].items():
            obj = by_key.get(key)
            if obj is None:
                raise SnapshotCapabilityError(f"articulation identity drift during restore: {key}")
            obj.set_qpos(np.array(row["qpos"], copy=True))
            obj.set_qvel(np.array(row["qvel"], copy=True))
            if "qacc" in row and not _set_attr(obj, ("set_qacc", "qacc"), np.array(row["qacc"], copy=True)):
                raise SnapshotCapabilityError("cannot restore articulation qacc")
            if "qf" in row and not _set_attr(obj, ("set_qf", "qf"), np.array(row["qf"], copy=True)):
                raise SnapshotCapabilityError("cannot restore articulation qf")
            if "root_pose" in row:
                _set_root_pose(obj, row["root_pose"])
            if "root_linear_velocity" in row and not _set_attr(
                obj,
                ("set_root_linear_velocity", "root_linear_velocity"),
                np.array(row["root_linear_velocity"], copy=True),
            ):
                raise SnapshotCapabilityError("cannot restore articulation root linear velocity")
            if "root_angular_velocity" in row and not _set_attr(
                obj,
                ("set_root_angular_velocity", "root_angular_velocity"),
                np.array(row["root_angular_velocity"], copy=True),
            ):
                raise SnapshotCapabilityError("cannot restore articulation root angular velocity")
            joints = obj.get_active_joints()
            if len(joints) != len(row["joints"]):
                raise SnapshotCapabilityError("active-joint identity drift during restore")
            for joint, saved in zip(joints, row["joints"]):
                joint.set_drive_target(np.array(saved["drive_target"], copy=True))
                joint.set_drive_velocity_target(
                    np.array(saved["drive_velocity_target"], copy=True)
                )
        # ``set_qpos`` recomputes articulation-link poses and SAPIEN 3 can
        # round their quaternion components by a few float32 ulps even when
        # qpos is unchanged.  Re-apply the scene-owned packed pose blob only
        # after qpos/qvel/controller restoration.  Besides preserving actors
        # and articulation-link poses exactly, this gives the paired evaluator
        # a stable state hash without weakening its bitwise drift gate.
        scene.unpack_poses(bytes(state["scene_poses"]))
        # Entity setters do not expose the PhysX solver/contact warm-start
        # state.  Restore the scene-owned CPU-system blob last so the next
        # integration begins from the exact captured physical state rather
        # than merely matching visible poses and generalized coordinates.
        # This is intentionally mandatory: snapshots created by schema v1 or
        # runtimes without pack/unpack must never be accepted as CARE data.
        if "physx_state" not in state:
            raise SnapshotCapabilityError("snapshot lacks PhysX system state")
        _physx_system(scene).unpack(bytes(state["physx_state"]))
        for key, value in state.get("env_attrs", {}).items():
            setattr(base, key, _restore_numeric(value))
        robot = getattr(base, "robot", None)
        if robot is not None:
            for key, value in state.get("robot_attrs", {}).items():
                setattr(robot, key, _restore_numeric(value))
    _restore_wrapper_state(env, state.get("wrapper_attrs"))
    rng = state.get("rng", {})
    if "python" in rng:
        random.setstate(deepcopy(rng["python"]))
    if "numpy" in rng:
        np.random.set_state(deepcopy(rng["numpy"]))
    if "torch" in rng:
        torch.set_rng_state(torch.as_tensor(rng["torch"]).cpu())
    if torch.cuda.is_available() and rng.get("cuda"):
        torch.cuda.set_rng_state_all([torch.as_tensor(value).cpu() for value in rng["cuda"]])
    # Do not implicitly call ``_update_render`` here.  BiCoord's Base_Task
    # synchronises camera/link poses in that method, and SAPIEN 3 may rewrite
    # packed scene poses by a few bytes even though the physical state is
    # unchanged.  Such a render side effect breaks the exact state hash used
    # by paired CARE validation.  Callers that need observations explicitly
    # render (``get_obs``) after this function returns.


def max_abs(first: Any, second: Any) -> float:
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        if not isinstance(first, Mapping) or not isinstance(second, Mapping) or set(first) != set(second):
            return math.inf
        return max((max_abs(first[key], second[key]) for key in first), default=0.0)
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)) or len(first) != len(second):
            return math.inf
        return max((max_abs(a, b) for a, b in zip(first, second)), default=0.0)
    a, b = np.asarray(first), np.asarray(second)
    if a.shape != b.shape:
        return math.inf
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
        return 0.0 if np.array_equal(a, b) else math.inf
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0


def state_sha256(value: Any) -> str:
    """Hash a snapshot tree including dtype/shape and structural boundaries."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=lambda child: repr(child)):
                update(key)
                update(item[key])
            digest.update(b"}")
            return
        if isinstance(item, tuple):
            digest.update(f"tuple:{len(item)}[".encode("ascii"))
            for child in item:
                update(child)
            digest.update(b"]")
            return
        if isinstance(item, list):
            digest.update(f"list:{len(item)}[".encode("ascii"))
            for child in item:
                update(child)
            digest.update(b"]")
            return
        if torch.is_tensor(item):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"array:")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
            return
        if isinstance(item, bytes):
            digest.update(f"bytes:{len(item)}:".encode("ascii"))
            digest.update(item)
            return
        if item is None or isinstance(item, (bool, int, float, str)):
            digest.update(f"{type(item).__name__}:{item!r};".encode("utf-8"))
            return
        raise SnapshotCapabilityError(
            f"snapshot hash encountered unsupported value: {type(item)!r}"
        )

    update(value)
    return digest.hexdigest()


def restore_probe(
    env: Any,
    state: Mapping[str, Any],
    rollout: Callable[[Any], Mapping[str, Any]],
    *,
    repeats: int = 2,
    tolerance: float = SNAPSHOT_TOLERANCE,
) -> dict[str, Any]:
    """Run identical post-restore rollouts and return a deterministic gate."""
    if repeats < 2:
        raise ValueError("restore probe needs at least two repeats")
    rows: list[Mapping[str, Any]] = []
    for _ in range(repeats):
        restore_state(env, state)
        rows.append(deepcopy(rollout(env)))
    error = max((max_abs(rows[0], row) for row in rows[1:]), default=math.inf)
    return {"schema": SNAPSHOT_SCHEMA, "repeats": repeats, "max_abs_error": error, "tolerance": tolerance, "passed": bool(error <= tolerance)}


__all__ = ["SNAPSHOT_SCHEMA", "SNAPSHOT_TOLERANCE", "SnapshotCapabilityError", "capability_report", "capture_state", "restore_state", "require_capability", "restore_probe", "max_abs", "state_sha256"]
