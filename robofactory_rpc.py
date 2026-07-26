"""Small fail-closed RPC primitives for cross-environment RoboFactory rollouts.

The simulator and M1 policy intentionally run in different Python environments.
This module therefore depends only on NumPy and the Python standard library.  It
uses JSON metadata plus contiguous, lossless ndarray bytes; no pickle payload is
accepted and RGB is never JPEG-compressed.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import socket
import struct
from typing import Any, Optional

import numpy as np


PROTOCOL_VERSION = "wam.robofactory.closed_loop/1"
FORMAL_LIFTBARRIER_M1_CONFIG_SHA256 = (
    "c2627a5176c53e36b2cd34079a3dc9c058c819912763006c8f213a9f81e1fa89"
)
DEFAULT_MAX_FRAME_BYTES = 64 * 1024 * 1024
_FRAME_LENGTH = struct.Struct("!Q")
_HEADER_LENGTH = struct.Struct("!I")
_MAX_HEADER_BYTES = 1024 * 1024
_MAX_ARRAYS = 32
_ALLOWED_DTYPE_KINDS = frozenset({"b", "u", "i", "f"})


class RoboFactoryRPCError(RuntimeError):
    """Raised when the peer violates the closed-loop wire contract."""


def configure_socket(connection: socket.socket, *, timeout_seconds: float) -> None:
    """Apply the common low-latency and failure-detection socket settings."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive and finite")
    connection.settimeout(float(timeout_seconds))
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if connection.family in {socket.AF_INET, socket.AF_INET6}:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def send_message(
    connection: socket.socket,
    message: Mapping[str, Any],
    arrays: Optional[Mapping[str, np.ndarray]] = None,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> None:
    """Send one versioned message and zero or more lossless ndarrays."""

    if not isinstance(message, Mapping):
        raise TypeError("message must be a mapping")
    message_type = message.get("type")
    if not isinstance(message_type, str) or not message_type:
        raise ValueError("message.type must be a non-empty string")
    if max_frame_bytes <= 0:
        raise ValueError("max_frame_bytes must be positive")

    values = {} if arrays is None else dict(arrays)
    if len(values) > _MAX_ARRAYS:
        raise ValueError(f"a message may contain at most {_MAX_ARRAYS} arrays")
    descriptors: list[dict[str, Any]] = []
    payloads: list[memoryview] = []
    for name, raw_value in values.items():
        if not isinstance(name, str) or not name or len(name) > 128:
            raise ValueError("array names must be non-empty strings of at most 128 chars")
        array = np.asarray(raw_value)
        _validate_dtype(array.dtype)
        if array.ndim > 8:
            raise ValueError(f"array {name!r} has too many dimensions")
        contiguous = np.ascontiguousarray(array)
        descriptors.append(
            {
                "name": name,
                "dtype": contiguous.dtype.str,
                "shape": [int(value) for value in contiguous.shape],
                "nbytes": int(contiguous.nbytes),
            }
        )
        payloads.append(memoryview(contiguous).cast("B"))

    header = dict(message)
    if "protocol" in header and header["protocol"] != PROTOCOL_VERSION:
        raise ValueError("message protocol does not match this implementation")
    if "arrays" in header:
        raise ValueError("message key 'arrays' is reserved by the wire protocol")
    header["protocol"] = PROTOCOL_VERSION
    header["arrays"] = descriptors
    try:
        header_bytes = json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("message metadata must be finite JSON data") from exc
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ValueError("message JSON header is too large")
    frame_size = _HEADER_LENGTH.size + len(header_bytes) + sum(
        len(value) for value in payloads
    )
    if frame_size > max_frame_bytes:
        raise ValueError(
            f"message frame is {frame_size} bytes; limit is {max_frame_bytes}"
        )

    connection.sendall(_FRAME_LENGTH.pack(frame_size))
    connection.sendall(_HEADER_LENGTH.pack(len(header_bytes)))
    connection.sendall(header_bytes)
    for payload in payloads:
        connection.sendall(payload)


def receive_message(
    connection: socket.socket,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Receive and validate one complete message from the peer."""

    if max_frame_bytes <= 0:
        raise ValueError("max_frame_bytes must be positive")
    frame_size = _FRAME_LENGTH.unpack(_receive_exact(connection, _FRAME_LENGTH.size))[0]
    if frame_size < _HEADER_LENGTH.size or frame_size > max_frame_bytes:
        raise RoboFactoryRPCError(
            f"invalid frame size {frame_size}; maximum is {max_frame_bytes}"
        )
    frame = _receive_exact(connection, int(frame_size))
    header_size = _HEADER_LENGTH.unpack_from(frame, 0)[0]
    if header_size <= 0 or header_size > _MAX_HEADER_BYTES:
        raise RoboFactoryRPCError("invalid JSON header size")
    header_end = _HEADER_LENGTH.size + int(header_size)
    if header_end > len(frame):
        raise RoboFactoryRPCError("JSON header exceeds the received frame")
    try:
        message = json.loads(
            frame[_HEADER_LENGTH.size : header_end].decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise RoboFactoryRPCError("message has an invalid JSON header") from exc
    if not isinstance(message, dict):
        raise RoboFactoryRPCError("message JSON root must be an object")
    if message.get("protocol") != PROTOCOL_VERSION:
        raise RoboFactoryRPCError(
            f"protocol mismatch: expected {PROTOCOL_VERSION!r}, "
            f"received {message.get('protocol')!r}"
        )
    if not isinstance(message.get("type"), str) or not message["type"]:
        raise RoboFactoryRPCError("message.type must be a non-empty string")
    raw_descriptors = message.pop("arrays", None)
    if not isinstance(raw_descriptors, list) or len(raw_descriptors) > _MAX_ARRAYS:
        raise RoboFactoryRPCError("message arrays descriptor is invalid")

    arrays: dict[str, np.ndarray] = {}
    cursor = header_end
    for descriptor in raw_descriptors:
        if not isinstance(descriptor, dict):
            raise RoboFactoryRPCError("array descriptor must be an object")
        name = descriptor.get("name")
        dtype_name = descriptor.get("dtype")
        shape = descriptor.get("shape")
        nbytes = descriptor.get("nbytes")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 128
            or name in arrays
        ):
            raise RoboFactoryRPCError("array descriptor has an invalid/duplicate name")
        if not isinstance(dtype_name, str) or len(dtype_name) > 16:
            raise RoboFactoryRPCError(f"array {name!r} has an invalid dtype")
        try:
            dtype = np.dtype(dtype_name)
            _validate_dtype(dtype)
        except (TypeError, ValueError) as exc:
            raise RoboFactoryRPCError(f"array {name!r} has an invalid dtype") from exc
        if (
            not isinstance(shape, list)
            or len(shape) > 8
            or any(not isinstance(value, int) or value < 0 for value in shape)
        ):
            raise RoboFactoryRPCError(f"array {name!r} has an invalid shape")
        elements = math.prod(shape)
        expected_nbytes = elements * dtype.itemsize
        if not isinstance(nbytes, int) or nbytes != expected_nbytes:
            raise RoboFactoryRPCError(f"array {name!r} byte count is inconsistent")
        end = cursor + nbytes
        if end > len(frame):
            raise RoboFactoryRPCError(f"array {name!r} exceeds the received frame")
        arrays[name] = np.frombuffer(frame, dtype=dtype, count=elements, offset=cursor).reshape(
            tuple(shape)
        ).copy()
        cursor = end
    if cursor != len(frame):
        raise RoboFactoryRPCError("message contains unclaimed trailing bytes")
    return message, arrays


def extract_liftbarrier_observation(
    observation: Mapping[str, Any],
    *,
    camera_name: str = "head_camera_global",
) -> tuple[np.ndarray, np.ndarray]:
    """Map one RoboFactory RGB observation to the exact M1 online contract."""

    try:
        agents = observation["agent"]
        sensor_data = observation["sensor_data"]
    except (KeyError, TypeError) as exc:
        raise ValueError("RoboFactory observation lacks agent/sensor_data") from exc
    state_parts: list[np.ndarray] = []
    for agent_name in ("panda-0", "panda-1"):
        try:
            agent = agents[agent_name]
            qpos = _unbatch_vector(agent["qpos"], expected=9, name=f"{agent_name}.qpos")
            qvel = _unbatch_vector(agent["qvel"], expected=9, name=f"{agent_name}.qvel")
        except (KeyError, TypeError) as exc:
            raise ValueError(f"RoboFactory observation lacks state for {agent_name}") from exc
        state_parts.extend((qpos, qvel))
    state = np.ascontiguousarray(np.concatenate(state_parts), dtype=np.float32)
    if state.shape != (36,) or not np.isfinite(state).all():
        raise ValueError("RoboFactory centralized state must be finite float32[36]")

    try:
        rgb = _to_numpy(sensor_data[camera_name]["rgb"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"RoboFactory observation lacks RGB camera {camera_name!r}") from exc
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"{camera_name}.rgb must be [1,H,W,3] or [H,W,3]")
    if rgb.dtype != np.uint8:
        raise ValueError(f"{camera_name}.rgb must be lossless uint8")
    return state, np.ascontiguousarray(rgb)


def split_liftbarrier_action(action: Any) -> dict[str, np.ndarray]:
    """Split the centralized raw pd_joint_pos command into two Panda actions."""

    value = np.asarray(action, dtype=np.float32)
    if value.shape != (16,) or not np.isfinite(value).all():
        raise ValueError("LiftBarrier action must be finite float32[16]")
    return {
        "panda-0": np.ascontiguousarray(value[:8]),
        "panda-1": np.ascontiguousarray(value[8:]),
    }


def extract_robofactory_observation(
    observation: Mapping[str, Any],
    *,
    agent_order: tuple[str, ...],
    camera_name: str = "head_camera_global",
) -> tuple[np.ndarray, np.ndarray]:
    """Map a native RoboFactory multi-Panda observation to M2 arrays."""

    if not agent_order or len(agent_order) != len(set(agent_order)):
        raise ValueError("agent_order must be non-empty and unique")
    try:
        agents = observation["agent"]
        sensor_data = observation["sensor_data"]
    except (KeyError, TypeError) as exc:
        raise ValueError("RoboFactory observation lacks agent/sensor_data") from exc
    state_parts: list[np.ndarray] = []
    for agent_name in agent_order:
        try:
            agent = agents[agent_name]
            qpos = _unbatch_vector(agent["qpos"], expected=9, name=f"{agent_name}.qpos")
            qvel = _unbatch_vector(agent["qvel"], expected=9, name=f"{agent_name}.qvel")
        except (KeyError, TypeError) as exc:
            raise ValueError(f"RoboFactory observation lacks state for {agent_name}") from exc
        state_parts.extend((qpos, qvel))
    state = np.ascontiguousarray(np.concatenate(state_parts), dtype=np.float32)
    if state.shape != (18 * len(agent_order),) or not np.isfinite(state).all():
        raise ValueError("RoboFactory centralized M2 state is invalid")
    try:
        rgb = _to_numpy(sensor_data[camera_name]["rgb"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"RoboFactory observation lacks RGB camera {camera_name!r}") from exc
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"{camera_name}.rgb must be lossless uint8 HWC")
    return state, np.ascontiguousarray(rgb)


def extract_robofactory_multiview_observation(
    observation: Mapping[str, Any],
    *,
    agent_order: tuple[str, ...],
    camera_names: Mapping[str, str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Map one native observation to centralized state plus named RGB views."""

    normalized = {
        str(name): str(source)
        for name, source in camera_names.items()
    }
    if (
        not normalized
        or len(normalized) != len(camera_names)
        or len(set(normalized.values())) != len(normalized)
    ):
        raise ValueError("camera_names must map unique logical views to sources")
    first_source = next(iter(normalized.values()))
    state, first_rgb = extract_robofactory_observation(
        observation,
        agent_order=agent_order,
        camera_name=first_source,
    )
    images: dict[str, np.ndarray] = {}
    sensor_data = observation.get("sensor_data")
    if not isinstance(sensor_data, Mapping):
        raise ValueError("RoboFactory observation lacks sensor_data")
    for logical_name, source_name in normalized.items():
        if source_name == first_source:
            rgb = first_rgb
        else:
            try:
                rgb = _to_numpy(sensor_data[source_name]["rgb"])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"RoboFactory observation lacks RGB camera {source_name!r}"
                ) from exc
            if rgb.ndim == 4 and rgb.shape[0] == 1:
                rgb = rgb[0]
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError(
                    f"{source_name}.rgb must be lossless uint8 HWC"
                )
        images[logical_name] = np.ascontiguousarray(rgb)
    shapes = {tuple(value.shape) for value in images.values()}
    if len(shapes) != 1:
        raise ValueError(f"RoboFactory camera shapes differ: {sorted(shapes)}")
    return state, images


def split_robofactory_action(
    action: Any, *, agent_order: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Split one canonical-order raw command into 8D Panda controller inputs."""

    value = np.asarray(action, dtype=np.float32)
    if value.shape != (8 * len(agent_order),) or not np.isfinite(value).all():
        raise ValueError("RoboFactory M2 action has the wrong finite shape")
    return {
        agent: np.ascontiguousarray(value[index * 8 : (index + 1) * 8])
        for index, agent in enumerate(agent_order)
    }


def scalar_bool(value: Any, *, name: str) -> bool:
    """Convert one tensor/array scalar to bool without ambiguous broadcasting."""

    array = _to_numpy(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one value")
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean scalar")
    return bool(array.reshape(-1)[0])


def scalar_float(value: Any, *, name: str) -> float:
    """Convert one tensor/array scalar to a finite Python float."""

    array = _to_numpy(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one scalar")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def wilson_interval(
    successes: int,
    episodes: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return the two-sided Wilson interval for a Bernoulli success rate."""

    if episodes <= 0 or successes < 0 or successes > episodes:
        raise ValueError("successes/episodes are inconsistent")
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (proportion + z * z / (2.0 * episodes)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / episodes
            + z * z / (4.0 * episodes * episodes)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    payload = bytearray(size)
    view = memoryview(payload)
    offset = 0
    while offset < size:
        received = connection.recv_into(view[offset:])
        if received == 0:
            raise ConnectionError("peer closed the rollout connection")
        offset += received
    return bytes(payload)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _validate_dtype(dtype: np.dtype[Any]) -> None:
    if dtype.fields is not None or dtype.subdtype is not None:
        raise ValueError("structured/subarray dtypes are not supported")
    if dtype.kind not in _ALLOWED_DTYPE_KINDS or dtype.itemsize > 8:
        raise ValueError(f"unsupported ndarray dtype {dtype}")


def _to_numpy(value: Any) -> np.ndarray:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    return np.asarray(current)


def _unbatch_vector(value: Any, *, expected: int, name: str) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (expected,):
        raise ValueError(f"{name} must be [1,{expected}] or [{expected}]")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return np.ascontiguousarray(array)


__all__ = [
    "DEFAULT_MAX_FRAME_BYTES",
    "FORMAL_LIFTBARRIER_M1_CONFIG_SHA256",
    "PROTOCOL_VERSION",
    "RoboFactoryRPCError",
    "configure_socket",
    "extract_liftbarrier_observation",
    "extract_robofactory_multiview_observation",
    "extract_robofactory_observation",
    "receive_message",
    "scalar_bool",
    "scalar_float",
    "send_message",
    "split_liftbarrier_action",
    "split_robofactory_action",
    "wilson_interval",
]
