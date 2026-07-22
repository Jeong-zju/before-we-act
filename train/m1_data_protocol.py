"""Protocol dispatch for canonical visual-cue and generic trajectory M1 data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Sequence, TypeAlias

from train.generic_m1_trajectory_dataset import (
    GENERIC_M1_DATASET_PROTOCOL,
    GenericM1ManifestIndex,
    GenericM1WindowDataset,
)
from train.m1_manifest_dataset import M1ManifestIndex, M1WindowDataset


VISUAL_CUE_PAIR_PROTOCOL = "visual_cue_pair"

M1DataManifest: TypeAlias = M1ManifestIndex | GenericM1ManifestIndex
M1DataWindowDataset: TypeAlias = M1WindowDataset | GenericM1WindowDataset


@dataclass(frozen=True)
class M1DataCapabilities:
    """Features that a protocol may expose outside deployable samples."""

    dataset_protocol: str
    causal_pairs: bool
    event_probe_labels: bool
    decision_window_sampling: bool


def detect_m1_data_protocol(manifest_path: str | Path) -> str:
    """Read only the JSON header and return the explicit protocol name."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    try:
        raw = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON manifest: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be a JSON object")
    protocol = raw.get("dataset_protocol")
    if protocol == GENERIC_M1_DATASET_PROTOCOL:
        return GENERIC_M1_DATASET_PROTOCOL
    if protocol is None and raw.get("format_version") == "wam.multimodal.m0.dataset/2":
        return VISUAL_CUE_PAIR_PROTOCOL
    if protocol == VISUAL_CUE_PAIR_PROTOCOL:
        return VISUAL_CUE_PAIR_PROTOCOL
    raise ValueError(
        "unsupported M1 dataset protocol; expected "
        f"{GENERIC_M1_DATASET_PROTOCOL!r} or the canonical visual-cue manifest"
    )


def load_m1_data_manifest(
    manifest_path: str | Path,
    *,
    verify_hdf5_sha256: bool = True,
    verify_hdf5_contract: bool = True,
    verify_normalization: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> M1DataManifest:
    """Load one M1 manifest without weakening either protocol's contract."""

    protocol = detect_m1_data_protocol(manifest_path)
    if protocol == GENERIC_M1_DATASET_PROTOCOL:
        return GenericM1ManifestIndex.from_path(
            manifest_path,
            verify_hdf5_sha256=verify_hdf5_sha256,
            verify_hdf5_contract=verify_hdf5_contract,
            verify_normalization=verify_normalization,
            progress_callback=progress_callback,
        )
    return M1ManifestIndex.from_path(
        manifest_path,
        verify_hdf5_sha256=verify_hdf5_sha256,
        verify_hdf5_contract=verify_hdf5_contract,
        progress_callback=progress_callback,
    )


def build_m1_window_dataset(
    manifest: M1DataManifest | str | Path,
    *,
    split: str,
    state_history: int = 32,
    action_chunk: int = 8,
    cameras: Sequence[str] | None = None,
    visual_history: int = 2,
    future_horizons: Sequence[int] = (1, 2, 4, 8),
    stride: int = 1,
    decision_window_radius: int = 8,
    hdf5_cache_size: int = 8,
    verify_hdf5_sha256: bool = True,
    verify_hdf5_contract: bool = True,
    verify_normalization: bool = True,
) -> M1DataWindowDataset:
    """Build the protocol-specific dataset behind one deployable sample API."""

    if isinstance(manifest, (str, Path)):
        manifest = load_m1_data_manifest(
            manifest,
            verify_hdf5_sha256=verify_hdf5_sha256,
            verify_hdf5_contract=verify_hdf5_contract,
            verify_normalization=verify_normalization,
        )
    if isinstance(manifest, GenericM1ManifestIndex):
        return GenericM1WindowDataset(
            manifest,
            split=split,
            state_history=state_history,
            action_chunk=action_chunk,
            cameras=cameras,
            visual_history=visual_history,
            future_horizons=future_horizons,
            stride=stride,
            hdf5_cache_size=hdf5_cache_size,
        )
    if not isinstance(manifest, M1ManifestIndex):  # pragma: no cover - type guard.
        raise TypeError(f"unsupported M1 manifest index {type(manifest).__name__}")
    selected_cameras = ("fixed",) if cameras is None else tuple(cameras)
    return M1WindowDataset(
        manifest,
        split=split,
        state_history=state_history,
        action_chunk=action_chunk,
        cameras=selected_cameras,
        visual_history=visual_history,
        future_horizons=future_horizons,
        stride=stride,
        decision_window_radius=decision_window_radius,
        hdf5_cache_size=hdf5_cache_size,
    )


def m1_data_capabilities(manifest: M1DataManifest) -> M1DataCapabilities:
    """Return fail-closed capability flags used by training/evaluation code."""

    if isinstance(manifest, GenericM1ManifestIndex):
        return M1DataCapabilities(
            dataset_protocol=GENERIC_M1_DATASET_PROTOCOL,
            causal_pairs=False,
            event_probe_labels=False,
            decision_window_sampling=False,
        )
    if isinstance(manifest, M1ManifestIndex):
        return M1DataCapabilities(
            dataset_protocol=VISUAL_CUE_PAIR_PROTOCOL,
            causal_pairs=True,
            event_probe_labels=True,
            decision_window_sampling=True,
        )
    raise TypeError(f"unsupported M1 manifest index {type(manifest).__name__}")


def m1_data_protocol_evidence(manifest: M1DataManifest) -> dict[str, Any]:
    capabilities = m1_data_capabilities(manifest)
    return {
        "dataset_protocol": capabilities.dataset_protocol,
        "causal_pairs": capabilities.causal_pairs,
        "event_probe_labels": capabilities.event_probe_labels,
        "decision_window_sampling": capabilities.decision_window_sampling,
    }


__all__ = [
    "M1DataCapabilities",
    "M1DataManifest",
    "M1DataWindowDataset",
    "VISUAL_CUE_PAIR_PROTOCOL",
    "build_m1_window_dataset",
    "detect_m1_data_protocol",
    "load_m1_data_manifest",
    "m1_data_capabilities",
    "m1_data_protocol_evidence",
]
