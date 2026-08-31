"""Stream-audit all BiCoord demonstrations and compute native normalization."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    ACTION_DIM,
    ACTION_ENCODING,
    DATASET_REVISION,
    EPISODES_PER_TASK,
    GRIPPER_ENCODING,
    GRIPPER_NATIVE_RANGE,
    SOURCE_FREQUENCY_HZ,
    STATE_DIM,
    TASKS,
    TOTAL_EPISODES,
)
from .data import (
    compute_normalization,
    discover_bicoord_episodes,
    episode_manifest,
    project_local_observation,
    write_normalization_receipt,
)
from .hdf5_data import BiCoordHDF5Reader, load_stage_segments, validate_hdf5_schema
from .stage_common import artifact, assert_common_paths, atomic_json, common_parser, publish_result, read_json, sha256_file, utc_now


def _verify_download_receipt(dataset: Path) -> dict[str, Any]:
    path = dataset / "dataset_receipt.json"
    receipt = read_json(path)
    expected = {
        "schema": "before-we-act.bicoord-dataset-download/1",
        "status": "PASSED",
        "dataset_revision": DATASET_REVISION,
        "episodes": TOTAL_EPISODES,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"dataset download receipt differs at {key}: {receipt.get(key)!r} != {value!r}"
            )
    counts = receipt.get("episodes_per_task")
    if not isinstance(counts, dict) or counts != {
        task: EPISODES_PER_TASK for task in TASKS
    }:
        raise ValueError("dataset download receipt task coverage differs")
    return receipt


def _audit_local_projection(episode) -> dict[str, Any]:
    reader = BiCoordHDF5Reader(episode.path, task=episode.task, episode_id=episode.episode_id)
    returned_fields: list[str] | None = None
    for focal_arm, peer_side in ((0, "right"), (1, "left")):
        raw = {
            "observation": {
                "head_camera": {"rgb": reader.frame("head_camera", 0)},
                "left_camera": {"rgb": reader.frame("left_camera", 0)},
                "right_camera": {"rgb": reader.frame("right_camera", 0)},
            },
            "joint_action": {
                "left_arm": reader.state(0, 0)[:6],
                "left_gripper": reader.state(0, 0)[6],
                "right_arm": reader.state(0, 1)[:6],
                "right_gripper": reader.state(0, 1)[6],
                "vector": np.r_[reader.state(0, 0), reader.state(0, 1)],
            },
        }
        first = project_local_observation(raw, focal_arm)
        raw["joint_action"][f"{peer_side}_arm"] = (
            np.asarray(raw["joint_action"][f"{peer_side}_arm"]) + 10_000
        )
        raw["joint_action"][f"{peer_side}_gripper"] = (
            float(raw["joint_action"][f"{peer_side}_gripper"]) + 10_000
        )
        raw["observation"][f"{peer_side}_camera"]["rgb"] = np.zeros_like(
            raw["observation"][f"{peer_side}_camera"]["rgb"]
        )
        second = project_local_observation(raw, focal_arm)
        invariant = (
            np.array_equal(first["head_rgb"], second["head_rgb"])
            and np.array_equal(first["wrist_rgb"], second["wrist_rgb"])
            and np.array_equal(first["state"], second["state"])
        )
        if not invariant:
            raise RuntimeError(
                f"peer mutation changed focal-arm policy input for arm {focal_arm}"
            )
        fields = sorted(first)
        if returned_fields is not None and fields != returned_fields:
            raise RuntimeError("left/right local projections expose different fields")
        returned_fields = fields
    return {
        "strict_local_projection": True,
        "both_arms_peer_mutation_invariant": True,
        "returned_fields": returned_fields,
    }


def _audit_numeric_ranges(episode) -> dict[str, Any]:
    """Check native finite ranges without clipping or reparameterization."""

    reader = BiCoordHDF5Reader(episode.path, task=episode.task, episode_id=episode.episode_id)
    with reader._open() as handle:
        values: dict[str, np.ndarray] = {
            name: np.asarray(handle[f"joint_action/{name}"])
            for name in ("left_arm", "right_arm", "left_gripper", "right_gripper")
        }
    for name, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite native action values in {episode.path}:{name}")
    for name in ("left_gripper", "right_gripper"):
        value = values[name]
        if np.any(value < 0.0) or np.any(value > 1.0):
            raise ValueError(
                f"native gripper range is outside [0,1] in {episode.path}:{name}"
            )
    return {
        "state_finite": True,
        "action_finite": True,
        "gripper_range": list(GRIPPER_NATIVE_RANGE),
        "gripper_encoding": GRIPPER_ENCODING,
        "left_gripper_unique_values": int(np.unique(values["left_gripper"]).size),
        "right_gripper_unique_values": int(np.unique(values["right_gripper"]).size),
        "gripper_thresholding": False,
        "state_clipping": False,
        "action_clipping": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_common_paths(args, need_dataset=True)
    formal = args.operation == "formal-audit"
    download_receipt = _verify_download_receipt(args.dataset) if formal else None
    episodes = discover_bicoord_episodes(
        args.dataset, require_formal=formal, verify_schema=False
    )
    counts = Counter(item.task for item in episodes)
    schema_rows: list[dict[str, Any]] = []
    stage_files = instruction_files = 0
    total_pairs = 0
    min_length: int | None = None
    max_length = 0
    for index, episode in enumerate(episodes):
        metadata = validate_hdf5_schema(episode.path, check_images=True)
        if int(metadata["length"]) != episode.length:
            raise RuntimeError(f"episode length changed during audit: {episode.path}")
        if episode.stage_path:
            segments = load_stage_segments(episode.stage_path, length=episode.length)
            if not segments:
                raise ValueError(f"empty stage sidecar: {episode.stage_path}")
            stage_files += 1
        if episode.instruction_path:
            try:
                instruction = json.loads(Path(episode.instruction_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid instruction sidecar: {episode.instruction_path}") from error
            if instruction in (None, {}, []):
                raise ValueError(f"empty instruction sidecar: {episode.instruction_path}")
            instruction_files += 1
        numeric_ranges = _audit_numeric_ranges(episode)
        total_pairs += (episode.length - 1) * 2
        min_length = episode.length if min_length is None else min(min_length, episode.length)
        max_length = max(max_length, episode.length)
        schema_rows.append(
            {
                "task": episode.task,
                "episode_id": episode.episode_id,
                "length": episode.length,
                "hdf5_sha256": episode.hdf5_sha256,
                "stage": bool(episode.stage_path),
                "instruction": bool(episode.instruction_path),
                "numeric_ranges": numeric_ranges,
            }
        )
        if index == 0 or (index + 1) % 100 == 0:
            print(json.dumps({"event": "dataset_audit", "episodes": index + 1, "total": len(episodes)}), flush=True)
    if formal:
        expected_counts = Counter({task: EPISODES_PER_TASK for task in TASKS})
        if len(episodes) != TOTAL_EPISODES or counts != expected_counts:
            raise RuntimeError(f"formal BiCoord coverage differs: {counts}")
        if stage_files != TOTAL_EPISODES or instruction_files != TOTAL_EPISODES:
            raise RuntimeError(
                f"formal metadata is incomplete: stages={stage_files}, instructions={instruction_files}"
            )
    normalization = compute_normalization(episodes, status="PASSED" if formal else "SMOKE")
    gripper_unique_max = max(
        max(
            int(row["numeric_ranges"]["left_gripper_unique_values"]),
            int(row["numeric_ranges"]["right_gripper_unique_values"]),
        )
        for row in schema_rows
    )
    if formal and gripper_unique_max <= 2:
        raise RuntimeError(
            "formal source did not exhibit the expected continuous/interpolated gripper values"
        )
    for key in ("qpos_min", "qpos_max", "action_min", "action_max"):
        values = np.asarray(normalization.get(key), dtype=np.float64)
        if values.shape != (STATE_DIM,) or not np.isfinite(values).all():
            raise RuntimeError(f"normalization range receipt is invalid at {key}")
    for key in ("qpos_min", "action_min"):
        if float(normalization[key][-1]) < 0.0:
            raise RuntimeError(f"native gripper minimum is below zero: {key}")
    for key in ("qpos_max", "action_max"):
        if float(normalization[key][-1]) > 1.0:
            raise RuntimeError(f"native gripper maximum exceeds one: {key}")
    audit_root = args.run / "artifacts" / "dataset_audit"
    normalization_path = write_normalization_receipt(
        audit_root / "normalization.json", normalization
    )
    manifest = episode_manifest(episodes, root=args.dataset)
    manifest.update(
        {
            "schema": "before-we-act.bicoord-audit/1",
            "status": "PASSED" if formal else "SMOKE",
            "dataset_revision": DATASET_REVISION,
            "training_pairs": total_pairs,
            "stage_files": stage_files,
            "instruction_files": instruction_files,
            "min_episode_length": min_length,
            "max_episode_length": max_length,
            "source_frequency_hz": SOURCE_FREQUENCY_HZ,
            "alignment": "observation_row_t_to_action_row_t_plus_1",
            "held_out_episodes": 0,
            "state_clipping": False,
            "action_clipping": False,
            "gripper_reparameterization": False,
            "gripper_encoding": GRIPPER_ENCODING,
            "gripper_unique_values_max_per_episode": gripper_unique_max,
            "schema_rows": schema_rows,
            "local_projection": _audit_local_projection(episodes[0]),
        }
    )
    manifest_path = atomic_json(audit_root / "manifest.json", manifest)
    numeric_contract = {
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "source_frequency_hz": SOURCE_FREQUENCY_HZ,
        "alignment": "observation_row_t_to_action_row_t_plus_1",
        "action_encoding": ACTION_ENCODING,
        "gripper_encoding": GRIPPER_ENCODING,
        "gripper_native_range": list(GRIPPER_NATIVE_RANGE),
        "state_clipping": False,
        "action_clipping": False,
        "normalization_from_all_data": formal,
        "gripper_thresholding": False,
        "gripper_reparameterization": False,
    }
    audit_value = {
        "schema": "before-we-act.bicoord-dataset-audit/1",
        "status": "PASSED" if formal else "SMOKE",
        "dataset_revision": DATASET_REVISION,
        "episodes": len(episodes),
        "episodes_per_task": {
            task: int(counts.get(task, 0)) for task in TASKS
        },
        "held_out_episodes": 0,
        "training_pairs": total_pairs,
        "numeric_contract": numeric_contract,
        "normalization": {
            "path": str(normalization_path.resolve()),
            "sha256": sha256_file(normalization_path),
            "population": "all_demonstrations_both_local_arms_lag_one",
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "download_receipt": (
            {
                "path": str((args.dataset / "dataset_receipt.json").resolve()),
                "sha256": sha256_file(args.dataset / "dataset_receipt.json"),
            }
            if download_receipt is not None
            else None
        ),
        "stage_text_model_input": False,
        "episode_instruction_model_input": False,
        "peer_state_action_wrist_model_input": False,
        "audited_at": utc_now(),
    }
    audit_path = atomic_json(audit_root / "audit.json", audit_value)
    return publish_result(
        args,
        stage="dataset_audit",
        artifacts=[
            artifact(manifest_path, kind="dataset_manifest"),
            artifact(normalization_path, kind="normalization"),
            artifact(audit_path, kind="dataset_audit"),
        ],
        episodes=len(episodes),
        episodes_per_task={task: int(counts.get(task, 0)) for task in TASKS},
        held_out_episodes=0,
        dataset_revision=DATASET_REVISION,
        numeric_contract=numeric_contract,
        training_pairs=total_pairs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = common_parser(__doc__, ("formal-audit", "smoke-audit"))
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
