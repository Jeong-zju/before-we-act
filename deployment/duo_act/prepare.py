from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from .dataset import TASKS
from .action_target import (
    ACTION_TARGET_CONTRACT_ID,
    ACTION_TARGET_CONTRACT_SHA256,
    action_target_contract,
    canonicalize_controller_action_with_audit,
    validate_controller_action,
)
from .protocol import (
    DUOBENCH_CODE_REVISION,
    FORMAL_DATASET_REVISION,
    FORMAL_EPISODES_PER_TASK,
    FORMAL_IMAGE_SIZE,
    IMAGE_PREPROCESS_ID,
    FORMAL_CONTROLLER_CORRECTION_ENTRIES,
    FORMAL_CONTROLLER_CORRECTIONS_BY_JOINT,
    FORMAL_CONTROLLER_CORRECTIONS_BY_TASK,
    FORMAL_RCS_API_OUT_OF_RANGE_ENTRIES_DIAGNOSTIC,
    FORMAL_SIM_PARQUET_SHA256,
    RCS_LEROBOT_CONVERTER_REVISION,
    VALIDATION_HORIZON_METHOD,
    VALIDATION_HORIZON_QUANTILE,
    VALIDATION_MAX_STEPS,
    validate_task_length_contract,
)


ACTION_AUDIT_RECEIPT_NAME = "action_target_audit.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _aggregate_action_audits(tasks: dict[str, dict]) -> dict:
    audits = [tasks[task]["action_target_audit"] for task in TASKS]
    sum_keys = (
        "raw_values",
        "rows",
        "changed_values",
        "changed_joint_values",
        "changed_gripper_values",
        "out_of_controller_range_entries",
        "outside_rcs_api_limits_diagnostic_entries",
        "nonbinary_gripper_entries",
    )
    vector_sum_keys = (
        "out_of_controller_range_by_joint",
        "outside_rcs_api_limits_diagnostic_by_joint",
    )
    return {
        **{key: sum(int(row[key]) for row in audits) for key in sum_keys},
        **{
            key: np.sum(
                [np.asarray(row[key], dtype=np.int64) for row in audits], axis=0
            ).astype(int).tolist()
            for key in vector_sum_keys
        },
        "max_abs_delta": max(float(row["max_abs_delta"]) for row in audits),
        "max_abs_joint_delta": max(
            float(row["max_abs_joint_delta"]) for row in audits
        ),
        "rcs_api_limits_used_for_canonicalization": False,
    }


def decode_video(path: Path, size: int) -> np.ndarray:
    """Decode official LeRobot RGB without applying a second resize.

    RCS already resized native frames with torchvision ``v2.Resize`` before
    encoding the released 224x224 AV1 streams.  Resampling the decoded video a
    second time would create a silent train/deploy mismatch, so any unexpected
    source resolution is a hard error.
    """

    container = av.open(str(path))
    frames = []
    try:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != (size, size, 3) or rgb.dtype != np.uint8:
                raise RuntimeError(
                    f"{path}: official LeRobot RGB must already be "
                    f"{size}x{size} uint8, got {rgb.shape}/{rgb.dtype}"
                )
            frames.append(np.ascontiguousarray(rgb))
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"{path}: video contains no decodable RGB frames")
    return np.asarray(frames, dtype=np.uint8)


def prepare_task(dataset: Path, output: Path, size: int, task: str):
    root = dataset / task / "sim"
    parquet_files = sorted((root / "data").glob("**/*.parquet"))
    if len(parquet_files) != 1:
        raise RuntimeError(f"{task}: expected one parquet shard, found {len(parquet_files)}")
    table = pq.read_table(parquet_files[0]).to_pydict()
    episodes = np.asarray(table["episode_index"])
    states = np.asarray(table["observation.state"], dtype=np.float32)
    raw_actions = np.asarray(table["action"], dtype=np.float32)
    if (
        states.ndim != 2
        or raw_actions.ndim != 2
        or states.shape[1] != 16
        or raw_actions.shape != states.shape
    ):
        raise RuntimeError(
            f"{task}: expected raw state/action [N,16], got "
            f"{states.shape}/{raw_actions.shape}"
        )
    # The official action column is the raw successful Pin IK output.  RCS'
    # narrow Gym Box is not the physical controller range and is never used
    # here.  MuJoCo's pinned position-actuator ctrlrange is the last operation
    # actually applied to the command, so make that saturation explicit before
    # computing normalization or exposing a training target.
    actions_local, action_audit = canonicalize_controller_action_with_audit(
        raw_actions.reshape(-1, 2, 8)
    )
    validate_controller_action(actions_local)
    source_parquet_sha256 = _sha256_file(parquet_files[0])
    if source_parquet_sha256 != FORMAL_SIM_PARQUET_SHA256[task]:
        raise RuntimeError(
            f"{task}: simulation parquet hash differs from the immutable "
            f"DuoBench snapshot"
        )
    if (
        action_audit["out_of_controller_range_by_joint"]
        != FORMAL_CONTROLLER_CORRECTIONS_BY_TASK[task]
    ):
        # The parquet hash matched one line above, so the source bytes are
        # identical and the data cannot be what changed. These counts are a
        # deterministic function of (parquet, controller bounds), which leaves
        # exactly one explanation -- and blaming the dataset for it has cost
        # real debugging time.
        raise RuntimeError(
            f"{task}: the parquet hash matches, so the controller joint bounds "
            f"in deployment.duo_act.action_target changed. Saturation counts "
            f"per joint are now "
            f"{action_audit['out_of_controller_range_by_joint']}, recorded as "
            f"{FORMAL_CONTROLLER_CORRECTIONS_BY_TASK[task]}. Restore the bounds, "
            f"or re-freeze the recorded counts if the change is intended."
        )
    actions = actions_local.reshape(-1, 16)
    if states.ndim != 2 or actions.ndim != 2 or states.shape[1] != 16 or actions.shape[1] != 16:
        raise RuntimeError(f"{task}: expected state/action [N,16], got {states.shape}/{actions.shape}")
    out = output / task
    out.mkdir(parents=True, exist_ok=True)
    unique = np.unique(episodes)
    lengths = [int(np.sum(episodes == episode)) for episode in unique]
    if len(unique) != FORMAL_EPISODES_PER_TASK:
        raise RuntimeError(f"{task}: expected all 50 demos, found {len(unique)}")
    try:
        length_contract = validate_task_length_contract(task, lengths)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if not all((out / f"{name}.npy").is_file() for name in ("head", "left", "right")):
        videos = {}
        for name, source in (("head", "observation.images.head"), ("left", "observation.images.left_wrist"), ("right", "observation.images.right_wrist")):
            video_files = sorted((root / "videos" / source).glob("**/*.mp4"))
            if len(video_files) != 1:
                raise RuntimeError(f"{task}/{source}: expected one video, found {len(video_files)}")
            videos[name] = decode_video(video_files[0], size)
            if len(videos[name]) != len(states):
                raise RuntimeError(f"{task}/{name}: video has {len(videos[name])} frames, table has {len(states)}")
        for name, frames in videos.items():
            np.save(out / f"{name}.npy", frames)
    # Always refresh numeric arrays so action-range fixes do not require the
    # expensive videos to be decoded again.
    np.save(out / "state.npy", states)
    # Keep the unmodified converter output next to the canonical target.  It
    # is not consumed by training, but lets the audit recompute every change
    # instead of trusting only extrema/counts in a JSON receipt.
    np.save(out / "raw_action.npy", raw_actions)
    np.save(out / "action.npy", actions)
    np.save(out / "episodes.npy", episodes)
    action_audit.update(
        {
            "contract_id": ACTION_TARGET_CONTRACT_ID,
            "contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
            "action_encoding": "absolute_joint7_binary_gripper1",
            "raw_ik_array_sha256": _sha256_array(raw_actions),
            "controller_equivalent_array_sha256": _sha256_array(actions),
            "source_parquet_sha256": source_parquet_sha256,
            "source_parquet": str(parquet_files[0]),
        }
    )
    return task, {
        "episodes": len(unique),
        "frames": len(states),
        "min_demo_steps": min(lengths),
        "mean_demo_steps": float(np.mean(lengths)),
        "max_demo_steps": max(lengths),
        "validation_max_steps": int(length_contract["validation_max_steps"]),
        "validation_q99_steps": float(length_contract["q99"]),
        "validation_horizon_method": VALIDATION_HORIZON_METHOD,
        "validation_horizon_quantile": VALIDATION_HORIZON_QUANTILE,
        "state_min": states.min(0).tolist(),
        "state_max": states.max(0).tolist(),
        "action_min": actions.min(0).tolist(),
        "action_max": actions.max(0).tolist(),
        "raw_action_min": raw_actions.min(0).tolist(),
        "raw_action_max": raw_actions.max(0).tolist(),
        "controller_canonicalized_joint_values": action_audit[
            "changed_joint_values"
        ],
        "binarized_action_values": action_audit["changed_gripper_values"],
        "action_target_audit": action_audit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.image_size != FORMAL_IMAGE_SIZE:
        raise ValueError(
            f"formal Duo preparation requires the released {FORMAL_IMAGE_SIZE}x"
            f"{FORMAL_IMAGE_SIZE} LeRobot videos"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "duobench-act-prepared-v1",
        "image_size": args.image_size,
        "image_preprocessing": {
            "id": IMAGE_PREPROCESS_ID,
            "training_video_resolution": [FORMAL_IMAGE_SIZE, FORMAL_IMAGE_SIZE],
            "training_decode_resize": "none_already_converter_resized",
            "runtime_source_resolution": [720, 1280],
            "runtime_resize": "torchvision_v2_uint8_bilinear_antialias_true",
            "views_resized_independently": True,
            "rcs_converter_revision": RCS_LEROBOT_CONVERTER_REVISION,
        },
        "validation_horizon": {
            "method": VALIDATION_HORIZON_METHOD,
            "quantile": VALIDATION_HORIZON_QUANTILE,
            "population": "all_50_successful_demo_lengths_per_task",
            "per_task_max_steps": dict(VALIDATION_MAX_STEPS),
        },
        "tasks": {},
        "action_target_contract": action_target_contract(),
        "recording_alignment": {
            "source_row_semantics": "post_action_observation_and_same_row_executed_action",
            "evidence": "RCS_StorageWrapper_step_then_record_and_JointDatasetConverter_same_row_join",
            "policy_decision_pair": "observation_row_i_to_action_row_i_plus_1",
            "action_lag_rows": 1,
            "source_dataset_immutable": True,
            "duplicate_action_filter": "RCS_converter_allclose_atol_1e-4",
            "released_timestamps": "regenerated_fixed_30Hz_after_filter",
        },
    }
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(prepare_task, args.dataset, args.output, args.image_size, task) for task in TASKS]
        for future in as_completed(futures):
            task, result = future.result()
            manifest["tasks"][task] = result
            print(json.dumps({"event": "task_prepared", "task": task, **result}), flush=True)
    manifest["tasks"] = {task: manifest["tasks"][task] for task in TASKS}
    qposes, actions = [], []
    for task in TASKS:
        states = np.load(args.output / task / "state.npy", mmap_mode="r").reshape(-1, 2, 8)
        task_actions = np.load(args.output / task / "action.npy", mmap_mode="r").reshape(-1, 2, 8)
        episodes = np.load(args.output / task / "episodes.npy", mmap_mode="r")
        starts = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1]])
        ends = np.r_[starts[1:], len(episodes)]
        for start, end in zip(starts, ends, strict=True):
            if end - start < 2:
                raise RuntimeError(f"{task}: episode at row {start} has no causal decision pair")
            qposes.append(states[start : end - 1])
            actions.append(task_actions[start + 1 : end])
    qpos = np.concatenate(qposes).astype(np.float64)
    action = np.concatenate(actions).astype(np.float64)
    q_std = np.maximum(qpos.std((0, 1)), 1e-4)
    a_std = np.maximum(action.std((0, 1)), 1e-4)
    manifest["normalization"] = {
        "qpos_mean": qpos.mean((0, 1)).tolist(), "qpos_std": q_std.tolist(),
        "qpos_min": qpos.min((0, 1)).tolist(), "qpos_max": qpos.max((0, 1)).tolist(),
        "action_mean": action.mean((0, 1)).tolist(), "action_std": a_std.tolist(),
        "action_min": action.min((0, 1)).tolist(), "action_max": action.max((0, 1)).tolist(),
        "action_encoding": "absolute_joint7_binary_gripper1",
        "action_target_contract_id": ACTION_TARGET_CONTRACT_ID,
        "action_target_contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "population": "all_causal_pairs_all_50_demos_all_11_tasks_both_local_arms",
        "qpos_rows": "post_action_observation_rows_except_each_episode_last",
        "action_rows": "same_episode_next_rows_except_each_episode_first",
        "action_lag_rows": 1,
    }
    manifest["total_episodes"] = sum(row["episodes"] for row in manifest["tasks"].values())
    manifest["total_frames"] = sum(row["frames"] for row in manifest["tasks"].values())
    manifest["total_policy_samples"] = manifest["total_frames"] - manifest["total_episodes"]
    manifest["dataset_revision"] = FORMAL_DATASET_REVISION
    manifest["duobench_code_revision"] = DUOBENCH_CODE_REVISION
    action_totals = _aggregate_action_audits(manifest["tasks"])
    if (
        action_totals["out_of_controller_range_entries"]
        != FORMAL_CONTROLLER_CORRECTION_ENTRIES
        or action_totals["out_of_controller_range_by_joint"]
        != FORMAL_CONTROLLER_CORRECTIONS_BY_JOINT
        or action_totals["outside_rcs_api_limits_diagnostic_entries"]
        != FORMAL_RCS_API_OUT_OF_RANGE_ENTRIES_DIAGNOSTIC
    ):
        raise RuntimeError(f"formal Duo action audit totals drifted: {action_totals}")
    receipt = {
        "schema": "before-we-act.duobench.action-target-audit/1",
        "status": "PASSED",
        "dataset_revision": FORMAL_DATASET_REVISION,
        "duobench_code_revision": DUOBENCH_CODE_REVISION,
        "rcs_lerobot_converter_revision": RCS_LEROBOT_CONVERTER_REVISION,
        "contract": action_target_contract(),
        "contract_id": ACTION_TARGET_CONTRACT_ID,
        "contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "source_semantics": {
            "released_action_column": "raw_successful_RCS_Pin_IK_joint_output",
            "raw_cartesian_pose_present_in_released_lerobot": False,
            "raw_cartesian_pose_required_for_controller_equivalent_target": False,
            "ik_success_rows": "converter_dropped_failed_IK_before_release",
            "canonical_target": "explicit_MuJoCo_position_actuator_ctrlrange_saturation",
            "normalization_population": (
                "controller_equivalent_actions_all_frames_all_50_demos_"
                "all_11_tasks_both_local_arms"
            ),
        },
        "forbidden_repairs": [
            "periodic_angle_guessing",
            "next_qpos_substitution",
            "RCS_API_joint_limits_clipping",
            "unreceipted_clipping",
        ],
        "totals": action_totals,
        "tasks": {
            task: manifest["tasks"][task]["action_target_audit"] for task in TASKS
        },
    }
    receipt_path = args.output / ACTION_AUDIT_RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    manifest["action_target_audit"] = {
        "schema": receipt["schema"],
        "status": "PASSED",
        "path": ACTION_AUDIT_RECEIPT_NAME,
        "sha256": _sha256_file(receipt_path),
        "contract_id": ACTION_TARGET_CONTRACT_ID,
        "contract_sha256": ACTION_TARGET_CONTRACT_SHA256,
        "totals": action_totals,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", "episodes": manifest["total_episodes"], "frames": manifest["total_frames"]}))


if __name__ == "__main__":
    main()
