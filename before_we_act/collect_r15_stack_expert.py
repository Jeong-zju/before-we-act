"""Record successful Stack oracle trajectories at the deployed 480x640 RGB contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


ENV_ID = "ThreeRobotsStackCube-rf"
CAMERAS = (
    "head_camera_global",
    "head_camera_agent0",
    "head_camera_agent1",
    "head_camera_agent2",
)
RGB_SHAPE = (480, 640, 3)


def validate_recorded_source(
    hdf5_path: str | Path,
    metadata_path: str | Path,
    expected_episodes: int,
) -> dict[str, object]:
    import h5py

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    rows = metadata.get("episodes")
    if (
        not isinstance(rows, list)
        or len(rows) != expected_episodes
        or not all(isinstance(row, dict) and bool(row.get("success")) for row in rows)
    ):
        raise ValueError("R15 expert metadata does not contain exact successful episodes")
    shapes: dict[str, tuple[int, ...]] = {}
    with h5py.File(hdf5_path, "r") as handle:
        trajectories = sorted(key for key in handle if key.startswith("traj_"))
        if len(trajectories) != expected_episodes:
            raise ValueError("R15 expert HDF5 trajectory count differs")
        for trajectory in trajectories:
            sensors = handle[f"{trajectory}/obs/sensor_data"]
            if set(sensors) != set(CAMERAS):
                raise ValueError("R15 expert camera identities differ")
            for camera in CAMERAS:
                shape = tuple(sensors[camera]["rgb"].shape)
                if shape[1:] != RGB_SHAPE:
                    raise ValueError(
                        f"R15 expert RGB differs at {trajectory}/{camera}: {shape}"
                    )
                shapes[f"{trajectory}/{camera}"] = shape
    return {
        "episodes": expected_episodes,
        "cameras": list(CAMERAS),
        "rgb_shape": list(RGB_SHAPE),
        "recorded_shapes": {key: list(value) for key, value in shapes.items()},
    }


def collect(args: argparse.Namespace) -> tuple[Path, Path, dict[str, object]]:
    import gymnasium as gym
    import numpy as np
    import robofactory  # noqa: F401
    from robofactory.planner.solutions.three_robots_stack_cube import solve
    from robofactory.utils.wrappers.record import RecordEpisodeMA

    output_dir = args.output_root / ENV_ID / "motionplanning"
    hdf5_path = output_dir / f"{args.trajectory_name}.h5"
    metadata_path = hdf5_path.with_suffix(".json")
    if hdf5_path.exists() or metadata_path.exists():
        raise FileExistsError("R15 native expert output already exists")
    env = gym.make(
        ENV_ID,
        config=str(args.config),
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="sensors",
        reward_mode="dense",
        sensor_configs={
            "shader_pack": "default",
            "width": RGB_SHAPE[1],
            "height": RGB_SHAPE[0],
        },
        human_render_camera_configs={"shader_pack": "default"},
        viewer_camera_configs={"shader_pack": "default"},
        sim_backend="cpu",
    )
    env = RecordEpisodeMA(
        env,
        output_dir=str(output_dir),
        trajectory_name=args.trajectory_name,
        save_video=False,
        source_type="motionplanning",
        source_desc="R15 native 480x640 RoboFactory motion-planning oracle",
        avoid_overwriting_video=True,
        save_on_reset=False,
        record_reward=False,
        record_env_state=True,
        record_observation=True,
    )
    passed = attempts = failed_motion_plans = 0
    seed = args.start_seed
    lengths: list[int] = []
    try:
        while passed < args.episodes:
            if attempts >= args.max_attempts:
                raise RuntimeError(
                    f"native expert collected {passed}/{args.episodes} successes "
                    f"after {attempts} attempts"
                )
            attempts += 1
            result = solve(env, seed=seed, debug=False, vis=False)
            success = result != -1 and bool(result[-1]["success"].item())
            if result == -1:
                failed_motion_plans += 1
            if not success:
                env.flush_trajectory(save=False)
                print(
                    json.dumps(
                        {"attempt": attempts, "seed": seed, "success": False},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                seed += 1
                continue
            length = int(np.asarray(result[-1]["elapsed_steps"].item()).item())
            lengths.append(length)
            env.flush_trajectory()
            passed += 1
            print(
                json.dumps(
                    {
                        "attempt": attempts,
                        "completed_successes": passed,
                        "target_successes": args.episodes,
                        "seed": seed,
                        "success": True,
                        "steps": length,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            seed += 1
    finally:
        env.close()
    validation = validate_recorded_source(hdf5_path, metadata_path, args.episodes)
    return hdf5_path, metadata_path, {
        **validation,
        "attempts": attempts,
        "failed_motion_plans": failed_motion_plans,
        "min_steps": min(lengths),
        "max_steps": max(lengths),
        "mean_steps": sum(lengths) / len(lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--trajectory-name", required=True)
    parser.add_argument("--max-attempts", type=int, default=0)
    args = parser.parse_args()
    if args.episodes < 1 or args.start_seed < 1 or not args.trajectory_name:
        raise ValueError("invalid native expert collection options")
    args.config = args.config.resolve(strict=True)
    args.output_root = args.output_root.resolve()
    args.max_attempts = args.max_attempts or args.episodes * 5
    started = time.monotonic()
    hdf5_path, metadata_path, receipt = collect(args)
    print(
        json.dumps(
            {
                **receipt,
                "hdf5": str(hdf5_path.resolve()),
                "metadata_json": str(metadata_path.resolve()),
                "elapsed_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
