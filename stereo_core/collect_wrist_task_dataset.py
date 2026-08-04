"""Reusable expert-data collector for every canonical 2/3/4-agent task.

No data are collected merely by importing this file.  When invoked, the output
contains all local wrist RGB streams plus own qpos/actions for later single-task
and pooled multi-task ACT experiments.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import gymnasium as gym
import robofactory
from robofactory.planner.run import MP_SOLUTIONS
from robofactory.utils.wrappers.record import RecordEpisodeMA
from two_three_task_manifest import get_task


ROOT = Path("/workspace/RoboFactory")


def worker(task_name, worker_index, requested, seed_start, stride, output_dir, gpus, camera_width, camera_height, obs_mode):
    spec = get_task(task_name)
    # The supervisor assigns a task-level GPU mask before spawning workers.
    # Do not overwrite it here: doing so made every task silently render on
    # physical GPU 0 when several tasks were launched together.
    # Must precede importing the camera retrofit: CameraConfig dimensions are
    # fixed when each worker installs its per-process task patch.
    os.environ["ROBOFACTORY_WRIST_WIDTH"] = str(camera_width)
    os.environ["ROBOFACTORY_WRIST_HEIGHT"] = str(camera_height)
    import wrist_camera_patch  # noqa: F401 -- mounted panda_hand cameras
    env = gym.make(
        spec["env_id"], config=str(ROOT / spec["config"]), obs_mode=obs_mode,
        control_mode="pd_joint_pos", render_mode="sensors", reward_mode="dense",
        sim_backend="cpu", sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    recorder = RecordEpisodeMA(
        env, output_dir=output_dir, trajectory_name=f"worker_{worker_index:02d}",
        save_video=False, save_on_reset=False, record_reward=True,
        record_env_state=False, record_observation=True, source_type="motionplanning",
        source_desc=f"{task_name}; local RGB cameras mounted on matching panda_hand links",
    )
    solver = MP_SOLUTIONS[spec["env_id"]]
    saved, attempts, seed, records = 0, 0, seed_start + worker_index, []
    while saved < requested:
        result = solver(recorder, seed=seed, debug=False, vis=False)
        success = result != -1 and bool(result[-1]["success"].item())
        steps = int(result[-1]["elapsed_steps"].item()) if result != -1 else 0
        attempts += 1
        if success:
            recorder.flush_trajectory(); saved += 1
        else:
            recorder.flush_trajectory(save=False)
        records.append({"seed": seed, "success": success, "steps": steps})
        print({"task": task_name, "worker": worker_index, "saved": saved, "target": requested, **records[-1]}, flush=True)
        seed += stride
    result = {"worker": worker_index, "h5": recorder._h5_file.filename, "successes": saved, "attempts": attempts, "records": records}
    recorder.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(__import__("two_three_task_manifest").TASKS))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--obs-mode", choices=("rgb", "rgbd"), default="rgb",
                        help="rgb for the DINO baseline; rgbd for single-camera local RGB-D collection.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    workers = min(args.workers, args.count)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    targets = [args.count // workers + int(index < args.count % workers) for index in range(workers)]
    jobs = [(args.task, index, targets[index], args.seed_start, workers, str(output), args.gpus,
             args.camera_width, args.camera_height, args.obs_mode) for index in range(workers)]
    if workers == 1:
        reports = [worker(*jobs[0])]
    else:
        mp.set_start_method("spawn", force=True)
        with mp.Pool(workers) as pool:
            reports = pool.starmap(worker, jobs)
    (output / "manifest.json").write_text(json.dumps({
        "task": args.task, "spec": get_task(args.task),
        "camera": {"mount": "panda_hand", "width": args.camera_width, "height": args.camera_height,
                   "observation_mode": args.obs_mode,
                   # ManiSkill rgbd stores metric depth as int16 millimeters;
                   # the Stereo-ACT loader converts raw_depth * 0.001 to metres.
                   "depth_storage_unit": "millimeters" if args.obs_mode == "rgbd" else None},
        "reports": reports,
    }, indent=2))


if __name__ == "__main__":
    main()
