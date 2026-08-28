from __future__ import annotations

import argparse, copy, json
from pathlib import Path
import numpy as np
import torch

from .common import TASKS, local_observation, make_env
from .evaluate import decode_actions, load_policy, policy_inputs, run_episode
from collections import deque


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--robofactory-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--closed-loop", action="store_true"); args = p.parse_args()
    device = torch.device("cuda:0"); model, q_mean, q_std, a_mean, a_std = load_policy(args.checkpoint, device); rows = []
    for task_id, task in enumerate(TASKS):
        env = make_env(task, args.robofactory_root); obs, _ = env.reset(seed=20260824 + task_id)
        arms = tuple(range(task.arms)); histories = [deque(maxlen=model.config.history) for _ in arms]
        previous = [None for _ in arms]
        images, qposes, raw_qposes, history, history_mask = policy_inputs(obs, arms, histories, previous, q_mean, q_std, a_mean, a_std, model.config.image_size, device)
        ids = torch.full((task.arms,), task_id, dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): chunks, selected = model.act(images, qposes, ids, history, history_mask)
        if chunks.shape != (task.arms, model.config.horizon, 8) or not torch.isfinite(chunks).all(): raise RuntimeError(f"invalid action tensor for {task.name}")
        # Prove an arm's extracted input is invariant to every other arm's private qpos.
        modified = copy.deepcopy(obs)
        focal_image, focal_qpos = local_observation(obs, 0)
        for other in range(1, task.arms): modified["agent"][f"panda-{other}"]["qpos"] = torch.randn_like(modified["agent"][f"panda-{other}"]["qpos"]) * 100
        changed_image, changed_qpos = local_observation(modified, 0)
        if not np.array_equal(focal_image, changed_image) or not np.array_equal(focal_qpos, changed_qpos): raise RuntimeError("private teammate input leaked into focal observation")
        action_rows = decode_actions(chunks, raw_qposes, a_mean, a_std)[:, 0]
        action = {f"panda-{arm}": np.clip(action_rows[arm], env.action_space.spaces[f"panda-{arm}"].low, env.action_space.spaces[f"panda-{arm}"].high) for arm in range(task.arms)}
        _obs, _reward, _terminated, _truncated, info = env.step(action); env.close()
        rows.append({"task": task.name, "arms": task.arms, "action_shape": list(chunks.shape), "strict_local_invariance": True, "environment_step": True, "success_after_one_step": bool(np.asarray(info.get("success", False)).all())})
    closed_loop = []
    if args.closed_loop:
        stats = (q_mean, q_std, a_mean, a_std)
        for task_id, task in enumerate(TASKS):
            closed_loop.append(run_episode(model, task, args.robofactory_root, task_id * 100000, device, stats, task.max_steps))
        if not any(row["success"] for row in closed_loop):
            raise RuntimeError(f"closed-loop smoke retained the zero-success failure: {[(x['task'], x['success']) for x in closed_loop]}")
    report = {"status": "passed", "checks": rows, "closed_loop": closed_loop}; args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report))


if __name__ == "__main__": main()
