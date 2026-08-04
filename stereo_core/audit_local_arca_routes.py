from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from stereo_decoder_variants import StereoARCA


def js_divergence(p, q):
    p = np.asarray(p, np.float64) + 1e-12
    q = np.asarray(q, np.float64) + 1e-12
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    model = StereoARCA(
        config["state_dim"], config["action_dim"], horizon=100, d_model=384,
        enc_layers=4, dec_layers=7, roles=config.get("experts", 4),
        role_rank=config.get("role_rank", 32),
    )
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    groups = defaultdict(lambda: {"gates": [], "top1": []})
    phase_groups = defaultdict(list)
    with torch.no_grad():
        for path in sorted(glob.glob(args.features)):
            with h5py.File(path, "r") as source:
                for episode in source.values():
                    task, split = str(episode.attrs["task"]), str(episode.attrs["split"])
                    if split != "test":
                        continue
                    for agent_name, agent in episode.items():
                        arm = int(agent_name.split("_")[-1])
                        observation = torch.from_numpy(np.asarray(agent["observation"], np.float32))
                        qpos = torch.from_numpy(np.asarray(agent["qpos"], np.float32))
                        times = np.asarray(agent["time"], np.float32)
                        phase_bins = np.minimum((5 * times / max(float(times.max()) + 1.0, 1.0)).astype(np.int64), 4)
                        for start in range(0, len(qpos), 512):
                            obs = observation[start:start + 512]
                            state = model.state(qpos[start:start + 512])
                            query = model.query.expand(len(state), -1, -1)
                            context = model.route_state(state) + model.route_observation(obs)
                            features = model.route_mlp(query + context.unsqueeze(1))
                            logits = torch.matmul(features, model.role_prototypes.t()) / np.sqrt(features.shape[-1])
                            values, ids = logits.topk(2, dim=-1)
                            gates = torch.zeros_like(logits).scatter_(-1, ids, values.softmax(-1))
                            groups[(task, arm)]["gates"].append(gates.mean(1).numpy())
                            groups[(task, arm)]["top1"].append(gates.argmax(-1).numpy())
                            gate_mean = gates.mean(1).numpy()
                            for offset, phase in enumerate(phase_bins[start:start + len(gate_mean)]):
                                phase_groups[(task, arm, int(phase))].append(gate_mean[offset])

    result = {"by_task_agent": {}, "within_task_agent_js": {}, "by_task_agent_phase": {}, "phase_agent_js": {}, "interpretation": {}}
    task_usage = defaultdict(list)
    for (task, arm), value in sorted(groups.items()):
        gates = np.concatenate(value["gates"], 0)
        top1 = np.concatenate(value["top1"], 0).reshape(-1)
        mean = gates.mean(0); mean /= mean.sum()
        sample_entropy = -(gates + 1e-12) * np.log(gates + 1e-12)
        key = f"{task}/robot_{arm}"
        result["by_task_agent"][key] = {
            "mean_role_usage": mean.tolist(),
            "mean_query_gate_entropy_nats": float(sample_entropy.sum(-1).mean()),
            "top1_fraction": [(top1 == role).mean().item() for role in range(len(mean))],
            "frames": int(len(gates)),
        }
        task_usage[task].append((arm, mean))
    for task, values in task_usage.items():
        pairs = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                pairs.append({"agents": [values[i][0], values[j][0]], "js_nats": js_divergence(values[i][1], values[j][1])})
        result["within_task_agent_js"][task] = pairs
    phase_usage = {}
    for (task, arm, phase), values in sorted(phase_groups.items()):
        mean = np.mean(np.asarray(values), axis=0); mean /= mean.sum()
        phase_usage[(task, arm, phase)] = mean
        result["by_task_agent_phase"][f"{task}/robot_{arm}/phase_{phase}"] = mean.tolist()
    for task in sorted(task_usage):
        arms = sorted(arm for arm, _mean in task_usage[task])
        for phase in range(5):
            pairs = []
            for i in range(len(arms)):
                for j in range(i + 1, len(arms)):
                    left, right = phase_usage[(task, arms[i], phase)], phase_usage[(task, arms[j], phase)]
                    pairs.append({"agents": [arms[i], arms[j]], "js_nats": js_divergence(left, right)})
            result["phase_agent_js"][f"{task}/phase_{phase}"] = pairs
    all_js = [x["js_nats"] for pairs in result["within_task_agent_js"].values() for x in pairs]
    result["interpretation"] = {
        "mean_within_task_agent_js_nats": float(np.mean(all_js)),
        "max_within_task_agent_js_nats": float(np.max(all_js)),
        "note": "JS near zero means visually distinct local agents still receive the same aggregate role allocation."
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
