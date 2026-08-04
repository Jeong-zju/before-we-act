#!/usr/bin/env python3
"""One exact-batch forward/backward canary for the no-wrist policy."""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

from no_wrist_pair_model import NoWristPAIRRoute


path = next(Path("/workspace/datasets/robofactory_multitask/camera_alignment/hdf5").glob("*.hdf5"))
with h5py.File(path, "r") as handle:
    data = handle["data"]
    global_rgb = torch.from_numpy(data["observation"]["images"]["global"][0]).permute(2, 0, 1)
    local_rgb = torch.from_numpy(data["observation"]["images"]["agent_0"][0]).permute(2, 0, 1)
    qpos = torch.from_numpy(data["observation"]["agents"]["panda_0"]["qpos"][0])

batch = 40
global_rgb = global_rgb.unsqueeze(0).repeat(batch, 1, 1, 1).float().div_(255).cuda()
local_rgb = local_rgb.unsqueeze(0).repeat(batch, 1, 1, 1).float().div_(255).cuda()
qpos = qpos.unsqueeze(0).repeat(batch, 1).cuda()
actions = torch.randn(batch, 100, 8, device="cuda")
model = NoWristPAIRRoute(
    9,
    8,
    dino_model="/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m",
).cuda().train()
started = time.time()
with torch.autocast("cuda", dtype=torch.bfloat16):
    prediction, mu, logvar, _, routes, counterfactual, target = model(
        global_rgb,
        local_rgb,
        qpos,
        actions,
        return_routing=True,
        counterfactual=True,
    )
    action_loss = (prediction - actions).square().mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
    errors = (counterfactual - target.unsqueeze(2)).square().mean(-1)
    temperature = errors.detach().std(-1, keepdim=True).clamp_min(1e-3)
    capability = (-errors.detach() / temperature).softmax(-1)
    coupling = F.kl_div(routes[:1].clamp_min(1e-8).log(), capability, reduction="none").sum(-1).mean()
    loss = action_loss + 1e-3 * kl + 0.05 * coupling
loss.backward()
torch.cuda.synchronize()
print(
    {
        "batch": batch,
        "loss": float(loss.detach()),
        "seconds": round(time.time() - started, 3),
        "max_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "finite": bool(torch.isfinite(loss)),
    },
    flush=True,
)
