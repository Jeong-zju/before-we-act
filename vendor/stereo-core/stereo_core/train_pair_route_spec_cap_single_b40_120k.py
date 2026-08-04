"""Single-GPU PAIR-Route formal trainer with exact local-action batch 40."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from pair_route_model import StereoPAIRRoute
from role_observability_teacher import TeamRoleTeacher
from train_act import _trajectories, seed_everything
from train_pair_dataparallel import CompactPAIRWristDataset
from train_pair_route_from_scratch_dp import teacher_edges, relation_loss, fingerprint
from train_stereo_act import DEPTH_MM_TO_M


class Exact40TeamBlockSampler(Sampler):
    """Task-balanced blocks with exact batch 40 and intact synchronized teams.

    Two- and four-agent tasks divide 40 exactly. For three-agent tasks, 39
    samples form 13 complete synchronized teams and one extra local sample is
    marked group=-1. The extra sample participates in local action learning but
    is excluded from the synchronized relation loss.
    """

    def __init__(self, dataset, updates, block_updates, seed):
        self.dataset, self.updates = dataset, updates
        self.block_updates, self.seed, self.epoch = block_updates, seed, 0
        self.by_task = {}
        for episode in dataset.episodes:
            self.by_task.setdefault(episode[0], []).append(episode)
        self.tasks = sorted(self.by_task)
        if len(self.tasks) != 5:
            raise ValueError(f"expected all five tasks, got {self.tasks}")

    def __len__(self):
        return self.updates

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch); self.epoch += 1
        done = block = 0
        while done < self.updates:
            task = self.tasks[block % len(self.tasks)]
            _task, streams, length = self.by_task[task][rng.randrange(len(self.by_task[task]))]
            team_size = len(streams); complete = 40 // team_size; remainder = 40 % team_size
            for _ in range(min(self.block_updates, self.updates - done)):
                batch = []
                for group in range(complete):
                    time_index = rng.randrange(length)
                    batch.extend((stream, time_index, group) for stream in streams)
                for extra in range(remainder):
                    batch.append((streams[(done + extra) % team_size], rng.randrange(length), -1))
                if len(batch) != 40:
                    raise RuntimeError(f"exact batch invariant failed: {len(batch)}")
                rng.shuffle(batch); yield batch; done += 1
            block += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--normalization", required=True)
    p.add_argument("--teacher", required=True); p.add_argument("--output", required=True)
    p.add_argument("--updates", type=int, default=120000); p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--router-lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500); p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--relation-weight", type=float, default=.05); p.add_argument("--capability-weight", type=float, default=.05)
    p.add_argument("--specialization-weight", type=float, default=.01); p.add_argument("--anchor-weight", type=float, default=.02)
    p.add_argument("--counterfactual-every", type=int, default=4); p.add_argument("--discovery-fraction", type=float, default=.30)
    p.add_argument("--cache-episodes", type=int, default=16); p.add_argument("--block-updates", type=int, default=64)
    p.add_argument("--save-updates", default="60000,80000,100000,120000")
    p.add_argument("--log-every", type=int, default=50); p.add_argument("--seed", type=int, default=20260801)
    p.add_argument("--experiment-label", default="pair_route_full")
    p.add_argument("--gpu-label", default="GPU0")
    p.add_argument("--allow-preflight", action="store_true")
    args = p.parse_args()
    if args.batch_size != 40:
        raise ValueError("this locked control requires exact batch_size=40")
    if not args.allow_preflight and (args.updates != 120000 or args.batch_size * args.updates != 4_800_000):
        raise ValueError("formal protocol is batch40 x 120k optimizer updates")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    seed_everything(args.seed); torch.backends.cudnn.benchmark = True
    stats = torch.load(args.normalization, map_location="cpu", weights_only=False)["stats"]
    paths = sorted({x for pattern in args.data.split(",") for x in glob.glob(pattern)})
    trajectories = _trajectories(paths, (0, 1, 2, 3))
    dataset = CompactPAIRWristDataset(trajectories, 100, stats, True, args.cache_episodes)
    sampler = Exact40TeamBlockSampler(dataset, args.updates, args.block_updates, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)
    sample = dataset[(0, 0, 0)]; state_dim, action_dim = len(sample[2]), len(sample[3][0])

    model = StereoPAIRRoute(state_dim, action_dim, horizon=100, d_model=384, enc_layers=4,
                            dec_layers=7, roles=4, role_rank=32).to(device)
    teacher_saved = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher = TeamRoleTeacher(action_dim).to(device)
    teacher.load_state_dict(teacher_saved["model"]); teacher.eval().requires_grad_(False)
    router_prefix = ("compatibility", "role_prototypes", "route_state", "route_observation", "route_mlp")
    router, body = [], []
    for name, parameter in model.named_parameters():
        (router if name.startswith(router_prefix) else body).append(parameter)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": router, "lr": args.router_lr}],
                                  weight_decay=1e-4)

    def multiplier(step):
        warmup = min(1.0, (step + 1) / max(args.warmup, 1))
        cosine = .5 * (1 + math.cos(math.pi * min(1.0, (step + 1) / args.updates)))
        return warmup * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    discovery = int(args.updates * args.discovery_fraction)
    milestones = {int(value) for value in args.save_updates.split(",") if value}
    config = vars(args) | {
        "policy_variant": "stereo_pair_route", "training": "from scratch",
        "experiment_label": args.experiment_label,
        "parallelism": f"single RTX 5090 {args.gpu_label}", "global_batch": 40, "sample_budget": 4_800_000,
        "optimizer_updates": 120000, "horizon": 100, "enc_layers": 4, "dec_layers": 7,
        "camera_width": 640, "camera_height": 480, "patch_grid": [30, 40],
        "depth_storage_unit": "millimeters", "depth_to_meters_scale": DEPTH_MM_TO_M,
        "strict_policy_input": "single local panda_hand wrist RGB-D and own qpos; no ID/language/communication/global/peer/right-camera",
        "training_only_signal": {
            "synchronized_action_relation_teacher": args.relation_weight > 0,
            "real_target_counterfactual_expert_error": args.capability_weight > 0,
            "specialization_regularizer": args.specialization_weight > 0,
            "stable_capability_anchor": args.anchor_weight > 0,
        },
        "three_agent_batch_contract": "39 samples in 13 complete teams + 1 local-action-only sample excluded from relation loss",
        "task_sampling": "equal 64-update task blocks over all five tasks",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))
    torch.save({"stats": stats}, output / "normalization.pt")

    capability_ema = capability_reference = None; started = time.time(); last = {}
    for update, batch in enumerate(loader, 1):
        rgb, depth, qpos, actions, mask, groups = [item.to(device, non_blocking=True) for item in batch]
        synchronized = groups >= 0; do_counterfactual = update % args.counterfactual_every == 0
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, mu, logvar, _aux, routes, counterfactual, counterfactual_target = model(
                rgb.float().div_(255), depth, qpos, actions, True, do_counterfactual
            )
            action = (((prediction - actions).square().mean(-1) * mask).sum() / mask.sum().clamp_min(1))
            kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
            relation = routes.sum() * 0.
            if args.relation_weight > 0:
                relation = relation_loss(routes[synchronized], model.compatibility,
                                         teacher_edges(teacher, actions[synchronized], groups[synchronized]))
            sample_entropy = -(routes.clamp_min(1e-8).log() * routes).sum(-1).mean()
            marginal = routes.mean((0, 1)); marginal_entropy = -(marginal.clamp_min(1e-8).log() * marginal).sum()
            specialization = sample_entropy - marginal_entropy
            starvation = F.relu(.05 - marginal).square().sum()
            coupling = routes.sum() * 0.; anchor = routes.sum() * 0.
            if do_counterfactual:
                errors = (counterfactual - counterfactual_target.unsqueeze(2)).square().mean(-1)
                temperature = errors.detach().std(-1, keepdim=True).clamp_min(1e-3)
                qcap = (-errors.detach() / temperature).softmax(-1)
                coupling = F.kl_div(routes[:1].clamp_min(1e-8).log(), qcap, reduction="none").sum(-1).mean()
                if args.anchor_weight > 0:
                    fp = fingerprint(counterfactual)
                    capability_ema = fp.detach() if capability_ema is None else .99 * capability_ema + .01 * fp.detach()
                    if update >= discovery and capability_reference is None:
                        capability_reference = capability_ema.clone()
                    if capability_reference is not None:
                        anchor = F.mse_loss(fp, capability_reference)
            loss = (action + args.beta * kl + args.relation_weight * relation + args.capability_weight * coupling
                    + args.specialization_weight * (specialization + starvation) + args.anchor_weight * anchor)
        loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step(); scheduler.step()
        last = {"loss": float(loss), "action": float(action), "kl": float(kl), "relation": float(relation),
                "coupling": float(coupling), "anchor": float(anchor), "route_entropy": float(sample_entropy),
                "marginal_entropy": float(marginal_entropy),
                "near_half": float((routes.topk(2, -1).values.diff(dim=-1).abs() < .05).float().mean())}
        if update == 1 or update % args.log_every == 0:
            elapsed = time.time() - started
            print(json.dumps({"update": update, **last, "updates_per_hour": update / elapsed * 3600,
                              "eta_hours": (args.updates - update) * elapsed / update / 3600,
                              "gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
        if update in milestones:
            torch.save({"model": model.state_dict(), "stats": stats, "config": config, "update": update,
                        "capability_reference": capability_reference, "last_metrics": last,
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                       output / f"checkpoint_{update:06d}.pt")
    print(json.dumps({"complete": True, "checkpoint": str(output / 'checkpoint_120000.pt')}), flush=True)


if __name__ == "__main__":
    main()
