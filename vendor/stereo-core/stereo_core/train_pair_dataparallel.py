"""Four-GPU DataParallel PAIR trainer.

This is the safe host-specific fallback when NCCL DDP is unavailable.  It
still executes the frozen RGB-D policy's forward/backward replicas on all four
5090s; gradients are reduced by PyTorch to GPU0 rather than NCCL.  The action
relation teacher is tiny and stays on GPU0 after the local policy outputs have
been gathered.  Global B128 x 25k preserves the previous 3.2M-sample budget.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from train_act import _stats, _trajectories, seed_everything
from train_stereo_act import DEPTH_MM_TO_M
from stereo_decoder_variants import PAIRActionTeacher, StereoPAIRAdapter


class CompactPAIRWristDataset(Dataset):
    """Stream-indexed RGB-D data without a Python object per control step.

    The earlier dataset expands every frame of all 500 demonstrations into a
    giant list before the first update.  PAIR's synchronized sampler knows a
    stream and time directly, so this compact representation eliminates that
    multi-minute CPU/RAM bottleneck without changing any image/action value.
    """
    def __init__(self, trajectories, horizon, stats, train, cache_limit=64):
        self.horizon, self.stats, self.cache_limit = horizon, stats, cache_limit
        kept = [item for index, item in enumerate(trajectories) if (index % 10 != 0) == train]
        self.streams, self.episodes = [], []
        for path, key, length, present, task in kept:
            stream_ids = []
            for arm in present:
                stream_ids.append(len(self.streams))
                self.streams.append((path, key, arm, task, length))
            self.episodes.append((task, tuple(stream_ids), length))
        self.cache = OrderedDict()

    def __len__(self):
        return sum(length * len(streams) for _task, streams, length in self.episodes)

    def _episode(self, stream_id):
        if stream_id not in self.cache:
            path, key, arm, _task, _length = self.streams[stream_id]
            with h5py.File(path, "r") as handle:
                trajectory = handle[key]
                sensor = trajectory["obs"]["sensor_data"][f"head_camera_agent{arm}"]
                rgb, depth = sensor["rgb"][:], sensor["depth"][:]
                if tuple(rgb.shape[1:]) != (480, 640, 3) or tuple(depth.shape[1:]) != (480, 640, 1):
                    raise ValueError(f"strict 640x480 RGB-D required for {path}:{key}:panda-{arm}")
                self.cache[stream_id] = (rgb, depth,
                    trajectory["obs"]["agent"][f"panda-{arm}"]["qpos"][:].astype(np.float32),
                    trajectory["actions"][f"panda-{arm}"][:].astype(np.float32))
            while len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(stream_id)
        return self.cache[stream_id]

    def __getitem__(self, request):
        stream_id, time, group = request
        rgb, depth, qpos, actions = self._episode(stream_id)
        future = actions[time:time + self.horizon]
        valid = len(future)
        padded = np.empty((self.horizon, actions.shape[1]), np.float32)
        padded[:valid], padded[valid:] = future, future[-1]
        mask = np.zeros(self.horizon, np.bool_); mask[:valid] = True
        return (torch.from_numpy(rgb[time]).permute(2, 0, 1).contiguous(),
                torch.from_numpy(depth[time]).permute(2, 0, 1).contiguous(),
                torch.from_numpy((qpos[time] - self.stats["q_mean"]) / self.stats["q_std"]),
                torch.from_numpy((padded - self.stats["a_mean"]) / self.stats["a_std"]),
                torch.from_numpy(mask), torch.tensor(group, dtype=torch.long))


class SameEpisodeTeamBlockSampler(Sampler):
    """One cached synchronized demonstration per 64-update block.

    Each batch contains complete teams at many independently selected times.
    This is both permutation invariant and I/O efficient: it keeps exactly
    2/3/4 local RGB-D streams resident rather than repeatedly decoding scores
    of long 640x480 demonstrations just to make one 120-sample batch.
    """
    def __init__(self, dataset, batch_size, updates, block_updates, seed):
        self.dataset, self.batch_size, self.updates = dataset, batch_size, updates
        self.block_updates, self.seed, self.epoch = block_updates, seed, 0
        self.by_task = defaultdict(list)
        for episode in dataset.episodes:
            self.by_task[episode[0]].append(episode)
        self.tasks = sorted(self.by_task)
        if len(self.tasks) != 5:
            raise ValueError(f"expected all five tasks, got {self.tasks}")

    def __len__(self): return self.updates

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch); self.epoch += 1
        done, block = 0, 0
        while done < self.updates:
            task = self.tasks[block % len(self.tasks)]
            _task, streams, length = self.by_task[task][rng.randrange(len(self.by_task[task]))]
            team_size = len(streams)
            if self.batch_size % team_size:
                raise ValueError("global batch must be divisible by every 2/3/4-agent team size")
            for _ in range(min(self.block_updates, self.updates - done)):
                batch = []
                for group in range(self.batch_size // team_size):
                    time = rng.randrange(length)
                    batch.extend((stream, time, group) for stream in streams)
                rng.shuffle(batch)
                yield batch; done += 1
            block += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--normalization", default=None,
                        help="existing same-corpus normalization.pt; avoids a redundant multi-GB scan")
    parser.add_argument("--shared-arms", default="0,1,2,3")
    parser.add_argument("--updates", type=int, default=26666)
    parser.add_argument("--batch-size", type=int, default=120,
                        help="global batch divisible by 2/3/4-agent synchronized teams")
    parser.add_argument("--episode-block-updates", type=int, default=64)
    parser.add_argument("--cache-episodes", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--warmup-updates", type=int, default=500)
    parser.add_argument("--beta", type=float, default=1e-3); parser.add_argument("--roles", type=int, default=4)
    parser.add_argument("--role-rank", type=int, default=32)
    parser.add_argument("--distill-weight", type=float, default=.50)
    parser.add_argument("--teacher-reconstruct-weight", type=float, default=.10)
    parser.add_argument("--teacher-relation-weight", type=float, default=.10)
    parser.add_argument("--teacher-usage-weight", type=float, default=.01)
    parser.add_argument("--save-updates", default="26666"); parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260730); parser.add_argument("--allow-preflight", action="store_true")
    args = parser.parse_args()
    if not args.allow_preflight and abs(args.batch_size * args.updates - 3_200_000) > args.batch_size:
        raise ValueError("formal PAIR run must match the 3.2M-sample B40 x 80k budget within one batch")
    if args.batch_size % 4:
        raise ValueError("DataParallel batch must split equally across four GPUs")
    seed_everything(args.seed); torch.backends.cudnn.benchmark = True
    arms = tuple(int(value) for value in args.shared_arms.split(","))
    paths = sorted({path for pattern in args.data.split(",") for path in glob.glob(pattern)})
    trajectories = _trajectories(paths, arms)
    if args.normalization:
        stats = torch.load(args.normalization, map_location="cpu", weights_only=False)["stats"]
    else:
        stats = _stats(trajectories, arms)
    dataset = CompactPAIRWristDataset(trajectories, 100, stats, True, cache_limit=args.cache_episodes)
    sampler = SameEpisodeTeamBlockSampler(dataset, args.batch_size, args.updates, args.episode_block_updates, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)
    sample = dataset[(0, 0, 0)]; state_dim, action_dim = len(sample[2]), len(sample[3][0])
    base = StereoPAIRAdapter(state_dim, action_dim, horizon=100, d_model=384, enc_layers=4,
                             dec_layers=7, roles=args.roles, role_rank=args.role_rank).cuda(0)
    policy = DataParallel(base, device_ids=[0, 1, 2, 3], output_device=0)
    teacher = PAIRActionTeacher(action_dim, roles=args.roles).cuda(0)
    optimizer = torch.optim.AdamW(list(policy.parameters()) + list(teacher.parameters()), lr=args.lr, weight_decay=1e-4)
    def schedule_multiplier(step):
        warmup = min(1.0, (step + 1) / max(args.warmup_updates, 1))
        return warmup * .5 * (1 + math.cos(math.pi * min(1.0, (step + 1) / args.updates)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_multiplier)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"horizon":100,"enc_layers":4,"dec_layers":7,"d_model":384,
        "vision_backbone":"stereo_act_cross_relbias","dino_model":"facebook/dinov3-vitb16-pretrain-lvd1689m",
        "defm_model":base.defm_model_name,"camera_width":640,"camera_height":480,"patch_grid":[30,40],
        "fusion_layers":2,"depth_storage_unit":"millimeters","depth_to_meters_scale":DEPTH_MM_TO_M,
        "arms":arms,"state_dim":state_dim,"action_dim":action_dim,"files":paths,"episodes":len(trajectories),
        "policy_variant":"stereo_pair_adapter","parallelism":"four-GPU PyTorch DataParallel; NCCL-free host fallback",
        "global_batch":args.batch_size,"sample_budget":args.batch_size*args.updates,
        "strict_policy_input":"current local panda_hand wrist RGB-D and local qpos only; no task/agent ID, peer/global/right-camera/language input",
        "training_only_teacher":"permutation-invariant synchronized action-chunk relation teacher; absent at deployment"}
    (out / "config.json").write_text(json.dumps(config, indent=2)); torch.save({"stats":stats},out / "normalization.pt")
    milestones = {int(value) for value in args.save_updates.split(",") if value}; totals = {key:0.0 for key in ("loss","action","kl","distill","teacher_reconstruct","teacher_relation","teacher_usage")}
    for update, batch in enumerate(loader, start=1):
        rgb, depth, qpos, actions, mask, groups = [x.cuda(0, non_blocking=True) for x in batch]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, mu, logvar, _aux, local_roles = policy(rgb.float().div_(255), depth, qpos, actions)
            action_loss = (((prediction-actions).square().mean(-1)*mask).sum()/mask.sum().clamp_min(1))
            kl = -.5*(1+logvar-mu.square()-logvar.exp()).sum(-1).mean()
            teacher_roles, reconstruction, relation, usage = teacher(actions, groups)
            distillation = F.kl_div(local_roles.clamp_min(1e-8).log(), teacher_roles.detach(), reduction="batchmean")
            loss = action_loss + args.beta*kl + args.distill_weight*distillation + args.teacher_reconstruct_weight*reconstruction + args.teacher_relation_weight*relation + args.teacher_usage_weight*usage
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(policy.parameters())+list(teacher.parameters()),1.0)
        optimizer.step(); scheduler.step()
        values = {"loss":loss,"action":action_loss,"kl":kl,"distill":distillation,"teacher_reconstruct":reconstruction,"teacher_relation":relation,"teacher_usage":usage}
        for name,value in values.items(): totals[name] += float(value.detach())
        if update % args.log_every == 0 or update in milestones:
            print(json.dumps({"update":update,"global_batch":args.batch_size,"lr":scheduler.get_last_lr()[0], **{key:value/update for key,value in totals.items()}}),flush=True)
        if update in milestones:
            torch.save({"model":policy.module.state_dict(),"stats":stats,"config":config,"update":update,"pair_teacher":teacher.state_dict()},out/f"checkpoint_{update:06d}.pt")


if __name__ == "__main__":
    main()
