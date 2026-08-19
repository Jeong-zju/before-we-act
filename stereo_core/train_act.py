"""Decentralized local-observation ACT training for RoboFactory tasks."""
import argparse
import copy
import glob
import json
import os
import random
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models import resnet18
try:
    from .five_task_contract import hierarchical_item_weights, task_from_path
except ImportError:
    from five_task_contract import hierarchical_item_weights, task_from_path


# Kept module-global deliberately so a parent process can preload a corpus and
# fork independent CUDA training children.  NumPy image arrays then remain
# read-only copy-on-write pages shared by the children; ordinary one-process
# training has exactly the same semantics as before.
EPISODE_CACHE = OrderedDict()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _trajectories(paths, arms):
    result = []
    for path in paths:
        with h5py.File(path, "r") as f:
            # The shared six-task WAM export stores one episode per HDF5 file
            # under data/{observation,action}.  Normalize it to the historical
            # trajectory tuple so ACT can use the same sampler and loss.
            if "data" in f and "observation" in f["data"]:
                data = f["data"]
                present = tuple(
                    arm for arm in arms
                    if f"agents/panda_{arm}/qpos" in data["observation"]
                    and f"images/agent_{arm}" in data["observation"]
                    and f"agents/panda_{arm}/commanded" in data["action"]
                )
                if present:
                    n = min(
                        len(data["observation"][f"agents/panda_{arm}"]["qpos"])
                        for arm in present
                    )
                    n = min(
                        n,
                        *(len(data["action"][f"agents/panda_{arm}"]["commanded"])
                          for arm in present),
                    )
                    result.append((path, "data", n, present, task_from_path(path)))
                continue
            for key in sorted(f.keys()):
                if key.startswith("traj_"):
                    tr = f[key]
                    # A pooled 2+3-agent corpus deliberately contains both
                    # two-arm and three-arm episodes.  Retain each local policy
                    # stream that actually exists, instead of requiring the
                    # absent panda-2 stream in two-arm demonstrations.
                    present = tuple(
                        arm for arm in arms
                        if f"panda-{arm}" in tr["actions"]
                        and f"panda-{arm}" in tr["obs"]["agent"]
                        and f"head_camera_agent{arm}" in tr["obs"]["sensor_data"]
                    )
                    if not present:
                        continue
                    n = min(
                        len(tr["actions"][f"panda-{arm}"])
                        for arm in present
                    )
                    n = min(
                        n, *(len(tr["obs"]["agent"][f"panda-{arm}"]["qpos"]) for arm in present),
                    )
                    # The source task is retained so mixed-task training can
                    # balance task probability despite unequal arm counts and
                    # episode lengths (notably LongPipelineDelivery).
                    # Canonical sampling label only.  It is never passed to the
                    # policy, and fixes per-seed LPD files being mistaken for
                    # separate tasks by mixed-task samplers.
                    result.append((path, key, n, present, task_from_path(path)))
    return result


def _stats(trajectories, arms):
    qs, acts = [], []
    for path, key, _, present, _ in trajectories:
        with h5py.File(path, "r") as f:
            tr = f[key]
            for arm in present:
                if key == "data":
                    qs.append(np.asarray(tr["observation"]["agents"][f"panda_{arm}"]["qpos"], np.float32))
                    acts.append(np.asarray(tr["action"]["agents"][f"panda_{arm}"]["commanded"], np.float32))
                else:
                    qs.append(np.asarray(tr["obs"]["agent"][f"panda-{arm}"]["qpos"], np.float32))
                    acts.append(np.asarray(tr["actions"][f"panda-{arm}"], np.float32))
    q, a = np.concatenate(qs), np.concatenate(acts)
    return {"q_mean": q.mean(0), "q_std": q.std(0).clip(1e-4),
            "a_mean": a.mean(0), "a_std": a.std(0).clip(1e-4)}


def _stats_from_manifests(paths, arms, stats_root=None):
    """Load the audited task moments without rescanning RGB-heavy HDF5 files."""
    task_stats = {}
    for path in paths:
        task = task_from_path(path)
        if stats_root is not None:
            norm_path = Path(stats_root) / task / "normalization.npz"
        else:
            root = Path(path)
            while root != root.parent and not (root / "normalization.npz").is_file():
                root = root.parent
            norm_path = root / "normalization.npz"
        if task in task_stats or not norm_path.is_file():
            continue
        payload = np.load(norm_path)
        state_mean = np.asarray(payload["state_mean"], np.float32)
        state_std = np.asarray(payload["state_std"], np.float32)
        action_mean = np.asarray(payload["action_mean"], np.float32)
        action_std = np.asarray(payload["action_std"], np.float32)
        q_means, q_stds, a_means, a_stds = [], [], [], []
        for arm in arms:
            state_offset = arm * 18
            action_offset = arm * 8
            if state_offset + 9 > state_mean.size or action_offset + 8 > action_mean.size:
                continue
            q_means.append(state_mean[state_offset:state_offset + 9])
            q_stds.append(state_std[state_offset:state_offset + 9])
            a_means.append(action_mean[action_offset:action_offset + 8])
            a_stds.append(action_std[action_offset:action_offset + 8])
        if q_means and a_means:
            task_stats[task] = {
                "q_mean": np.mean(q_means, axis=0), "q_std": np.maximum(np.mean(q_stds, axis=0), 1e-4),
                "a_mean": np.mean(a_means, axis=0), "a_std": np.maximum(np.mean(a_stds, axis=0), 1e-4),
            }
    if not task_stats:
        raise ValueError("no compatible normalization.npz found for ACT data")
    return {key: np.mean([value[key] for value in task_stats.values()], axis=0).astype(np.float32)
            for key in ("q_mean", "q_std", "a_mean", "a_std")}


class RoboFactoryACTDataset(Dataset):
    def __init__(self, trajectories, arms, horizon, stats, train, *, preload=True, cache_limit=0, windowed_io=False):
        self.arms, self.horizon, self.stats = tuple(arms), horizon, stats
        self.cache_limit = int(cache_limit)
        self.windowed_io = bool(windowed_io)
        # Keep entire episodes together: predictable held-out demonstrations.
        kept = [x for i, x in enumerate(trajectories) if (i % 10 != 0) == train]
        # A/B views of one joint episode always share a split: no paired leakage.
        self.items = [
            (p, k, t, arm, task)
            for p, k, n, present, task in kept
            for arm in present
            for t in range(n)
        ]
        self.item_tasks = [task for _, _, _, _, task in self.items]
        self.item_weights = hierarchical_item_weights(kept, self.items)
        self.stream_indices = defaultdict(list)
        for index, (path, key, _, arm, task) in enumerate(self.items):
            self.stream_indices[(path, key, arm, task)].append(index)
        # This is deliberately RAM-resident. With random ACT batches, lazy episode
        # caching repeatedly decompresses 20+ MB RGB trajectories for one frame.
        # The host has 192 GB RAM; keeping this single-task corpus in memory turns
        # that I/O bottleneck into continuous GPU training.
        self.cache = EPISODE_CACHE
        if preload:
            for path, key, _, present, _ in kept:
                for arm in present:
                    self._episode(path, key, arm)

    def __len__(self): return len(self.items)

    def _episode(self, path, key, arm):
        tag = (path, key, arm)
        if self.windowed_io:
            return None, None, None
        if tag not in self.cache:
            with h5py.File(path, "r") as f:
                tr = f[key]
                if key == "data":
                    cam = tr["observation"]["images"][f"agent_{arm}"][:]
                    qpos = tr["observation"]["agents"][f"panda_{arm}"]["qpos"][:]
                    actions = tr["action"]["agents"][f"panda_{arm}"]["commanded"][:].astype(np.float32)
                else:
                    cam = tr["obs"]["sensor_data"][f"head_camera_agent{arm}"]["rgb"][:]
                    qpos = tr["obs"]["agent"][f"panda-{arm}"]["qpos"][:]
                    actions = tr["actions"][f"panda-{arm}"][:].astype(np.float32)
            self.cache[tag] = (cam, qpos.astype(np.float32), actions)
            if self.cache_limit > 0:
                while len(self.cache) > self.cache_limit:
                    self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(tag)
        return self.cache[tag]

    def __getitem__(self, idx):
        path, key, t, arm, _ = self.items[idx]
        if self.windowed_io:
            with h5py.File(path, "r") as f:
                tr = f[key]
                if key == "data":
                    image = tr["observation"]["images"][f"agent_{arm}"][t]
                    qpos = tr["observation"]["agents"][f"panda_{arm}"]["qpos"][t]
                    actions = tr["action"]["agents"][f"panda_{arm}"]["commanded"][t:t + self.horizon].astype(np.float32)
                else:
                    image = tr["obs"]["sensor_data"][f"head_camera_agent{arm}"]["rgb"][t]
                    qpos = tr["obs"]["agent"][f"panda-{arm}"]["qpos"][t]
                    actions = tr["actions"][f"panda-{arm}"][t:t + self.horizon].astype(np.float32)
        else:
            image, qpos, actions = self._episode(path, key, arm)
        # RGB observations are local to this arm only; no ID or global view.
        # Keep the camera frame as uint8 until it reaches the GPU. Per-sample
        # CPU resizing starves two 5090s; batched resize is done in _loss.
        im = torch.from_numpy(image[t]).permute(2, 0, 1).contiguous()
        q = (qpos[t] - self.stats["q_mean"]) / self.stats["q_std"]
        future = actions[t:t + self.horizon]
        valid = len(future)
        padded = np.empty((self.horizon, actions.shape[1]), np.float32)
        padded[:valid] = future
        padded[valid:] = future[-1]
        padded = (padded - self.stats["a_mean"]) / self.stats["a_std"]
        mask = np.zeros(self.horizon, np.bool_); mask[:valid] = True
        return im, torch.from_numpy(q), torch.from_numpy(padded), torch.from_numpy(mask)


class EpisodeBlockBatchSampler(Sampler):
    """Task-balanced local episode blocks for bounded RGB caching."""
    def __init__(self, dataset, batch_size, updates, block_updates, seed, task_balanced):
        if batch_size % 4:
            raise ValueError("episode-block batching requires batch size divisible by 4")
        self.batch_size = batch_size
        self.updates = updates
        self.block_updates = block_updates
        self.seed = seed
        self.epoch = 0
        self.per_stream = batch_size // 4
        # Preserve the requested hierarchy: task -> demonstration -> local
        # arm -> time.  Flat stream sampling would over-represent a four-arm
        # demonstration simply because it contributes four streams.
        self.by_task_episode = defaultdict(lambda: defaultdict(list))
        for (path, key, arm, task), indices in dataset.stream_indices.items():
            self.by_task_episode[task][(path, key)].append(indices)
        self.tasks = sorted(self.by_task_episode)
        self.task_balanced = task_balanced
        self.all_episodes = [streams for episodes in self.by_task_episode.values() for streams in episodes.values()]

    def __len__(self):
        return self.updates

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        produced = 0
        while produced < self.updates:
            episodes = self.all_episodes
            if self.task_balanced:
                task = self.tasks[rng.randrange(len(self.tasks))]
                episodes = list(self.by_task_episode[task].values())
            # Choose a demonstration first and then an arm uniformly within it.
            streams = [episode[rng.randrange(len(episode))]
                       for episode in (episodes[rng.randrange(len(episodes))] for _ in range(4))]
            for _ in range(min(self.block_updates, self.updates - produced)):
                batch = [
                    stream[rng.randrange(len(stream))]
                    for stream in streams
                    for _ in range(self.per_stream)
                ]
                rng.shuffle(batch)
                yield batch
                produced += 1


class ACT(nn.Module):
    """ACT-style CVAE with independently configurable action encoder/decoder."""
    def __init__(self, state_dim, action_dim, horizon=100, d_model=384, enc_layers=4, dec_layers=7,
                 latent_dim=32, vision_backbone="resnet18",
                 dino_model="facebook/dinov3-vitb16-pretrain-lvd1689m"):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.dino_model = dino_model
        if vision_backbone == "resnet18":
            backbone = resnet18(weights=None)
            self.vision = nn.Sequential(*list(backbone.children())[:-2])
            self.vision_proj = nn.Conv2d(512, d_model, 1)
        elif vision_backbone == "dinov3_vitb16_frozen":
            # DINOv3 is intentionally a frozen visual head: only the ACT
            # projection/transformers/policy layers receive gradients.
            from transformers import AutoImageProcessor, AutoModel
            token = os.environ.get("HF_TOKEN")
            processor = AutoImageProcessor.from_pretrained(dino_model, token=token)
            self.vision = AutoModel.from_pretrained(dino_model, token=token)
            self.vision.requires_grad_(False)
            self.vision.eval()
            self.register_buffer("dino_mean", torch.tensor(processor.image_mean).view(1, -1, 1, 1))
            self.register_buffer("dino_std", torch.tensor(processor.image_std).view(1, -1, 1, 1))
            self.vision_proj = nn.Linear(self.vision.config.hidden_size, d_model)
        else:
            raise ValueError(f"unknown vision backbone: {vision_backbone}")
        self.state = nn.Sequential(nn.Linear(state_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.action = nn.Linear(action_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, horizon, d_model) * .02)
        self.query = nn.Parameter(torch.randn(1, horizon, d_model) * .02)
        enc = nn.TransformerEncoderLayer(d_model, 8, d_model * 4, dropout=.1,
                                         batch_first=True, norm_first=True, activation="gelu")
        dec = nn.TransformerDecoderLayer(d_model, 8, d_model * 4, dropout=.1,
                                         batch_first=True, norm_first=True, activation="gelu")
        self.posterior = nn.TransformerEncoder(enc, num_layers=enc_layers)
        self.decoder = nn.TransformerDecoder(dec, num_layers=dec_layers)
        self.latent = nn.Linear(d_model, latent_dim * 2)
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.out = nn.Linear(d_model, action_dim)
        self.horizon = horizon

    def _vision_tokens(self, image):
        if self.vision_backbone == "resnet18":
            image = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)
            return self.vision_proj(self.vision(image)).flatten(2).transpose(1, 2)
        # 640×480 is retained natively: both dimensions are divisible by the
        # DINOv3 ViT-B/16 patch size, yielding a 40×30 local-image token grid.
        if tuple(image.shape[-2:]) != (480, 640):
            raise ValueError(f"strict 640x480 protocol required, got {tuple(image.shape[-2:])}")
        image = (image - self.dino_mean) / self.dino_std
        self.vision.eval()  # model.train() must never enable frozen-head dropout
        with torch.no_grad():
            all_tokens = self.vision(pixel_values=image).last_hidden_state
            # Drop CLS and DINO register tokens; retain exactly the 40×30
            # patch grid produced by an unresized 640×480 ViT-B/16 image.
            first_patch = 1 + int(getattr(self.vision.config, "num_register_tokens", 0))
            tokens = all_tokens[:, first_patch:]
        if tokens.shape[1] != 30 * 40:
            raise ValueError(f"strict 30x40 DINO grid required, got {tokens.shape[1]} tokens")
        return self.vision_proj(tokens)

    def forward(self, image, qpos, actions=None):
        x = self._vision_tokens(image)
        state = self.state(qpos).unsqueeze(1)
        if actions is not None:
            h = self.posterior(self.action(actions) + self.pos)
            mu, logvar = self.latent(h.mean(1)).chunk(2, -1)
            z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar)
        else:
            mu = logvar = None
            z = torch.zeros((image.shape[0], self.z_proj.in_features), device=image.device)
        memory = torch.cat((state, self.z_proj(z).unsqueeze(1), x), dim=1)
        pred = self.out(self.decoder(self.query.expand(image.shape[0], -1, -1), memory))
        return pred, mu, logvar


def _loss(model, image, qpos, actions, mask, beta):
    """Return differentiable total loss plus reporting tensors for one microbatch."""
    image = image.float().div_(255)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred, mu, logvar = model(image, qpos, actions)
        mse = ((pred - actions).square().mean(-1) * mask).sum() / mask.sum().clamp_min(1)
        kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
    return mse + beta * kl, mse, kl


def _sync_to_replica(master, replica, replica_device):
    """Copy the one authoritative shared-policy state to the second GPU."""
    with torch.no_grad():
        for src, dst in zip(master.parameters(), replica.parameters()):
            dst.copy_(src.to(replica_device))
        for src, dst in zip(master.buffers(), replica.buffers()):
            dst.copy_(src.to(replica_device))


def _aggregate_replica_grads(master, replica, master_device):
    """Sum already globally weighted replica gradients without NCCL."""
    with torch.no_grad():
        for p0, p1 in zip(master.parameters(), replica.parameters()):
            if p0.grad is None:
                p0.grad = p1.grad.to(master_device).clone()
            elif p1.grad is not None:
                p0.grad.add_(p1.grad.to(master_device))
        # Keep BatchNorm running statistics representative of both local halves.
        for b0, b1 in zip(master.buffers(), replica.buffers()):
            if b0.is_floating_point():
                b0.add_(b1.to(master_device)).mul_(0.5)


def epoch(model, loader, opt, device, beta, max_updates=None, scheduler=None, replica=None, replica_device=None):
    training = opt is not None
    model.train(training)
    total = {"loss": 0., "mse": 0., "kl": 0., "n": 0}
    ctx = torch.enable_grad if training else torch.no_grad
    updates = 0
    with ctx():
        for image, qpos, actions, mask in loader:
            # A second replica receives the other half of the *global* batch.
            # This is manual synchronous data parallelism, needed because this
            # Vast host cannot bootstrap NCCL even though both CUDA devices work.
            use_replica = replica is not None and image.shape[0] >= 2
            if use_replica:
                split = image.shape[0] // 2
                first = tuple(x[:split].to(device, non_blocking=True) for x in (image, qpos, actions, mask))
                second = tuple(x[split:].to(replica_device, non_blocking=True) for x in (image, qpos, actions, mask))
                loss0, mse0, kl0 = _loss(model, *first, beta)
                loss1, mse1, kl1 = _loss(replica, *second, beta)
                action_weight0 = float(first[3].sum().item())
                action_weight1 = float(second[3].sum().item())
                action_total = max(action_weight0 + action_weight1, 1.)
                sample_total = float(image.shape[0])
                # Weight gradients exactly as the loss over the unsharded batch.
                # Metrics live on the master GPU only. The actual backward below
                # stays separate on each GPU and never needs a cross-device graph.
                mse = mse0.detach() * (action_weight0 / action_total) + mse1.detach().to(device) * (action_weight1 / action_total)
                kl = kl0.detach() * (first[0].shape[0] / sample_total) + kl1.detach().to(device) * (second[0].shape[0] / sample_total)
                loss = mse + beta * kl
                n = image.shape[0]
            else:
                image, qpos, actions, mask = (x.to(device, non_blocking=True) for x in (image, qpos, actions, mask))
                loss, mse, kl = _loss(model, image, qpos, actions, mask, beta)
                n = image.shape[0]
            if training:
                opt.zero_grad(set_to_none=True)
                if use_replica:
                    for p in replica.parameters(): p.grad = None
                    # Backpropagate weighted local terms before averaging.
                    loss0w = mse0 * (action_weight0 / action_total) + beta * kl0 * (first[0].shape[0] / sample_total)
                    loss1w = mse1 * (action_weight1 / action_total) + beta * kl1 * (second[0].shape[0] / sample_total)
                    loss0w.backward(); loss1w.backward()
                    _aggregate_replica_grads(model, replica, device)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                if use_replica: _sync_to_replica(model, replica, replica_device)
                if scheduler is not None: scheduler.step()
                updates += 1
            for k, v in (("loss", loss), ("mse", mse), ("kl", kl)):
                total[k] += float(v.detach()) * n
            total["n"] += n
            if training and max_updates is not None and updates >= max_updates:
                break
    return ({k: v / total["n"] for k, v in total.items() if k != "n"}, updates)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Glob or comma-separated HDF5 paths")
    p.add_argument("--arm", type=int, choices=(0, 1), help="single-arm ablation only")
    p.add_argument("--shared", action="store_true", help="pool listed agents' local data into one shared policy")
    p.add_argument("--shared-arms", default="0,1", help="comma-separated agents for --shared, e.g. 0,1,2")
    p.add_argument("--devices", default="0", help="one shared DataParallel model, e.g. 0,1")
    p.add_argument("--output", required=True)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--enc-layers", type=int, default=4)
    p.add_argument("--dec-layers", type=int, default=7)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--vision-backbone", choices=("resnet18", "dinov3_vitb16_frozen"), default="resnet18")
    p.add_argument("--dino-model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    p.add_argument("--camera-width", type=int, default=320)
    p.add_argument("--camera-height", type=int, default=240)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--updates", type=int, default=60000)
    p.add_argument("--save-updates", default="20000,40000,60000",
                   help="comma-separated exact optimizer-update checkpoints")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lazy-cache-episodes", type=int, default=0,
                   help="Bound RGB cache and use episode-block batches (0 keeps full preloading).")
    p.add_argument("--episode-block-updates", type=int, default=64,
                   help="Updates reusing four local streams in lazy-cache mode.")
    p.add_argument("--task-balanced", action="store_true",
                   help="Sample each source task equally in mixed-task training.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--stats-root", default=None,
                   help="Root containing audited per-task normalization.npz files")
    p.add_argument("--windowed-io", action="store_true",
                   help="Read only sampled RGB frames instead of caching whole episodes")
    a = p.parse_args()
    assert a.shared != (a.arm is not None), "set exactly one of --shared or --arm"
    arms = tuple(int(x) for x in a.shared_arms.split(",")) if a.shared else (a.arm,)
    assert arms and len(set(arms)) == len(arms) and all(x >= 0 for x in arms)
    seed_everything(a.seed); torch.backends.cudnn.benchmark = True
    device_ids = [int(x) for x in a.devices.split(",")]
    device = torch.device(f"cuda:{device_ids[0]}")
    paths = sorted({p for item in a.data.split(",") for p in glob.glob(item)})
    assert paths, f"no HDF5 files match {a.data}"
    tr = _trajectories(paths, arms); assert len(tr) >= 10, "need at least 10 successful demonstrations"
    stats = (_stats_from_manifests(paths, arms, a.stats_root) if a.stats_root is not None else _stats(tr, arms))
    lazy_cache = a.lazy_cache_episodes > 0
    if lazy_cache and a.workers:
        raise ValueError("lazy RGB cache requires --workers 0")
    train = RoboFactoryACTDataset(tr, arms, a.horizon, stats, True,
                                  preload=not lazy_cache, cache_limit=a.lazy_cache_episodes,
                                  windowed_io=a.windowed_io)
    valid = RoboFactoryACTDataset(tr, arms, a.horizon, stats, False,
                                  preload=not lazy_cache, cache_limit=a.lazy_cache_episodes,
                                  windowed_io=a.windowed_io)
    kwargs = dict(batch_size=a.batch_size, num_workers=a.workers, pin_memory=True,
                  persistent_workers=a.workers > 0)
    # Forking workers after CUDA is initialized can deadlock (and h5py is not
    # fork-friendly either). Spawn keeps the two GPU training jobs independent.
    if a.workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
    sampler = None
    if lazy_cache:
        sampler = EpisodeBlockBatchSampler(train, a.batch_size, a.updates,
                                           a.episode_block_updates, a.seed, a.task_balanced)
        train_loader = DataLoader(train, batch_sampler=sampler, num_workers=0, pin_memory=True)
    elif a.task_balanced:
        counts = Counter(train.item_tasks)
        if len(counts) > 1:
            weights = torch.as_tensor(train.item_weights, dtype=torch.double)
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    if not lazy_cache:
        train_loader = DataLoader(train, shuffle=sampler is None, sampler=sampler, drop_last=True, **kwargs)
    val_loader = DataLoader(valid, shuffle=False, **kwargs)
    sample = train[0]
    if tuple(sample[0].shape[-2:]) != (a.camera_height, a.camera_width):
        raise ValueError(f"dataset frame {tuple(sample[0].shape[-2:])} does not match requested "
                         f"{a.camera_height}x{a.camera_width}")
    model = ACT(len(sample[1]), len(sample[2][0]), a.horizon, a.d_model, a.enc_layers, a.dec_layers,
                vision_backbone=a.vision_backbone, dino_model=a.dino_model).to(device)
    replica = None; replica_device = None
    if len(device_ids) > 1:
        assert len(device_ids) == 2, "manual synchronous mode currently supports exactly two GPUs"
        replica_device = torch.device(f"cuda:{device_ids[1]}")
        replica = copy.deepcopy(model).to(replica_device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.updates)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "normalization.npz", **stats)
    best = float("inf")
    info = vars(a) | {"arms": arms, "files": paths, "episodes": len(tr), "train_steps": len(train), "val_steps": len(valid),
                      "train_task_item_counts": dict(Counter(train.item_tasks)),
                      "state_dim": len(sample[1]), "action_dim": len(sample[2][0])}
    (out / "config.json").write_text(json.dumps(info, indent=2))
    milestones = {int(x) for x in a.save_updates.split(",") if x}
    updates = 0; e = 0
    while updates < a.updates:
        e += 1
        next_stop = min([a.updates] + [m for m in milestones if m > updates])
        train_metrics, ran = epoch(model, train_loader, opt, device, a.beta, next_stop - updates, sched, replica, replica_device)
        updates += ran
        val_metrics, _ = epoch(model, val_loader, None, device, a.beta, replica=replica, replica_device=replica_device)
        report = {"epoch": e, "updates": updates, "lr": sched.get_last_lr()[0], "train": train_metrics, "val": val_metrics}
        print(json.dumps(report), flush=True)
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": e,
                 "updates": updates, "stats": stats, "config": info}
        torch.save(state, out / "last.pt")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]; torch.save(state, out / "best.pt")
        if updates in milestones:
            torch.save(state, out / f"checkpoint_{updates:06d}.pt")


if __name__ == "__main__": main()
