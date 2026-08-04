"""Wrist-only RGB-D Stereo-ACT-cross_relbias trainer for RoboFactory.

One shared policy sees only one local Panda wrist RGB-D stream at a time. RGB
uses frozen DINOv3-B/16; native ManiSkill depth (stored in millimetres) uses
frozen DeFM-S/14. Both become 30x40 patches before two region-aligned
cross_relbias fusion blocks and the standard ACT 4/7 CVAE policy.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from .rgbd_patch_fusion import RGBDPatchFusion
    from .train_act import ACT, EpisodeBlockBatchSampler, _stats, _trajectories, seed_everything
    from .five_task_contract import hierarchical_item_weights
except ImportError:
    from rgbd_patch_fusion import RGBDPatchFusion
    from train_act import ACT, EpisodeBlockBatchSampler, _stats, _trajectories, seed_everything
    from five_task_contract import hierarchical_item_weights


DEPTH_MM_TO_M = 0.001


class WristRGBDACTDataset(Dataset):
    def __init__(self, trajectories, horizon, stats, train, *, preload=True, cache_limit=0):
        self.horizon, self.stats, self.cache_limit = horizon, stats, int(cache_limit)
        kept = [item for index, item in enumerate(trajectories) if (index % 10 != 0) == train]
        self.items = [(path, key, t, arm, task) for path, key, n, present, task in kept
                      for arm in present for t in range(n)]
        self.item_tasks = [task for _, _, _, _, task in self.items]
        self.item_weights = hierarchical_item_weights(kept, self.items)
        self.stream_indices = defaultdict(list)
        for index, (path, key, _t, arm, task) in enumerate(self.items):
            self.stream_indices[(path, key, arm, task)].append(index)
        self.cache = OrderedDict()
        if preload:
            for path, key, _, present, _ in kept:
                for arm in present:
                    self._episode(path, key, arm)

    def __len__(self):
        return len(self.items)

    def _episode(self, path, key, arm):
        tag = (path, key, arm)
        if tag not in self.cache:
            with h5py.File(path, "r") as h5:
                tr = h5[key]
                sensor = tr["obs"]["sensor_data"][f"head_camera_agent{arm}"]
                if "depth" not in sensor:
                    raise ValueError(f"Stereo-ACT needs rgbd corpus; missing depth in {path}:{key}")
                rgb, depth = sensor["rgb"][:], sensor["depth"][:]
                if tuple(rgb.shape[1:]) != (480, 640, 3) or tuple(depth.shape[1:]) != (480, 640, 1):
                    raise ValueError(
                        f"strict 640x480 RGB-D required; {path}:{key}:panda-{arm} has "
                        f"rgb={tuple(rgb.shape[1:])}, depth={tuple(depth.shape[1:])}"
                    )
                self.cache[tag] = (
                    rgb, depth,
                    tr["obs"]["agent"][f"panda-{arm}"]["qpos"][:].astype(np.float32),
                    tr["actions"][f"panda-{arm}"][:].astype(np.float32),
                )
            if self.cache_limit > 0:
                while len(self.cache) > self.cache_limit:
                    self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(tag)
        return self.cache[tag]

    def __getitem__(self, index):
        path, key, t, arm, _ = self.items[index]
        rgb, depth, qpos, actions = self._episode(path, key, arm)
        future = actions[t:t + self.horizon]
        valid = len(future)
        padded = np.empty((self.horizon, actions.shape[1]), np.float32)
        padded[:valid], padded[valid:] = future, future[-1]
        mask = np.zeros(self.horizon, np.bool_); mask[:valid] = True
        return (
            torch.from_numpy(rgb[t]).permute(2, 0, 1).contiguous(),
            torch.from_numpy(depth[t]).permute(2, 0, 1).contiguous(),
            torch.from_numpy((qpos[t] - self.stats["q_mean"]) / self.stats["q_std"]),
            torch.from_numpy((padded - self.stats["a_mean"]) / self.stats["a_std"]),
            torch.from_numpy(mask),
        )


class StereoACT(ACT):
    def __init__(self, state_dim, action_dim, horizon=100, d_model=384, enc_layers=4, dec_layers=7,
                 dino_model="facebook/dinov3-vitb16-pretrain-lvd1689m", defm_model="defm_vit_s14"):
        super().__init__(state_dim, action_dim, horizon, d_model, enc_layers, dec_layers,
                         vision_backbone="dinov3_vitb16_frozen", dino_model=dino_model)
        from defm.model_factory import create_defm_model
        self.defm_model_name = defm_model
        # Use an explicitly versioned local checkpoint when supplied.  This
        # prevents an interrupted Hub download from silently changing the
        # depth encoder during a formal run.
        checkpoint = os.environ.get("DEFM_CHECKPOINT")
        self.defm = create_defm_model(
            defm_model, pretrained=True, pretrained_path=checkpoint
        ).eval()
        self.defm.requires_grad_(False)
        self.depth_proj = nn.Linear(384, d_model)
        self.fusion = RGBDPatchFusion(d_model=d_model, heads=8, grid_h=30, grid_w=40,
                                      layers=2, ffn_dim=d_model * 4)
        self.fusion_pos = nn.Parameter(torch.randn(1, 30 * 40, d_model) * 0.02)

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval(); self.defm.eval()
        return self

    def _rgbd_tokens(self, rgb, depth_mm):
        # Parent RGB method preserves native 640x480 -> 30x40 DINO patches.
        rgb_tokens = super()._vision_tokens(rgb)
        from defm.utils.utils import preprocess_depth_batch
        depth_m = depth_mm.float().squeeze(1).mul(DEPTH_MM_TO_M)
        prepared = preprocess_depth_batch(depth_m, target_size=(420, 560), patch_size=14, device=rgb.device)
        self.defm.eval()
        with torch.no_grad():
            spatial, _ = self.defm.to(rgb.device).get_intermediate_layers(
                prepared.float(), n=1, reshape=True, return_class_token=True
            )[0]
        depth_tokens = self.depth_proj(spatial.flatten(2).transpose(1, 2).to(dtype=rgb_tokens.dtype))
        if rgb_tokens.shape[1] != 30 * 40 or depth_tokens.shape[1] != 30 * 40:
            raise ValueError(
                f"strict aligned 30x40 tokens required, got RGB={rgb_tokens.shape[1]} "
                f"and depth={depth_tokens.shape[1]}"
            )
        return self.fusion(rgb_tokens, depth_tokens, self.fusion_pos.to(dtype=rgb_tokens.dtype))

    def forward(self, image, depth_mm, qpos, actions=None):
        x = self._rgbd_tokens(image, depth_mm)
        state = self.state(qpos).unsqueeze(1)
        if actions is not None:
            h = self.posterior(self.action(actions) + self.pos)
            mu, logvar = self.latent(h.mean(1)).chunk(2, -1)
            z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar)
        else:
            mu = logvar = None
            z = torch.zeros((image.shape[0], self.z_proj.in_features), device=image.device)
        memory = torch.cat((state, self.z_proj(z).unsqueeze(1), x), dim=1)
        return self.out(self.decoder(self.query.expand(image.shape[0], -1, -1), memory)), mu, logvar


def loss(model, rgb, depth, qpos, actions, mask, beta):
    rgb = rgb.float().div_(255)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred, mu, logvar = model(rgb, depth, qpos, actions)
        mse = ((pred - actions).square().mean(-1) * mask).sum() / mask.sum().clamp_min(1)
        kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
    return mse + beta * kl, mse, kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--shared-arms", default="0,1")
    parser.add_argument("--output", required=True)
    # Verified on the target 32 GB RTX 5090 with a real 640×480 RGB-D canary:
    # batch=32 completes DINO+DeFM plus two 30×40 fusion blocks' backward pass.
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates", type=int, default=60000)
    parser.add_argument("--save-updates", default="20000,40000,60000")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-episodes", type=int, default=0,
                        help="Bound decoded episode cache; 0 preloads all trajectories.")
    parser.add_argument("--episode-block-updates", type=int, default=64)
    parser.add_argument("--task-balanced", action="store_true")
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    import glob
    arms = tuple(int(item) for item in args.shared_arms.split(","))
    paths = sorted({path for item in args.data.split(",") for path in glob.glob(item)})
    trajectories = _trajectories(paths, arms)
    if len(trajectories) < 10:
        raise ValueError("need at least 10 successful RGB-D demonstrations")
    seed_everything(args.seed); torch.backends.cudnn.benchmark = True
    stats = _stats(trajectories, arms)
    lazy_cache = args.cache_episodes > 0
    if lazy_cache and args.workers:
        raise ValueError("bounded RGB-D cache requires --workers 0 to keep a single cache")
    train = WristRGBDACTDataset(trajectories, 100, stats, True, preload=not lazy_cache,
                                cache_limit=args.cache_episodes)
    valid = WristRGBDACTDataset(trajectories, 100, stats, False, preload=False,
                                cache_limit=args.cache_episodes)
    counts = Counter(train.item_tasks)
    sampler = None
    if lazy_cache:
        sampler = EpisodeBlockBatchSampler(train, args.batch_size, args.updates,
                                           args.episode_block_updates, args.seed, args.task_balanced)
        loader = DataLoader(train, batch_sampler=sampler, num_workers=0, pin_memory=True)
    elif args.task_balanced and len(counts) > 1:
        weights = torch.as_tensor(train.item_weights, dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    if not lazy_cache:
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                            drop_last=True, num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)
    device = torch.device("cuda:0")
    sample = train[0]
    model = StereoACT(len(sample[2]), len(sample[3][0])).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.updates)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"horizon": 100, "enc_layers": 4, "dec_layers": 7, "d_model": 384,
        "vision_backbone": "stereo_act_cross_relbias",
        "dino_model": "facebook/dinov3-vitb16-pretrain-lvd1689m", "defm_model": model.defm_model_name,
        "defm_checkpoint": os.environ.get("DEFM_CHECKPOINT"),
        "camera_width": 640, "camera_height": 480, "patch_grid": [30, 40], "fusion_layers": 2,
        "depth_storage_unit": "millimeters", "depth_to_meters_scale": DEPTH_MM_TO_M,
        "arms": arms, "state_dim": len(sample[2]), "action_dim": len(sample[3][0]),
        "files": paths, "episodes": len(trajectories), "train_task_item_counts": dict(counts)}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    np.savez(output / "normalization.npz", **stats)
    milestones, updates = {int(x) for x in args.save_updates.split(",") if x}, 0
    while updates < args.updates:
        model.train(); totals = {"loss": 0.0, "mse": 0.0, "kl": 0.0, "n": 0}
        for rgb, depth, qpos, actions, mask in loader:
            rgb, depth, qpos, actions, mask = (item.to(device, non_blocking=True) for item in (rgb, depth, qpos, actions, mask))
            optimizer.zero_grad(set_to_none=True)
            total, mse, kl = loss(model, rgb, depth, qpos, actions, mask, args.beta)
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); updates += 1
            batch = len(rgb)
            for name, value in (("loss", total), ("mse", mse), ("kl", kl)):
                totals[name] += float(value.detach()) * batch
            totals["n"] += batch
            if updates in milestones:
                torch.save({"model": model.state_dict(), "stats": stats, "config": config, "update": updates},
                           output / f"checkpoint_{updates:06d}.pt")
            if updates % 100 == 0:
                print(json.dumps({"update": updates, **{key: value / totals["n"] for key, value in totals.items() if key != "n"}}), flush=True)
            if updates >= args.updates:
                break


if __name__ == "__main__":
    main()
