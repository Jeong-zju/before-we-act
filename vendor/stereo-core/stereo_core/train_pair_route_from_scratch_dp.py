"""Three-GPU from-scratch PAIR-Route formal trainer.

The deployed policy is strictly local. Synchronized team actions are used only
during training to supervise directed relations and expert capability. Global
B120 x 26,667 matches the B40 x 80k baseline sample budget.
"""
from __future__ import annotations

import argparse, glob, json, math, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from pair_route_model import StereoPAIRRoute
from role_observability_teacher import TeamRoleTeacher
from train_act import _trajectories, seed_everything
from train_pair_dataparallel import CompactPAIRWristDataset, SameEpisodeTeamBlockSampler
from train_stereo_act import DEPTH_MM_TO_M


def teacher_edges(teacher, actions, groups):
    unique = groups.unique(sorted=True)
    packed = actions.new_zeros((len(unique), 4, actions.shape[1], actions.shape[2]))
    mask = torch.zeros((len(unique), 4), dtype=torch.bool, device=actions.device)
    locations = []
    for gi, group in enumerate(unique):
        ids = (groups == group).nonzero(as_tuple=False).flatten()
        packed[gi, :len(ids)] = actions.index_select(0, ids)
        mask[gi, :len(ids)] = True; locations.append(ids)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _score, roles = teacher(packed, mask)
        q, k = teacher.q(roles), teacher.k(roles)
        edges = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(q.shape[-1])
    return [(ids, edges[i, :len(ids), :len(ids)].float())
            for i, ids in enumerate(locations)]


def relation_loss(routes, compatibility, targets):
    pooled = routes.mean(1); losses = []
    for ids, target in targets:
        p = pooled.index_select(0, ids)
        local = torch.einsum("ir,rs,js->ij", p, compatibility, p)
        valid = ~torch.eye(len(ids), dtype=torch.bool, device=local.device)
        losses.append(F.binary_cross_entropy_with_logits(local[valid], target.sigmoid()[valid]))
    return torch.stack(losses).mean()


def fingerprint(cf):
    # E x (mean action, action spread, mean temporal effect)
    mean = cf.mean((0, 1)).transpose(0, 1)
    spread = cf.float().std((0, 1)).transpose(0, 1).to(cf.dtype)
    velocity = (cf[:, 1:] - cf[:, :-1]).abs().mean((0, 1)).transpose(0, 1)
    return torch.cat((mean, spread, velocity), -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--normalization", required=True)
    p.add_argument("--teacher", required=True); p.add_argument("--output", required=True)
    p.add_argument("--updates", type=int, default=26667); p.add_argument("--batch-size", type=int, default=120)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--router-lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500); p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--relation-weight", type=float, default=.05); p.add_argument("--capability-weight", type=float, default=.05)
    p.add_argument("--specialization-weight", type=float, default=.01); p.add_argument("--anchor-weight", type=float, default=.02)
    p.add_argument("--counterfactual-every", type=int, default=4); p.add_argument("--discovery-fraction", type=float, default=.30)
    p.add_argument("--cache-episodes", type=int, default=48); p.add_argument("--block-updates", type=int, default=64)
    p.add_argument("--save-updates", default="8000,16000,26667"); p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260801); p.add_argument("--allow-preflight", action="store_true")
    args = p.parse_args()
    if not args.allow_preflight and abs(args.batch_size * args.updates - 3_200_000) > args.batch_size:
        raise ValueError("formal run must match the B40 x 80k = 3.2M sample budget")
    if args.batch_size % 12: raise ValueError("batch must divide all 2/3/4-agent teams and three GPUs")
    seed_everything(args.seed); torch.backends.cudnn.benchmark = True
    saved = torch.load(args.normalization, map_location="cpu", weights_only=False); stats = saved["stats"]
    paths = sorted({x for pattern in args.data.split(",") for x in glob.glob(pattern)})
    trajectories = _trajectories(paths, (0, 1, 2, 3))
    dataset = CompactPAIRWristDataset(trajectories, 100, stats, True, args.cache_episodes)
    sampler = SameEpisodeTeamBlockSampler(dataset, args.batch_size, args.updates,
                                           args.block_updates, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)
    sample = dataset[(0, 0, 0)]; state_dim, action_dim = len(sample[2]), len(sample[3][0])
    base = StereoPAIRRoute(state_dim, action_dim, horizon=100, d_model=384,
                           enc_layers=4, dec_layers=7, roles=4, role_rank=32).cuda(0)
    policy = DataParallel(base, device_ids=[0, 1, 2], output_device=0)
    teacher_saved = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher = TeamRoleTeacher(action_dim).cuda(0)
    teacher.load_state_dict(teacher_saved["model"]); teacher.eval().requires_grad_(False)
    router_prefix = ("compatibility", "role_prototypes", "route_state", "route_observation", "route_mlp")
    router, body = [], []
    for name, parameter in base.named_parameters():
        (router if name.startswith(router_prefix) else body).append(parameter)
    optimizer = torch.optim.AdamW([{"params": body, "lr": args.lr},
                                   {"params": router, "lr": args.router_lr}], weight_decay=1e-4)
    def mult(step):
        warm = min(1., (step + 1) / max(args.warmup, 1))
        return warm * .5 * (1 + math.cos(math.pi * min(1., (step + 1) / args.updates)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, mult)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    discovery = int(args.updates * args.discovery_fraction); milestones = {int(x) for x in args.save_updates.split(",") if x}
    config = vars(args) | {"policy_variant":"stereo_pair_route", "training":"from scratch",
        "parallelism":"three-GPU DataParallel on physical GPU1-3", "global_batch":args.batch_size,
        "sample_budget":args.batch_size*args.updates, "horizon":100, "enc_layers":4, "dec_layers":7,
        "camera_width":640, "camera_height":480, "patch_grid":[30,40], "depth_storage_unit":"millimeters",
        "depth_to_meters_scale":DEPTH_MM_TO_M,
        "strict_policy_input":"single local panda_hand wrist RGB-D and own qpos; no ID, language, communication, global/peer/right-camera",
        "training_only_signal":"synchronized action relation teacher + real-target counterfactual expert error",
        "specialization":"low per-sample entropy, high batch diversity, weak starvation floor",
        "capability_basis":"discover first 30%, then anchor action-effect fingerprints"}
    (out/"config.json").write_text(json.dumps(config, indent=2)); torch.save({"stats":stats}, out/"normalization.pt")
    capability_ema = None; capability_ref = None; started = time.time(); last = {}
    for update, batch in enumerate(loader, 1):
        rgb, depth, qpos, actions, mask, groups = [x.cuda(0, non_blocking=True) for x in batch]
        do_cf = update % args.counterfactual_every == 0
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, mu, logvar, _aux, routes, cf, cf_target = policy(
                rgb.float().div_(255), depth, qpos, actions, True, do_cf)
            action = (((pred-actions).square().mean(-1)*mask).sum()/mask.sum().clamp_min(1))
            kl = -.5*(1+logvar-mu.square()-logvar.exp()).sum(-1).mean()
            relation = relation_loss(routes, base.compatibility, teacher_edges(teacher, actions, groups))
            per_sample_entropy = -(routes.clamp_min(1e-8).log()*routes).sum(-1).mean()
            marginal = routes.mean((0,1)); marginal_entropy = -(marginal.clamp_min(1e-8).log()*marginal).sum()
            specialization = per_sample_entropy - marginal_entropy
            starvation = F.relu(.05-marginal).square().sum()
            coupling = routes.sum()*0.; anchor = routes.sum()*0.
            if do_cf:
                errors = (cf-cf_target.unsqueeze(2)).square().mean(-1)
                temp = errors.detach().std(-1, keepdim=True).clamp_min(1e-3)
                qcap = (-errors.detach()/temp).softmax(-1)
                selected = routes[torch.arange(len(cf), device=routes.device)* (len(routes)//len(cf))]
                coupling = F.kl_div(selected.clamp_min(1e-8).log(), qcap, reduction="none").sum(-1).mean()
                fp = fingerprint(cf)
                capability_ema = fp.detach() if capability_ema is None else .99*capability_ema + .01*fp.detach()
                if update >= discovery and capability_ref is None: capability_ref = capability_ema.clone()
                if capability_ref is not None: anchor = F.mse_loss(fp, capability_ref)
            loss = (action + args.beta*kl + args.relation_weight*relation +
                    args.capability_weight*coupling + args.specialization_weight*(specialization+starvation) +
                    args.anchor_weight*anchor)
        loss.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        last = {"loss":float(loss),"action":float(action),"kl":float(kl),"relation":float(relation),
                "coupling":float(coupling),"anchor":float(anchor),"route_entropy":float(per_sample_entropy),
                "marginal_entropy":float(marginal_entropy),"near_half":float(((routes.topk(2,-1).values.diff(-1).abs())<.05).float().mean())}
        if update == 1 or update % args.log_every == 0:
            elapsed=time.time()-started
            print(json.dumps({"update":update,**last,"updates_per_hour":update/elapsed*3600,
                              "eta_hours":(args.updates-update)*elapsed/update/3600,
                              "gpu_memory_gb":[round(torch.cuda.max_memory_allocated(i)/2**30,2) for i in range(3)]}),flush=True)
        if update in milestones:
            torch.save({"model":base.state_dict(),"stats":stats,"config":config,"update":update,
                        "capability_reference":capability_ref,"last_metrics":last}, out/f"checkpoint_{update:06d}.pt")
    print(json.dumps({"complete":True,"checkpoint":str(out/f'checkpoint_{args.updates:06d}.pt')}),flush=True)


if __name__ == "__main__": main()
