"""Decoder capacity/routing variants for strict local-wrist Stereo-ACT.

Both variants retain frozen DINOv3 + DeFM and the 30x40 RGB->depth
cross_relbias front end from StereoACT.  They use only current local wrist
RGB-D tokens and local qpos; no task/agent ID, language, peer, or global view.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from train_stereo_act import StereoACT


class _Expert(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, d_model), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)


class Top2SparseMoE(nn.Module):
    """Four FFN experts; each local decoder token selects exactly top-2."""
    def __init__(self, d_model, ffn_dim, experts=4, dropout=.1):
        super().__init__(); self.experts_n = experts
        self.router = nn.Linear(d_model, experts, bias=False)
        self.experts = nn.ModuleList([_Expert(d_model, ffn_dim, dropout) for _ in range(experts)])

    def forward(self, x):
        shape, flat = x.shape, x.reshape(-1, x.shape[-1])
        logits = self.router(flat)
        top_logits, top_ids = logits.topk(2, dim=-1)
        gates = top_logits.softmax(-1)
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            chosen = (top_ids == expert_id).nonzero(as_tuple=False)
            if chosen.numel() == 0: continue
            token_ids, slots = chosen[:, 0], chosen[:, 1]
            y = expert(flat.index_select(0, token_ids))
            out.index_add_(0, token_ids, y * gates[token_ids, slots].unsqueeze(-1))
        # Switch-style differentiable importance/load balancing; its minimum is one.
        importance = logits.softmax(-1).mean(0)
        load = torch.bincount(top_ids.reshape(-1), minlength=self.experts_n).to(flat.dtype) / (2.0 * flat.shape[0])
        aux = self.experts_n * (importance * load).sum()
        return out.reshape(shape), aux


class MoEDecoderLayer(nn.Module):
    def __init__(self, d_model, heads=8, ffn_dim=None, dropout=.1, experts=4):
        super().__init__(); ffn_dim = ffn_dim or 4*d_model
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.moe = Top2SparseMoE(d_model, ffn_dim, experts=experts, dropout=dropout)

    def forward(self, x, memory):
        x = x + self.drop1(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        x = x + self.drop2(self.cross_attn(self.norm2(x), memory, memory, need_weights=False)[0])
        ff, aux = self.moe(self.norm3(x)); return x + ff, aux


class MoEDecoder(nn.Module):
    def __init__(self, d_model, layers=7, experts=4, dropout=.1):
        super().__init__(); self.layers = nn.ModuleList([MoEDecoderLayer(d_model, dropout=dropout, experts=experts) for _ in range(layers)])
    def forward(self, x, memory):
        aux = x.new_zeros(())
        for layer in self.layers:
            x, value = layer(x, memory); aux = aux + value
        return x, aux / len(self.layers)


class StereoFFNMoE(StereoACT):
    """Stereo front end + top-2 four-expert FFN replacement in every decoder block."""
    def __init__(self, *args, experts=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.experts_n = experts
        self.decoder = MoEDecoder(self.query.shape[-1], layers=len(self.decoder.layers), experts=experts)

    def forward(self, image, depth_mm, qpos, actions=None):
        x = self._rgbd_tokens(image, depth_mm); state = self.state(qpos).unsqueeze(1)
        if actions is not None:
            h = self.posterior(self.action(actions) + self.pos)
            mu, logvar = self.latent(h.mean(1)).chunk(2, -1); z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar)
        else:
            mu = logvar = None; z = torch.zeros((image.shape[0], self.z_proj.in_features), device=image.device)
        memory = torch.cat((state, self.z_proj(z).unsqueeze(1), x), dim=1)
        decoded, aux = self.decoder(self.query.expand(image.shape[0], -1, -1), memory)
        return self.out(decoded), mu, logvar, aux


class RoleCrossAdapter(nn.Module):
    """Small role-specific cross-attention from one action query to current observation tokens."""
    def __init__(self, d_model, rank=32):
        super().__init__()
        self.q = nn.Linear(d_model, rank, bias=False); self.k = nn.Linear(d_model, rank, bias=False)
        self.v = nn.Linear(d_model, rank, bias=False); self.out = nn.Linear(rank, d_model, bias=False)
        self.rank = rank
    def forward(self, query, observation):
        scores = torch.matmul(self.q(query), self.k(observation).transpose(-1, -2)) / math.sqrt(self.rank)
        return self.out(torch.matmul(scores.softmax(-1), self.v(observation)))


class ARCADecoderLayer(nn.Module):
    def __init__(self, d_model, roles=4, rank=32, heads=8, dropout=.1, sparse_roles=False):
        super().__init__(); ffn = 4*d_model
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.ff = _Expert(d_model, ffn, dropout)
        self.adapters = nn.ModuleList([RoleCrossAdapter(d_model, rank) for _ in range(roles)])
        self.sparse_roles = sparse_roles

    def forward(self, x, memory, observation, gates):
        x = x + self.drop1(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        h = self.norm2(x)
        base = self.cross_attn(h, memory, memory, need_weights=False)[0]
        role = torch.zeros_like(base)
        for role_id, adapter in enumerate(self.adapters):
            if not self.sparse_roles:
                role = role + gates[..., role_id:role_id+1] * adapter(h, observation)
                continue
            # MSA routes exactly top-2 roles per action query.  Execute an
            # adapter only for selected (batch, query) pairs rather than
            # evaluating every role densely and multiplying most outputs by
            # zero.  This makes MSA a genuine sparse decoder-MoE and retains
            # gradients through the selected gate values.
            chosen = (gates[..., role_id] > 0).nonzero(as_tuple=False)
            if chosen.numel() == 0:
                continue
            batch_ids, query_ids = chosen[:, 0], chosen[:, 1]
            selected_query = h[batch_ids, query_ids].unsqueeze(1)
            selected_observation = observation.index_select(0, batch_ids)
            # Preserve the exact sparse role computation while discarding its
            # large RGB-D attention activations until backward recomputation.
            selected = checkpoint(adapter, selected_query, selected_observation,
                                  use_reentrant=False).squeeze(1)
            selected = selected * gates[batch_ids, query_ids, role_id].unsqueeze(-1)
            role = role.index_put((batch_ids, query_ids), selected, accumulate=True)
        x = x + self.drop2(base + role)
        return x + self.ff(self.norm3(x))


class ARCADecoder(nn.Module):
    def __init__(self, d_model, layers=7, roles=4, rank=32, dropout=.1, sparse_roles=False):
        super().__init__(); self.layers = nn.ModuleList([
            ARCADecoderLayer(d_model, roles, rank, dropout=dropout, sparse_roles=sparse_roles)
            for _ in range(layers)
        ])
    def forward(self, x, memory, observation, gates):
        for layer in self.layers: x = layer(x, memory, observation, gates)
        return x


class StereoARCA(StereoACT):
    """Action-role conditioned observation cross-attention inside every decoder layer."""
    def __init__(self, *args, roles=4, role_rank=32, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.query.shape[-1]; self.roles_n, self.role_rank = roles, role_rank
        self.decoder = ARCADecoder(d, layers=len(self.decoder.layers), roles=roles, rank=role_rank)
        self.route_state, self.route_observation = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.route_mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d, bias=False))
        self.role_prototypes = nn.Parameter(torch.randn(roles, d) * .02)

    def _route(self, state, observation, batch):
        # This deliberately excludes ACT's posterior z: z is zero at deployment.
        q = self.query.expand(batch, -1, -1)
        context = self.route_state(state) + self.route_observation(observation.mean(1))
        features = self.route_mlp(q + context.unsqueeze(1))
        logits = torch.matmul(features, self.role_prototypes.t()) / math.sqrt(features.shape[-1])
        values, ids = logits.topk(2, dim=-1); gates = torch.zeros_like(logits).scatter_(-1, ids, values.softmax(-1).to(logits.dtype))
        importance = logits.softmax(-1).mean((0, 1))
        load = (gates.gt(0).to(logits.dtype).mean((0, 1)) / 2.0)
        aux = self.roles_n * (importance * load).sum()
        return gates, aux

    def forward(self, image, depth_mm, qpos, actions=None):
        x = self._rgbd_tokens(image, depth_mm); state_vec = self.state(qpos); state = state_vec.unsqueeze(1)
        gates, aux = self._route(state_vec, x, image.shape[0])
        if actions is not None:
            h = self.posterior(self.action(actions) + self.pos)
            mu, logvar = self.latent(h.mean(1)).chunk(2, -1); z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar)
        else:
            mu = logvar = None; z = torch.zeros((image.shape[0], self.z_proj.in_features), device=image.device)
        memory = torch.cat((state, self.z_proj(z).unsqueeze(1), x), dim=1)
        decoded = self.decoder(self.query.expand(image.shape[0], -1, -1), memory, x, gates)
        return self.out(decoded), mu, logvar, aux


class StereoPAIRAdapter(StereoACT):
    """PAIR: permutation-invariant action-relation informed adapters.

    The deployed policy is strictly local.  ``local_role_head`` consumes only
    the current wrist RGB-D tokens and qpos.  During training, its soft role
    distribution is distilled from :class:`PAIRActionTeacher`, which consumes
    *only the simultaneously recorded action chunks of an unordered team*.
    The teacher is intentionally not a policy input and is not needed at test
    time.  Role probabilities condition low-rank cross-attention adapters in
    every shared ACT decoder layer.
    """
    def __init__(self, *args, roles=4, role_rank=32, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.query.shape[-1]
        self.roles_n, self.role_rank = roles, role_rank
        self.decoder = ARCADecoder(d, layers=len(self.decoder.layers), roles=roles,
                                   rank=role_rank, sparse_roles=False)
        self.local_state = nn.Linear(d, d, bias=False)
        self.local_observation = nn.Linear(d, d, bias=False)
        self.local_role_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, roles)
        )
        # The 100 learned ACT action queries may prefer different functional
        # roles, but every decision remains conditioned by the local role
        # posterior.  Zero initialization starts from a role-agnostic policy.
        self.query_role_bias = nn.Parameter(torch.zeros(self.query.shape[1], roles))
        self.last_role_probs = None
        self.last_gates = None

    def _local_roles(self, state, observation):
        context = self.local_state(state) + self.local_observation(observation.mean(1))
        return self.local_role_head(context)

    def forward(self, image, depth_mm, qpos, actions=None):
        x = self._rgbd_tokens(image, depth_mm)
        state_vec = self.state(qpos)
        role_logits = self._local_roles(state_vec, x)
        role_probs = role_logits.softmax(-1)
        gates = (role_logits.unsqueeze(1) + self.query_role_bias.unsqueeze(0)).softmax(-1)
        self.last_role_probs, self.last_gates = role_probs, gates
        if actions is not None:
            h = self.posterior(self.action(actions) + self.pos)
            mu, logvar = self.latent(h.mean(1)).chunk(2, -1)
            # The posterior variance is an internal training quantity.  A
            # bounded log-variance prevents a rare chunk from overflowing the
            # VAE sample while preserving the standard ACT posterior pathway.
            logvar = logvar.clamp(-10., 5.)
            z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar)
        else:
            mu = logvar = None
            z = torch.zeros((image.shape[0], self.z_proj.in_features), device=image.device)
        memory = torch.cat((state_vec.unsqueeze(1), self.z_proj(z).unsqueeze(1), x), dim=1)
        decoded = self.decoder(self.query.expand(image.shape[0], -1, -1), memory, x, gates)
        # This auxiliary is exactly zero: PAIR avoids an artificial uniform
        # route prior.  Role identifiability comes from the action-relation
        # teacher rather than a load-balancing assumption.
        return self.out(decoded), mu, logvar, x.new_zeros(()), role_probs


class PAIRActionTeacher(nn.Module):
    """Training-only equivariant teacher over unordered synchronized actions.

    For each same-time robot set, it encodes action chunks, builds an
    order-free per-agent relation summary, and assigns a soft functional role.
    It is anchored by reconstructing action features and their pairwise action
    relation matrix.  No robot identifier, task name, visual observation, or
    simulator state enters this module.
    """
    def __init__(self, action_dim, roles=4, width=192):
        super().__init__()
        self.roles_n, self.width = roles, width
        self.step = nn.Sequential(nn.Linear(action_dim, width), nn.GELU(), nn.Linear(width, width))
        self.feature = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.pair = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, width))
        self.role_head = nn.Sequential(nn.LayerNorm(3 * width), nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, roles))
        self.role_basis = nn.Parameter(torch.randn(roles, width) * .02)
        self.reconstruct = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def _action_features(self, actions):
        # Differences remove static joint offsets; mean and variation preserve
        # what the local arm is about to do over ACT's standard target chunk.
        delta = actions[:, 1:] - actions[:, :-1]
        z = self.step(delta)
        return self.feature(torch.cat((z.mean(1), z.square().mean(1).sqrt()), dim=-1))

    def forward(self, actions, groups):
        features = self._action_features(actions.float())
        # Keep teacher posteriors in fp32 under the policy's bf16 autocast;
        # they are KL targets rather than a high-throughput vision activation.
        roles = torch.zeros((len(actions), self.roles_n), device=actions.device, dtype=torch.float32)
        recon_losses, relation_losses = [], []
        for group in groups.unique(sorted=True):
            ids = (groups == group).nonzero(as_tuple=False).flatten()
            f = features.index_select(0, ids)
            n = len(ids)
            team = f.mean(0, keepdim=True).expand(n, -1)
            fi, fj = f.unsqueeze(1).expand(n, n, -1), f.unsqueeze(0).expand(n, n, -1)
            pair = self.pair(torch.cat((fi, fj, fi - fj, fi * fj), dim=-1))
            if n > 1:
                context = (pair.sum(1) - pair.diagonal(dim1=0, dim2=1).transpose(0, 1)) / (n - 1)
            else:
                context = torch.zeros_like(f)
            probs = self.role_head(torch.cat((f, context, team), dim=-1)).softmax(-1)
            roles.index_copy_(0, ids, probs.float())
            reconstructed = self.reconstruct(probs @ self.role_basis)
            recon_losses.append(F.mse_loss(reconstructed, f))
            if n > 1:
                target = F.normalize(f, dim=-1, eps=1e-6) @ F.normalize(f, dim=-1, eps=1e-6).t()
                predicted = F.normalize(reconstructed, dim=-1, eps=1e-6) @ F.normalize(reconstructed, dim=-1, eps=1e-6).t()
                off_diagonal = ~torch.eye(n, dtype=torch.bool, device=f.device)
                relation_losses.append(F.mse_loss(predicted[off_diagonal], target[off_diagonal]))
        # A weak use penalty prevents the trivial one-role teacher, without
        # dictating equal role frequencies within any individual task.
        usage = roles.mean(0)
        usage_penalty = (usage - (1.0 / self.roles_n)).square().mean()
        zero = features.new_zeros(())
        return roles, (torch.stack(recon_losses).mean() if recon_losses else zero), \
            (torch.stack(relation_losses).mean() if relation_losses else zero), usage_penalty


class StereoMSA(StereoACT):
    """Mode-Structured Action routing for local-wrist Stereo-ACT.

    Unlike the token-wise top-2 router in :class:`StereoARCA`, this model
    treats the 100 action queries as a short latent state sequence.  Local
    RGB-D/qpos emits evidence for each mode, while a learned transition
    matrix propagates the route across adjacent action queries.  Thus a mode
    persists by default but may change when the *current local* observation
    supports a different action segment.  No previous observation, wall-clock
    time, task/agent ID, peer observation/action, or global camera is used.

    ``interaction_graph`` is used only by the optional training-time
    synchronized-demo contrastive loss.  It is never an inference input.
    """
    def __init__(self, *args, roles=4, role_rank=32, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.query.shape[-1]
        self.roles_n, self.role_rank = roles, role_rank
        # MSA uses a top-2 active role set.  The sparse adapter implementation
        # is enabled only under the memory-safe micro-batch protocol.
        self.decoder = ARCADecoder(d, layers=len(self.decoder.layers), roles=roles, rank=role_rank,
                                   sparse_roles=True)
        self.route_state = nn.Linear(d, d, bias=False)
        self.route_observation = nn.Linear(d, d, bias=False)
        self.route_mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d, bias=False))
        self.mode_prototypes = nn.Parameter(torch.randn(roles, d) * .02)
        # Diagonal-biased initialization gives a stable, but learnable,
        # semi-Markov prior.  Evidence from local RGB-D can still switch mode.
        self.transition_logits = nn.Parameter(torch.eye(roles) * 1.5)
        self.start_logits = nn.Parameter(torch.zeros(roles))
        # Directed role relation used exclusively for the graph supervision.
        self.interaction_graph = nn.Parameter(torch.empty(roles, roles))
        nn.init.xavier_uniform_(self.interaction_graph)

    def _route(self, state, observation, batch):
        q = self.query.expand(batch, -1, -1)
        context = self.route_state(state) + self.route_observation(observation.mean(1))
        features = self.route_mlp(q + context.unsqueeze(1))
        emissions = torch.matmul(features, self.mode_prototypes.t()) / math.sqrt(features.shape[-1])
        log_transition = F.log_softmax(self.transition_logits, dim=-1)
        log_prob = emissions[:, 0] + F.log_softmax(self.start_logits, dim=-1)
        routes = [log_prob.softmax(-1)]
        for index in range(1, emissions.shape[1]):
            # p(z_t|x_1:t) with a learned role-transition graph.  This is a
            # differentiable forward recursion, not a task-time condition.
            log_prob = emissions[:, index] + torch.logsumexp(
                log_prob.unsqueeze(-1) + log_transition.unsqueeze(0), dim=1
            )
            routes.append(log_prob.softmax(-1))
        dense_gates = torch.stack(routes, dim=1)
        values, ids = dense_gates.topk(min(2, self.roles_n), dim=-1)
        gates = torch.zeros_like(dense_gates).scatter_(-1, ids, values)
        gates = gates / gates.sum(-1, keepdim=True).clamp_min(1e-8)
        importance = dense_gates.mean((0, 1))
        usage = gates.gt(0).to(gates.dtype).mean((0, 1)) / 2.0
        load_aux = self.roles_n * (importance * usage).sum()
        switch_rate = 1.0 - (gates[:, 1:] * gates[:, :-1]).sum(-1).mean()
        self.last_route_stats = {
            "route_entropy": -(gates.clamp_min(1e-8).log() * gates).sum(-1).mean().detach(),
            "route_switch_rate": switch_rate.detach(),
        }
        return gates, load_aux

    def orthogonal_basis_penalty(self):
        """Keep action-role adapters as a reusable, non-collapsed skill basis.

        This is the *SMP-inspired* capacity factor.  It operates on learned
        parameters only and adds neither labels nor deployment inputs.
        """
        vectors = [self.mode_prototypes]
        for layer in self.decoder.layers:
            vectors.append(torch.stack([adapter.out.weight.flatten() for adapter in layer.adapters]))
        penalties = []
        for basis in vectors:
            normal = F.normalize(basis.float(), dim=-1)
            gram = normal @ normal.t()
            penalties.append((gram - torch.eye(self.roles_n, device=gram.device)).square().mean())
        return torch.stack(penalties).mean()

    def forward(self, image, depth_mm, qpos, actions=None):
        x = self._rgbd_tokens(image, depth_mm)
        state_vec = self.state(qpos)
        gates, aux = self._route(state_vec, x, image.shape[0])
        if actions is not None:
            h = self.posterior(self.action(actions) + self.pos)
            mu, logvar = self.latent(h.mean(1)).chunk(2, -1)
            z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar)
        else:
            mu = logvar = None
            z = torch.zeros((image.shape[0], self.z_proj.in_features), device=image.device)
        memory = torch.cat((state_vec.unsqueeze(1), self.z_proj(z).unsqueeze(1), x), dim=1)
        decoded = self.decoder(self.query.expand(image.shape[0], -1, -1), memory, x, gates)
        return self.out(decoded), mu, logvar, aux, gates


class StereoSyncARCA(StereoARCA):
    """Stereo-ARCA with a training-only synchronized action-stage teacher.

    At inference ``phase_target`` is never provided.  Each local policy predicts
    the phase from its own current wrist RGB-D tokens and qpos, then conditions
    action-query routing on that *predicted* soft phase.  The optional target is
    used only for the CE loss in the trainer.
    """
    def __init__(self, *args, phases=8, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.query.shape[-1]
        self.phases_n = phases
        self.phase_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, phases))
        self.phase_embed = nn.Parameter(torch.randn(phases, d) * .02)

    def _sync_route(self, state, observation, phase_probs, batch):
        q = self.query.expand(batch, -1, -1)
        context = self.route_state(state) + self.route_observation(observation.mean(1))
        phase_context = torch.matmul(phase_probs, self.phase_embed)
        features = self.route_mlp(q + (context + phase_context).unsqueeze(1))
        logits = torch.matmul(features, self.role_prototypes.t()) / math.sqrt(features.shape[-1])
        values, ids = logits.topk(2, dim=-1)
        gates = torch.zeros_like(logits).scatter_(-1, ids, values.softmax(-1).to(logits.dtype))
        importance = logits.softmax(-1).mean((0, 1))
        load = gates.gt(0).to(logits.dtype).mean((0, 1)) / 2.0
        aux = self.roles_n * (importance * load).sum()
        return gates, aux

    def forward(self, image, depth_mm, qpos, actions=None):
        x = self._rgbd_tokens(image, depth_mm)
        state_vec = self.state(qpos)
        local_context = self.route_state(state_vec) + self.route_observation(x.mean(1))
        phase_logits = self.phase_head(local_context)
        phase_probs = phase_logits.softmax(-1)
        gates, aux = self._sync_route(state_vec, x, phase_probs, image.shape[0])
        state = state_vec.unsqueeze(1)
        if actions is not None:
            h = self.posterior(self.action(actions) + self.pos)
            mu, logvar = self.latent(h.mean(1)).chunk(2, -1)
            z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar)
        else:
            mu = logvar = None
            z = torch.zeros((image.shape[0], self.z_proj.in_features), device=image.device)
        memory = torch.cat((state, self.z_proj(z).unsqueeze(1), x), dim=1)
        decoded = self.decoder(self.query.expand(image.shape[0], -1, -1), memory, x, gates)
        return self.out(decoded), mu, logvar, aux, phase_logits, gates
