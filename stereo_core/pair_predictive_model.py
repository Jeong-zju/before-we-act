"""Predictability-Aware Agent Interaction Routing (PAIR).

The policy remains strictly local at deployment.  A training-only teacher sees
synchronised action sets and the future *local* consequence of each arm.  The
policy is not regressed to an arbitrary privileged role label; instead, its
local interaction representation is contrastively aligned to the part of the
teacher event that is predictable from current wrist RGB-D and qpos.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo_decoder_variants import StereoARCA


class StereoPredictabilityPAIR(StereoARCA):
    """Local-ARCA plus a locally inferred interaction-event representation.

    ``interaction_to_query`` is zero-initialised.  Loading a Local-ARCA
    checkpoint therefore starts from the exact deployed baseline and lets the
    new signal prove useful instead of perturbing the policy at update zero.
    """

    def __init__(self, *args, event_dim=128, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.query.shape[-1]
        self.event_dim = event_dim
        self.local_event_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, event_dim),
        )
        self.interaction_to_query = nn.Linear(event_dim, d, bias=False)
        nn.init.zeros_(self.interaction_to_query.weight)
        self.last_local_event = None
        self.last_gates = None

    def _route_with_event(self, state, observation, batch):
        query = self.query.expand(batch, -1, -1)
        context = self.route_state(state) + self.route_observation(observation.mean(1))
        local_event = F.normalize(self.local_event_head(context).float(), dim=-1, eps=1e-6)
        event_bias = self.interaction_to_query(local_event.to(context.dtype))
        features = self.route_mlp(query + context.unsqueeze(1) + event_bias.unsqueeze(1))
        logits = torch.matmul(features, self.role_prototypes.t()) / math.sqrt(features.shape[-1])
        values, ids = logits.topk(2, dim=-1)
        gates = torch.zeros_like(logits).scatter_(-1, ids, values.softmax(-1).to(logits.dtype))
        importance = logits.softmax(-1).mean((0, 1))
        load = gates.gt(0).to(logits.dtype).mean((0, 1)) / 2.0
        auxiliary = self.roles_n * (importance * load).sum()
        return gates, auxiliary, local_event

    def forward(self, image, depth_mm, qpos, actions=None):
        observation = self._rgbd_tokens(image, depth_mm)
        state_vector = self.state(qpos)
        gates, auxiliary, local_event = self._route_with_event(
            state_vector, observation, image.shape[0]
        )
        self.last_local_event, self.last_gates = local_event, gates
        if actions is not None:
            hidden = self.posterior(self.action(actions) + self.pos)
            mu, logvar = self.latent(hidden.mean(1)).chunk(2, -1)
            logvar = logvar.clamp(-10.0, 5.0)
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            mu = logvar = None
            latent = torch.zeros(
                (image.shape[0], self.z_proj.in_features), device=image.device
            )
        memory = torch.cat(
            (
                state_vector.unsqueeze(1),
                self.z_proj(latent).unsqueeze(1),
                observation,
            ),
            dim=1,
        )
        decoded = self.decoder(
            self.query.expand(image.shape[0], -1, -1),
            memory,
            observation,
            gates,
        )
        return self.out(decoded), mu, logvar, auxiliary, local_event


class PredictiveInteractionTeacher(nn.Module):
    """Training-only event teacher over unordered synchronised teams.

    The teacher is permutation equivariant: each target arm is represented by
    its own action/consequence and the mean of the other arms' action features.
    No task name, agent identifier, global view, or simulator state is used.
    """

    def __init__(self, action_dim, effect_dim, event_dim=128, width=192):
        super().__init__()
        self.event_dim = event_dim
        self.action_step = nn.Sequential(
            nn.Linear(action_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.action_feature = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.effect_feature = nn.Sequential(
            nn.LayerNorm(effect_dim),
            nn.Linear(effect_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.event = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.Linear(width, event_dim),
        )
        self.own_effect = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, effect_dim),
        )
        self.interaction_effect = nn.Sequential(
            nn.Linear(event_dim, width),
            nn.GELU(),
            nn.Linear(width, effect_dim),
        )
        self.sync_score = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def action_features(self, actions):
        delta = actions[:, 1:] - actions[:, :-1]
        encoded = self.action_step(delta.float())
        return self.action_feature(
            torch.cat((encoded.mean(1), encoded.square().mean(1).sqrt()), dim=-1)
        )

    @staticmethod
    def peer_means(features, groups):
        peers = torch.zeros_like(features)
        for group in groups.unique(sorted=True):
            ids = (groups == group).nonzero(as_tuple=False).flatten()
            values = features.index_select(0, ids)
            if len(ids) > 1:
                peer = ((values.sum(0, keepdim=True) - values) / (len(ids) - 1)).to(
                    features.dtype
                )
            else:
                peer = torch.zeros_like(values)
            peers.index_copy_(0, ids, peer)
        return peers

    @staticmethod
    def shuffled_peer_means(peer, groups):
        unique = groups.unique(sorted=True)
        if len(unique) < 2:
            return peer.roll(1, dims=0)
        shuffled = peer.clone()
        # Shift complete time groups, preserving within-group team structure.
        for index, group in enumerate(unique):
            source_group = unique[(index + 1) % len(unique)]
            target_ids = (groups == group).nonzero(as_tuple=False).flatten()
            source_ids = (groups == source_group).nonzero(as_tuple=False).flatten()
            source = peer.index_select(0, source_ids)
            if len(source) != len(target_ids):
                source = source.mean(0, keepdim=True).expand(len(target_ids), -1)
            shuffled.index_copy_(0, target_ids, source)
        return shuffled

    def forward(self, actions, future_effect, groups):
        own = self.action_features(actions)
        effect = self.effect_feature(future_effect.float())
        peer = self.peer_means(own, groups)
        teacher_event = F.normalize(
            self.event(torch.cat((own, peer, effect), dim=-1)).float(),
            dim=-1,
            eps=1e-6,
        )

        own_prediction = self.own_effect(own)
        full_prediction = own_prediction + self.interaction_effect(teacher_event)
        own_loss = F.mse_loss(own_prediction, future_effect.float())
        full_loss = F.mse_loss(full_prediction, future_effect.float())

        true_logits = self.sync_score(torch.cat((own, peer, effect), dim=-1))
        shuffled_peer = self.shuffled_peer_means(peer, groups)
        false_logits = self.sync_score(torch.cat((own, shuffled_peer, effect), dim=-1))
        sync_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(true_logits, torch.ones_like(true_logits))
            + F.binary_cross_entropy_with_logits(false_logits, torch.zeros_like(false_logits))
        )
        sync_accuracy = 0.5 * (
            (true_logits > 0).float().mean() + (false_logits < 0).float().mean()
        )
        return teacher_event, own_loss, full_loss, sync_loss, sync_accuracy


def symmetric_contrastive_alignment(local_event, teacher_event, temperature=0.1):
    """Same-sample alignment with all other same-task/time samples as negatives."""
    local = F.normalize(local_event.float(), dim=-1, eps=1e-6)
    teacher = F.normalize(teacher_event.float(), dim=-1, eps=1e-6)
    logits = local @ teacher.detach().t() / temperature
    targets = torch.arange(len(local), device=local.device)
    student_to_teacher = F.cross_entropy(logits, targets)
    # The second direction stabilises the shared subspace without propagating
    # gradients into the policy through the teacher branch.
    teacher_to_student = F.cross_entropy(logits.t(), targets)
    return 0.5 * (student_to_teacher + teacher_to_student)
