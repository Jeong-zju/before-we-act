"""PAIR-Route policy: strict-local routing with training-only team relations.

The deployed forward path is exactly local wrist RGB-D plus own qpos.  Team
actions are consumed only by the trainer to form relational and capability
losses; this module never receives a task/agent ID, language, peer observation,
global image, or communication message.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from stereo_decoder_variants import StereoARCA


class StereoPAIRRoute(StereoARCA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compatibility = torch.nn.Parameter(torch.empty(self.roles_n, self.roles_n))
        torch.nn.init.xavier_uniform_(self.compatibility)
        self.last_dense_routes = None
        self.last_sparse_routes = None
        self._counterfactual_cache = None

    def _pair_route(self, state, observation, batch):
        query = self.query.expand(batch, -1, -1)
        context = self.route_state(state) + self.route_observation(observation.mean(1))
        features = self.route_mlp(query + context.unsqueeze(1))
        logits = torch.matmul(features, self.role_prototypes.t()) / math.sqrt(features.shape[-1])
        dense = logits.softmax(-1)
        values, ids = logits.topk(2, dim=-1)
        sparse = torch.zeros_like(logits).scatter_(-1, ids, values.softmax(-1).to(logits.dtype))
        self.last_dense_routes, self.last_sparse_routes = dense, sparse
        return sparse

    def forward(self, image, depth_mm, qpos, actions=None, return_routing=False,
                counterfactual=False):
        observation = self._rgbd_tokens(image, depth_mm)
        state_vec = self.state(qpos)
        gates = self._pair_route(state_vec, observation, image.shape[0])
        if actions is not None:
            encoded = self.posterior(self.action(actions) + self.pos)
            mu, logvar = self.latent(encoded.mean(1)).chunk(2, -1)
            logvar = logvar.clamp(-10.0, 5.0)
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            mu = logvar = None
            latent = torch.zeros((image.shape[0], self.z_proj.in_features), device=image.device)
        memory = torch.cat((state_vec.unsqueeze(1), self.z_proj(latent).unsqueeze(1), observation), dim=1)
        decoded = self.decoder(self.query.expand(image.shape[0], -1, -1), memory, observation, gates)
        self._counterfactual_cache = (memory, observation)
        prediction = self.out(decoded)
        if not return_routing:
            return prediction, mu, logvar, observation.new_zeros(())

        # One intervention sample per replica keeps the capability target exact
        # while bounding memory. DataParallel gathers one sample from each GPU.
        cf_predictions = prediction.new_empty((0, self.horizon, self.roles_n,
                                                prediction.shape[-1]))
        cf_targets = prediction.new_empty((0, self.horizon, prediction.shape[-1]))
        if counterfactual and actions is not None:
            cf_memory, cf_observation = memory[:1], observation[:1]
            cf_query = self.query.expand(1, -1, -1)
            role_predictions = []
            for role in range(self.roles_n):
                forced = prediction.new_zeros((1, self.horizon, self.roles_n))
                forced[..., role] = 1
                role_predictions.append(self.out(self.decoder(
                    cf_query, cf_memory, cf_observation, forced)))
            cf_predictions = torch.stack(role_predictions, dim=2)
            cf_targets = actions[:1]
        return (prediction, mu, logvar, observation.new_zeros(()),
                self.last_dense_routes, cf_predictions, cf_targets)

    @torch.no_grad()
    def counterfactual_errors(self, target, sample_count=4):
        """Per-query error under each forced role adapter.

        The result is detached and therefore only supervises the router.  The
        normal imitation path remains responsible for learning policy/expert
        weights, avoiding a winner-take-all self-reinforcing expert update.
        """
        if self._counterfactual_cache is None:
            raise RuntimeError("forward must run before counterfactual_errors")
        memory, observation = self._counterfactual_cache
        count = min(int(sample_count), len(target))
        memory, observation, target = memory[:count].detach(), observation[:count].detach(), target[:count]
        query = self.query.expand(count, -1, -1)
        errors = []
        was_training = self.decoder.training
        self.decoder.eval()
        for role in range(self.roles_n):
            gate = target.new_zeros((count, target.shape[1], self.roles_n))
            gate[..., role] = 1
            prediction = self.out(self.decoder(query, memory, observation, gate))
            errors.append((prediction - target).square().mean(-1))
        self.decoder.train(was_training)
        return torch.stack(errors, -1)

    def local_relation_logits(self, groups):
        """Directed local relation logits for each synchronized team group."""
        routes = self.last_dense_routes.mean(1)
        result = []
        for group in groups.unique(sorted=True):
            ids = (groups == group).nonzero(as_tuple=False).flatten()
            p = routes.index_select(0, ids)
            result.append(torch.einsum("ir,rs,js->ij", p, self.compatibility, p))
        return result

    def routing_regularizers(self):
        routes = self.last_dense_routes
        per_sample = -(routes.clamp_min(1e-8).log() * routes).sum(-1).mean()
        marginal = routes.mean((0, 1))
        marginal_entropy = -(marginal.clamp_min(1e-8).log() * marginal).sum()
        mutual_information_loss = per_sample - marginal_entropy
        capacity_floor = F.relu(0.05 - marginal).square().sum()
        top = routes.topk(2, -1).values
        near_half = ((top[..., 0] - top[..., 1]).abs() < 0.05).float().mean()
        return mutual_information_loss, capacity_floor, per_sample, near_half
