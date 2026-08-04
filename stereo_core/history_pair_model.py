"""Strict-local temporal routing variants for the PAIR observability pilot.

Both variants keep the frozen Local-ARCA policy body.  A causal local history
encoder can only change Local-ARCA's action-query router.  PAIR-Belief differs
from History-only solely by a training-time cosine target from the validated
synchronized-action complementarity teacher.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo_decoder_variants import StereoARCA


class StereoHistoryARCA(StereoARCA):
    def __init__(self, *args, history=32, history_stride=4, role_dim=64, **kwargs):
        state_dim = int(args[0] if args else kwargs["state_dim"])
        super().__init__(*args, **kwargs)
        d = self.query.shape[-1]
        self.history_length = int(history)
        self.history_stride = int(history_stride)
        self.history_input = nn.Sequential(
            nn.LayerNorm(d + state_dim),
            nn.Linear(d + state_dim, d), nn.GELU(),
        )
        self.history_gru = nn.GRU(d, d, num_layers=2, batch_first=True, dropout=.1)
        self.route_history = nn.Linear(d, d, bias=False)
        self.belief_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, role_dim))
        nn.init.zeros_(self.route_history.weight)
        self._online_history = None
        self.last_belief = None

    def reset_history(self):
        self._online_history = None

    def _encode_history(self, history):
        encoded = self.history_input(history)
        _, hidden = self.history_gru(encoded)
        summary = hidden[-1]
        self.last_belief = F.normalize(self.belief_head(summary), dim=-1)
        return summary

    def _online_window(self, observation, qpos):
        current = torch.cat((observation.mean(1).detach(), qpos.detach()), -1)
        if self._online_history is None or len(self._online_history) != len(current):
            self._online_history = [[] for _ in range(len(current))]
        windows = []
        for index, row in enumerate(current):
            self._online_history[index].append(row)
            maximum = 1 + (self.history_length - 1) * self.history_stride
            self._online_history[index] = self._online_history[index][-maximum:]
            values = self._online_history[index][::-self.history_stride][::-1]
            pad = [torch.zeros_like(row) for _ in range(self.history_length - len(values))]
            windows.append(torch.stack(pad + values))
        return torch.stack(windows)

    def _route_with_history(self, state, observation, history_summary, batch):
        q = self.query.expand(batch, -1, -1)
        context = (self.route_state(state) + self.route_observation(observation.mean(1))
                   + self.route_history(history_summary))
        features = self.route_mlp(q + context.unsqueeze(1))
        logits = torch.matmul(features, self.role_prototypes.t()) / features.shape[-1] ** .5
        values, ids = logits.topk(2, dim=-1)
        gates = torch.zeros_like(logits).scatter_(-1, ids, values.softmax(-1).to(logits.dtype))
        importance = logits.softmax(-1).mean((0, 1))
        load = gates.gt(0).to(logits.dtype).mean((0, 1)) / 2.0
        return gates, self.roles_n * (importance * load).sum()

    def forward(self, image, depth_mm, qpos, actions=None, history=None):
        x = self._rgbd_tokens(image, depth_mm)
        state_vec = self.state(qpos)
        if history is None:
            history = self._online_window(x, qpos)
        summary = self._encode_history(history.to(dtype=x.dtype))
        gates, aux = self._route_with_history(state_vec, x, summary, image.shape[0])
        if actions is not None:
            h = self.posterior(self.action(actions) + self.pos)
            mu, logvar = self.latent(h.mean(1)).chunk(2, -1)
            logvar = logvar.clamp(-10., 5.)
            z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar)
        else:
            mu = logvar = None
            z = torch.zeros((image.shape[0], self.z_proj.in_features), device=image.device)
        memory = torch.cat((state_vec.unsqueeze(1), self.z_proj(z).unsqueeze(1), x), dim=1)
        decoded = self.decoder(self.query.expand(image.shape[0], -1, -1), memory, x, gates)
        return self.out(decoded), mu, logvar, aux, self.last_belief
