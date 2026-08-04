"""PAIR: Predictive Agent-Interaction Representation.

The deployment policy is strictly local.  Synchronised peer actions and future
local effects are available only to the training-time teacher.  Unlike the
previous predictability pilot, PAIR never changes Local-ARCA's action-role
router.  A zero-initialised low-rank residual adapter can only modulate the
decoded action tokens, so update zero is exactly the measured Local-ARCA policy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo_decoder_variants import StereoARCA


class StereoPAIRResidual(StereoARCA):
    """PAIR-conditioned interaction residual over a Local-ARCA policy.

    ``joint_training=False`` reproduces the conservative frozen-checkpoint
    probe.  ``joint_training=True`` is the formal from-scratch PAIR policy:
    the same local observation/action path, local interaction student and
    residual adapter are optimized jointly while the synchronized teacher
    remains training-only.
    """

    def __init__(
        self, *args, event_dim=96, residual_rank=32, joint_training=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        d = self.query.shape[-1]
        self.event_dim = event_dim
        self.residual_rank = residual_rank
        self.local_event_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, event_dim),
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )
        self.residual_down = nn.Linear(d, residual_rank, bias=False)
        self.event_scale = nn.Linear(event_dim, residual_rank)
        self.query_scale = nn.Parameter(torch.zeros(self.query.shape[1], residual_rank))
        self.residual_up = nn.Linear(residual_rank, d, bias=False)
        # Hard baseline-preservation property: before training, PAIR produces
        # exactly Local-ARCA's decoded hidden state and action chunk.
        nn.init.zeros_(self.residual_up.weight)
        nn.init.zeros_(self.event_scale.weight)
        nn.init.zeros_(self.event_scale.bias)
        self.last_local_event = None
        self.last_confidence = None
        self.last_gates = None
        self.last_residual_norm = None
        self.joint_training = bool(joint_training)

    def freeze_local_arca(self):
        """Freeze every parameter inherited from Local-ARCA.

        New PAIR modules remain trainable.  This makes role-route preservation
        structural rather than a soft regularisation preference.
        """
        new_prefixes = (
            "local_event_head.",
            "confidence_head.",
            "residual_down.",
            "event_scale.",
            "residual_up.",
        )
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name == "query_scale" or name.startswith(new_prefixes)

    def forward(self, image, depth_mm, qpos, actions=None):
        def base_forward():
            observation = self._rgbd_tokens(image, depth_mm)
            state_vector = self.state(qpos)
            gates, auxiliary = self._route(state_vector, observation, image.shape[0])
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
            base_action = self.out(decoded)

            return observation, state_vector, gates, auxiliary, decoded, base_action, mu, logvar

        # The conservative probe structurally freezes Local-ARCA.  The formal
        # model executes the identical graph with gradients enabled.
        if self.joint_training:
            observation, state_vector, gates, auxiliary, decoded, base_action, mu, logvar = base_forward()
        else:
            with torch.no_grad():
                observation, state_vector, gates, auxiliary, decoded, base_action, mu, logvar = base_forward()

        context = (
            self.route_state(state_vector if self.joint_training else state_vector.detach())
            + self.route_observation(
                (observation if self.joint_training else observation.detach()).mean(1)
            )
        )
        local_event = F.normalize(
            self.local_event_head(context).float(), dim=-1, eps=1e-6
        )
        confidence = torch.sigmoid(self.confidence_head(context).float())
        scale = torch.sigmoid(
            self.event_scale(local_event).unsqueeze(1)
            + self.query_scale.unsqueeze(0)
        )
        residual_hidden = self.residual_up(
            F.gelu(
                self.residual_down(decoded if self.joint_training else decoded.detach())
            )
            * scale.to(decoded.dtype)
        )
        residual_hidden = confidence.to(decoded.dtype).unsqueeze(1) * residual_hidden
        action = self.out(
            (decoded if self.joint_training else decoded.detach()) + residual_hidden
        )

        self.last_local_event = local_event
        self.last_confidence = confidence
        self.last_gates = gates
        # Epsilon avoids the undefined d(sqrt(x))/dx at the deliberately
        # zero-initialised residual while preserving the reported RMS value.
        self.last_residual_norm = (
            residual_hidden.float().square().mean() + 1e-12
        ).sqrt()
        return (
            action,
            mu,
            logvar,
            auxiliary,
            local_event,
            confidence,
            base_action,
            gates,
        )


class PairwiseLagInteractionTeacher(nn.Module):
    """Training-only pairwise, directed and lag-selective interaction teacher.

    Each target arm attends over all (peer, lag) candidates in its synchronised
    unordered team.  The target's future *local* effect is privileged training
    supervision, never a deployment input.
    """

    def __init__(
        self,
        action_dim,
        effect_dim,
        lags,
        event_dim=96,
        width=192,
    ):
        super().__init__()
        self.effect_dim = effect_dim
        self.lags = tuple(int(value) for value in lags)
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
        self.lag_embedding = nn.Embedding(len(self.lags), width)
        self.pair_value = nn.Sequential(
            nn.LayerNorm(4 * width),
            nn.Linear(4 * width, width),
            nn.GELU(),
            nn.Linear(width, event_dim),
        )
        self.pair_score = nn.Sequential(
            nn.LayerNorm(4 * width),
            nn.Linear(4 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.own_effect = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, len(self.lags) * effect_dim),
        )
        self.interaction_effect = nn.Sequential(
            nn.Linear(event_dim, width),
            nn.GELU(),
            nn.Linear(width, len(self.lags) * effect_dim),
        )

    def action_features(self, actions):
        delta = actions[:, 1:] - actions[:, :-1]
        encoded = self.action_step(delta.float())
        return self.action_feature(
            torch.cat((encoded.mean(1), encoded.square().mean(1).sqrt()), dim=-1)
        )

    def forward(self, actions, future_effects, groups):
        """Return event, predictive-gain confidence target and diagnostics."""
        own = self.action_features(actions)
        effect_features = self.effect_feature(future_effects.float())
        events = torch.zeros(
            (len(actions), self.interaction_effect[0].in_features),
            device=actions.device,
            dtype=torch.float32,
        )
        attention_entropy = []
        for group in groups.unique(sorted=True):
            ids = (groups == group).nonzero(as_tuple=False).flatten()
            values = own.index_select(0, ids)
            group_effects = effect_features.index_select(0, ids)
            for local_index, target_id in enumerate(ids):
                candidates, scores = [], []
                target = values[local_index]
                peer_indices = [
                    index for index in range(len(ids)) if index != local_index
                ]
                if not peer_indices:
                    continue
                for peer_index in peer_indices:
                    peer = values[peer_index]
                    for lag_index in range(len(self.lags)):
                        effect = group_effects[local_index, lag_index]
                        lag = self.lag_embedding.weight[lag_index]
                        pair = torch.cat(
                            (target, peer, target - peer, effect + lag), dim=-1
                        )
                        candidates.append(self.pair_value(pair))
                        scores.append(self.pair_score(pair))
                score = torch.cat(scores, dim=0)
                weight = score.softmax(dim=0)
                candidate = torch.stack(candidates, dim=0)
                event = (weight.unsqueeze(-1) * candidate).sum(0)
                events[target_id] = event.float()
                attention_entropy.append(
                    -(weight.clamp_min(1e-8).log() * weight).sum()
                )

        teacher_event = F.normalize(events, dim=-1, eps=1e-6)
        target = future_effects.float().flatten(1)
        own_prediction = self.own_effect(own)
        full_prediction = own_prediction + self.interaction_effect(teacher_event)
        own_error = (own_prediction - target).square().mean(-1)
        full_error = (full_prediction - target).square().mean(-1)
        # Only the locally consequential part receives a strong deployment
        # confidence target.  Detaching prevents the teacher from gaming it.
        gain = ((own_error - full_error) / own_error.clamp_min(1e-6)).clamp(0, 1).detach()
        zero = own_error.new_zeros(())
        entropy = torch.stack(attention_entropy).mean() if attention_entropy else zero
        return (
            teacher_event,
            gain.unsqueeze(-1),
            own_error.mean(),
            full_error.mean(),
            entropy,
        )


def interaction_alignment(student_event, teacher_event, temperature=0.15):
    student = F.normalize(student_event.float(), dim=-1, eps=1e-6)
    teacher = F.normalize(teacher_event.detach().float(), dim=-1, eps=1e-6)
    logits = student @ teacher.t() / temperature
    target = torch.arange(len(student), device=student.device)
    return 0.5 * (
        F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)
    )
