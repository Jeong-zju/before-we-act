from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    horizon: int = 16
    action_dim: int = 8
    qpos_dim: int = 9
    candidates: int = 6
    tasks: int = 4
    hidden: int = 768
    image_size: int = 224
    history: int = 16
    history_dim: int = 256


class CAREPolicy(nn.Module):
    """Shared strict-local proposal/response/scoring policy used independently by every arm."""

    def __init__(self, config: ModelConfig = ModelConfig(), pretrained: bool = False):
        super().__init__(); self.config = config
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.vision = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.task = nn.Embedding(config.tasks, 64)
        self.history = nn.GRU(17, config.history_dim, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(768 + config.qpos_dim + 64 + config.history_dim, config.hidden), nn.LayerNorm(config.hidden), nn.GELU(), nn.Linear(config.hidden, config.hidden), nn.GELU())
        total = config.horizon * config.action_dim
        self.reference = nn.Linear(config.hidden, total)
        self.offsets = nn.Linear(config.hidden, (config.candidates - 1) * total)
        self.candidate_encoder = nn.Sequential(nn.Linear(total, 256), nn.GELU(), nn.Linear(256, 256))
        self.response = nn.Sequential(nn.Linear(config.hidden + 256, 384), nn.GELU(), nn.Linear(384, config.qpos_dim))
        self.score = nn.Sequential(nn.Linear(config.hidden + 256, 384), nn.GELU(), nn.Linear(384, 1))

    def forward(self, image, qpos, task_id, history=None, history_mask=None):
        cfg = self.config
        visual = self.pool(self.vision(image)).flatten(1)
        if history is None:
            history = torch.zeros(len(image), cfg.history, 17, device=image.device, dtype=qpos.dtype)
            history[:, -1, :cfg.qpos_dim] = qpos
        if history_mask is not None:
            history = history * history_mask[..., None]
        _sequence, hidden = self.history(history.float())
        feature = self.fuse(torch.cat((visual, qpos, self.task(task_id), hidden[-1].to(qpos.dtype)), -1))
        reference = self.reference(feature).view(-1, 1, cfg.horizon, cfg.action_dim)
        offsets = self.offsets(feature).view(-1, cfg.candidates - 1, cfg.horizon, cfg.action_dim)
        candidates = torch.cat((reference, reference + 0.25 * torch.tanh(offsets)), 1)
        encoded = self.candidate_encoder(candidates.flatten(2))
        fused = torch.cat((feature[:, None].expand(-1, cfg.candidates, -1), encoded), -1)
        return {"candidates": candidates, "scores": self.score(fused).squeeze(-1), "responses": self.response(fused)}

    def loss_from_output(self, output, batch):
        candidates, scores = output["candidates"], output["scores"]
        target = batch["actions"][:, None]
        errors = (candidates - target).square().mean((2, 3))
        oracle = errors.argmin(1)
        rows = torch.arange(len(oracle), device=oracle.device)
        response = output["responses"][rows, oracle]
        base = F.smooth_l1_loss(candidates[:, 0], target[:, 0])
        proposal = errors.min(1).values.mean()
        score_target = (-errors.detach() / 0.25).clamp(-20, 0)
        scoring = F.smooth_l1_loss(scores, score_target) + F.cross_entropy(scores, oracle)
        response_loss = F.smooth_l1_loss(response, batch["future_delta"])
        total = base + 0.5 * proposal + 0.25 * scoring + 0.2 * response_loss
        return total, {"loss": total.detach(), "base": base.detach(), "proposal": proposal.detach(), "scoring": scoring.detach(), "response": response_loss.detach()}

    def loss(self, batch):
        return self.loss_from_output(self(batch["image"], batch["qpos"], batch["task_id"], batch["history"], batch["history_mask"]), batch)

    @torch.no_grad()
    def act(self, image, qpos, task_id, history=None, history_mask=None):
        output = self(image, qpos, task_id, history, history_mask)
        selected = output["scores"].argmax(1)
        return output["candidates"][torch.arange(len(selected), device=selected.device), selected], selected

    def config_dict(self): return asdict(self.config)


class LegacyCAREPolicy(nn.Module):
    """Read-only loader for the v1 absolute-action checkpoint."""

    def __init__(self, config: ModelConfig):
        super().__init__(); self.config = config
        from torchvision.models import convnext_tiny
        backbone = convnext_tiny(weights=None)
        self.vision = backbone.features; self.pool = nn.AdaptiveAvgPool2d(1)
        self.task = nn.Embedding(config.tasks, 64)
        self.fuse = nn.Sequential(nn.Linear(768 + config.qpos_dim + 64, config.hidden), nn.LayerNorm(config.hidden), nn.GELU(), nn.Linear(config.hidden, config.hidden), nn.GELU())
        total = config.horizon * config.action_dim
        self.reference = nn.Linear(config.hidden, total)
        self.offsets = nn.Linear(config.hidden, (config.candidates - 1) * total)
        self.candidate_encoder = nn.Sequential(nn.Linear(total, 256), nn.GELU(), nn.Linear(256, 256))
        self.response = nn.Sequential(nn.Linear(config.hidden + 256, 384), nn.GELU(), nn.Linear(384, config.qpos_dim))
        self.score = nn.Sequential(nn.Linear(config.hidden + 256, 384), nn.GELU(), nn.Linear(384, 1))

    def forward(self, image, qpos, task_id, history=None, history_mask=None):
        cfg = self.config; visual = self.pool(self.vision(image)).flatten(1)
        feature = self.fuse(torch.cat((visual, qpos, self.task(task_id)), -1))
        reference = self.reference(feature).view(-1, 1, cfg.horizon, cfg.action_dim)
        offsets = self.offsets(feature).view(-1, cfg.candidates - 1, cfg.horizon, cfg.action_dim)
        candidates = torch.cat((reference, reference + 0.25 * torch.tanh(offsets)), 1)
        encoded = self.candidate_encoder(candidates.flatten(2))
        fused = torch.cat((feature[:, None].expand(-1, cfg.candidates, -1), encoded), -1)
        return {"candidates": candidates, "scores": self.score(fused).squeeze(-1), "responses": self.response(fused)}

    @torch.no_grad()
    def act(self, image, qpos, task_id, history=None, history_mask=None):
        output = self(image, qpos, task_id)
        selected = output["scores"].argmax(1)
        return output["candidates"][torch.arange(len(selected), device=selected.device), selected], selected
