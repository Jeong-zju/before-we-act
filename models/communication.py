from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math
import torch

from models.free_energy import (
    multi_hypothesis_expected_free_energy,
    normalize_hypothesis_weights,
)


@dataclass
class CommunicationConfig:
    codebook_size: int = 64
    residual_dim: int = 64
    residual_bits: int = 8
    envelope_bits: int = 32
    request_envelope_bits: int = 32
    uncertainty_bits: int = 8
    lambda_bits: float = 1e-4
    lambda_delay: float = 0.05
    lambda_redundancy: float = 0.1
    delay_steps: float = 1.0
    delta_margin: float = 0.0


class CommunicationTrigger:
    def __init__(self, cfg: CommunicationConfig):
        self.cfg = cfg

    def message_bits(self) -> int:
        code_bits = int(math.ceil(math.log2(max(2, self.cfg.codebook_size))))
        return int(
            code_bits
            + self.cfg.residual_dim * self.cfg.residual_bits
            + self.cfg.envelope_bits
            + self.cfg.uncertainty_bits
        )

    def communication_cost(self, redundancy: torch.Tensor | None = None, device=None, dtype=None) -> torch.Tensor:
        if redundancy is None:
            redundancy = torch.tensor(0.0, device=device, dtype=dtype or torch.float32)
        bits = torch.as_tensor(float(self.message_bits()), device=redundancy.device, dtype=redundancy.dtype)
        delay = torch.as_tensor(float(self.cfg.delay_steps), device=redundancy.device, dtype=redundancy.dtype)
        return (
            self.cfg.lambda_bits * bits
            + self.cfg.lambda_delay * delay
            + self.cfg.lambda_redundancy * redundancy
        )

    def redundancy_score(self, inferred_code: torch.Tensor, message_code: torch.Tensor) -> torch.Tensor:
        # Redundant if inferred discrete belief already matches the communicated code.
        return (inferred_code == message_code).float()

    def decide(
        self,
        G_no_comm: torch.Tensor,
        G_comm: torch.Tensor,
        inferred_code: torch.Tensor,
        message_code: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        delta_G = G_no_comm - G_comm
        redundancy = self.redundancy_score(inferred_code, message_code)
        C_comm = self.communication_cost(redundancy=redundancy)
        trigger = delta_G > (C_comm + self.cfg.delta_margin)
        return {
            "trigger": trigger,
            "delta_G": delta_G,
            "C_comm": C_comm,
            "redundancy": redundancy,
            "bits": torch.full_like(delta_G, float(self.message_bits())),
        }


class VPICommunicationTrigger:
    """Request information using value of perfect information.

    Unlike the legacy post-hoc ``CommunicationTrigger.decide`` API, this
    pre-request decision never accepts a true/message plan code.  It compares
    the best posterior-robust action with the action that could be chosen after
    a hypothetical perfect reply.
    """

    def __init__(self, cfg: CommunicationConfig):
        self.cfg = cfg
        self._cost_model = CommunicationTrigger(cfg)

    def message_bits(self) -> int:
        """Legacy-compatible reply payload size (code + residual + envelope)."""
        return self._cost_model.message_bits()

    def request_bits(self) -> int:
        return int(self.cfg.request_envelope_bits)

    def reply_bits(self) -> int:
        return self.message_bits()

    def round_trip_bits(self) -> int:
        return self.request_bits() + self.reply_bits()

    def request_cost(self, reference: torch.Tensor) -> torch.Tensor:
        # Reply redundancy is unknowable before requesting.  Do not inspect a
        # privileged reply code.  Charge both request-envelope and reply-plan
        # bits; cfg.delay_steps is interpreted as the configured round-trip
        # planning delay.
        bits = torch.full_like(reference, float(self.round_trip_bits()))
        delay = torch.full_like(reference, float(self.cfg.delay_steps))
        return self.cfg.lambda_bits * bits + self.cfg.lambda_delay * delay

    def decide_request(
        self,
        hypothesis_G: torch.Tensor,
        hypothesis_weights: torch.Tensor,
        request_cost: torch.Tensor | float | None = None,
    ) -> Dict[str, torch.Tensor]:
        aggregate = multi_hypothesis_expected_free_energy(hypothesis_G, hypothesis_weights)
        vpi = aggregate["VPI"]

        if request_cost is None:
            cost = self.request_cost(vpi)
        else:
            cost = torch.as_tensor(request_cost, device=vpi.device, dtype=vpi.dtype)
            try:
                cost = torch.broadcast_to(cost, vpi.shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"request_cost shape {tuple(cost.shape)} is not broadcastable to {tuple(vpi.shape)}"
                ) from exc
            if not torch.isfinite(cost).all() or (cost < 0).any():
                raise ValueError("request_cost must be finite and non-negative")

        trigger = vpi > (cost + self.cfg.delta_margin)
        return {
            **aggregate,
            "trigger": trigger,
            "delta_G": vpi,
            "C_comm": cost,
            "bits": torch.full_like(vpi, float(self.round_trip_bits())),
            "request_bits": torch.full_like(vpi, float(self.request_bits())),
            "reply_bits": torch.full_like(vpi, float(self.reply_bits())),
        }

    # A concise alias for callers that treat trigger objects uniformly.
    decide = decide_request


def reply_plan_diagnostics(
    prior_code_probabilities: torch.Tensor,
    reply_code: torch.Tensor,
    prior_plan_index: torch.Tensor | None = None,
    revised_plan_index: torch.Tensor | None = None,
    prior_actions: torch.Tensor | None = None,
    revised_actions: torch.Tensor | None = None,
    prior_residual_mu_by_code: torch.Tensor | None = None,
    prior_residual_logvar_by_code: torch.Tensor | None = None,
    reply_residual: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Measure posterior surprise and behavioral change after a real reply.

    This function belongs after a request/reply has completed.  It is not used
    by ``VPICommunicationTrigger`` and therefore cannot leak the reply into the
    pre-request decision.
    """
    if prior_code_probabilities.ndim != 2:
        raise ValueError("prior_code_probabilities must have shape [B, codebook_size]")
    probs = normalize_hypothesis_weights(prior_code_probabilities)
    B, codebook_size = probs.shape
    code = reply_code.to(device=probs.device, dtype=torch.long).reshape(-1)
    if code.shape[0] != B:
        raise ValueError(f"reply_code must contain one code per batch item, expected {B}")
    if (code < 0).any() or (code >= codebook_size).any():
        raise ValueError(f"reply_code must be in [0, {codebook_size - 1}]")

    replied_probability = probs.gather(1, code[:, None]).squeeze(1)
    code_surprise = -replied_probability.clamp_min(eps).log()
    out: Dict[str, torch.Tensor] = {
        "reply_probability": replied_probability,
        "code_surprise": code_surprise,
    }

    residual_args = (
        prior_residual_mu_by_code,
        prior_residual_logvar_by_code,
        reply_residual,
    )
    if any(value is not None for value in residual_args) and not all(
        value is not None for value in residual_args
    ):
        raise ValueError(
            "prior_residual_mu_by_code, prior_residual_logvar_by_code, and "
            "reply_residual must be provided together"
        )
    if prior_residual_mu_by_code is not None:
        mu_all = prior_residual_mu_by_code.to(device=probs.device)
        logvar_all = prior_residual_logvar_by_code.to(device=probs.device, dtype=mu_all.dtype)
        residual = reply_residual.to(device=probs.device, dtype=mu_all.dtype)
        if mu_all.ndim != 3 or mu_all.shape[:2] != (B, codebook_size):
            raise ValueError("prior_residual_mu_by_code must have shape [B, codebook_size, D]")
        if logvar_all.shape != mu_all.shape or residual.shape != (B, mu_all.shape[-1]):
            raise ValueError(
                "residual log-variance must match means and reply_residual must have shape [B, D]"
            )
        gather = code[:, None, None].expand(B, 1, mu_all.shape[-1])
        mu = mu_all.gather(1, gather).squeeze(1)
        logvar = logvar_all.gather(1, gather).squeeze(1).clamp(min=-20.0, max=20.0)
        standardized_sq_error = (residual - mu).pow(2) * torch.exp(-logvar)
        residual_surprise = 0.5 * (
            math.log(2.0 * math.pi) + logvar + standardized_sq_error
        ).sum(dim=-1)
        out["residual_surprise"] = residual_surprise
        out["residual_mahalanobis_sq"] = standardized_sq_error.sum(dim=-1)
        out["plan_surprise"] = code_surprise + residual_surprise
    else:
        out["residual_surprise"] = torch.zeros_like(code_surprise)
        out["residual_mahalanobis_sq"] = torch.zeros_like(code_surprise)
        out["plan_surprise"] = code_surprise

    if (prior_plan_index is None) != (revised_plan_index is None):
        raise ValueError("prior_plan_index and revised_plan_index must be provided together")
    if prior_plan_index is not None:
        before = prior_plan_index.to(device=probs.device, dtype=torch.long).reshape(-1)
        after = revised_plan_index.to(device=probs.device, dtype=torch.long).reshape(-1)
        if before.shape[0] != B or after.shape[0] != B:
            raise ValueError("plan indices must contain one value per batch item")
        out["replanned"] = before != after
        out["prior_plan_index"] = before
        out["revised_plan_index"] = after

    if (prior_actions is None) != (revised_actions is None):
        raise ValueError("prior_actions and revised_actions must be provided together")
    if prior_actions is not None:
        before_actions = prior_actions.to(device=probs.device)
        after_actions = revised_actions.to(device=probs.device, dtype=before_actions.dtype)
        if before_actions.shape != after_actions.shape or before_actions.shape[0] != B:
            raise ValueError("prior_actions and revised_actions must share shape [B, ...]")
        delta = after_actions - before_actions
        flat = delta.reshape(B, -1)
        out["action_change_l2"] = torch.linalg.vector_norm(flat, dim=1)
        out["action_change_mean_abs"] = flat.abs().mean(dim=1)
        out["action_change_max_abs"] = flat.abs().amax(dim=1)

    return out


def make_config_from_args(args) -> CommunicationConfig:
    return CommunicationConfig(
        codebook_size=args.codebook_size,
        residual_dim=args.residual_dim,
        residual_bits=args.residual_bits,
        envelope_bits=args.envelope_bits,
        request_envelope_bits=getattr(args, "request_envelope_bits", 32),
        uncertainty_bits=args.uncertainty_bits,
        lambda_bits=args.lambda_bits,
        lambda_delay=args.lambda_delay,
        lambda_redundancy=args.lambda_redundancy,
        delay_steps=args.delay_steps,
        delta_margin=args.delta_margin,
    )
