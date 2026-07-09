from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math
import torch


@dataclass
class CommunicationConfig:
    codebook_size: int = 64
    residual_dim: int = 64
    residual_bits: int = 8
    envelope_bits: int = 32
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


def make_config_from_args(args) -> CommunicationConfig:
    return CommunicationConfig(
        codebook_size=args.codebook_size,
        residual_dim=args.residual_dim,
        residual_bits=args.residual_bits,
        envelope_bits=args.envelope_bits,
        uncertainty_bits=args.uncertainty_bits,
        lambda_bits=args.lambda_bits,
        lambda_delay=args.lambda_delay,
        lambda_redundancy=args.lambda_redundancy,
        delay_steps=args.delay_steps,
        delta_margin=args.delta_margin,
    )
