"""Commitment-consistent decentralized planner for Research-v2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import torch

from models.research_v2 import (
    BlockTransitionWorldModelV2,
    IntentionPosteriorV2,
    PlanProposalV2,
    PlanTokenizerV2,
    decode_plan_batch,
)
from models.research_v2_decision import (
    CalibrationV2,
    RiskV2Config,
    calibrated_posterior_probabilities,
    candidate_hypothesis_risk,
    counterfactual_vpi,
)


@dataclass(frozen=True)
class MessageCodecV2:
    residual_dim: int = 16
    payload_dim: int = 16
    quantization_scale: float = 32.0

    def __post_init__(self) -> None:
        if self.residual_dim <= 0 or self.payload_dim not in (0, 8, 16):
            raise ValueError("V2 message payload must be code-only, 8D, or 16D")
        if self.payload_dim > self.residual_dim:
            raise ValueError("payload cannot exceed the tokenizer residual")
        if self.quantization_scale <= 0:
            raise ValueError("quantization_scale must be positive")

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape[-1] != self.residual_dim:
            raise ValueError("residual dimension differs from codec")
        return (
            residual[..., : self.payload_dim]
            .mul(self.quantization_scale)
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
        )

    def decode(
        self,
        code: torch.Tensor,
        payload: torch.Tensor,
        residual_prior_by_code: torch.Tensor,
    ) -> torch.Tensor:
        prior = residual_prior_by_code.to(payload.device)[code.long()].clone()
        if payload.shape[-1] != self.payload_dim:
            raise ValueError("message payload dimension differs from codec")
        if self.payload_dim:
            prior[..., : self.payload_dim] = payload.to(prior.dtype) / self.quantization_scale
        return prior

    def canonicalize(
        self,
        code: torch.Tensor,
        residual: torch.Tensor,
        residual_prior_by_code: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(code, self.encode(residual), residual_prior_by_code)

    @property
    def reply_bits(self) -> int:
        # 6-bit code, quantized int8 payload, and the existing 72-bit envelope.
        return 6 + 8 * self.payload_dim + 72


@dataclass(frozen=True)
class PlanMessageV2:
    sender_id: int
    sequence: int
    episode_sequence: int
    step: int
    valid_until_step: int
    code: int
    residual_payload: torch.Tensor

    def __post_init__(self) -> None:
        if self.sender_id not in (0, 1) or min(self.sequence, self.episode_sequence, self.step) < 0:
            raise ValueError("invalid plan-message envelope")
        if self.valid_until_step < self.step or self.code < 0:
            raise ValueError("invalid plan-message validity/code")
        payload = torch.as_tensor(self.residual_payload).detach().clone()
        if payload.ndim != 1 or payload.dtype != torch.int8:
            raise ValueError("residual payload must be one int8 vector")
        object.__setattr__(self, "residual_payload", payload)


@dataclass(frozen=True)
class PlannerV2Config:
    num_candidates: int = 8
    num_hypotheses: int = 4
    cooldown_steps: int = 8
    # A reply commits the responder's current action only.  Multi-step TTLs
    # require an explicit shifted commitment queue and are intentionally not
    # advertised until that runtime contract exists.
    plan_valid_steps: int = 0
    communication_cost: float = 0.0
    action_clip: float = 1.0
    unmodeled_tail_penalty: float = 0.5
    residual_sigma_scale: float = 0.5

    def __post_init__(self) -> None:
        if min(self.num_candidates, self.num_hypotheses) <= 0:
            raise ValueError("candidate/hypothesis counts must be positive")
        if self.plan_valid_steps != 0:
            raise ValueError(
                "Research-v2 currently supports current-step plan messages only"
            )
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative")
        if self.action_clip <= 0:
            raise ValueError("action_clip must be positive")
        if self.unmodeled_tail_penalty < 0:
            raise ValueError("unmodeled_tail_penalty must be non-negative")
        if self.residual_sigma_scale < 0:
            raise ValueError("residual_sigma_scale must be non-negative")


@dataclass(frozen=True)
class PreparedPlanV2:
    agent_id: int
    step: int
    request: bool
    provisional_index: int
    provisional_message: PlanMessageV2
    vpi: float


@dataclass(frozen=True)
class LocalDecisionV2:
    agent_id: int
    step: int
    action: torch.Tensor
    plan_code: int
    plan_residual: torch.Tensor
    request_sent: bool
    reply_received: bool
    locked_as_responder: bool
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class PairDecisionV2:
    joint_action: torch.Tensor
    agents: tuple[LocalDecisionV2, LocalDecisionV2]
    routed_messages: int
    requester: int | None


@dataclass
class _PendingV2:
    belief: torch.Tensor
    codes: torch.Tensor
    residuals: torch.Tensor
    actions: torch.Tensor
    prepared: PreparedPlanV2
    risk: dict[str, torch.Tensor]
    vpi: dict[str, torch.Tensor]
    posterior_diagnostics: Mapping[str, object]


class DeterministicRequestArbiterV2:
    """Content-blind arbitration independently reproducible on both robots."""

    @staticmethod
    def requester(request_bits: Mapping[int, bool], episode_sequence: int, step: int) -> int | None:
        if set(request_bits) != {0, 1}:
            raise ValueError("arbitration requires request bits for agents 0 and 1")
        active = [agent for agent in (0, 1) if bool(request_bits[agent])]
        if not active:
            return None
        if len(active) == 1:
            return active[0]
        return int((int(episode_sequence) + int(step)) & 1)


class LocalPlannerV2:
    """One strictly local planner; model instances need not be shared."""

    def __init__(
        self,
        agent_id: int,
        *,
        tokenizer: PlanTokenizerV2,
        proposal: PlanProposalV2,
        intention: IntentionPosteriorV2,
        world_ensemble: Sequence[BlockTransitionWorldModelV2],
        active_code_mask: torch.Tensor,
        residual_prior_by_code: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        artifact_hash: str,
        codec: MessageCodecV2 | None = None,
        config: PlannerV2Config | None = None,
        risk_config: RiskV2Config | None = None,
        calibration: CalibrationV2 | None = None,
        epistemic_available: bool | None = None,
    ) -> None:
        if agent_id not in (0, 1) or not world_ensemble:
            raise ValueError("planner requires agent 0/1 and a non-empty world ensemble")
        self.agent_id = int(agent_id)
        self.tokenizer = tokenizer.eval()
        self.proposal = proposal.eval()
        self.intention = intention.eval()
        self.world_ensemble = tuple(model.eval() for model in world_ensemble)
        self.active_code_mask = active_code_mask.bool()
        self.residual_prior_by_code = residual_prior_by_code
        self.action_mean = action_mean
        self.action_std = action_std
        self.artifact_hash = str(artifact_hash)
        self.codec = codec or MessageCodecV2()
        self.calibration = calibration or CalibrationV2()
        resolved_config = config or PlannerV2Config()
        if self.calibration.communication_price_frozen:
            assert self.calibration.communication_price is not None
            resolved_config = replace(
                resolved_config,
                communication_cost=float(self.calibration.communication_price),
            )
        self.config = resolved_config
        self.risk_config = risk_config or RiskV2Config()
        self.epistemic_available = (
            len(self.world_ensemble) >= 2
            if epistemic_available is None
            else bool(epistemic_available)
        )
        if self.epistemic_available and len(self.world_ensemble) < 2:
            raise ValueError("epistemic availability requires at least two world models")
        if self.active_code_mask.ndim != 1 or not bool(self.active_code_mask.any()):
            raise ValueError("planner requires a non-empty one-dimensional active code mask")
        active_count = int(self.active_code_mask.sum().item())
        if self.config.num_candidates > active_count:
            raise ValueError("num_candidates exceeds the active plan-code count")
        if self.config.num_hypotheses > active_count:
            raise ValueError("num_hypotheses exceeds the active plan-code count")
        self.reset()

    def reset(self, *, episode_sequence: int = 0) -> None:
        self.episode_sequence = int(episode_sequence)
        self.step = 0
        self.sequence = 0
        self.last_request_step: int | None = None
        self.last_received: PlanMessageV2 | None = None
        self._pending: _PendingV2 | None = None

    def _cooldown(self) -> bool:
        return self.last_request_step is not None and self.step - self.last_request_step < self.config.cooldown_steps

    def _world_grid(
        self,
        belief: torch.Tensor,
        ego_actions: torch.Tensor,
        peer_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, H, A = ego_actions.shape
        M = peer_actions.shape[1]
        own = ego_actions[:, :, None].expand(B, K, M, H, A).reshape(B * K * M, H, A)
        peer = peer_actions[:, None].expand(B, K, M, H, A).reshape(B * K * M, H, A)
        state = belief[:, None, None].expand(B, K, M, *belief.shape[1:]).reshape(
            B * K * M, *belief.shape[1:]
        )
        quantiles, constraints = [], []
        with torch.no_grad():
            for model in self.world_ensemble:
                out = model(state, own, peer)
                quantiles.append(out["return_quantiles"].reshape(B, K, M, -1))
                constraints.append(out["constraint_logits"].reshape(B, K, M))
        return torch.stack(quantiles), torch.stack(constraints)

    def _risk(
        self,
        quantiles: torch.Tensor,
        constraints: torch.Tensor,
        actions: torch.Tensor,
        *,
        residual_variance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return candidate_hypothesis_risk(
            ensemble_return_quantiles=quantiles,
            ensemble_constraint_logits=constraints,
            ego_actions=actions,
            hypothesis_residual_variance=residual_variance,
            config=self.risk_config,
            calibration=self.calibration,
            epistemic_available=self.epistemic_available,
        )

    def _marginal_peer_hypotheses(
        self,
        belief: torch.Tensor,
        ego_id: torch.Tensor,
        proposals: Mapping[str, torch.Tensor],
        codes: torch.Tensor,
        residuals: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Mapping[str, object],
    ]:
        """Marginalize q(peer | belief, candidate) over all ego candidates.

        This is a single batched intention forward pass.  It removes the old
        top-1 conditioning bug while retaining a common peer-hypothesis support
        across candidates, which is required for a valid expectation-of-min
        VPI calculation.
        """

        B, K = codes.shape
        C = self.active_code_mask.numel()
        D = residuals.shape[-1]
        device = belief.device
        active = self.active_code_mask.to(device)

        proposal_logits = proposals["code_logits"].masked_fill(
            ~active.reshape(1, C), -torch.inf
        )
        proposal_probabilities = proposal_logits.softmax(dim=-1)
        candidate_mass = proposal_probabilities.gather(1, codes)
        candidate_coverage = candidate_mass.sum(dim=-1)
        candidate_weights = candidate_mass / candidate_coverage.unsqueeze(-1).clamp_min(1e-8)

        candidate_belief = belief[:, None].expand(B, K, *belief.shape[1:]).reshape(
            B * K, *belief.shape[1:]
        )
        candidate_ego_id = ego_id[:, None].expand(B, K).reshape(B * K)
        metadata = belief.new_zeros(B * K, self.intention.message_metadata_dim)
        posterior = self.intention(
            candidate_belief,
            codes.reshape(B * K),
            residuals.reshape(B * K, D),
            candidate_ego_id,
            metadata,
        )
        conditional_probabilities = calibrated_posterior_probabilities(
            posterior["code_logits"].reshape(B, K, C),
            active_code_mask=active,
            calibration=self.calibration,
        )
        conditional_mu = posterior["residual_mu_by_code"].reshape(B, K, C, D)
        conditional_variance = (
            posterior["residual_logvar_by_code"].reshape(B, K, C, D).exp()
            * self.calibration.posterior_variance_scale
        )

        joint = candidate_weights.unsqueeze(-1) * conditional_probabilities
        marginal_probabilities = joint.sum(dim=1)
        denominator = marginal_probabilities.unsqueeze(-1).clamp_min(1e-8)
        marginal_mu = (joint.unsqueeze(-1) * conditional_mu).sum(dim=1) / denominator
        marginal_second_moment = (
            joint.unsqueeze(-1) * (conditional_variance + conditional_mu.square())
        ).sum(dim=1) / denominator
        marginal_variance = (marginal_second_moment - marginal_mu.square()).clamp_min(0.0)

        hypothesis_count = self.config.num_hypotheses
        use_antithetic_residuals = (
            hypothesis_count >= 2 and self.config.residual_sigma_scale > 0
        )
        # Keep the world-grid budget fixed: one code receives an antithetic
        # two-point residual quadrature and the remaining hypotheses represent
        # distinct high-probability codes.  Thus residual uncertainty reaches
        # the nonlinear world model instead of becoming a candidate-invariant
        # scalar that would cancel out of VPI.
        represented_code_count = (
            hypothesis_count - 1 if use_antithetic_residuals else hypothesis_count
        )
        code_weights, represented_codes = marginal_probabilities.topk(
            represented_code_count, dim=-1
        )
        gather = represented_codes.unsqueeze(-1).expand(B, represented_code_count, D)
        represented_mu = marginal_mu.gather(1, gather)
        represented_variance = marginal_variance.gather(1, gather)
        if use_antithetic_residuals:
            dimension = torch.arange(D, device=device).reshape(1, D)
            direction = (
                ((dimension + represented_codes[:, :1]) & 1).to(represented_mu.dtype)
                .mul(2.0)
                .sub(1.0)
                / float(D) ** 0.5
            )
            offset = (
                represented_variance[:, 0].sqrt()
                * direction
                * self.config.residual_sigma_scale
            )
            peer_codes = torch.cat(
                (represented_codes[:, :1], represented_codes[:, :1], represented_codes[:, 1:]),
                dim=1,
            )
            peer_residuals = torch.cat(
                (
                    represented_mu[:, :1] + offset.unsqueeze(1),
                    represented_mu[:, :1] - offset.unsqueeze(1),
                    represented_mu[:, 1:],
                ),
                dim=1,
            )
            weights = torch.cat(
                (code_weights[:, :1] * 0.5, code_weights[:, :1] * 0.5, code_weights[:, 1:]),
                dim=1,
            )
            peer_variance = torch.cat(
                (
                    represented_variance[:, :1].mean(dim=-1),
                    represented_variance[:, :1].mean(dim=-1),
                    represented_variance[:, 1:].mean(dim=-1),
                ),
                dim=1,
            )
        else:
            peer_codes = represented_codes
            peer_residuals = represented_mu
            weights = code_weights
            peer_variance = represented_variance.mean(dim=-1)
        peer_residuals = self.codec.canonicalize(
            peer_codes, peer_residuals, self.residual_prior_by_code.to(device)
        )
        tail_weight = (1.0 - code_weights.sum(dim=-1)).clamp(0.0, 1.0)
        entropy = -(
            marginal_probabilities
            * marginal_probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        diagnostics: Mapping[str, object] = {
            "intention_conditioning": "proposal_marginalized_over_all_candidates",
            "intention_candidate_count": K,
            "proposal_candidate_coverage": float(candidate_coverage[0].item()),
            "posterior_top_m_coverage": float(weights[0].sum().item()),
            "posterior_tail_probability": float(tail_weight[0].item()),
            "posterior_entropy": float(entropy[0].item()),
            "posterior_mean_residual_variance": float(peer_variance[0].mean().item()),
            "posterior_distinct_code_hypotheses": represented_code_count,
            "posterior_antithetic_residual_hypotheses": int(use_antithetic_residuals) * 2,
        }
        return (
            peer_codes,
            peer_residuals,
            weights,
            tail_weight,
            peer_variance,
            diagnostics,
        )

    @torch.inference_mode()
    def prepare(self, belief: torch.Tensor) -> PreparedPlanV2:
        if self._pending is not None:
            raise RuntimeError("prepare called twice without finalize")
        if belief.ndim == 2:
            belief = belief.unsqueeze(0)
        if belief.shape[0] != 1:
            raise ValueError("local online planner accepts exactly one ego belief")
        device = belief.device
        ego_id = torch.tensor([self.agent_id], device=device)
        proposals = self.proposal.topk(
            belief, ego_id, self.active_code_mask.to(device), k=self.config.num_candidates
        )
        codes = proposals["topk_codes"]
        residuals = self.codec.canonicalize(
            codes,
            proposals["topk_residuals"],
            self.residual_prior_by_code.to(device),
        )
        actions = decode_plan_batch(
            self.tokenizer, codes, residuals, self.action_mean.to(device), self.action_std.to(device)
        ).clamp(-self.config.action_clip, self.config.action_clip)
        valid_message = self.last_received
        if valid_message is not None and not (
            valid_message.sender_id == 1 - self.agent_id
            and valid_message.step <= self.step <= valid_message.valid_until_step
        ):
            valid_message = None
        if valid_message is None:
            (
                peer_codes,
                peer_residuals,
                weights,
                tail_weight,
                peer_variance,
                posterior_diagnostics,
            ) = self._marginal_peer_hypotheses(
                belief, ego_id, proposals, codes, residuals
            )
        else:
            if not 0 <= valid_message.code < self.active_code_mask.numel() or not bool(
                self.active_code_mask[valid_message.code]
            ):
                raise ValueError("received plan message contains an inactive/invalid code")
            peer_codes = torch.tensor([[valid_message.code]], device=device)
            peer_residuals = self.codec.decode(
                peer_codes,
                valid_message.residual_payload.to(device).reshape(1, 1, -1),
                self.residual_prior_by_code.to(device),
            )
            weights = torch.ones(1, 1, device=device)
            tail_weight = torch.zeros(1, device=device)
            peer_variance = torch.zeros(1, 1, device=device)
            posterior_diagnostics = {
                "intention_conditioning": "exact_received_message",
                "intention_candidate_count": 0,
                "proposal_candidate_coverage": 1.0,
                "posterior_top_m_coverage": 1.0,
                "posterior_tail_probability": 0.0,
                "posterior_entropy": 0.0,
                "posterior_mean_residual_variance": 0.0,
            }
        peer_actions = decode_plan_batch(
            self.tokenizer,
            peer_codes,
            peer_residuals,
            self.action_mean.to(device),
            self.action_std.to(device),
        ).clamp(-self.config.action_clip, self.config.action_clip)
        quantiles, constraints = self._world_grid(belief, actions, peer_actions)
        risk = self._risk(
            quantiles,
            constraints,
            actions,
            residual_variance=peer_variance,
        )
        # Omitted posterior codes remain explicit probability mass.  Their
        # outcomes were not rolled out, so use a conservative upper envelope
        # and do not award speculative reveal benefit for that tail.
        tail_risk = risk["G"].amax(dim=-1) + self.config.unmodeled_tail_penalty
        vpi = counterfactual_vpi(
            risk["G"],
            weights,
            tail_weight=tail_weight,
            tail_risk=tail_risk,
        )
        selected = int(vpi["no_comm_plan_index"].item())
        request = bool(
            valid_message is None
            and vpi["VPI"].item() > self.config.communication_cost
            and not self._cooldown()
        )
        message = PlanMessageV2(
            sender_id=self.agent_id,
            sequence=self.sequence,
            episode_sequence=self.episode_sequence,
            step=self.step,
            valid_until_step=self.step + self.config.plan_valid_steps,
            code=int(codes[0, selected].item()),
            residual_payload=self.codec.encode(residuals[0, selected]).cpu(),
        )
        self.sequence += 1
        prepared = PreparedPlanV2(
            agent_id=self.agent_id,
            step=self.step,
            request=request,
            provisional_index=selected,
            provisional_message=message,
            vpi=float(vpi["VPI"].item()),
        )
        self._pending = _PendingV2(
            belief,
            codes,
            residuals,
            actions,
            prepared,
            risk,
            vpi,
            posterior_diagnostics,
        )
        return prepared

    @torch.inference_mode()
    def finalize(
        self,
        *,
        reply: PlanMessageV2 | None = None,
        locked_as_responder: bool = False,
    ) -> LocalDecisionV2:
        if self._pending is None:
            raise RuntimeError("finalize requires prepare")
        pending = self._pending
        selected = pending.prepared.provisional_index
        if reply is not None:
            if locked_as_responder or not pending.prepared.request:
                raise RuntimeError("reply is only valid for the selected requester")
            if reply.sender_id != 1 - self.agent_id:
                raise ValueError("reply peer mismatch")
            if reply.episode_sequence != self.episode_sequence:
                raise ValueError("reply episode sequence mismatch")
            if not reply.step <= self.step <= reply.valid_until_step:
                raise ValueError("expired reply")
            if self.last_received is not None and reply.sequence <= self.last_received.sequence:
                raise ValueError("stale or replayed reply sequence")
            if not 0 <= reply.code < self.active_code_mask.numel() or not bool(
                self.active_code_mask[reply.code]
            ):
                raise ValueError("reply contains an inactive/invalid plan code")
            self.last_received = reply
            device = pending.belief.device
            peer_code = torch.tensor([[reply.code]], device=device)
            peer_payload = reply.residual_payload.to(device).reshape(1, 1, -1)
            peer_residual = self.codec.decode(
                peer_code, peer_payload, self.residual_prior_by_code.to(device)
            )
            peer_actions = decode_plan_batch(
                self.tokenizer,
                peer_code,
                peer_residual,
                self.action_mean.to(device),
                self.action_std.to(device),
            ).clamp(-self.config.action_clip, self.config.action_clip)
            quantiles, constraints = self._world_grid(pending.belief, pending.actions, peer_actions)
            revealed_risk = self._risk(quantiles, constraints, pending.actions)["G"]
            selected = int(revealed_risk[:, :, 0].argmin(dim=1).item())
        if pending.prepared.request:
            self.last_request_step = self.step
        code = int(pending.codes[0, selected].item())
        residual = pending.residuals[0, selected]
        decision = LocalDecisionV2(
            agent_id=self.agent_id,
            step=self.step,
            action=pending.actions[0, selected, 0].clone(),
            plan_code=code,
            plan_residual=residual.clone(),
            request_sent=pending.prepared.request,
            reply_received=reply is not None,
            locked_as_responder=bool(locked_as_responder),
            diagnostics={
                "VPI": pending.prepared.vpi,
                "provisional_index": pending.prepared.provisional_index,
                "selected_index": selected,
                "reply_bits": self.codec.reply_bits if reply is not None else 0,
                "communication_cost": self.config.communication_cost,
                "calibration": self.calibration.as_dict(),
                "world_ensemble_size": len(self.world_ensemble),
                "epistemic_available": self.epistemic_available,
                "quantile_crossing_rate_before_projection": float(
                    pending.risk["quantile_crossing_rate_before_projection"].item()
                ),
                **pending.posterior_diagnostics,
            },
        )
        self._pending = None
        self.step += 1
        return decision


class DecentralizedPairCoordinatorV2:
    """Simulation transport; joint action exists only after both local decisions."""

    def __init__(self, planner0: LocalPlannerV2, planner1: LocalPlannerV2):
        if planner0.agent_id != 0 or planner1.agent_id != 1:
            raise ValueError("pair coordinator requires planners 0 and 1")
        if planner0.artifact_hash != planner1.artifact_hash:
            raise ValueError("independent planners must load an identical artifact bundle")
        self.planners = (planner0, planner1)
        self.arbiter = DeterministicRequestArbiterV2()

    def reset(self, *, episode_sequence: int = 0) -> None:
        for planner in self.planners:
            planner.reset(episode_sequence=episode_sequence)

    def step(self, beliefs: Sequence[torch.Tensor]) -> PairDecisionV2:
        if len(beliefs) != 2:
            raise ValueError("pair coordinator receives exactly one local belief per robot")
        prepared = (self.planners[0].prepare(beliefs[0]), self.planners[1].prepare(beliefs[1]))
        if prepared[0].step != prepared[1].step:
            raise RuntimeError("public planner steps are not aligned")
        requester = self.arbiter.requester(
            {0: prepared[0].request, 1: prepared[1].request},
            prepared[0].provisional_message.episode_sequence,
            prepared[0].step,
        )
        replies: dict[int, PlanMessageV2] = {}
        responder: int | None = None
        if requester is not None:
            responder = 1 - requester
            replies[requester] = prepared[responder].provisional_message
        decisions: list[LocalDecisionV2 | None] = [None, None]
        if responder is not None:
            decisions[responder] = self.planners[responder].finalize(locked_as_responder=True)
        if requester is not None:
            decisions[requester] = self.planners[requester].finalize(reply=replies[requester])
        else:
            decisions[0] = self.planners[0].finalize()
            decisions[1] = self.planners[1].finalize()
        assert decisions[0] is not None and decisions[1] is not None
        # Commitment invariant compares the codec-canonical latent known to
        # both endpoints, not an untransmitted residual suffix.
        if responder is not None:
            message = prepared[responder].provisional_message
            canonical = self.planners[responder].codec.decode(
                torch.tensor(message.code),
                message.residual_payload,
                self.planners[responder].residual_prior_by_code,
            )
            if decisions[responder].plan_code != message.code or not torch.equal(
                decisions[responder].plan_residual.cpu(), canonical.cpu()
            ):
                raise RuntimeError("delivered plan differs from sender executed plan")
        pair = (decisions[0], decisions[1])
        return PairDecisionV2(
            joint_action=torch.cat((pair[0].action, pair[1].action), dim=-1),
            agents=pair,
            routed_messages=int(requester is not None),
            requester=requester,
        )
