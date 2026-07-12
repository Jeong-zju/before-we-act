"""Decentralized FE-PC-WAM policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Mapping, Sequence

import torch

from models.communication import VPICommunicationTrigger, reply_plan_diagnostics
from models.decentralized import EgoLocalWAM, LocalIntentionPosterior
from models.free_energy import FreeEnergyEvaluator
from models.plan_tokenizer import ActionOnlyPlanTokenizer, PlanCodeSupport


CommunicationMode = Literal[
    "selective",
    "no_comm",
    "always_reply",
    "periodic",
    "random",
]


@dataclass(frozen=True)
class DecentralizedPolicyConfig:
    """Execution-only settings for one copy of the decentralized policy.

    The default communication mode is the deployable VPI policy.  ``no_comm``
    and ``always_reply`` are evaluation ablations; neither changes the packet
    schema or grants access to peer state.
    """

    num_candidates: int = 8
    # Number of top code modes; residual sigma points expand each mode below.
    num_teammate_hypotheses: int = 4
    residual_sigma_points: int = 3
    residual_sigma_scale: float = 1.0
    candidate_residual_scale: float = 1.0
    ensure_candidate_code_diversity: bool = True
    action_clip: float | None = 1.0
    communication_mode: CommunicationMode = "selective"
    cooldown_steps: int = 8
    plan_valid_steps: int = 1
    periodic_interval: int = 8
    periodic_enabled: bool = True
    # Optional exact long-run request rate for budget-matched periodic
    # baselines.  ``None`` preserves the legacy ``periodic_interval`` schedule.
    # A cumulative-quota scheduler supports arbitrary rates without the
    # quantization error introduced by ``round(1 / rate)``.
    periodic_request_rate: float | None = None
    random_request_probability: float = 0.1
    seed: int = 0
    # The first fields in the model's message-metadata vector are reserved for
    # the local communication cache.  Indices outside the vector are ignored.
    metadata_available_index: int = 0
    metadata_age_index: int = 1
    metadata_confidence_index: int = 2
    metadata_delay_index: int = 3

    def __post_init__(self) -> None:
        if self.num_candidates <= 0 or self.num_teammate_hypotheses <= 0:
            raise ValueError("candidate and teammate-hypothesis counts must be positive")
        if self.residual_sigma_points not in (1, 3):
            raise ValueError("residual_sigma_points must be 1 (mean) or 3 (mean and +/- sigma)")
        if self.residual_sigma_scale < 0:
            raise ValueError("residual_sigma_scale cannot be negative")
        # A positive scale makes the deployable default sample the learned
        # per-code residual distribution rather than silently substituting 0.
        if self.candidate_residual_scale <= 0:
            raise ValueError("candidate_residual_scale must be positive")
        if self.action_clip is not None and self.action_clip <= 0:
            raise ValueError("action_clip must be positive when configured")
        if self.communication_mode not in (
            "selective",
            "no_comm",
            "always_reply",
            "periodic",
            "random",
        ):
            raise ValueError(f"unknown communication_mode={self.communication_mode!r}")
        if self.cooldown_steps < 0 or self.plan_valid_steps < 0:
            raise ValueError("cooldown_steps and plan_valid_steps cannot be negative")
        if self.periodic_interval <= 0:
            raise ValueError("periodic_interval must be positive")
        if self.periodic_request_rate is not None and not (
            0.0 <= self.periodic_request_rate <= 1.0
        ):
            raise ValueError("periodic_request_rate must be in [0, 1] when configured")
        if not 0.0 <= self.random_request_probability <= 1.0:
            raise ValueError("random_request_probability must be in [0, 1]")
        metadata_indices = (
            self.metadata_available_index,
            self.metadata_age_index,
            self.metadata_confidence_index,
            self.metadata_delay_index,
        )
        if any(index < -1 for index in metadata_indices):
            raise ValueError("metadata indices must be -1 (disabled) or non-negative")


@dataclass(frozen=True)
class LocalPlannerInput:
    """The complete online input of one robot's planner.

    There is deliberately no teammate observation, pose, state, action, or
    private slot in this type.  Teammate plan latents enter only through a
    routed :class:`PlanMessage` after a request.
    """

    ego_slots: torch.Tensor
    message_metadata: torch.Tensor


@dataclass(frozen=True)
class PlanMessage:
    """Low-rate plan-latent reply; it never carries state or an action chunk."""

    sender_id: int
    sequence: int
    start_step: int
    valid_until_step: int
    code: int
    residual: torch.Tensor
    confidence: float

    def __post_init__(self) -> None:
        if self.sender_id not in (0, 1):
            raise ValueError("sender_id must be 0 or 1")
        if self.sequence < 0 or self.start_step < 0:
            raise ValueError("sequence and start_step cannot be negative")
        if self.valid_until_step < self.start_step:
            raise ValueError("valid_until_step cannot precede start_step")
        if self.code < 0:
            raise ValueError("code cannot be negative")
        residual = torch.as_tensor(self.residual).detach().clone()
        if residual.ndim != 1 or not torch.is_floating_point(residual):
            raise ValueError("residual must be a floating-point vector")
        if not torch.isfinite(residual).all():
            raise ValueError("residual must be finite")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "confidence", float(self.confidence))

    def clone_for_transport(self) -> "PlanMessage":
        return PlanMessage(
            sender_id=self.sender_id,
            sequence=self.sequence,
            start_step=self.start_step,
            valid_until_step=self.valid_until_step,
            code=self.code,
            residual=self.residual,
            confidence=self.confidence,
        )


@dataclass(frozen=True)
class PlanProposal:
    """Public result of the pre-reply phase.

    ``request`` has already been decided from the local posterior and VPI.
    The router is allowed to read that boolean and, if true, transport the
    peer's ``message``.  All rollout internals remain inside the planner.
    """

    agent_id: int
    step: int
    request: bool
    raw_vpi_trigger: bool
    cooldown_remaining: int
    cached_message_used: bool
    selected_plan_index: int
    provisional_action: torch.Tensor
    message: PlanMessage
    vpi: float
    G_no: float
    G_reveal: float
    communication_cost: float


@dataclass(frozen=True)
class LocalDecision:
    agent_id: int
    step: int
    action: torch.Tensor
    plan_code: int
    plan_residual: torch.Tensor
    communicated: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairDecision:
    """Two independent ego actions joined only at the environment boundary."""

    joint_action: torch.Tensor
    agents: tuple[LocalDecision, LocalDecision]
    routed_messages: int


@dataclass
class _OwnPlan:
    code: int
    residual: torch.Tensor


@dataclass
class _PendingPreparation:
    local_input: LocalPlannerInput
    ego_slots: torch.Tensor
    effective_metadata: torch.Tensor
    candidate_codes: torch.Tensor
    candidate_residuals: torch.Tensor
    candidate_actions: torch.Tensor
    posterior: Dict[str, torch.Tensor]
    hypothesis_codes: torch.Tensor
    hypothesis_residuals: torch.Tensor
    hypothesis_weights: torch.Tensor
    score: Dict[str, torch.Tensor]
    trigger: Dict[str, torch.Tensor]
    proposal: PlanProposal


def _module_device_and_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype


def _scalar(value: torch.Tensor | float | int) -> float:
    return float(torch.as_tensor(value).detach().cpu().item())


class LocalAgentPlanner:
    """One robot's stateful, ego-local planner.

    Model objects may be shared by two planners, but the random generator,
    pending decision, cached message, cooldown, sequence counter, and committed
    own plan are strictly per-planner state.
    """

    def __init__(
        self,
        agent_id: int,
        *,
        tokenizer: ActionOnlyPlanTokenizer,
        wam: EgoLocalWAM,
        intention: LocalIntentionPosterior,
        support: PlanCodeSupport,
        free_energy: FreeEnergyEvaluator,
        communication: VPICommunicationTrigger,
        config: DecentralizedPolicyConfig | None = None,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
        seed: int | None = None,
    ):
        if agent_id not in (0, 1):
            raise ValueError("agent_id must be 0 or 1")
        self.agent_id = int(agent_id)
        self.tokenizer = tokenizer
        self.wam = wam
        self.intention = intention
        self.support = support
        self.free_energy = free_energy
        self.communication = communication
        self.config = config or DecentralizedPolicyConfig()
        self.device, self.dtype = _module_device_and_dtype(wam)

        self._validate_components()
        self._action_mean, self._action_std = self._validate_action_normalization(
            action_mean,
            action_std,
        )
        self._seed = int(self.config.seed + self.agent_id if seed is None else seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self._seed)
        self._communication_generator = torch.Generator(device="cpu")
        self._communication_generator.manual_seed(self._seed + 10_000)
        self._step = 0
        self._sequence = 0
        self._last_request_step: int | None = None
        self._last_received: PlanMessage | None = None
        self._last_receive_delay = 0.0
        self._own_plan: _OwnPlan | None = None
        self._pending: _PendingPreparation | None = None
        self._periodic_quota_accumulator = 0.0

        # This class is an inference controller.  Both local planners can call
        # these shared modules safely because no model-side mutable state is used.
        self.tokenizer.eval()
        self.wam.eval()
        self.intention.eval()

    @property
    def step_index(self) -> int:
        return self._step

    @property
    def cached_message(self) -> PlanMessage | None:
        return self._last_received

    def reset(self, *, seed: int | None = None) -> None:
        """Reset episode state and, optionally, reseed planner randomness.

        Evaluation uses the optional seed to give every communication mode the
        same candidate/random-baseline stream for a paired episode without
        reloading the shared model weights.  Existing callers that omit it keep
        the construction-time seed.
        """

        if seed is not None:
            self._seed = int(seed)
        self._generator.manual_seed(self._seed)
        self._communication_generator.manual_seed(self._seed + 10_000)
        self._step = 0
        self._sequence = 0
        self._last_request_step = None
        self._last_received = None
        self._last_receive_delay = 0.0
        self._own_plan = None
        self._pending = None
        self._periodic_quota_accumulator = 0.0

    def _validate_components(self) -> None:
        tokenizer_cfg = self.tokenizer.cfg
        wam_cfg = self.wam.cfg
        intention_cfg = self.intention.cfg
        codebook_size = int(tokenizer_cfg.codebook_size)
        residual_dim = int(tokenizer_cfg.latent_dim)
        if self.support.codebook_size != codebook_size:
            raise ValueError("support and tokenizer codebook sizes differ")
        if self.support.residual_dim != residual_dim:
            raise ValueError("support residual dimension differs from tokenizer latent dimension")
        if int(wam_cfg.plan_codebook_size) != codebook_size:
            raise ValueError("WAM and tokenizer codebook sizes differ")
        if int(intention_cfg.plan_codebook_size) != codebook_size:
            raise ValueError("intention and tokenizer codebook sizes differ")
        if int(wam_cfg.plan_latent_dim) != residual_dim:
            raise ValueError("WAM plan latent dimension differs from tokenizer")
        if int(intention_cfg.plan_latent_dim) != residual_dim:
            raise ValueError("intention plan latent dimension differs from tokenizer")
        if int(wam_cfg.horizon) != int(tokenizer_cfg.horizon):
            raise ValueError("WAM and tokenizer horizons differ")
        if int(wam_cfg.action_dim_per_agent) != int(tokenizer_cfg.action_dim):
            raise ValueError("WAM and tokenizer action dimensions differ")
        if int(self.communication.cfg.codebook_size) != codebook_size:
            raise ValueError("communication and tokenizer codebook sizes differ")
        if int(self.communication.cfg.residual_dim) != residual_dim:
            raise ValueError("communication and tokenizer residual dimensions differ")

        module_locations = [
            _module_device_and_dtype(module)
            for module in (self.tokenizer, self.wam, self.intention)
        ]
        devices = {device for device, _ in module_locations}
        if len(devices) != 1:
            raise ValueError("tokenizer, WAM, and intention model must be on the same device")
        dtypes = {dtype for _, dtype in module_locations}
        if len(dtypes) != 1:
            raise ValueError("tokenizer, WAM, and intention model must use the same parameter dtype")

    def _validate_action_normalization(
        self,
        action_mean: torch.Tensor | None,
        action_std: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (action_mean is None) != (action_std is None):
            raise ValueError("action_mean and action_std must be supplied together")
        if action_mean is None:
            return None, None
        action_dim = int(self.tokenizer.cfg.action_dim)
        mean = torch.as_tensor(action_mean, device=self.device, dtype=self.dtype).reshape(-1)
        std = torch.as_tensor(action_std, device=self.device, dtype=self.dtype).reshape(-1)
        if mean.shape != (action_dim,) or std.shape != (action_dim,):
            raise ValueError(f"action normalization must have shape [{action_dim}]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("action normalization must be finite with positive std")
        return mean, std

    def _coerce_input(self, local_input: LocalPlannerInput) -> tuple[torch.Tensor, torch.Tensor]:
        slots = torch.as_tensor(local_input.ego_slots, device=self.device, dtype=self.dtype)
        metadata = torch.as_tensor(local_input.message_metadata, device=self.device, dtype=self.dtype)
        expected_slots = (int(self.wam.cfg.slots_per_agent), int(self.wam.cfg.slot_dim))
        expected_metadata = (int(self.intention.cfg.message_metadata_dim),)
        if slots.shape != expected_slots:
            raise ValueError(f"ego_slots must have shape {expected_slots}, got {tuple(slots.shape)}")
        if metadata.shape != expected_metadata:
            raise ValueError(
                f"message_metadata must have shape {expected_metadata}, got {tuple(metadata.shape)}"
            )
        if not torch.isfinite(slots).all() or not torch.isfinite(metadata).all():
            raise ValueError("planner inputs must be finite")
        return slots.unsqueeze(0), metadata.unsqueeze(0)

    def _message_is_valid(self, message: PlanMessage | None) -> bool:
        return bool(
            message is not None
            and message.sender_id == 1 - self.agent_id
            and message.start_step <= self._step <= message.valid_until_step
        )

    def _effective_metadata(self, base: torch.Tensor) -> tuple[torch.Tensor, PlanMessage | None]:
        metadata = base.clone()
        message = self._last_received
        valid_message = message if self._message_is_valid(message) else None
        dimension = metadata.shape[-1]

        def assign(index: int, value: float) -> None:
            if 0 <= index < dimension:
                metadata[:, index] = value

        assign(self.config.metadata_available_index, float(valid_message is not None))
        if message is None:
            age, confidence, delay = 0.0, 0.0, 0.0
        else:
            age = float(max(0, self._step - message.start_step))
            confidence = message.confidence
            delay = self._last_receive_delay
        assign(self.config.metadata_age_index, age)
        assign(self.config.metadata_confidence_index, confidence)
        assign(self.config.metadata_delay_index, delay)
        return metadata, valid_message

    def _sample_candidates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled = self.support.sample(
            self.config.num_candidates,
            generator=self._generator,
            device=self.device,
            residual_scale=self.config.candidate_residual_scale,
            ensure_code_diversity=self.config.ensure_candidate_code_diversity,
        )
        codes = sampled["code_indices"].long()
        residuals = sampled["residual"].to(device=self.device, dtype=self.dtype)
        active = self.support.active_codes.to(device=self.device)
        if not torch.isin(codes, active).all():
            raise RuntimeError("PlanCodeSupport.sample returned a code outside active support")
        with torch.no_grad():
            decoded = self.tokenizer.decode_plan_latent(codes, residuals)
        actions = decoded["recon_actions"].to(device=self.device, dtype=self.dtype)
        expected = (
            self.config.num_candidates,
            int(self.tokenizer.cfg.horizon),
            int(self.tokenizer.cfg.action_dim),
        )
        if actions.shape != expected:
            raise ValueError(f"tokenizer decoded action shape {tuple(actions.shape)}, expected {expected}")
        if self._action_mean is not None:
            actions = actions * self._action_std.view(1, 1, -1) + self._action_mean.view(1, 1, -1)
        if self.config.action_clip is not None:
            actions = actions.clamp(-self.config.action_clip, self.config.action_clip)
        return codes, residuals, actions

    def _reference_plan(
        self,
        candidate_codes: torch.Tensor,
        candidate_residuals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._own_plan is not None:
            return (
                torch.tensor([self._own_plan.code], device=self.device, dtype=torch.long),
                self._own_plan.residual.to(device=self.device, dtype=self.dtype).reshape(1, -1),
            )
        support_probability = self.support.probabilities.to(self.device)[candidate_codes]
        reference_index = int(support_probability.argmax().item())
        return (
            candidate_codes[reference_index : reference_index + 1],
            candidate_residuals[reference_index : reference_index + 1],
        )

    def _infer_posterior(
        self,
        ego_slots: torch.Tensor,
        metadata: torch.Tensor,
        candidate_codes: torch.Tensor,
        candidate_residuals: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        reference_code, reference_residual = self._reference_plan(
            candidate_codes,
            candidate_residuals,
        )
        with torch.no_grad():
            posterior = self.intention(
                ego_slots=ego_slots,
                ego_plan_code=reference_code,
                ego_plan_residual=reference_residual,
                agent_id=torch.tensor([self.agent_id], device=self.device, dtype=torch.long),
                received_message_metadata=metadata,
            )

        probabilities = posterior["code_probabilities"]
        expected = (1, self.support.codebook_size)
        if probabilities.shape != expected:
            raise ValueError(
                f"intention code probabilities have shape {tuple(probabilities.shape)}, expected {expected}"
            )
        if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("intention code probabilities must be finite and non-negative")
        residual_mu = posterior["residual_mu_by_code"]
        residual_logvar = posterior["residual_logvar_by_code"]
        residual_shape = (1, self.support.codebook_size, self.support.residual_dim)
        if residual_mu.shape != residual_shape or residual_logvar.shape != residual_shape:
            raise ValueError(
                "intention residual statistics must have shape "
                f"{residual_shape}, got {tuple(residual_mu.shape)} and {tuple(residual_logvar.shape)}"
            )
        if not torch.isfinite(residual_mu).all() or not torch.isfinite(residual_logvar).all():
            raise ValueError("intention residual statistics must be finite")
        uncertainty = posterior["uncertainty"]
        if uncertainty.numel() != 1 or not torch.isfinite(uncertainty).all():
            raise ValueError("intention uncertainty must contain one finite local scalar")
        active = self.support.active_codes.to(device=self.device)
        mask = torch.zeros_like(probabilities, dtype=torch.bool)
        mask[:, active] = True
        masked = torch.where(mask, probabilities, torch.zeros_like(probabilities))
        mass = masked.sum(dim=-1, keepdim=True)
        if (mass <= 0).any():
            raise RuntimeError("intention posterior has no probability mass on artifact support")
        posterior = dict(posterior)
        posterior["unmasked_code_probabilities"] = probabilities
        posterior["code_probabilities"] = masked / mass
        posterior["active_code_mask"] = mask
        return posterior

    def _posterior_hypotheses(
        self,
        posterior: Mapping[str, torch.Tensor],
        valid_message: PlanMessage | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if valid_message is not None:
            return (
                torch.tensor([valid_message.code], device=self.device, dtype=torch.long),
                valid_message.residual.to(device=self.device, dtype=self.dtype).reshape(1, -1),
                torch.ones(1, device=self.device, dtype=self.dtype),
            )

        probabilities = posterior["code_probabilities"][0]
        active_count = int(self.support.active_codes.numel())
        top_count = min(self.config.num_teammate_hypotheses, active_count)
        code_weights, codes = probabilities.topk(top_count)
        code_weights = code_weights / code_weights.sum().clamp_min(1e-8)
        residual_mu_all = posterior["residual_mu_by_code"][0]
        residual_logvar_all = posterior["residual_logvar_by_code"][0]
        mu = residual_mu_all[codes]

        if self.config.residual_sigma_points == 1:
            return codes, mu, code_weights

        # Three deterministic diagonal sigma points per code preserve the
        # posterior residual scale in rollout/VPI without an O(D) expansion.
        sigma = torch.exp(0.5 * residual_logvar_all[codes]) * self.config.residual_sigma_scale
        codes = codes[:, None].expand(-1, 3).reshape(-1)
        residuals = torch.stack((mu, mu + sigma, mu - sigma), dim=1).reshape(
            -1,
            mu.shape[-1],
        )
        point_weights = code_weights[:, None] * code_weights.new_tensor([0.5, 0.25, 0.25])
        weights = point_weights.reshape(-1)
        weights = weights / weights.sum().clamp_min(1e-8)
        return codes, residuals, weights

    def _rollout_grid(
        self,
        ego_slots: torch.Tensor,
        candidate_codes: torch.Tensor,
        candidate_residuals: torch.Tensor,
        hypothesis_codes: torch.Tensor,
        hypothesis_residuals: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        K = int(candidate_codes.numel())
        M = int(hypothesis_codes.numel())
        ego_codes = candidate_codes[:, None].expand(K, M).reshape(-1)
        peer_codes = hypothesis_codes[None, :].expand(K, M).reshape(-1)
        ego_residuals = candidate_residuals[:, None, :].expand(K, M, -1).reshape(K * M, -1)
        peer_residuals = hypothesis_residuals[None, :, :].expand(K, M, -1).reshape(K * M, -1)
        codes = torch.stack((ego_codes, peer_codes), dim=1)
        residuals = torch.stack((ego_residuals, peer_residuals), dim=1)

        # q(z_j) belongs to the outer expectation.  The current WAM API still
        # requires a weight argument, so pass a neutral value and never rely on
        # posterior probability to change p(y | b, z_i, z_j).
        neutral_weights = torch.ones(K * M, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            rollout = self.wam.rollout(
                ego_slots=ego_slots.expand(K * M, -1, -1),
                plan_codes=codes,
                plan_residuals=residuals,
                teammate_hypothesis_weight=neutral_weights,
            )
        grid: Dict[str, torch.Tensor] = {}
        for key, value in rollout.items():
            if torch.is_tensor(value) and value.shape[0] == K * M:
                grid[key] = value.reshape(1, K, M, *value.shape[1:])
        return grid

    def _score_grid(
        self,
        rollout: Mapping[str, torch.Tensor],
        hypothesis_weights: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        weights = hypothesis_weights.reshape(1, -1)
        uncertainty_grid = uncertainty.reshape(1, 1, 1)
        return self.free_energy.total_score_hypotheses(
            dict(rollout),
            weights,
            uncertainty=uncertainty_grid,
        )

    def _cooldown_remaining(self) -> int:
        if self._last_request_step is None:
            return 0
        return max(0, self.config.cooldown_steps - (self._step - self._last_request_step))

    def _periodic_request_due(self) -> bool:
        """Return the next deterministic periodic-baseline quota decision.

        When an explicit rate is configured, this is the standard cumulative
        error (Bresenham/DDA) schedule: over any ``N`` calls it emits
        ``floor(N * rate)`` requests, up to floating-point tolerance.  Thus the
        finite-horizon rate error is bounded by one request instead of being
        quantized to a reciprocal integer interval.
        """

        rate = self.config.periodic_request_rate
        if rate is None:
            return self._step % self.config.periodic_interval == 0
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True

        self._periodic_quota_accumulator += rate
        if self._periodic_quota_accumulator + 1e-12 < 1.0:
            return False
        self._periodic_quota_accumulator = max(
            0.0,
            self._periodic_quota_accumulator - 1.0,
        )
        return True

    def _request_decision(
        self,
        trigger: Mapping[str, torch.Tensor],
        valid_message: PlanMessage | None,
    ) -> tuple[bool, bool, int]:
        raw_trigger = bool(trigger["trigger"].reshape(-1)[0].item())
        cooldown_remaining = self._cooldown_remaining()
        eligible = cooldown_remaining == 0 and valid_message is None
        if self.config.communication_mode == "no_comm":
            request = False
        elif self.config.communication_mode == "always_reply":
            request = eligible
        elif self.config.communication_mode == "periodic":
            periodic_due = self._periodic_request_due()
            request = (
                self.config.periodic_enabled
                and eligible
                and periodic_due
            )
        elif self.config.communication_mode == "random":
            draw = torch.rand((), generator=self._communication_generator).item()
            request = eligible and draw < self.config.random_request_probability
        else:
            request = raw_trigger and eligible
        return request, raw_trigger, cooldown_remaining

    def _message_confidence(self, code: int) -> float:
        active = self.support.active_codes
        probability = self.support.probabilities[active]
        if probability.sum() <= 0:
            probability = self.support.counts[active].float()
        probability = probability / probability.sum().clamp_min(1e-8)
        position = torch.nonzero(active == code, as_tuple=False).flatten()
        if position.numel() != 1:
            raise RuntimeError("selected plan code is outside active support")
        return float(probability[position.item()].item())

    def prepare(self, local_input: LocalPlannerInput) -> PlanProposal:
        """Perform local inference and decide whether to request, before reply."""

        if self._pending is not None:
            raise RuntimeError("prepare called twice without finalizing the previous decision")
        ego_slots, base_metadata = self._coerce_input(local_input)
        effective_metadata, valid_message = self._effective_metadata(base_metadata)
        candidate_codes, candidate_residuals, candidate_actions = self._sample_candidates()
        posterior = self._infer_posterior(
            ego_slots,
            effective_metadata,
            candidate_codes,
            candidate_residuals,
        )
        hypothesis_codes, hypothesis_residuals, hypothesis_weights = self._posterior_hypotheses(
            posterior,
            valid_message,
        )
        rollout = self._rollout_grid(
            ego_slots,
            candidate_codes,
            candidate_residuals,
            hypothesis_codes,
            hypothesis_residuals,
        )
        score = self._score_grid(
            rollout,
            hypothesis_weights,
            posterior["uncertainty"],
        )
        trigger = self.communication.decide_request(
            score["G"],
            hypothesis_weights.reshape(1, -1),
        )
        request, raw_trigger, cooldown_remaining = self._request_decision(trigger, valid_message)
        selected_index = int(score["no_comm_plan_index"][0].item())
        selected_code = int(candidate_codes[selected_index].item())
        message = PlanMessage(
            sender_id=self.agent_id,
            sequence=self._sequence,
            start_step=self._step,
            valid_until_step=self._step + self.config.plan_valid_steps,
            code=selected_code,
            residual=candidate_residuals[selected_index].detach().cpu(),
            confidence=self._message_confidence(selected_code),
        )
        self._sequence += 1
        proposal = PlanProposal(
            agent_id=self.agent_id,
            step=self._step,
            request=request,
            raw_vpi_trigger=raw_trigger,
            cooldown_remaining=cooldown_remaining,
            cached_message_used=valid_message is not None,
            selected_plan_index=selected_index,
            provisional_action=candidate_actions[selected_index, 0].detach().clone(),
            message=message,
            vpi=_scalar(trigger["VPI"][0]),
            G_no=_scalar(trigger["G_no"][0]),
            G_reveal=_scalar(trigger["G_reveal"][0]),
            communication_cost=_scalar(trigger["C_comm"][0]),
        )
        self._pending = _PendingPreparation(
            local_input=local_input,
            ego_slots=ego_slots,
            effective_metadata=effective_metadata,
            candidate_codes=candidate_codes,
            candidate_residuals=candidate_residuals,
            candidate_actions=candidate_actions,
            posterior=posterior,
            hypothesis_codes=hypothesis_codes,
            hypothesis_residuals=hypothesis_residuals,
            hypothesis_weights=hypothesis_weights,
            score=score,
            trigger=trigger,
            proposal=proposal,
        )
        return proposal

    def _validate_reply(self, message: PlanMessage) -> None:
        if message.sender_id != 1 - self.agent_id:
            raise ValueError("reply sender is not this planner's peer")
        if not message.start_step <= self._step <= message.valid_until_step:
            raise ValueError("reply is not valid at the current planner step")
        if message.code >= self.support.codebook_size:
            raise ValueError("reply code lies outside the codebook")
        if message.residual.shape != (self.support.residual_dim,):
            raise ValueError("reply residual dimension differs from plan support")
        active = self.support.active_codes
        if not bool((active == message.code).any().item()):
            raise ValueError("reply code lies outside artifact support")

    def _base_diagnostics(
        self,
        pending: _PendingPreparation,
        *,
        communicated: bool,
        actual_request_bits: int,
        actual_reply_bits: int,
        actual_delay: float,
    ) -> Dict[str, Any]:
        proposal = pending.proposal
        return {
            "request_sent": bool(proposal.request),
            "reply_received": bool(communicated),
            "raw_vpi_trigger": bool(proposal.raw_vpi_trigger),
            "cooldown_remaining": int(proposal.cooldown_remaining),
            "cached_message_used": bool(proposal.cached_message_used),
            "VPI": float(proposal.vpi),
            "G_no": float(proposal.G_no),
            "G_reveal": float(proposal.G_reveal),
            "communication_cost": float(proposal.communication_cost),
            "intention_uncertainty": _scalar(pending.posterior["uncertainty"][0]),
            "actual_request_bits": int(actual_request_bits),
            "actual_reply_bits": int(actual_reply_bits),
            "actual_round_trip_bits": int(actual_request_bits + actual_reply_bits),
            "actual_delay_steps": float(actual_delay),
            "posterior_active_codes": self.support.active_codes.tolist(),
            "candidate_codes": pending.candidate_codes.detach().cpu().tolist(),
            "hypothesis_codes": pending.hypothesis_codes.detach().cpu().tolist(),
        }

    def finalize(self, reply: PlanMessage | None = None) -> LocalDecision:
        """Optionally condition on a routed reply, then choose before execution."""

        if self._pending is None:
            raise RuntimeError("finalize requires a preceding prepare call")
        pending = self._pending
        proposal = pending.proposal
        prior_index = proposal.selected_plan_index
        revised_index = prior_index
        communicated = reply is not None

        if communicated and not proposal.request:
            raise RuntimeError("unsolicited plan message: no local request was issued")

        if reply is None:
            request_bits = self.communication.request_bits() if proposal.request else 0
            reply_bits = 0
            delay = self.communication.cfg.delay_steps if proposal.request else 0.0
            diagnostics = self._base_diagnostics(
                pending,
                communicated=False,
                actual_request_bits=request_bits,
                actual_reply_bits=reply_bits,
                actual_delay=delay,
            )
            diagnostics.update(
                {
                    "G_before": float(proposal.G_no),
                    "G_after": float(proposal.G_no),
                    "plan_surprise": 0.0,
                    "code_surprise": 0.0,
                    "residual_surprise": 0.0,
                    "replanned": False,
                    "action_change_l2": 0.0,
                }
            )
        else:
            self._validate_reply(reply)
            transported = reply.clone_for_transport()
            self._last_received = transported
            self._last_receive_delay = float(self.communication.cfg.delay_steps)
            actual_codes = torch.tensor([reply.code], device=self.device, dtype=torch.long)
            actual_residuals = reply.residual.to(device=self.device, dtype=self.dtype).reshape(1, -1)
            actual_rollout = self._rollout_grid(
                pending.ego_slots,
                pending.candidate_codes,
                pending.candidate_residuals,
                actual_codes,
                actual_residuals,
            )
            # The actual message removes teammate-plan posterior uncertainty for
            # this rollout.  Physical/process uncertainty remains in the WAM.
            actual_score = self._score_grid(
                actual_rollout,
                torch.ones(1, device=self.device, dtype=self.dtype),
                torch.zeros(1, device=self.device, dtype=self.dtype),
            )
            revised_index = int(actual_score["no_comm_plan_index"][0].item())
            G_before = actual_score["G"][0, prior_index, 0]
            G_after = actual_score["G"][0, revised_index, 0]
            reply_diagnostics = reply_plan_diagnostics(
                prior_code_probabilities=pending.posterior["code_probabilities"],
                reply_code=actual_codes,
                prior_plan_index=torch.tensor([prior_index], device=self.device),
                revised_plan_index=torch.tensor([revised_index], device=self.device),
                prior_actions=pending.candidate_actions[prior_index].unsqueeze(0),
                revised_actions=pending.candidate_actions[revised_index].unsqueeze(0),
                prior_residual_mu_by_code=pending.posterior["residual_mu_by_code"],
                prior_residual_logvar_by_code=pending.posterior["residual_logvar_by_code"],
                reply_residual=actual_residuals,
            )
            diagnostics = self._base_diagnostics(
                pending,
                communicated=True,
                actual_request_bits=self.communication.request_bits(),
                actual_reply_bits=self.communication.reply_bits(),
                actual_delay=float(self.communication.cfg.delay_steps),
            )
            diagnostics.update(
                {
                    "G_before": _scalar(G_before),
                    "G_after": _scalar(G_after),
                    "received_code": int(reply.code),
                    "plan_surprise": _scalar(reply_diagnostics["plan_surprise"][0]),
                    "code_surprise": _scalar(reply_diagnostics["code_surprise"][0]),
                    "residual_surprise": _scalar(reply_diagnostics["residual_surprise"][0]),
                    "replanned": bool(reply_diagnostics["replanned"][0].item()),
                    "action_change_l2": _scalar(reply_diagnostics["action_change_l2"][0]),
                    "action_change_mean_abs": _scalar(
                        reply_diagnostics["action_change_mean_abs"][0]
                    ),
                }
            )

        if proposal.request:
            self._last_request_step = self._step
        selected_code = int(pending.candidate_codes[revised_index].item())
        selected_residual = pending.candidate_residuals[revised_index].detach().clone()
        self._own_plan = _OwnPlan(selected_code, selected_residual.detach().cpu())
        action = pending.candidate_actions[revised_index, 0].detach().clone()
        diagnostics["prior_plan_index"] = int(prior_index)
        diagnostics["revised_plan_index"] = int(revised_index)
        diagnostics["selected_plan_code"] = selected_code
        decision = LocalDecision(
            agent_id=self.agent_id,
            step=self._step,
            action=action,
            plan_code=selected_code,
            plan_residual=selected_residual,
            communicated=communicated,
            diagnostics=diagnostics,
        )
        self._pending = None
        self._step += 1
        return decision


class SelectivePlanRouter:
    """A content-blind request/reply router for exactly two robots."""

    def route(
        self,
        requests: Mapping[int, bool],
        reply_suppliers: Mapping[int, Callable[[], PlanMessage]],
    ) -> Dict[int, PlanMessage]:
        if set(requests) != {0, 1} or set(reply_suppliers) != {0, 1}:
            raise ValueError("router requires request and supplier entries for agents 0 and 1")
        deliveries: Dict[int, PlanMessage] = {}
        for requester in (0, 1):
            if bool(requests[requester]):
                # The supplier is intentionally not called for a non-requesting
                # robot, so peer plan content cannot affect the trigger.
                peer = 1 - requester
                deliveries[requester] = reply_suppliers[peer]().clone_for_transport()
        return deliveries


class DecentralizedPairCoordinator:
    """Runs two local planners and concatenates actions at the physics edge."""

    def __init__(
        self,
        planner0: LocalAgentPlanner,
        planner1: LocalAgentPlanner,
        router: SelectivePlanRouter | None = None,
    ):
        if planner0.agent_id != 0 or planner1.agent_id != 1:
            raise ValueError("pair coordinator requires planners with IDs 0 and 1")
        # In a single-process deployment test both robots use literally the
        # same version/weights.  Their recurrent controller state remains local.
        shared_names = ("tokenizer", "wam", "intention", "support")
        for name in shared_names:
            if getattr(planner0, name) is not getattr(planner1, name):
                raise ValueError(f"both planners must share the exact same {name} object")
        self.planners = (planner0, planner1)
        self.router = router or SelectivePlanRouter()

    @classmethod
    def from_shared_components(
        cls,
        *,
        tokenizer: ActionOnlyPlanTokenizer,
        wam: EgoLocalWAM,
        intention: LocalIntentionPosterior,
        support: PlanCodeSupport,
        free_energy: FreeEnergyEvaluator,
        communication: VPICommunicationTrigger,
        config: DecentralizedPolicyConfig | None = None,
        action_mean: torch.Tensor | None = None,
        action_std: torch.Tensor | None = None,
        router: SelectivePlanRouter | None = None,
    ) -> "DecentralizedPairCoordinator":
        config = config or DecentralizedPolicyConfig()
        planner0 = LocalAgentPlanner(
            0,
            tokenizer=tokenizer,
            wam=wam,
            intention=intention,
            support=support,
            free_energy=free_energy,
            communication=communication,
            config=config,
            action_mean=action_mean,
            action_std=action_std,
            seed=config.seed,
        )
        planner1 = LocalAgentPlanner(
            1,
            tokenizer=tokenizer,
            wam=wam,
            intention=intention,
            support=support,
            free_energy=free_energy,
            communication=communication,
            config=config,
            action_mean=action_mean,
            action_std=action_std,
            seed=config.seed + 1,
        )
        return cls(planner0, planner1, router=router)

    def reset(self, *, seed: int | None = None) -> None:
        for agent_id, planner in enumerate(self.planners):
            planner.reset(seed=None if seed is None else int(seed) + agent_id)

    def step(self, local_inputs: Sequence[LocalPlannerInput]) -> PairDecision:
        if len(local_inputs) != 2:
            raise ValueError("pair step requires exactly two local planner inputs")
        # Each prepare call receives one ego packet only.  No joint tensor is
        # formed or consumed by either model.
        proposals = (
            self.planners[0].prepare(local_inputs[0]),
            self.planners[1].prepare(local_inputs[1]),
        )
        deliveries = self.router.route(
            requests={0: proposals[0].request, 1: proposals[1].request},
            reply_suppliers={
                0: lambda: proposals[0].message,
                1: lambda: proposals[1].message,
            },
        )
        decisions = (
            self.planners[0].finalize(deliveries.get(0)),
            self.planners[1].finalize(deliveries.get(1)),
        )
        joint_action = torch.cat((decisions[0].action, decisions[1].action), dim=-1)
        return PairDecision(
            joint_action=joint_action,
            agents=decisions,
            routed_messages=len(deliveries),
        )
