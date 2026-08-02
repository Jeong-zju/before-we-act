"""S4-R7 active-parent Flow with token-preserving utility-calibrated evidence."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import inspect
from typing import Callable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.wam_multimodal.agent_factorized_flow_wam import AgentFactorizedFlowWAM
from models.wam_multimodal.cross_agent_world_conditioned_flow import (
    CrossAgentWorldConditionedFlow,
    PredictedFutureLatents,
    WorldToFlowResidualAdapter,
)
from models.wam_multimodal.local_future_predictor import (
    LocalActionConditionedFuturePredictor,
)
from models.wam_multimodal.protected_team_future_predictor import (
    ProtectedTeamFuturePredictor,
)
from models.wam_multimodal.world_evidence_router import (
    EvidenceTokens,
    FutureEvidenceRouter,
    LowRankEvidenceAdapterBank,
    S4WorldEvidenceProvider,
    UtilityCalibratedResidual,
    WorldEvidenceRouterConfig,
)


@dataclass(frozen=True)
class ActiveParentVelocityCache:
    """Minimum cache an S4-compatible active R6 parent must expose."""

    active_parent_velocity: Tensor
    flow_features: Tensor
    clean_actions: Tensor
    predicted_futures: object | None = None


@dataclass(frozen=True)
class ForcedEvidenceAudit:
    """Detached forced-group errors and the resulting utility target."""

    velocity_errors: Tensor
    utility_target: Tensor
    temperature: Tensor
    group_mask: Tensor
    valid_query_mask: Tensor


class S4ActiveTeamFutureProvider(nn.Module):
    """Trainable R6L-own plus R5-P0 peer/shared future clone.

    The protected R5 class is intentionally not executed: it hard-codes an
    ``eval()+no_grad()`` own tower.  Instead, this module clones only its P0 team
    mixer/heads and always derives state, visual, action, future-position, and
    grid-position features from the one active local predictor supplied here.
    One forward returns the full own/peer/shared prediction used by both the
    legacy R6 adapter and the new evidence provider.
    """

    def __init__(
        self,
        local_predictor: LocalActionConditionedFuturePredictor,
        r5_p0_source: ProtectedTeamFuturePredictor,
    ) -> None:
        super().__init__()
        if r5_p0_source.team_config.team_mixer != "shared":
            raise ValueError("S4 active team provider must initialize from R5-P0")
        if r5_p0_source.peer_mixer is not None:
            raise ValueError("R5-P0 must not contain a private peer mixer")
        if not r5_p0_source._protected_loaded:
            raise RuntimeError("R5-P0 protected-own checkpoint must be loaded first")
        if local_predictor.config != r5_p0_source.local_config:
            raise ValueError("R6L local and R5-P0 team configs must match exactly")
        if local_predictor is r5_p0_source.protected_own:
            raise ValueError("active local predictor must be a separate clone")
        self.local_predictor = local_predictor
        self.local_config = local_predictor.config
        self.shared_projection = copy.deepcopy(r5_p0_source.shared_projection)
        self.team_agent_norm = copy.deepcopy(r5_p0_source.team_agent_norm)
        self.slot_embedding = nn.Parameter(
            r5_p0_source.slot_embedding.detach().clone()
        )
        self.shared_mixer = copy.deepcopy(r5_p0_source.shared_mixer)
        self.peer_state_head = copy.deepcopy(r5_p0_source.peer_state_head)
        self.peer_visual_head = copy.deepcopy(r5_p0_source.peer_visual_head)
        self.shared_visual_head = copy.deepcopy(r5_p0_source.shared_visual_head)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.train(True)

    def forward(
        self,
        current_state: Tensor,
        current_visual_latent: Tensor,
        shared_visual_latent: Tensor,
        candidate_actions: Tensor,
        valid_agent_mask: Tensor,
        *,
        actions_by_focal: Tensor | None = None,
    ) -> PredictedFutureLatents:
        config = self.local_config
        batch_size = current_state.shape[0]
        if shared_visual_latent.shape != (
            batch_size,
            config.visual_grid_tokens,
            config.visual_latent_dim,
        ):
            raise ValueError("shared_visual_latent has an invalid shape")
        if valid_agent_mask.shape != (batch_size, config.max_agents):
            raise ValueError("valid_agent_mask must be [B,A]")
        if valid_agent_mask.dtype != torch.bool:
            raise TypeError("valid_agent_mask must have dtype bool")
        if actions_by_focal is None:
            actions_by_focal = candidate_actions[:, None].expand(
                -1, config.max_agents, -1, -1, -1
            )
        if actions_by_focal.shape != (
            batch_size,
            config.max_agents,
            config.max_agents,
            config.action_horizon,
            config.action_dim,
        ):
            raise ValueError(
                "actions_by_focal must be [B,focal,target,H,action_dim]"
            )

        # This is the sole own prediction.  Team roles below reuse projections
        # from this exact trainable local clone and never call protected_own.
        own_context = self.local_predictor.encode_context(
            current_state,
            current_visual_latent,
            candidate_actions,
            valid_agent_mask,
            valid_agent_mask,
        )
        own_state, own_visual = self.local_predictor.decode_future(
            own_context, valid_agent_mask
        )
        state_token = self.local_predictor.state_projection(current_state)
        visual_token = self.local_predictor.visual_projection(
            current_visual_latent
        ).mean(dim=2)
        action_token = self.local_predictor.action_projection(actions_by_focal)
        action_token = (
            action_token + self.local_predictor.action_position[:, None]
        ).mean(dim=3)
        agent_tokens = self.team_agent_norm(
            state_token[:, None] + visual_token[:, None] + action_token
        )
        shared_token = self.shared_projection(shared_visual_latent).mean(dim=1)
        shared_tokens = shared_token[:, None, None].expand(
            -1, config.max_agents, -1, -1
        )
        tokens = torch.cat((shared_tokens, agent_tokens), dim=2)
        tokens = tokens + self.slot_embedding
        valid_sequence = torch.cat(
            (
                torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=valid_agent_mask.device,
                ),
                valid_agent_mask,
            ),
            dim=1,
        )
        padding = (~valid_sequence[:, None].expand(
            -1, config.max_agents, -1
        )).reshape(batch_size * config.max_agents, config.max_agents + 1)
        encoded = self.shared_mixer(
            tokens.reshape(
                batch_size * config.max_agents,
                config.max_agents + 1,
                config.d_model,
            ),
            padding,
        ).reshape(
            batch_size,
            config.max_agents,
            config.max_agents + 1,
            config.d_model,
        )
        encoded_shared = encoded[:, :, 0]
        encoded_agents = encoded[:, :, 1:]
        focal_index = torch.arange(config.max_agents, device=current_state.device)
        focal_context = encoded_agents[:, focal_index, focal_index]
        pair_context = encoded_agents + focal_context[:, :, None]
        future_position = self.local_predictor.future_position
        grid_position = self.local_predictor.grid_position
        pair_future = pair_context[:, :, :, None] + future_position[:, None]
        peer_state = self.peer_state_head(pair_future)
        peer_visual = self.peer_visual_head(
            pair_future[:, :, :, :, None] + grid_position[:, None]
        )
        shared_future = encoded_shared[:, :, None] + future_position
        shared_visual = self.shared_visual_head(
            shared_future[:, :, :, None] + grid_position
        )
        pair_valid = valid_agent_mask[:, :, None] & valid_agent_mask[:, None, :]
        focal_valid = valid_agent_mask[:, :, None, None, None]
        return PredictedFutureLatents(
            own_state=own_state,
            own_visual=own_visual,
            peer_state=peer_state * pair_valid[:, :, :, None, None],
            peer_visual=peer_visual * pair_valid[:, :, :, None, None, None],
            shared_visual=shared_visual * focal_valid,
        )


class ScaleAlignedActiveWorldFlow(nn.Module):
    """Trainable clone of the R6L Flow/local-world/legacy-adapter path."""

    def __init__(
        self,
        base_flow: AgentFactorizedFlowWAM,
        future_predictor: S4ActiveTeamFutureProvider,
        legacy_adapter: WorldToFlowResidualAdapter,
    ) -> None:
        super().__init__()
        if base_flow.config.d_model != legacy_adapter.config.flow_dim:
            raise ValueError("base Flow and legacy adapter widths differ")
        if base_flow.config.action_dim != legacy_adapter.config.action_dim:
            raise ValueError("base Flow and legacy adapter action dimensions differ")
        self.base_flow = base_flow
        self.future_predictor = future_predictor
        self.legacy_adapter = legacy_adapter
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.train(True)

    @classmethod
    def from_legacy_reference(
        cls,
        legacy_reference: CrossAgentWorldConditionedFlow,
        r5_p0_source: ProtectedTeamFuturePredictor,
    ) -> "ScaleAlignedActiveWorldFlow":
        """Deep-clone accepted ancestors without modifying either reference."""

        if legacy_reference.future_scope != "local" or not legacy_reference.injection:
            raise ValueError("S4 must initialize from the accepted gated R6L path")
        if not isinstance(
            legacy_reference.future_predictor,
            LocalActionConditionedFuturePredictor,
        ):
            raise TypeError("accepted R6L future predictor must be local")
        active_local = copy.deepcopy(legacy_reference.future_predictor)
        for parameter in active_local.parameters():
            parameter.requires_grad_(True)
        active_team = S4ActiveTeamFutureProvider(active_local, r5_p0_source)
        return cls(
            copy.deepcopy(legacy_reference.base_flow),
            active_team,
            copy.deepcopy(legacy_reference.adapter),
        )

    def velocity(
        self,
        base_vision_tokens: Tensor,
        current_state: Tensor,
        current_visual_latent: Tensor,
        shared_visual_latent: Tensor,
        action_inputs: Tensor,
        flow_time: Tensor,
        valid_agent_mask: Tensor,
        *,
        force_gate_zero: bool = False,
        return_cache: bool = False,
        future_intervention: Callable[
            [PredictedFutureLatents], PredictedFutureLatents
        ] | None = None,
    ) -> tuple[Tensor, dict[str, object], ActiveParentVelocityCache] | tuple[
        Tensor, dict[str, object]
    ]:
        if base_vision_tokens.ndim != 4 or current_state.ndim != 3:
            raise ValueError("base vision/state must retain [B,A,...]")
        batch_size, agents = current_state.shape[:2]
        expected_actions = (
            batch_size,
            agents,
            self.base_flow.config.horizon,
            self.base_flow.config.action_dim,
        )
        if action_inputs.shape != expected_actions:
            raise ValueError(f"action_inputs must have shape {expected_actions}")
        if flow_time.shape != (batch_size,):
            raise ValueError("flow_time must be [B]")
        if valid_agent_mask.shape != (batch_size, agents):
            raise ValueError("valid_agent_mask must be [B,A]")
        base_velocity, router_aux, flow_features = self.base_flow.forward_features(
            base_vision_tokens.flatten(0, 1),
            current_state.flatten(0, 1),
            action_inputs.flatten(0, 1),
            flow_time[:, None].expand(-1, agents).reshape(-1),
        )
        base_velocity = base_velocity.reshape(expected_actions)
        flow_features = flow_features.reshape(
            batch_size,
            agents,
            self.base_flow.config.horizon,
            -1,
        )
        valid = valid_agent_mask[:, :, None, None].to(base_velocity)
        base_velocity = base_velocity * valid
        if force_gate_zero and not return_cache:
            return base_velocity, {
                "gate": base_velocity.new_zeros(()),
                "router_aux": router_aux.detach(),
                "residual_rms": base_velocity.new_zeros(()),
            }

        clean_actions = action_inputs + (
            1.0 - flow_time[:, None, None, None]
        ) * base_velocity
        predicted_futures = self.future_predictor(
            current_state,
            current_visual_latent,
            shared_visual_latent,
            clean_actions.detach(),
            valid_agent_mask,
        )
        if future_intervention is not None:
            predicted_futures = future_intervention(predicted_futures)
            if not isinstance(predicted_futures, PredictedFutureLatents):
                raise TypeError(
                    "future_intervention must return PredictedFutureLatents"
                )
        if force_gate_zero:
            active_velocity = base_velocity
            legacy_residual = torch.zeros_like(base_velocity)
            legacy_gate = base_velocity.new_zeros(())
        else:
            legacy_residual, legacy_gate = self.legacy_adapter(
                flow_features,
                predicted_futures,
                valid_agent_mask,
                future_scope="local",
            )
            active_velocity = base_velocity + legacy_residual
        auxiliary: dict[str, object] = {
            "gate": legacy_gate.detach(),
            "router_aux": router_aux.detach(),
            "residual_rms": legacy_residual.float().square().mean().sqrt().detach(),
        }
        if not return_cache:
            return active_velocity, auxiliary
        return active_velocity, auxiliary, ActiveParentVelocityCache(
            active_parent_velocity=active_velocity,
            flow_features=flow_features,
            clean_actions=clean_actions,
            predicted_futures=predicted_futures,
        )


class UtilityCalibratedWorldFlow(nn.Module):
    """Compose a scale-aligned active parent with the new R7 residual.

    ``active_parent`` is a mutable clone, not the immutable legacy reference.
    Its ``velocity`` method should expose ``flow_features`` and ``clean_actions``
    through ``return_cache=True``.  No checkpoint-specific code is required;
    small synthetic parents can implement the same public protocol in tests.
    """

    def __init__(
        self,
        active_parent: nn.Module,
        evidence_provider: S4WorldEvidenceProvider,
        config: WorldEvidenceRouterConfig,
        *,
        evidence_adapter: LowRankEvidenceAdapterBank | None = None,
        router: FutureEvidenceRouter | None = None,
        residual: UtilityCalibratedResidual | None = None,
    ) -> None:
        super().__init__()
        self.active_parent = active_parent
        self.evidence_provider = evidence_provider
        self.config = config
        self.evidence_adapter = evidence_adapter or LowRankEvidenceAdapterBank(config)
        self.router = router or FutureEvidenceRouter(config)
        self.residual = residual or UtilityCalibratedResidual(config)

    def velocity(
        self,
        base_vision_tokens: Tensor,
        current_state: Tensor,
        current_visual_latent: Tensor,
        shared_visual_latent: Tensor,
        action_inputs: Tensor,
        flow_time: Tensor,
        valid_agent_mask: Tensor,
        *,
        force_world_evidence_gate_zero: bool = False,
        force_all_world_gates_zero: bool = False,
        forced_group: int | Tensor | None = None,
        execute_evidence_when_gate_zero: bool = False,
        future_intervention: Callable[
            [PredictedFutureLatents], PredictedFutureLatents
        ] | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        """Evaluate the active candidate under a normal or causal intervention.

        ``force_world_evidence_gate_zero`` disables only the new S4 residual.
        ``force_all_world_gates_zero`` additionally asks the active R6 parent to
        disable its legacy world adapter.  In either zero-gate path the returned
        velocity is the exact parent tensor; no multiply/add round trip is used.
        """

        if valid_agent_mask.dtype != torch.bool:
            raise TypeError("valid_agent_mask must have dtype bool")
        new_gate_disabled = bool(
            force_world_evidence_gate_zero or force_all_world_gates_zero
        )
        need_cache = not new_gate_disabled or execute_evidence_when_gate_zero
        parent_velocity, parent_aux, parent_cache = self._active_parent_velocity(
            base_vision_tokens,
            current_state,
            current_visual_latent,
            shared_visual_latent,
            action_inputs,
            flow_time,
            valid_agent_mask,
            force_all_world_gates_zero=force_all_world_gates_zero,
            return_cache=need_cache,
            future_intervention=future_intervention,
        )
        if new_gate_disabled and not execute_evidence_when_gate_zero:
            return parent_velocity, self._disabled_diagnostics(
                parent_velocity,
                parent_aux,
                force_all_world_gates_zero=force_all_world_gates_zero,
            )
        if parent_cache is None:
            raise RuntimeError(
                "active_parent.velocity(return_cache=True) did not expose the "
                "S4 flow_features/clean_actions cache"
            )
        cache = _normalize_parent_cache(parent_velocity, parent_cache)
        q = cache.flow_features
        if q.shape[:3] != parent_velocity.shape[:3]:
            raise ValueError("active parent Flow features and velocity disagree")
        if cache.clean_actions.shape != action_inputs.shape:
            raise ValueError("active parent clean_actions violate the action contract")
        if cache.predicted_futures is None:
            raise RuntimeError(
                "active parent cache must contain the complete own/peer/shared "
                "predicted_futures object"
            )
        evidence = self.evidence_provider(
            current_state,
            current_visual_latent,
            shared_visual_latent,
            cache.clean_actions.detach(),
            valid_agent_mask,
            predicted_futures=cache.predicted_futures,
        )
        adapted = self.evidence_adapter(q, evidence)
        routed = self.router(q, evidence, group_mask=adapted.group_mask)
        effective_pi = (
            routed.pi
            if forced_group is None
            else forced_group_distribution(
                routed.group_mask,
                query_count=q.shape[2],
                forced_group=forced_group,
                dtype=routed.pi.dtype,
            )
        )
        residual, gate, mixture = self.residual(
            q,
            adapted.z,
            effective_pi,
            valid_agent_mask,
            force_gate_zero=new_gate_disabled,
        )
        # The causal zero-gate contract is bit-exact even when evidence was
        # deliberately executed to audit off-path side effects.
        velocity = parent_velocity if new_gate_disabled else parent_velocity + residual
        diagnostics: dict[str, object] = {
            "active_parent_velocity": parent_velocity,
            "flow_features": q,
            "clean_actions": cache.clean_actions,
            "predicted_futures": cache.predicted_futures,
            "evidence_tokens": evidence.tokens,
            "evidence_mask": evidence.mask,
            "evidence_z": adapted.z,
            "router_logits": routed.logits,
            "router_pi": routed.pi,
            "effective_pi": effective_pi,
            "group_mask": routed.group_mask,
            "group_summary": routed.group_summary,
            "new_gate": gate,
            "new_residual": residual,
            "evidence_mixture": mixture,
            "forced_group": forced_group,
            "force_world_evidence_gate_zero": force_world_evidence_gate_zero,
            "force_all_world_gates_zero": force_all_world_gates_zero,
            "parent": parent_aux,
        }
        return velocity, diagnostics

    @torch.no_grad()
    def forced_evidence_audit(
        self,
        diagnostics: Mapping[str, object],
        target_velocity: Tensor,
        valid_agent_mask: Tensor,
        *,
        valid_action_query_mask: Tensor | None = None,
    ) -> ForcedEvidenceAudit:
        """Build the stop-gradient utility target from one cached forward."""

        return forced_evidence_velocity_errors(
            active_parent_velocity=_tensor_item(
                diagnostics, "active_parent_velocity"
            ),
            flow_features=_tensor_item(diagnostics, "flow_features"),
            evidence_z=_tensor_item(diagnostics, "evidence_z"),
            group_mask=_tensor_item(diagnostics, "group_mask"),
            target_velocity=target_velocity,
            valid_agent_mask=valid_agent_mask,
            residual=self.residual,
            valid_action_query_mask=valid_action_query_mask,
        )

    def _active_parent_velocity(
        self,
        base_vision_tokens: Tensor,
        current_state: Tensor,
        current_visual_latent: Tensor,
        shared_visual_latent: Tensor,
        action_inputs: Tensor,
        flow_time: Tensor,
        valid_agent_mask: Tensor,
        *,
        force_all_world_gates_zero: bool,
        return_cache: bool,
        future_intervention: Callable[
            [PredictedFutureLatents], PredictedFutureLatents
        ] | None,
    ) -> tuple[Tensor, Mapping[str, object], object | None]:
        velocity_method = getattr(self.active_parent, "velocity", None)
        if not callable(velocity_method):
            raise TypeError("active_parent must expose a callable velocity method")
        arguments: dict[str, object] = {}
        if force_all_world_gates_zero:
            if _accepts_keyword(velocity_method, "force_all_world_gates_zero"):
                arguments["force_all_world_gates_zero"] = True
            elif _accepts_keyword(velocity_method, "force_gate_zero"):
                arguments["force_gate_zero"] = True
            else:
                raise TypeError(
                    "active parent cannot execute the all-world-gates-zero audit"
                )
        if return_cache and _accepts_keyword(velocity_method, "return_cache"):
            arguments["return_cache"] = True
        if future_intervention is not None:
            if not _accepts_keyword(velocity_method, "future_intervention"):
                raise TypeError("active parent cannot apply a future intervention")
            arguments["future_intervention"] = future_intervention
        result = velocity_method(
            base_vision_tokens,
            current_state,
            current_visual_latent,
            shared_visual_latent,
            action_inputs,
            flow_time,
            valid_agent_mask,
            **arguments,
        )
        if not isinstance(result, tuple) or len(result) not in {2, 3}:
            raise TypeError(
                "active_parent.velocity must return (velocity, aux[, cache])"
            )
        velocity = result[0]
        if not isinstance(velocity, Tensor):
            raise TypeError("active parent velocity must be a Tensor")
        auxiliary = result[1] if isinstance(result[1], Mapping) else {}
        cache: object | None = result[2] if len(result) == 3 else None
        if cache is None and isinstance(result[1], Mapping):
            cache = result[1].get("cache")
            if cache is None and {
                "flow_features",
                "clean_actions",
            }.issubset(result[1]):
                cache = result[1]
        return velocity, auxiliary, cache

    @staticmethod
    def _disabled_diagnostics(
        parent_velocity: Tensor,
        parent_aux: Mapping[str, object],
        *,
        force_all_world_gates_zero: bool,
    ) -> dict[str, object]:
        return {
            "active_parent_velocity": parent_velocity,
            "new_gate": parent_velocity.new_zeros(()),
            "new_residual": torch.zeros_like(parent_velocity),
            "router_logits": None,
            "router_pi": None,
            "effective_pi": None,
            "group_mask": None,
            "forced_group": None,
            "force_world_evidence_gate_zero": True,
            "force_all_world_gates_zero": force_all_world_gates_zero,
            "parent": parent_aux,
        }


def forced_group_distribution(
    group_mask: Tensor,
    *,
    query_count: int,
    forced_group: int | Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """Return a one-hot group distribution without unmasking invalid evidence."""

    if group_mask.ndim != 3 or group_mask.dtype != torch.bool:
        raise ValueError("group_mask must be bool [B,A,M]")
    if query_count <= 0:
        raise ValueError("query_count must be positive")
    batch_size, agents, groups = group_mask.shape
    if isinstance(forced_group, int):
        if not 0 <= forced_group < groups:
            raise ValueError("forced_group index is outside the evidence bank")
        indices = torch.full(
            (batch_size, agents, query_count),
            forced_group,
            dtype=torch.long,
            device=group_mask.device,
        )
    elif isinstance(forced_group, Tensor):
        if forced_group.shape != (batch_size, agents, query_count):
            raise ValueError("forced_group tensor must be [B,A,Q]")
        if forced_group.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("forced_group tensor must contain integer indices")
        indices = forced_group.to(dtype=torch.long)
        if bool(((indices < 0) | (indices >= groups)).any()):
            raise ValueError("forced_group tensor contains an invalid index")
    else:
        raise TypeError("forced_group must be an int or integer Tensor")
    one_hot = F.one_hot(indices, num_classes=groups).to(dtype=dtype)
    available = group_mask[:, :, None].expand(-1, -1, query_count, -1)
    return one_hot * available.to(dtype=dtype)


@torch.no_grad()
def forced_evidence_velocity_errors(
    *,
    active_parent_velocity: Tensor,
    flow_features: Tensor,
    evidence_z: Tensor,
    group_mask: Tensor,
    target_velocity: Tensor,
    valid_agent_mask: Tensor,
    residual: UtilityCalibratedResidual,
    valid_action_query_mask: Tensor | None = None,
) -> ForcedEvidenceAudit:
    """Evaluate every valid evidence group and construct ``q_util``.

    The function is unconditionally no-grad.  Callers should also put the model
    in ``eval`` mode for the scheduled audit so stochastic layers cannot alter
    the forced-group comparison.
    """

    if active_parent_velocity.shape != target_velocity.shape:
        raise ValueError("target_velocity must match active_parent_velocity")
    if flow_features.shape[:3] != active_parent_velocity.shape[:3]:
        raise ValueError("flow_features and velocity must share B/A/Q axes")
    if evidence_z.shape[:3] != flow_features.shape[:3]:
        raise ValueError("evidence_z and flow_features must share B/A/Q axes")
    if group_mask.shape != evidence_z.shape[:2] + (evidence_z.shape[3],):
        raise ValueError("group_mask must be [B,A,M]")
    if group_mask.dtype != torch.bool or valid_agent_mask.dtype != torch.bool:
        raise TypeError("group and agent masks must have dtype bool")
    batch_size, agents, queries, groups = evidence_z.shape[:4]
    if valid_agent_mask.shape != (batch_size, agents):
        raise ValueError("valid_agent_mask must be [B,A]")
    query_valid = _query_mask(
        valid_agent_mask,
        queries,
        valid_action_query_mask=valid_action_query_mask,
    )
    per_group: list[Tensor] = []
    for group in range(groups):
        pi = forced_group_distribution(
            group_mask,
            query_count=queries,
            forced_group=group,
            dtype=flow_features.dtype,
        )
        delta = residual(
            flow_features,
            evidence_z,
            pi,
            valid_agent_mask,
        )[0]
        velocity = active_parent_velocity + delta
        error = (velocity - target_velocity).float().square().mean(dim=-1)
        per_group.append(error)
    errors = torch.stack(per_group, dim=-1)
    expanded_group_mask = group_mask[:, :, None].expand(
        -1, -1, queries, -1
    )
    safe_errors = torch.where(expanded_group_mask, errors, 0)
    count = expanded_group_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = safe_errors.sum(dim=-1, keepdim=True) / count
    variance = (
        torch.where(expanded_group_mask, (errors - mean).square(), 0).sum(
            dim=-1, keepdim=True
        )
        / count
    )
    temperature = variance.sqrt().clamp_min(1.0e-3)
    utility_target = _masked_softmax(
        -errors / temperature,
        expanded_group_mask,
        dim=-1,
    )
    valid_query = query_valid & group_mask.sum(dim=-1)[:, :, None].ge(2)
    utility_target = utility_target * valid_query[..., None]
    return ForcedEvidenceAudit(
        velocity_errors=torch.where(expanded_group_mask, errors, 0).detach(),
        utility_target=utility_target.detach(),
        temperature=temperature.squeeze(-1).detach(),
        group_mask=group_mask.detach(),
        valid_query_mask=valid_query.detach(),
    )


def world_utility_coupling_loss(
    router_pi: Tensor,
    utility_target: Tensor,
    group_mask: Tensor,
    valid_agent_mask: Tensor,
    *,
    valid_action_query_mask: Tensor | None = None,
) -> Tensor:
    """Masked ``KL(q_util || pi_route)`` with a detached utility target."""

    if router_pi.shape != utility_target.shape or router_pi.ndim != 4:
        raise ValueError("router_pi and utility_target must match [B,A,Q,M]")
    batch_size, agents, queries, groups = router_pi.shape
    if group_mask.shape != (batch_size, agents, groups):
        raise ValueError("group_mask must be [B,A,M]")
    if group_mask.dtype != torch.bool or valid_agent_mask.dtype != torch.bool:
        raise TypeError("group and agent masks must have dtype bool")
    query_valid = _query_mask(
        valid_agent_mask,
        queries,
        valid_action_query_mask=valid_action_query_mask,
    )
    query_valid = query_valid & group_mask.sum(dim=-1)[:, :, None].ge(2)
    expanded_mask = group_mask[:, :, None].expand(-1, -1, queries, -1)
    target = torch.where(
        expanded_mask, utility_target.detach().float(), 0
    )
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    route = torch.where(expanded_mask, router_pi.float(), 0)
    route = route / route.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    positive = target > 0
    terms = torch.where(
        positive,
        target * (target.clamp_min(1.0e-8).log() - route.clamp_min(1.0e-8).log()),
        0,
    )
    per_query = terms.sum(dim=-1)
    if not bool(query_valid.any()):
        return router_pi.sum() * 0
    return per_query[query_valid].mean()


def _normalize_parent_cache(
    parent_velocity: Tensor,
    value: object,
) -> ActiveParentVelocityCache:
    if isinstance(value, ActiveParentVelocityCache):
        if not torch.equal(value.active_parent_velocity, parent_velocity):
            raise ValueError("cached active parent velocity differs from return value")
        return value
    flow_features = _cache_value(value, "flow_features")
    clean_actions = _cache_value(value, "clean_actions")
    cached_velocity = _optional_cache_value(value, "active_parent_velocity")
    if cached_velocity is not None and not torch.equal(cached_velocity, parent_velocity):
        raise ValueError("cached active parent velocity differs from return value")
    predicted_futures = None
    for name in ("predicted_futures", "futures", "local_futures"):
        predicted_futures = _optional_object_value(value, name)
        if predicted_futures is not None:
            break
    return ActiveParentVelocityCache(
        active_parent_velocity=parent_velocity,
        flow_features=flow_features,
        clean_actions=clean_actions,
        predicted_futures=predicted_futures,
    )


def _cache_value(value: object, name: str) -> Tensor:
    item = _optional_object_value(value, name)
    if not isinstance(item, Tensor):
        raise TypeError(f"active parent cache {name} must be a Tensor")
    return item


def _optional_cache_value(value: object, name: str) -> Tensor | None:
    item = _optional_object_value(value, name)
    if item is None:
        return None
    if not isinstance(item, Tensor):
        raise TypeError(f"active parent cache {name} must be a Tensor")
    return item


def _optional_object_value(value: object, name: str) -> object | None:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _tensor_item(value: Mapping[str, object], name: str) -> Tensor:
    item = value.get(name)
    if not isinstance(item, Tensor):
        raise TypeError(f"diagnostics[{name!r}] must be a Tensor")
    return item


def _query_mask(
    valid_agent_mask: Tensor,
    queries: int,
    *,
    valid_action_query_mask: Tensor | None,
) -> Tensor:
    base = valid_agent_mask[:, :, None].expand(-1, -1, queries)
    if valid_action_query_mask is None:
        return base
    if valid_action_query_mask.shape != base.shape:
        raise ValueError("valid_action_query_mask must be [B,A,Q]")
    if valid_action_query_mask.dtype != torch.bool:
        raise TypeError("valid_action_query_mask must have dtype bool")
    return base & valid_action_query_mask


def _accepts_keyword(function: object, name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _masked_softmax(logits: Tensor, mask: Tensor, *, dim: int) -> Tensor:
    mask = mask.expand_as(logits)
    masked = logits.masked_fill(~mask, float("-inf"))
    any_valid = mask.any(dim=dim, keepdim=True)
    safe = torch.where(any_valid, masked, torch.zeros_like(masked))
    probabilities = torch.softmax(safe, dim=dim)
    return torch.where(mask, probabilities, 0)


__all__ = [
    "ActiveParentVelocityCache",
    "ForcedEvidenceAudit",
    "S4ActiveTeamFutureProvider",
    "ScaleAlignedActiveWorldFlow",
    "UtilityCalibratedWorldFlow",
    "forced_evidence_velocity_errors",
    "forced_group_distribution",
    "world_utility_coupling_loss",
]
