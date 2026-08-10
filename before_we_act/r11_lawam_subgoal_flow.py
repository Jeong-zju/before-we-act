"""R11 candidate D: official LaWAM latent-subgoal flow policy."""
from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch
from torch import nn

from before_we_act.r11_vendor import validate_asset_bundle_receipt, verify_vendor_checkout


MODEL_NAME = "R11LaWAMSubgoalFlow"
INFERENCE_MODES = ("normal", "prediction_off", "prediction_shuffled")
ACTION_CONDITION_MODES = ("normal", "action_shuffled")
ACTION_HORIZON = 100
ACTION_DIM = 8
EXECUTION_CADENCE = 100


def _move(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _latest_valid_indices(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if mask.ndim != 2 or mask.shape[1] != 4:
        raise ValueError("LaWAM adapter requires the four frozen future offsets")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("LaWAM sample has no valid future target")
    weights = torch.arange(1, 5, device=mask.device).view(1, 4)
    return (mask.long() * weights).argmax(dim=1)


class LaWAMRoboFactoryAdapter:
    """Translate the common six-task batch through official LaWAM builders."""

    def __init__(self, train_collator, infer_batch_builder, spatial_preprocess=None) -> None:
        self.train_collator = train_collator
        self.infer_batch_builder = infer_batch_builder
        self.spatial_preprocess = spatial_preprocess

    @staticmethod
    def _agent(batch: Mapping, index: int) -> int:
        value = batch["agent"]
        if isinstance(value, torch.Tensor):
            return int(value[index].item())
        return int(value[index])

    def training_batch(self, batch: Mapping, device: torch.device) -> dict:
        current = batch["current_rgb"].detach().cpu()
        future = batch["future_rgb"].detach().cpu()
        future_index = _latest_valid_indices(batch["future_mask"].detach().cpu())
        qpos = batch["qpos"].detach().cpu()
        actions = batch["action"].detach().cpu()
        action_mask = batch["action_mask"].detach().cpu().bool()
        features = []
        for index in range(current.shape[0]):
            valid_actions = int(action_mask[index].sum().item())
            if valid_actions < 1:
                raise ValueError("LaWAM training action has no valid timestep")
            target = future[index, int(future_index[index].item()), 0]
            primary_video = torch.stack((current[index, 0], target), dim=0).unsqueeze(0)
            wrist_images = current[index, 1].unsqueeze(0)
            if self.spatial_preprocess is not None:
                primary_video = self.spatial_preprocess(primary_video[0]).unsqueeze(0)
                wrist_images = self.spatial_preprocess(wrist_images)
            features.append(
                {
                    "primary_videos": primary_video,
                    "wrist_images": wrist_images,
                    "lang": str(batch["task_text"][index]),
                    "state": qpos[index].unsqueeze(0),
                    "action": actions[index, :valid_actions],
                    # Category zero is reserved for non-robot data upstream.
                    "embodiment_id": self._agent(batch, index) + 1,
                    # With horizon_sec=1 this makes the official time-grid mask
                    # match tail-truncated demonstrations exactly.
                    "action_hz": float(valid_actions),
                }
            )
        return _move(self.train_collator(features), device)

    def inference_batch(self, batch: Mapping, device: torch.device) -> dict:
        current = batch["current_rgb"].detach().cpu()
        qpos = batch["qpos"].detach().cpu()
        examples = []
        for index in range(current.shape[0]):
            global_hwc = current[index, 0].permute(1, 2, 0).contiguous().numpy()
            local_hwc = current[index, 1].permute(1, 2, 0).contiguous().numpy()
            examples.append(
                {
                    "primary_image": [np.asarray(global_hwc)],
                    "wrist_image": [np.asarray(local_hwc)],
                    "lang": str(batch["task_text"][index]),
                    "state": qpos[index].numpy(),
                    "embodiment_id": self._agent(batch, index) + 1,
                    "action_hz": float(ACTION_HORIZON),
                }
            )
        return _move(self.infer_batch_builder.build_infer_batch(examples), device)


class R11LaWAMSubgoalFlow(nn.Module):
    """Official LaWAM backend with measurable future/action interventions."""

    def __init__(self, policy_backend: nn.Module, adapter: LaWAMRoboFactoryAdapter) -> None:
        super().__init__()
        self.policy_backend = policy_backend
        self.adapter = adapter
        self.prediction_mode = "normal"
        self.action_condition_mode = "normal"
        self.provenance: dict = {}
        self._future_hook_handle = self.policy_backend.lam.decoder.register_forward_hook(
            self._future_hook
        )
        self._action_hook_handle = self.policy_backend.vlm_to_lam.register_forward_hook(
            self._action_condition_hook
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _intervene(value: torch.Tensor, *, off: bool) -> torch.Tensor:
        if off:
            return torch.zeros_like(value)
        if value.shape[0] > 1:
            return value.roll(1, dims=0)
        if value.shape[1] > 1:
            return value.flip(1)
        return -value

    def _future_hook(self, _module, _inputs, output):
        if self.prediction_mode == "normal":
            return output
        tensor = output[0] if isinstance(output, tuple) else output
        changed = self._intervene(
            tensor, off=self.prediction_mode == "prediction_off"
        )
        if self.prediction_mode not in INFERENCE_MODES:
            raise ValueError(self.prediction_mode)
        if isinstance(output, tuple):
            return (changed, *output[1:])
        return changed

    def _action_condition_hook(self, _module, _inputs, output):
        if self.action_condition_mode == "normal":
            return output
        if self.action_condition_mode != "action_shuffled":
            raise ValueError(self.action_condition_mode)
        return self._intervene(output, off=False)

    def training_step(self, batch: Mapping, update: int) -> dict:
        self.prediction_mode = "normal"
        self.action_condition_mode = "normal"
        self.policy_backend.set_flow_train_step(update)
        output = self.policy_backend(
            batch=self.adapter.training_batch(batch, self.device)
        )
        return {
            "loss": output["loss_total"],
            "action_loss": output["loss_flow"],
            "world_loss": output["loss_perceptual"],
            "distill_loss": output["loss_distill"],
            "scheduled_prediction_probability": self.policy_backend._flow_h_t1_pred_prob(),
        }

    def forward(
        self,
        batch: Mapping,
        *,
        mode: str = "normal",
        action_condition_mode: str = "normal",
        update: int = 0,
    ) -> dict:
        del update
        if mode not in INFERENCE_MODES:
            raise ValueError(mode)
        if action_condition_mode not in ACTION_CONDITION_MODES:
            raise ValueError(action_condition_mode)
        prepared = self.adapter.inference_batch(batch, self.device)
        self.prediction_mode = mode
        self.action_condition_mode = action_condition_mode
        try:
            actions, intermediates = self.policy_backend.predict_action(
                batch=prepared,
                return_intermediates=True,
                return_padded=True,
            )
        finally:
            self.prediction_mode = "normal"
            self.action_condition_mode = "normal"
        if actions.shape[1:] != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"LaWAM action contract drift: {tuple(actions.shape)}")
        future = intermediates["h_t1_pred"]
        if isinstance(future, torch.Tensor):
            future = future.to(actions.device)
        return {
            "action": actions,
            "future_prediction": future,
            "current_prediction_tokens": intermediates["h_t"],
            "execution_cadence": EXECUTION_CADENCE,
            "mode": mode,
            "action_condition_mode": action_condition_mode,
        }

    def causal_probe(
        self, batch: Mapping, *, action_condition_mode: str = "normal"
    ) -> dict:
        """Expose official LAM h_t/h_t1_pred/h_t1_gt without pixel proxies."""

        if action_condition_mode not in ACTION_CONDITION_MODES:
            raise ValueError(action_condition_mode)
        prepared = self.adapter.training_batch(batch, self.device)
        self.prediction_mode = "normal"
        self.action_condition_mode = action_condition_mode
        try:
            shared = self.policy_backend._run_shared_encoding_train(
                prepared_batch=prepared,
                source="R11LaWAMSubgoalFlow.causal_probe",
                lam_features_with_no_grad=True,
            )
        finally:
            self.prediction_mode = "normal"
            self.action_condition_mode = "normal"
        return {
            "future_prediction": shared.h_t1_pred,
            "future_target": shared.h_t1_gt.detach(),
            "persistence_prediction": shared.h_t.detach(),
            "action_condition_mode": action_condition_mode,
        }


def _load_compatible_pretrain(backend: nn.Module, checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("LaWAM pretrain checkpoint must be a state dict")
    target = backend.state_dict()
    compatible = {}
    mismatched = []
    unexpected = []
    for original_key, value in checkpoint.items():
        key = original_key.removeprefix("policy_backend.")
        if key not in target:
            unexpected.append(original_key)
        elif tuple(value.shape) != tuple(target[key].shape):
            mismatched.append(
                {
                    "key": original_key,
                    "checkpoint": list(value.shape),
                    "model": list(target[key].shape),
                }
            )
        else:
            compatible[key] = value
    load_result = backend.load_state_dict(compatible, strict=False)
    required_prefixes = ("flow.DiT.", "vlm_to_lam.", "flow_action_query")
    for prefix in required_prefixes:
        if not any(key == prefix or key.startswith(prefix) for key in compatible):
            raise ValueError(f"LaWAM pretrain has no compatible {prefix} weights")
    return {
        "loaded": len(compatible),
        "checkpoint_keys": len(checkpoint),
        "missing": len(load_result.missing_keys),
        "unexpected": len(unexpected),
        "shape_mismatches": mismatched,
        "required_prefixes": list(required_prefixes),
    }


def _build_official_backend(config: Mapping, artifacts: Mapping[str, str]):
    from starVLA.dataloader.latent_world_train_collator import LatentWorldTrainCollator
    from starVLA.model.framework.latent_world.batch_builder import (
        LatentWorldPolicyInferBatchBuilder,
    )
    from starVLA.model.framework.latent_world.processor_utils import (
        build_latent_world_processor_spec,
    )
    from starVLA.model.framework.latent_world.batch_utils import (
        prepare_frame_spatial_uint8,
    )
    from starVLA.model.framework.latent_world.runtime.freeze_policy import (
        LatentWorldPolicyFreezeConfig,
        apply_policy_freeze,
    )
    from starVLA.model.framework.latent_world.vlm_adapter import LatentWorldPolicyVLMAdapter
    from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingConfig
    from starVLA.model.framework.vlas.lawam import (
        LatentWorldPolicyBackend,
        LatentWorldPolicyConfig,
    )

    model_cfg = config["model_config"]
    flow_cfg = ConditionalFlowMatchingConfig(**model_cfg["flow_cfg"])
    policy_cfg = LatentWorldPolicyConfig(
        flow_cfg=flow_cfg,
        future_action_window_size=ACTION_HORIZON - 1,
        past_action_window_size=0,
        action_horizon=ACTION_HORIZON,
        hf_cache_dir=config["foundation"]["hf_cache"],
        lam_ckpt_path=artifacts["lam_checkpoint"],
        lam_yaml_path=artifacts["lam_yaml"],
        perceptual_weight=model_cfg["perceptual_weight"],
        enable_loss_distill=True,
        lam_encoder_distill_weight=model_cfg["lam_encoder_distill_weight"],
        future_prediction=True,
        repeated_diffusion_steps=model_cfg["repeated_diffusion_steps"],
        enable_flow_h_t1_scheduled_sampling=True,
        flow_h_t1_pred_prob_start=0.0,
        flow_h_t1_pred_prob_end=1.0,
        flow_h_t1_pred_ramp_steps=model_cfg["scheduled_sampling_ramp_updates"],
        detach_future_feature=False,
        num_action_queries=model_cfg["num_action_queries"],
        flow_action_num_queries=model_cfg["flow_action_num_queries"],
    )
    backend = LatentWorldPolicyBackend.build(
        policy_cfg, vlm_model_id=artifacts["qwen_directory"]
    )
    apply_policy_freeze(
        backend,
        LatentWorldPolicyFreezeConfig(
            freeze_vision_backbone=False,
            freeze_llm_backbone=False,
            freeze_last_llm_layer=True,
            freeze_embedding=True,
            unfreeze_vision_merger=True,
            unfreeze_lam_decoder=True,
            keep_llm_first_n_layers=model_cfg["keep_llm_first_n_layers"],
        ),
    )
    if hasattr(backend.vlm, "gradient_checkpointing_enable"):
        backend.vlm.gradient_checkpointing_enable()
    if hasattr(backend.vlm, "enable_input_require_grads"):
        backend.vlm.enable_input_require_grads()
        # Transformers marks the output of a frozen input embedding as a
        # requires-grad leaf. The pinned LaWAM backend then replaces action
        # placeholders in-place, which PyTorch correctly rejects for a leaf.
        # Keep the official query-injection path intact while returning a
        # differentiable non-leaf tensor from the embedding boundary.
        embedding = backend.vlm.get_input_embeddings()
        backend._r11_embedding_clone_hook = embedding.register_forward_hook(
            _clone_requires_grad_leaf
        )
    pretrain = _load_compatible_pretrain(backend, artifacts["pretrain_checkpoint"])

    prompt_config = SimpleNamespace(datasets=None)
    vlm_adapter = LatentWorldPolicyVLMAdapter(
        model=backend.vlm,
        processor=backend.processor,
        config=prompt_config,
        placeholder_token=policy_cfg.latent_action_placeholder_token,
        act_queries=int(backend.num_action_queries),
        flow_queries=int(backend.flow_action_query.shape[0]),
    )
    infer_builder = LatentWorldPolicyInferBatchBuilder(
        policy_cfg=policy_cfg,
        policy_backend=backend,
        policy_vlm_adapter=vlm_adapter,
        enable_primary_random_resized_crop=False,
    )
    processor_spec = build_latent_world_processor_spec(
        policy_cfg=policy_cfg, vlm_model_id=artifacts["qwen_directory"]
    )
    train_collator = LatentWorldTrainCollator(
        policy_cfg=policy_cfg,
        processor_spec=processor_spec,
        act_queries=int(backend.num_action_queries),
        flow_queries=int(backend.flow_action_query.shape[0]),
        enable_primary_video_aug=False,
        enable_primary_random_resized_crop=False,
    )
    # Reuse the already loaded processor and exact placeholder id.
    train_collator._processor = backend.processor
    train_collator._placeholder_token_id = backend.placeholder_token_id
    train_spatial_preprocess = partial(
        prepare_frame_spatial_uint8,
        target_hw=infer_builder.DEFAULT_INFER_IMAGE_HW,
        apply_center_crop_90=False,
    )
    return (
        backend,
        LaWAMRoboFactoryAdapter(
            train_collator,
            infer_builder,
            spatial_preprocess=train_spatial_preprocess,
        ),
        pretrain,
    )


def _clone_requires_grad_leaf(_module, _inputs, output):
    """Make frozen embedding outputs safe for official in-place query injection."""

    if isinstance(output, torch.Tensor) and output.requires_grad and output.is_leaf:
        return output.clone()
    return output


def build_lawam_subgoal_flow(config: Mapping, project_root: str | Path) -> R11LaWAMSubgoalFlow:
    project_root = Path(project_root).resolve()
    vendor = Path(os.environ.get(config["vendor_env"], config["vendor_default"])).resolve()
    source = verify_vendor_checkout(project_root / config["source_receipt"], vendor)
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    foundation = config["foundation"]
    assets = validate_asset_bundle_receipt(
        foundation["receipt"],
        {
            "repositories": foundation["repositories"],
            "licenses": foundation["licenses"],
            "task_sft_checkpoint": "none",
        },
    )
    backend, adapter, pretrain = _build_official_backend(config, assets["artifacts"])
    model = R11LaWAMSubgoalFlow(backend, adapter)
    model.provenance = {
        "model": MODEL_NAME,
        "source": source,
        "foundation_assets": assets,
        "w10_init": "none",
        "lawam_pretrain": pretrain,
        "task_sft_checkpoint": "none",
        "action_horizon": ACTION_HORIZON,
        "lam_spatial_preprocess": "official prepare_frame_spatial_uint8 256x256",
        "scheduled_sampling": {"start": 0.0, "end": 1.0, "updates": 10000},
        "trainable_parameter_map": {
            name: bool(parameter.requires_grad) for name, parameter in model.named_parameters()
        },
    }
    return model


def load_candidate_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("model") != MODEL_NAME:
        raise ValueError("candidate D config model mismatch")
    if tuple(config.get("inference_modes", ())) != INFERENCE_MODES:
        raise ValueError("candidate D inference mode drift")
    flow = config["model_config"]["flow_cfg"]
    if flow["action_dim"] != ACTION_DIM or config["model_config"]["action_horizon"] != ACTION_HORIZON:
        raise ValueError("candidate D action contract drift")
    if float(flow["horizon_sec"]) != 1.0:
        raise ValueError("candidate D 100-step natural-time contract drift")
    return config
