from __future__ import annotations

import dataclasses

import openpi.models.pi0_config as pi0_config
import openpi.transforms as transforms
from openpi.policies.duobench_policy import CastFloat32, DuoBenchInputs, DuoBenchOutputs
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders


@dataclasses.dataclass(frozen=True)
class DuoBenchDataConfig(_config.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        repack = transforms.Group(inputs=[transforms.RepackTransform({
            "head": "observation/head", "wrist": "observation/wrist",
            "state": "observation/state", "actions": "actions", "prompt": "prompt",
        })])
        data = transforms.Group(
            inputs=[DuoBenchInputs(model_type=model_config.model_type)],
            outputs=[DuoBenchOutputs()],
        ).push(
            inputs=[transforms.DeltaActions(transforms.make_bool_mask(7, -1))],
            outputs=[transforms.AbsoluteActions(transforms.make_bool_mask(7, -1))],
        )
        model_transforms = _config.ModelTransformFactory()(model_config).push(inputs=[CastFloat32()])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack, data_transforms=data,
            model_transforms=model_transforms, action_sequence_keys=("actions",),
        )


def make_config():
    model = pi0_config.Pi0Config(
        pi05=True, action_dim=32, action_horizon=16,
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
    )
    return _config.TrainConfig(
        name="pi05_duobench_lora", exp_name="all550_4gpu_dp_b128_25k",
        model=model,
        data=DuoBenchDataConfig(repo_id="duobench", assets=_config.AssetsConfig(asset_id="duobench")),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=5e-5, decay_steps=25000, decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=model.get_freeze_filter(), ema_decay=None,
        assets_base_dir="/workspace/runs/pi05_duo/assets",
        checkpoint_base_dir="/workspace/runs/pi05_duo/checkpoints",
        batch_size=128, num_train_steps=25000, save_interval=1000,
        keep_period=5000, num_workers=12, fsdp_devices=1, wandb_enabled=False,
        policy_metadata={
            "benchmark": "DuoBench", "episodes": 550,
            "protocol": "shared_weights_decentralized_head_local_wrist_own_state_to_own_action8",
            "action_lag_rows": 1, "global_batch_size": 128,
            "sample_budget": 3200000, "parallelism": "four-way data parallel",
        },
    )
