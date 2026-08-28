"""Data config for decentralized RoboFactory π0.5 LoRA fine-tuning."""
from __future__ import annotations

import dataclasses

import openpi.models.pi0_config as pi0_config
import openpi.transforms as transforms
from openpi.models.model import ModelType
from openpi.policies.robofactory_policy import CastFloat32, RoboFactoryInputs, RoboFactoryOutputs
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders


@dataclasses.dataclass(frozen=True)
class RoboFactoryDataConfig(_config.DataConfigFactory):
    """Use all manifest episodes and only local observation/action fields."""

    default_prompt: str | None = None

    def create(self, assets_dirs, model_config):
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "image": "observation/image",
                        "state": "observation/state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data = transforms.Group(
            inputs=[RoboFactoryInputs(model_type=model_config.model_type)],
            outputs=[RoboFactoryOutputs()],
        )
        # The stored commanded values are absolute joint-position targets.
        # Convert arm joints to the delta convention used by the π0 base model;
        # gripper dimension 7 remains absolute.
        data = data.push(
            inputs=[transforms.DeltaActions(transforms.make_bool_mask(7, -1))],
            outputs=[transforms.AbsoluteActions(transforms.make_bool_mask(7, -1))],
        )
        model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        model_transforms = model_transforms.push(inputs=[CastFloat32()])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
        )


def make_config() -> _config.TrainConfig:
    model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    return _config.TrainConfig(
        name="pi05_robofactory_lora",
        exp_name="decentralized_all150",
        model=model,
        data=RoboFactoryDataConfig(
            repo_id="robofactory", assets=_config.AssetsConfig(asset_id="robofactory")
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=5e-5, decay_steps=120000, decay_lr=5e-6
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/workspace/bwa_vla_runs/openpi/assets",
        checkpoint_base_dir="/workspace/bwa_vla_runs/openpi/checkpoints",
        batch_size=32,
        num_train_steps=120000,
        save_interval=1000,
        keep_period=10000,
        num_workers=0,
        # The deployment target is one H200.  FSDP=1 keeps the same mesh API
        # while avoiding the four-device assumption from the earlier A100 run.
        fsdp_devices=1,
        wandb_enabled=False,
        policy_metadata={"protocol": "decentralized_local_rgb_qpos_action", "episodes_per_task": 150},
    )
