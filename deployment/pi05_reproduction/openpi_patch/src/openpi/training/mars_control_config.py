"""π0.5 LoRA config for four-task MARS-Control."""
from __future__ import annotations
import dataclasses
import openpi.models.pi0_config as pi0_config
import openpi.transforms as transforms
from openpi.policies.robofactory_policy import CastFloat32,RoboFactoryInputs,RoboFactoryOutputs
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
@dataclasses.dataclass(frozen=True)
class MarsControlDataConfig(_config.DataConfigFactory):
    default_prompt:str|None=None
    def create(self,assets_dirs,model_config):
        repack=transforms.Group(inputs=[transforms.RepackTransform({"image":"observation/image","state":"observation/state","actions":"actions","prompt":"prompt"})])
        data=transforms.Group(inputs=[RoboFactoryInputs(model_type=model_config.model_type)],outputs=[RoboFactoryOutputs()]); data=data.push(inputs=[transforms.DeltaActions(transforms.make_bool_mask(7,-1))],outputs=[transforms.AbsoluteActions(transforms.make_bool_mask(7,-1))])
        model_transforms=_config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config).push(inputs=[CastFloat32()])
        return dataclasses.replace(self.create_base_config(assets_dirs,model_config),repack_transforms=repack,data_transforms=data,model_transforms=model_transforms,action_sequence_keys=("actions",))
def make_config():
    model=pi0_config.Pi0Config(pi05=True,action_dim=32,action_horizon=16,paligemma_variant="gemma_2b_lora",action_expert_variant="gemma_300m_lora")
    # Four RTX PRO 6000 cards use a global batch of 128.  Keeping the product
    # batch_size * num_train_steps at 3.84M examples preserves the traversal
    # budget of the original 32 x 120k reference run while reducing optimizer
    # steps to 30k.  The decay schedule follows optimizer steps, not examples.
    return _config.TrainConfig(name="pi05_mars_control_lora",exp_name="all600_4gpu_dp_b128",model=model,data=MarsControlDataConfig(repo_id="mars_control",assets=_config.AssetsConfig(asset_id="mars_control")),weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=250,peak_lr=5e-5,decay_steps=30000,decay_lr=5e-6),optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),freeze_filter=model.get_freeze_filter(),ema_decay=None,assets_base_dir="/workspace/runs/pi05_mars/assets",checkpoint_base_dir="/workspace/runs/pi05_mars/checkpoints",batch_size=128,num_train_steps=30000,save_interval=1000,keep_period=10000,num_workers=8,fsdp_devices=1,wandb_enabled=False,policy_metadata={"benchmark":"MARS-Control","protocol":"shared_weights_decentralized_local_rgb_qpos_to_local_action8","episodes":600,"global_batch_size":128,"sample_budget":3840000,"parallelism":"four-way data parallel"})
