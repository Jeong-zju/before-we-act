# FE-PC WAM

FE-PC WAM 是一个只使用双机器人 proprioception 的 Joint World-Action Model 项目。当前 cooperative-stop 任务使用 22 维集中式状态和 8 维连续动作；主方法以 recurrent world model 的 belief 为条件生成 stateful action-flow chunk，并使用冻结 action prior 作为训练监督与部署 anchor。

当前正式结果已经通过：关闭 fallback 的 prior-anchored Joint WAM direct 与 action prior 在 standard 和 challenge 各 500 个未见 seeds 上均为 100%，`policy_acceptable=true`。现有任务已饱和，因此该结果不用于声称 joint training 带来了额外控制收益。完整判断见 [Benchmark 与 Baseline 计划](docs/FE_PC_WAM_BENCHMARK_AND_BASELINE_PLAN_V1.0_ZH.md)。

从当前纯 proprioception 系统逐步扩展到同时使用机器人 state、RGB 视觉并对齐 DreamZero 一类开源 World Action Model 的实施顺序，见 [多模态扩模技术路线](docs/FE_PC_WAM_MULTIMODAL_SCALING_ROADMAP_V1.0_ZH.md)。

Joint coupling 保持 prior-anchored contract：flow 学习冻结 action prior rollout，world model 在专家动作与生成动作上提供 action-conditioned world/risk coupling；部署动作固定为 `anchor + 0.1 × (flow - anchor)`。独立 action prior 同时保留为主 baseline 和可选 fallback。

## 目录

```text
configs/wam/                  # 数据、world model、action prior、Joint WAM 配置
data/                         # 轨迹 schema 与 HDF5/LeRobot exporter
envs/                         # cooperative-stop 环境、runner、视频与标注
models/wam/                   # recurrent world model、action prior、stateful action flow
policies/                     # 数据采集、action prior 与 Joint WAM runtime policy
train/                        # 数据加载、训练目标与自包含 checkpoint I/O
eval/                         # open-loop、closed-loop 与正式验收
scripts/                      # 数据采集、训练和评测入口
tests/                        # 数据流程、模型、checkpoint 与最终验收测试
datasets/cooperative_stop_wam_proprio/
checkpoints/joint_wam/
outputs/                       # 本地可再生产物，全部由 Git 忽略
```

历史阶段方案已移入 `docs/archive`，不再作为现行入口说明。

## 数据收集、训练与验证全流程

以下命令均从仓库根目录执行，并且必须等待前一步成功结束后再进入下一步：

```bash
cd ~/zeno/wam/fe_pc_wam
```

### 1. 准备环境

项目只支持 Python 3.11，依赖由 `uv` 管理：

```bash
uv sync --frozen
uv run pytest -q
```

### 2. 收集并审计数据

正式 proprioceptive 数据集使用 `wam.proprio/1.0` schema，包含 commanded/executed action、行为与扰动配置、任务标签和 episode/seed 隔离的 train/validation/test 划分，不包含图像或 policy 可见的 privileged state。

采集 10,000 条 episode：

```bash
uv run python scripts/collect_wam_proprio_dataset.py \
  --out-dir datasets/cooperative_stop_wam_proprio \
  --episodes 10000 \
  --seed 0 \
  --mixture-seed 20260714
```

采集目录必须不存在或为空，不能把不同 schema 或上一次残留 episode 混入同一数据集。采集完成后运行完整审计：

```bash
uv run python scripts/audit_dataset.py \
  --data-dir datasets/cooperative_stop_wam_proprio/hdf5 \
  --report outputs/dataset_audit.json \
  --expected-episodes 10000
```

命令退出码必须为 0，且 `outputs/dataset_audit.json` 中的 `passed` 必须为 `true`。需要同时导出 RGB、HDF5 或 LeRobot 数据时，可查看通用采集入口：

```bash
uv run python scripts/collect_modular_dataset.py --help
```

### 2.1 转换 RoboFactory 数据集

`scripts/convert_robofactory_dataset.py` 可直接读取 RoboFactory/ManiSkill 生成的 `.h5` 和同名 `.json` sidecar。源文件保持只读，转换过程按 episode 流式读取，不会把整套 RGB 数据加载进内存。HDF5 输出采用本项目的一 episode 一文件布局；LeRobot 输出使用 Dataset v3 writer，并在结束时执行 `finalize()`。

只转换为本项目的标准 HDF5：

```bash
uv run python scripts/convert_robofactory_dataset.py \
  --input ../RoboFactory/data/h5_data/LiftBarrier-rf.h5 \
  --out-dir datasets/robofactory_lift_barrier \
  --format hdf5 \
  --fps 20 \
  --task "Lift the barrier together"
```

为 LiftBarrier M1 scratch 训练生成版本化 HDF5（DINO 之外的任务侧模型从零训练）使用独立 profile。下面的命令固定 20 Hz、只导出 global 相机，并显式声明 `action.executed` 是 `action.commanded` 的 command echo：

```bash
uv run python scripts/convert_robofactory_dataset.py \
  --input ../RoboFactory/data/h5_data/LiftBarrier-rf.h5 \
  --metadata-json ../RoboFactory/data/h5_data/LiftBarrier-rf.json \
  --out-dir datasets/robofactory_lift_barrier_m1_v1 \
  --profile m1-scratch \
  --format hdf5 \
  --camera global \
  --fps 20 \
  --task-id lift_barrier \
  --task "Lift the barrier together" \
  --executed-action-source command-echo \
  --canonical-only \
  --compression gzip
```

该 profile 的动作语义是：

- `action.commanded` 是权威动作监督，也是历史动作的来源。
- `action.executed` 是 commanded action 的逐元素精确副本，仅作为显式标注的 command echo 兼容字段；它不是独立测得的 actuator feedback。
- state 按 agent 自然顺序排列，每个 agent 内固定为 `qpos` 后接 `qvel`；action 使用相同 agent 顺序。
- current/next RGB 分别来自源数据的 `rgb[t]` 和 `rgb[t+1]`，图像与控制频率均为 20 Hz。

转换完成后，生成 seed-disjoint 的训练 manifest 和仅使用 train transition 拟合的归一化统计。该命令默认加载 LiftBarrier 的 16D 双 Panda `pd_joint_pos` action codec；HDF5 保持 raw controller action，训练 window 和 action 统计变换到 canonical `[-1,1]`：

```bash
uv run python scripts/prepare_robofactory_m1_training_artifacts.py \
  --dataset-dir datasets/robofactory_lift_barrier_m1_v1 \
  --transition-selection through-first-done-inclusive
```

如果目录中已有 codec 引入前的 `training_manifest.json`/`normalization.npz`，需要显式增加 `--overwrite` 重新生成；scratch 训练入口会拒绝 `codec.applied=false` 或 raw-domain normalization。

该命令默认使用 `split_seed=7` 和 `0.8/0.1/0.1`，对当前 150 个唯一 seed 生成 120/15/15 个 episode 的 train/validation/test 划分。它会全量校验 HDF5 contract、记录每个文件的 SHA256，并输出：

- `training_manifest.json`：相对 HDF5 路径、文件 hash、episode/seed/split、数据语义、终止截断口径和 split 汇总；
- `training_manifest.json.sha256`：训练 manifest 自身的 SHA256；
- `normalization.npz`：仅从 train split 的 selected transitions 计算；state/delta 保持物理域，action 是 codec 变换后的 canonical-domain population moments。

源数据在首次 `done=true` 后仍保存了一段 transition，因此 `--transition-selection` 是必填项。推荐的 `through-first-done-inclusive` 保留首次 terminal transition 并排除其后记录；若确实要训练全部记录，必须显式改为 `all-recorded`。codec 使用 ManiSkill Panda 关节限制做逐维仿射变换，不使用本数据集 observed min/max，且越界 raw action 会直接中止产物生成。

用协议分发器严格校验 manifest/HDF5/normalization，并实际读取三个 split 的首尾 M1 window：

```bash
uv run python scripts/smoke_m1_data_protocol.py \
  --manifest datasets/robofactory_lift_barrier_m1_v1/training_manifest.json \
  --splits train validation test \
  --state-history 32 \
  --action-chunk 8 \
  --visual-history 2 \
  --future-horizons 1 2 4 8 \
  --camera global
```

该 manifest 会分发到 `generic_multimodal_trajectory` loader：它不读取/伪造 cue、event 或 causal-pair 字段，历史动作来自 `action.commanded`，并严格只索引 manifest 中选定的 episode 前缀。原 M0 visual-cue manifest 仍分发到旧的严格 loader，保留 pair/event 门禁。快速开发时可临时增加 `--skip-hdf5-hashes`，正式预检不要跳过文件 hash。

完成上述产物更新后，scratch 专用训练入口为：

```bash
uv run python scripts/train_liftbarrier_m1_scratch.py \
  --config configs/wam_multimodal/m1_liftbarrier_scratch.yaml \
  --device auto
```

该入口只构建一次随机任务侧模型，冻结 DINOv3，不读取 legacy checkpoint/action prior，依次运行 dynamics、action-flow、multimodal fusion、future-joint 四阶段，并保存 `wam.multimodal.m1.scratch_checkpoint/1`。模型内部始终使用 canonical 16D action；`ScratchM1Policy` 在观测历史入口 encode，在控制器输出前 decode 回 raw `pd_joint_pos`。

正式 checkpoint 完成后，使用两个隔离 Python 环境进行 RoboFactory 闭环成功率评测、逐集视频渲染及结果固化，参见 [`docs/ROBOFACTORY_M1_CLOSED_LOOP_ROLLOUT_ZH.md`](docs/ROBOFACTORY_M1_CLOSED_LOOP_ROLLOUT_ZH.md)。

正式入口默认启用 Rich 终端显示：启动时列出 `1/4..4/4` 的阶段表，训练时同时显示总阶段进度和当前阶段的 step、loss、gradient norm、GPU allocated memory、耗时与 ETA。Rich 输出写到 stderr，训练结束的 JSON 仍单独写到 stdout，便于重定向保存。

LiftBarrier 配置同时启用以下吞吐策略：

- 启动时把 train split 所需的 7 个数值/RGB HDF5 字段解压到连续的 PyTorch shared-memory tensor；当前 manifest 为 7,277 个 window、约 3.54 GiB，共享给所有 worker，不产生每 worker 一份的 RAM 副本；
- 预载前同时检查 `MemAvailable` 和 `/dev/shm`，默认最多使用可用 RAM 的 50% 和 `/dev/shm` 的 90%，不足时 fail closed；
- `num_workers: auto` 使用一半逻辑 CPU、上限 12，配合 `spawn`、`persistent_workers=true`、`prefetch_factor=4`、CUDA pinned memory 和每 worker 单 Torch thread；
- 每个训练阶段只读取真实需要的字段：前两个 state/action 阶段完全不搬运 RGB，fusion 阶段不读取 future RGB，只有 future-joint 阶段读取 future RGB；
- pinned batch 使用独立 CUDA stream 提前搬运下一批，与当前批计算重叠；固定输入尺寸启用 cuDNN benchmark，FP32 参数默认允许 TF32 tensor-core matmul。

正式训练前可检查 RAM 条件：

```bash
free -h
df -h /dev/shm
```

当前配置至少需要约 3.54 GiB 空闲 `/dev/shm`，还应为 DataLoader 预取、模型和系统保留余量。正式训练不要使用 `--skip-hdf5-sha256`；只有无交互 CI 才需要 `--no-rich-progress`。

同时生成 HDF5 和 LeRobot v3；LeRobot 是可选依赖：

```bash
uv run --with "lerobot>=0.4" python scripts/convert_robofactory_dataset.py \
  --input ../RoboFactory/data/h5_data/LiftBarrier-rf.h5 \
  --out-dir datasets/robofactory_lift_barrier \
  --format hdf5 \
  --format lerobot \
  --repo-id local/robofactory-lift-barrier \
  --fps 20 \
  --task "Lift the barrier together"
```

主要字段映射如下：

| RoboFactory/ManiSkill 源字段 | 目标字段 | 规则 |
|---|---|---|
| `obs/agent/<agent>/qpos,qvel` | `observation.state` | 按自然排序的 agent 顺序拼接，每个 agent 内先 qpos 后 qvel |
| `actions/<agent>` | `action` | 按同一 agent 顺序拼接 |
| 分 agent qpos/qvel/action | `observation.agents.*` / `agents.*.action` | 默认同时保留；`--canonical-only` 可关闭 |
| `head_camera_agent0` 等 | `observation.images.agent_0` 等 | agent 相机规范为 `agent_N`，全局相机规范为 `global` |
| `rewards` | `next.reward` | 仅当源数据真实存在 reward 时输出，不伪造零 reward |
| `terminated` / `truncated` | `next.terminated` / `next.truncated` / `next.done` | `done = terminated OR truncated` |
| `success` / `fail` | `next.success` / `next.failure` | 保留原始逐帧标签 |

源 observation 有 `T+1` 帧、action 有 `T` 帧，转换后严格保存为 `observation[t], action[t], next_observation[t+1]`。相机内外参默认保存在 `observation.camera_calibration.*`；可用 `--no-images`、`--no-calibration` 做纯 proprioception 转换，或用 `--success-only` 过滤失败 episode。默认任务文本由 `env_id` 规范化得到，建议通过 `--task` 提供准确指令。

输出根目录会生成 `manifest.json`，其中记录 agent/camera 原名映射、集中状态和动作的切片范围、源 episode id、shape、标签可用性及完整字段 schema。为防止混入旧数据，选中的输出子目录和 manifest 必须事先不存在。

转换默认使用与训练入口一致的 Rich 分阶段进度条，显示源 episode、目标 episode 及当前 frame；可用 `--progress-refresh-hz` 调整刷新率，CI 或重定向日志时可用 `--no-progress` 关闭动态显示。

### 3. 训练离线动力学基线

该步骤生成 linear、MLP 和独立 action-prior 的离线对比结果：

```bash
uv run python scripts/train_baselines.py \
  --data-dir datasets/cooperative_stop_wam_proprio/hdf5 \
  --output-dir outputs/baselines \
  --device cuda
```

这些结果用于 world-model benchmark；正式闭环中的 `action_prior` 则来自下一步与 frozen world model 配套训练的 checkpoint。

### 4. 训练 recurrent world model 初始化资产

```bash
uv run python scripts/train_world_model.py \
  --config configs/wam/world_model.yaml \
  --device cuda
```

成功后应生成 `checkpoints/joint_wam/initialization/world_model/model.safetensors`、normalization、dataset manifest 和训练 metrics。

### 5. 训练 action prior baseline 与 anchor

```bash
uv run python scripts/train_action_prior.py \
  --config configs/wam/action_prior.yaml \
  --device cuda
```

该 checkpoint 同时承担两种角色：独立 action-prior 闭环 baseline，以及 prior-anchored Joint WAM 的冻结训练监督和部署 anchor。

### 6. 训练 prior-anchored Joint WAM

```bash
uv run python scripts/train_joint_wam.py \
  --config configs/wam/joint_wam.yaml \
  --device cuda
```

该入口依次完成 action-flow warm-up、两轮 on-policy distillation、joint coupling、离线验收、最终 checkpoint 保存与严格重载。action-flow warm-up 训练 10 个完整数据轮次；joint coupling 渐进解冻并固定执行 `flow_only=64 steps`、`world_heads=128 steps`、`full_joint=512 steps`，共 704 steps。

训练中间 action-flow checkpoint 成功后会自动清理。诊断性截断训练必须显式指定独立的 `--checkpoint-dir`，不能覆盖正式 checkpoint。

### 7. 运行快速闭环诊断

正式 500-seed 验收前，先用 50 seeds 检查 direct policy 和 action-prior baseline：

```bash
rm -rf outputs/joint_wam_diagnostic_prior_anchored_v1
uv run python scripts/evaluate_joint_wam.py \
  --config configs/wam/joint_wam.yaml \
  --device cpu \
  --episodes 50 \
  --policies joint_wam_direct action_prior \
  --skip-videos \
  --output-dir outputs/joint_wam_diagnostic_prior_anchored_v1
```

汇总关键指标：

```bash
jq -r '
.metrics
| to_entries[] as $suite
| $suite.value
| to_entries[]
| "\($suite.key)  \(.key)  episodes=\(.value.episodes)  success=\(.value.success_rate)  fallback=\(.value.fallback_trigger_rate)"
' outputs/joint_wam_diagnostic_prior_anchored_v1/closed_loop_metrics.json
```

两个 suite 的 `joint_wam_direct` success rate 都应达到 `0.90`，再进入正式验证。诊断报告的 `policy_acceptable=false` 不代表训练失败：覆盖 episode、policy、seed、checkpoint、视频或输出目录后，评测会按设计标记为 diagnostic。

### 8. 运行正式闭环验证

正式验证固定使用 standard/challenge 各 500 个 held-out seeds，并运行 Joint WAM direct、action prior、stationary、scripted oracle 与 report-only fallback deployment。评测器拒绝混入旧证据，因此先清空可再生的正式输出目录：

```bash
rm -rf outputs/joint_wam
uv run python scripts/evaluate_joint_wam.py \
  --config configs/wam/joint_wam.yaml \
  --device cpu
```

正式命令不能添加 `--episodes`、`--policies`、`--skip-videos`、seed、checkpoint 或 `--output-dir` 覆盖，否则结果只会被视为 diagnostic。

查看最终结论和每个 suite/policy 的闭环结果：

```bash
jq '{
  formal_protocol,
  policy_acceptable,
  joint_benefit,
  metrics: (
    .metrics
    | with_entries(
        .value |= with_entries(
          .value |= {
            episodes,
            success_rate,
            failure_rate,
            fallback_trigger_rate
          }
        )
      )
  )
}' outputs/joint_wam/closed_loop_metrics.json
```

正式验收的总门禁是 `formal_protocol=true` 且 `policy_acceptable=true`。完整报告、逐 episode 记录和视频证据位于 `outputs/joint_wam/`。

### 9. 准备 DINOv3 权重并训练 Phase M1

当前 Phase M1 canonical 视觉 teacher 已重构为冻结 DINOv3。项目默认别名 `dinov3_vitl16_lvd` 严格映射官方 Hugging Face 模型 `facebook/dinov3-vitl16-pretrain-lvd1689m`；policy 仍接收 `96×96@10Hz` raw RGB，encoder 内部按官方预处理放大到 `256×256`，原生 `1024` 维 patch token 经可训练投影进入 `512` 维 resampler，future head 监督目标为冻结 teacher 的 CLS feature。runtime 会缓存每个新 10Hz frame 的 detached encoder feature，供相邻 20Hz 控制步复用。

该官方模型是 gated model，并受 [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) 约束。先在[官方模型页](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)接受许可，再登录并执行一次显式准备：

```bash
uv sync --frozen
uv run hf auth login
uv run python scripts/prepare_dinov3_encoder.py \
  --encoder dinov3_vitl16_lvd \
  --output-dir artifacts/vision/dinov3_vitl16_lvd
```

`hf auth login` 显示已登录并不等于 gated 权限已经获批。若准备入口报告 `Access denied`，先用 `uv run hf auth whoami` 确认浏览器与 CLI 是同一账号，在模型页提交访问申请并等待批准；fine-grained token 还必须启用 `Read access to contents of all public gated repos you can access`。以下命令只检查权限与文件元数据，不下载权重：

```bash
uv run hf download facebook/dinov3-vitl16-pretrain-lvd1689m \
  config.json model.safetensors \
  --revision dd0a398fa8e84f2a37179332f6c561d20276300b \
  --dry-run
```

准备入口会下载并校验固定 revision 的 `config.json` 与 `model.safetensors`。训练、评测和 rollout 只读取已校验的本地文件，不会在运行中自动联网，也不会静默回退到随机或其他视觉权重。准备完成后启动正式 GPU 训练：

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run python scripts/train_multimodal_wam.py \
  --config configs/wam_multimodal/m1_latent_wam_dinov3.yaml \
  --device cuda:0 \
  --torch-threads 16
```

canonical 配置已针对当前单机训练路径启用 train/validation/causal-pair `4/2/2` 个 DataLoader worker、pinned memory、persistent prefetch，并将 DINOv3 encoder microbatch 设为 `16`；`--torch-threads` 只控制进程内算子线程，不能替代这些 DataLoader worker。启动阶段的 manifest/hash 校验仍是一次性的 CPU/I/O 阶段，此时 GPU 空闲属于预期；进入各训练 stage 后才应观察 GPU 利用率。

需要切换视觉 encoder 时，复制为独立配置并同时修改 encoder 别名、官方模型 ID、固定 revision、配置/权重路径及两项预期 SHA-256，再用 `--config` 指向该配置；不要覆盖 canonical DINOv3 配置。旧 `configs/wam_multimodal/m1_latent_wam.yaml` 与 `outputs/phase_m1/` 仅用于复现历史 ResNet-18 pilot。

DINOv3 版本已完成正式训练与 Gate M1 验收。`outputs/phase_m1_dinov3/phase_m1_acceptance.json` 使用 acceptance/bundle `/2` 修订统计协议并记录 `formal_protocol=true`、`passed=true`、`claim_allowed=true`：视觉价值与 state-shuffle 保留总体固定阈值和 clustered 正 CI，并要求至少 2/3 训练 seed 的配对均值同向；逐 seed CI 继续报告但不再单 seed 否决。视觉干预与 future probe 的逐 seed 严格要求保持不变，具体结果与限制见技术路线 5.7。

### 10. 运行历史 Phase M1 闭环 rollout 并录制 MP4

以下命令显式使用 `outputs/phase_m1/` 中已通过旧 Gate 的 ResNet-18 `state_vision_future` checkpoint，在 visual-required MuJoCo 环境中直接闭环执行动作，同时统计成功率并录制实际仿真相机画面。该入口是历史 pilot 的 diagnostic rollout，不会改写 canonical acceptance，也不构成 DINOv3 版本的验收。

下面的命令运行 3 个训练 seed、3 个任务、每任务 10 个新 physical seeds × 2 个 cue，共 180 个闭环 episode；每个训练 seed/任务录制前 2 个 episode，因此生成 18 段 MP4：

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run python scripts/rollout_multimodal_wam.py \
  --config configs/wam_multimodal/m1_latent_wam.yaml \
  --device cuda:0 \
  --torch-threads 16 \
  --train-seeds 101 202 303 \
  --tasks visual_event_stop visual_target_select visual_obstacle_avoid \
  --physical-seeds 10 \
  --cue-variants 0 1 \
  --video-episodes-per-task 2 \
  --video-width 640 \
  --video-height 360 \
  --output-dir outputs/phase_m1/rollouts/clean_fresh_10
```

未指定 `--physical-seed-start` 时，入口默认从 canonical M1 正式范围之后的 `710100` 开始，不会静默复用 `710000..710099`。若要录制本次运行的每个 episode，将 `--video-episodes-per-task` 设为 `physical-seeds × cue 数`；上例对应 `20`。

主要产物：

```text
outputs/phase_m1/rollouts/clean_fresh_10/
├── rollout_summary.json
├── rollout_episodes.jsonl
└── videos/train_seed_<seed>/<task>/
    ├── physical_seed_<seed>_cue_<cue>.mp4
    └── physical_seed_<seed>_cue_<cue>.json
```

查看总体、分任务、分训练 seed 和分 cue 成功率：

```bash
jq '.aggregation | {overall, by_task, by_train_seed, by_cue}' \
  outputs/phase_m1/rollouts/clean_fresh_10/rollout_summary.json
```

策略输入流保持为原正式协议的 `fixed` 96×96@10Hz；MP4 使用独立的 `rollout` 640×360@20Hz 旁路，永不进入 policy observation。每段视频写完后都会重新解码并核验 `frames == rollout steps`、尺寸、FPS、字节数和 SHA-256，且保存包含终态的 post-action frame。同名 JSON sidecar 记录 success、failure reason、action source、fallback/privileged 检查与视频证据。MP4 是有损可视证据，成功率真值仍以本次 JSONL 为准；这里录制的是实际仿真闭环，不是 M1 future latent head 生成的视频。

## 训练与产物约定

已整理的资产布局如下：

```text
checkpoints/joint_wam/
├── world_model.safetensors
├── action_flow.safetensors
├── normalization.npz
├── config.yaml / schema.json
├── dataset_manifest.json / metrics.json / provenance.json
└── initialization/
    ├── world_model/          # 单个 recurrent world model，训练初始化与冻结 teacher
    └── action_prior/         # 行为克隆 baseline 与冻结 anchor 来源
```

最终 checkpoint 是自包含的；部署加载不依赖 `initialization/` 或 action-flow warm-up 中间产物。`initialization/` 仅用于从头复现训练和运行 baseline。

direct policy 使用 `anchor_residual_scale=0.10`；flow 只生成对冻结 prior chunk 的有界修正。验收时仍关闭 fallback，以区分 prior anchor 与运行时 fallback。

所有 `outputs/` 内容均是可再生产物并由 Git 忽略，包括数据审计、离线 baseline、快速诊断、正式闭环聚合指标与视频。需要保留某次实验时，应将整个输出目录复制到仓库之外的实验归档位置。

## 约束

- runtime policy 只能读取 proprioception 与过去已执行动作。
- direct 验收必须关闭 fallback；fallback 结果单独报告。
- generated-action world target 必须来自冻结 world model 在同一动作上的预测，不能使用不同 demonstration action 的未来状态冒充 ground truth。
- 正式 checkpoint 使用 `safetensors`，所有组件严格加载并由 artifact SHA-256 绑定。
- 数据、初始化资产、训练 seeds 与正式评测 seeds 必须保持可审计且互不污染。
