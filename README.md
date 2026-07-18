# FE-PC WAM

FE-PC WAM 是一个只使用双机器人 proprioception 的 Joint World-Action Model 项目。当前 cooperative-stop 任务使用 22 维集中式状态和 8 维连续动作；主方法以 recurrent world model 的 belief 为条件生成 stateful action-flow chunk，并使用冻结 action prior 作为训练监督与部署 anchor。

当前正式结果已经通过：关闭 fallback 的 prior-anchored Joint WAM direct 与 action prior 在 standard 和 challenge 各 500 个未见 seeds 上均为 100%，`policy_acceptable=true`。现有任务已饱和，因此该结果不用于声称 joint training 带来了额外控制收益。完整判断见 [Benchmark 与 Baseline 计划](docs/FE_PC_WAM_BENCHMARK_AND_BASELINE_PLAN_V1.0_ZH.md)。

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
uv sync
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
