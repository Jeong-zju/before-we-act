# FE-PC WAM

FE-PC WAM 是一个只使用双机器人 proprioception 的 Joint World-Action Model 项目。当前
cooperative-stop 任务使用 22 维集中式状态和 8 维连续动作；最终策略通过 stateful Flow
Matching 生成动作块，并与 recurrent world model 联合训练。

当前正式结果已经通过：关闭 fallback 的 Joint WAM direct 与 action prior 在 standard 和
challenge 各 500 个未见 seeds 上均为 100%，`policy_acceptable=true`。现有任务已饱和，
因此该结果不用于声称 joint training 带来了额外控制收益。完整判断见
[Benchmark 与 Baseline 计划](docs/FE_PC_WAM_BENCHMARK_AND_BASELINE_PLAN_V1.0_ZH.md)。

## 目录

```text
configs/wam/                  # 数据、world model、action prior、Joint WAM 配置
data/                         # 轨迹 schema 与 HDF5/LeRobot exporter
envs/                         # cooperative-stop 环境、runner、视频与标注
models/wam/                   # recurrent world model、action prior、stateful action flow
policies/                     # 数据采集、action prior 与 Joint WAM runtime policy
train/                        # 数据加载、训练目标与自包含 checkpoint I/O
eval/                         # open-loop、uncertainty、closed-loop 与正式验收
scripts/                      # 数据采集、训练和评测入口
tests/                        # 数据流程、模型、checkpoint 与最终验收测试
datasets/cooperative_stop_wam_proprio/
checkpoints/joint_wam/
outputs/joint_wam/
```

历史阶段方案已移入 `docs/archive`，不再作为现行入口说明。

## 环境

项目只支持 Python 3.11，依赖由 `uv` 管理：

```bash
uv sync
uv run pytest -q
```

已有 Python 3.11 环境也可以运行：

```bash
python -m pytest -q
```

## 数据采集

正式 proprioceptive 数据集使用 `wam.proprio/1.0` schema，包含 commanded/executed action、
行为与扰动配置、任务标签和 episode/seed 隔离的 train/validation/test 划分，不包含图像或
policy 可见的 privileged state。

完整采集与审计：

```bash
python scripts/collect_wam_proprio_dataset.py \
  --out-dir datasets/cooperative_stop_wam_proprio \
  --episodes 10000 \
  --seed 0 \
  --mixture-seed 20260714

python scripts/audit_dataset.py \
  --data-dir datasets/cooperative_stop_wam_proprio/hdf5 \
  --report outputs/dataset_audit.json
```

需要同时导出 RGB、HDF5 或 LeRobot 数据时使用通用入口：

```bash
python scripts/collect_modular_dataset.py --help
```

采集目录必须为空，不能把不同 schema 或上一次残留 episode 混入同一数据集。

## 训练

已整理的资产布局如下：

```text
checkpoints/joint_wam/
├── world_model.safetensors
├── action_flow.safetensors
├── normalization.npz
├── config.yaml / schema.json
├── dataset_manifest.json / metrics.json / provenance.json
└── initialization/
    ├── world_model/          # 五成员 uncertainty ensemble，训练初始化与离线诊断
    └── action_prior/         # 行为克隆 baseline 与冻结 anchor 来源
```

最终 checkpoint 是自包含的；部署加载不依赖 `initialization/` 或 action-flow warm-up 中间产物。
`initialization/` 仅用于从头复现训练和运行 baseline。

在初始化资产已存在时，一条命令完成 action-flow warm-up、joint coupling、离线验收、保存和
严格重载：

```bash
python scripts/train_joint_wam.py \
  --config configs/wam/joint_wam.yaml \
  --device cuda
```

训练中间 action-flow checkpoint 成功后会自动清理。诊断性截断训练必须显式指定独立的
`--checkpoint-dir`，不会覆盖正式 checkpoint。

如需从数据开始重建初始化资产，依次运行：

```bash
python scripts/train_baselines.py \
  --data-dir datasets/cooperative_stop_wam_proprio/hdf5 \
  --output-dir outputs/baselines
python scripts/train_world_model.py --config configs/wam/world_model.yaml --device cuda
python scripts/train_world_model_ensemble.py \
  --config configs/wam/world_model_ensemble.yaml \
  --device cuda
python scripts/train_action_prior.py \
  --config configs/wam/action_prior.yaml \
  --device cuda
```

## 验证

正式验证固定 standard/challenge 各 500 个 held-out seeds，并同时运行 Joint WAM direct、
action prior、stationary、scripted oracle 与 report-only fallback deployment：

```bash
python scripts/evaluate_joint_wam.py \
  --config configs/wam/joint_wam.yaml \
  --device cpu
```

任何 episode 数、seed、policy、checkpoint、视频或输出路径覆盖都会被标记为 diagnostic；此时
必须提供独立 `--output-dir`。正式最小证据保存在：

```text
outputs/joint_wam/closed_loop_metrics.json
outputs/joint_wam/report.md
outputs/joint_wam/video_manifest.json
outputs/joint_wam/world_model/
```

原始 episode JSONL 与 MP4 保留在本地但不纳入 Git；聚合 JSON/Markdown 和视频 manifest
纳入版本控制。

## 约束

- runtime policy 只能读取 proprioception 与过去已执行动作。
- direct 验收必须关闭 fallback；fallback 结果单独报告。
- generated-action world target 必须来自冻结 world model 在同一动作上的预测，不能使用不同
  demonstration action 的未来状态冒充 ground truth。
- 正式 checkpoint 使用 `safetensors`，所有组件严格加载并由 artifact SHA-256 绑定。
- 数据、初始化资产、训练 seeds 与正式评测 seeds 必须保持可审计且互不污染。
