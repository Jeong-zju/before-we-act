# P1 多机器人闭环模型技术路线 V6.0：以六任务 W10 为新基线

> 更新日期：2026-08-10
> 活动分支：`feat/model-improvements`
> 当前状态：六任务 W10 已完成训练和固定种子 Validation20；R11 及以后已有修改全部撤回，新的 R11 尚未开始
> 活动任务：Lift Barrier、Camera Alignment、Long Pipeline Delivery、Take Photo、Pass Shoe、Place Food；不包含任何 Stack Cube 任务

## 1. 当前决定

从现在起，后续模型改进只与六任务 W10 比较。原 R11、R12、R13/R13N、R14 和 R15 的代码、配置、训练入口、验收脚本及实验清单不再属于活动路线，也不能作为新候选的隐式组成部分。

本次采用普通 Git 回撤提交，不改写历史。需要追溯旧实验时只读以下归档：

- [V5.0 及以前完整历史](../archive/20260725_P1_MULTI_ROBOT_MODEL_ARCHITECTURE_ACTION_GENERATION_ROADMAP_V2.0_FULL_HISTORY_ZH.md)

归档只用于审计，里面的阶段状态、命令和分支不代表当前执行计划。

## 2. W10 六任务基线

### 2.1 模型和训练口径

W10 使用 `NoWristPAIRRoute`：每个机器人读取当前全局固定相机 RGB、与自己对应的固定相机 RGB 和自己的 qpos，输出本机器人 8 维动作块。视觉主干为冻结的 DINOv3 ViT-B/16，保留完整 `30×40` token 网格；动作 horizon 为 100。模型不读取任务 ID、机器人 ID、语言、腕部图像、深度、其他机器人的状态或未来信息。

六任务训练固定为：

- 120,000 optimizer updates；batch size 48；每个 batch 每任务恰好 8 个样本。
- 共 5,760,000 个局部动作样本，六任务采样预算完全相同。
- qpos/action 归一化统计直接从训练 HDF5 的物理量重新计算。
- Place Food 没有 agent camera；训练与推理都确定性复用同一时刻的 global RGB 作为 local RGB。这是输入接口对齐，不是额外信息。
- 训练代码集成提交：`87a6153cb9d0b7899066a82c69daeea1c70996a9`。
- 远程正式训练提交：`a335a2dfdf429058f4543a15c66a1d6c6738c77b`（远程实验分支 `bwa/w10-six-task`）。远程验证时使用的 complete-resume 和 Place Food 相机回退修复已纳入当前主分支回撤提交。

### 2.2 正式 Validation20 结果

完成时间为 2026-08-10 07:48:27 UTC。六项任务使用固定 seed 文件，每项 20 回合，共 120 回合。

| 任务 | 成功数 | 成功率 | 基线含义 |
|---|---:|---:|---|
| Lift Barrier | 20/20 | 100% | 后续候选必须保护 |
| Camera Alignment | 8/20 | 40% | 第一改进目标之一 |
| Long Pipeline Delivery | 20/20 | 100% | 后续候选必须保护 |
| Take Photo | 20/20 | 100% | 后续候选必须保护 |
| Pass Shoe | 20/20 | 100% | 后续候选必须保护 |
| Place Food | 0/20 | 0% | 最高优先级改进目标 |
| **合计** | **88/120** | **73.33%** | **后续模型的唯一活动 baseline** |

这里结构化结果中的 `status=PASSED` 只表示 120 个验证回合完整执行、产物齐全，并不表示六任务都解决。模型质量结论必须读取逐任务成功数：Place Food 尚未成功，Camera Alignment 仍不稳定。

可审计结果：

- 仓库内结构化摘要：[20260810_W10_SIX_TASK_VALIDATION20.json](../reports/20260810_W10_SIX_TASK_VALIDATION20.json)
- 远程 summary：`/workspace/bwa_runs/w10-six-task-v1/evaluation/validation/summary.json`
- 远程总日志：`/workspace/bwa_runs/w10-six-task-v1/logs/validation_supervisor.log`
- 远程逐任务日志：`/workspace/bwa_runs/w10-six-task-v1/logs/validation_<task>.log`
- 本地验证产物备份：`/home/jeong/zeno/wam/remote_backups/w10_six_task_120k/validation/`

### 2.3 Checkpoint 和运行环境

| 项目 | 值 |
|---|---|
| checkpoint | `/workspace/bwa_runs/w10-six-task-v1/train/formal/checkpoint_120000.pt` |
| checkpoint SHA256 | `e1b07b2cf7bff37428bf54a27f545632c8a1013930d96f6e646d8ca055f2f574` |
| 训练状态 | `120000/120000`，`PASSED` |
| 数据根目录 | `/workspace/datasets/robofactory_multitask` |
| DINOv3 | `/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m` |
| Hugging Face 缓存 | `/workspace/.cache/huggingface`；继续沿用 S0 缓存和鉴权约定 |
| Python | `/venv/robofactory-act/bin/python` |
| GPU | 4× NVIDIA RTX 5090；训练使用 GPU0，验证分两波并行使用 GPU0–3 |
| 训练输出 | `/workspace/bwa_runs/w10-six-task-v1/train/formal` |
| 验证输出 | `/workspace/bwa_runs/w10-six-task-v1/evaluation/validation` |
| 固定 seeds | `/workspace/bwa_runs/w10-six-task-v1/seeds/validation` |

验证结束后训练和验证进程均已自然退出，没有 OOM、NaN 或残留 W10 GPU 进程。`checkpoint_120000.pt` 的哈希由正式 summary 固化；不得用同名不同哈希文件替换它。

## 3. R11 及以后回撤范围

活动代码树已撤回以下内容：

| 旧阶段 | 已撤回内容 | 当前处理 |
|---|---|---|
| R11 | team-belief / V-JEPA2 predictor、四卡训练验收和上游组件移植 | 不继承；新 R11 从 W10 重新设计 |
| R12 | action generator、ACT/P2/R4/evolution 路线、缓存与恢复诊断 | 不继承 |
| R13 / R13N | world/action baseline、六任务 R13N 训练评测链 | 不继承；其成绩不再是 baseline |
| R14 | world-guided decision 组件和验收路线 | 不继承 |
| R15 | Stack 专项、专家数据、portfolio/role-query 尝试 | 不继承；Stack 不在活动任务集 |

回撤也覆盖相应模型注册、白名单、配置、requirements、许可证副本、测试、monitor、一键启动和停止脚本。保留的只有 W10 六任务训练/验证链、W10 之前仍通用的仓库能力，以及只读历史归档。

## 4. 新的后续改进约束

新的 R11 尚未预选架构。可以调研并移植优秀论文的开源实现，但必须从回撤后的 `feat/model-improvements` 建立独立候选，先写清来源、许可证、最小适配 diff 和可归因假设。

候选晋级必须使用与 W10 完全相同的六任务 Validation20 seed、环境、max steps 和成功判据：

1. Lift Barrier、Long Pipeline Delivery、Take Photo、Pass Shoe 均不得低于各自 `20/20`。
2. Camera Alignment 不得低于 `8/20`，Place Food 不得低于 `0/20`。
3. Camera Alignment 或 Place Food 至少一项严格提升。
4. 六任务总成功数必须严格高于 `88/120`。
5. 结果必须来自候选自身 checkpoint；禁止 fallback 到 W10 动作，禁止更换 seed、放宽 horizon 或改成功条件。

由于 20 回合样本量较小，Validation20 只用于快速筛选。通过后还要在预注册的更大样本上复核置信区间，才能宣布稳定提升。当前优化顺序为：先解决 Place Food 的 `0/20`，其次提高 Camera Alignment 的 `8/20`，同时冻结并回归保护四个 `20/20` 任务。

## 5. 可直接复现的命令

### 5.1 训练或从 checkpoint 续跑

```bash
cd /workspace/fe-pc-wam
CUDA_VISIBLE_DEVICES=0 \
W10_SIX_DATA_ROOT=/workspace/datasets/robofactory_multitask \
W10_SIX_DINO_MODEL=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m \
W10_SIX_OUTPUT=/workspace/bwa_runs/w10-six-task-v1/train/formal \
scripts/before_we_act/launch_w10_six_task.sh
```

launcher 检测到 `checkpoint_latest.pt` 时会自动续跑；如果 checkpoint 已达到 120,000 updates，则只校验并保持 `PASSED`，不会重复训练。

### 5.2 验证前安全检查

```bash
cd /workspace/fe-pc-wam
W10_SIX_RUN_ROOT=/workspace/bwa_runs/w10-six-task-v1 \
W10_SIX_SEED_ROOT=/workspace/bwa_runs/w10-six-task-v1/seeds/validation \
scripts/before_we_act/validate_w10_six_task.sh --dry-run
```

### 5.3 固定种子六任务正式验证

```bash
cd /workspace/fe-pc-wam
tmux new-session -d -s bwa-w10-six-validation \
  "cd /workspace/fe-pc-wam && \
   W10_SIX_RUN_ROOT=/workspace/bwa_runs/w10-six-task-v1 \
   W10_SIX_SEED_ROOT=/workspace/bwa_runs/w10-six-task-v1/seeds/validation \
   scripts/before_we_act/validate_w10_six_task.sh \
   >> /workspace/bwa_runs/w10-six-task-v1/logs/validation_supervisor.log 2>&1"
tmux attach -t bwa-w10-six-validation
```

已完整生成的逐任务 JSON 会被保留，重跑时只补未完成任务。查看最终结果：

```bash
jq '{status,successes,episodes,tasks}' \
  /workspace/bwa_runs/w10-six-task-v1/evaluation/validation/summary.json
```

## 6. 当前里程碑

| 里程碑 | 状态 | 完成条件 |
|---|---|---|
| W10 六任务训练 | **完成** | 120,000 updates 和 checkpoint 固化 |
| W10 Validation20 | **完成** | 120/120 回合完整，结果 88/120 |
| R11+ 旧实现回撤 | **完成** | 活动树不再包含 R11/R12/R13/R13N/R14/R15 实现 |
| 新 R11 方案设计 | **未开始** | 从 W10 的两个失败点提出可归因候选 |
| 新 R11 训练与验收 | **未开始** | 满足第 4 节全部门槛并完成大样本复核 |

当前没有训练或验证任务在后台运行。下一步不是恢复任何旧 R11+ 分支，而是以本文件冻结的 W10 checkpoint、六任务数据和 Validation20 为统一起点重新设计候选。
