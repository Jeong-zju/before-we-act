# WAM Modular Runtime

面向 VLA / World Action Model 实验的模块化仿真与数据基础设施。模型、仿真环境和
数据集保持单向依赖：

```text
models/   纯张量模型，不读取环境或数据集
envs/     MuJoCo 几何任务、runner、RGB/MP4
data/     字段契约、HDF5/LeRobot 流式导出
scripts/  组合环境与导出器的应用入口
```

## 当前任务：双机器人协同刹停

两台平面移动机器人同时闭合夹爪，通过两个 XML grip site 的几何中点共同搬运一根
横向长条物体。两台机器人先沿世界坐标 `+Y` 方向共同运动；环境随后按 seed 随机选择：

- 一台 braking robot；
- `2.0～5.0 s` 内的一个 braking start time。

事件触发后，braking robot 的底盘目标速度被环境强制置零，并受加速度上限约束逐渐
停止；responding robot 仍由策略控制。只有当 responding robot 也从有效巡航状态逐渐
减速，并且两台机器人稳定停止 8 个控制步时，任务才成功。预先让两台机器人保持静止
不会成功。

环境只保留 `standard` 场景。没有窄道、目标区、私有门、遮挡、false belief、隐藏阻塞
通道或虚拟碰撞力代理。

环境仍是几何任务环境：底盘位姿由加速度受限的运动学控制器直接更新，双夹爪闭合时
物体按夹持点中点更新；MuJoCo 负责 XML 几何、接触查询、广义执行器 effort 和 RGB
渲染，不调用 `mj_step` 积分电机动力学。XML 是出生点、机器人/物体尺寸、夹持点和
任务边界的唯一持久化几何入口。

## 运行

机器人局部观测默认包含机载 RGB；无显示器服务器使用 EGL：

```bash
MUJOCO_GL=egl python -m envs.run --scenario standard --episodes 3
```

实时运行并打开 viewer：

```bash
python -m envs.run --scenario standard --realtime --viewer
```

Viewer 会在随机选中的机器人上方显示 `BRAKE ROBOT | starts/since T`，另一台显示
`RESPONDER`；标签使用 MuJoCo `user_scn`，不会写入环境相机图像。

流式写固定相机视频：

```bash
MUJOCO_GL=egl python -m envs.run \
  --realtime --camera fixed --video outputs/cooperative_stop.mp4
```

通过 `envs.run --video` 生成的 rollout 视频默认带有 braking robot、预定刹车时刻、
倒计时/已触发时间以及 responding robot 状态。数据集脚本不安装该 annotator；HDF5、
LeRobot 及其 `--stream-video` 输出始终消费原始机载/场景 RGB，不含这些文本。

## 观测契约

```text
robot_0 / robot_1
├── state          float32[11]
├── base_pose      float32[3] = x, y, yaw
├── base_velocity  float32[3] = vx, vy, wz
├── gripper        float32[2] = command, closed
├── base_effort    float32[3] = Fx, Fy, Tz
└── image          uint8[H,W,3]，固定在机器人 body 上的 RGB 相机

proprioception     float32[22] = 两台 robot state 拼接
privileged_state
├── object_pose / object_velocity
├── task_bounds / object_half_size
├── task[10]
├── braking_event[10]
├── contact[5]
└── state[34]
```

`braking_event` 包含随机刹车机器人、预定/实际触发步、响应机器人、响应延迟、减速步数和
稳定停止计数，只用于 centralized critic、WAM、评测或数据标签，不应直接交给去中心化
策略。`SimulationRunner` 默认会在调用 policy 前移除整个 `privileged_state`；集中式
基线必须显式设置 `expose_privileged_state_to_policy=True` 才能访问。`base_effort` 来自
`qfrc_actuator`，是几何控制器广义执行器力/力矩，不是真实
动力学积分后的关节扭矩。

## 基础失败条件

- 已建立几何抓取后任一夹爪松开；
- 两台机器人距离超过上限；
- 机器人或物体越出 XML 任务边界；
- 随机刹车事件后未在响应时限内完成协同停止；
- episode 总超时。

## 数据集采集

```bash
MUJOCO_GL=egl python scripts/collect_modular_dataset.py \
  --out-dir datasets/cooperative_stop \
  --format hdf5 --profile wam --episodes 100
```

采集脚本默认使用 Rich 显示总体 episode 进度、当前 step、任务阶段、成功/失败数以及
预计剩余时间。可变文字列保持稳定宽度，bar 自动占用终端的剩余宽度。显示默认
限制为 4 Hz，并每 5 step 刷新一次 step，避免终端频繁重排；
可通过 `--progress-refresh-hz` 和 `--progress-step-interval` 调整。自动化日志或 CI 中可添加
`--no-progress` 关闭动态显示。

同一次 rollout 可同时导出 HDF5 和 LeRobot。默认 profile 为 `vla`、`wam`、
`robocasa`、`rmbench`；WAM profile 额外保存 `response_progress`、
`coordination_error` 和 `braking_agent`。

## WAM Phase 0：纯本体感知接口与基线

`wam_proprio` 是不含图像和特权状态的 HDF5 契约。它使用
`wam.proprio/1.0` schema，并分别保存 policy 发出的 `commanded_action` 与经过环境覆盖、
限加速度执行后的 `executed_action`，同时记录行为、扰动、环境和随机化配置：

```bash
python scripts/collect_wam_proprio_dataset.py \
  --out-dir datasets/cooperative_stop_wam_proprio_v1 \
  --episodes 10000 \
  --seed 0 \
  --mixture-seed 20260714
```

采集目录必须是新的空目录，避免把旧 schema 或上一次残留 episode 混入数据集。Phase 0
入口默认使用 `phase0_mixed_v1`，每 100 个 episode 精确执行第 11.2 节比例：

| `behavior_id` | 比例 |
|---|---:|
| `scripted_oracle_v1` | 30% |
| `oracle_ou_noise_v1` | 25% |
| `delayed_response_v1` | 15% |
| `response_rate_v1` | 10% |
| `smooth_random_v1` | 10% |
| `counterfactual_stop_v1` | 5% |
| `induced_failure_v1` | 5% |

采集完成后必须先运行 Gate A 审计；脚本任一检查失败都会返回非零状态：

```bash
python scripts/audit_phase0_gate_a.py \
  --data-dir datasets/cooperative_stop_wam_proprio_v1/hdf5 \
  --report outputs/wam_phase0_gate_a_v1.json \
  --expected-episodes 10000 \
  --mixture-tolerance 0
```

序列 loader 以 episode 为边界构造 `states[32,22]`、`past_actions[31,8]` 和未来
监督，并通过 `valid_mask`、`forecast_mask` 标识 padding。数据集按 episode/seed 分组为
80/10/10，禁止随机拆 transition。

仅在 Gate A 报告中 `"passed": true` 后训练线性动力学、单步 MLP 和 action prior。
正式训练必须关闭 legacy 兼容：

```bash
python scripts/train_phase0_baselines.py \
  --data-dir datasets/cooperative_stop_wam_proprio_v1/hdf5 \
  --output-dir outputs/wam_phase0_v2_mixed \
  --no-allow-legacy-wam
```

Phase 0 baseline 默认在 episode split 完成后，将
`state_t/commanded_action_t/next_state_t/reward_t/done_t/success_t/failure_t` 一次性载入 RAM，并在数据统计、
三个模型、所有 epoch 和评测之间复用。当前 1,097,241 条 transition 的核心缓存约为
234 MiB；完整序列 loader 仍保留给后续多步 WAM。内存受限环境可添加
`--no-preload-data` 回退到原 HDF5 序列读取路径。

采集和训练默认显示 Rich 进度条。采集展示 episode、step、成功/失败数、任务阶段和 ETA；
训练为每个阶段保留独立行，显示 `stage=当前/总数`、数据统计、各模型优化 loss 以及
train/validation/test 评估进度。优化阶段会在单行进度条下方实时绘制 5 行、按终端可用
宽度分箱降采样的 batch-loss Braille 点图；点之间不连线，每个字符提供 `2×4` 点阵
分辨率。横轴始终覆盖 `step 1 → 当前 step`，并随训练持续将完整历史重新分箱到终端
宽度；纵轴使用 1%/99% 稳健范围和边距，避免少量 loss 峰值把主体变化压成直线。
阶段完成后 spinner 变为 `✓`，
进度条、最终 loss 和点图固定保留，下一阶段在新行显示。预加载、统计和评估阶段仍只占
一行。各弹性列和进度 bar 会随终端宽度分配空间，并使用 4 Hz 刷新上限。
可用 `--progress-refresh-hz 2` 进一步降低刷新频率。CI、重定向日志或仍无法正确处理 ANSI 的
终端可显式添加 `--no-progress`。最小进度条依赖可通过
`pip install -r requirements-wam.txt` 安装。

线性和 MLP dynamics baseline 现在同时输出 `done_logit`、`success_logit` 和
`failure_logit`。三个稀有标签的 positive class weight 只由 train split 计算；
每个标签的决策阈值只在 validation split 上按最大 F1 选择，F1 并列时优先更高
recall，然后冻结用于 train/test。评估同时报告 AUROC、AUPRC、多数类 accuracy、
prevalence AUPRC baseline、Brier score、10-bin ECE 和阈值后的 confusion matrix/F1。
test split 不参与阈值选择。

训练输出包含 `baseline_metrics.json`、`dataset_manifest.json`、`normalization.npz`、
三个 `*.safetensors` checkpoint 及对应模型配置，以及
`outcome_label_stats.json`、`linear_outcome_calibration.json` 和
`mlp_outcome_calibration.json`。默认允许读取旧 `wam` layout 作为
基线对照；对正式数据可加 `--no-allow-legacy-wam` 强制使用 `wam.proprio/1.0`。
新 checkpoint 格式为 `wam.phase0/2`，因输出维度由 24 变为 26，不应将旧
`wam.phase0/1` dynamics checkpoint 复制到新输出目录混用。
对应的可复现参数见 `configs/wam/phase0_baselines_v2.yaml`。旧 checkpoint 格式归档在
`configs/wam/phase0_baselines_v1_legacy.yaml`，不得用于 Phase 1。

Phase 0/Gate A、Phase 1/Gate B 与 Phase 2/Gate C 已于 2026-07-15 全量通过，允许进入
Phase 3。正式证据位于 `outputs/wam_phase0_gate_a_v1.json`、
`outputs/wam_phase0_v2_mixed`、`outputs/wam_phase1_open_loop_v1` 和
`outputs/wam_phase2_uncertainty_v1`。关键验收数字见
[技术方案 V1.0 第 21～23 节](docs/PROPRIOCEPTIVE_WAM_TECHNICAL_PLAN_V1.0_ZH.md#21-phase-0--gate-a-验收记录)。

Phase 1 的唯一任务配置入口为 `configs/wam/phase1_rwm_ar_v1.yaml`。已通过 Gate B 的
RWM-AR 训练、评估和损失模块冻结为 `rwm_ar_*`；Phase 2 的 `rwm_u_*` 模块在共享
`models/wam/` 组件上扩展，不覆盖 Phase 1 的可复现实验入口。

## WAM Phase 1：RWM-AR

Phase 1 已实现单模型 RWM-AR 工程链路：状态标准化与 yaw `sin/cos` 特征、两层历史
GRU belief、1→4→8→16 步 outer autoregression、显式 mean MSE + Gaussian NLL state
head、symlog reward head、
done/success/failure 和辅助 heads、episode-safe RAM 预载、safetensors checkpoint、
1/5/10/20/40 步 open-loop 指标及 SVG rollout。该入口只负责 Phase 1 单模型实验，不混入
后续 ensemble 或 MPPI 逻辑。

正式训练前先跑 256 个片段的 overfit 证据：

```bash
python scripts/train_phase1_rwm_ar.py \
  --config configs/wam/phase1_rwm_ar_v1.yaml \
  --data-dir datasets/cooperative_stop_wam_proprio_v1/hdf5 \
  --checkpoint-dir checkpoints/wam_cooperative_stop_v1_overfit \
  --overfit-samples 256 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0 \
  --curriculum-epochs 50 50 50 100 \
  --no-use-amp \
  --device cuda
```

配置会仅在 overfit 模式追加 50 epochs、学习率 `3e-4` 的 H=1 refinement；overfit
容量测试固定用 FP32，正式全量训练不会追加该阶段且仍可使用 AMP。

训练完成后先独立审计 overfit checkpoint；该命令不依赖正式训练 checkpoint：

```bash
python scripts/evaluate_phase1_rwm_ar.py \
  --config configs/wam/phase1_rwm_ar_v1.yaml \
  --overfit-checkpoint-dir checkpoints/wam_cooperative_stop_v1_overfit \
  --overfit-only \
  --output-dir outputs/wam_phase1_overfit_v1 \
  --device cuda
```

overfit 同时要求 256 个训练片段的 1-step continuous-state NRMSE 不高于 `0.02`、
gripper-closed RMSE 不高于 `0.05`，且分布诊断全部 finite。负 Gaussian NLL 本身不再
作为拟合成功证据。

然后使用全量 train split 正式训练：

```bash
python scripts/train_phase1_rwm_ar.py \
  --config configs/wam/phase1_rwm_ar_v1.yaml \
  --data-dir datasets/cooperative_stop_wam_proprio_v1/hdf5 \
  --checkpoint-dir checkpoints/wam_cooperative_stop_v1 \
  --device cuda
```

评测命令会在 validation 校准 outcome 阈值，在 test 上同时递归运行常速度与 Phase 0 MLP，
并结合 overfit checkpoint 输出 Gate B 结论：

```bash
python scripts/evaluate_phase1_rwm_ar.py \
  --config configs/wam/phase1_rwm_ar_v1.yaml \
  --checkpoint-dir checkpoints/wam_cooperative_stop_v1 \
  --overfit-checkpoint-dir checkpoints/wam_cooperative_stop_v1_overfit \
  --phase0-output-dir outputs/wam_phase0_v2_mixed \
  --output-dir outputs/wam_phase1_open_loop_v1 \
  --device cuda
```

正式结果为 `gate_b.passed=true`：test H=1 continuous NRMSE 为 `0.03140`，低于 Phase 0
MLP 的 `0.03590` 和常速度的 `0.37113`；H=16 finite rate 为 `1.0`、越界率为 `0`，
严格重载差异为 `0`。该结论不代表 Gate C、uncertainty 或闭环控制已经通过。

训练每个 curriculum horizon 和 validation 都是独立阶段。优化阶段保持 Phase 0 的终端
风格：首列 spinner、`stage=当前/总数`、单行自适应进度条，以及下方 5 行完整历史
Braille loss 点图；阶段完成后固定留在终端，新阶段从下一行开始。可用
`--progress-refresh-hz` 调整刷新率，或用 `--no-progress` 关闭动态输出。
AMP 会在支持的 CUDA 上优先使用 BF16，否则使用 FP16；缺失的 previous action 使用
训练集 action mean 作为哨兵，归一化 std floor 固定为 `1e-3`，避免常数动作维在低精度
线性层中溢出。定位设备相关数值问题时可临时添加 `--no-use-amp`，正式结果仍需与 AMP
配置分开记录。

内存预载只保存每个原始 transition 一份，再按需切 history/forecast window；全量数据不
会物化数百万个重叠序列。内存不足时训练和评测均可加 `--no-preload-data` 回退 HDF5。
checkpoint 目录包含 `model.safetensors`、`ema_model.safetensors`（Phase 1 与 model
字节一致，尚未声称使用 EMA）、`config.yaml`、`normalization.npz`、`schema.json`、
`dataset_manifest.json`、`metrics.json` 和 `provenance.json`。加载时会严格检查 schema、
维度、normalization SHA-256 和全部 state-dict keys。

`evaluate_phase1_rwm_ar.py` 只有 Gate B 四项全量通过才返回 0；未提供 overfit
checkpoint、任一
baseline 未被击败、16 步出现非有限/越界或严格重载不一致时返回 2。使用
`--max-batches` 或 `--max-episodes` 得到的部分评测永远不能把 Gate B 标记为通过。

## WAM Phase 2：RWM-U ensemble

Phase 2 在冻结的 RWM-AR member 上实现 5 个完整独立模型的 episode-bootstrap ensemble。
每个 imagined trajectory 在整个 horizon 内固定使用同一个 member；公开 risk API 同时输出
normalized epistemic/aleatoric uncertainty、failure probability 和 action OOD penalty。
训练还会使用 member 0 的同一 bootstrap episode、初始化 seed 和 schedule 训练
teacher-forcing 对照，避免把 ensemble 增益误报为 autoregressive training 增益。

正式训练会依次训练 5 个 member 和 1 个 teacher-forcing ablation，终端进度条、阶段完成
记录及 loss 点图与 Phase 1 保持一致：

```bash
python scripts/train_phase2_rwm_u.py \
  --config configs/wam/phase2_rwm_u_v1.yaml \
  --data-dir datasets/cooperative_stop_wam_proprio_v1/hdf5 \
  --phase1-checkpoint-dir checkpoints/wam_cooperative_stop_v1 \
  --checkpoint-dir checkpoints/wam_cooperative_stop_phase2_rwm_u_v1 \
  --device cuda
```

每个完整 member 训练后立即写入独立 safetensors 和部分训练状态。中断后使用完全相同的
配置、数据和 normalization 执行以下命令；run signature 不一致时会拒绝续训：

```bash
python scripts/train_phase2_rwm_u.py \
  --config configs/wam/phase2_rwm_u_v1.yaml \
  --checkpoint-dir checkpoints/wam_cooperative_stop_phase2_rwm_u_v1 \
  --resume \
  --device cuda
```

训练完成后先仅使用 validation 拟合逐维 variance scale，再在完整 test split 上评测
1/5/10/20/40 步 ensemble mean、50%/90%/95% coverage、uncertainty-error Spearman、
bounded action OOD AUROC、event-aligned 反平均刹车和 teacher-forcing 消融：

```bash
python scripts/evaluate_phase2_uncertainty.py \
  --config configs/wam/phase2_rwm_u_v1.yaml \
  --data-dir datasets/cooperative_stop_wam_proprio_v1/hdf5 \
  --checkpoint-dir checkpoints/wam_cooperative_stop_phase2_rwm_u_v1 \
  --phase1-metrics outputs/wam_phase1_open_loop_v1/open_loop_metrics.json \
  --output-dir outputs/wam_phase2_uncertainty_v1 \
  --device cuda
```

输出包含 `uncertainty_calibration.json`、`uncertainty_metrics.json` 和
`uncertainty_report.md`。只有版本化配置中的全部 Gate C 条款及完整 test split 同时通过，
评测脚本才返回 0；`--max-batches`、`--max-episodes` 或缺少 teacher-forcing checkpoint
时只能用于 smoke test，不能通过 Gate C。

正式 5-member GPU 训练和 1,000 个 test episodes 的评测已经完成，`gate_c.passed=true`：
H=20 ensemble NRMSE 为 `0.18434`，相对 Phase 0 MLP 的 `0.32066` 下降 `42.51%`；
autoregressive member 0 相对 teacher-forcing 改善 `12.42%`；H=20 uncertainty-error
Spearman 为 `0.89098`；bounded action OOD AUROC/epistemic ratio 为
`0.98853/29.25`；H=5 dominant-agent accuracy/ambiguous braking rate 为
`97.58%/2.24%`。五个 member 均独立，严格重载差异为 `0`。

## WAM Phase 3：准备状态

Phase 3 以 `checkpoints/wam_cooperative_stop_phase2_rwm_u_v1` 作为只读 world-model
输入，不覆盖 Phase 2 权重。计划使用以下版本化名称：

- 配置：`configs/wam/phase3_wam_mppi_v1.yaml`；
- policy：`policies/wam_mppi_policy.py`；
- 闭环评测：`eval/closed_loop.py`；
- 入口：`scripts/rollout_wam_policy.py`；
- 输出：`outputs/wam_phase3_closed_loop_v1`。

这些 Phase 3 文件将在实现阶段创建；准备阶段不放置不可运行的空壳模块。

## 验证

```bash
pytest -q
ruff check .
git diff --check
```

任务公式、奖励、数据字段、后续阶段和验收门槛见
[技术方案 V1.0](docs/PROPRIOCEPTIVE_WAM_TECHNICAL_PLAN_V1.0_ZH.md)。
