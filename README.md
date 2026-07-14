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
预计剩余时间。自动化日志或 CI 中可添加 `--no-progress` 关闭动态显示。

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
  --episodes 100
```

Phase 0 采集入口使用 `scripted_oracle_v1` 并将其写入 `behavior_id`。该数据适合验证契约
和训练链路；正式训练 RWM 前仍需按技术方案补充噪声、延迟响应、随机动作和失败行为。

序列 loader 以 episode 为边界构造 `states[32,22]`、`past_actions[31,8]` 和未来
监督，并通过 `valid_mask`、`forecast_mask` 标识 padding。数据集按 episode/seed 分组为
80/10/10，禁止随机拆 transition。

在仓库已有 legacy WAM 数据上直接训练线性动力学、单步 MLP 和 action prior：

```bash
python scripts/train_wam_baselines.py \
  --data-dir datasets/cooperative_stop_wam/hdf5 \
  --output-dir outputs/wam_phase0_v1
```

采集和训练默认显示 Rich 进度条。采集展示 episode、step、成功/失败数、任务阶段和 ETA；
训练展示数据统计、各模型优化 loss 以及 train/validation/test 评估进度。CI 或重定向日志时
可显式添加 `--no-progress`。最小进度条依赖可通过
`pip install -r requirements-wam-phase0.txt` 安装。

训练输出包含 `baseline_metrics.json`、`dataset_manifest.json`、`normalization.npz`、
三个 `*.safetensors` checkpoint 及对应模型配置。默认允许读取旧 `wam` layout 作为
基线对照；对正式数据可加 `--no-allow-legacy-wam` 强制使用 `wam.proprio/1.0`。

Phase 0 工程验收已通过：三个 checkpoint 可严格加载，数据 split 无 episode/seed
泄漏，MLP test state NRMSE 为 `0.02592`，线性模型为 `0.36232`。但该结果来自 100 个
全成功 legacy scripted-oracle episode；Gate A 数据准入尚未通过，不能把该结果解释为
done/failure 预测、多步 rollout 或闭环 WAM 已有效。完整证据和进入 Phase 1 前的条件见
[技术方案 V1.0 第 21 节](docs/PROPRIOCEPTIVE_WAM_TECHNICAL_PLAN_V1.0_ZH.md#21-phase-0-验收记录)。

## 验证

```bash
pytest -q
ruff check .
git diff --check
```

任务公式、奖励和数据字段详见
[模块化架构设计](docs/MODULAR_ARCHITECTURE_ZH.md)。

纯本体感知 WAM 的后续阶段和验收门槛见
[技术方案 V1.0](docs/PROPRIOCEPTIVE_WAM_TECHNICAL_PLAN_V1.0_ZH.md)。
