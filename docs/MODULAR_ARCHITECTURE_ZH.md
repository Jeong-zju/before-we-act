# 模块化仿真与数据管线

## 1. 模块边界

```text
models/                         envs/
  纯张量计算                      环境、runner、实时节拍、RGB/MP4
  不读取数据集/仿真状态             不导入 data/models
       ▲                            │
       │                            ▼
       └── application adapters   data/
                                   字段契约、HDF5/LeRobot 导出
```

`tests/test_modular_architecture.py` 通过 AST import guard 固定依赖方向：

- `models/` 禁止导入 `data`、`envs`；
- `envs/` 禁止导入 `data`、`models`；
- `data/` 只通过 `envs.runtime.SimulationTransition` 消费轨迹；
- `scripts/` 负责组合环境、策略和导出器。

## 2. 当前任务定义

公共环境类是 `TwoRobotCooperativeStopEnv`，任务指令为：

> Carry the object together; when one robot slows to a stop, the other robot
> should gradually slow and stop.

两台机器人均为平面全向底盘，动作是：

```text
[r0_vx, r0_vy, r0_wz, r0_grip, r1_vx, r1_vy, r1_wz, r1_grip]
```

动作位于 `[-1,1]`，`vx/vy` 是世界坐标速度指令。底盘使用加速度受限的运动学更新：

```text
desired_velocity = normalized_action × velocity_limit
executed_velocity[t+1] = acceleration_limit(
    executed_velocity[t], desired_velocity
)
pose[t+1] = pose[t] + executed_velocity[t+1] × control_dt
```

它不是电机动力学积分；环境只调用 `mj_forward` 更新 MuJoCo 派生状态、接触和渲染。

### 2.1 几何抓取

当两个 grip action 都大于 `0.5` 时：

- 物体中心等于两个 XML grip site 的几何中点；
- 物体 yaw 等于两个 grip site 连线方向；
- 物体速度由相邻控制步的几何位姿差计算。

建立抓取后任一夹爪松开会触发 `grasp_lost`。当前任务仍不模拟真实夹爪接触、摩擦和
物体掉落动力学。

### 2.2 随机刹车事件

每次 reset 都通过环境 RNG 采样：

```text
braking_agent ∈ {0, 1}
brake_start_time ∈ [2.0 s, 5.0 s]
responding_agent = 1 - braking_agent
```

事件随机性即使在 `randomize=False` 时也保留；该参数只控制出生位姿扰动。相同 seed 在
两种 reset 模式下产生相同的 braking agent 和 start step。

到达触发步后，环境忽略 braking agent 的底盘动作，将其目标速度强制置零。实际速度仍
受与正常控制相同的线加速度和角加速度上限约束，因此不会瞬时跳到零。responding agent
的底盘动作不被覆盖。

### 2.3 防止伪成功

事件触发时必须满足：

- 两个夹爪已经建立几何抓取；
- 两台机器人 `+Y` 速度均不低于 `min_cruise_forward_speed`；
- 两台机器人的速度差不超过 `max_pre_brake_speed_error`。

该条件记录为 `pre_brake_motion_valid`。因此“两台机器人从 reset 起一直不动”即使最终
速度都是零，也不会满足成功条件。

responding agent 的速度相对事件触发速度至少下降 `response_speed_delta` 后，才记为
`response_started`。实际底盘受加速度上限约束，并且成功还要求至少经历
`min_gradual_brake_steps` 个减速步，所以单步将 action 写成零不会在状态层形成瞬时停止。

### 2.4 成功条件

以下条件必须同时成立：

1. 随机刹车事件已经触发；
2. `pre_brake_motion_valid = true`；
3. responding agent 已开始响应；
4. responding agent 的减速过程达到最小步数；
5. 两台机器人线速度和角速度均低于停止阈值；
6. 两台机器人保持停止 `stop_hold_steps`，默认 8 步；
7. 两个夹爪仍闭合。

### 2.5 基础失败条件

当前只保留与新任务直接相关的条件：

| 原因 | 含义 |
|---|---|
| `grasp_lost` | 已抓取后任一夹爪松开 |
| `robot_too_far` | 两台机器人距离超过安全协作上限 |
| `object_out_of_bounds` | 搬运物体越出 XML 任务边界 |
| `robot_out_of_bounds` | 任意机器人越出 XML 任务边界 |
| `response_timeout` | 刹车事件后未在响应时限内完成稳定停止 |
| `episode_timeout` | episode 达到总步数上限 |

窄道穿透、目标区、私有门、遮挡、false belief、虚拟 blocked lane、物体通道 yaw 和
private-event mismatch 均已删除。

### 2.6 奖励

公共项：

```text
- reward_time_cost
- reward_energy_scale × executed_base_action²
- reward_ungrasped_cost                    # 当前未保持双夹爪闭合
```

事件触发前：

```text
+ reward_cruise_scale × 两台机器人共同的 +Y 巡航速度
- reward_speed_match_scale × 两台机器人速度差
```

事件触发后：

```text
+ reward_response_progress_scale × responding agent 最佳减速进度增量
- reward_speed_tracking_scale × 两台机器人速度差
- reward_acceleration_tracking_scale × 两台机器人减速度差
+ reward_stop_hold                         # 两台机器人均已停止
```

成功额外 `+50`，失败额外 `-20`。减速进度只对历史最佳值的增加给奖励，避免通过反复
加速/减速重复获取同一段 shaping reward。

## 3. 标准场景和 XML 真值

场景只接受 `scenario="standard"`。传入旧场景名会立即抛出 `ValueError`。

`envs/assets/two_robot_carry.xml` 是唯一持久化几何入口，定义：

- `home` keyframe：两台机器人和物体的出生位姿；
- `task_bounds`：基础越界判定；
- robot/object geom：尺寸和接触几何；
- robot grip site：几何绑定点；
- 两个机载相机以及 fixed/topdown 场景相机。

XML 中不再包含 wall、goal、private gate 或 blocked-lane custom numeric。Python 中也不
维护这些几何的平行常量。

## 4. 观测和特权真值

```text
robot_0 / robot_1                  # 去中心化策略可用
  state[11]
    base_pose[3]                   # x, y, yaw
    base_velocity[3]               # vx, vy, wz
    gripper[2]                     # command, closed
    base_effort[3]                 # Fx, Fy, Tz
  image[H,W,3]                     # XML body-mounted RGB

proprioception[22]                 # 两台 robot state 拼接

privileged_state                   # critic / WAM / 评测 / 标签
  object_pose[3]
  object_velocity[6]
  task_bounds[4]
  object_half_size[3]
  task[10]
  braking_event[10]
  contact[5]
  state[34]
```

`braking_event` 是随机外生事件真值，包含 braking/responding agent、触发步、响应状态、
响应延迟、减速步数和稳定停止计数。它不得直接进入去中心化 policy，否则会形成标签
泄漏。`SimulationRunner` 默认在调用 policy 前删除 `privileged_state`，但 transition 和
数据导出仍保留完整 observation；只有显式设置
`RunnerConfig.expose_privileged_state_to_policy=True` 的集中式基线才能读取该字段。
机器人可以通过自身 `base_velocity`、机载视觉以及时间上下文判断队友是否减速。

`base_effort` 来自 `data.qfrc_actuator`，表示运动学控制器对应的广义执行器 effort，不应
解释为真实动力学积分后的关节扭矩。

### 4.1 人类可视化标注与训练视频隔离

标注只存在于显式的人类可视化路径：

- Passive viewer：在 `viewer.user_scn` 中为两个机器人添加 `mjGEOM_LABEL`。随机刹车
  机器人显示 `BRAKE ROBOT | starts/since T`，另一台显示 `RESPONDER`；标签位置每步
  跟随机器人底盘。
- `python -m envs.run --video ...`：`RenderRequest.annotator` 在原始 RGB 的副本上绘制
  braking agent、预定时刻、倒计时/elapsed time、双方速度和响应状态。
- `scripts/collect_modular_dataset.py`：所有 `RenderRequest` 均不设置 annotator；机载图像
  也直接来自原始 environment observation。因此 HDF5/LeRobot 数组以及
  `--stream-video` MP4 都不包含可视化文本。

`annotate_cooperative_stop_frame()` 总是先复制输入帧再绘字，不能反向修改 renderer 或
observation 中的原始数组。这个隔离避免模型从视频文字直接读取 braking-agent 标签。

## 5. Runtime 与数据字段

`SimulationTransition` 严格表示：

```text
observation[t] -> action[t] -> observation[t+1]
```

默认 profile：

| Profile | 关键字段 |
|---|---|
| `vla` / `robocasa` | 双机器人本体状态、两路机载 RGB、action、task、reward/done |
| `wam` | 当前/下一步 agent state、object、privileged state、`response_progress`、`coordination_error`、`braking_agent` |
| `rmbench` / `robotwin` | 当前/下一步 state、action、success、`response_progress` |

HDF5 逐 transition 写入 extendable dataset；LeRobot 导出调用官方 writer contract。同一
rollout 可以同时 fan-out 到多个 exporter，不重复运行仿真。

## 6. 运行和采集

```bash
MUJOCO_GL=egl python -m envs.run --scenario standard --episodes 3
```

```bash
MUJOCO_GL=egl python scripts/collect_modular_dataset.py \
  --out-dir datasets/cooperative_stop \
  --format hdf5 --format lerobot \
  --profile vla --episodes 100 --stream-video
```

WAM 数据：

```bash
MUJOCO_GL=egl python scripts/collect_modular_dataset.py \
  --out-dir datasets/cooperative_stop_wam \
  --format hdf5 --profile wam --episodes 100
```

## 7. 证伪与发布门槛

这一定义成立需要以下测试持续通过：

- 多个 seed 能覆盖两种 braking agent 和多个 start step；
- 从 reset 起保持静止不能成功；
- oracle baseline 在两种 braking agent 下都能逐渐停止并成功；
- responding agent 不响应时必须以 `response_timeout` 失败；
- Gym observation space 与实际嵌套 observation 完全匹配；
- HDF5 的 braking event 标签、图像帧数和 transition 数严格对齐；
- `pytest -q`、`ruff check .`、`git diff --check` 全部通过。

[判断] 对 VLA 训练而言，这个任务的研究信号来自时间因果响应，而不是静态目标到达。
因此评测时必须保证 policy 看不到 `privileged_state.braking_event`，并按 seed 分层报告
braking agent、trigger time、response delay 和成功率，而不能只报告总体平均成功率。
