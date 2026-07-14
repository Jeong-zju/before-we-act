# 纯本体感知 World-Action Model 技术方案 V1.0

> 文档版本：V1.0
>
> 更新时间：2026-07-14
>
> 状态：Phase 0 工程验收通过；Gate A 数据准入、Phase 1～5 待完成
>
> 适用仓库：`fe_pc_wam`，分支 `refactor/environment-overhaul`

## 1. 执行结论

本项目不直接部署 DreamZero、LingBot-VA、WorldVLA 等以视频生成为核心的大型
World-Action Model，而采用面向低维机器人状态的组合方案：

1. 以开源 RWM/RWM-U 的历史条件 GRU、自回归训练和不确定性建模为显式动力学骨干；
2. 引入 TD-MPC2 的 policy/value heads 与 MPPI 规划方法；
3. 使用当前任务的本体感知数据从头训练任务专用 checkpoint；
4. 在推理时通过多粒子 imagined rollout 评估候选动作，只向真实环境执行第一个动作，
   获得新观测后立即重新规划；
5. 将 `privileged_state` 限制为训练期辅助监督，禁止作为部署输入。

该组合同时满足两个目标：

- 显式生成未来 proprioception，便于验证 WAM 的 rollout 质量；
- 使用 learned world model 在真实环境中闭环选动作，而不只是离线预测下一状态。

第一版默认采用集中式观测：两台机器人的 `state[11]` 拼接为
`proprioception[22]`，统一生成 `action[8]`。如果部署要求每台机器人只能看到自身
`state[11]`，需要另行设计双局部 belief/WAM 和通信协议，不属于本版本范围。

## 2. 目标和非目标

### 2.1 目标

- 仅使用策略可见的机器人本体感知和历史动作建立 belief；
- 预测未来本体状态、奖励、终止、成功和失败风险；
- 支持不确定性感知的多步 imagined rollout；
- 通过 MPPI/MPC 在真实 `TwoRobotCooperativeStopEnv` 中闭环运行；
- 形成可复现的数据、训练、评测、checkpoint 和部署流程；
- 保持当前仓库 `models → envs → data → scripts` 的单向依赖原则；
- 为未来动力学升级和真实机器人部署预留接口。

### 2.2 非目标

- V1.0 不接收 RGB 图像或自然语言；
- 不下载和部署数十亿参数的视频扩散 WAM；
- 不将 `privileged_state`、刹车机器人编号或刹车时刻作为策略输入；
- 不在第一版实现跨具身、跨任务基础模型；
- 不将当前几何环境中的成功直接解释为真实接触物理或 sim-to-real 证据。

## 3. 当前仓库基线

### 3.1 环境和观测

当前任务为双机器人协同搬运与刹停：

- 控制周期：`control_dt = 0.05 s`，即 20 Hz；
- 单机器人状态：11 维；
- 集中式本体感知：22 维；
- 联合动作：8 维；
- 特权状态：34 维；
- 随机隐藏事件：环境在 2～5 秒之间选择一台机器人开始刹车；
- 策略必须根据可观测速度、姿态和 effort 变化识别事件并协调停止。

相关代码：

- 环境配置与任务定义：[`envs/two_robot_carry_env.py`](../envs/two_robot_carry_env.py)
- Gymnasium 适配：[`envs/wrappers.py`](../envs/wrappers.py)
- 真实环境 rollout runner：[`envs/runtime.py`](../envs/runtime.py)
- 数据字段契约：[`data/trajectory.py`](../data/trajectory.py)

`SimulationRunner` 默认在调用 policy 前删除整个 `privileged_state`。新策略必须保持
这一约束，不能通过闭包、环境引用或调试字段绕过。

### 3.2 当前模型的缺口

[`models/world_action_model.py`](../models/world_action_model.py) 中的模型是单步 MLP：

```text
state + action -> next_state + reward + done
```

当前实现没有：

- 历史状态和历史动作；
- partial observability belief；
- 随机或多模态未来；
- epistemic/aleatoric uncertainty；
- 多步自回归训练；
- action prior、terminal value 或 planner；
- WAM checkpoint 训练和加载流程；
- open-loop 与 closed-loop 评测。

### 3.3 当前数据的缺口

现有 `datasets/cooperative_stop_wam` 包含 100 个 episode、约 9,277 个 transition，
全部来自成功 scripted oracle。它可以验证字段对齐和 HDF5 流程，但不能支持可靠的
model-based control：

- 没有失败轨迹；
- 动作覆盖范围很窄；
- 没有响应延迟、过快/过慢刹车等行为差异；
- 没有针对 WAM/MPPI 访问的 OOD 状态动作；
- MPPI 容易利用数据分布外的模型误差。

## 4. 环境能力边界

当前环境适合验证完整 WAM 工程链路，但不足以单独支撑“模型学习了复杂物理规律”
的研究结论。

### 4.1 当前状态转移的主要形式

环境直接执行加速度受限的运动学更新：

```text
desired_velocity = action * velocity_limit
next_velocity = move_toward(current_velocity,
                            desired_velocity,
                            acceleration_limit * dt)
next_pose = current_pose + next_velocity * dt
```

刹车事件发生后，环境直接把 braking robot 的 `desired_velocity` 改为零。物体主要按
两个 grip site 的几何中点更新。MuJoCo 用于 XML 几何、接触查询、广义 actuator
effort 和 RGB 渲染；当前任务不通过 `mj_step` 积分完整电机与刚体动力学。

### 4.2 当前 WAM 可以验证的能力

- 历史条件是否有助于识别隐藏刹车事件；
- 自回归训练是否降低多步误差；
- 模型能否学习动作、速度和位姿之间的时序关系；
- ensemble uncertainty 能否识别 OOD 状态动作；
- learned rollout 能否支持 MPPI 规划；
- 数据、训练、checkpoint、部署和评测能否形成闭环；
- 集中式多机器人协调是否能仅靠 proprioception 完成。

### 4.3 当前 WAM 不能证明的能力

- 学会质量、惯性、摩擦、滑移或接触冲量；
- 学会物体摆动、夹爪柔顺性和力矩传递；
- 学会电机、传动、控制延迟和关节耦合；
- 能够从当前环境直接迁移到真实双机器人系统；
- 掌握可跨任务、跨具身泛化的通用物理规律。

因此，V1.0 的研究表述应限制为“纯本体感知、多步预测和模型闭环控制”。如果后续要
研究物理世界模型，需要把环境升级为真实 `mj_step` 动力学，并加入质量、摩擦、执行器
延迟、传感器噪声、抓取柔顺性和 domain randomization。

## 5. 开源路线调研与选型

| 路线 | 核心形式 | 开源情况 | 与当前任务的关系 |
|---|---|---|---|
| DreamZero | 14B 视频扩散骨干，联合生成视觉未来和动作 | 论文和代码/训练资料逐步开放 | 依赖图像、语言和大算力，不适合纯 22 维状态 |
| LingBot-VA | 自回归 video-action world model | Apache-2.0，开放 checkpoint；单卡 offload 推理约 18 GB VRAM | 适合视频 WAM，不适合当前低维状态模型 |
| WorldVLA | `text+image→action` 与 `image+action→image` 统一 token 模型 | Apache-2.0，Chameleon-7B 基底 | 输入模态和模型规模不匹配 |
| FastWAM | 训练时视频 co-training，推理时可跳过显式未来生成 | 开放代码和 LIBERO/RoboTwin checkpoint | 支持“测试时不必生成视频”的方向判断，但训练仍是视觉路线 |
| UnifoLM-WMA | 视频世界模型、action head、交互模拟和部署 | 代码开放；模型访问受条件约束，checkpoint 为 CC BY-NC-SA 4.0 | 仍依赖主视角视频，且许可证限制较强 |
| DreamerV3 | RSSM imagined rollout + actor critic | JAX 开源实现 | 可作为 generative latent baseline，但集成栈与当前 PyTorch 仓库差异大 |
| TD-MPC2 | decoder-free latent dynamics + reward/Q/policy + MPPI | MIT，支持 state/pixel，开放 324 个 checkpoint | 规划和控制基线最佳，但不显式还原未来 proprio |
| RWM/RWM-U | 历史 observation-action、GRU 双重自回归、显式未来、ensemble uncertainty | Apache-2.0，开放代码和 ANYmal 示例 checkpoint | 与当前“纯本体感知 + 显式 rollout”最匹配 |

主要一手资料：

- [RWM 论文](https://arxiv.org/abs/2501.10100)
- [RWM-U 论文](https://arxiv.org/abs/2504.16680)
- [RWM/RWM-U 官方代码](https://github.com/leggedrobotics/robotic_world_model)
- [TD-MPC2 论文](https://arxiv.org/abs/2310.16828)
- [TD-MPC2 官方代码](https://github.com/nicklashansen/tdmpc2)
- [TD-MPC2 官方 checkpoint 列表](https://www.tdmpc2.com/models)
- [DreamZero 论文](https://arxiv.org/abs/2602.15922)
- [LingBot-VA 官方代码](https://github.com/Robbyant/lingbot-va)
- [FastWAM 论文](https://arxiv.org/abs/2603.16666)
- [FastWAM 官方代码](https://github.com/yuantianyuan01/FastWAM)
- [WorldVLA 官方模型卡](https://huggingface.co/Alibaba-DAMO-Academy/WorldVLA/blob/main/README.md)
- [UnifoLM-WMA 官方代码](https://github.com/unitreerobotics/unifolm-world-model-action)

### 5.1 最终选型

主模型采用 RWM-AR/RWM-U 风格的显式概率递归世界模型；控制器采用 TD-MPC2 风格
的 policy/value heads 和 MPPI planner。两者使用宽松开源许可证，但不直接依赖其任务
checkpoint。

选择依据：

1. RWM 的输入输出语义与本项目接近，可以直接预测未来 proprio；
2. 历史 GRU 能处理刹车事件带来的 partial observability；
3. outer autoregression 让训练分布更接近实际 rollout 分布；
4. ensemble 能为离线数据之外的 MPC 候选动作提供不确定性信号；
5. TD-MPC2 已验证 policy prior、terminal value 和 MPPI 的组合可用于连续控制；
6. 小型状态模型可以在单 GPU 上迭代，不需要视频 WAM 的多 GPU 训练成本。

## 6. 总体架构

```text
真实 observation
  └── proprioception[22]
          │
          ▼
历史缓冲区: states[B,32,22] + actions[B,31,8]
          │
          ▼
StateFeatureEncoder
  ├── 角度 sin/cos
  ├── 数值标准化
  └── 状态分组编码
          │
          ▼
2-layer GRU belief encoder, hidden=256
          │
          ├────────► Action Prior Head
          ├────────► Terminal Value / Q Head
          └────────► Probabilistic Dynamics Ensemble × 5
                               │
                               ├── next proprio distribution
                               ├── reward
                               ├── done/success/failure
                               ├── response progress
                               ├── coordination error
                               └── executed action auxiliary
                                      │
                                      ▼
                         多粒子自回归 imagined rollout
                                      │
                                      ▼
                           Risk-aware MPPI scoring
                                      │
                                      ▼
                          只执行第一个 action[8]
                                      │
                                      ▼
                                真实 env.step
                                      │
                                      └── 新 proprio，重新规划
```

## 7. 模型接口

### 7.1 输入

```text
states          float32[B, T_context, 22]
past_actions    float32[B, T_context-1, 8]
candidate_actions float32[B, H, 8]
valid_mask      bool[B, T_context]
```

默认参数：

```yaml
state_dim: 22
action_dim: 8
history_horizon: 32
train_forecast_horizon: 16
planning_horizon: 20
```

32 步历史对应 1.6 秒；16 步训练 rollout 对应 0.8 秒；20 步规划对应 1 秒。

### 7.2 状态特征

原始状态按机器人分组：

```text
base_pose       [x, y, yaw]
base_velocity   [vx, vy, wz]
gripper         [command, closed]
base_effort     [Fx, Fy, Tz]
```

特征处理：

- 输入中的两个 yaw 分别展开为 `sin(yaw), cos(yaw)`；
- target 中仍保留原始 yaw，训练 yaw delta 时使用 wrap-to-pi；
- 连续量使用训练集 mean/std，并为 std 设置下限；
- `gripper.closed` 使用独立二分类 loss；
- 所有标准化统计量随 checkpoint 固化；
- 运行时遇到 schema、维度或统计量哈希不一致时直接拒绝加载。

### 7.3 输出

```text
state_delta_mean       [E,B,H,22]
state_delta_log_std    [E,B,H,22]
reward                 [E,B,H,1]
done_logit             [E,B,H,1]
success_logit          [E,B,H,1]
failure_logit          [E,B,H,1]
response_progress      [E,B,H,1]
coordination_error     [E,B,H,1]
executed_action        [E,B,H,8]
terminal_value         [E,B,1]
epistemic_uncertainty  [B,H,*]
aleatoric_uncertainty  [B,H,*]
```

其中 `E` 是 ensemble size。

## 8. 概率递归动力学

### 8.1 Belief encoder

基础配置：

```yaml
encoder_hidden_dim: 256
gru_hidden_dim: 256
gru_layers: 2
dropout: 0.0
```

GRU inner autoregression 顺序读取历史 `(state, action)`，形成当前 belief。这样模型可以
利用速度变化、effort 变化和之前的控制命令推断不可直接观测的事件状态。

### 8.2 Dynamics ensemble

第一版使用 5 个完整独立成员：

```yaml
ensemble_size: 5
bootstrap: true
predict_delta: true
min_log_std: -8.0
max_log_std: 2.0
```

每个成员使用不同随机初始化和 bootstrap episode 采样。规划时为每条 particle 固定一个
ensemble member，保持单条 imagined trajectory 的动力学一致性。

### 8.3 多模态事件处理

刹车机器人和刹车时刻在事件发生前不可由 proprio 完全确定，因此未来本身是多模态的。
V1.0 先使用 ensemble + Gaussian state distribution 表示。如果 event-aligned 评测出现
以下现象，则升级为 mixture density head：

- 两台机器人被预测为同时轻微刹车；
- 预测均值准确但任何单条 rollout 都不真实；
- NLL 与实际规划质量明显背离；
- ensemble 成员未形成合理模式分化。

升级时采用 3～5 个 mixture components，并在整条 particle rollout 中保持采样模式。

## 9. Action prior、reward 和 value

### 9.1 Action prior

共享 belief 后增加 tanh-Gaussian action prior：

```text
belief -> mean[8], log_std[8] -> tanh -> action[-1,1]
```

它有三个用途：

- 作为行为克隆基线；
- 为 MPPI 提供部分高质量候选轨迹；
- 在没有 GPU 或超时降级时直接执行。

Action prior 主要使用成功、高回报和 oracle-like 片段训练，避免随机失败数据主导动作
生成。世界模型和风险 heads 则使用全量数据训练。

### 9.2 Reward head

reward 包含较大的成功/失败终端项，不能直接与状态 MSE 等权相加。候选实现：

- 首选 symlog + Huber；
- 或采用 TD-MPC2 风格 two-hot categorical regression；
- 评测时恢复为原始 reward 标度。

### 9.3 Terminal value/Q head

规划 horizon 只有 1 秒，不能只累计窗口内 reward。使用 value/Q head估计窗口末端的
剩余 return：

```text
J = sum(gamma^t * predicted_reward_t) + gamma^H * terminal_value_H
```

Value head 使用真实轨迹的 n-step/Monte Carlo return 和 target network 训练。

## 10. 训练目标

总目标：

```text
L_total =
    λ_state      * L_state_nll
  + λ_ar         * L_multistep_autoregressive
  + λ_reward     * L_reward
  + λ_done       * L_done
  + λ_terminal   * (L_success + L_failure)
  + λ_aux        * (L_progress + L_coordination + L_executed_action)
  + λ_action     * L_action_prior
  + λ_value      * L_value
```

### 10.1 状态损失

连续状态采用 Gaussian NLL，yaw 使用 wrapped delta；gripper closed 使用 BCE。各状态组
分别统计 loss，避免位置维度掩盖速度或 effort 的失败。

### 10.2 多步自回归损失

训练不能只使用真实 `state_t` 预测 `state_{t+1}`。从第二个 forecast step 开始，将模型
自己的预测状态重新输入：

```text
pred_state[t+1] = model(pred_state[t], action[t])
```

远期步使用衰减权重 `rho^k`。初始 `forecast_horizon=4`，稳定后依次提升为 8 和 16，
减少训练早期完全发散。

### 10.3 Privileged learning

`privileged_state` 只作为辅助 target，可用于训练：

- contact/failure head；
- response progress；
- coordination error；
- event posterior 的训练期诊断。

禁止将其拼接到 state encoder 输入。导出的部署模型也不应包含 privileged input 参数。

### 10.4 类别不平衡

done、success 和 failure 只出现在少量时间步，需要：

- 以 episode 为单位平衡成功和失败；
- 提高终端附近窗口的采样概率；
- 使用 positive class weights 或 focal BCE；
- 单独报告 AUROC、AUPRC 和 calibration，而不只报告 accuracy。

## 11. 数据采集方案

### 11.1 数据规模

第一阶段目标：

```text
episodes:       10,000
mean steps:     约 90
transitions:    约 900,000
image streams:  disabled
```

这不是固定的最终规模。应通过数据规模曲线比较 1k、2k、5k 和 10k episode，确认模型
是否已经饱和。

### 11.2 行为分布

建议初始构成：

| 比例 | 行为 |
|---:|---|
| 30% | scripted oracle |
| 25% | oracle + temporally correlated Gaussian noise |
| 15% | responder 延迟开始刹车 |
| 10% | 过快或过慢的响应刹车 |
| 10% | 平滑随机动作或随机 action chunks |
| 5% | 提前停止、保持静止等反事实行为 |
| 5% | 松夹爪、越界、距离过大等失败行为 |

噪声策略应覆盖多个幅度，而不是固定一个 `sigma`。动作扰动要保持一定时间相关性，避免
产生只有白噪声、真实策略永远不会访问的轨迹。

### 11.3 新增字段

必须区分：

- `commanded_action`：policy 发送的动作；
- `executed_action`：环境覆盖和加速度限制后的实际动作。

建议增加：

```text
behavior_id
perturbation_config
failure_reason
environment_config
randomization_config
schema_version
```

世界模型输入使用 commanded candidate action；executed action 只能是预测目标或历史
诊断信息，因为未来实际动作在规划时尚未发生。

### 11.4 数据划分

按 episode 和 seed 划分：

```text
train: 80%
validation: 10%
test: 10%
```

禁止随机拆 transition，否则相邻状态会跨越训练和测试集，产生严重时序泄漏。额外保留
完整 OOD 测试集：未见的噪声幅度、响应延迟、初始位姿和动力学参数。

### 11.5 闭环数据迭代

首个 WAM+MPPI 版本完成后：

1. 在 held-out seeds 中运行；
2. 保存失败、高 uncertainty、模型与环境分歧最大的轨迹；
3. 使用安全策略或 oracle 恢复；
4. 将新数据加入 replay dataset；
5. 重新训练或增量微调；
6. 重复 2～3 轮。

这一步用于解决 planner 主动寻找模型漏洞的问题。

## 12. Risk-aware MPPI

### 12.1 初始配置

```yaml
planning_horizon: 20
num_samples: 512
num_elites: 64
iterations: 4
num_policy_trajectories: 32
particles_per_candidate: 5
temperature: 0.5
min_std: 0.05
max_std: 1.0
warm_start: true
execute_steps: 1
```

配置参考 TD-MPC2 的 512 samples、64 elites 和多轮 MPPI，但把 horizon 调整为当前任务
所需的 1 秒显式状态 rollout。实际数值必须根据 GPU latency 和 closed-loop 结果调优。

### 12.2 候选序列

候选动作由两部分组成：

- action prior 在 predicted beliefs 上递归生成的候选；
- 围绕上一时刻最优序列 warm-start mean 的 Gaussian candidates。

上一规划序列向前平移一位，最后一位使用 action prior 或零动作补齐。

### 12.3 风险评分

```text
score =
    E[discounted return]
  + terminal value
  - β_epistemic * ensemble disagreement
  - β_aleatoric * predicted variance
  - β_failure * P(failure)
  - β_action * action magnitude/change cost
```

除了期望回报，还应报告 worst-particle return 或 CVaR。隐藏刹车事件发生前，规划器应
避免只对某一个乐观未来有效的动作。

### 12.4 真实环境闭环

```text
reset env
clear history

while not done:
    read observation["proprioception"]
    update history with actual observation
    sample and evaluate candidate action sequences in WAM
    execute only first action in real environment
    append commanded action
```

必须每个 control step 重新规划，不允许把整段 imagined action chunk 无条件执行完。只有
在后续实测证明稳定时，才允许 `execute_steps > 1`。

## 13. 代码结构和依赖方向

建议新增：

```text
models/wam/
├── __init__.py
├── config.py
├── normalizer.py
├── state_features.py
├── recurrent_dynamics.py
├── ensemble.py
├── heads.py
└── rollout.py

policies/
└── wam_mppi_policy.py

train/
├── trajectory_dataset.py
├── losses.py
├── checkpointing.py
└── trainer.py

eval/
├── open_loop.py
├── uncertainty.py
└── closed_loop.py

configs/wam/
└── cooperative_stop_v1.yaml

scripts/
├── collect_wam_proprio_dataset.py
├── train_wam.py
├── evaluate_wam.py
└── rollout_wam_policy.py
```

依赖约束：

- `models/`：只依赖 PyTorch 和张量接口；不导入 `envs` 或 `data`；
- `data/`：负责 HDF5/LeRobot schema 和序列切片；
- `train/`：组合 dataset、model、optimizer 和 checkpoint；
- `policies/`：将 NumPy observation 适配为 tensor，并调用 planner；
- `envs/`：不能导入 `models`、`policies` 或 `data`；
- `scripts/`：唯一的应用组合入口。

### 13.1 API 调整

保留现有单步接口用于 baseline，并新增序列和 rollout API：

```python
WorldModelSequenceInputs(
    states,
    past_actions,
    valid_mask,
)

WorldModelRolloutInputs(
    history,
    candidate_actions,
    num_particles,
)

WorldModelRolloutOutput(
    state_distribution,
    rewards,
    termination,
    uncertainty,
    diagnostics,
)
```

`WAMMPPIActionPolicy` 实现当前 `Policy.act(observation) -> np.ndarray`，因此不需要修改
`SimulationRunner` 的主控制循环。

## 14. Checkpoint 与外部依赖

### 14.1 外部代码

需要记录和固定：

- `leggedrobotics/robotic_world_model`：Apache-2.0；
- `nicklashansen/tdmpc2`：MIT。

集成时固定 commit SHA，并在 `THIRD_PARTY.md` 记录来源、修改和许可证。优先移植所需的
小型模块，不把 Isaac Lab 或完整 TD-MPC2 环境栈变成运行时依赖。

### 14.2 外部 checkpoint

V1.0 不依赖外部权重：

- RWM 的 `pretrain_rnn_ens.pt` 约 23.9 MB，但状态和动作对应 ANYmal；
- TD-MPC2 的 checkpoint 对应 DMControl、Meta-World、ManiSkill2 或 MyoSuite；
- 两者的输入层、输出层、normalization 和任务价值语义均与本任务不兼容。

这些 checkpoint 最多用于 loader smoke test 或部分初始化消融，不应成为主训练路径。

如果进行迁移实验，必须比较：

```text
scratch
vs.
reinitialize I/O layers + load compatible hidden layers
```

迁移没有显著收益时立即删除该依赖。

### 14.3 本任务 checkpoint 结构

```text
checkpoints/wam_cooperative_stop_v1/
├── model.safetensors
├── ema_model.safetensors
├── config.yaml
├── normalization.npz
├── schema.json
├── dataset_manifest.json
├── metrics.json
└── provenance.json
```

`provenance.json` 至少包含：

- git commit；
- 数据版本和 episode IDs；
- 外部代码 commit 和许可证；
- Python/PyTorch/CUDA 版本；
- 随机 seed；
- 训练命令；
- checkpoint format version。

加载时默认使用 `safetensors`，优化器和 trainer state 单独保存，避免推理端反序列化任意
Python 对象。

## 15. 训练与部署资源

当前工作区检测结果：

- Python 3.11；
- PyTorch 已安装；
- `torch.cuda.is_available()` 为 `False`；
- `nvidia-smi` 无法连接 NVIDIA driver；
- 工作盘剩余空间足够。

在 CUDA 恢复前可以完成：

- 数据 schema 与采集策略；
- dataset loader；
- 小批量 CPU 单元测试；
- 模型 shape 和 checkpoint 测试；
- 小数据 overfit 测试。

正式训练和 512-candidate、5-particle 的 20 Hz MPPI 建议使用 GPU。部署性能目标应以
实测为准，不提前承诺固定 latency。

### 15.1 推理优化顺序

1. `torch.inference_mode()`；
2. vectorized ensemble 和 particle rollout；
3. `torch.compile`，保留 eager fallback；
4. BF16/FP16，在数值评测通过后启用；
5. 减少 `num_samples` 或 planning horizon；
6. action-prior distillation；
7. 最后再考虑 ONNX/TensorRT。

如果 planner 超时，安全降级顺序为：

```text
MPPI -> reduced-sample MPPI -> action prior -> scripted safe stop
```

## 16. 评测体系

### 16.1 基线

至少比较：

1. 常速度/解析运动学 baseline；
2. 当前单步 MLP；
3. history MLP；
4. GRU teacher-forcing；
5. RWM autoregressive；
6. RWM ensemble + risk-aware MPPI；
7. TD-MPC2 state baseline；
8. scripted oracle 和 stationary negative control。

### 16.2 Open-loop 指标

按 `1/5/10/20/40` steps 报告：

- 每个状态组的 RMSE/NRMSE；
- yaw wrapped error；
- reward MAE/NLL；
- done/success/failure AUROC 和 AUPRC；
- event-aligned velocity/deceleration error；
- rollout state constraint violation；
- ensemble calibration 和 50%/90%/95% interval coverage。

必须单独对齐刹车事件时刻评测。普通平均指标可能被大量简单巡航帧淹没。

### 16.3 Closed-loop 指标

在至少 500 个未见 seeds 上报告：

- success rate；
- episode return；
- response delay；
- coordination error；
- gradual brake steps；
- stop hold stability；
- failure reason 分布；
- planner latency P50/P95/P99；
- OOD fallback 触发率；
- model exploitation 事件数。

### 16.4 Privileged leakage 审计

- policy observation 中不得出现 `privileged_state`；
- model forward 不接受 braking agent/time 等字段；
- training auxiliary head 与 runtime input 代码路径分离；
- 测试通过打乱/置零 privileged labels，确认推理输出不受影响；
- 记录每次 policy 调用的 observation keys，并在测试中断言。

## 17. 阶段门槛

### Gate A：数据与接口

- 新数据包含成功、失败和行为多样性；
- commanded/executed action 对齐通过；
- episode split 无泄漏；
- proprio-only 采集不生成图片；
- sequence dataset 能跨 batch 正确 padding，但不跨 episode。

### Gate B：模型可训练

- 100～500 个片段可以 overfit；
- 单步误差优于常速度和当前 MLP；
- 16 步 rollout 不发生 NaN、爆炸或明显状态越界；
- checkpoint 重载后输出逐元素一致。

### Gate C：模型有研究价值

- 20 步 open-loop NRMSE 至少比当前 MLP 递归 rollout 低 20%；
- event-aligned 预测不存在明显的双机器人“平均刹车”；
- autoregressive training 明显优于同架构 teacher forcing；
- uncertainty 与实际 rollout error 正相关，并能识别 OOD 扰动。

### Gate D：闭环有效

- WAM+MPPI 在 held-out seeds 上比 action-prior/BC success rate 高至少 10 个百分点；
- 失败率不能通过提前静止或投机奖励下降；
- planner latency 满足控制频率，或存在验证过的安全降级路径；
- 不发生 privileged-state leakage。

以上门槛是项目验收条件，不是已经取得的实验结果。第一轮数据完成后可以根据基线难度
调整，但调整必须记录原因，不能为通过实验事后降低标准。

## 18. 致命实验与停止条件

### 18.1 环境过于简单

如果线性模型、解析运动学或当前小 MLP 在多步误差和闭环成功率上与 RWM 相当，则停止
扩大模型，不引入 transformer 或视频 WAM。下一步应升级环境动力学和任务复杂度。

### 18.2 不确定性无效

如果 ensemble disagreement 在 OOD 动作下不增加，或与真实误差无相关性，则不得把它
作为安全信号。需要重新检查数据覆盖、bootstrap 方法和 calibration，必要时改用 mixture
或其他显式 stochastic latent。

### 18.3 Planner 利用模型漏洞

如果 imagined return 很高而真实环境 return 很低，说明 planner 正在 model exploit。
处理顺序：

1. 保存并回放该动作序列；
2. 将真实轨迹加入数据；
3. 增加 uncertainty/OOD penalty；
4. 缩短 horizon 或限制动作变化；
5. 重新训练后复测。

在问题解决前不得仅通过提高 reward scale 掩盖。

### 18.4 无闭环收益

如果 WAM+MPPI 不优于 action prior/BC，保留 WAM 作为 evaluator 或 failure predictor，
停止把它作为主控制器。只有在多步模型误差或规划评分获得可证实改进后再恢复。

## 19. 实施阶段与交付物

### Phase 0：接口与基线

- 新增 proprio-only schema；
- 导出 executed action 和行为元数据；
- 实现 episode sequence loader；
- 完成线性、MLP 和 action-prior 基线。

交付：数据契约测试、baseline metrics、最小训练命令。

### Phase 1：RWM-AR

- 实现 state feature encoder；
- 实现 GRU inner/outer autoregression；
- 实现概率 state/reward/done heads；
- 完成多步 open-loop 评测。

交付：单模型 checkpoint、1/5/10/20/40 步指标和 rollout 可视化。

### Phase 2：RWM-U ensemble

- 独立 bootstrap ensemble；
- uncertainty 和 calibration；
- OOD 数据集与测试；
- risk scoring API。

交付：ensemble checkpoint、不确定性报告和 Gate C 结论。

### Phase 3：MPPI 闭环

- action prior/value heads；
- vectorized particle rollout；
- MPPI policy adapter；
- 接入 `SimulationRunner`；
- latency 和 closed-loop 评测。

交付：`rollout_wam_policy.py`、视频/日志、500-seed 报告和 Gate D 结论。

### Phase 4：数据闭环与部署

- 收集 planner failures；
- 进行 2～3 轮模型更新；
- actor distillation 和 fallback；
- 固化 checkpoint provenance；
- 编写部署说明。

交付：V1 task checkpoint、可复现部署包和最终评测报告。

### Phase 5：可选的物理环境升级

- 使用 `mj_step` 和真实 actuator dynamics；
- 质量、惯性、摩擦与控制延迟随机化；
- 传感器噪声和观测延迟；
- 更真实的双夹爪约束与物体运动；
- 重新验证模型规模和 stochastic architecture。

该阶段完成后，才适合讨论 contact-rich physics、sim-to-real 或通用物理建模结论。

## 20. 待确认决策

实施前需要最终确认以下产品约束：

1. 部署输入是集中式 22 维 proprio，还是每台机器人独立的 11 维 proprio；
2. 目标是 explicit imagined-state rollout、实际环境控制，还是两者都需要；
3. 是否允许训练期使用 privileged targets；
4. 可用 GPU 型号、显存和部署控制频率；
5. 后续是否需要真实机器人或完整 MuJoCo 动力学迁移；
6. checkpoint 是否存在商业使用要求。

V1.0 默认答案为：集中式 22 维、两种 rollout 都需要、允许 privileged auxiliary
supervision、部署目标 20 Hz、暂不承诺 sim-to-real、仅采用允许商业使用的代码依赖。

## 21. Phase 0 验收记录

验收日期：2026-07-14。

验收输入：`outputs/wam_phase0_v1`。该目录为本地运行产物，由 `.gitignore` 排除，
不随代码提交；可通过第 13 节命令重新生成。

### 21.1 结论

- `[判断]` **Phase 0 工程验收通过**：约定的 schema、动作语义、序列 loader、三类
  baseline、测试、checkpoint、指标和最小命令均已交付并可运行。
- `[判断]` **Gate A 数据准入不通过**：本次 baseline 使用的是 legacy `wam` 数据，
  仅能确认工程链路与单步基线，不能作为 Phase 1 正式训练数据或 WAM 研究结论。

这两个结论不矛盾：Phase 0 验收的是接口和基线工程；Gate A 验收的是下一阶段所需的
数据覆盖与语义完整性。

### 21.2 输出完整性

`[事实]` 输出目录包含：

- `linear.safetensors`、`mlp.safetensors`、`action_prior.safetensors`；
- 三份模型配置、运行配置、normalization 和 dataset manifest；
- train/validation/test baseline metrics。

三个 safetensors 均通过严格 `state_dict` 加载，无 missing/unexpected keys；使用零输入
执行 forward 时输出 shape 分别为 `[2,24]`、`[2,24]` 和 `[2,8]`，且全部为有限值。
normalization 的八组数组维度正确且不存在 NaN/Inf。

数据按 episode/seed 切分：

| Partition | Episodes | Transitions | 与其他 partition 的 episode/seed 交集 |
|---|---:|---:|---:|
| train | 80 | 7,411 | 0 |
| validation | 10 | 926 | 0 |
| test | 10 | 940 | 0 |

### 21.3 Baseline 结果

| Model | Test metric | 结果 |
|---|---|---:|
| Linear dynamics | state RMSE | 0.63433 |
| Linear dynamics | state NRMSE | 0.36232 |
| Linear dynamics | reward MAE | 1.35479 |
| MLP dynamics | state RMSE | 0.21002 |
| MLP dynamics | state NRMSE | 0.02592 |
| MLP dynamics | reward MAE | 1.32401 |
| Action prior | action RMSE | 0.04798 |

`[事实]` MLP test state NRMSE 相对线性模型下降 92.84%，state RMSE 下降 66.89%；
train/validation/test 指标接近，没有观察到明显的 episode-split 泛化断层。三类训练 loss
均下降且为有限值，首末轮降幅分别为 linear 39.81%、MLP 88.54%、action prior
98.87%。这些结果足以证明 Phase 0 baseline 训练链路有效。

### 21.4 不通过 Gate A 的证据

本次 100 个 episode、9,277 个 transition 全部来自成功 scripted oracle：

- 成功 episode：100；失败 episode：0；
- 数据 profile：legacy `wam`，不是 `wam.proprio/1.0`；
- 100 个源 HDF5 均包含图像数据；
- 100 个源 HDF5 均没有 `executed_action`；
- test set 只有 10 个 done frame；MLP `done_accuracy=0.98936` 与“永不预测 done”的
  多数类准确率完全相同，因此不能认定 done head 已学会终止事件；
- 没有失败行为、动作扰动、延迟响应或 OOD 覆盖。

因此不得用本次低 state NRMSE 推导“模型已能多步 rollout”“终止/失败预测有效”或
“可以进入 MPPI 闭环”。这也符合第 10.2 节指出的边界：低维 learned dynamics baseline
不是完整生成式 WAM，单步拟合更不是可执行世界模型。

### 21.5 进入 Phase 1 前必须完成

1. 使用 `wam.proprio/1.0` 重新采集包含成功、失败和第 11.2 节行为混合的数据；
2. 验证所有 episode 都含 commanded/executed action，且不含图像或 privileged input；
3. 重新生成三类 baseline，并对 done/success/failure 报告 AUROC、AUPRC 和多数类基线；
4. Gate A 全项通过后，才允许开始正式 RWM-AR 训练；
5. Phase 1 仍须独立通过 16 步稳定 rollout、checkpoint 重载一致性和 Gate B，不能沿用
   本节的单步结果替代。
