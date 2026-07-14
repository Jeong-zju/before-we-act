# FE-PC-WAM V1 人工验收审计

本文只回答“当前实现是否遵守已讨论的信息边界”。它不把单元测试或 smoke
训练解释成正式模型效果。

## 1. 原始疑问与结论

### 疑问 1：模型实际输入是否包含不可采集真值

结论：部署前向输入不再包含物体真值、队友位姿、队友 proprio 或全局状态。

实际 belief encoder 输入为：

```text
ego_id
local_history[L, 13 + 3J]
history_mask[L]
object_observation_history[L, 3]
object_valid_history[L]
object_confidence_history[L]
object_age_history[L]
```

其中当前移动底盘仿真 `J=0`。`local_history` 由本机 base twist、关节
`q/dq/tau`、本地 force/contact/grasp、ego-frame task goal 和上一条本机动作组成。

证据：

- `data/local_observation.py` 定义 deployable packet；
- `data/decentralized_dataset.py::INPUT_KEYS` 定义实际 model input keys；
- `/privileged` 与 `/observations/*/deployable` 在 `data/schema.py` 中隔离；
- `tests/test_data_contract.py` 修改队友 deployable stream 后，本机输入保持不变；
- `tests/test_online_parity.py` 证明在线 ring buffer 与离线历史逐元素一致。

状态：代码契约已证明；真实传感驱动的 RGB perception adapter 尚未实现。

### 疑问 2：object slot 是否仍然必要

结论：保留 `object-belief`，但它不是 object truth slot。它融合可选物体估计、
valid/confidence/age、本地力、接触和时序记忆。完全缺失时使用 `valid=0`，不回退真值。

固定四角色为：

```text
self / object-belief / teammate-belief / task-context
```

辅助监督 head 只能读取绑定的单一角色 slot；仿真标签只进入 loss。

证据：

- `models/slot_encoder.py::LocalBeliefSlotEncoder`；
- `tests/test_slot_encoder.py` 覆盖缺失/陈旧物体估计、padding 屏蔽、角色绑定与
  privileged target 防泄漏；
- belief 训练的 object target 使用 ego-frame truth，仅作为辅助标签，不再要求从
  本地输入预测不可观测的 world pose。

状态：实现与信息边界已证明；object-belief 在真实任务上的收益仍需消融实验验证。

### 疑问 3：推理时如何获得队友信息

结论：WAM 不读取队友 private slots。未通信时使用本地 intention posterior
`q_i(z_j)`；只有 VPI 超过 request/reply/delay/margin 成本时请求。回复只包含
`code + residual + envelope metadata`。

通信前：

```text
G_no     = min_z_i E_q G(b_i, z_i, z_j)
G_reveal = E_q min_z_i G(b_i, z_i, z_j)
VPI      = G_no - G_reveal
```

通信后才使用实际 reply 重做 rollout，并记录 surprise、G 变化、replan 和动作变化。

证据：

- `models/decentralized.py::EgoLocalWAM` 没有 teammate slots 参数；
- `models/communication.py::VPICommunicationTrigger` 的 request API 不接收真实回复；
- `policies/decentralized.py` 实现 support-only hypotheses、request/reply、缓存、
  cooldown 和执行前重规划；
- `tests/test_dynamics.py`、`tests/test_policy.py` 覆盖 request 因果性、
  reply 后动作修复、消息字段防火墙和共享权重/独立状态。

状态：算法与单进程仿真协议已证明；真实网络 transport、丢包和时钟同步尚未实现。

## 2. D1–D4 实现审计

- D1：两个 local planner 使用同一模型/权重，各自维护历史、消息缓存、随机状态和
  cooldown；只在 physics boundary 拼接动作。已实现。
- D2：四角色 token 保留，物体输入只允许 perception estimate。已实现。
- D3：唯一队友语义消息为选择性 plan latent；no-comm、always-reply、periodic、
  random 和 selective VPI 可作为配对基线。已实现。
- D4：数据改为严格 `o_t --a_t--> o_(t+1)`；计划 encoder action-only；候选来自
  checkpoint 内经验 `PlanCodeSupport`；旧固定 code 默认值已移除；训练增加 usage
  balance、dead-code 指标和严格健康门槛。已实现。

## 3. 验证证据

- 全量测试：当前验收回归全部通过（精确数量以本次测试输出为准）；
- 小数据五阶段训练：`plan → belief → wam → intention → wam_robust` 全部完成；
- 最终 artifact audit：通过 schema、input firewall、checkpoint tag、经验 code
  support 和 upstream lineage 检查；
- 不兼容的 checkpoint 会被 loader 拒绝，正式 validation/test 只接受当前严格契约。

## 4. 不应被视为已经完成的事项

- 尚未完成严格 per-agent contact/force 数据上的正式规模训练与验证；
- 尚未证明 selective VPI 优于 no-comm/always-reply；
- 尚未实现多视角 RGB perception 主干；
- 尚未完成真实机器人通信、时延、丢包和安全控制集成。

因此当前可以人工认可的是：**设计结论、信息防火墙和可执行训练/推理流程已经落实**。
不能认可为：**模型性能或实机可部署性已经通过实验验证**。
