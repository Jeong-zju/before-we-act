# FE-PC-WAM：本地 belief、选择性计划通信与分散执行

## 1. 已确认的系统边界

 的部署假设是：

- 两台机器人部署同一版本、共享权重的模型；
- 每台机器人只读取自己的本地传感、历史动作、任务上下文和已经实际收到的消息；
- 不存在持续读取两台机器人状态并输出联合动作的中央控制器；
- 不持续广播机器人状态或计划；
- 唯一的机器人间语义消息是按需 request 后返回的 compact plan latent；
- 仿真全局状态只能用于标签、teacher、离线诊断和 oracle upper bound。

 首先定位为 **decentralized state-estimate-based WAM**。多视角 RGB 会在数据接口中保留，但本阶段不实现视觉感知主干。

## 2. 部署期信息防火墙

机器人 `i` 的部署输入可以包含：

- 底盘三维速度 `(vx, vy, wz)`；
- 机器人关节角、关节速度和关节力矩；
- 本机夹爪、局部力和接触估计；
- 到达当前观测的上一条本机动作 `a^i_{t-1}`；
- 可选的物体相对状态估计，以及 `valid / confidence / age`；
- 任务目标或 task context；
- 实际收到的 plan message、时间戳和有效期；
- 历史有效性与 padding mask。

机器人 `i` 的部署输入不得包含：

- 另一机器人私有的 proprio、局部观测、位姿或 slots；
- 仿真物体真值或另一机器人真值；
- 全局 force/contact；
- oracle phase；
- 尚未收到的 plan message；
- 未来真实物体轨迹或任务结果。

多视角 RGB 可被记录，未来由独立 perception adapter 产生物体估计。本阶段缺少物体估计时必须显式使用 `valid=0`，不能回退到仿真真值。

## 3. 四个 belief tokens

本地时序 encoder 固定输出四个角色 token：

1. `self`：本机动力学、本体状态和历史动作；
2. `object-belief`：由本机物体估计、力、接触和时序记忆形成；
3. `teammate-belief`：不读取队友状态，由本机历史和物体耦合形成先验；已收到的 plan latent 作为下游 WAM 的显式条件更新该 belief 的使用结果，而不伪装成本地传感字段；
4. `task-context`：目标、角色和任务上下文。

仿真物体/队友真值可以监督辅助 probe，但不得参与上述 token 的前向构造。

## 4. 严格的 transition 时间语义

数据按 transition 保存：

```text
observation_t -- action_t --> observation_t+1, outcome_t+1
```

因此：

- observations 长度为 `T+1`；
- actions 和 transition outcomes 长度为 `T`；
- 决策时刻 `t` 的历史最后一行是 `(observation_t, action_{t-1})`；
- WAM 的第一个动作条件/目标从 `action_t` 开始；
- WAM 的第一个后果目标从 `observation_{t+1}` 开始；
- reset padding 必须伴随 mask，不能把当前帧重复多次冒充历史。

Dataset 样本索引为 `(episode, t, ego_id)`。输入 loader 只能读取该 `ego_id` 的 deployable stream；队友动作和全局状态只能进入训练 target。

## 5. Plan latent 的因果定义

Plan Tokenizer 的 encoder 只编码发送方在决策时能够知道的内容：

- 本机未来 command/action chunk；
- 可选的、由规划器预先产生的 ego-relative reference。

未来真实物体轨迹和执行结果不是计划输入，只能作为 WAM outcome target 或训练期辅助 target。

在线候选必须来自训练支持域。训练 artifact 需要保存：

- code usage count/probability；
- active code 集合；
- 每个 active code 的 residual mean/std；
- codebook usage、entropy 和 perplexity。

闭环禁止硬编码 active codes，也禁止默认从 zero residual 周围无约束采样。

动作段统一采用发送方自己的 ego/base frame。当前仿真器内部仍使用 world-axis
控制，因此 collector 会在写入  action stream 前转换坐标；原始仿真动作只保存在
`/privileged/transitions/environment_action_world`。

## 6. Ego-local WAM 与 teammate intention

每台机器人运行：

```text
local history
  -> four belief tokens b_i
  -> own plan candidates z_i^1 ... z_i^K
  -> teammate plan posterior q_i(z_j)
  -> WAM rollouts p(y | b_i, z_i, z_j)
  -> expected free energy / risk
  -> execute ego action only
```

WAM 不接收队友 private slots。训练时的真实队友 plan 可以作为 conditional target；部署时由 intention posterior 或实际收到的 reply 替代。

`q_i(z_j)` 的 hypothesis weights 只用于 WAM 外部的
`E_q[G]` 聚合，不进入条件动力学 `p(y|b_i,z_i,z_j)`。相同 belief 和计划条件下，
仅修改 posterior probability 不允许改变 WAM rollout。

Intention 的不确定性由 code posterior entropy 和 residual predictive variance 构成，并在 validation set 上校准。独立但无监督的 uncertainty head 不再作为通信依据。

## 7. Request–reply 中的自由能

在没有消息时，本机对 teammate plan posterior `q_i(z_j)` 求期望：

\[
G_{no}=\min_{z_i}\mathbb{E}_{z_j\sim q_i}[G(b_i,z_i,z_j)].
\]

若计划被揭示后可以重新选择本机计划，则理想的期望代价为：

\[
G_{reveal}=\mathbb{E}_{z_j\sim q_i}\left[\min_{z_i}G(b_i,z_i,z_j)\right].
\]

消息前的期望信息价值为：

\[
VPI=G_{no}-G_{reveal}\ge 0.
\]

请求规则为：

\[
request \iff VPI > C_{request}+C_{reply}+C_{delay}+margin.
\]

该决定只依赖本机 belief，不得读取真实 teammate plan。收到 reply 后：

1. 计算收到计划在原 posterior 下的 surprise；
2. 用收到的 `(code, residual)` 重新 rollout；
3. 比较消息前后最优计划、动作和自由能；
4. 在执行前替换错误或高风险动作；
5. 记录 `plan_surprise / action_changed / G_before / G_after / actual_bits / delay`。

当前“先使用真实消息计算 `G_comm`，再决定是否通信”的逻辑只保留为 oracle upper bound，不得用于部署策略。

## 8. 训练顺序

1. 构建并验证 transition/local-observation 数据；
2. 训练 action-only Plan Tokenizer，并导出真实 code support；
3. 训练四角色 Local Belief Slot Encoder；
4. 使用真实 teammate plan 训练 oracle-conditioned ego-local WAM；
5. 训练 Local Intention Posterior；
6. 使用 inferred/missing/delayed/corrupted plan 条件微调 WAM；
7. 校准 uncertainty、free-energy 权重和通信成本；
8. 比较 no-comm、always-reply、selective request–reply 和 centralized-oracle upper bound。

## 9. 最低验收条件

- 固定本机输入和已收到消息后，任意修改队友 private state，不改变本机模型输出；
- Dataset 历史和在线 ring buffer 在相同 replay 上逐元素一致；
- request 决定在真实 reply 内容改变但本机 posterior 不变时保持不变；
- reply 到达后允许 plan/action 改变，并能记录修复量；
- 在线 code/residual 全部来自 artifact 支持域；
- active codes 不由源码硬编码；
- selective 模式以真实 bits/delay 计费；
- 与 no-comm、always-reply、随机/周期通信和 oracle upper bound 在相同 seeds 下比较。

## 10. 已实现入口与当前边界

- 本地 packet /  HDF5 / ego-indexed dataset：`data/local_observation.py`、
  `data/schema.py`、`data/decentralized_dataset.py`；
- action-only tokenizer 与经验支持集：`models/plan_tokenizer.py`；
- 四角色 belief encoder：`models/slot_encoder.py`；
- ego-local WAM / intention / VPI：`models/decentralized.py`、
  `models/free_energy.py`、`models/communication.py`；
- 分散控制与 checkpoint-to-sensor runtime：`policies/decentralized.py`、
  `policies/runtime.py`；
- 五阶段训练与契约审计：`train/train_decentralized.py`、
  `scripts/train_fe_pc_wam_pipeline.py`、`scripts/audit_contract.py`。

本轮已经跑通小数据的五阶段 smoke pipeline；它只证明接口、反向传播、checkpoint
lineage 和 runtime 可执行。正式模型的 codebook 健康度、任务成功率、安全性、通信收益
仍必须在重新采集的训练/验证集上训练和配对评估，不能用 smoke loss 代替。
