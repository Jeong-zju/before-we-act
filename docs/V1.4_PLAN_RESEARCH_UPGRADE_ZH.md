# FE-PC-WAM 研究级升级方案 V1.4

> 状态：最终研究设计建议
> 日期：2026-07-13
> 范围：模型参数、模型架构、信号流、时序因果性、自回归 rollout、DiT/Flow Matching、分阶段训练、数据与实验协议

## 1. 总体结论

FE-PC-WAM V2 应定位为：

> 基于局部信念、匹配动作干预、块因果 rollout 和承诺一致选择性通信的研究级世界模型。

首期不应直接将模型整体替换为 DiT，也不应将全部预测目标改成 Flow Matching。当前模型仍处于 toy setting 的主要原因不是参数量不足，而是数据分布单一、动作条件信号过弱、部分训练目标语义错配，以及尚缺少研究级闭环对照实验。

合理的升级顺序为：

```text
正确的信号流
→ 匹配的数据和训练目标
→ 因果 rollout
→ 闭环与通信验证
→ 最后考虑规模和生成式建模
```

### 1.1 去中心化执行硬性契约

以下约束的优先级高于本文后续所有架构、训练和实验建议。任何 V2 实现只要违反其中一项，就不能称为 FE-PC-WAM 的去中心化执行版本。

1. **相同权重、独立实例**：机器人 0 和机器人 1 分别在本地进程或本地设备上运行模型实例
   `Mθ(0)` 和 `Mθ(1)`。两者加载相同 checkpoint，模型参数和 artifact hash 必须一致；“共享权重”不表示依赖同一个中央推理服务或共享内存中的模型实例。
2. **运行状态完全独立**：两台机器人不得共享 observation history、belief、KV cache、message cache、RNG、cooldown、pending proposal、committed plan 或其他 controller state。参数共享不允许退化为 activation、cache 或隐状态共享。
3. **在线输入白名单**：机器人 `i` 的模型只能读取本机 deployable observation/history、本机历史动作、task context、公开的 agent/sequence/step 元数据，以及已经通过允许协议实际收到的 compact plan message。
4. **在线输入黑名单**：模型推理不得读取 teammate observation、proprioception、pose、private slots、真实当前/未来动作、未收到的真实 plan、全局状态、simulator truth、联合 observation tensor 或未来真实 outcome。
5. **Teammate 条件来源受限**：未通信时，teammate plan/action 只能由本机 posterior
   `q_i(z_j)` 的假设解码得到；通信后只能使用实际收到且通过有效期检查的 `(code, residual)`。真实 teammate plan 只能作为训练标签、离线诊断或 oracle upper bound。
6. **本地世界模型、本地动作输出**：本机 WAM 只预测本机视角下的 future belief/outcome；本机 planner 只输出本机动作。双方动作只允许在 simulator/robot transport 的物理执行边界拼接，拼接结果不得反馈为任一模型的联合输入。
7. **Privileged 数据 target-only**：joint actions、global state、simulator snapshot、teammate truth 和 branch outcome 只能用于数据生成、训练 target、loss、audit 或 oracle evaluation。`snapshot/decision id` 只能用于样本分组，不能成为模型 feature；导出的 runtime artifact 和 forward signature 不得依赖任何 privileged 字段。
8. **通信仲裁去中心化**：同时 request 的仲裁只能使用双方可知的 agent ID、episode/sequence/step 和 request bits。两台机器人必须能够独立计算出相同 winner；禁止由读取双方 belief、VPI、候选计划或私有状态的中央仲裁器决策。
9. **单进程仿真不构成部署依赖**：测试中允许两个 planner 引用同一个无状态、`eval()` 模型对象以节省显存，但这只是一种仿真优化。必须另有测试证明，将其替换为两个从同一 checkpoint 独立加载的模型实例后，给定相同本地输入时结果等价。
10. **信息防火墙必须可执行审计**：固定本机输入和已收到消息时，任意修改 teammate private state、global truth 或未送达消息，不得改变本机模型输出。删除所有 privileged 数据后，完整 runtime 必须仍可执行。

对机器人 `i`，合法的本地 rollout 形式为：

\[
\hat Y_i^{k,m}
=
F_\theta\left(
B_i,
A_i^k,
\operatorname{Decode}(z_{j}^{m})
\right),
\]

其中 `B_i`、`A_i^k` 和 `z_j^m` 全部来自机器人 `i` 的本地计算或已接收消息。每台机器人独立执行相同计算；不存在接收两个机器人 observation 并输出 joint action 的中央世界模型。

## 2. 当前实现诊断

当前默认模型总参数量约为 1.12 亿：

| 模块 | 参数量 |
|---|---:|
| Plan tokenizer | 约 0.4M |
| Belief encoder | 约 1.2M |
| WAM | 约 58M |
| Intention model | 约 53M |

因此参数量本身并不算特别小。当前限制主要来自：

1. WAM 一次并行输出全部未来，没有真正的预测—反馈递推；
2. WAM 主要接收压缩计划 token，动作干预关系过弱；
3. `wam_robust` 将错误 teammate plan 与真实 plan 产生的 outcome 配对，会鼓励 WAM 忽略计划条件；
4. belief 输入虽然只包含历史，但辅助 target 是没有当前动作条件的 `t+1` 状态；
5. 当前正式数据全部来自单一 scripted policy，成功率为 100%；
6. counterfactual branch 只有动作和标量结果，agent 1 还存在 ego 顺序反转风险；
7. 双方同时 request 时，可能各自发送 provisional plan 后又同时修改执行计划；
8. 参数过度集中在 intention model，而不是 action-conditioned dynamics；
9. 当前 WAM/intention 尚无对应的研究级完整训练与闭环验收产物。

## 3. V2 总体信号流

```text
ego-local history o≤t
        ↓
Causal Belief Encoder
        ↓
self / teammate / object / task belief tokens
        ├────────→ Own-plan Proposal qφ(zi | bi)
        └────────→ Teammate Posterior qψ(zj | bi, message)

zi, zj
        ↓ 本机 plan decoder（两机权重相同、实例独立）
本机候选动作 ai,1:H
+ 本地假设或已接收计划解码出的 teammate 动作 aj,1:H
        ↓
本机 Block Transition World Model
        ↓
未来局部 belief、progress、reward quantiles、contact、force、constraint
        ↓
G(zi, zj)
        ↓
对 teammate posterior 外部边缘化
        ↓
VPI、通信决策、计划选择
        ↓
只执行第一个真实动作
        ↓
用最新真实观测重新编码和规划
```

最关键的变化是：世界模型必须接收双方解码后的逐时间步动作，而不是只接收两个计划摘要 token。
这里的“双方动作”是本机候选动作与本机持有的 teammate-plan hypothesis，不是从中央状态读取的双方真实动作。以上完整信号流由每台机器人使用相同权重、独立本地状态分别执行一次。

## 4. 自回归 rollout

### 4.1 最终选择

引入块间自回归，不引入逐动作自回归。

当前策略已经采用 receding-horizon control：内部评估完整候选计划，但只执行第一个动作，并在下一个真实环境步重新规划。逐动作生成会增加串行延迟和 exposure error，而候选计划的完整动作序列在 rollout 前已经已知，不需要让 WAM 再生成一次动作。

### 4.2 块转移模型

令块长度 `L=4`、预测 horizon `H=16`：

\[
B_{r+1},Y_r
=
F_\theta(B_r,A^i_r,A^j_r,e_r),
\qquad r=0,1,2,3.
\]

- 每块内部四步并行预测；
- 块间自回归；
- transition cell 在四块间共享参数；
- 训练时逐渐将上一块真实 belief 替换为预测 belief；
- 最终 self-rollout 比例初始建议为 50%；
- target belief 由冻结或 EMA belief encoder 在真实历史上生成；
- 未来真实状态只能作为 target，不能进入模型输入。

部署过程保持为：

```text
内部想象 16 步
→ 执行 1 步
→ 获取真实观测
→ 覆盖预测状态
→ 再规划
```

NOVA 的异步 Foresight Reasoning 暂不实现。只有实测 WAM 的 p95 推理耗时超过控制周期的 25%，才考虑增加预测—执行异步流水线。

## 5. 时序因果性

不同模块需要分别判断：

- **Plan tokenizer**：可以双向读取完整计划动作段，因为该计划在发送时已经形成；
- **Belief encoder**：只能读取 `o≤t`，并监督当前 `t` 状态；
- **Dynamics**：读取 `B_t` 和计划动作 `a_t:t+H` 预测未来，其中 teammate 动作只能来自本地 posterior hypothesis 或实际收到的 plan message；
- **Intention**：只能读取本地 belief、历史消息和消息元数据，不能读取真实 teammate plan；
- **Communication**：request 决策必须发生在真实 reply 到达之前。

V2 可以称为：

- temporally causal；
- action-conditioned；
- trained with paired simulator interventions。

但不能宣称已经实现通用的因果发现或因果识别。从同一 simulator snapshot 强制执行不同联合动作，支持的是动作干预评估：

\[
p(y\mid b,\operatorname{do}(a_i),\operatorname{do}(a_j)).
\]

部分可观测隐藏状态仍然存在。

## 6. V2-Base 模型配置

| 模块 | 建议配置 |
|---|---|
| Belief | `d=256`，6 层，4 role tokens |
| Plan tokenizer | 64 codes，16 维 residual |
| Proposal | `d=256`，4 层 |
| Intention | `d=256`，4 层 |
| Context encoder | `d=512`，4 层 |
| Shared block transition | `d=512`，6 层，8 heads，FFN 2048 |
| Horizon/block | 16/4 |
| Candidates/hypotheses | `K=8, M=4` |
| 总参数量 | 约 55–70M |

新配置虽然小于当前约 112M，但有效 dynamics 容量更大，intention 不再占据近一半参数。

建议规模梯度：

- Small：约 30–35M；
- Base：约 55–70M；
- Large：约 130–145M。

只有 Small→Base 在相同数据下表现出稳定收益，并且数据 scaling curve 尚未饱和，才训练 Large。参数量本身不作为研究贡献。

## 7. 计划通信与承诺一致性

最终协议不增加额外 ACK，而是限制每个环境步最多服务一个 request：

1. 两个 agent 独立执行 `prepare()`；
2. 分别产生 provisional plan 和 request bit；
3. 若只有一方 request，正常返回对方 provisional plan；
4. 若双方同时 request，使用公开 step parity 和 agent ID 做内容不可见的确定性仲裁；
5. responder 锁定并执行它回复的计划；
6. requester 收到 reply 后可以修改自己的计划；
7. 下一真实环境步重新决策。

必须满足以下协议不变量：

```text
对每个 delivered reply：
reply.code/residual == sender executed plan.code/residual
```

该方案不需要中央控制器读取任何私有内容，并能避免双方同时修改计划导致的 stale reply。
仲裁逻辑必须由两台机器人根据相同的公开 sequence/step 和 request bits 各自在本地计算；单进程 router 只能作为 transport 仿真，不得读取或聚合双方 belief、VPI 或候选计划。

### 7.1 消息预算

按照当前协议字段估算：

| 消息表示 | Round-trip bits |
|---|---:|
| 当前 64D residual | 约 590 |
| 16D residual | 约 206 |
| Code-only | 约 78 |

正式实验应比较 code-only、code+8D、code+16D 和现有 code+64D，而不是只报告通信次数。

## 8. 自由能与不确定性

候选相关风险建议定义为：

\[
G_{km}
=
-Q_{0.5}^{km}
+\lambda_{\mathrm{tail}}(Q_{0.5}^{km}-Q_{0.1}^{km})
+\lambda_cP(C\mid k,m)
+\lambda_e\operatorname{Var}_{ensemble}(Q_{0.5}^{km})
+\lambda_u\lVert a_i^k\rVert^2.
\]

然后计算：

\[
G_{\mathrm{no}}
=
\min_k\sum_m q_mG_{km},
\]

\[
G_{\mathrm{reveal}}
=
\sum_mq_m\min_kG_{km},
\]

\[
VPI=G_{\mathrm{no}}-G_{\mathrm{reveal}}.
\]

teammate posterior entropy 不能作为对所有候选相同的标量直接加进 `G`，因为它会在候选比较中抵消。它应通过 `q_m` 和 VPI 发挥作用。

## 9. 训练 DAG

### S0：正确性修复

- branch action 转换成 ego-first；
- belief auxiliary target 从无动作 `t+1` 改为当前 `t`；
- 停用 `wam_robust`；
- 增加同时 request 仲裁；
- 真正隔离 train/validation/test；
- 增加 best checkpoint 和 early stopping；
- 增加 sent-plan/executed-plan 一致性测试。

### S1A/S1B：并行训练表示

- 训练 action-only plan tokenizer；
- 训练 current-state causal belief encoder；
- 执行 codebook/residual rate–distortion 实验；
- privileged 信息只作为 role-bound target。
- 导出的 encoder/runtime forward signature 不包含任何 privileged 字段。

### S2：世界模型

先训练同参数量的 direct-parallel baseline，再训练 block model。目标包括：

- matched joint actions；
- future belief；
- progress/contact/force；
- reward quantiles；
- success/constraint；
- paired branch ranking；
- rollout consistency。

动作预测只保留为诊断项，不再作为主要 target。
这里的 matched joint actions 是训练期 action-intervention 条件：每个 ego 样本必须先转换成 ego-first，再作为本机 WAM 的动作条件。它们不能成为部署期直接读取的双方真实动作流。

### S3A/S3B：Proposal 与 intention

- 状态条件化 own-plan top-K proposal；
- teammate plan posterior；
- behavior cloning；
- branch-oracle coverage/ranking；
- candidate diversity；
- posterior calibration。

### S4：闭环数据迭代

重新采集：

- 失败与恢复；
- 高 VPI 状态；
- safety near-miss；
- proposal 产生的新计划；
- matched simulator branches。

不得再通过破坏 plan label 伪造鲁棒训练样本。

### S5：校准与冻结测试

- return quantile coverage；
- constraint Brier/ECE；
- branch regret；
- predicted VPI 与 simulator-realized communication value 的校准；
- 冻结通信价格后只运行一次 test。

### S6：可选生成式扩展

仅在 Base 模型通过全部因果、闭环和通信门槛后，评估 latent Flow Matching 和 DiT-style denoiser。

## 10. 数据建设

当前 2400/400/400 scripted、100% success 的数据只能作为 D0 诊断集。建议采用数据 scaling tiers，而不是把某个 episode 数当成研究级门槛：

- D0：当前数据，用于实现正确性检查；
- D1：约 8k mixed-policy episodes；
- D2：约 16k episodes，用于数据 scaling；
- 若 D1→D2 的验证收益低于预注册阈值，可以停止扩充。

数据必须覆盖：

- scripted success；
- noisy control；
- recovery；
- exploratory/on-policy；
- failure 与 constraint-near-miss；
- layout/dynamics/sensor OOD。

branch 数据需要从当前“六个固定计划和标量结果”升级为：

```text
branch_group_id
snapshot/decision id
ego_id
forced joint action sequence
future ego-local observations
future belief targets
reward/progress/contact/force
success/constraint/terminal
valid mask
```

其中 `snapshot/decision id` 只用于将同一干预组的样本配对，不得输入 belief、proposal、intention、WAM 或 free-energy 模型。

计划对的选择应同时包含固定 anchor 和随机或自适应 proposal，并在给定 snapshot 后独立于 branch outcome。

如果论文要宣称跨任务泛化，至少应增加一个不同协作机制，例如 asymmetric load/force-limit；否则论文声明必须限定为 private-information cooperative carrying。

## 11. DiT 与 Flow Matching

首期主模型不使用 DiT 或 Flow Matching。

DiT 的主要优势来自连续高维 latent、noise/time conditioning、多模态生成和大规模并行训练。当前任务是低维局部 belief 和少量风险变量，普通 transformer transition 更高效，也更容易校准。

只有当 Base 模型出现明确多模态欠拟合，并且 quantile heads 与三模型 ensemble 仍不足时，才增加 conditional latent flow head。Flow 只建模连续 future-belief residual；success、constraint、contact 等离散 target 继续使用分类损失。

Flow/DiT 的准入条件：

- CRPS/NLL 改善至少 5%；
- branch regret 改善至少 5%；
- 闭环性能不下降；
- p95 延迟不超过 Base 的 1.5 倍。

否则不采用。

## 12. 关键实验和验收条件

### 12.1 核心对照

- direct-parallel vs block-causal；
- summary-plan conditioning vs per-step action conditioning；
- global code support vs state-conditioned proposal；
- observational/mismatched training vs matched simulator interventions；
- no-comm / always-reply / random / periodic / selective；
- code-only / 8D / 16D / 64D residual；
- Small / Base / Large；
- ID 与 layout/dynamics/sensor OOD。

### 12.2 验收门槛

- selective 相对 no-comm 的成功率差，其 95% CI 下界大于 0；
- selective 相对 always-reply 非劣不超过 5 个百分点；
- selective 通信量不超过 always-reply 的 50%；
- decisive-private 事件请求率显著高于 redundant 事件；
- block model 的 branch regret 至少相对下降 10%，闭环成功率不下降超过 1pp；
- proposal top-K 的 branch-oracle coverage 至少达到 90%；
- 模型输入和消息继续满足 ego-local 信息防火墙；
- delivered plan 与发送方实际执行计划完全一致；
- 两个独立加载但 checkpoint hash 相同的本地模型实例，与单进程共享无状态权重对象的输出逐元素一致；
- 删除 privileged 字段后 runtime 仍可完成端到端决策；修改未送达的 teammate private/global truth 不改变本机输出；
- 每个本地 planner 只产生 ego action，joint action 只在物理环境边界形成。

每个核心实验至少使用三个独立训练 seed、相同的配对环境 seed 和 paired bootstrap confidence interval。

## 13. 论文贡献与声明边界

首篇成果建议围绕：

1. 匹配动作干预如何改善局部世界模型的计划敏感性；
2. 块因果 rollout 如何改善长时反事实排序；
3. 校准后的 counterfactual VPI 如何改善通信—协作性能前沿；
4. 承诺一致协议如何保证所传计划与实际执行一致。

暂时不应宣称：

- 通用于任意数量智能体；
- video foundation world model；
- 已完成真实机器人部署；
- DiT 或 Flow Matching 优于普通 transformer；
- 已实现一般意义上的因果识别；
- 已获得形式化隐私保证。

参数量、DiT 和 Flow Matching 本身不应包装成研究贡献。最终结论必须来自匹配实验、闭环性能、通信效率、校准指标和 OOD 结果。

## 14. 推荐实施顺序

```text
PR0  冻结 V1 baseline，并修复协议/数据 P0 问题
PR1  Research-v2 数据协议与 matched branches
PR2  V2 tokenizer、causal belief、state-conditioned proposal
PR3  Direct baseline 与 block transition dynamics
PR4  Intention posterior、统一风险和 VPI
PR5  承诺一致通信和 on-policy 数据迭代
PR6  完整闭环、OOD、校准和 scaling 实验
PR7  仅在 Go/No-Go 通过后尝试 latent flow/DiT
```

V1 应保留为可复现基线；V2 使用独立 schema、checkpoint contract 和输出目录，不覆盖现有结果。
