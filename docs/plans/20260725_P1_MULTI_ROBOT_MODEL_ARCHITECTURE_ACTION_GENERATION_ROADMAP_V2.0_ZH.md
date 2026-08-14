# P1 多机器人闭环模型技术路线 V7.3：Measurement 收口后的 ARB-first PBT 协作模型

> 更新日期：2026-08-14
> 活动分支：`feat/model-improvements`
> 当前状态：**第 1 步 Measurement 按负责人后验的 signal-first 决策收口，模块设计和训练获准开始。** 第 2 步先建立 `TeamTemporalSample` 与公平 B0-H，第 3 步再训练继承 ARB schema、direct residual 和可靠度回退的 B-core，随后逐级进入 BP、BT、BPT。
> 证据边界：R4-C 在 72 条全新成功 episode 上相对冻结 HC 改善 `26.05%`，95% CI `[24.15%, 27.90%]`，三个 seed、六个任务全正；但 hidden-only、time-only、row-shuffle 和同阶段 shuffle 仍更强。因此可声称“整套 ARB_hat + residual 栈有稳定纸面收益”，不可声称“ARB 语义独立贡献了这些收益”。
> 优先级调整：M4 partner-change、M5 W10 失败归因和 ARB-vs-hidden residual 隔离不删除，但后置为稳定候选的机制消融，不再阻断现在的模块训练。详细结论和全部原始验收状态见[第 1 步结果文档](../reports/20260814_P1_STEP1_MEASUREMENT_CONCLUSIONS_AND_ACCEPTANCE_ZH.md)。
> 活动任务：Lift Barrier、Camera Alignment、Long Pipeline Delivery、Take Photo、Pass Shoe、Place Food；不包含任何 Stack Cube 任务

## 0. 先用一句话说明这条路线

我们要验证一件事：

> 如果机器人不仅看自己的图像和关节，还能判断“队友正在干什么、任务做到哪一步、团队还缺什么”，它能不能更好地合作，同时不丢掉 W10 已经具备的基础动作能力？

### 0.1 三十秒看懂最终结果

第 1 步已经回答了“这条接口值不值得继续工程化”：

- 完整旧 B 在收敛修复后仍为负收益，所以停止；
- oracle ARB 有动作相关信号，ARB 表示优于清洗旧 B，但 query 融合显著输给 direct residual；
- `ARB_hat + direct residual` 在一次性全新 test 上相对 HC 改善 `26.05%`，预测头可校准且 gate-off 精确回退；
- hidden-only 和时间/打乱对照仍更强，说明主要收益可以由通用 residual capacity 解释，ARB 独立语义增量尚未证明。

负责人据此采用工程优先的路线：接受整套栈的稳定纸面信号，进入具体模块设计与训练；把严格机制归因放到稳定候选产生后统一完成。

> **第 1 步完成不等于 ARB 机制已经证明。它只表示当前证据足以支持继续开模，同时把所有归因限制原样带入后续路线。**

阅读建议：

- 只想知道现在做什么：看本节和第 16 节；
- 想查看第 1 步的完整数字和验收：[打开详细结果文档](../reports/20260814_P1_STEP1_MEASUREMENT_CONCLUSIONS_AND_ACCEPTANCE_ZH.md)；
- 想实现模块：从第 2 节架构和第 5 节执行入口开始。

<details>
<summary>展开查看原计划路线和术语说明</summary>

修订后的路线按下面的顺序执行：

```text
第 0 步：准备工作（已完成）
  ↓
第 1 步：Measurement（已完成）
  ↓
第 2 步：TeamTemporalSample + B0-H
  ↓
第 3 步：ARB-B-core
  ↓
第 4/5 步：BP / BT
  ↓
第 6 步：BPT
  ↓
联合训练 + Validation/Confirmation
  ↓
稳定候选后的 M4 / M5 / residual 机制消融
```

第 2 步以后的模块仍逐级走漏斗；后置 M4/M5 不阻断开模，但在它们完成前不得提出严格 ARB 因果机制 claim。

文档里的几个常用词可以这样理解：

| 词 | 人话解释 |
|---|---|
| gate | 一道“通过/停止”的检查 |
| oracle | simulator 给出的真实答案，只能用于出题和判卷，不能给部署模型偷看 |
| probe | 为了验证某种信息有没有用而训练的小模型，不是最终模型 |
| sidecar | 不修改原 HDF5，另外保存的一份逐帧标签文件 |
| receipt | 用 commit、配置和 SHA256 证明“这次到底用了什么”的记录 |
| belief | 模型对团队当前状态的结构化判断，不是自然语言感想 |
| ARB | Action-Relevant Belief；不追求复原全部团队状态，只保留能改变当前自身动作的少量关系、事件和可靠度 |
| cross-attention | 一组信息主动读取另一组信息；谁是 Query 就是谁在问，谁提供 Key/Value 就是谁在回答 |
| coordination queries | 少量“协调员”token，负责把 B、P、T 中与下一段动作有关的信息整理后交给动作模型 |
| CoordinationAdapter | ACT decoder 里的小型旁路，让 action query 读取 coordination queries；它不是第四个 PBT 模块 |
| macro | 先分别计算六个任务，再对六个任务等权平均，避免长任务支配结果 |
| paired bootstrap | 对相同 seed 的两个模型做成对重采样，用来估计结果是否稳定 |
| 95% CI | 对结果稳定范围的估计；本文要求最保守的一端仍然大于 0 |
| 统计功效 | 现有样本是否足以看出预设大小的差异；样本不够时只能说“证据不足” |
| 分叉点 | 从同一个模拟器状态复制出几条路线，只改变队友后续行为 |
| `INCONCLUSIVE` | 证据不足，既不算通过，也不能当成路线已被证明失败 |

</details>

## 1. 我们从哪里开始

### 1.1 R11、R12 已经结束

R11 和 R12 只保留作历史审计，不再继续训练，也不能从它们的 checkpoint 续训。

| 阶段 | 最终结论 | 已经确认的事实 | 没有发生的事 |
|---|---|---|---|
| R11 | `NO_WINNER` | A/B/D 完成真实 F1 后在 Discovery gate 失败；C 因冻结 foundation revision 403 关闭 | Validation20、Confirmation50、winner merge |
| R12 | `ARCHIVED_NO_WINNER` | Measurement future-vs-persistence `+91.604%`、K=4 oracle paired-win `87.5%`；F0/F1 通过；旧 global-clip 尝试只算诊断，L0/L1 的诊断 Discovery 失败 | 合格 Discovery、Validation5/20、Confirmation50、winner merge；L2/L3 没有合格实验终态，修正版正式重跑未完成 |

R12 是用户在证据还不完整时主动结束的阶段。因此可以说“R12 没有 winner”，但不能补写成“L2/L3 已经被正式实验判败”。

历史材料：

- [R11/R12 失败技术路线完整归档](../archive/20260811_R11_R12_FAILED_TECHNICAL_ROUTES_ZH.md)
- [更早的完整路线历史](../archive/20260725_P1_MULTI_ROBOT_MODEL_ARCHITECTURE_ACTION_GENERATION_ROADMAP_V2.0_FULL_HISTORY_ZH.md)
- R11 账本：`feat/r11-four-way-integration@678e67780e6960749410ee0649ce961b10495950:docs/experiments/r11/`
- R12 账本：`feat/r12-lawam-controlled-ablation@6d45a108098fcdf5b06060d0d5860639b1513617:docs/experiments/r12/`

远端历史数据、checkpoint、日志、receipt 和 HF cache 继续只读保留。归档不代表授权删除这些内容。

### 1.2 唯一公平基线是 W10 六任务模型

W10 做的事情很简单：每个机器人读取当前 global RGB、自己的 local RGB 和自己的 qpos，然后输出本机器人的 8 维动作。它不读取任务文本、机器人 ID、队友状态、腕部图像、深度或未来信息。

W10 的固定配置是：

- `NoWristPAIRRoute`
- 冻结 DINOv3 ViT-B/16，保留完整 `30×40` token 网格
- action horizon 100
- 120,000 optimizer updates
- effective batch 48，每次 update 六个任务各取 8 个 local-agent 样本

W10 当前 Validation20：

| 任务 | 成功数 |
|---|---:|
| Lift Barrier | 20/20 |
| Camera Alignment | 8/20 |
| Long Pipeline Delivery | 20/20 |
| Take Photo | 20/20 |
| Pass Shoe | 20/20 |
| Place Food | 0/20 |
| **合计** | **88/120** |

- checkpoint：`/workspace/bwa_runs/w10-six-task-v1/train/formal/checkpoint_120000.pt`
- SHA256：`e1b07b2cf7bff37428bf54a27f545632c8a1013930d96f6e646d8ca055f2f574`

W10 在新路线中只有三个用途：

1. 复用已经验证的数据加载、训练、保存、恢复和 evaluator 脚手架；
2. 提供公平比较数字；
3. 提供 B0-H 的基础 action backbone 和训练 recipe。

不能做的事：

- 不能加载 W10 checkpoint 开始训练新候选；
- 不能继承 W10 optimizer、RNG 或 sample cursor；
- 不能把 W10 当 teacher、action prior、fallback 或评测兜底；
- 最终 checkpoint 里不能藏一个可调用的独立 W10 policy。

## 2. 第 1 步完成后，要实现什么模型

### 2.1 先纠正旧设计

旧设计把 P、T、B 当成三份并列说明书，分别从同一段图像历史产生，然后一起塞进 action decoder。这样虽然代码容易写，但模型可能只读其中一份，也可能把三份重复信息当成“现在是第几帧”的线索。它没有回答三件最重要的事：

1. 谁是其他模块的基础；
2. 谁应该读取谁；
3. 三者意见冲突时，动作模型应该相信什么。

V7.3 不再采用“三组 token 直接拼接”的正式方案。P、B、T 是同一条推理链上的三个阶段，不是三个互不相干的插件。第 1 步现已收口，本节从条件设计转为活动模块路线；具体训练仍按第 5–13 节逐级执行。

### 2.2 P、B、T 各自只负责什么

为了避免三者重复记同一件事，边界固定如下：

| 模块 | 用人话解释 | 它负责的内容 | 它不负责的内容 |
|---|---|---|---|
| B：team belief | “我现在认为现场是什么状态” | 谁可见、谁接触/抓住了什么、物体由谁持有、最近确认过的交接、遮挡和置信度 | 不直接判断任务完成百分比，不直接预测队友未来动作 |
| P：progress | “按照任务规则，现在做完了什么、还缺什么” | 已完成/未完成谓词、当前阶段、阶段内连续进度、剩余目标 | 不重新看原始图像猜角色，不复制 B 的 agent-object 关系 |
| T：teammate future | “根据当前现场和剩余任务，队友接下来可能怎么做” | 队友未来若干步的目标、角色、动作模式及每种模式的不确定性 | 不保存长期物体记忆，不重新定义任务进度 |

`per_agent_contribution` 继续作为监督和审计标签，但不再单独做一组输入 token。它用于检查 B 中的角色/物体关系和 P 中的任务完成判断是否一致。

### 2.3 PBT 到底做不做 cross-attention

做，但只做**有方向的 cross-attention**，不做 P、B、T 两两互相乱读。

完整的一次前向按下面顺序执行：

```text
合法的当前观察、最近 16 步历史、上一时刻 B
                      │
                      ▼
       B：更新“现场现在是什么状态”
                      │
                      ▼
       P 用自己的 Query 去读取 B
       得到“做完了什么、还缺什么”
                      │
                      ▼
       T 用自己的 Query 去读取 B + P
       得到“队友接下来可能怎么做”
                      │
                      ▼
  8 个 coordination queries 读取 B + P + T
       只留下与下一段自身动作有关的信息
                      │
                      ▼
       ACT action queries 读取协调结果并输出动作
```

用 Transformer 的 Query/Key/Value 写法表示：

| 顺序 | 谁在问（Query） | 读取谁（Key/Value） | 输出 |
|---|---|---|---|
| 1 | B 的 entity/memory queries | 当前观察、16 步历史、上一时刻 B | 更新后的 `B_t` |
| 2 | P 的 task-predicate queries | `B_t` 和 task graph | `P_t` |
| 3 | T 的 future-mode queries | `B_t` 和 `P_t` | `T_t` |
| 4 | 8 个 coordination queries | `B_t`、`P_t`、`T_t` | `C_t` |
| 5 | 100 个 ACT action queries | 原视觉 memory，以及单独的 `C_t` 旁路 | 100 步 action chunk |

这里有一个刻意的限制：在同一次前向中，`B_t` 不再反过来读取 P 或 T。否则会形成“B 依赖 P、P 又依赖 B”的即时循环，很难判断信息从哪里来。T 预测出的未来只监督一个单独的 `future_B` 预测头，用来检查动力学是否合理；它不会在同一时刻覆盖当前 `B_t`。下一控制步到来后，B 再用真正看到的新观察更新。

这条 `B → P → T → C → action` 的单向链是第一版正式实现。更复杂的多轮 PBT 反复交互只有在单向版本通过因果干预后，才允许作为新路线预注册，不能在训练途中临时加入。

### 2.4 每组张量长什么样

基础 action backbone 继续使用 `d_model=384` 和 100 个 action queries。社会状态使用同一宽度：

```text
B_t: [batch, n_B, 384]    现场中的 agent/object/relation/memory slots
P_t: [batch, n_P, 384]    任务谓词和剩余目标 slots
T_t: [batch, n_T, 384]    队友未来模式和时间段 slots
C_t: [batch,   8, 384]    固定 8 个协调员 slots
```

`n_B`、`n_P`、`n_T` 不能拍脑袋决定。ARB 字段已经由第 1 步冻结；token 数在第 2 步根据六个任务需要的最大 agent、object、predicate 和未来模式数量一次性冻结，并使用 padding mask 处理较小任务。冻结后不得因为 Validation 结果不好而改 token 数。8 个 coordination queries 是第一版 B-core 的候选规格，目的是迫使模型整理信息，而不是把全部 PBT 原样复制给动作模型。

agent slot 不读取显式机器人 ID。ego 由自己的 local view/qpos 确认，其余 teammate slots 共享参数；训练时一致交换 teammate slot，B/T/C 和动作影响也必须相应交换。

### 2.5 CoordinationAdapter 到底是什么

它不是一个新的“大协调模型”，只是放在现有 ACT decoder 每一层里的小旁路。

当前 ARCA decoder 每层已经做三件事：action query 先互相交流，再读取基础视觉 memory，最后通过 role adapter 补充观察信息。V7.3 保留这些原有路径，只增加第四条残差：让 action query 读取 8 个 `C_t`。

```text
原来的动作特征
  + 原来的视觉 cross-attention
  + 原来的 role adapter
  + social_gate × CrossAttention(action_query, C_t)
  = 本层新的动作特征
```

第一版在 7 个 ACT decoder layers 中都使用下面这个固定计算，不同时尝试 FiLM、AdaLN 或其他融合器：

```text
h_l       : [batch, 100, 384]，第 l 层的 action query
C_t       : [batch,   8, 384]，统一协调结果
social_l  = LowRankCrossAttention(query=LayerNorm(h_l), key/value=C_t, rank=32)
gate_l    = sigmoid(MLP(LayerNorm(h_l), mean(C_t)))       # [batch, 100, 1]
h_l       = h_l + gate_l × ZeroInitOutput(social_l)
```

`ZeroInitOutput` 表示最后一个线性层初始为 0：第一次 forward 时社会旁路不会破坏基础动作，训练后它才能逐渐变成非零。`rank=32` 沿用现有 ARCA role adapter 的低秩宽度，避免又引入一个未经控制的大参数块。

具体约束：

- `action_query` 是 Query，8 个 coordination tokens 是 Key/Value；
- 每个 decoder layer 都有自己的 adapter，但使用同一组 `C_t`；
- gate 按 layer 和 action query 分别计算，因此“马上执行的动作”和“动作块后半段”可以读取不同社会信息；
- social cross-attention 的输出投影从 0 初始化，所以训练开始时模型与基础动作路径一致，之后才逐渐学会使用协调信息；
- 沿用 R4 已验证的可靠度回退：不可靠的社会残差必须衰减，不能无界污染主路；如果高不确定应该触发等待或观察，要通过独立、可监督的 uncertainty/observe head 明确表达并重新过 gate，不能让未校准的残差暗中承担两种相反职责；
- P、B、T 不再分别直接接入 action decoder。action decoder 只读取统一的 `C_t`，否则又会退回三组 token 竞争的旧设计。

本项目已有的 [ARCADecoderLayer](../../vendor/stereo-core/stereo_core/stereo_decoder_variants.py) 已经提供 query-conditioned residual cross-attention 的代码骨架，因此第一版只需要增加一个 `CoordinationAdapter` 和对应 gate，不需要替换整个 ACT。

### 2.6 训练和部署时如何避免偷看答案

oracle P/B/T 只用于生成监督目标和测上限。正式模型在训练和部署时都必须沿着下面这条可微路径走：

```text
合法输入 → 预测 B → 预测 P → 预测 T → C → 动作
```

不能在训练时把 oracle B/P/T embedding 喂给下游模块，再在部署时突然换成预测值。各类 oracle label 只计算 `L_B`、`L_P`、`L_T` 和一致性 loss；主动作 loss 始终经过预测的 PBT。最终 checkpoint 同时包含 history/belief、progress、teammate、coordination adapter 和 action decoder，部署时不需要 teacher、reward model 或外部通信。

最终要验证的不是“P/B/T 分类准确”，而是：切断 `B→P`、`B/P→T`、`PBT→C` 或 `C→action` 中任一条边时，动作和对应合作行为是否按预注册方向退化。

### 2.7 研究锚点怎样落到这套模型里

第 15 节给出了论文、官方仓库、license 和代码边界的完整审计。这里把最重要的对应关系前移：**每篇工作到底支撑哪一块设计，以及它不能替我们证明什么。**

| V7.3 要解决的问题 | 主要近期锚点 | 具体落到本项目的设计 | 当前证据边界 |
|---|---|---|---|
| B 怎样记住被遮挡前的现场 | [MemoryVLA，ICLR 2026](https://arxiv.org/abs/2508.19236) | 当前 token 用 cross-attention 检索历史，再通过 gate 合并新观察与旧记忆；每个 episode 必须 reset | 它验证的是单机器人记忆，不能直接证明多机器人 belief 有效；仓库 license 未澄清前只读机制，不复制源码 |
| P 怎样表示“做到哪了” | [PALM，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_PALM_Progress-Aware_Policy_Learning_via_Affordance_Reasoning_for_Long-Horizon_Robotic_CVPR_2026_paper.html)、[ProcVLM，2026](https://procvlm.github.io/) 和 [ProgVLA，2026](https://arxiv.org/abs/2605.28231) | 联合学习 action 与 progress；progress 按任务谓词、程序步骤和剩余动作定义；长历史先压成少量 control-ready tokens | PALM 没有同等完整的官方实现；ProcVLM 是 progress reward/VLM，不是动作策略；ProgVLA 目前只作机制参考 |
| T 应该学习哪些合作变化 | [Sequential Asymmetric Imitation，2026](https://arxiv.org/abs/2606.16490) | 数据和反事实必须覆盖队友延迟、阶段不一致、让行和错误分工，T 不能只在正常专家轨迹上学“队友总会配合” | 它主要支撑数据与干预设计，不是可直接搬来的 T 模块源码 |
| P、B、T 为什么用单向 cross-attention | [AffordanceVLA，2026](https://arxiv.org/abs/2606.06155) | 借鉴严格单向的 block-causal attention，把它改成 `B→P→T→C→Action`；同一次前向不允许反向形成循环 | 借的是 mask 和连接原则，不搬它的 π0 权重、训练集或大规模训练栈 |
| 为什么先汇总成少量 C token | [Gamma-World，2026](https://arxiv.org/abs/2605.28816) | 借鉴 Sparse Hub Attention：P/B/T 先写入 8 个协调 token，ACT 只读取这 8 个 token；teammate slot 使用共享参数并做置换检查 | 它是视频 world model，只支撑 hub/slot 机制，不证明 ACT 上一定有效 |
| action chunk 交界处怎样不断片 | [ChainVLA，2026](https://arxiv.org/abs/2608.02326) | 下一次决策保留上一 chunk 的工作状态和未执行动作尾部，作为 B/P 的跨 chunk 连续性参考 | 工作很新且尚未完成代码审计，第一版只做消融参考 |
| C 怎样接入现有 ACT | 本仓库 [ARCADecoderLayer](../../vendor/stereo-core/stereo_core/stereo_decoder_variants.py) | 沿用已有的低秩、query-conditioned residual cross-attention 骨架，增加 zero-init social residual 和逐 query gate | 这是本地工程锚点，不是外部论文对 BPT 有效性的证据 |

这些锚点不是参考文献装饰，而是要变成可检查的实现和消融：MemoryVLA 对应 B 的检索、更新门和 episode reset；PALM/ProcVLM 对应“进度不能等于帧号”和 action-progress 联合头；AffordanceVLA 对应单向 attention mask 与禁止反向边；Gamma-World 对应 8 个协调 token 和机器人换位测试；Sequential Asymmetric Imitation 对应 delay、yield、wrong-role 数据；ChainVLA 对应 action chunk 交界测试。任何一项没有通过本项目的 Measurement 和闭环实验，都不能只凭论文写成“已经有效”。

## 3. 第 0 步：冻结 Measurement 边界（已完成）

### 3.1 先说结论

2026-08-11 已在远程服务器完成原 `SSC-V7-M0` 的第 0 步，历史结论是：

> `M0_STEP0_PASSED`：基础代码、W10、六任务数据、归一化、固定 seed、evaluator、分支、run root 和原预注册合同都已对上。这个结论只证明 M0 当时具备执行条件，不等于模型有效。

正式实验还没开始时，我们复查了 M0，发现三处问题：M2 没说清标签怎样才算合格；M4 要求模型在还看不出队友变化时就猜中变化；统计规则也没有充分处理 P/T/B 同时比较带来的偶然性。因此 M0 作废，由修正版 `SSC-V7-M1` 取代。旧目录永久只读，已经核验过的数据、W10、归一化和评测代码可以继续作为来源证据。M1 后来通过新 dry-run；严格版重放门禁先失败，用户针对固定 benchmark 明确放宽 qpos/终局 success 后，`SSC-V7-M1-R1` 全量重跑通过。当前状态见 3.5 和 4.4 节。

### 3.2 到底检查了什么

| 检查项 | 人话结论 | 结果 |
|---|---|---|
| 基础代码 | 本地和远端 `feat/model-improvements` 都精确指向 `945d1b49247612f6e67d79104726b67915cf86bf`，远端工作树干净，与刚刷新过的 origin 没有 ahead/behind | 通过 |
| W10 checkpoint | 对 734,198,133 字节 checkpoint 重新计算 SHA256，结果仍是 `e1b07b2...f574` | 通过 |
| W10 训练源码 | 当前主分支不直接包含训练 commit 对象，但只读 bundle 是完整 Git 历史，能解析到 `a335a2df...c77b`，patch 署名为 `Jeong Li <lzh_jeong@qq.com>` | 通过 |
| 六任务 HDF5 | 不是只数文件名，而是对 900 个文件逐一检查实际大小并重算 SHA256；744,660,714,054 字节全部与 manifest 一致，缺失/大小不符/哈希不符都是 0 | 通过 |
| 归一化 | W10 `normalization.pt` 哈希已核对；六个任务各自的 `normalization.npz` 也逐一重算并与 manifest 对上 | 通过 |
| Validation seeds | 六个 W10 seed 文件均存在且已冻结哈希；SSC-V7 又按固定 SHA256 算法生成 1,349 个唯一新 seed，与 W10 的 120 个 seed 零重叠 | 通过 |
| evaluator | evaluator 固定读取 RoboFactory 返回的 `info["success"]`，不是人工看视频判成功；代码和任务文件均已记录哈希 | 通过 |
| 服务器 | 4 张 RTX 5090 均可见，没有正在运行的训练或 W10 evaluator，约有 321 GiB 空间 | 通过 |
| 输出隔离 | SSC-V7 使用全新 run root，八个子目录为空，没有写入任何历史 run | 通过 |

六任务的闭环上限也已经固定：Lift Barrier、Pass Shoe、Place Food 是 500 步；Camera Alignment、Long Pipeline Delivery、Take Photo 是 1500 步。成功条件不是另写一套口径，而是冻结 RoboFactory commit `5868242322414a91454e22f1dd9641f613ba1bcf` 中原任务的物理判定；对应的人话和公式见 [oracle label 合同](../experiments/ssc_v7/oracle_label_spec.json)。

### 3.3 新阶段放在哪里

统一阶段名称是 `SSC-V7`。旧版和修正版的边界如下：

| 版本 | 状态 | 远端目录 |
|---|---|---|
| `SSC-V7-M0` | 已通过历史 dry-run，但在正式 Measurement 前被门槛审计取代，只读保留 | `/workspace/bwa_runs/ssc-v7-social-state-cooperation-v1` |
| `SSC-V7-M1 / SSC-V7-M1-R1` | 严格版失败回执只读保留；benchmark 放宽修订已全量重跑通过，后续 M2/M3/R4 也已完成 | `/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2` |

严格版 Measurement 审计实现冻结在 `169240db1514f2550ff18b8d495a48d598e75a57`；放宽版在独立分支 `feat/ssc-v7-measurement-relaxed` 冻结为 `a6ecfa5d9dbb6e4206da4da93873244b68145dab`。旧 M1 合同中预留的 P、T、B、PT、PTB 空分支没有训练代码，也不再代表 V7.3 的正式架构顺序；它们只作为历史占位保留，不能直接启动。

第 1 步完成后，要先生成新的模块训练合同，再建立下面这些兄弟分支。分支名和目录名必须在新合同中冻结，不能沿用旧名字假装架构没有变化：

| 新路线 | 计划分支 | 独立输出目录 |
|---|---|---|
| Measurement | `feat/ssc-v7-measurement` | `measurement/` |
| M3-R4/ARB | `feat/ssc-v7-m3-r4-arb` | `measurement/m3_r4_arb/` |
| B0-H | `feat/ssc-v7-b0-history` | `b0-history/` |
| B-core | `feat/ssc-v7-b-core` | `b-core/` |
| BP | `feat/ssc-v7-bp-progress` | `bp-progress/` |
| BT | `feat/ssc-v7-bt-teammate` | `bt-teammate/` |
| BPT | `feat/ssc-v7-bpt-directed` | `bpt-directed/` |
| 汇总 | `feat/ssc-v7-integration` | `integration/` |

所有正式模型仍从同一个冻结 base 随机初始化，不从前一路 checkpoint 续训。表中分支已经获得路线级设计/训练资格，但仍要在启动各阶段前冻结数据、预算、seed、停止规则和验收合同。

### 3.4 已经冻结的实验规矩

用人话概括如下：

- Measurement 已按 signal-first 路线级决策完成；原始 M1/M2/M3/R4 回执继续保留，M4/M5 改为稳定候选后的机制审计；
- 模型只能看合法图像、自己的 qpos、固定任务文本和 16 步合法历史；不能偷看队友真值、对象真值、未来、最终成功或虚构通信；
- Measurement 每个任务先用 4 个 seed 调试，再准备 60 个成功专家 episode。W10 先跑 20 个新 seed；如果样本仍不足，只能六个任务一起每次增加 5 个，最多增加到每任务 40 个。每个 W10 episode 最多选 24 个时刻做反事实分叉；
- 将来每条正式训练路线统一走 F0、4-update F1、5,000-update Discovery、Validation5、120,000-update Formal 和 Validation20；
- B0-H/B-core/BP/BT/BPT 使用相同 seed、数据、48 effective batch、100 action horizon、evaluator 和预算；
- 自动执行时不准看到结果后重抽 seed、加预算、换 evaluator 或降低门槛。用户因固定 benchmark 的已知限制明确改变研究口径时，必须建立新的只读 gate revision、保留旧结论并原样重跑，不能覆盖旧 receipt；
- BP 必须等 B-core 通过自身模块门禁并冻结 P 合同；BT 必须等 B-core 通过并冻结 T 合同；BPT 必须等 BP、BT 均通过各自漏斗，不能同时开工后挑最好结果。

历史数字和停止码见 [阶段合同](../experiments/ssc_v7/stage_contract.json)、[seed 合同](../experiments/ssc_v7/seed_contract.json)、[Measurement gate](../experiments/ssc_v7/measurement_gate.json)、[M1 benchmark 放宽修订](../experiments/ssc_v7/m1_relaxed_gate.json) 和[第 1 步详细结果](../reports/20260814_P1_STEP1_MEASUREMENT_CONCLUSIONS_AND_ACCEPTANCE_ZH.md)。旧合同中的 P/T/B/PT/PTB 训练顺序已被本路线替代；模块训练前必须生成 V7.3 新 schema 的合同，不能修改已经产生过 receipt 的旧文件。

### 3.5 冻结产物和 dry-run

M1 修正版产物已经在本地生成：

- [执行提示词](../experiments/ssc_v7/EXECUTION_PROMPT_ZH.md)
- [输入白名单](../experiments/ssc_v7/input_contract.json)
- [oracle label 规范](../experiments/ssc_v7/oracle_label_spec.json)
- [Measurement 判定门槛](../experiments/ssc_v7/measurement_gate.json)
- [seed 合同](../experiments/ssc_v7/seed_contract.json)
- [阶段、预算和停止规则](../experiments/ssc_v7/stage_contract.json)
- [来源核验回执](../experiments/ssc_v7/source_receipt.json)
- [run manifest](../experiments/ssc_v7/run_manifest.json)
- [冻结时的 dry-run 占位回执](../experiments/ssc_v7/dry_run_receipt.json)
- [实际通过的 M1 dry-run 回执](../experiments/ssc_v7/m1_dry_run_receipt.json)
- [实际 dry-run 日志](../experiments/ssc_v7/step0_audit/dry_run.log)
- [全部预注册文件 SHA256](../experiments/ssc_v7/sha256sums.txt)
- [六任务归一化补充核验回执](../experiments/ssc_v7/normalization_verification_receipt.json)
- [dry-run launcher](../../scripts/before_we_act/launch_ssc_v7_measurement.sh)

M0 冻结文件仍位于：

```text
/workspace/bwa_runs/ssc-v7-social-state-cooperation-v1/pre_registration
```

这里的远端文件权限为 `0444`、目录为 `0555`；其历史 dry-run 检查了 11 个文件哈希、JSON、W10、六个数据 manifest、900 文件核验回执、1,349 个 M0 seed、八个分支、4 张 GPU、空输出目录和进程冲突，最终输出 `SSC_V7_DRY_RUN_PASSED`。这些结果只属于 M0。

M1 已同步到 `/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/pre_registration` 并设为只读。2026-08-11 的实际 dry-run 重新核验了全部预注册文件、代码和数据来源、W10、八个分支、4 张 GPU、空输出目录和进程冲突；固定算法生成的 1,470 个新 seed 全部唯一，并排除了 120 个历史 W10 seed。严格版 M1 随后因重放精度失败；2026-08-12 用户明确要求把 qpos 和终局 success 改成诊断项，新修订冻结后全量重跑通过。M2-R3、M3-R4 与 R4-C 也已完成，后续状态统一见第 4 节和详细结果文档。

### 3.6 两个不能假装不存在的限制

1. 服务器的 `/workspace` 不是持久卷。实例 stop/start 会保留，但 recycle/destroy 会丢失。因此每个不可替代的 Measurement gate 产物必须同步到服务器外，没同步就不能算真正完成。
2. RoboFactory 有 MIT License，但项目仓库没有顶层 LICENSE。当前只冻结为本服务器内部研究使用；在所有者补充明确许可前，不对外分发项目代码、数据或 checkpoint。

另外，本机 `8080` 已被现有进程占用，因此用户给出的 `-L 8080:localhost:8080` 无法绑定。本次改用不带本地转发的同一 SSH 连接完成命令行工作，不影响本次 M1 命令行审计，也没有停止占用 8080 的用户进程。

因此服务器不可持久风险和许可边界仍然有效，但它们不再改变当前进入第 2 步模块训练的路线状态。

## 4. 第 1 步：Measurement（已完成）

> 详细结论、历次门禁、完整数字、状态码、哈希与远端位置见：[第 1 步 Measurement 结论与验收记录](../reports/20260814_P1_STEP1_MEASUREMENT_CONCLUSIONS_AND_ACCEPTANCE_ZH.md)。本节只保留后续路线需要继承的结论。

### 4.1 路线级裁决

第 1 步按负责人 2026-08-14 的后验 signal-first 决策收口：现有证据已经足以支持进入模块设计和训练。这个决定不把原严格 R4-B 的归因失败改写成通过，也不声称 ARB 已经贡献独立净增量。

| 问题 | 当前答案 | 后续影响 |
|---|---|---|
| 数据与标签能否支持训练 | 可以 | M1 放宽版和 M2-R3 已通过 |
| 完整旧 192 维 B 是否值得继续 | 不值得 | 旧 concat B 路线停止 |
| 正确 ARB 是否有动作相关信号 | 有 | R4-A1 已通过 |
| 合法输入能否产生可用 ARB_hat 栈 | 可以 | R4-C 在全新密封 test 上确认纸面收益 |
| ARB 是否胜过通用 residual / 时间阶段解释 | 没有证明 | 禁止 ARB 独立语义 claim |
| 是否可以开始模块设计与训练 | 可以 | 第 2 步及之后按漏斗执行 |
| M4/M5 是否仍是开模前置门禁 | 不是 | 后置为稳定候选的机制消融 |

路线级状态为：

> `COMPLETED_STEP1_SIGNAL_FIRST_MODULE_AUTHORIZED`

它是负责人在结果已知后的研究优先级裁决，不覆盖任何原始机器回执。

### 4.2 最小证据摘要

- M1：900/900 文件、284,183 帧的哈希/schema/时序检查通过；30/30 保存状态分叉确定。跨进程从头复刻原 qpos/success 只作 benchmark 诊断。
- M2：按专家实际物理轨迹重定义阶段后，30 条、9,494 帧通过自动门禁；121 个片段完成 AI 辅助、物理证据和因果重放复核，并由负责人明确授权。
- M3-R3：收敛问题修好后，完整旧 oracle B 相对 HC 为 `-6.20%`，B_hat 为 `-13.97%`，所以旧 B 停止。
- R4-A1：oracle ARB 相对 HC 改善 `19.39%`，并胜过 zero/noise/label-shuffle/episode-shuffle。
- R4-A2：ARB 表示主效应 `+1.79%`；query 相对 direct 为 `-14.14%`，所以第一版锁定 direct residual。
- R4-B：ARB_hat + direct 在 confirmation 上相对 HC 改善 `25.19%`，36/36 活动头校准优于常数，gate-off 精确回到 HC；原严格归因闸门仍失败。
- R4-C：72 条全新成功 episode、13,028 个评测行、test 只打开一次；候选相对 HC 改善 `26.05%`，95% CI `[24.15%, 27.90%]`，三 seed、六任务全正。
- 关键反证：hidden-only、time-only、row-shuffle 和同阶段 shuffle 仍更强；当前主要收益由 direct residual capacity 足以解释。

### 4.3 进入模块路线后继承什么

1. B-core 继承 ARB schema，只保留 contact/grasp/custody、handoff、teammate motion、blocking/collision、visibility/staleness 和 uncertainty/missingness。
2. 禁止把 frame index、episode ID、固定机器人编号、任务完成百分比、remaining goals、未来队友动作或旧 192 维 B 偷渡进 ARB。
3. direct residual 是第一版默认接口；residual 最后一层 zero-init，低可靠度和 gate-off 必须回到本候选基础动作。
4. query-attention 只作为历史反方消融，不因进入正式模块而复活。
5. B0-H、B-core、BP、BT、BPT 使用相同 TeamTemporalSample、seed、sample cursor、action target 和预算。
6. hidden-only、time/phase、shuffle、stale 和 gate-off 继续作为诊断；它们不再阻断开始训练，但继续限制论文归因。
7. 纸面未来 16 步 NRMSE 只用于筛选，最终 winner 仍必须通过闭环 Validation/Confirmation。

### 4.4 M4/M5 的新优先级

M4 和 M5 不删除，改到候选架构与训练 recipe 稳定以后统一执行：

- M4 检查 partner change 是否因果改变正确动作，以及模型能否在变化可见后及时响应；
- M5 检查 W10 和最终候选的失败中有多少真正属于合作根因；
- residual 隔离使用嵌套结构，在冻结 hidden residual 后比较真实 ARB、ARB 置零、同阶段 shuffle 和 oracle ARB。

这些实验决定后续能否提出严格 ARB 机制和合作因果 claim，不再决定现在能否进入模块设计与训练。

## 5. 第 2 步：先修数据单元，再训练 B0-H

本节现在是活动入口。先冻结 `TeamTemporalSample`、数据/seed/预算合同和 B0-H 公平基线，再进入 B-core；M4/M5 不再是本节的前置条件。

### 5.1 当前数据加载器为什么不能直接训练 PBT

现有 `NoWristFrameDataset` 每次只读取一个 episode、一个 arm、一个时刻，并预测这个 arm 后续的动作；Sampler 也独立抽取 `(episode, arm, time_index)`。它没有返回连续历史，也没有把同一时刻的团队信息组织在一起。直接在这个 Dataset 上加 PBT head，只会得到几个缺少时序依据的分类器。

第一项实现工作是增加统一的 `TeamTemporalSample`：

```text
样本身份：task + episode + ego arm + 当前时刻 t
模型输入：t-15 到 t 的合法 global/local RGB、own qpos、own action history、task text
动作目标：ego 从 t 开始的 100 步 commanded action
训练标签：同一 episode/t 对应的 B、P、T sidecar；只作 target，不作输入
```

同一条 episode 中的连续样本必须能按时间顺序组成训练组，才能正确更新和重置 B。随机抽到另一个 episode 时，上一条 episode 的 memory 必须清空。agent slot permutation、padding mask、历史缺帧和 episode 边界都要在 F0/F1 中测试。

### 5.2 B0-H 是什么

B0-H 是公平基础模型，不是旧 W10 的逐字复制。它与社会模型读取完全相同的 task text 和 16 步历史，但不预测 B/P/T，也没有 `C_t→action` 的真实社会信息。

为了排除“多了历史或参数所以变强”，B0-H 必须有两种控制：

- 与社会路线相同的 history encoder；
- 与社会路线等参数的 constant 或 input-independent noise slots，但这些 slots 不携带社会标签。

B0-H、B-core、BP、BT、BPT 使用完全相同的 `TeamTemporalSample`、seed、sample cursor 和 action target。所有路线从同一冻结 base 随机初始化。

B0-H 最低资格保持不变：总成功 `>=80/120`；Lift/Long/Photo/Shoe 合计 `>=72/80` 且每项 `>=16/20`；Camera `>=6/20`；Camera+Food `>=8/40`。达到 `>=88/120` 才能声称在原始成功数上达到 W10。

## 6. 第 3 步：B-core——先学会保存“现场状态”

B-core 只增加 belief 和 `B→C→action` 路径，不启用 P、T。它已获得路线级设计资格；完成第 2 步、冻结模块合同并确认 B0-H 合格后，才启动正式训练。

正式 B-core 不再放大旧 192 维 B，而是继承通过 Measurement 的 `B^AR` schema、action-facing queries、zero-init residual、可靠度回退和全部负对照。`M3-R4` 中没有通过的字段、融合器或记忆机制不得在这里复活。

**研究锚点：**B 的“历史检索→门控更新→episode reset”参考 [MemoryVLA](https://arxiv.org/abs/2508.19236)；memory representation 与集成消融参考 [RoboMME](https://arxiv.org/abs/2603.04639)；agent 对称性和少量 hub token 参考 [Gamma-World](https://arxiv.org/abs/2605.28816)。这些结果分别来自单机器人 VLA benchmark 或生成式 world model，不能替本项目证明多机器人 B-core 有效。

每个控制步执行：

1. entity/event queries 读取当前观察和最近 16 步历史，只更新已通过 R4 的 contact、custody、handoff、motion、blocking、visibility/staleness 与 reliability；
2. 用 resettable gate 把新证据与上一时刻 `B^AR_{t-1}` 合并；episode 边界强制清空，过期关系显式衰减；
3. teammate slots 共享权重并使用置换等变 agent encoding，禁止固定 ID 捷径；
4. 8 个 coordination queries 只读取 `B^AR_t`，同时保留 R4 已验证的 4 个 action-facing query 职责；扩成 8 个必须在正式合同中作为一次冻结选择，不得中途搜索；
5. CoordinationAdapter 用 zero-init residual 接入 action queries；低可靠度时回退本候选自己的基础动作主路。强制 gate-off 时结构上退化为与 B0-H 相同的无社会信息路径，但正式 B-core 仍按第 13 节从共同 base 独立初始化，不能加载独立 B0-H checkpoint 充当外部 fallback。

B-core 不预测 remaining goals 和 progress。这样 BP 后续增加的收益才不会被 B 提前吃掉。

B-core 必须额外通过：

- episode 切换时 memory 被完全清空，同一 episode 连续执行时已经发生的交接不会在短暂遮挡后立刻丢失；
- 一致交换 teammate slots 后，B、C 和动作影响按同样方式交换；
- 遮住队友或关键物体后 uncertainty 上升；
- 加入无关历史或上一 episode 状态时，正式结果不能随干扰量持续恶化；若会恶化，必须先加入检索/过期淘汰并重新走切边实验；
- 切断 `B→C` 或打乱 B 后，动作和风险行为按预注册方向退化；
- oracle B 只报告上限，不成为正式模型输入或 fallback。

如果 B 标签准确，但切断 `B→C` 后动作不变，或 B-core 只在 reset-off/stale 条件下“变好”，B-core 失败。

## 7. 第 4 步：BP——让进度读取 belief

BP 从共同 base 随机初始化，同时训练 B 和 P；它不从 B-core checkpoint 续训。

**研究锚点：**PALM 支撑 action-progress 联合学习，ProcVLM 支撑按程序步骤和剩余动作定义进度，ProgVLA 支撑把长历史压成少量控制 token。三者都不直接证明本项目的 `B→P→C→action` 有效，因此 BP 仍要用 time-only、切断 `B→P` 和切断 `P→C` 三组对照验证。

P 不直接读取原始 RGB。P 的 task-predicate queries 读取 `B_t` 和冻结的 task graph，输出：

- 已完成和剩余任务谓词；
- `stage_id` 与 stage 内连续 progress；
- 哪些并行子目标都可以先做，而不是强迫唯一顺序。

task graph 只提供“这个任务有哪些谓词、哪些可以并行、哪些完成后才能进入下一步”的空结构，不提供当前谓词真值。它由合法的 canonical task text 映射得到，映射和图结构在 M2 后冻结；当前完成状态仍必须由 B/P 自己预测，不能把 simulator 的 remaining-goal truth 填进图里。

coordination queries 在 BP 中读取 B+P。动作 hidden 还要预测与 action horizon 对齐的 progress delta，用来检查“执行这段动作以后任务应该推进多少”。这使 progress 与动作共同学习，而不是在旁边完成一个阶段分类任务。

BP 必须通过：

- 切断 `B→P` 后，P 的物理一致性和动作表现退化；
- stage swap、remaining-goal swap 后，重复劳动或错误切换按预注册方向增加；
- time-only probe 不能复现收益，同一物理/历史状态只改 frame index 时 P 不变；
- 切断 `P→C` 后动作发生可解释退化；
- B0-H、B-core、BP 的参数和历史输入被公平控制。

## 8. 第 5 步：BT——让队友未来读取 belief

BT 也从共同 base 随机初始化。在这条受控路线中没有真实 P，T 只读取 B；这条路线用来回答：“知道当前团队状态以后，继续预测队友未来是否还有新增价值？”

**研究锚点：**Sequential Asymmetric Imitation 主要支撑 delay、phase mismatch、yield 和 targeted intervention 这些合作数据；Gamma-World 只支撑 teammate slot 的共享参数和置换对称。当前没有一套近期开放代码能直接证明“视觉双机器人操作中的 T future-mode tokens”有效，这正是 BT 必须单独做、也允许被实验否决的研究空白。

T 不能只给出一个确定的“队友下一动作”。遇到遮挡、等待或角色切换时，未来本来就可能有多种。T 要输出若干 future-mode tokens，每种模式包含目标、角色、未来动作摘要和置信度；模式数量和时间分桶在 M2 后冻结。

coordination queries 在 BT 中读取 B+T。当前数据没有合法通信，所以 T 只能从 global/local 图像、own qpos、own action history 和 B 推断，不能读取 oracle peer action、peer qpos 或虚构消息。

BT 必须通过：

- 一致交换 teammate slots 后，T 的未来模式和动作影响相应交换；
- 打乱 teammate history 或切断 `B→T` 后，未来预测和合作动作退化；
- hold、delay、wrong-role 变得可观察以后，ego 在 16 步内产生合理的 wait/yield/接手变化；
- 切断 `T→C` 后，上述伙伴条件行为消失或减弱；
- uncertainty 有校准报告，不能用过度自信的单一未来掩盖多种可能。

## 9. 第 6 步：BPT——三者怎样真正交融

只有 B、P、T 的 Measurement 信号通过，且 BP、BT 均证明新增边被动作使用后，才启动 BPT。

**研究锚点：**AffordanceVLA 提供单向 block-causal attention 的近期实现依据，Gamma-World 提供少量 hub token 交换信息的依据，本仓库 ARCA 提供 C→action 低秩残差旁路的工程骨架。三者的组合是本项目的新假设，不是任何一篇论文已经验证过的现成结论，所以必须保留 `BPT-flat`、`BPT-no-PtoT`、`BPT-C-off` 和逐边切断对照。

BPT 严格执行一遍：

```text
B = BeliefUpdate(legal history, previous B)
P = ProgressCrossAttention(query=P, key/value=B + task graph)
T = TeammateCrossAttention(query=T, key/value=B + P)
C = CoordinationCrossAttention(query=8 coordinator slots, key/value=B + P + T)
action = ACT(observation) + CoordinationAdapter(action queries, C)
```

这里新增且必须单独验证的是 `P→T`：同样的现场状态下，剩余目标不同，队友下一步角色和动作预测也应该不同。例如物体已经完成交接时，T 不应继续预测原队友重复交接。

以下对照在看到正式结果前冻结：

| 对照 | 改了什么 | 回答什么问题 |
|---|---|---|
| `BPT-flat` | P/B/T 预测保持不变，但原样拼到 decoder memory，不经过 C | 统一协调瓶颈是否比“直接塞 token”更好 |
| `BPT-no-PtoT` | T 只读 B，不读取 P | P 是否真的改变队友未来判断 |
| `BPT-C-off` | C 不进入 action decoder | PBT 是否只是辅助预测 |
| `BPT-edge-off` | 分别切断 B→P、B→T、P→T、P/B/T→C | 每条交融边是否有独立作用 |
| `BPT-oracle` | 使用 oracle PBT | 只测理论上限，不具备部署资格 |

`BPT-flat` 和 `BPT-oracle` 只作诊断，不参加 winner 选择。正式候选是有方向的 BPT。

## 10. 第 7 步：联合训练和最终门槛

正式 BPT 使用一个 checkpoint、一次从随机初始化开始的联合训练。loss 至少包含：

```text
L = L_action
  + λB × L_belief
  + λP × L_progress
  + λT × L_teammate_future
  + λF × L_future_belief
  + λC × L_cross_module_consistency
```

这些 loss 的权重和是否逐步升高，必须在 Discovery 前冻结。可以在同一次训练中先让结构化 loss warm up，再逐渐增加 action loss，但不能保存一个模块 checkpoint 后把它当作下一路线的初始化。

BPT 只有同时满足以下条件才有 winner 资格：

- 不损害 W10 已经很强的四个 protected tasks；
- 在 Camera Alignment、Place Food 或预注册 cooperation 指标上产生真实收益；
- `B→P`、`B/P→T`、`PBT→C`、`C→action` 的因果 gate 全部通过；
- 相比 `BPT-flat` 的优势来自协调机制，而不是更多参数或不同历史输入；
- 单 checkpoint 直接输出动作，不调用 W10、oracle、teacher、reward model 或外部通信；
- 延迟、显存、memory reset 和非法动作满足部署要求。

如果 BP 或 BT 通过而完整 BPT 失败，保留相应部分模型的科学结论，但不能宣称 PBT 交融成功。模块越多不自动代表模型越好。

## 11. 每条训练路线都走同一个漏斗

```text
F0 静态检查
  ↓
F1 真实数据集成检查
  ↓
Discovery 短预算筛选
  ↓
Validation5 小规模闭环
  ↓
Formal 120,000 updates
  ↓
Validation20 正式闭环
  ↓
Selection
  ↓
Confirmation50 与 W10 配对确认
```

每一层做什么：

| 阶段 | 目的 | 失败后怎么办 |
|---|---|---|
| F0 | 检查 shape、mask、参数路径、输入白名单、无未来泄漏 | 修实现，不开始训练 |
| F1 | 用真实 HDF5 跑 forward/backward/optimizer/save/resume | 修集成，不进入 Discovery |
| Discovery | 用冻结短预算检查 loss、因果干预和机制是否被使用 | 该路线停止，不降低门槛 |
| Validation5 | 用 5 个固定 seed 快速检查闭环是否值得正式训练 | 失败则不进入 Formal |
| Formal | 完成 120k updates，保存完整 receipt | 训练异常则按同一 manifest 恢复 |
| Validation20 | 计算正式六任务成绩和 cooperation 指标 | 不合格就没有 winner 资格 |
| Confirmation50 | 临时 winner 与 W10 在每任务 50 个新 seed 上配对比较 | 非劣下界不通过则无最终 winner |

Discovery 的具体 update 数、loss 门槛和 Validation5 晋级规则必须在 execution prompt 中预注册；本文不在没有测量依据时编造这些数字。

## 12. 最终怎么判断模型合不合格

### 12.1 基础闭环门槛

若数据和 evaluator receipt 与当前 W10 相同，正式候选最低资格是：

- 六任务总成功 `>=80/120`；
- Lift/Long/Photo/Shoe 合计 `>=72/80`；
- 上述四任务每项 `>=16/20`；
- Camera `>=6/20`；
- Camera+Food `>=8/40`；
- 对应机制的因果 gate 通过。

如果论文要直接声称“达到或超过 W10”，相同 Validation20 的原始总成功必须 `>=88/120`。

### 12.2 合作指标

除了成功率，还必须在查看结果前冻结以下指标：

- 完成任务用了多少步；
- 每个机器人 idle/wait 的比例；
- 是否重复处理已经完成的目标；
- 是否互相阻挡或争抢；
- 碰撞、安全投影和高风险动作；
- 团队贡献是否严重失衡。

只提高 auxiliary accuracy、不改善动作或合作行为，不算合作提升。只保持成功率、不改善任何预注册 cooperation 指标，只能写“社会状态可建模，但没有产生有效合作增益”。

### 12.3 Confirmation50

临时 winner 与 W10 使用每任务 50 个全新、配对 seed。报告 point estimate 和 paired-bootstrap 95% CI。

非劣要求：相对 W10 的成功率差值，95% CI 下界不得低于 `-6.67pp`。

如果没有候选同时满足动作能力、合作收益和后置因果 gate，最终结论就是“V7.3 无 winner”。不能为了必须产出模型而继续放宽标准。

## 13. 所有路线必须共享的公平条件

B0-H、B-core、BP、BT、BPT 必须共享：

- 数据 receipt；
- sample cursor；
- 随机种子；
- 训练更新数和 effective batch；
- optimizer/scheduler policy；
- action horizon 和 temporal ensemble；
- Validation5/20/50 seeds；
- evaluator、max steps 和成功条件。

新增模块带来的参数必须通过缩小 common width 抵消，或者增加同参数但不含真实社会信息的 capacity control。不能把“参数更多”写成“社会状态更有效”。

所有正式路线从同一个 base commit 建兄弟分支。执行顺序是预算顺序，不是 checkpoint 继承顺序：

```text
错误：B0-H checkpoint -> B-core checkpoint -> BP checkpoint -> BPT checkpoint
正确：同一个 base -> B0-H/B-core/BP/BT/BPT 分别从头训练
```

## 14. 工程入口和产物要求

正式启动前需要为新路线建立：

1. 独立 stage ID，不能复用历史 R13/R13N；
2. Measurement、B0-H、B-core、BP、BT、BPT 独立兄弟分支和共同 integration 分支；
3. 独立 run root，禁止写入 R11/R12 目录；
4. 不可变 run manifest、数据 receipt、seed receipt 和源码 receipt；
5. F0/F1、launcher、monitor、graceful stop、resume 和 acceptance 脚本；
6. 每个阶段的结构化 JSON、日志、checkpoint hash 和最终结论。

monitor 至少要显示：当前路线、branch/commit、GPU、PID、stage、update、loss、ETA、checkpoint、Validation、因果 gate、显存、温度、OOM/NaN/stale 和 acceptance 状态。

R11/R12 的 runbook 只能参考工程结构，不能作为活动入口。第 1 步详细状态见独立结果文档；当前活动入口是为 `TeamTemporalSample`、B0-H 和后续 ARB-B-core 冻结新的模块合同。M4/M5 与 residual 隔离保留为稳定候选后的机制审计。

## 15. 研究依据、反证和开源采用边界

### 15.1 先区分论文事实、本项目推断和路线决定

本节只使用论文主页、arXiv/CVF 等原始论文页面和作者官方仓库。外部工作回答“哪些机制值得测”，本项目实验才回答“它在六任务上是否有效”。证据分三档：

1. **可迁移代码锚点**：官方代码和明确 license 都存在；完成 commit、文件 hash、NOTICE 与符号映射 receipt 后，才可迁移小型机制；
2. **机制锚点**：论文有实验，但仓库不完整、license 不清楚或模型规模差异过大；只允许独立实现思想；
3. **反证/边界证据**：结果提醒我们某机制并不普遍有效，必须把失败模式写成对照，不能只摘正面数字。

### 15.2 为什么把路线修成 ARB

| 外部原始证据 | 论文/仓库实际说明了什么 | 对本项目的可证伪推断 | 因而写入的路线决定 |
|---|---|---|---|
| [GuidedVLA，RSS 2026](https://arxiv.org/abs/2605.12369) 与[官方仓库](https://github.com/GuidedVLA/GuidedVLA) | 用专门 attention heads 学 object/geometry/skill，并通过 zero-init control branch 加到主路 | 旧 B 负收益可能部分来自“整份状态直接污染动作”，小残差更容易测清增量 | R4 冻结 HC，只训练 action-facing ARB residual；`g_B=0` 必须精确回退 HC |
| [Action QFormer，2026](https://arxiv.org/abs/2607.14635) | instruction-conditioned action queries 把继承的多模态信息重组为 action-facing representation，并减少上游被动作监督大范围改写 | 动作自己提问可能优于把 192 维 B 原样拼接 | 该推断已在 successor 2×2 中实测；本项目小探针上 query 输给 direct `14.14%`，因此不进入 R4-B 默认方案 |
| [Event-VLA，2026](https://arxiv.org/abs/2606.29384) | action queries 经 gated cross-attention 选择性融合 event tokens | 新模态/新状态不必直接混入主干，门控选择值得小规模验证 | selective fusion 作为反方消融保留；当前负结果优先于外部论文，R4-B 采用 direct residual |
| [LangForce，ICML 2026](https://arxiv.org/abs/2601.15197) | 同时建模无语言 prior 与有语言 posterior，以条件 PMI 抑制视觉捷径 | 与其期待网络“自己使用 B”，不如显式比较无 B 主路和有 B 增量 | 使用共享 HC prior + ARB residual 双分支；R4 第一版不用复杂 PMI loss，先用 matched/shuffle/stale 对照验证增量 |
| [RoboMME，ICML 2026](https://arxiv.org/abs/2603.04639)、[policy learning](https://github.com/RoboMME/robomme_policy_learning) 与 [benchmark](https://github.com/RoboMME/robomme_benchmark) | 16 个 temporal/spatial/object/procedural memory 任务、14 种 π0.5 memory variants；表示效果高度依赖任务 | “加 memory 就会好”不成立 | R4 不先上 recurrent memory；先比较无记忆、frame selection、stale/reset 控制 |
| [RoboMME-Interference，2026](https://arxiv.org/abs/2606.22338) | perceptual memory 在无干扰时受益，但随无关 session 增加而持续衰减；检索相关演示可恢复 | 长记忆会污染，不只是遗忘 | episode 强制 reset，增加 previous-episode、stale 和无关历史干扰测试；需要时先检索再写入 |
| [MemoryVLA，ICLR 2026](https://arxiv.org/abs/2508.19236) 与[官方仓库](https://github.com/shihao1895/MemoryVLA) | perceptual/cognitive memory 用于长时依赖 | 遮挡和交接可能需要可更新的短时状态，但不能从单机器人结果外推多机器人收益 | 只有无记忆 ARB 已通过且遮挡保留率不足，才允许 R4-D resettable memory；license 未明前不复制源码 |
| [Gamma-World，2026](https://arxiv.org/abs/2605.28816) 与[官方仓库](https://github.com/nv-tlabs/Gamma-World) | Simplex Rotary Agent Encoding 支持 permutation-symmetric agent conditioning；Sparse Hub Attention 用少量 hub 交换多主体信息 | teammate slot 不应绑定固定编号，大量两两 attention 也不是第一选择 | B-core 使用共享 slot、置换测试和少量 coordination queries；不迁移视频生成主干 |
| [CHORUS，2026](https://arxiv.org/abs/2606.12352) | 单个共享 VLA 可只凭各机器人本地观察和 robot-identifying prompt 做去中心化协作，无需推理时通信 | 显式 B 不是协作唯一道路，强直接策略是必要反方 | 保留不依赖 B 的 reactive/shared-policy 基线路线；若 ARB 失败，不把“无 B”误判成“无法协作” |
| [Sequential Asymmetric Imitation，2026](https://arxiv.org/abs/2606.16490) | staged curriculum 暴露 delay、phase mismatch、insufficient yielding 和 conflict | 仅正常专家轨迹可能缺少让 B 改变动作的关键分歧样本 | 后置 M4/M5 与后续数据修订必须覆盖 delay/yield/wrong-role；不能靠结构弥补完全缺失的数据 |
| [Embodied Interpretability，ICML 2026](https://arxiv.org/abs/2605.00321) 与[作者代码](https://github.com/robot-future/vla-explain) | ISS 用干预式 masking 估计视觉区域对动作的因果影响，NMR 衡量 nuisance 依赖 | 只看 attention 或标签准确率不能证明动作在用正确原因 | R4 把 ARB-off/shuffle/stale 作为主 gate；ISS/NMR 只作诊断，不替代 paired action/闭环指标 |
| [VLA-ATTC，2026](https://arxiv.org/abs/2605.01194) 与 [VLAConf，2026](https://arxiv.org/abs/2605.29605) | 前者用不确定性 clutch 切换额外推理，后者以轻量 head 做单次前向 confidence | 不确定性有用的前提是被校准，并且有明确的 fallback 行为 | R4 先用可靠度衰减 ARB residual；主动观察/候选动作 critic 属于以后独立路线，不能用来救 oracle 负收益 |

上表故意同时保留支持和反对证据。最强的反对意见有三个：RoboMME 说明 memory design 是 task-dependent；其官方 policy 仓库直接披露 recurrent variants 仍 underperforming；CHORUS 说明无需显式 B 也可能合作。因此 V7.3 的判断不是“ARB 一定成功”，而是“当前纸面信号足以支持先完成模块工程，再用后置消融判断机制”。

### 15.3 2026-08-13 官方仓库只读核查

| 官方仓库 | 核查 HEAD | license / 成熟度 | V7.3 采用边界 |
|---|---|---|---|
| [GuidedVLA](https://github.com/GuidedVLA/GuidedVLA) | `04be059e0d6bd448be5cb45fdbafc775f7eb5e38` | Apache-2.0；含训练、评测、checkpoint 与数据入口；第三方模型另受各自条款约束 | 可参考/迁移 zero-init control attention 小机制；不迁移 π0/openpi 主干和权重 |
| [RoboMME policy learning](https://github.com/RoboMME/robomme_policy_learning) | `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b` | Apache-2.0；含 symbolic/perceptual/recurrent variants、训练评测和 checkpoint；官方注明 recurrent 仍 underperforming | 参考 memory 表示/融合消融和可恢复评测流程，不迁移 π0.5 主干 |
| [RoboMME benchmark](https://github.com/RoboMME/robomme_benchmark) | `0bdbb1789c77642f93bcb4100dc4477e2b999f29` | Apache-2.0；16 任务、1,600 demos、固定 train/val/test seeds | 参考 memory 单元测试、固定 seed 和干扰评测组织；不把其任务数字当本项目证据 |
| [Gamma-World](https://github.com/nv-tlabs/Gamma-World) | `6a95de85c439d8ea73eae34c88fbfd4e89ea02e2` | Apache-2.0；2026-06-16 已发布训练 pipeline，含 THIRD_PARTY_NOTICES | 可参考 agent encoding/hub attention；不迁移生成式视频模型 |
| [MemoryVLA](https://github.com/shihao1895/MemoryVLA) | `d732ea9072bc063399ccc817aed74ab172eb50be` | 有代码、权重和数据入口；当前 HEAD 未发现顶层 LICENSE | 只读分析 retrieval/gate/reset；license 澄清前不复制任何源码 |
| [LangForce](https://github.com/ZGC-EmbodyAI/LangForce) | `ff35aab1c9c6a02b4daf73c71248350f30d22048` | 有训练说明和权重；当前 HEAD 未发现顶层 LICENSE | 只借 dual-branch/conditional-information 思想，独立实现 |
| [vla-explain](https://github.com/robot-future/vla-explain) | `202d2a9a00fb4b99083559525d54d8f2a7eb4d3f` | 已发布 ISS/NMR toolkit；当前 HEAD 未发现顶层 LICENSE | 只作只读诊断参考；license 澄清前不复制工具代码 |
| [AffordanceVLA](https://github.com/Skywalker-yqz/AffordanceVLA) | `7689e423fc264a16ce9a662dd10e4b0470066f98` | MIT；含 model/training/attention mask | 通过 ARB 后才可参考后续 B→P→T block-causal mask |
| [ProcVLM](https://github.com/RUCKBReasoning/ProcVLM) | `377523a31f05bab9c0db5ac8b9edfa7b7f03968a` | 当前 HEAD 未发现顶层 LICENSE | 仅供以后 P 标签/进度审计，不进入 R4 |

这张表仍不是源码迁移 receipt。真正复制任何符号前，必须再冻结 commit、逐文件 hash、LICENSE/NOTICE/SPDX、依赖/权重/数据条款和“外部符号→本项目符号”映射。论文声称“will release code”不等于本轮已核验到可迁移代码；[Action QFormer](https://arxiv.org/abs/2607.14635)、[Event-VLA](https://arxiv.org/abs/2606.29384)、[CHORUS](https://arxiv.org/abs/2606.12352) 和 [VLA-ATTC](https://arxiv.org/abs/2605.01194) 本轮均按机制锚点处理。

### 15.4 不进入 R4 第一版的东西

- 不引入 RMT、TTT、跨 episode 长记忆或生成式 world model；先证明短窗口 ARB 有动作价值；
- 不引入 P、T、progress critic、teammate future 或多轮 PBT 互读；它们会让 oracle B 失败原因重新混在一起；
- 不换 π0/π0.5/OpenVLA，不迁移外部 checkpoint；R4 仍用本项目小探针回答局部因果问题；
- 不以 attention map、B 分类准确率、论文 SOTA 数字或单个好 seed 代替未来 16 步动作门槛；
- 不删除 direct/reactive baseline。若 ARB 失败，应允许另立 shared-policy、数据课程或纯动作架构路线，而不是无限加深 B。

[PALM](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_PALM_Progress-Aware_Policy_Learning_via_Affordance_Reasoning_for_Long-Horizon_Robotic_CVPR_2026_paper.html)、[ProcVLM](https://procvlm.github.io/)、[ProgVLA](https://arxiv.org/abs/2605.28231) 继续作为以后 P 的近期锚点；[AffordanceVLA](https://arxiv.org/abs/2606.06155) 继续支撑单向 `B→P→T→C→Action` mask。SARM、MARIE、GPL、LIAM、ROMA、MAMBA 等较早工作只保留为历史来源，不能承担 V7.3 的通过证明。

## 16. 现在按什么顺序做

一句话：**第 1 步已经按 signal-first 路线决策完成；现在先把数据单元、公平基础模型和 ARB-B-core 做成稳定候选，再逐级训练 BP/BT/BPT，最后集中完成 M4/M5 与 residual 机制消融。**

### 16.1 最小可行执行清单

1. **归档第 1 步（已完成）。** 原始严格状态和后续 signal-first 状态全部保留；详细数字只在独立结果文档维护，主路线不再重复展开。
2. **冻结 V7.3 模块合同。** 固定 base commit、TeamTemporalSample schema、数据 receipt、seed、sample cursor、参数预算、训练预算、停止规则、Validation/Confirmation 和兄弟分支。
3. **实现 TeamTemporalSample 并完成 F0/F1。** 正确组织同一 episode、同一时刻、同一 ego 的 16 步合法历史、动作目标和只作监督的 sidecar；episode 切换必须 reset。
4. **训练公平 B0-H。** 与社会路线使用相同历史、参数预算和动作目标，先确认基础动作能力达到第 5.2 节门槛。
5. **设计并训练 ARB-B-core。** 只启用 B→C→action，继承 ARB schema、direct residual、zero-init、可靠度回退、置换和 stale/reset 约束；不复活完整旧 B 或 query 默认融合。
6. **逐级执行 BP、BT、BPT。** 每条路线从共同 base 独立训练，先过前一模块的漏斗再开下一模块，禁止 checkpoint 串行继承。
7. **完成闭环 Selection 与 Confirmation。** 纸面 NRMSE 只作早期筛选；最终候选必须用闭环成功率、合作指标和 W10 配对确认。
8. **对稳定候选执行后置机制审计。** M4 做 partner-change，M5 做失败根因，嵌套 residual 对照隔离 ARB 相对 hidden-only 的净增量。

### 16.2 当前仍然禁止什么

- 不能再次生成、打开或重评已经一次性完成的 R4-C 密封 test；
- 不能继续旧 192 维 B/B_hat 路线，也不能用 R2 的历史正数覆盖 R3；
- 不能把 `26.05%` 写成 ARB 语义收益、闭环成功率或因果合作提升；
- 不能把 hidden-only、time-only、row-shuffle 和同阶段 shuffle 的反证从报告中删除；
- 不能在正式训练途中搜索 token 数、memory 类型、fusion、loss、历史长度、label schema 或挑最好 seed；
- 不能让 B0-H、B-core、BP、BT、BPT 从彼此 checkpoint 续训后再冒充公平兄弟比较；
- 不能用外部论文收益替代本项目闭环结果，也不能在 license 未明确时复制外部源码；
- M4/M5 虽然后置，但完成前不能提出严格 ARB 独立机制或 partner-change 因果 claim。

当前路线级状态是 `COMPLETED_STEP1_SIGNAL_FIRST_MODULE_AUTHORIZED`。原始实验账本仍同时保留 `FAILED_STRICT_M3_R4_B_OBSERVABILITY_GATE` 与 `PASSED_M3_R4_C_SIGNAL_FIRST_SEALED_TEST`：前者限制归因，后者支持继续工程化。二者不冲突，也不再阻断第 2 步。
