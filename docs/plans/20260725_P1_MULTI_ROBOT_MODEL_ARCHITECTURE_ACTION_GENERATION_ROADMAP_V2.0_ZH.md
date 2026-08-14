# P1 多机器人闭环模型技术路线 V7.3：Measurement 收口后的自动团队信念 PBT 协作模型

> 更新日期：2026-08-14
> 活动分支：`feat/model-improvements`
> 当前状态：**第 1 步 Measurement 按负责人后验的 signal-first 决策收口，模块设计和训练获准开始。** 第 2 步不返工原始数据和标签，而是把已经验收的数据整理成所有候选共同使用的 16 步时序样本，并训练一个“有同样历史和 residual、但没有团队信念”的强 B0-H；第 3 步按 `3-N1` 原始数据新信号、`3-N2` 完整 B-core、`3-N3` 机制归因、`3-N4` 正式验收逐级推进，随后再进入 BP、BT、BPT。
> 证据边界：R4-C 在 72 条全新成功 episode 上相对冻结 HC 改善 `26.05%`，95% CI `[24.15%, 27.90%]`，三个 seed、六个任务全正；但 hidden-only、time-only、row-shuffle 和同阶段 shuffle 仍更强。因此可声称“整套 ARB_hat + residual 栈有稳定纸面收益”，不可声称“ARB 语义独立贡献了这些收益”。
> 优先级调整：B-core 的“新信号—整体结构—普通容量”主归因放在 3-N3；M4 partner-change 和 M5 W10 失败归因保留为稳定候选后的补充因果审计，不再阻断现在的模块训练。详细结论和全部原始验收状态见[第 1 步结果文档](../reports/20260814_P1_STEP1_MEASUREMENT_CONCLUSIONS_AND_ACCEPTANCE_ZH.md)。
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
第 2 步：统一 16 步训练样本 + 公平 B0-H
  ↓
第 3 步：自动团队信念 B-core
  ├─ 3-N1：无人工标签的原始数据团队表示
  ├─ 3-N2：完整 B-core 训练充分后的积极信号
  ├─ 3-N3：整体结构与新信号的机制归因
  └─ 3-N4：冻结唯一方案并正式闭环验收
  ↓
第 4/5 步：BP / BT
  ↓
第 6 步：BPT
  ↓
联合训练 + Validation/Confirmation
  ↓
稳定候选后的 M4 / M5 补充因果审计
```

第 2 步以后的模块仍逐级走漏斗；3-N3 负责 B-core 的主机制归因，后置 M4/M5 不阻断开模，但在它们完成前不得提出更强的 partner-change 或失败根因因果结论。

文档里的几个常用词可以这样理解：

| 词 | 人话解释 |
|---|---|
| gate | 一道“通过/停止”的检查 |
| oracle | simulator 给出的真实答案，只能用于出题和判卷，不能给部署模型偷看 |
| probe | 为了验证某种信息有没有用而训练的小模型，不是最终模型 |
| sidecar | 不修改原 HDF5，另外保存的一份逐帧标签文件 |
| receipt | 用 commit、配置和 SHA256 证明“这次到底用了什么”的记录 |
| TeamTemporalSample | 一条统一训练样本：把同一任务、同一 episode、同一机器人当前时刻的 16 步过去、100 步动作答案和监督标签对齐放好 |
| residual | “动作修正支路”：基础模型先给动作，旁路再学习加上一小段修正；它本身也可能带来性能，所以必须单独控制 |
| B0-H | 不使用团队信念的公平基础模型；它拥有与 B-core 相同的 16 步历史和 residual 容量，用来测清历史与通用修正本身能做到多好 |
| belief | 模型根据当前与过去证据形成的团队状态分布，同时包含“认为是什么”和“有多不确定” |
| ARB | Action-Relevant Belief；第 1 步冻结的动作相关语义集合。第 3 步中只作探针、解释和审计，不再作为 B-core 必须人工标注的训练输入 |
| 运行分支 | 部署时真正保留的分支，只读取当前和过去的合法观测 |
| 训练教师分支 | 只在训练时读取同步多视角和未来 `0.2/0.4/0.8/1.6` 秒的四个原始观测锚点，用来教运行分支形成更完整的团队信念；部署时删除 |
| 团队信念 token | B-core 自动学习的少量潜在状态位置，包括智能体锚点和自由交互状态，不与人工 ARB 字段逐维绑定 |
| cross-attention | 一组信息主动读取另一组信息；谁是 Query 就是谁在问，谁提供 Key/Value 就是谁在回答 |
| coordination queries | 完整 PBT 阶段才启用的少量“协调员”token，负责把 B、P、T 中与下一段动作有关的信息整理后交给动作模型；B-core 单独阶段不重复增加这一层 |
| CoordinationAdapter | 完整 PBT 阶段 ACT decoder 的小型旁路；B-core 单独阶段直接用可靠度控制的 zero-init belief residual 接入动作 |
| macro | 先分别计算六个任务，再对六个任务等权平均，避免长任务支配结果 |
| paired bootstrap | 对相同 seed 的两个模型做成对重采样，用来估计结果是否稳定 |
| 95% CI | 对结果稳定范围的估计；本文要求最保守的一端仍然大于 0 |
| 统计功效 | 现有样本是否足以看出预设大小的差异；样本不够时只能说“证据不足” |
| 分叉点 | 从同一个模拟器状态复制出几条路线，只改变队友后续行为 |
| `INCONCLUSIVE` | 证据不足，既不算通过，也不能当成路线已被证明失败 |
| `INCONCLUSIVE_TRAINING_NOT_CONVERGED` | 已达到冻结训练上限但曲线仍明显改善，暂时不能判断路线有没有信号 |
| `INCONCLUSIVE_ATTRIBUTION` | 各自训练充分后的排序与相同训练量下的排序冲突，暂时不能判断收益来自哪里 |

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
| B：team belief | “根据目前能看到的证据，我认为团队现场是什么状态，而且我有多确定” | 从原始多视角轨迹自动学习智能体、物体交互、接触、运动、遮挡和关键事件的潜在状态分布 | 不要求逐维复刻人工 ARB，不直接判断任务完成百分比，不显式输出队友未来模式 |
| P：progress | “按照任务规则，现在做完了什么、还缺什么” | 已完成/未完成谓词、当前阶段、阶段内连续进度、剩余目标 | 不重新看原始图像猜角色，不复制 B 的 agent-object 关系 |
| T：teammate future | “根据当前现场和剩余任务，队友接下来可能怎么做” | 队友未来若干步的目标、角色、动作模式及每种模式的不确定性 | 不保存长期物体记忆，不重新定义任务进度 |

第 1 步的 ARB sidecar 与 `per_agent_contribution` 继续保留作探针、解释和审计，但不再要求 B-core 依靠这些人工定义字段训练。B 的主要训练信号来自原始轨迹本身：同步多视角、未来 `0.2/0.4/0.8/1.6` 秒的视觉特征、队友当前状态和状态变化。P、T 仍各自保留独立语义边界，避免 B 预先吞掉后续模块的研究问题。

### 2.3 PBT 到底做不做 cross-attention

做，但分两个阶段处理：**B-core 单独阶段直接把 B 接到动作；完整 PBT 阶段才用有方向的 cross-attention 和统一协调层。** 这样 B-core 的研究主张集中在“团队信念怎样从原始数据自动形成”，不会被一个额外协调瓶颈混淆。

完整的一次前向按下面顺序执行：

```text
合法的当前观察、最近 16 步历史、上一时刻 B
                      │
                      ▼
       B：更新带不确定性的团队潜在状态
          │
          ├──────── B-core 阶段：可靠度控制的直接 residual → action
          │
          └──────── 完整 PBT 阶段继续向下
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
  少量 coordination queries 读取 B + P + T
       只留下与下一段自身动作有关的信息
                      │
                      ▼
       ACT action queries 读取协调结果并输出动作
```

用 Transformer 的 Query/Key/Value 写法表示：

| 顺序 | 谁在问（Query） | 读取谁（Key/Value） | 输出 |
|---|---|---|---|
| 1 | B 的智能体锚点和自由交互 queries | 当前观察、16 步历史、上一时刻 B、当前 episode 的关键事件记忆 | 更新后的信念分布 `B_t=(μ_t,σ_t)` |
| 2a | B-core 的 direct belief residual | `B_t` 与动作主干 hidden | 只由 B 带来的动作修正 |
| 2b | P 的 task-predicate queries | `B_t` 和 task graph | 完整 PBT 中的 `P_t` |
| 3 | T 的 future-mode queries | `B_t` 和 `P_t` | 完整 PBT 中的 `T_t` |
| 4 | 少量 coordination queries | `B_t`、`P_t`、`T_t` | 完整 PBT 中的 `C_t` |
| 5 | 100 个 ACT action queries | 原视觉 memory，以及 `B_t` 或 `C_t` 的 zero-init 旁路 | 100 步 action chunk |

这里有一个刻意的限制：在同一次前向中，`B_t` 不再反过来读取 P 或 T。否则会形成“B 依赖 P、P 又依赖 B”的即时循环，很难判断信息从哪里来。T 预测出的未来只监督一个单独的 `future_B` 预测头，用来检查动力学是否合理；它不会在同一时刻覆盖当前 `B_t`。下一控制步到来后，B 再用真正看到的新观察更新。

B-core 的链路固定为 `raw history → B → action`；完整模型的链路固定为 `raw history → B → P → T → C → action`。同一次前向中 B 不反过来读取 P 或 T。更复杂的多轮 PBT 交互只有在单向版本通过因果干预后才允许另立路线，不能在训练途中临时加入。

### 2.4 每组张量长什么样

基础 action backbone 继续使用 `d_model=384` 和 100 个 action queries。社会状态使用同一宽度：

```text
B_mu:    [batch, n_B, 384]    团队状态的当前中心判断
B_sigma: [batch, n_B, 384]    对每个状态判断的不确定性
M_event: [batch, n_M, 384]    当前 episode 中按预测意外程度保留的关键事件
P_t: [batch, n_P, 384]    任务谓词和剩余目标 slots
T_t: [batch, n_T, 384]    队友未来模式和时间段 slots
C_t: [batch, n_C, 384]    完整 PBT 才启用的协调 slots
```

`n_B` 不再由人工 ARB 字段数决定。3-N1 先观察表示容量与预测/动作探针是否已经趋于饱和，再在 3-N2 启动前冻结一个规模；默认可以从十几个团队信念 token 起步，但不能看到闭环结果后继续搜索。B 中至少保留智能体锚点，其余位置作为自由交互状态，不强迫每个 token 对应某个手工物体类别。`n_P`、`n_T`、`n_C` 在各自模块启动前用相同原则冻结。

agent slot 不读取显式机器人 ID。ego 由自己的 local view/qpos 确认，其余 teammate slot 共享参数并使用相对角色编码；一致交换两个机器人的输入和动作目标后，B/T/C 及其动作影响必须相应交换。

### 2.5 B-core 和完整 PBT 怎样接入动作

动作接入不承担本轮的主要研究创新。第 1 步已经表明 direct residual 优于当时的 query 融合，因此 B-core 使用最短、可切断的旁路，完整 PBT 才恢复统一协调层：

```text
B-core:
action = ACT(observation, history)
       + reliability(B_t) × ZeroInit(DirectBeliefResidual(action_hidden, B_t))

完整 PBT:
action = ACT(observation, history)
       + reliability(C_t) × ZeroInit(CoordinationAdapter(action_hidden, C_t))
```

`ZeroInit` 保证新增旁路初始不破坏基础动作；强制关闭 B 或 C 时必须结构上回到同一候选的无社会信息路径。B 的不确定性直接控制 residual 强度，不再旁挂一套与 belief 无关的可靠度分类器。本项目已有的 [ARCADecoderLayer](../../vendor/stereo-core/stereo_core/stereo_decoder_variants.py) 可继续提供低秩残差骨架，但具体融合形式在 3-N2 前一次冻结，3-N2 内不同时搜索多种 adapter。

### 2.6 训练和部署时如何避免偷看答案

第 3 步把“运行时可见信息”和“训练时用于教学的信息”严格分开：

```text
运行分支：合法当前/过去输入 → 预测 B → 动作
训练教师分支：同步多视角 + t+4/8/16/32 的未来原始观测 → 更完整的 B 教师状态
训练约束：让运行分支逼近教师状态；部署时删除教师分支
```

原始 720 条轨迹的控制与图像频率都是 `20 Hz`，因此未来窗口冻结为最多 `32` 个控制步，也就是 `1.6` 秒。教师不连续读取 32 张高度重复的图像，只读取各相机在 `t+4、t+8、t+16、t+32` 的四个锚点，分别对应 `0.2、0.4、0.8、1.6` 秒，并在冻结 DINO 潜在特征空间中教学。episode 尾部不存在的锚点必须用 mask 排除，禁止复制最后一帧冒充未来。如果以后改变采样频率，应保持这四个秒级时间点并重新换算步数，不能机械沿用 4/8/16/32。

训练教师分支还可以读取同步的当前队友状态和多视角一致性证据，但不能把未来 ego 动作答案、未来队友动作、最终成功、remaining goals 或 simulator truth 直接送入运行分支。未来队友动作若被使用，只能作为训练目标，不能作为教师或运行分支输入。

人工 ARB、P、T sidecar 仍可作 probe、oracle 上限和解释性审计，但 B-core 的动作主路始终经过自动学习的 B。最终部署 checkpoint 不包含训练教师、oracle、reward model 或外部通信。到完整 PBT 阶段，仍需逐边验证 `B→P`、`B/P→T`、`PBT→C` 和 `C→action`。

### 2.7 研究锚点怎样落到这套模型里

第 15 节给出了论文、官方仓库、license 和代码边界的完整审计。这里把最重要的对应关系前移：**每篇工作到底支撑哪一块设计，以及它不能替我们证明什么。**

| V7.3 要解决的问题 | 主要近期锚点 | 具体落到本项目的设计 | 当前证据边界 |
|---|---|---|---|
| 怎样不用人工 B 标签形成动作相关状态 | [Being-H0.7，2026](https://arxiv.org/abs/2605.00078) | 采用结构匹配的运行分支和训练教师分支；未来四锚点只负责塑造潜在团队状态，部署时删除教师分支 | 它验证的是单机器人 latent world-action model，不能直接证明多机器人 team belief 有效 |
| 怎样让团队状态对遮挡和关键事件敏感 | [MemoryVLA，ICLR 2026](https://arxiv.org/abs/2508.19236) 与 [RoboMemArena/PrediMem，2026](https://arxiv.org/abs/2605.10921) | 保留短期工作状态；用下一时刻潜在预测误差选择关键事件；每个 episode 强制 reset | 两者主要研究单机器人长时记忆，不能替本项目证明预测误差选出的事件会改善双机器人动作 |
| 为什么预测潜在特征而不是未来像素 | [AHEAD，2026](https://arxiv.org/abs/2606.02486) 与 [ω-0，2026](https://arxiv.org/abs/2608.06375) | 冻结视觉主干，在紧凑特征空间预测场景和队友状态变化；不训练大型未来视频生成器 | 两者任务和数据规模不同，只支撑潜在预测这一训练形式 |
| 多机器人怎样保持可区分又不绑定固定编号 | [Gamma-World，2026](https://arxiv.org/abs/2605.28816) | 智能体锚点使用共享参数和相对角色编码，自由交互 token 形成共享团队状态；一致交换机器人时结果相应交换 | 它以生成式多智能体 world model 为主，只支撑置换对称和共享状态原则 |
| P 怎样表示“做到哪了” | [PALM，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_PALM_Progress-Aware_Policy_Learning_via_Affordance_Reasoning_for_Long-Horizon_Robotic_CVPR_2026_paper.html)、[ProcVLM，2026](https://procvlm.github.io/) 和 [ProgVLA，2026](https://arxiv.org/abs/2605.28231) | 联合学习 action 与 progress；progress 按任务谓词、程序步骤和剩余动作定义；长历史先压成少量 control-ready tokens | PALM 没有同等完整的官方实现；ProcVLM 是 progress reward/VLM，不是动作策略；ProgVLA 目前只作机制参考 |
| T 应该学习哪些合作变化 | [Sequential Asymmetric Imitation，2026](https://arxiv.org/abs/2606.16490) | 数据和反事实必须覆盖队友延迟、阶段不一致、让行和错误分工，T 不能只在正常专家轨迹上学“队友总会配合” | 它主要支撑数据与干预设计，不是可直接搬来的 T 模块源码 |
| P、B、T 为什么用单向 cross-attention | [AffordanceVLA，2026](https://arxiv.org/abs/2606.06155) | 借鉴严格单向的 block-causal attention，把它改成 `B→P→T→C→Action`；同一次前向不允许反向形成循环 | 借的是 mask 和连接原则，不搬它的 π0 权重、训练集或大规模训练栈 |
| 完整 PBT 为什么再汇总成少量 C token | [Gamma-World，2026](https://arxiv.org/abs/2605.28816) | 只在 P、B、T 同时存在时用少量共享 hub 整理三路信息；B-core 阶段由 B 直接接 action，不重复压缩 | 它不证明 ACT 上的协调层一定有效，因此 C 仍需在 BPT 中单独消融 |
| action chunk 交界处怎样不断片 | [ChainVLA，2026](https://arxiv.org/abs/2608.02326) | 下一次决策保留上一 chunk 的工作状态和未执行动作尾部，作为 B/P 的跨 chunk 连续性参考 | 工作很新且尚未完成代码审计，第一版只做消融参考 |
| B/C 怎样安全接入现有 ACT | 本仓库 [ARCADecoderLayer](../../vendor/stereo-core/stereo_core/stereo_decoder_variants.py) | 沿用已有低秩残差骨架；B-core 使用 direct belief residual，完整 PBT 使用 CoordinationAdapter，输出都 zero-init | 这是本地工程锚点，不是外部论文对 B-core/PBT 有效性的证据 |

这些锚点不是参考文献装饰，而是要变成可检查的实现和消融：Being-H0.7 对应运行/教师双分支；AHEAD、ω-0 对应潜在未来预测；MemoryVLA、PrediMem 对应短期状态和预测意外驱动的事件记忆；Gamma-World 对应机器人换位和共享团队状态；PALM/ProcVLM 对应“进度不能等于帧号”；AffordanceVLA 对应单向 attention mask；Sequential Asymmetric Imitation 对应 delay、yield、wrong-role 数据。任何一项没有通过本项目的信号实验和最终闭环，都不能只凭论文写成“已经有效”。

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
| M3-R4/ARB（历史只读） | `feat/ssc-v7-m3-r4-arb` | `measurement/m3_r4_arb/` |
| B0-H | `feat/ssc-v7-b0-history` | `b0-history/` |
| B-core 3-N1 | `feat/ssc-v7-b-core` | `b-core/n1-raw-signal/` |
| B-core 3-N2 | `feat/ssc-v7-b-core` | `b-core/n2-architecture/` |
| B-core 3-N3 | `feat/ssc-v7-b-core` | `b-core/n3-attribution/` |
| B-core 3-N4 | `feat/ssc-v7-b-core` | `b-core/n4-formal/` |
| BP | `feat/ssc-v7-bp-progress` | `bp-progress/` |
| BT | `feat/ssc-v7-bt-teammate` | `bt-teammate/` |
| BPT | `feat/ssc-v7-bpt-directed` | `bpt-directed/` |
| 汇总 | `feat/ssc-v7-integration` | `integration/` |

3-N1～3-N3 是同一条 B-core 研究路线内部的递进实验，可以在只读 receipt 完整的前提下继承上一小步的表示权重和代码产物，但这些继承结果只能用于 Discovery，不具备正式候选资格。3-N4、B0-H、BP、BT、BPT 等正式模型仍从同一个冻结 base 按各自完整 recipe 重新训练，不能把诊断 checkpoint 冒充正式初始化。表中分支已经获得路线级设计/训练资格，但仍要在启动各阶段前冻结数据边界、比较方向、seed 和停止规则。

### 3.4 已经冻结的实验规矩

用人话概括如下：

- Measurement 已按 signal-first 路线级决策完成；原始 M1/M2/M3/R4 回执继续保留，M4/M5 改为稳定候选后的机制审计；
- 运行分支只能看合法图像、自己的 qpos、固定任务文本和 16 步合法历史；训练教师和辅助目标只能按第 2.6 节读取预先冻结的原始数据字段，并在部署时删除，任何分支都不能使用最终成功、remaining goals 或虚构通信；
- Measurement 每个任务先用 4 个 seed 调试，再准备 60 个成功专家 episode。W10 先跑 20 个新 seed；如果样本仍不足，只能六个任务一起每次增加 5 个，最多增加到每任务 40 个。每个 W10 episode 最多选 24 个时刻做反事实分叉；
- B0-H、3-N4、BP、BT、BPT 等正式候选统一走 F0、4-update F1、Discovery、Validation5、120,000-update Formal 和 Validation20；3-N1～3-N3 是前置探索，不分别重复 Validation20/Confirmation50，但训练本身必须达到第 6.0 节的最低暴露和收敛平台，不能把 4-update F1 或 5,000-step smoke test 当成信号实验；
- B0-H、3-N4 正式 B-core、BP、BT、BPT 使用完全相同的统一 16 步样本、seed、48 effective batch、100 action horizon、evaluator 和正式预算；3-N1～3-N3 在各自训练充分性合同内做数据、sample cursor 和 matched-compute 比较。B0-H 同时保留 history-only 和 hidden-residual 两种读法，后者是主要强基线；
- 自动执行时不准看到结果后重抽 seed、加预算、换 evaluator 或降低门槛。用户因固定 benchmark 的已知限制明确改变研究口径时，必须建立新的只读 gate revision、保留旧结论并原样重跑，不能覆盖旧 receipt；
- BP 必须等 3-N4 正式验收 B-core 并冻结 P 合同；BT 必须等 3-N4 正式验收 B-core 并冻结 T 合同；BPT 必须等 BP、BT 均通过各自漏斗，不能同时开工后挑最好结果。

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

1. 第 1 步的 ARB schema 继续定义“哪些团队语义可能与动作有关”，但在第 3 步只作 probe、解释和审计；B-core 的团队信念必须主要从原始轨迹的多视角、时序变化和队友状态中自动学习，不依赖新增人工标注。
2. 禁止把 frame index、episode ID、固定机器人编号、任务完成百分比、remaining goals、未来队友动作、最终成功或旧 192 维 B 偷渡进运行分支。`t+4/8/16/32` 的未来观测只允许进入训练教师分支，部署时必须删除。
3. direct residual 是 B-core 第一版默认动作接口；residual 最后一层 zero-init，低可靠度和 gate-off 必须回到本候选基础动作。
4. B-core 不再额外增加一层 C；自动团队信念 B 直接修正动作。C 只在后续 P/B/T 同时存在时负责汇总三路信息。
5. B0-H、3-N4 正式 B-core、BP、BT、BPT 必须读取完全相同的 16 步样本、走相同 sample cursor、预测相同动作目标并使用相同正式预算；3-N1～3-N3 的比较组必须在各自训练充分性合同内匹配，否则差异没有可比性。
6. hidden-only、time/phase、shuffle、stale 和 gate-off 继续作为诊断；3-N1～3-N2 只要求出现跨对照的一致积极信号，完整归因集中在 3-N3。
7. 纸面未来 16 步 NRMSE 只用于筛选，最终 winner 仍必须通过闭环 Validation/Confirmation。

### 4.4 M4/M5 的新优先级

M4 和 M5 不删除，改到候选架构与训练 recipe 稳定以后统一执行：

- M4 检查 partner change 是否因果改变正确动作，以及团队信念能否在变化可见后及时响应；
- M5 对失败轨迹做时序错误分析，区分信念形成、动作生成和控制执行问题；
- 归因时至少打乱跨机器人对应关系、替换错误 teammate、删除训练教师或未来预测目标、删除关键事件记忆，并比较 no-B residual 与 B-increment。

这些实验用于判断性能信号是否来自团队信念和 B-core 整体结构。Step 1 的 ARB/oracle 只保留为补充解释探针，不再作为训练依赖或主归因手段。

## 5. 第 2 步：统一时序训练样本，并建立公平基础模型

本节现在是活动入口。这里说的“统一数据单元”不是清洗数据，更不是重做标签。M1 已经确认原始 HDF5 可用，M2 已经确认标签可用；第 2 步只负责把这些已经验收的材料整理成正式模型都能读取的同一种训练样本。

用人话说，这一步要先回答三个很朴素的问题：

1. 每个模型到底看到了哪 16 步历史；
2. 这段历史、要预测的动作和只用于判卷的 B/P/T 标签有没有对齐；
3. 不给模型 ARB，只给它相同历史和相同 residual 容量，它本身能做到多好。

只有这三个问题先固定，后面 B-core、BP、BT、BPT 的结果才可以互相比较。

### 5.1 为什么现有两个数据加载器都不能直接当正式入口

现有 W10 `NoWristFrameDataset` 每次只取一个 episode、一个机器人、一个时刻，然后预测这个机器人后续动作。它适合训练 W10，但没有返回完整的 16 步历史，也没有对齐同一时刻的 B/P/T 监督。

Measurement 的临时 loader 已经证明“从合法输入中取出 16 步历史并对齐 ARB 标签”是做得到的，但它把图像压成小块均值，再展平成独立的 MLP 行，并且只稀疏抽取少量时刻。它的用途是快速判断有没有纸面信号，不是训练最终视觉动作模型。

所以现在缺的不是新数据，而是连接两者的正式接口：保留 W10 的真实图像和动作训练方式，同时加入 Measurement 已验证过的合法历史、标签对齐和信息边界。

### 5.2 一条统一样本到底是什么

统一样本的代码名继续叫 `TeamTemporalSample`。一条样本用人话表示如下：

| 这一部分是什么 | 具体内容 | 模型能不能看到 |
|---|---|---|
| “这条样本是谁” | task、episode、当前 ego 机器人、当前时刻 `t` | 只作索引和审计；只有 canonical task text 会在下一行作为正常输入，episode ID、frame index 不喂给模型 |
| “刚才发生了什么” | `t-15` 到 `t` 的 global/local RGB 和 ego qpos，`t-16` 到 `t-1` 已经执行过的 ego action，以及 task text | 可以看到；全部来自当前或过去，不包含从 `t` 开始的动作答案 |
| “接下来该怎么动” | ego 从 `t` 开始的 100 步 commanded action 和有效位 mask | 训练目标，模型不能提前看到 |
| “原始数据教师材料” | 同步多视角、当前队友状态、`t+4/8/16/32`（20 Hz 下为 `0.2/0.4/0.8/1.6` 秒）的未来图像锚点，以及从原始轨迹自动得到的队友/场景状态变化 | 只允许 3-N1/3-N2 的训练教师分支和辅助目标读取；运行分支、动作主路和部署模型都不能看到；尾部缺失锚点只做 mask，不复制末帧 |
| “团队现场的标准答案” | 同一 episode、同一时刻的 B/P/T sidecar | 只用来计算监督损失和验收，绝不能拼进模型输入 |
| “这里是不是新一局” | history mask、episode reset、agent/padding mask | 可以看到，用来防止把上一局记忆带进下一局 |

episode 开头不足 16 步时，只能用合同中提前冻结的 padding 规则补齐，并明确提供 mask；不能为了凑满历史去读未来。切换 episode 时必须清空上一局记忆。机器人槽位交换后，相关标签、mask 和动作目标也必须一起交换。

第一版每条样本自己携带完整的 16 步窗口，B0-H 不需要在两个随机训练批次（batch）之间保存隐藏状态。接口同时保留“同一 episode 的短序列”和 reset 标记，供 B-core 以后按时间更新 B；但第 2 步不建设跨训练批次、跨 episode 的复杂长期记忆。

### 5.3 B0-H 到底在对照什么

B0-H 不是旧 W10 的复制品，也不是一个故意做弱的陪跑模型。它要回答的是：

> **如果模型拥有与 B-core 完全相同的 16 步历史、动作 backbone 和 direct residual（也就是在基础动作上再加一条可学习的修正支路），但不给它自动团队信念和训练教师信号，它能做到多好？**

第 1 步里 `HC-hidden-only + direct` 比 `ARB_hat + direct` 更强，因此这个对照不能省略。B0-H 至少保留下面两种读法：

1. **history-only：**只增加统一的 16 步历史，用来判断“多看历史”本身带来多少收益；这条先跑探索漏斗（Discovery/Validation5）诊断，不默认重复完整正式判卷；
2. **hidden-residual：**在相同历史上增加与社会路线等容量的 direct residual，但这条修正支路只读取动作主干已经算出的普通内部特征（hidden），团队信念输入恒为零，也不使用训练教师或 B/P/T 目标。这是正式 B0-H，也是后续 B-core 必须认真比较的强基线。

如果 B-core 的张量形状必须预留社会信息位置（slots），B0-H 可以放全零占位符，但这些占位符不能随样本变化，也不能携带时间、阶段或标签信息。B0-H、B-core、BP、BT、BPT 都从同一个冻结 base 独立初始化，不能从彼此 checkpoint 续训。

这里再加一条硬规则：**正式 B0-H 的效果只由闭环 rollout 成功率评判，不由未来 16 步动作的 MSE/NRMSE 评判。** 16 步误差可以帮助发现 loss 爆炸、动作尺度错误或模型完全没学会，但即使它很好看，也不能代替机器人真正执行整局任务；反过来，只要训练稳定，不能因为 16 步误差略差就提前淘汰一个闭环可能更好的 B0-H。

### 5.4 这一步明确不做什么

- 不重新采集 episode，不修改原 HDF5，不重新划分 train/validation/test；
- 不重新标注 ARB，也不因为 time-only 很强就人工改标签；
- 不人为删除正常存在的任务阶段和时间相关性，只禁止显式 frame index、未来信息和 simulator truth 泄漏；
- 不在这里做 M4 partner-change、M5 失败归因或完整 ARB 机制消融；
- 不引入跨 episode 长记忆、生成式 world model 或其他与统一样本无关的新结构；
- 不看完训练结果后再改变历史长度、padding、样本抽法或 B0-H 定义。

如果 F0/F1 发现的是索引错位、未来泄漏、episode 串线或 mask 错误，就修 loader；不能没有新证据便把问题归咎于原始轨迹或人工标注。

### 5.5 按什么顺序执行，什么叫完成

1. **先冻结合同。** 写清样本字段、哪些字段只作审计、16 步 padding、episode reset、sidecar 对齐、训练恢复后“下一批从哪里继续”的 sample cursor、seed、参数和训练预算。
2. **做 F0 人工可读检查。** 六个任务分别查看 episode 开头、中间、结尾的样本，确认历史没有越界、动作目标属于同一 ego、标签只出现在 target 中。
3. **做 F1 小训练和恢复检查。** 跑 4 个 update，验证同一个 cursor 能重现同一批样本，暂停恢复后不换数据，episode 切换不串记忆，机器人换位后所有对应字段一起换位。
4. **训练正式 B0-H。** `history-only` 先跑探索漏斗，用来解释历史收益；`hidden-residual` 走完整正式漏斗，作为主要强基线。两者都要每 5,000 updates 保存未见 episode 学习曲线，其中正式 `hidden-residual` 按第 6.0 节冻结 `U_B0H`，`history-only` 曲线只作解释；B-core 的代码和小规模冒烟检查（smoke test）可以同时进行，但在 B0-H 合格且训练充分性明确前，不签发 B-core 的信号或正式晋级结论。

数据单元通过的标准不是“loss 看起来不错”，而是同一个样本身份始终对应同一段合法历史、同一个动作目标和同一份监督标签，并且没有未来泄漏、episode 串线或不可恢复的 sample cursor。

B0-H 的正式判分来自闭环 Validation20：六个任务各运行 20 局，共 120 局。最低动作资格保持不变：总成功 `>=80/120`；Lift/Long/Photo/Shoe 合计 `>=72/80` 且每项 `>=16/20`；Camera `>=6/20`；Camera+Food `>=8/40`。达到 `>=88/120` 才能声称在原始成功数上达到 W10。未来 16 步 MSE/NRMSE 无论多好，都不能让一个未达到这些闭环门槛的 B0-H 通过。

如果数据单元检查通过但 B0-H 不合格，优先检查 history encoder、动作 backbone 和训练 recipe，不自动返工标签。如果 B0-H 合格，第 2 步完成，B-core 才有一个可信的闭环比较对象。

## 6. 第 3 步：B-core——从原始轨迹自动形成团队信念

第 3 步不再把一组人工 ARB 字段直接放大成正式模型，也不一次性完成所有训练与归因。它把同一个 B-core 研究问题按证据强度拆成四段：

```text
3-N1：原始轨迹中是否存在无需人工标签、又对动作有用的团队新信号？
  ↓ 积极信号
3-N2：完整的预测式团队信念架构能否把新信号转化为动作和闭环趋势？
  ↓ 积极信号
3-N3：趋势究竟来自新信号、整体结构，还是普通容量与时间捷径？
  ↓ 归因方向成立
3-N4：冻结唯一 recipe，从共同 base 正式训练并以闭环结果验收
```

前三段是初期研究探索，核心是尽快判断方向是否值得继续，不为尚无测量依据的问题编造精确百分比。每段都必须提前写清比较对象、希望看到的方向和什么结果会否定当前解释；但只有 3-N4 和后续正式候选使用第 12 节的闭环硬门槛。

### 6.0 N1～N3 先过“训练充分性”门禁

“探索实验”只表示不重复完整的 Validation20/Confirmation50，不表示模型只训练几步。积极信号、弱信号和无信号都必须在训练充分以后判断；训练仍在明显改善时只能写 `INCONCLUSIVE_TRAINING_NOT_CONVERGED`，不能据此淘汰路线。

当前 720 条训练轨迹共有约 22.7 万个时刻；每个时刻按两个 ego 机器人展开后约为 45.5 万条样本。六任务平衡采样器每次更新固定给每个任务 8 条样本，最长的 Long Pipeline Delivery 约有 19.5 万条 ego-time 样本，因此 `5,000` updates 只相当于该任务约 `0.2` 个暴露周期，不能承担 N1～N3 的信号裁决。训练充分性统一按下面的合同执行：

1. **最低数据暴露。** N1 的表示模型和动作探针必须分别训练至少 `25,000` updates；N2 完整 B-core 与 N3 的每个训练比较组不得早于 `max(25,000, U_B0H)` 做信号裁决。`U_B0H` 是第 2 步正式 `hidden-residual` B0-H 在未见 episode 上首次满足下述平台条件的 update，由其学习曲线事先给出，不能看 N2/N3 结果后修改；若它到 `120,000` updates 仍未形成平台，N2/N3 不得作信号裁决，先把第 2 步记为训练充分性未解决。
2. **固定频率看曲线。** 每 `5,000` updates 在同一组未见 episode 上记录训练/验证动作损失、各未来锚点损失、教师对齐、表示坍缩、gate 使用和任务级结果；不能只保存最后一个总 loss。
3. **平台条件。** 过了最低数据暴露和 warmup 以后，预注册主要验证分数的平滑值在连续三个评测点、也就是至少 `15,000` updates 内相对改善均不足 `1%`，并且该平台经过一次预注册的学习率下降后仍未被突破，同时没有某个关键任务或关键辅助目标仍持续明显改善，才算基本训练到位。这里的 `1%` 只用于判断曲线是否还在动，不是模型效果的通过数字。
4. **统一训练上限。** N1～N3 单次训练的上限统一冻结为 `120,000` updates。若到上限曲线仍明显改善，结论是训练预算不足，不能写 `NO_SIGNAL`；后续是否增加预算必须另立路线修订，不能只给落后组临时加步数。
5. **过拟合也算可诊断结果。** 若最低暴露后训练损失继续下降、未见 episode 连续三个评测点恶化，则按预注册规则选择恶化前 checkpoint，并写明 `SATURATED_BY_OVERFIT`；这表示模型已经充分拟合但不能泛化，不属于“训练步数不够”。
6. **先过门禁，再做闭环。** N2/N3 的 Validation5 只能在训练充分 checkpoint 上执行。任何 F1、5,000-step smoke test 或尚未到平台的中间 checkpoint 只检查实现和趋势，不能签发 `POSITIVE_SIGNAL`、`WEAK_SIGNAL` 或 `NO_SIGNAL`。

每个 seed 都必须独立通过训练充分性门禁，不能用一个已收敛 seed 带着两个仍在学习的 seed 投票。N1 的表示模型与动作探针分别判断平台，不能用表示损失收敛代替探针收敛。N3 则同时报告两种比较：一是每组各自训练充分后的结果，防止复杂模型因学得慢而吃亏；二是在相同 update 的 matched-compute 截面，防止把更多训练写成结构收益。两种比较方向冲突时，归因结论必须是 `INCONCLUSIVE_ATTRIBUTION`。

### 6.1 3-N1：无人工标签的原始数据团队表示

3-N1 只回答一个问题：

> **不用人工 ARB sidecar，只用轨迹天然记录的多视角、时序和机器人状态，能否提取出超出普通 history hidden、并对下一段动作有用的团队信号？**

运行侧输入只包含部署合法信息：当前与过去 16 步 global/local RGB、ego qpos、已经执行过的 ego action 和合法 task text。训练侧可以自动使用：

- 同步多视角之间的一致性；
- 当前队友 qpos 和可由相邻帧计算的状态变化；
- `t+4/8/16/32` 四个时间锚点的冻结 DINO 特征；
- 队友短期动作或状态变化作为预测目标，但不能作为运行分支或训练教师分支的输入。

这一小步只训练紧凑团队表示和轻量动作探针，不建设完整记忆系统，不接完整 ACT，不做闭环正式训练。这里的“轻量”指模型结构和验证范围较小，不表示少训练；表示模型和动作探针都必须分别通过第 6.0 节的最低暴露与平台条件。人工 ARB 只可作为事后 probe，检查潜在表示是否碰巧包含已知动作相关语义，不能进入主训练损失。

3-N1 的积极信号标准是方向性的：

- 在未见 episode 上，真实原始目标应比持久值、全零或打乱目标更可预测；
- 使用该表示的动作探针相对同容量 hidden-only 探针出现跨 seed、跨任务较一致的改善方向，而不是只靠单个任务或单个最好 seed；
- 同任务阶段打乱、time-only 或 row-shuffle 不能完整复现收益；
- 表示不能坍缩成常数，也不能只编码 episode 身份、帧号或任务阶段。

本阶段不要求某个提前臆造的最小提升百分比。若表示只能预测原始目标却不能改善动作探针，结论应写成“可建模但尚无动作价值”，不得带入 3-N2。若真实目标与打乱目标无稳定差别，停止当前目标设计；只有出现明确实现错误时才修复重跑，不能看到结果后不断增加新目标直到变正。

### 6.2 3-N2：对称预测式团队信念模型

3-N1 出现动作相关信号后，3-N2 才建设完整 B-core。它不是手工状态机，而是一个面向部分可观测多机器人协作的潜在状态模型：

```text
                              仅训练时存在
          同步多视角 + 当前队友状态 + 未来四锚点图像
                              │
                              ▼
                       训练教师分支
                              │  潜在状态对齐
                              ▼
合法当前/过去观测 ──→ 运行分支 ──→ 团队信念 B_t=(μ_t,σ_t)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              当前团队状态       关键事件记忆
                                 由预测意外程度选择
                    └─────────┬─────────┘
                              ▼
                 可靠度控制的 zero-init residual
                              ▼
                  基础动作 + 团队信念动作修正
```

**未来窗口固定为 `1.6` 秒，不是笼统的“短期”。** 原始 720 条轨迹是 `20 Hz`。只看 `0.8` 秒，很多样本仍处在同一小段运动里，容易只学到“下一帧会延续现在”；只看 `3.2` 秒，又会跨过多个接触或分工分支，而且在 Lift Barrier 和 Camera Alignment 中只有约四成时刻还存在合法远端帧。对 720 条轨迹的只读时间尺度审计还显示，六个任务的双机器人关节变化从 `0.8` 秒到 `1.6` 秒都继续明显增大，Long Pipeline Delivery 的中位变化尤其从近乎静止进入可辨运动。因此把 `t+32=1.6` 秒设为最远“后果锚点”，同时保留 `t+4/8/16` 三个较容易学习的近端锚点；这比连续预测 32 张图像更省算力，也避免模型只会短期外推或被单个远未来压垮。

3-N2 的配置和 receipt 必须原样记录下面的合同，不能在看过 Validation5 后改成另一个窗口：

```text
future_observation_offsets_steps   = [4, 8, 16, 32]
future_observation_offsets_seconds = [0.2, 0.4, 0.8, 1.6]
source_frequency_hz                = 20
maximum_future_seconds             = 1.6
tail_policy                        = mask_missing_anchor
teacher_target_space               = frozen_DINO_latent
```

架构固定为以下六个组成部分：

1. **多视角证据压缩。** 继续使用冻结 DINOv3 patch 特征，用少量可学习 query 把每个时刻的高分辨率视觉证据压成可做时序建模的 token；不训练像素级视频生成器。
2. **智能体中心表示。** 显式保留 ego/teammate 锚点，但使用共享参数和相对角色编码，不绑定左臂、右臂或固定机器人 ID；其余为自由交互 token，不逐维绑定人工物体/关系标签。
3. **因果团队状态更新。** 一个 block-causal temporal Transformer 只允许过去影响现在，用上一时刻 B、当前证据和已执行动作更新 `B_t`；episode reset 时全部清空。
4. **分布式 belief。** 每个团队 token 同时输出中心 `μ_t` 与不确定性 `σ_t`。遮挡、证据冲突或预测失败时不确定性应上升，动作旁路据此减弱。
5. **预测式关键事件记忆。** 模型先预测下一时刻潜在特征；真正观测到下一时刻后，用预测误差判断刚才是否发生值得保存的事件。只在当前 episode 内保留少量高价值事件，禁止跨 episode 污染。
6. **直接动作旁路。** B 直接通过 zero-init residual 修正 ACT，不再先压成另一组 C token。C 留到 P、B、T 同时存在的完整模型阶段。

训练目标由动作损失、运行/教师潜在状态对齐、未来潜在特征预测、队友状态变化预测、机器人交换一致性和防坍缩正则组成。四个未来锚点在各自归一化后等权汇总，并分别报告结果，不能用近端锚点的好成绩掩盖 `1.6` 秒后果锚点完全没学会。教师分支只提供训练信号，部署时完全删除。B 不预测 remaining goals、task progress 或显式 teammate future modes，避免提前吞掉 P、T 的研究空间。

模型宽度继续与现有动作主干对齐为 `d_model=384`。团队 token 数、关键事件容量和 temporal block 数在 3-N1 后根据表示饱和、显存和延迟一次冻结；文档不把未经测量的某个 token 数写成真理，也不允许在 3-N2 看到 Validation 结果后继续搜索。

3-N2 只验证完整架构是否出现值得继续的积极效果，不承担机制归因；但积极效果只能从第 6.0 节定义的训练充分 checkpoint 上判断。验收关注：

- 训练稳定、运行分支确实学到非坍缩的团队状态，教师分支删除后仍可独立推理；
- `0.2/0.4/0.8/1.6` 秒四个锚点分别判卷；最远 `1.6` 秒锚点相对持久值和打乱目标也应出现可重复的积极方向，否则只能说明模型学会局部运动延续，不能声称学到了动作后果；
- 相对正式 B0-H，动作诊断和 Validation5 的总体方向更好，多个 seed/任务不出现由单个异常点支配的假象；
- 至少一个预注册合作现象出现改善方向，例如更少重复劳动、争抢、阻挡或不必要等待，同时不能明显破坏强任务；
- 遮挡或多视角冲突时不确定性方向合理；强制 gate-off 时严格回到本候选的无 B 动作路径；
- 计算量、显存和延迟没有出现使后续正式训练明显不可行的问题。

这里仍不规定“必须提升 X%”。如果只改善离线动作误差而闭环完全无响应，只能记录为弱信号，不应直接进入正式训练；如果 Validation5 总体方向为负或强任务明显受损，则停止当前架构，先分析失败而不是直接加大模型。

### 6.3 3-N3：整体结构与新信号的机制归因

3-N3 不再发明新模型，只使用 3-N2 冻结的训练 recipe 做一个最小四组比较：

| 方案 | 3-N1 新团队信号 | B-core 时序/不确定性/事件结构 | 回答的问题 |
|---|---:|---:|---|
| B0-H | 无 | 无 | 相同历史和通用 residual 本身能做到什么 |
| 只有新信号 | 有 | 无 | 新信号直接接动作是否已经足够 |
| 只有结构 | 无 | 有 | 额外时序结构或参数本身能解释多少收益 |
| 完整 B-core | 有 | 有 | 新信号与整体结构结合后是否最好 |

“只有新信号”把 3-N1 表示直接送入同容量 residual，不使用团队状态更新、分布式不确定性和关键事件记忆。“只有结构”保留同样规模的 B-core 骨架，但移除运行/教师对齐及原始团队目标，只读取普通 history hidden。四组使用相同数据、seed、sample cursor 和总参数控制；每组都先训练到自身平台，再补充相同 update 的 matched-compute 截面，不能拿一个已收敛模型去比较另一个仍在快速学习的模型。

在四组比较之外，只保留少量直接切边：关闭 `B→action`、同阶段打乱 B、关闭运行/教师对齐、关闭事件写入，以及一致交换机器人。它们回答模型是否真的读取新信号、是否使用时序结构、是否依赖固定身份捷径。

3-N3 的积极归因标准同样以方向为主：

- 完整 B-core 相对 B0-H 继续保持 3-N2 的积极趋势；
- 完整 B-core 应优于“只有结构”，否则收益仍可由容量解释；
- 完整 B-core 应优于“只有新信号”，否则复杂 B-core 没有必要，应退化成更简单的新信号 residual；
- 切断 B 或在同阶段打乱 B 后，动作/合作收益按预期减弱，而不是基本不变；
- 机器人一致换位后，belief 和动作影响也对应换位。

探索样本较小时，不以一次置信区间是否刚好跨零作为唯一裁决；更重要的是四组排序、多个 seed/任务的方向和干预结果是否形成同一解释。若“只有新信号≈完整 B-core”，保留新信号结论并简化结构；若“只有结构≈完整 B-core”，不得声称 team belief 有效；若完整模型没有稳定优于 B0-H，第三步停止且没有正式 B-core 候选。

### 6.4 3-N4：冻结方案并正式验收

只有 3-N3 支持一个可解释的唯一方案后，才进入 3-N4。3-N4 不再改 token 数、记忆形式、融合器、loss 组合或训练数据，只负责：

1. 把 N1 的原始信号学习、N2 的完整架构和 N3 选定的简化/保留项冻结成一份不可变 recipe；
2. 从与 B0-H 相同的共同 base 重新执行完整 recipe，不加载 N1～N3 的诊断 checkpoint；
3. 对 B0-H 和必要 capacity control 匹配正式数据、sample cursor、训练更新数和总计算预算；
4. 走 Formal、Validation20、Selection 和 Confirmation50；
5. 按第 12 节以闭环成功率、合作指标、安全退化、因果切边和最终配对结果签发正式结论。

因此，N1～N3 的“通过”只表示值得继续研究，不能写成 B-core 已正式有效；只有 N4 达到闭环门槛，才能把 B-core 作为后续 BP、BT 的合格基础。

### 6.5 第 3 步始终不做什么

- 不新增人工 ARB 标注，不把人工 sidecar 作为运行分支输入或主训练依赖；
- 不生成未来像素视频，不从 720 条轨迹从头训练大型生成式 world model；
- 不把未来 ego action、未来 peer action、最终成功、remaining goals 或 simulator truth 送进运行分支；
- 不在 N1～N3 为追求正结果反复搜索 token 数、历史长度、未来窗口、目标集合、memory 类型和 seed；
- 不跨 episode 保存团队记忆；
- 不在 B-core 阶段引入 P、T 或重复的 C bottleneck；
- 不用外部论文数字、人工 ARB probe 准确率或未来 16 步 NRMSE 代替最终闭环验收。

## 7. 第 4 步：BP——让进度读取 belief

BP 只有在 3-N4 正式验收 B-core 后才启动。它从共同 base 按冻结的 B-core recipe 重新训练 B 和 P，不从 3-N4 checkpoint 续训。

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

BT 也只有在 3-N4 正式验收 B-core 后才启动，并从共同 base 按冻结 recipe 重新训练。在这条受控路线中没有真实 P，T 只读取 B；这条路线用来回答：“知道当前团队状态以后，继续预测队友未来是否还有新增价值？”

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

其中 `L_belief` 沿用 N2 冻结的自动学习配方，包括运行分支与训练教师分支的潜在状态对齐、未来潜在变化预测、不确定性和机器人换位一致性；它不要求人工 ARB 标签。

这些 loss 的权重和是否逐步升高，必须在 Discovery 前冻结。可以在同一次训练中先让结构化 loss warm up，再逐渐增加 action loss，但不能保存一个模块 checkpoint 后把它当作下一路线的初始化。

BPT 只有同时满足以下条件才有 winner 资格：

- 不损害 W10 已经很强的四个 protected tasks；
- 在 Camera Alignment、Place Food 或预注册 cooperation 指标上产生真实收益；
- `B→P`、`B/P→T`、`PBT→C`、`C→action` 的因果 gate 全部通过；
- 相比 `BPT-flat` 的优势来自协调机制，而不是更多参数或不同历史输入；
- 单 checkpoint 直接输出动作，不调用 W10、oracle、teacher、reward model 或外部通信；
- 延迟、显存、memory reset 和非法动作满足部署要求。

如果 BP 或 BT 通过而完整 BPT 失败，保留相应部分模型的科学结论，但不能宣称 PBT 交融成功。模块越多不自动代表模型越好。

## 11. 探索小步与正式候选使用不同强度的漏斗

完整漏斗仍保留，但不再要求 3-N1～3-N3 各自重复 Validation20 和 Confirmation50。这里节省的是大规模闭环判卷和正式候选流程，不是把训练截断在尚未学会的位置：N1～N3 必须先通过第 6.0 节的训练充分性门禁，探索阶段才有资格发现和解释积极信号，正式阶段再给最终结论。

```text
F0 静态检查
  ↓
F1 真实数据集成检查
  ├─ N1～N3 探索分支
  │    ↓
  │  最低数据暴露 → 训练到平台 → 方向/归因检查
  │    ↓
  │  N2/N3 必要时做 Validation5
  │    ↓
  │  只签发研究状态
  │
  └─ N4 和其他正式候选
       ↓
     Discovery 稳定性筛选 → Validation5
       ↓
     Formal 120,000 updates → Validation20
       ↓
     Selection → Confirmation50
```

每一层做什么：

| 阶段 | 目的 | 失败后怎么办 |
|---|---|---|
| F0 | 检查 shape、mask、参数路径、输入白名单、无未来泄漏 | 修实现，不开始训练 |
| F1 | 用真实 HDF5 跑 forward/backward/optimizer/save/resume；它只证明代码能跑 | 修集成，不进入 Discovery；不得从 F1 判断模型效果 |
| Discovery | N1～N3 负责最低暴露、训练平台、方向和反证；正式路线只作进入 Formal 前的稳定性筛选 | N1～N3 未到平台只能记训练未收敛；正式路线异常则不进入 Formal |
| Validation5 | 用固定小规模闭环检查积极信号是否能触及真实动作和合作行为 | 总体方向为负或强任务明显受损时不进入 Formal |
| Formal | 完成 120k updates，保存完整 receipt | 训练异常则按同一 manifest 恢复 |
| Validation20 | 计算正式六任务成绩和 cooperation 指标 | 不合格就没有 winner 资格 |
| Confirmation50 | 临时 winner 与 W10 在每任务 50 个新 seed 上配对比较 | 非劣下界不通过则无最终 winner |

各阶段实际走到哪里：

| 阶段 | 使用的漏斗 | 能签发什么结论 |
|---|---|---|
| 3-N1 | F0/F1 + 表示模型与动作探针分别训练到平台 | 原始数据中是否存在动作相关团队新信号 |
| 3-N2 | F0/F1 + 完整 B-core 训练到平台 + Validation5 | 完整 B-core 是否出现值得继续的动作/闭环趋势 |
| 3-N3 | 四组分别训练到平台 + matched-compute 截面 + 必要的 Validation5 | 新信号、整体结构和容量解释的相对方向 |
| 3-N4 | 完整正式漏斗 | B-core 是否正式合格 |
| B0-H、BP、BT、BPT 正式候选 | 完整正式漏斗 | 对应模块是否具备正式候选资格 |

N1～N3 的 execution prompt 必须预注册比较对象、预期方向、禁止项、训练充分性和停止条件，但不要求在没有先验测量依据时编造“必须提升 X%”一类效果数字。探索验收先看曲线是否已经训练充分，再看方向是否跨 seed/任务较一致、是否胜过关键对照、干预后是否按解释退化、是否存在明显安全或强任务副作用。只有 N4 和后续正式候选按第 12 节的闭环数字签发最终结论。

## 12. 最终怎么判断模型合不合格

本节只适用于 3-N4 以及 B0-H、BP、BT、BPT 等正式候选，不反向要求 3-N1～3-N3 用小样本提前达到这些数字。

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

- 同一版 `TeamTemporalSample` 字段、16 步 padding、history mask 和 episode reset；
- 数据 receipt；
- sample cursor；
- 随机种子；
- 训练更新数和 effective batch；
- optimizer/scheduler policy；
- action horizon 和 temporal ensemble；
- Validation5/20/50 seeds；
- evaluator、max steps 和成功条件。

新增模块带来的参数必须通过缩小 common width 抵消，或者增加同参数但不含真实社会信息的 capacity control。不能把“参数更多”写成“社会状态更有效”。

其中主要基础比较是 B0-H `hidden-residual`：它拥有相同历史和同容量 direct residual，但不使用团队信念、训练教师或原始团队辅助目标。它的效果以及正式 B-core 相对它的效果都以相同配对 seed 的闭环成功率为主，16 步 MSE/NRMSE 只作训练诊断。B-core 如果只胜过旧 W10、却没有在闭环中胜过这个强基线，只能说明新训练栈整体更强，不能说明自动团队信念提供了额外收益。

3-N1～3-N3 属于同一 B-core 路线内部的研究递进，可以为了节约探索时间继承上一小步的表示权重和代码产物，但必须在 receipt 中明确记录来源，且这些 checkpoint 不得参加正式排名。3-N4 必须从共同 base 重新执行包含 N1 预训练在内的完整冻结 recipe；四组正式/机制对照还要匹配原始数据使用量、总更新数和主要参数容量，避免把额外预训练算成“belief 结构收益”。

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

B-core 至少使用 `B3-N1-RAW-SIGNAL`、`B3-N2-ARCHITECTURE`、`B3-N3-ATTRIBUTION`、`B3-N4-FORMAL` 四个互不覆盖的 stage ID。N1～N3 只有通过训练充分性门禁后才能写 `POSITIVE_SIGNAL`、`WEAK_SIGNAL` 或 `NO_SIGNAL`；未收敛必须写 `INCONCLUSIVE_TRAINING_NOT_CONVERGED`，归因冲突写 `INCONCLUSIVE_ATTRIBUTION`，不能复用 `PASSED_FORMAL`。只有 N4 可以签发正式通过/失败。

N1～N3 每个 seed 和比较组还必须生成独立的 `training_sufficiency.json`，至少记录最低暴露是否满足、全部 5,000-update 评测点、平滑方法、最近三个相对改善量、学习率下降前后曲线、平台 checkpoint、训练上限、`U_B0H` 和最终训练充分性状态。acceptance 脚本缺少这份 receipt 时必须拒绝签发任何信号结论。

monitor 至少要显示：当前路线、branch/commit、GPU、PID、stage、update、最低暴露进度、各验证曲线斜率、平台计数、loss、ETA、checkpoint、Validation、因果 gate、显存、温度、OOM/NaN/stale 和 acceptance 状态。

R11/R12 的 runbook 只能参考工程结构，不能作为活动入口。第 1 步详细状态见独立结果文档；当前活动入口是把已经验收的数据整理成统一 `TeamTemporalSample`，训练 history-only 与 hidden-residual 两种 B0-H，再按 N1～N4 建立自动团队信念 B-core。M4/M5 保留为正式稳定候选后的补充因果审计。

## 15. 研究依据、反证和开源采用边界

### 15.1 先区分论文事实、本项目推断和路线决定

本节只使用论文主页、arXiv/CVF 等原始论文页面和作者官方仓库。外部工作回答“哪些机制值得测”，本项目实验才回答“它在六任务上是否有效”。证据分三档：

1. **可迁移代码锚点**：官方代码和明确 license 都存在；完成 commit、文件 hash、NOTICE 与符号映射 receipt 后，才可迁移小型机制；
2. **机制锚点**：论文有实验，但仓库不完整、license 不清楚或模型规模差异过大；只允许独立实现思想；
3. **反证/边界证据**：结果提醒我们某机制并不普遍有效，必须把失败模式写成对照，不能只摘正面数字。

### 15.2 为什么把路线修成自动团队信念

| 外部原始证据 | 论文/仓库实际说明了什么 | 对本项目的可证伪推断 | 因而写入的路线决定 |
|---|---|---|---|
| [Being-H0.7，2026](https://arxiv.org/abs/2605.00078) | 用可部署 prior 与训练期 future-informed posterior 对齐潜在推理状态，推理时删除 posterior 且不生成未来视频 | 未来原始观测可以只在训练期塑造动作相关 latent，而不成为部署输入 | 3-N1/N2 使用运行/教师双分支；未来只进教师，B 主路不依赖人工 ARB |
| [AHEAD，2026](https://arxiv.org/abs/2606.02486)、[ω-0，2026](https://arxiv.org/abs/2608.06375) | 都把未来预测放在紧凑视觉特征而非完整像素空间 | 720 条轨迹更适合训练轻量潜在预测，不适合从头训练大型视频生成器 | B-core 预测未来 DINO latent 和队友状态变化，不生成未来像素 |
| [DLPWM，2025](https://arxiv.org/abs/2511.06136) | 无监督对象中心表示虽能做好重建和预测，但多物体交互中的 latent drift 会让下游策略弱于普通 world model | “看起来可解释的对象槽”不自动等于更好的控制状态 | B-core 只固定智能体锚点，其余使用自由交互 token；不强迫每个 token 对应人工物体类别，N3 检查表示漂移和动作使用 |
| [RoboMemArena/PrediMem，2026](https://arxiv.org/abs/2605.10921) | 用 recent/keyframe memory 与训练期 predictive coding 提高对关键状态转移的敏感性 | 无人工事件标签时，预测误差可作为“值得记住”的候选信号 | 3-N2 用下一时刻 latent 预测误差驱动 episode 内关键事件写入，3-N3 单独关闭验证 |
| [GuidedVLA，RSS 2026](https://arxiv.org/abs/2605.12369) 与[官方仓库](https://github.com/GuidedVLA/GuidedVLA) | 用专门 attention heads 学 object/geometry/skill，并通过 zero-init control branch 加到主路 | 旧 B 负收益可能部分来自“整份状态直接污染动作”，小残差更容易测清增量 | N2 冻结基础动作主干，通过 zero-init direct belief residual 注入 B；`g_B=0` 必须精确回退 B0-H |
| [Action QFormer，2026](https://arxiv.org/abs/2607.14635) | instruction-conditioned action queries 把继承的多模态信息重组为 action-facing representation，并减少上游被动作监督大范围改写 | 动作自己提问可能优于把 192 维 B 原样拼接 | 该推断已在 successor 2×2 中实测；本项目小探针上 query 输给 direct `14.14%`，所以 N2 默认采用 direct belief residual |
| [Event-VLA，2026](https://arxiv.org/abs/2606.29384) | action queries 经 gated cross-attention 选择性融合 event tokens | 新模态/新状态不必直接混入主干，门控选择值得小规模验证 | selective fusion 只作反方消融保留；当前本项目负结果优先于外部论文，N2 默认采用 direct belief residual |
| [LangForce，ICML 2026](https://arxiv.org/abs/2601.15197) | 同时建模无语言 prior 与有语言 posterior，以条件 PMI 抑制视觉捷径 | 与其期待网络“自己使用 B”，不如显式比较无 B 主路和有 B 增量 | N2 保留无 B 主路与 B 增量的双分支接口；N3 用 matched/shuffle/stale 和 no-B 对照验证增量，不先引入复杂 PMI loss |
| [RoboMME，ICML 2026](https://arxiv.org/abs/2603.04639)、[policy learning](https://github.com/RoboMME/robomme_policy_learning) 与 [benchmark](https://github.com/RoboMME/robomme_benchmark) | 16 个 temporal/spatial/object/procedural memory 任务、14 种 π0.5 memory variants；表示效果高度依赖任务 | “加 memory 就会好”不成立，结构必须服从具体遮挡和时序问题 | N2 只冻结一种短期因果状态+关键事件结构；N3 用 structure-only、关闭 event write 和 stale/reset 控制判断它是否必要 |
| [RoboMME-Interference，2026](https://arxiv.org/abs/2606.22338) | perceptual memory 在无干扰时受益，但随无关 session 增加而持续衰减；检索相关演示可恢复 | 长记忆会污染，不只是遗忘 | episode 强制 reset，增加 previous-episode、stale 和无关历史干扰测试；需要时先检索再写入 |
| [MemoryVLA，ICLR 2026](https://arxiv.org/abs/2508.19236) 与[官方仓库](https://github.com/shihao1895/MemoryVLA) | perceptual/cognitive memory 用于长时依赖，相关性检索、门控融合和记忆合并优于朴素堆叠 | 遮挡和交接需要可更新状态，但不能从单机器人结果外推多机器人收益 | B-core 使用 episode 内短期状态、检索和 reset；不复制外部源码，不跨 episode 保存 |
| [Gamma-World，2026](https://arxiv.org/abs/2605.28816) 与[官方仓库](https://github.com/nv-tlabs/Gamma-World) | Simplex Rotary Agent Encoding 支持 permutation-symmetric agent conditioning；Sparse Hub Attention 用少量 hub 交换多主体信息 | teammate slot 不应绑定固定编号，共享团队状态应避免固定 roster 捷径 | B-core 使用共享智能体锚点和置换测试，B token 自身作为共享状态；C 只在完整 PBT 中启用，不迁移视频生成主干 |
| [CHORUS，2026](https://arxiv.org/abs/2606.12352) | 单个共享 VLA 可只凭各机器人本地观察和 robot-identifying prompt 做去中心化协作，无需推理时通信 | 显式 B 不是协作唯一道路，强直接策略是必要反方 | 保留不依赖 B 的 reactive/shared-policy 基线路线；若 B-core 失败，不把“无 B”误判成“无法协作” |
| [Sequential Asymmetric Imitation，2026](https://arxiv.org/abs/2606.16490) | staged curriculum 暴露 delay、phase mismatch、insufficient yielding 和 conflict | 仅正常专家轨迹可能缺少让 B 改变动作的关键分歧样本 | 后置 M4/M5 与后续数据修订必须覆盖 delay/yield/wrong-role；不能靠结构弥补完全缺失的数据 |
| [Embodied Interpretability，ICML 2026](https://arxiv.org/abs/2605.00321) 与[作者代码](https://github.com/robot-future/vla-explain) | ISS 用干预式 masking 估计视觉区域对动作的因果影响，NMR 衡量 nuisance 依赖 | 只看 attention 或标签准确率不能证明动作在用正确原因 | N3 把 B-off/shuffle/stale 作为主归因；ISS/NMR 只作诊断，不替代 paired action/闭环指标 |
| [VLA-ATTC，2026](https://arxiv.org/abs/2605.01194) 与 [VLAConf，2026](https://arxiv.org/abs/2605.29605) | 前者用不确定性 clutch 切换额外推理，后者以轻量 head 做单次前向 confidence | 不确定性有用的前提是被校准，并且有明确的 fallback 行为 | N2 用 B 的不确定性衰减 direct belief residual；主动观察/候选动作 critic 属于以后独立路线，不能用来挽救 B-core 负结果 |

上表故意同时保留支持和反对证据。最强的反对意见有三个：RoboMME 说明 memory design 是 task-dependent；无监督对象中心状态可能发生 latent drift；CHORUS 说明无需显式 B 也可能合作。因此 V7.3 的判断不是“自动 team belief 一定成功”，而是“原始数据潜在预测、智能体对称和短期记忆的组合值得按 N1～N3 逐级寻找信号，再由 N4 闭环裁决”。

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
| [AffordanceVLA](https://github.com/Skywalker-yqz/AffordanceVLA) | `7689e423fc264a16ce9a662dd10e4b0470066f98` | MIT；含 model/training/attention mask | 3-N4 正式验收 B-core 后才可参考后续 B→P→T block-causal mask |
| [ProcVLM](https://github.com/RUCKBReasoning/ProcVLM) | `377523a31f05bab9c0db5ac8b9edfa7b7f03968a` | 当前 HEAD 未发现顶层 LICENSE | 仅供以后 P 标签/进度审计，不进入 B-core N1～N3 |

这张表仍不是源码迁移 receipt。真正复制任何符号前，必须再冻结 commit、逐文件 hash、LICENSE/NOTICE/SPDX、依赖/权重/数据条款和“外部符号→本项目符号”映射。论文声称“will release code”不等于本轮已核验到可迁移代码；[Action QFormer](https://arxiv.org/abs/2607.14635)、[Event-VLA](https://arxiv.org/abs/2606.29384)、[CHORUS](https://arxiv.org/abs/2606.12352) 和 [VLA-ATTC](https://arxiv.org/abs/2605.01194) 本轮均按机制锚点处理。

### 15.4 不进入 B-core N1～N3 的东西

- 不引入跨 episode 长记忆、像素级生成式 world model 或测试时未来视频 rollout；先证明轻量潜在团队状态有动作价值；
- 不引入 P、T、progress critic、显式 teammate future modes 或多轮 PBT 互读；它们会让 B-core 的新信号与结构来源重新混在一起；
- 不换 π0/π0.5/OpenVLA，不迁移外部大模型 checkpoint；仍用冻结 DINOv3 和现有 ACT 回答本项目问题；
- 不强迫自由交互 token 对齐人工对象类别，也不把 ARB probe 准确率、attention map、论文 SOTA 或单个好 seed 当作积极信号；
- 不在 N1～N3 编造没有测量依据的固定提升百分比，也不把方向性通过写成正式闭环成功；
- 不删除 B0-H/direct/reactive baseline。若 B-core 失败，应保留“原始信号可用但结构无效”或“无需显式 B 也可合作”的结论，而不是无限加深 B。

[PALM](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_PALM_Progress-Aware_Policy_Learning_via_Affordance_Reasoning_for_Long-Horizon_Robotic_CVPR_2026_paper.html)、[ProcVLM](https://procvlm.github.io/)、[ProgVLA](https://arxiv.org/abs/2605.28231) 继续作为以后 P 的近期锚点；[AffordanceVLA](https://arxiv.org/abs/2606.06155) 继续支撑单向 `B→P→T→C→Action` mask。SARM、MARIE、GPL、LIAM、ROMA、MAMBA 等较早工作只保留为历史来源，不能承担 V7.3 的通过证明。

## 16. 现在按什么顺序做

一句话：**第 1 步已经按 signal-first 路线决策完成；现在先完成统一 16 步样本和强 B0-H，再按 N1 新信号、N2 完整架构、N3 机制归因、N4 正式闭环的顺序，用原始轨迹建立不依赖新增人工标签的 B-core。**

### 16.1 最小可行执行清单

1. **归档第 1 步（已完成）。** 原始严格状态和后续 signal-first 状态全部保留；详细数字只在独立结果文档维护，主路线不再重复展开。
2. **冻结 V7.3 模块合同。** 固定 base commit、统一样本里什么能给运行分支看、什么只能给训练教师或判卷，以及 16 步 padding、episode reset、数据 receipt、seed、sample cursor、参数预算、训练预算、停止规则、Validation/Confirmation 和兄弟分支。
3. **实现 TeamTemporalSample 并完成 F0/F1。** 正确组织同一 episode、同一时刻、同一 ego 的 16 步合法历史、100 步动作目标、训练教师所需原始字段和只作审计的 sidecar；检查 episode 开头/中间/结尾、暂停恢复、机器人换位和 episode 切换，确保未来材料不会进入运行分支。
4. **训练两种公平 B0-H。** history-only 先跑探索漏斗，说明历史本身的价值；hidden-residual 在相同历史上使用同容量 direct residual、但团队信念恒为零且不使用训练教师，并走完整正式漏斗，作为主要强基线。两者都记录训练平台并冻结 `U_B0H`；正式资格按第 5.5 节的闭环 Validation20 判断，16 步 MSE/NRMSE 只作训练诊断，不能代替成功率。
5. **执行 3-N1 原始数据新信号。** 只训练团队表示和动作探针；真实目标、打乱目标、hidden-only 与 time/phase 对照形成一致积极方向后才进入 N2。
6. **执行 3-N2 完整 B-core。** 冻结运行/教师双分支、智能体对称、因果团队状态、分布式不确定性、关键事件记忆和 direct belief residual；先达到最低数据暴露并训练到平台，再用 Validation5 寻找动作和闭环积极信号。
7. **执行 3-N3 机制归因。** 比较 B0-H、只有新信号、只有结构、完整 B-core，并做最少必要切边；确认收益不能由普通容量或时间捷径解释。
8. **执行 3-N4 正式验收。** 冻结唯一 recipe，从共同 base 完整重训，走 Formal、Validation20 和 Confirmation50；只有这里使用第 12 节闭环硬门槛签发 B-core 结论。
9. **逐级执行 BP、BT、BPT。** 每条路线从共同 base 独立训练，先过前一模块的正式漏斗再开下一模块，禁止正式 checkpoint 串行继承。
10. **对稳定候选执行补充因果审计。** M4 做 partner-change，M5 做失败根因；N3 已完成 B-core 主机制归因，M4/M5 进一步限定合作因果 claim。

### 16.2 当前仍然禁止什么

- 不能再次生成、打开或重评已经一次性完成的 R4-C 密封 test；
- 不能继续旧 192 维 B/B_hat 路线，也不能用 R2 的历史正数覆盖 R3；
- 不能把 `26.05%` 写成 ARB 语义收益、闭环成功率或因果合作提升；
- 不能把 hidden-only、time-only、row-shuffle 和同阶段 shuffle 的反证从报告中删除；
- 不能把第 2 步扩大成没有新证据支持的重新采集、数据清洗或重新标注工程；
- 不能把 episode ID、frame index、未来动作、未来标签或 simulator truth 混进统一样本的运行分支；`t+4/8/16/32` 的未来图像锚点只允许进入训练教师并在部署时删除；
- 不能用未来 16 步 MSE/NRMSE 给 B0-H 签发通过结论，也不能用它代替闭环成功率比较 B0-H 与 B-core；
- 不能在 N1～N3 看到结果后反复搜索 token 数、memory 类型、fusion、目标集合、历史长度或最好 seed；N4 及后续正式训练更不得修改冻结 recipe；
- 不能让 B0-H、B-core、BP、BT、BPT 从彼此 checkpoint 续训后再冒充公平兄弟比较；
- 不能用外部论文收益替代本项目闭环结果，也不能在 license 未明确时复制外部源码；
- 不能把 N1～N3 的 `POSITIVE_SIGNAL` 写成正式 B-core 通过；
- M4/M5 虽然后置，但完成前不能提出严格 partner-change 或合作根因因果 claim。

当前路线级状态是 `COMPLETED_STEP1_SIGNAL_FIRST_MODULE_AUTHORIZED`。原始实验账本仍同时保留 `FAILED_STRICT_M3_R4_B_OBSERVABILITY_GATE` 与 `PASSED_M3_R4_C_SIGNAL_FIRST_SEALED_TEST`：前者限制归因，后者支持继续工程化。二者不冲突，也不再阻断第 2 步。
