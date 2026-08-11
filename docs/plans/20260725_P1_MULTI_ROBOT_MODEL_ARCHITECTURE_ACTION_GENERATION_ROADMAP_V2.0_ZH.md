# P1 多机器人闭环模型技术路线 V7.0：Social-State Cooperation

> 更新日期：2026-08-11
> 活动分支：`feat/model-improvements`
> 当前状态：R11、R12 均已关闭并归档，无 winner、无 Confirmation50、无候选 checkpoint 或模型代码合入 baseline。唯一活动研究方向是从头训练的 social-state-conditioned cooperative action generation；W10 只作训练脚手架、架构复现起点和公平比较基线，不作为 checkpoint prior、teacher fallback 或部署依赖
> 活动任务：Lift Barrier、Camera Alignment、Long Pipeline Delivery、Take Photo、Pass Shoe、Place Food；不包含任何 Stack Cube 任务

## 1. 当前决定与历史终局

R11 与 R12 不再属于活动模型路线。不得从其 checkpoint 续训、warm-start、拼接机制或在原 run root 重启；历史代码、runbook、receipt 和结果只用于审计。活动路线不得用后续结果追溯修改两个阶段的失败结论。

| 阶段 | 终局 | 可确认事实 | 未发生事项 |
|---|---|---|---|
| R11 | `NO_WINNER` | A/B/D 完成真实 F1 后在 Discovery gate 失败；C 因冻结 foundation revision 403 关闭 | Validation20、Confirmation50、winner merge |
| R12 | `ARCHIVED_NO_WINNER` | Measurement future-vs-persistence `+91.604%`、K=4 oracle paired-win `87.5%`；F0/F1 通过；旧 global-clip 尝试被判 diagnostic-only，L0/L1 的诊断 Discovery 失败 | 合格 Discovery、Validation5/20、Confirmation50、winner merge；L2/L3 没有合格实验终态，修正版正式重跑未完成 |

R12 的阶段关闭是用户在证据不完整时作出的研究终止决定。它足以得出“R12 无胜者、方向归档”，但不能声称 L2/L3 已经完成实验并触发失败门槛。

历史入口：

- [R11/R12 失败技术路线完整归档](../archive/20260811_R11_R12_FAILED_TECHNICAL_ROUTES_ZH.md)
- [更早的完整路线历史](../archive/20260725_P1_MULTI_ROBOT_MODEL_ARCHITECTURE_ACTION_GENERATION_ROADMAP_V2.0_FULL_HISTORY_ZH.md)
- R11 完整执行账本：`feat/r11-four-way-integration@678e67780e6960749410ee0649ce961b10495950:docs/experiments/r11/`
- R12 完整执行账本：`feat/r12-lawam-controlled-ablation@6d45a108098fcdf5b06060d0d5860639b1513617:docs/experiments/r12/`
- 远端 R11/R12 产物继续只读保留；archive 不授权删除数据、checkpoint、日志、receipt、HF cache 或不明 session/process

## 2. 唯一公平基线：W10 六任务

### 2.1 模型和训练协议

W10 使用 `NoWristPAIRRoute`：每个机器人读取当前全局固定相机 RGB、与自己对应的固定相机 RGB 和自己的 qpos，输出本机器人 8 维动作块。视觉主干为冻结 DINOv3 ViT-B/16，保留完整 `30×40` token 网格；action horizon 为 100。模型不读取任务 ID、机器人 ID、语言、腕部图像、深度、其他机器人的状态或未来信息。

冻结训练协议为 120,000 optimizer updates、effective batch 48，每个 update 六任务各 8 个 local-agent 样本。数据、归一化、temporal ensemble、fixed seeds、max steps、成功条件和 evaluator receipt 必须在新阶段开工前重新核验。

### 2.2 Validation20 与 checkpoint

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
- 唯一活动比较基线：六任务 W10；数据 receipt 改变时先重新训练和验证等价 baseline，再冻结新数值

### 2.3 W10 在新论文中的边界

W10 只承担三种角色：复用已经验证的训练/evaluator 脚手架；提供同协议比较数字；定义 B0 的基础 action backbone 与 recipe。新路线必须从头训练，不得加载 W10 checkpoint、optimizer、RNG 或 sample cursor，不得把 W10 policy 作为 action prior、teacher fallback 或评测兜底，导出的 checkpoint 中不得保留可调用的独立 W10 handle。

W10 的代码与工程贡献必须明确归因。许可证、代码使用和 author contribution 边界需在新阶段实现前书面冻结；清晰归因解决来源问题，但不能代替方法 novelty。

## 3. 活动研究问题与最终模型

研究问题固定为：显式建模“队友正在做什么、团队知道什么、联合任务完成到哪里”，能否改善多机器人 cooperation，同时使最终策略在同一六任务协议下不弱于 W10。

最终模型是一个从头联合训练、单 checkpoint 可部署的 social-state-conditioned policy：

```text
pi(a_i | local_observation_i, task_text, proprio_i,
         joint_task_progress, teammate_latent, team_belief)
```

`joint_task_progress`、`teammate_latent` 和 `team_belief` 必须通过 cross-attention、conditioned action queries、FiLM/gating 或预注册等价路径进入 action decoder。只增加 auxiliary loss、训练期 teacher、外挂 scorer 或 inference 时不影响 action chunk 的表示不合格，也不得把 W10 外挂模块重新命名为新模型。

## 4. Social-headroom Measurement gate

大模型实现前先验证当前数据和任务是否包含可学习的 cooperation 信号。Measurement 与训练集、Validation5/20/50 隔离，固定 anchor、seed、simulator state、agent identity、history window、可见性/通信条件、排除规则和逐文件 hash；真实未来、队友隐藏状态与成功标签只作 label，不进入部署输入。

必须形成逐任务结构化报告：

1. **Schema/observability**：HDF5 是否保存时间同步的多 agent observation、action、proprio、agent ID、合法通信、任务实体状态和成功条件。
2. **Social action headroom**：控制自身 observation/task/proprio 后，oracle teammate state、joint progress 或 team belief 是否改善 action/chunk prediction 或闭环 oracle score。
3. **Cooperation sensitivity**：交换 teammate history/ID、打乱对方 action prefix、遮掉通信或改变已完成子目标时，oracle-correct ego action 是否发生一致变化。
4. **Progress validity**：progress 由成功谓词、实体状态或核验过的 subtask boundary 构造；禁止用 frame index、episode length 或未来终止时刻泄漏时间。
5. **Failure taxonomy**：量化 W10 失败中的重复劳动、互相阻挡/争抢、错误等待、错误分工和 progress 误判；区分社会问题与单臂操作/视觉感知失败。

阈值必须在读取结果前写入新阶段 `measurement_gate.json`。若 oracle social state 没有 paired headroom，或增益来自时间泄漏、agent-ID shortcut、privileged evaluator state，立即写 `FAILED_MEASUREMENT/NO_SOCIAL_HEADROOM`，停止全部社会模型训练。

## 5. 受控路线

| 路线 | 唯一机制 | 部署输入与目标 | 研究问题 |
|---|---|---|---|
| B0 / reproduction | W10-equivalent action backbone 与 recipe，从头训练 | local observation、task text、proprio | 能否不加载 W10 checkpoint 而复现强 action baseline；不是论文贡献 |
| P / progress | B0 + joint task-progress tokens | global stage、stage 内连续 progress、per-agent contribution、remaining goals | 是否减少停滞、重复劳动和错误阶段切换 |
| T / teammate | B0 + teammate intent/action latent | 合法队友历史 observation/action/communication；预测下一 chunk、目标/角色与 uncertainty | 是否减少冲突并改善分工 |
| B / belief | B0 + structured team belief | agent-object-role slot/graph、任务缺口与 uncertainty | 部分可观测下的联合 belief 是否有独立增益 |
| PT | P+T，除组合外不加新机制 | progress 与 teammate latent 联合条件化动作 | 两种单机制互补还是重复编码历史 |
| PTB | P+T+B 单一联合策略 | 三类 social tokens 共同条件化 action chunk | 是否在保持 W10 级动作能力时产生 cooperation 增益 |

预算顺序为 `Measurement -> B0 -> P -> T -> B -> PT -> PTB`。所有正式路线从同一冻结 base 建兄弟分支，不得从前一路 checkpoint warm-start 下一路。B0 未在预注册容忍区间复现 W10 时先诊断基础训练，不得加载 W10 checkpoint 补齐；P/T/B 的结构化结果未齐全前不启动 PT/PTB。

## 6. 共同控制、参考边界和源码政策

所有路线共享数据 receipt、sample cursor、随机种子、训练更新、effective batch、optimizer policy、action horizon、temporal ensemble、Validation5/20/50 和 evaluator。新增参数必须由 common width 补偿或加入等参数 capacity-control；不能把参数量增益冒充社会建模增益。

机制参考候选：

- progress：[SARM](https://arxiv.org/abs/2509.25358) 与 [LeRobot SARM](https://github.com/huggingface/lerobot/blob/main/docs/source/sarm.mdx)
- teammate representation：[GPL](https://github.com/uoe-agents/GPL)、[LIAM](https://github.com/uoe-agents/LIAM)
- cooperative world model：[MAMBA](https://arxiv.org/abs/2205.15023)
- teammate latent/ToM：[Dreaming of Others](https://arxiv.org/abs/2605.31361)，目前只作论文机制参考

2026-08-11 初步只读 recheck 只确认仓库 HEAD 存在：LeRobot `59ab28620f3f2385f808bd4bcac7fc50cf14217a`、GPL `83bf42d9b02a4a520381f37bed3cb662d86df701`、LIAM `8545b9e4237eb60ad45b7cb8ed6caec6bc4263b5`、MAMBA `2c97258f71bf1c421c40ce14fd2f7cc3fe7fe19f`。这不是源码迁移 receipt；使用前仍需冻结论文/仓库对应、commit、license、依赖、逐文件 hash、LICENSE/NOTICE/SPDX 和逐符号映射。MARL 或单机器人论文结果不得当成本项目成绩。

## 7. 因果验收与闭环非劣

除 action loss、gradient、checkpoint、resume、inference shape/range 外，每个机制必须通过对应干预：

| 干预 | 必须验证的结论 |
|---|---|
| `progress_shuffled` / stage swap | 错误进度使动作或阶段切换按预注册方向恶化 |
| `teammate_history_shuffled` | 错误队友轨迹降低动作预测或 cooperation score |
| `teammate_id_swap` / role swap | 策略对角色合理变化，排除固定 ID shortcut |
| `communication_off` / visibility mask | 信息减少时 belief uncertainty 上升，风险/等待行为校准变化 |
| `oracle_social_state` | 存在性能上界；oracle 只作诊断，不得成为 fallback |
| equal-parameter capacity control | 同等参数但无真实社会信息时不产生相同收益 |

闭环验收继续保护 Lift/Long/Photo/Shoe 四任务，并单独报告 Camera、Food、总成功数、p95 latency、峰值显存、非法动作和漏跑。若数据/评测 receipt 与当前 W10 一致：

- 正式候选最低资格仍为总成功 `>=80/120`、protected-four `>=72/80` 且单项 `>=16/20`、Camera `>=6/20`、Camera+Food `>=8/40`，并通过本路线因果 gate。
- 论文直接声称 “matches or exceeds W10” 还要求相同 Validation20 原始总成功数 `>=88/120`。
- 临时 winner 与 W10 用查看结果前冻结的每任务 50 个新 seeds 做 paired Confirmation50；报告 point estimate 与 paired bootstrap 95% CI，非劣下界不得低于 `-6.67pp`。

在读取结果前另行冻结 cooperation-specific 指标与 tie rule，至少包括完成步数、双方 idle/wait 比例、重复目标动作、互相阻挡/争抢、安全投影/碰撞风险和贡献不平衡。若只保持动作成功率却不改善预注册 cooperation 指标，只能报告“社会状态可建模但未产生合作增益”。

## 8. 正式执行入口

本路线尚未分配 stage ID、分支、run root、GPU、预算或 execution prompt，当前只完成研究问题与 fail-fast 设计。开始实现前必须：

1. 确认 W10 代码使用、署名和共同数据边界；
2. 完成 multi-agent schema 与 social-headroom Measurement 预注册；
3. 解析并冻结唯一 base commit、数据/seed receipt 和 B0 reproduction policy；
4. 为 B0/P/T/B/PT/PTB 建立独立兄弟分支、共同 integration、不可变 run manifest、F0/F1、launcher、monitor、graceful stop 和 acceptance；
5. 新建独立 run root，禁止复用或写入 R11/R12 目录。

R11/R12 的 runbook 仅用于历史审计，不再是活动入口。没有通过 Measurement 和正式执行 prompt 冻结前，不启动新训练。
