# FE-PC WAM Benchmark 与 Baseline 计划

> 更新日期：2026-07-18
>
> 当前状态：主 pipeline 已恢复为 prior-anchored Joint WAM。关闭 fallback 的 direct policy 与 action prior 在 standard/challenge 各 500 个未见 seeds 上均为 100%，但任务已经饱和，`joint_benefit` 仍未评测。

## 1. 当前主方法

当前系统使用双机器人 22 维 proprioception 与 8 维连续动作。recurrent world model 编码历史 belief；stateful action flow 学习冻结 action prior 的 8-step rollout，并每次执行 2 步、按实际执行步数平移旧 chunk 作为 warm start。

部署动作采用冻结 prior anchor：

```text
deployed = anchor + 0.10 * (flow_generated - anchor)
```

world model 在专家动作和生成动作上提供 action-conditioned state、risk、progress 与 reward coupling。action prior 既是 flow 的冻结监督/anchor，也是独立 baseline 和可选 fallback；正式 direct 验收关闭 fallback。

训练入口包含两部分：action-flow warm-up 训练 10 个完整数据轮次；joint coupling 按 `flow_only=64 steps`、`world_heads=128 steps`、`full_joint=512 steps` 渐进解冻，共 704 steps。最终 checkpoint 自包含 world model、action flow、normalization、数据划分、训练指标与 provenance。

## 2. 当前主表

| 方法 | 用途 | 当前结果 |
|---|---|---|
| Stationary | 检查提前静止是否能利用奖励漏洞 | standard/challenge 均为 0% |
| Scripted oracle | 环境上界与任务可解性检查 | standard/challenge 均为 100% |
| Frozen WM belief + action prior | 主 baseline 与可选安全 fallback | standard/challenge 均为 100% |
| Prior-anchored Joint WAM direct | 当前主方法；验收时关闭 fallback | standard/challenge 均为 100% |
| Joint WAM with fallback | 仅报告部署表现，不参与 direct 验收 | 两套 suite 均为 100%，触发率 0% |
| Standalone Flow Matching | 判断 world coupling 是否带来控制收益 | 尚未评测 |

正式结果位于 `outputs/joint_wam`。两套 suite 共 1,000 个 held-out seeds，训练/评测 seed overlap 为 0；direct action-source coverage 为 100%，fallback trigger 为 0，动作全部 finite/bounded，无 privileged-state leakage 或提前静止成功。

## 3. 公平比较规则

- 所有学习方法使用同一 train/validation/test 划分，评测 seeds 不得进入训练、蒸馏或 relabel。
- direct policy 只能读取 proprioception 与过去已执行动作；privileged labels 只能作为训练 target 或离线审计字段。
- prior-anchored Joint WAM、standalone action flow 与 frozen-WM action prior 必须使用相同数据划分、solver、action horizon 和控制频率，并报告各自训练预算。
- prior anchor 属于 direct policy contract；运行时 fallback 成功必须单独归到 fallback deployment。
- 正式验收固定 standard/challenge 各 500 个未见 seeds；diagnostic CLI 覆盖必须写入独立 output 目录。
- 视频按确定性规则选择：每套成功 seed 取最小 3 个；若有失败，全局最多取最小 3 个。

## 4. 指标与验收

控制指标包括成功率、return、response delay、coordination error、失败原因、动作来源覆盖率、fallback trigger rate 与 P50/P95/P99 latency。世界模型指标包括多步 NRMSE、finite rollout、constraint violation 和 event-aligned dominant-agent accuracy。

`policy_acceptable` 的必要条件是：

- direct Joint WAM 在每套 500 episodes 上相对相同 seeds 的 action prior 回归不超过 10pp；
- direct execution rate 为 100%，fallback trigger rate 为 0；
- 动作 finite/bounded，无 privileged leakage、seed overlap 或提前静止成功；
- flow、world heads 与 shared history 确有非零参数变化和必要梯度，frozen teacher/anchor 与初始化资产保持不变；
- checkpoint 严格重载误差为 0，正式视频与 sidecar 完整。

## 5. 下一步 benchmark

若目标是证明 `joint_benefit`，应先构造 action prior 未饱和的 challenge 或新任务，然后用 paired seeds 至少比较：

1. prior-anchored Joint WAM；
2. standalone action flow；
3. frozen WM belief + action prior；
4. action prior only。

报告 paired effect、置信区间、动作误差、相同控制预算下延迟，并消融 `anchor_residual_scale` 与 generated-action world consistency。当前任务上的共同 100% 只能证明 pipeline 可执行，不能证明 world-action coupling 带来收益。

若目标是进入 RoboFactory、RoboTwin、ACT、Diffusion Policy、SmolVLA 或 OpenVLA-OFT 等外部 benchmark，必须先加入匹配的 RGB/语言接口。当前系统没有 RGB 和语言输入，只能称为 proprioceptive WAM。

## 6. 停止条件

- 新任务上的 direct policy 在小规模无 fallback 测试中低于 80%：停止全量评测，先修动作生成。
- prior anchor、generated-action teacher target 或 source-checkpoint immutability 被破坏：不得沿用当前 Joint WAM 结论。
- standalone/frozen-backbone 对照未按同预算运行：不得声称 joint training 带来控制收益。
- 单卡 smoke 无法稳定完成或 checkpoint 不能严格重载：停止扩大 benchmark。

阶段性设计与已退出路线的详细记录见 [`archive/PROPRIOCEPTIVE_WAM_TECHNICAL_PLAN_V1.0_ZH.md`](archive/PROPRIOCEPTIVE_WAM_TECHNICAL_PLAN_V1.0_ZH.md)。
