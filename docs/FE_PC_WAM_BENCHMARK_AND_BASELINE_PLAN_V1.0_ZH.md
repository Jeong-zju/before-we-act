# FE-PC WAM Benchmark 与 Baseline 计划

> 更新日期：2026-07-17
>
> 当前状态：proprioceptive Joint WAM 已完成训练与正式闭环验证。关闭 fallback 的 direct
> policy 在 standard/challenge 各 500 个未见 seeds 上均为 100%，相同 seeds 的 action
> prior 也均为 100%，`policy_acceptable=true`；`joint_benefit` 尚未评测。

## 1. 当前结论

当前模型使用双机器人 22 维 proprioception 与 8 维连续动作。它以 recurrent world model
的 belief 为共享条件，通过 stateful Flow Matching 生成 8-step action chunk，每次正常执行
2 步，使用 4-step Euler solver，并按实际执行步数平移动作块作为下一次 warm start。

训练包含 action-flow warm-up 与 joint coupling 两部分，但它们现在由同一配置和同一训练
入口管理；最终 checkpoint 自包含 world model、action flow、normalization、数据划分、训练
指标与 provenance，不依赖 warm-up 中间 checkpoint。完整 world-model ensemble 与 action
prior 只作为可复现初始化资产和 baseline，不是 direct policy 的在线依赖。

现有 cooperative-stop 任务已被 action prior 与 Joint WAM 同时饱和。因此当前结果证明
Joint WAM 可直接闭环执行，但不能证明 joint world modeling 相对 frozen-backbone action
policy 或 standalone flow 改善了控制。

## 2. 当前主表

| 方法 | 用途 | 当前结果 |
|---|---|---|
| Stationary | 检查提前静止是否能利用奖励漏洞 | standard/challenge 均为 0% |
| Scripted oracle | 环境上界与任务可解性检查 | standard/challenge 均为 100% |
| Action prior | 行为克隆 baseline 与可选安全 fallback | standard/challenge 均为 100% |
| Joint WAM direct | 正式方法，验收时关闭 fallback | standard/challenge 均为 100% |
| Joint WAM with fallback | 仅报告部署表现，不参与 direct 验收 | 两套 suite 均为 100%，触发率 0% |
| Standalone Flow Matching | 判断 world model 是否带来控制收益 | 尚未评测 |

正式结果位于 `outputs/joint_wam`。两套 suite 共 1,000 个 held-out seeds，训练/评测 seed
overlap 为 0；direct action-source coverage 为 100%，fallback trigger 为 0，动作全部
finite/bounded，无 privileged-state leakage 或提前静止成功。每套保留 3 条按 seed 排序选取
的 direct/no-fallback 成功视频；本次没有失败 episode。

## 3. 公平比较规则

- 所有学习方法使用同一 train / validation / test 划分，评测 seeds 不得进入任何训练、
  蒸馏或 relabel 流程。
- direct policy 只能读取 proprioception 与过去已执行动作；privileged labels 只能作为训练
  target 或离线审计字段。
- Joint WAM、standalone flow 与 frozen-backbone 对照必须使用相同数据、action-expert 参数量、
  训练 steps、solver steps、action horizon 和控制频率。
- fallback 成功必须归到 fallback deployment，不能计入 direct policy 成功率。
- 正式验收固定 standard/challenge 各 500 个未见 seeds；diagnostic CLI 覆盖必须写入独立
  output 目录，不能覆盖正式证据。
- 视频按确定性规则选择：每套成功 seed 取最小 3 个；若有失败，全局最多取最小 3 个。

## 4. 指标与验收

控制指标包括成功率、return、response delay、coordination error、失败原因、动作来源覆盖率、
fallback trigger rate 与 P50/P95/P99 latency。世界模型指标包括多步 NRMSE、finite rollout、
constraint violation、uncertainty-error Spearman、OOD AUROC 与 event-aligned dominant-agent
accuracy。

`policy_acceptable` 的必要条件是：

- direct Joint WAM 在每套 500 episodes 上相对相同 seeds 的 action prior 回归不超过 10pp；
- direct execution rate 为 100%，fallback trigger rate 为 0；
- 动作 finite/bounded，无 privileged leakage、seed overlap 或提前静止成功；
- joint branches 确有非零参数变化和必要梯度，frozen teacher/anchor 与初始化资产保持不变；
- checkpoint 严格重载误差为 0，正式视频与 sidecar 完整。

当前所有必要条件均已通过。正式 direct/prior 成功率均为 100%，回归 0pp。

## 5. 下一步 benchmark

若目标是证明 `joint_benefit`，应先构造 action prior 未饱和的 challenge 或新任务，然后用
paired seeds 比较 Joint WAM、frozen-backbone action policy 与 standalone Flow Matching，
报告 paired effect、置信区间、world-model 指标和相同计算预算下的延迟。

若目标是进入 RoboFactory、RoboTwin、ACT、Diffusion Policy、SmolVLA 或 OpenVLA-OFT 等
外部 benchmark，必须先为目标任务加入匹配的 RGB/语言接口。当前系统没有 RGB 和语言输入，
只能称为 proprioceptive WAM，不能按 VLA 结果宣传。

## 6. 停止条件

- 新任务上的 direct policy 在小规模无 fallback 测试中低于 80%：停止全量评测，先修动作生成。
- joint coupling 缺少同动作 teacher target、shared/world/action 非零梯度或初始化资产不可变性：
  不得称为 Joint WAM。
- standalone/frozen-backbone 对照未按同预算运行：不得声称 joint training 带来控制收益。
- 单卡 smoke 无法稳定完成或 checkpoint 不能严格重载：停止扩大 benchmark。

阶段性设计与已退出路线的详细记录见
[`archive/PROPRIOCEPTIVE_WAM_TECHNICAL_PLAN_V1.0_ZH.md`](archive/PROPRIOCEPTIVE_WAM_TECHNICAL_PLAN_V1.0_ZH.md)。
