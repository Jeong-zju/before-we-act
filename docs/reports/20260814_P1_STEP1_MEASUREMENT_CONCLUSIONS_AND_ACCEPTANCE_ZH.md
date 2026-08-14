# P1 第 1 步 Measurement 结论与验收记录

> 日期：2026-08-14
> 适用路线：P1 多机器人闭环模型技术路线 V7.3
> 活动分支：`feat/model-improvements`
> 远端实验工作树：`/workspace/fe-pc-wam`
> 远端最终实验提交：`c6a352145c15fb2d837dc910f6d991a3e3142f8f`
> 路线级状态：`COMPLETED_STEP1_SIGNAL_FIRST_MODULE_AUTHORIZED`

## 0. 这份文档负责什么

这份文档是第 1 步 Measurement 的详细结论、验收结果和证据边界。主技术路线只保留第 1 步的精简摘要，后续查询具体数字、历史失败、正式状态码和审计位置时，以本文为入口。

本文同时记录 2026-08-14 的负责人路线调整：M4 因果分叉和 M5 W10 失败归因不再阻断模块设计与训练，改为候选架构稳定后的机制消融。这个决定发生在 R4-C 结果已知以后，属于明确披露的后验研究优先级调整；它不追溯修改任何已经签发的严格实验回执。

## 1. 最终结论

第 1 步按 signal-first 研究目标完成，足以进入具体模块设计与训练：

1. 数据、字段、时序索引和可恢复状态分叉足以支撑后续实验；
2. 按专家真实物理轨迹构造的 P/T/B oracle sidecar 已通过自动审计和负责人授权复核；
3. 完整旧 192 维 B 没有显示稳定价值，停止使用；
4. 精简后的 oracle ARB 对未来 16 步动作预测有稳定正信号；
5. 只用部署合法输入得到的 `ARB_hat + direct residual` 在一次性全新密封 test 上相对 HC 改善 `26.05%`，三个 seed、六个任务全部同方向；
6. predictor 的活动关系头具有可用校准，低可靠度时可以精确回退 HC；
7. hidden-only、time-only、row-shuffle 和同阶段 shuffle 仍比候选更强，因此当前约 `26%` 的纸面动作预测收益不能归因为 ARB 关系语义本身。

因此工程结论与科学结论必须分开：

> **工程结论：`ARB_hat + zero-init direct residual + reliability fallback` 是值得进入正式模块设计和训练的可部署接口。**

> **归因结论：现有结果只证明整套栈的纸面动作预测收益；没有证明 ARB 相对 HC-hidden-only 通用 residual 的独立净增量，也没有证明闭环合作因果性。**

## 2. 路线级验收与原始实验状态

### 2.1 路线级最终裁决

负责人在完整查看 R4-C 与补充对照后作出以下后验路线裁决：

| 项目 | 当前状态 | 含义 |
|---|---|---|
| 第 1 步 Measurement | **完成** | 不再继续用前置机制消融阻断开模 |
| 第 2 步数据单元与 B0-H | **获准设计和训练** | 先建立公平的历史基础模型 |
| 第 3 步 ARB-B-core | **获准设计；完成第 2 步后训练** | 继承 ARB schema、direct residual、可靠度回退和负对照 |
| BP、BT、BPT | **按主路线逐级执行** | 各自仍需通过模块级漏斗，不能一次性全开 |
| M4 因果分叉 | **后置、非阻断** | 候选架构稳定后检查 partner-change 因果响应 |
| M5 W10 失败归因 | **后置、非阻断** | 候选架构稳定后判断合作失败占比和应用空间 |
| ARB 独立语义 claim | **未获准** | hidden/time/shuffle 对照没有被击败 |
| 闭环合作提升 claim | **未获准** | 当前结果是未来 16 步动作 NRMSE，不是闭环成功率 |

路线级状态记为：

> `COMPLETED_STEP1_SIGNAL_FIRST_MODULE_AUTHORIZED`

这个状态只表达“研究资源转入模块实现”，不覆盖下面任何原始机器回执。

### 2.2 原始实验状态账本

| 阶段 | 原始正式状态 | 当前怎样使用 |
|---|---|---|
| M1 严格版 | `FAILED_SCHEMA/REPLAY_NOT_DETERMINISTIC` | 永久保留历史；失败来自跨进程从头重放不一致 |
| M1 benchmark 放宽版 | `PASSED_M1_BENCHMARK_RELAXED` | 当前有效 M1 结论 |
| M2-R2 | `INCONCLUSIVE_M2_REQUIRES_CORRECTED_RERUN` | 检查器错误和预想阶段不符合专家真实轨迹 |
| M2-R3 | `PASSED_M2_ORACLE_LABEL_GATE` | 当前有效 M2 结论 |
| M3-R2 | `FAILED_MEASUREMENT/NO_SOCIAL_HEADROOM` | 修复前历史结果，不覆盖 R3/R4 |
| M3-R3 | train/tune 筛查失败 | 完整旧 B 路线终止，没有打开新 test |
| M3-R4 原合同 | `INCONCLUSIVE_TRAINING/CAP_REACHED` | 收敛不完整的历史运行，不用于通过判定 |
| R4 successor A1 | `PASSED_M3_R4_A1_CONFIRMED_ORACLE_UTILITY` | 支持正确 ARB 的动作相关信号 |
| R4-A2 | 已完成 | 支持 ARB 表示；否决 query superiority |
| R4-B 原严格闸门 | `FAILED_STRICT_M3_R4_B_OBSERVABILITY_GATE` | 继续限制 ARB 语义归因 |
| R4-B 探索状态 | `PROMISING_DEPLOYABLE_ARB_HAT_SIGNAL` | 支持继续工程探索 |
| R4-B signal-first 补充 | `PASSED_M3_R4_B_SIGNAL_FIRST_OWNER_AMENDMENT` | 在限制归因的前提下授权一次 R4-C |
| R4-C | `PASSED_M3_R4_C_SIGNAL_FIRST_SEALED_TEST` | 确认整套可部署栈的纸面收益可以泛化 |

## 3. M1 与 M2 验收结果

### 3.1 M1：数据与分叉可用，但不能从头复刻原轨迹

严格版读取 900 个 HDF5、284,183 帧，并对 30 条冻结 episode 做从头重放和保存状态分叉：

| 检查 | 结果 |
|---|---:|
| HDF5 哈希、schema、字段和时间对齐 | 900/900 文件、284,183 帧，0 问题 |
| 样本主键 | 284,183 个 `(task, episode_sha256, frame_index)` |
| 保存完整状态后重复相同联合动作 | 30/30 精确一致 |
| 从 episode 开头重放达到 qpos `<=1e-4` | 0/30 |
| 从头重放终局 success 与原记录一致 | 5/30 |

从头重放失败的原因边界是：现有 HDF5 没有保存完整对象位姿、速度、接触/抓取状态和可直接恢复的 `env_state`，所以 `seed + recorded action` 不是完整轨迹状态。它不等于“同一完整状态、同一动作不确定”。

负责人随后明确 benchmark 和 simulator 不修改，并把 qpos/终局复刻降为诊断项。放宽版沿用同一批 30 个 episode 从头重跑，900/900 文件、284,183 帧、30/30 完整执行和 30/30 保存状态分叉全部通过，正式状态为 `PASSED_M1_BENCHMARK_RELAXED`。

有效边界：后续允许做字段审计、样本索引和基于完整保存状态的确定性分叉；禁止把从头重放产生的偏移状态冒充原轨迹真值。

### 3.2 M2：标签必须跟随专家真实物理轨迹

M2-R2 成功采集 30 条、9,547 帧，但冻结检查器漏换历史 latch 键，且 Take Photo/Place Food 的四类预想状态在真实专家轨迹中零覆盖，因此状态为 `INCONCLUSIVE_M2_REQUIRES_CORRECTED_RERUN`。

M2-R3 不在 R2 上删项改判，而是按以下证据顺序重新冻结标签：

```text
环境 success 公式
  > 实际执行出的物理轨迹
  > planner 命令意图
  > 旧路线文字
```

R3 正式结果：

| 项目 | 结果 |
|---|---:|
| 正式成功 episode | 30；每任务 5 条 |
| 正式标签帧 | 9,494 |
| 必填字段、有限值、终态 success | 0 错误 |
| 确定性、机器人换位等变、合法状态转移 | 0 错误 |
| 歧义帧 | 7/9,494，`0.0737%` |
| 专家真实必需状态覆盖 | 每类 5 个独立 episode |
| 转移复核片段 | 121，覆盖六任务各 5 条 episode |
| predicate 一致率 | 100% |
| terminal、role、custody | 0 错误 |

复核方式必须如实披露：121 个片段由 AI 盲图检查、仿真物理证据和完整因果重放共同核对，负责人 `jeong` 明确豁免独立第二人工审核并签发 owner approval；不能描述成独立双人盲审。正式状态为 `PASSED_M2_ORACLE_LABEL_GATE`。

## 4. M3 历史结果与 R4 修订

### 4.1 M3-R2：修复前结果不用于最终方向判断

R2 使用 360 条成功轨迹并一次性打开 72 条密封 test。结果中 oracle P/B 为正，但部署预测量均为负，label-shuffle 又异常变好：

| 来源 | oracle 真值相对 HC | 合法输入预测相对 HC |
|---|---:|---:|
| P | `+15.96%` | `-4.63%` |
| T | `+1.95%` | `-8.50%` |
| B | `+13.64%` | `-8.94%` |
| PTB | `+9.28%` | `-28.01%` |

因为 predictor 未稳定、动作 checkpoint 又按 100 步 loss 选择而按 16 步判分，R2 只保留为历史诊断。

### 4.2 M3-R3：修复收敛后，完整旧 B 仍失败

R3 把 predictor 上限提高到 260 轮，按 episode 外折生成训练用 `B_hat`，动作模型直接用最终 16 步六任务等权指标选 checkpoint。predictor 验证 MSE 从 `0.10987` 降至 `0.08831`，但动作结果仍为负：

| 条件 | 相对 HC | 95% CI |
|---|---:|---:|
| oracle B | `-6.20%` | `[-8.10%, -4.31%]` |
| `B_hat` | `-13.97%` | `[-16.43%, -11.51%]` |
| `B_hat` 相对 label-shuffle | `-13.39%` | `[-16.41%, -10.54%]` |

预注册功效为 `90.56%`，oracle B 也已正常早停，因此不能再用“少训几步”解释。完整旧 192 维 B 路线到此停止。

### 4.3 R4：只保留会改变当前动作的 ARB

R4 把 B 收窄为 Action-Relevant Belief，只保留：

- contact / grasp / custody；
- handoff event；
- teammate motion state；
- blocking / collision relation；
- visibility / staleness；
- uncertainty / missingness。

明确排除 frame index、episode ID、固定机器人编号、任务完成百分比、remaining goals、完整队友未来动作和旧 192 维 B 的复制。进度属于 P，未来属于 T。

动作接口固定为：

```text
a_final = a_HC + g(reliability) * Delta_a_direct
```

HC 冻结；residual 最后一层零初始化；强制 `g_B=0` 时必须逐元素回到 HC。

## 5. R4-A1/A2：oracle ARB 有信号，direct 胜过 query

### 5.1 R4-A1 confirmed oracle utility

六任务各采 12 条全新 confirmation episode，共 72 条；24/24 个三种子正式分支全部正常早停，没有撞上 1,200 轮上限。

| 对比 | 配对改善 | 95% CI |
|---|---:|---:|
| oracle ARB 相对 HC | `+19.39%` | `[+17.91%, +20.85%]` |
| oracle ARB 相对 zero residual | `+5.27%` | `[+4.43%, +6.09%]` |
| oracle ARB 相对 input-independent noise | `+0.93%` | `[+0.47%, +1.37%]` |
| oracle ARB 相对 label-shuffle | `+7.03%` | `[+6.12%, +7.94%]` |
| oracle ARB 相对 episode-shuffle | `+5.66%` | `[+4.80%, +6.49%]` |

三个 seed 和六个任务全部同方向，正式状态为 `PASSED_M3_R4_A1_CONFIRMED_ORACLE_UTILITY`。它证明正确 ARB 内容在这个探针中具有动作相关信号，不证明部署机器人已经能估计 ARB，也不证明 ARB 胜过 hidden-only residual。

### 5.2 R4-A2 standardized 2x2

| 表示 × 融合 | 六任务 NRMSE |
|---|---:|
| ARB + direct residual | `0.04577` |
| sanitized legacy B + direct residual | `0.04620` |
| ARB + query attention | `0.05006` |
| sanitized legacy B + query attention | `0.05118` |

- ARB 表示主效应：`+1.79%`，95% CI `[+1.37%, +2.21%]`；
- query 相对 direct：`-14.14%`，95% CI `[-15.77%, -12.50%]`；
- 交互项：`+1.80%`，不足以挽回 query 的整体劣势。

因此正式模块默认继承 ARB 表示和 direct residual，不把 query-attention 作为第一版默认融合。

## 6. R4-B：`ARB_hat` 可用，但严格语义归因失败

R4-B 只使用合法当前观测和 16 步历史，训练集 ARB 全部由 episode-out-of-fold predictor 生成。

### 6.1 主结果与 predictor

| 结果 | 数值 |
|---|---:|
| `ARB_hat + direct` 相对 HC | `+25.19%`，95% CI `[+22.89%, +27.40%]` |
| 三个 seed | `+24.96% / +24.99% / +25.42%` |
| oracle direct 相对 HC | `+27.89%` |
| 相对 HC 的 oracle 收益保留率 | `90.33%` |
| 三种子中位 NRMSE | HC `0.06062`；oracle `0.04577`；`ARB_hat` `0.04749` |

36/36 个活动头在 confirmation 上的 Brier 都优于 train-incidence 常数；平均 Brier 为 `0.03691`，常数为 `0.14083`。从最低到最高可靠度四个区间，硬错误率为 `13.12% → 6.25% → 0.89% → 0.03%`。强制 gate 为 0 后与 HC 最大绝对差为 `0.0`。

### 6.2 原严格对照

下表正数表示 `ARB_hat` 更好：

| 对照 | `ARB_hat` 相对对照 | 95% CI |
|---|---:|---:|
| episode-shuffle | `+1.60%` | `[+0.31%, +2.92%]` |
| stale-8 | `+1.82%` | `[+0.78%, +2.83%]` |
| stale-16 | `+0.97%` | `[-0.72%, +2.66%]` |
| row-shuffle | `-1.08%` | `[-1.80%, -0.34%]` |
| time-only | `-2.48%` | `[-4.11%, -1.01%]` |

所以原严格状态保持 `FAILED_STRICT_M3_R4_B_OBSERVABILITY_GATE`，同时记录探索状态 `PROMISING_DEPLOYABLE_ARB_HAT_SIGNAL`。

### 6.3 hidden-only 与同阶段 shuffle 补充对照

`HC-hidden-only` 把 48 维 ARB 输入置为标准化零点、predictor 可靠度固定为 1，residual 只读取冻结 HC 的 256 维 hidden。`phase-matched row-shuffle` 只在同任务、同粗阶段内打乱 ARB 和可靠度。

| 方案 | 相对 HC | 95% CI |
|---|---:|---:|
| `ARB_hat + direct` | `+25.19%` | `[+22.89%, +27.40%]` |
| `HC-hidden-only + direct` | `+27.61%` | `[+25.43%, +29.79%]` |
| 同阶段 shuffle + direct | `+26.91%` | `[+24.70%, +28.98%]` |

hidden-only 比候选强 `3.23%`，同阶段 shuffle 比候选强 `2.19%`。所以约 `25%` 的改善不能诚实地写成“ARB 语义带来了 25%”。补充回执为 `PASSED_M3_R4_B_SIGNAL_FIRST_OWNER_AMENDMENT`，它只授权一次性 R4-C，不修改原严格失败。

## 7. R4-C 一次性密封测试

R4-C 在生成 test 前冻结代码、29 个 predictor/action checkpoint、69 个依赖工件、候选 seed 顺序和统计规则。六任务各顺序采集 12 条全新成功 episode，共 72 条、13,028 个合法评测行；test manifest 只打开一次。

### 7.1 主结果

| 结果 | 数值 |
|---|---:|
| `ARB_hat + direct` 相对 HC | `+26.05%`，95% CI `[+24.15%, +27.90%]` |
| 三个 residual seed | `+25.79% / +26.43% / +25.93%` |
| 六任务 | Camera `+6.63%`；Lift `+24.54%`；Long `+37.59%`；Photo `+19.52%`；Shoe `+29.21%`；Food `+38.83%` |
| oracle direct 相对 HC | `+28.85%`，95% CI `[+26.89%, +30.76%]` |
| 相对 HC 的 oracle 收益保留率 | `90.30%` |

新 test 上 36/36 个活动头仍优于 train-incidence 常数；平均 Brier `0.04032`，常数 `0.14171`。可靠度从低到高的错误率为 `13.90% → 7.31% → 0.90% → 0.0085%`。强制 `g_B=0` 后与 HC 最大绝对差仍为 `0.0`。

### 7.2 诊断对照

下表正数表示 `ARB_hat` 更好：

| 对照 | `ARB_hat` 相对对照 | 95% CI |
|---|---:|---:|
| episode-shuffle | `+2.11%` | `[+0.98%, +3.19%]` |
| stale-8 | `+3.04%` | `[+2.06%, +4.00%]` |
| stale-16 | `+3.10%` | `[+1.76%, +4.38%]` |
| row-shuffle | `-1.83%` | `[-2.59%, -1.11%]` |
| 同阶段 shuffle | `-2.18%` | `[-2.85%, -1.50%]` |
| time-only | `-3.22%` | `[-4.49%, -2.00%]` |
| HC-hidden-only | `-3.35%` | `[-4.03%, -2.69%]` |

正式状态为 `PASSED_M3_R4_C_SIGNAL_FIRST_SEALED_TEST`。它证明整套合法输入、可校准、可回退的 direct-residual 栈在新数据上的纸面动作预测趋势，不证明 ARB 独立机制或闭环合作收益。

## 8. 当前最合理的机制解释

现有证据支持三个不同强度的陈述：

1. **已证明充分性：** `MLP(h_t, 0)` 的通用 residual 能力足以复现并超过候选的纸面收益；
2. **已有 ARB 信号：** 正确 oracle ARB 胜过 label/episode shuffle，`ARB_hat` 胜过 episode-shuffle 和 stale 对照；
3. **未证明独立增量：** `ARB_hat` 没有胜过 hidden-only、time-only、row-shuffle 或同阶段 shuffle。

因此不能说“ARB 完全没有关系”，也不能说“26% 都来自 ARB”。当前最稳妥的解释是：

> **主要收益来自冻结 HC 上增加可训练 direct residual；ARB 的对应性和新鲜度显示了一些信号，但在现有数据、门控和融合下还没有形成超过通用 residual 的净增量。**

此外，hidden-only 读取的是 HC hidden，而不是常数。HC hidden 本身可能已经隐式编码任务阶段、视觉中的队友或团队线索，所以“通用 residual”只表示没有显式 ARB 输入，不等于完全没有社会信息。

## 9. 模块路线继承的设计约束

进入第 2 步及后续模块训练时，继承以下结论：

1. 不复活完整旧 192 维 concat B；
2. 第一版 B-core 使用 ARB schema，而不是无选择复制全部 P/T/B；
3. direct residual 是默认动作接口，query-attention 只保留为历史反方消融；
4. residual 最后一层 zero-init，强制 gate-off 必须精确回到本候选基础动作；
5. predictor 可靠度、missingness、staleness 和 episode reset 必须进入回退逻辑；
6. ARB、时间、进度、固定身份和未来信息继续保持信息卫生边界；
7. B0-H、B-core、BP、BT、BPT 必须共享数据、seed、sample cursor、action target 和训练预算；
8. 模块阶段继续保留 hidden-only、time/phase、shuffle、stale 和 gate-off 诊断，但这些不再作为开始开模的一票否决；
9. 最终 winner 仍需通过闭环 Validation/Confirmation，纸面 NRMSE 不能直接替代任务成功率。

## 10. M4/M5 的新位置

M4 和 M5 不删除，只下调为候选架构稳定后的机制审计。

后续至少保留三类问题：

1. **M4 partner-change：** 在相同初始状态下改变队友后续行为，检查可见变化出现后模型是否作出方向正确且及时的响应；
2. **M5 failure taxonomy：** 判断 W10 与最终候选的失败中有多少确属重复劳动、阻挡、争抢、错误等待、错误分工或进度误判；
3. **ARB residual 隔离：** 在同一个已冻结 hidden residual 上增加单独 ARB 分支，比较真实 ARB、ARB 置零、同阶段 shuffle 和 oracle ARB。

推荐的后置隔离结构为：

```text
a = a_base
  + Delta_a_hidden(h_t)
  + reliability * Delta_a_ARB(h_t, ARB_hat_t)
```

先冻结 `Delta_a_hidden`，再判断 ARB 分支是否提供额外收益。只有这种嵌套对照或等价设计通过，才允许提出“ARB 超过通用 residual 的独立动作增量”。

## 11. 审计入口

关键版本与远端产物：

| 阶段 | 分支 / commit | run root 或关键回执 |
|---|---|---|
| M1 放宽版 | `a6ecfa5d9dbb6e4206da4da93873244b68145dab` | `docs/experiments/ssc_v7/m1_relaxed_rerun/` |
| M2-R3 | `feat/ssc-v7-m2-oracle-sidecar` | `docs/experiments/ssc_v7/m2_r3_formal/`、`m2_r3_human_review/` |
| M3-R2 | `fa5543c6ce24fa4a5210b87a150d5255f24b4117` | `/workspace/bwa_runs/ssc-v7-social-state-cooperation-v2/measurement/m3_r2` |
| M3-R3 | `47cab7c9d8192433d00627718d2f11bc3217438e` | `measurement/m3_r3_convergence` |
| R4 successor | `acc3e818bd720d39c4109edeb2b26c3d034e046d` | `measurement/m3_r4_successor_a1_v1` |
| R4-B | `f26ff0fde4d0a0c4b9294a82d394e925aed4e25a` | `measurement/m3_r4_b_observability_v1` |
| R4-B supplement | `cc2442bcc91fd8342acc81f007aecb2cf4864cd0` | `measurement/m3_r4_b_supplement_v1` |
| R4-C | `c6a352145c15fb2d837dc910f6d991a3e3142f8f` | `measurement/m3_r4_c_sealed_test_v1` |

关键 R4 哈希：

- successor confirmation manifest：`76ecdc88001ba894e8289541d33625edca341cbe7eac8162299f367d1f66b3ad`；
- successor formal gate：`f3bb78b569f8800d42cd462d94582abc0c899eba468f1d3c6a17a0556ffc4904`；
- successor receipt：`b0209148bf7406b5cfd391a84c5a9d10906b098e717c065024445e9110e68395`；
- R4-B gate：`e0e87af7ad05b29f9ff2fda1f0be3cb09973361e473ab793fdaa44974d392f34`；
- R4-B predictor receipt：`6593ef6853a9fd10c8a0e103afdba80d3e4194651580c5d8e8190a6f6bb1e4f6`；
- R4-B final receipt：`43484fdc6a0121f7de4d952ec78daf50395a11dc8c7706661667c7d9b6b515b5`；
- supplement gate：`86a875267f46c0c31ab27b80b1d990eb0a3dc6ede4e8385c51aea8458eff55e0`；
- supplement receipt：`367653f8e55d69974c6ee4e34c837f6db2f2dadb9347ec1e9a16f08dc587d2a0`；
- R4-C gate：`9cd9649aef105241f9eef48a4aadf9a0fb8e10a3937640e21e588a462f96eb5a`；
- R4-C final receipt：`e25fe1298e5e127741972e417931714d45ca43a8cc26c99a5ee49ebc77dbdf86`。

远端实验数据和 checkpoint 继续保留在服务器；本文与 Git 中的实现、合同和回执负责长期审计，不能用后续模块结果追溯重写上述历史状态。
