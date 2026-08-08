# P1 多机器人模型架构与动作生成技术路线 V5.1（R13/R14 回退重定版）

> 状态：CURRENT / RESET
> 更新日期：2026-08-08
> 当前唯一分支：feat/model-improvements
> 完整历史归档：[V2.0 全量历史原稿](../archive/20260725_P1_MULTI_ROBOT_MODEL_ARCHITECTURE_ACTION_GENERATION_ROADMAP_V2.0_FULL_HISTORY_ZH.md)
> 归档原稿 SHA-256：8458d28fe3c4ba6235348bd042ecfca560df2031878da9bdc29d48e4c882dc3c

## 0. 文档定位

本文件是回退后的当前事实源。旧 R13/R14 的完整实验、日志、哈希和失败证据只保留在历史归档与 Git 历史中，不再属于活动架构。

新路线遵循三个原则：

1. 先恢复可执行、可复算的闭环基线，再训练候选。
2. 优先修复 observation、数据分布和失败恢复，再更换动作生成架构。
3. 所有重要判断必须由独立 closed-loop seeds 证伪；loss、offline score 和工程测试不能代替成功率。

## 1. 回退结论

### 1.1 当前活动系统

旧 R13 和旧 R14 已从活动代码中完整撤销：

- 移除 R13 TD-MPC2 world model、world window、训练评估、配置、上游代码、验收和运行脚本。
- 移除 R14 world-guided planner、decision evaluator、配置、验收和运行脚本。
- 将 contracts.py、audit_no_full_repo_dependency.py、r11_runtime.py、verify_upstream_source.py 精确恢复到 W12 终态。
- R13/R14 的全部非文档路径与 W12 终态 commit 8b90d9e 逐文件一致。

当前活动结构为：

| 层级 | 当前定义 | 状态 |
| --- | --- | --- |
| Observation | RGB、qpos、executed-action history | 活动 |
| W11 belief | V-JEPA2 belief/predictor 架构 | 需要重建权重 |
| W12 action | high-resolution ACT/action generator | 当前基线架构；需要重建权重 |
| 旧 W13 world | TD-MPC2 latent world | 已回退，不在活动代码 |
| 旧 R14 decision | world-guided planner | 已回退，不在活动代码 |
| R15 observation/action patch | role-query、bit-exact view dedup、phase-balanced continuation | 保留代码，尚无正式胜出证据 |

R15 与 R13/R14 没有 import 依赖，也没有修改同一批 R13/R14 文件，因此本次保留用户已选择的 observation/action 实现。它默认视为实验性能力，不自动成为新基线。

### 1.2 回退验证

- Python compile：PASSED。
- 全部 R15 shell syntax：PASSED。
- tests/before_we_act：110 passed，10 个既有 Transformer warning。
- Git diff check：PASSED。
- 回退使用普通新增提交保存，不重写 Git 历史。

## 2. 历史证据与关闭边界

历史 W12 正式 Gate20：

| 任务 | W12 历史成功数 |
| --- | ---: |
| lift_barrier | 20/20 |
| camera_alignment | 14/20 |
| three_robots_stack_cube | 3/20 |
| long_pipeline_delivery | 20/20 |
| take_photo | 20/20 |
| 合计 | 77/100 |

这些数字是历史参考线，不是当前可执行 checkpoint 的结果。历史 checkpoint 已全部删除，必须重新训练并复算。

旧阶段关闭结论：

| 阶段 | 历史结果 | 回退后的处理 |
| --- | --- | --- |
| R13 P0 TD-MPC2 | 工程门通过，但 off-path，动作 bit-exact | 代码撤销；不继承 W13 名称 |
| R14 P0/P1/P2/P3 | 77/77/75/76，均未严格高于 77 | 代码撤销；四路永久关闭 |
| phase-balanced | formal Stack 1/20 | 结果保留，路线淘汰 |
| robust/world-reactive | formal Stack 0/20 | 结果保留，路线淘汰 |
| phase-routed | discovery 1/20 vs 1/20 | 结果保留，路线淘汰 |
| role-query | discovery/validation 通过，formal 2/20 | 实现保留，不能声明提升 |
| role-query + view-dedup | discovery 1/11 后中止 | 实现保留，无质量结论 |

禁止把工程通过、人工合并或 protected task 复用描述成 R13/R14 的性能提升。

## 3. 根因判断

### 3.1 为什么四个任务看起来高、Stack 很低

历史 R12/R14 评测中，四个 protected tasks 直接复用冻结 W12 输出，合计 74/80。它们并不是 R13/R14 新组件重新执行后获得的泛化成绩。

Stack 是唯一真正调用候选逻辑的任务，因此它暴露了系统的实际改进能力：

- W12 formal 终态链约为 B=15、A=6、C=3。
- 成功通常约 400 步完成；失败常运行到 800 步，表现为早期小误差累积且缺少恢复。
- global、agent_0、agent_1、agent_2 在已审计 Stack 样本中 bit-exact，原多视角信息实际退化为重复单视角。
- DINO spatial bridge 约 6×8 token，而 cube half-size 约 0.02、最终垂直容差约 ±0.005。
- success-only BC 覆盖专家轨迹，却很少覆盖学生偏离后的恢复状态。
- 多条方法在 discovery/validation 小幅胜出、formal 回落，说明收益被 seed 方差和分布外状态吞没。

[判断] 当前最可能的瓶颈排序为：

1. 失败状态与恢复监督不足；
2. 观测视角退化及空间精度不足；
3. 长链 action chunk 的误差累积；
4. 动作分布建模能力；
5. 高层 world/decision 表达。

这个排序可能错的主要原因是：重新生成真实不同视角后，视觉几何可能一跃成为第一瓶颈。新路线用单变量分支和独立闭环 seeds 来区分。

## 4. 新的阶段命名

旧 R13/R14 标识永久保留给已关闭历史，避免新结果与旧结论混淆。

| 新阶段 | 名称 | 目标 |
| --- | --- | --- |
| R13N | Baseline & Observation Reset | 重建 W11/W12 基线，冻结 observation/data 事实 |
| R14N | Causal Policy Tournament | 四卡独立验证数据、几何、生成头、恢复控制 |
| R15N | Winner Composition & Confirmation | 只组合已独立胜出的轴，并做 50-seed 确认 |
| PIVOT-N | Task Portfolio Reset | Stack 持续失败时，从零建立新任务组合 |

## 5. R13N：基线与观测重置

### 5.1 R13N-0 权重重建

历史权重为零，第一步不是训练改进模型，而是重建可审计基线：

1. 固定 feat/model-improvements commit。
2. 固定数据 repo、revision、split、样本数、camera keys、action codec 和 normalization receipt。
3. 重建 W11 belief checkpoint。
4. 重建 W12 ACT checkpoint，不启用 R15 patch。
5. strict load 后在五任务上重新运行 paired Gate20。
6. 生成新的 B_reset 指标、seed-level JSON、视频和 checkpoint manifest。

历史 77/100 只作为外部参考。B_reset 是新实验的可执行对照；如果 B_reset 显著低于 77，先调查复现偏差，不允许直接把较低分数当作容易超越的新门槛。

### 5.2 R13N-1 observation 审计

对每个任务至少抽样 50 个 episode、每个 episode 的起始/中段/终态：

- 逐相机 pixel SHA-256、max-abs、PSNR 和 active mask；
- intrinsic、extrinsic、look_at、分辨率、裁剪和时间戳；
- 数据集与 evaluator 的相机定义是否一致；
- agent observation 是否真的对应不同机器人；
- padding view 是否在进入 encoder 前被 mask；
- RGB、qpos 和 action 是否时间对齐。

冻结结构化 observation_audit.json。只有以下两种合法路线：

- 真多视角：像素与外参确实不同，保留 view/role identity。
- 诚实单视角：bit-exact 重复视角只保留一路，不再伪装成多视角。

### 5.3 R13N-2 数据重置

为 Stack 生成两个彼此分离的数据增量：

1. Geometry set：真实不同的 global/agent cameras，或单视角高分辨率 crop；不得只改 view embedding。
2. Recovery set：从 B_reset 失败轨迹的首次偏离点开始，由合法 expert/planner 给出纠正动作，覆盖 B 未放稳、A 滑移、C 对齐、释放失败和中间层坍塌。

每条 recovery 样本记录：

- source rollout、seed、首次偏离 step；
- 介入前 observation；
- expert correction 和恢复后是否重新进入成功轨迹；
- cube stage；
- 是否使用训练期 privileged label；
- policy 输入中是否完全移除 privileged state。

R13N 通过条件：

- B_reset 五任务 Gate20 完整；
- observation audit 无未解释的 camera/time alignment 异常；
- Geometry/Recovery 数据 receipt 完整；
- train/validation/formal seeds 互不重叠；
- 数据和缓存检查不会覆盖历史产物。

## 6. R14N：四卡因果模型锦标赛

四个分支从同一个 R13N base commit 创建，共享 B_reset、原始数据和冻结增量数据；每个分支只改变一个主轴。

### 6.1 GPU0 / A：Recovery-DAgger ACT

目标：验证“训练分布缺少失败恢复”是否是主因。

- 保持 W12 ACT 架构和 observation 不变。
- 加入冻结 Recovery set。
- success expert、recovery prefix、recovery continuation 分层采样。
- 对严重 OOD 状态提高采样权重，但不改变动作损失定义。
- 记录 policy rollout 到 expert intervention 的 paired 改善。

参考 [HumanCompatibleAI DAgger 实现](https://github.com/HumanCompatibleAI/imitation/blob/master/docs/algorithms/dagger.rst)。只移植 dataset aggregation、round identity 和 intervention receipt，不引入其环境或 policy 框架。

建议分支：exp/r14n-a-recovery-dagger-act。

### 6.2 GPU1 / B：Geometry-Aware ACT

目标：验证“空间信息不足”是否是主因。

- 动作头和训练数据量保持与 B_reset 一致。
- 若审计确认真多视角，使用相机外参编码和 role-conditioned queries。
- 若审计确认单视角，使用 exact dedup，并取消重复 view embedding。
- 提升 object-region token 分辨率。
- 增加训练期 cube center、层级和相对位姿辅助头；辅助标签不得进入部署输入。
- 正式输出仍只有原 action codec。

建议分支：exp/r14n-b-geometry-act。

### 6.3 GPU2 / C：RGB Diffusion Policy Head

目标：验证 ACT/CVAE action chunk 是否限制了多峰精细动作。

- observation encoder、数据、action codec 与 B_reset 对齐。
- 只替换 action head 和采样器。
- 冻结 horizon、执行频率、最大 inference latency 和 seed 规则。
- 先做 tiny-batch overfit、shape parity、finite action、latency 和 deterministic seed preflight。

优先移植 [Diffusion Policy 官方仓库](https://github.com/real-stanford/diffusion_policy)。上游代码提供视觉策略、配置、日志和 checkpoint 复现结构；本项目只取最小 action diffusion 闭包。

建议分支：exp/r14n-c-rgb-diffusion。

### 6.4 GPU3 / D：Residual Recovery Controller

目标：验证小范围闭环纠偏是否比重生成完整 action chunk 更可靠。

- B_reset 生成主 action chunk。
- residual head 只在预注册的不确定度或阶段偏离条件触发。
- residual 有严格幅度、持续步数、deadline 和 bit-exact fallback。
- 只用 Recovery set 训练纠偏，不改 B_reset 主干。
- 必须报告 intervention rate、paired wins/losses、timeout 和 fallback。

建议分支：exp/r14n-d-residual-recovery。

### 6.5 条件性 3D 路线

[3D Diffusion Policy 官方实现](https://github.com/YanjieZe/3D-Diffusion-Policy)显示其输入和数据流程依赖 point cloud/depth，并提供自定义任务 wrapper。它只有在以下条件全部满足时才能替换 R14N-D：

- 训练和 evaluator 都能合法提供同定义 depth/point cloud；
- calibration 可复现；
- 不使用 policy 不可见的 simulator state；
- 数据审计在训练前完成；
- 替换决定在看到候选结果前预注册。

否则不启动 DP3，避免把训练期 privileged 3D 信息伪装成合法部署能力。

## 7. 外部论文代码移植规则

每条路线在写模型代码前完成：

1. 记录论文、官方仓库 URL、license 和 exact upstream commit。
2. 建立 component lock、source map、adaptation card 和最小 patch。
3. 禁止复制整个上游仓库或引入其训练环境作为隐式依赖。
4. 用 parity test 证明移植核心与上游一致。
5. 用 method-separation test 证明四个候选没有混入彼此改动。
6. 对输入合法性、action effect、checkpoint strict load、fallback 和 latency 做 preflight。
7. 论文指标只能解释方法动机，不能当作本项目成功证据。

W12 的 ACT 基线可对照 [ACT 官方实现](https://github.com/tonyzhaozh/act)，但当前仓库既有 ACT 路径仍是本项目基线事实源。

## 8. 统一验收规则

### 8.1 三层 seeds

- Discovery20：独立 seeds，对 B_reset 做 identical-seed paired comparison。
- Validation20：冻结候选后使用第二组独立 seeds。
- Formal20：使用原始冻结 Gate20 seeds。
- Confirmation50：仅对 Formal PASSED winner 使用第三组新 seeds。

任何层失败都停止该候选，不创建后续运行目录。

### 8.2 PASSED 条件

设 B_reset 为重建基线：

1. candidate total 必须严格大于 max(B_reset total, 77)。
2. Stack 必须严格大于 max(B_reset Stack, 3)。
3. 五个任务中任何任务不得低于 B_reset 对应成功数。
4. protected exact reuse 必须逐 seed 验证；若实际调用候选，则不能再标记为 exact reuse。
5. Formal 胜出后，Confirmation50 必须继续保持正 paired net improvement。
6. OOM、NaN、异常重启、无心跳、非有限动作或验收 JSON 缺失均 fail closed。

同时报告两套口径：

- System score：包含合法路由/fallback 的完整系统成绩。
- Candidate coverage：候选在每个任务实际介入的回合和 step 比例。

只有 system score 提升时可以声明综合闭环提升；只有 candidate coverage 足够时才可以声明新模型跨任务泛化。

### 8.3 Stack 阶段指标

除最终 success 外，固定报告：

- cubeB_placed；
- cubeA_on_cubeB；
- cubeC_on_cubeA；
- release success；
- first deviation step；
- recovery trigger 和 recovery success；
- 400/600/800 step 生存曲线；
- seed-level paired wins/losses。

阶段指标只用于根因分析，不替代最终 success。

## 9. R15N：winner 组合

只有两个不同主轴都独立通过 Formal20，才允许创建组合分支。例如：

- Recovery data winner + Geometry winner；
- Geometry winner + Diffusion head winner；
- Recovery data winner + Residual controller winner。

禁止：

- 将两个失败方案组合后直接跳过独立归因；
- 在组合阶段重新调 formal seeds；
- 用 discovery 最佳 checkpoint 替代冻结 validation winner；
- 因组合 total 上升而忽略单任务回退。

组合必须重新运行 Discovery20、Validation20、Formal20 和 Confirmation50。

## 10. PIVOT-N：任务组合切换

满足以下任一条件才触发：

- R14N 四路均无法在 formal 使 Stack 严格大于 max(B_reset Stack, 3)；
- observation audit 证明 Stack 的合法观测无法支持所需空间精度，且无法重新生成一致的训练/评估数据；
- 两轮 Recovery set 扩充后，首次偏离和终态链均无改善。

候选来自 Hugging Face zeno-ai 组织，优先：

- pick_meat；
- place_food；
- 其他 agent observations 与 global observation 在真实像素和视角上不同的任务。

camera_alignment 已在当前五任务中，除非选择不同 repo/revision/observation 定义，否则不算新增任务。

切换前必须：

1. 冻结新的任务列表和总分分母。
2. 审计每个数据集的 revision、split、相机、动作编码和成功定义。
3. 为新任务从零训练全部 task-dependent 权重。
4. 为旧基线和新候选分别运行相同 seeds。
5. 不得在看到结果后删除失败任务。

## 11. 四卡与运行约定

| GPU | R14N 候选 | tmux 建议 |
| ---: | --- | --- |
| 0 | A Recovery-DAgger ACT | bwa-r14n-a |
| 1 | B Geometry-Aware ACT | bwa-r14n-b |
| 2 | C RGB Diffusion Policy | bwa-r14n-c |
| 3 | D Residual Recovery | bwa-r14n-d |

要求：

- 独立 branch、worktree、run root、checkpoint、日志、状态、心跳和 tmux。
- 数据集与 Hugging Face cache 只读共享。
- producer 每 20 秒写原子 heartbeat。
- monitor 从结构化 status/acceptance 读取状态，不从日志存在性猜测。
- stop 只终止 manifest 中登记的 PID 和 session。
- 不覆盖任何历史 run root。

当前服务器在清理后无历史 checkpoint，禁止直接运行旧 handoff 脚本。新的一键 launch/monitor/stop 在 R13N receipt 完成后实现。

## 12. 当前资产和验证命令

保留资产：

| 资产 | 路径 |
| --- | --- |
| 多任务数据集 | /workspace/datasets/robofactory_multitask |
| Hugging Face cache | /workspace/.cache/huggingface |
| 高分辨率 feature cache | /workspace/bwa_runs/shared/r12r4_native_full_cache_v2 |
| 历史日志和结构化结果 | /workspace/bwa_runs |
| 远程主工作树 | /workspace/fe-pc-wam |

本地验证：

    cd /home/jeong/zeno/wam/before-we-act
    git switch feat/model-improvements
    git status --short
    git rev-parse HEAD
    .venv/bin/python -m compileall -q before_we_act scripts/before_we_act
    .venv/bin/python -m pytest -q tests/before_we_act
    git diff --check

回退完整性：

    cd /home/jeong/zeno/wam/before-we-act
    git diff --name-status 8b90d9e..HEAD -- before_we_act/world_model before_we_act/planner before_we_act/data/world_windows.py before_we_act/evaluate_team_world.py before_we_act/evaluate_world_guided_decision.py before_we_act/train_team_world.py

上述命令在本次回退提交之后应只显示历史 commit 比较；工作区不应重新出现这些路径。

远程同步：

    ssh -p 10328 root@69.176.92.104
    cd /workspace/fe-pc-wam
    git switch feat/model-improvements
    git pull --ff-only origin feat/model-improvements
    git status --short
    git rev-parse HEAD

Hugging Face token 只允许通过 S0 已有环境变量或受限 secret 机制注入，不得写入代码、配置、argv、日志、文档或 Git。

## 13. 当前声明边界

可以声明：

- 旧 R13/R14 模型、planner、配置、验收和运行代码已回退；
- 当前代码回到 W12 架构基线，并保留无依赖的 R15 observation/action 实现；
- 本次回退后本地完整测试为 110 passed；
- 历史 W12 参考是 77/100，但当前权重已删除；
- 新路线先重建基线，再进行数据/几何/动作头/恢复控制的独立比较。

禁止声明：

- 当前已有可运行的 W12、W13 或 R14 checkpoint；
- role-query/view-dedup 已正式提高成功率；
- 四个 protected tasks 证明新模型跨任务泛化；
- Diffusion Policy、DP3 或 DAgger 在本项目尚未运行时已经有效；
- 通过降低 baseline 或更换 seeds 获得提升。

后续只在本文件追加冻结阶段结论和当前有效命令。长日志、逐分钟快照、失败分支细节和淘汰产物继续进入 docs/archive。
