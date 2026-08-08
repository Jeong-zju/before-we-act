# P1 多机器人模型架构与动作生成技术路线 V5.0（当前状态精简版）

> 状态：CURRENT / DISTILLED
> 最近清洗：2026-08-08
> 当前唯一分支：feat/model-improvements
> 完整历史归档：[V2.0 全量历史原稿](../archive/20260725_P1_MULTI_ROBOT_MODEL_ARCHITECTURE_ACTION_GENERATION_ROADMAP_V2.0_FULL_HISTORY_ZH.md)
> 归档原稿 SHA-256：8458d28fe3c4ba6235348bd042ecfca560df2031878da9bdc29d48e4c882dc3c

## 0. 文档定位

本文件是当前实现、验收边界和后续决策的事实源。它只保留仍会影响工程或研究判断的信息。

完整的 M0–R15 时间线、逐分钟 monitor 快照、历史失败分支、旧 checkpoint 路径、重复命令和清理审计均移入上述只读归档。需要追责、复现实验过程或核对旧哈希时查阅归档；不要把归档中的历史运行状态当作当前状态。

## 1. 当前结论

1. 冻结的正式闭环基线是 W12：五任务 Gate20 为 Lift 20、Camera 14、Stack 3、LPD 20、Photo 20，合计 77/100。
2. R13 选择了 P0 TD-MPC2 latent world component，但它保持 off-path，动作输出与 W12 bit-exact，因此不产生闭环提升主张。
3. R14 四路 World-Guided Decision 的正式总分为 77、77、75、76，均未严格高于 77；正式结论是 no winner / no merge。
4. R14 后的 phase-balanced、world-reactive、phase-routed 和 role-query 路线均未在原始 formal seeds 上超过 W12。
5. exact-view-dedup 在 discovery 仅跑到 11/20 时被用户要求的资源清理中止，终态为 1/11；它没有完成 discovery、validation 或 formal，不能记为验收通过。
6. 用户随后明确指定 exact-view-dedup 为“人工选择的实现 winner”。该实现已合入 feat/model-improvements；这是一项工程选择，不追溯改写正式 Gate20 结论。
7. 历史 checkpoint 已按用户授权全部清理。当前代码、日志、结构化结果、数据集和缓存仍在，但不能直接续训或复跑依赖旧权重的 Gate20。

因此，当前允许的准确表述是：

- R13/R14 组件及后续 operator-selected 实现已进入主改进分支；
- 截至现有正式证据，尚未证明 R13+R14 使综合闭环性能高于 W12；
- 若继续研究，必须先重建可审计权重，再使用独立 seeds 和原始强门重新验证。

## 2. 当前系统定义

| 层级 | 当前定义 | 是否直接影响动作 | 证据边界 |
| --- | --- | --- | --- |
| Observation | 合法 fixed-view RGB、qpos、executed-action history | 是 | 禁止 privileged simulator state |
| W11 belief | R11-P0 V-JEPA2 predictor | 当前为 off-path 辅助 | screen winner；无新增闭环收益主张 |
| W12 action | R12-P2 high-resolution ACT/action generator | 是，冻结基线 | 正式 77/100 |
| W13 world | R13-P0 TD-MPC2 candidate-conditioned latent world | 否，off-path | action hash bit-exact |
| R14 decision | 四个候选均未通过正式门 | 否，回退 W12 | 77/77/75/76 |
| 当前附加实现 | role-conditioned spatial query + bit-exact view dedup + phase-balanced continuation | 是 | 人工合并；formal 未证明胜出 |

### 2.1 Role-conditioned spatial query

- 使用 16 个 spatial query，分成 4 个角色组，每组 4 个 query。
- 在 cross-attention 前叠加已有 agent_slot_embedding。
- Stack 的有效角色为 12 个 query；第 4 个 slot 的 4 个 query 被 mask。
- 复用已有参数，不新增独立 checkpoint tensor。

### 2.2 Exact-view-dedup

- 在添加 learned view embedding 之前比较 DINO spatial tensor。
- 只对 bit-exact 的 active view 去重，并保留第一路。
- near-equal view 不去重，避免将真实不同视角误判为重复。
- 在真实 Stack expert 样本中，原 mask [1,1,1,1,0] 变为 [1,0,0,0,0]；前四路特征两两 max-abs 为 0。

该实现修复的是“同一物理图像被多个 learned view embedding 重复计权”的结构风险。它是否提升闭环成功率仍需新的正式实验回答。

## 3. 冻结验收合同

### 3.1 Formal Gate20

五个任务各运行 20 个冻结 seed：

| 任务 | W12 成功数 | 候选规则 |
| --- | ---: | --- |
| lift_barrier | 20/20 | 必须逐 seed exact reuse |
| camera_alignment | 14/20 | 必须逐 seed exact reuse |
| three_robots_stack_cube | 3/20 | 必须严格大于 3/20 |
| long_pipeline_delivery | 20/20 | 必须逐 seed exact reuse |
| take_photo | 20/20 | 必须逐 seed exact reuse |
| 合计 | 77/100 | 必须严格大于 77/100 |

protected 四任务 74/80 是对 W12 输出的精确复用，不代表新组件在四个任务上重新训练并获得了高泛化成功率。

异常、非有限值、动作越界、低 utility 或 deadline 必须 fail closed 并回退 W12。工程 smoke、训练 loss、offline metric 或主观视频均不能替代 Gate20。

### 3.2 Discovery 与 validation

- discovery20 和 validation20 分别使用独立、预注册、identical-seed 的 W12 control。
- 两层 control 均为 1/20 时，候选必须严格大于 1/20 才能晋级。
- discovery/validation 只用于筛选；即使两层通过，也只有原始 formal Gate20 能决定最终 PASSED/FAILED。
- 不得看到结果后换 seed、覆盖目录、改变成功定义或降低阈值。

## 4. 关键阶段结论

| 阶段 | 结果 | 当前含义 |
| --- | --- | --- |
| R9/R10 | 冻结 W10；R10 无有效新 winner | 仅作更早基线和兼容参考 |
| R11 | P0 V-JEPA2 screen winner | 形成 W11；off-path |
| R12 | P2 ACT 唯一正式 winner | 形成 W12；77/100 |
| R13 | P0 TD-MPC2 工程 winner | 形成 W13；off-path，无闭环提升 |
| R14 | P0/P1/P2/P3 = 77/77/75/76 | 四路均 FAILED；no winner |
| Post-R14 | 多条 Stack 路线持续探索 | 均未形成 formal winner |
| 当前 | exact-view-dedup 人工合并 | 实现保留，质量结论未改变 |

R14 四路正式结果：

| 候选 | Lift | Camera | Stack | LPD | Photo | 总分 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P0 World-In-World | 20 | 14 | 3 | 20 | 20 | 77 | FAILED |
| P1 DINO-WM CEM | 20 | 14 | 3 | 20 | 20 | 77 | FAILED |
| P2 | 20 | 14 | 1 | 20 | 20 | 75 | FAILED |
| P3 | 20 | 14 | 2 | 20 | 20 | 76 | FAILED |

## 5. Post-R14 证据摘要

| 路线 | Discovery | Validation | Formal | 结论 |
| --- | --- | --- | --- | --- |
| phase-balanced | 2/20 vs 1/20，PASS | 2/20 vs 1/20，PASS | 1/20 vs W12 3/20 | FAILED |
| robust/world-reactive | 3/20 vs 1/20，PASS | 2/20 vs 1/20，PASS | 0/20 vs W12 3/20 | FAILED |
| phase-routed | 1/20 vs 1/20 | 未晋级 | 未运行 | FAILED at discovery |
| role-query | 2/20 vs 1/20，PASS | 3/20 vs 1/20，PASS | 2/20 vs W12 3/20 | FAILED |
| role-query + exact-view-dedup | 1/11 后中止 | 未运行 | 未运行 | STOPPED / 无质量结论 |

role-query formal 的受保护任务保持 74/80 exact，但 Stack 2/20、总分 76/100，未满足 Stack >3 和总分 >77。

exact-view-dedup 的来源提交为 bd1ae1e9f2f193c29cc851b529a36965d4df7aa0，人工合并 commit 为 03fda16。合并后的完整 before_we_act 测试为 118 passed；这只证明代码集成有效，不证明闭环提升。

## 6. ThreeRobotsStackCube 诊断

### 6.1 成功链

严格成功需要依次满足：

1. cube B 到达目标位置；
2. cube A 稳定放在 cube B 上；
3. cube C 稳定放在 cube A 上；
4. 垂直误差约束为 ±0.005，水平容差约 0.0333；
5. 机械臂释放并保持终态。

W12 formal 的终态链为 B=15、A=6、C=3，成功回合通常约 400 步完成；失败回合常持续到 800 步，说明主要问题不是完全不会操作，而是长链误差累积后缺少恢复。

### 6.2 主要原因

1. 多视角退化：global、agent_0、agent_1、agent_2 在抽样帧中 bit-exact，且 YAML look_at 相同；learned view embedding 曾把同一图像当作不同视角重复计权。
2. 空间分辨率不足：DINO bridge 的空间网格约 6×8，而 cube half-size 仅 0.02，最终堆叠容差远小于普通 pick-and-place。
3. 逐阶段乘法衰减：放 B、放 A、放 C 任一步的小误差都会降低后续成功概率。
4. 恢复数据不足：训练数据以成功 expert 为主，学生偏离轨迹后的 recovery state/label 很少。
5. seed 方差高：多条路线在 discovery/validation 小幅胜出，却在 formal seeds 回落，说明改进没有形成稳定分布外恢复能力。
6. 世界模型未闭环生效：W13 off-path；R14 planner 的介入不足或收益不稳，无法把 world signal 转化为净 paired win。

数据本身并非样本极少：Stack 有 120 条训练成功轨迹、48,892 timesteps；20 条新增 expert 的三阶段有效样本为 2,536 / 3,332 / 2,256。phase head 在 4 条完全留出轨迹上达到 raw accuracy 0.9722、authority accuracy 0.9598、boundary MAE 8.125 step。这证明阶段可观测，但不证明闭环恢复已解决。

## 7. 当前仓库与资产状态

### 7.1 Git

- 唯一保留分支：feat/model-improvements。
- GitHub 默认分支：feat/model-improvements。
- operator-selected winner 已合并。
- 当前精确 commit 请执行 git rev-parse HEAD；不要从历史日志复制旧 HEAD。

### 7.2 服务器

- 主工作树：/workspace/fe-pc-wam。
- 清理后没有训练、验证或 monitor GPU 进程。
- tmux 仅保留 0:0:bash。
- 数据集、Hugging Face cache、DINO foundation、feature cache、日志、JSON 和视频保留。
- 历史 checkpoint 与额外模型权重均已删除，最终复核计数为 0。

### 7.3 保留路径

| 资产 | 路径 |
| --- | --- |
| 多任务数据集 | /workspace/datasets/robofactory_multitask |
| Hugging Face cache | /workspace/.cache/huggingface |
| 高分辨率 feature cache | /workspace/bwa_runs/shared/r12r4_native_full_cache_v2 |
| 历史运行日志/JSON | /workspace/bwa_runs |
| 远程仓库 | /workspace/fe-pc-wam |

不要在文档、argv、日志或 Git 中写入 Hugging Face token。下载继续沿用 S0 的缓存、镜像、断点续传、离线复用和环境变量注入方式。

## 8. 当前代码入口

核心实现：

- before_we_act/action_generator/evolution.py
- before_we_act/action_generator/spatial_bridge.py
- before_we_act/train_action_generator_evolution.py
- before_we_act/evaluate_action_generator_evolution.py
- before_we_act/collect_r15_stack_expert.py
- before_we_act/train_r15_stack_expert.py

运行与验收基础设施：

- scripts/before_we_act/r15_runtime.py
- scripts/before_we_act/prepare_r15_stack_protocol.py
- scripts/before_we_act/launch_r15_stack_screens_tmux.sh
- scripts/before_we_act/monitor_r15_portfolio.sh
- scripts/before_we_act/stop_r15_stack_screens.sh
- scripts/before_we_act/handoff_r15_role_query_promotion.sh
- scripts/before_we_act/handoff_r15_role_query_view_dedup_promotion.sh

关键回归：

- tests/before_we_act/test_r15_role_query_specialist.py
- tests/before_we_act/test_r15_role_query_view_dedup.py
- tests/before_we_act/test_r15_stack_protocol.py
- tests/before_we_act/test_r15_portfolio_monitor.py
- tests/before_we_act/test_r15_runtime.py

这些脚本保留了历史编排能力，但部分 handoff/launcher 引用了已删除 checkpoint 或已删除历史分支。重建权重并更新 manifest 前，不得把它们描述为可直接恢复的训练命令。

## 9. 恢复研究的必要步骤

1. 固定 feat/model-improvements commit，并确认本地、origin、远程服务器三端一致且 clean。
2. 重新冻结 dataset repo/revision、split、camera keys、action codec、seed manifest 和缓存 receipt。
3. 决定是严格重建 W11/W12/W13 旧权重，还是把当前实现作为新一轮从零训练方案；两种证据链不得混用。
4. 对每个重建 checkpoint 做 strict load、model identity、normalization、action codec、finite output 和 fallback preflight。
5. 使用全新、不覆盖的 run root，依次运行 discovery、validation、formal。
6. monitor 必须读取 producer heartbeat 和结构化 acceptance，不得从日志存在性推断 PASSED。
7. 只有 protected=74 exact、Stack>3、total>77 同时成立，才能声明综合闭环提升。

历史权重已删除，因此当前不存在诚实的“一键继续训练”命令。任何新 launcher 必须先显式填入新 checkpoint manifest，并通过 dry-run。

## 10. 下一轮研究优先级

### 10.1 仍保留 Stack 时

优先级从高到低：

1. 修正原始 observation/camera 配置，让 global 与 agent views 在真实像素上不同，再从零训练；这是比在 bit-exact 重复图像上继续加 decision head 更直接的干预。
2. 采集失败后恢复数据，包括抓取偏差、放置滑移、遮挡、末端释放失败和中间层坍塌。
3. 使用显式 object/slot geometry 或更高空间分辨率，并在闭环中验证对毫米级终态的贡献。
4. 让 world model 真实参与可审计的 candidate scoring，同时限制 intervention rate 并保留 bit-exact fallback。
5. 参考优秀论文的开源实现时，必须保留许可证、上游 commit、最小 patch、方法分离测试和独立消融。

### 10.2 谨慎更换任务

若多条结构差异路线在独立 seeds 上仍无法使 Stack formal >3/20，可评估 Hugging Face zeno-ai 组织中的：

- camera_alignment；
- pick_meat；
- place_food；
- 其他 agent observations 与 global observation 在真实像素和视角上不同的任务。

换任务前必须预注册新任务集合和综合分母，并审计数据 revision、split、相机键、动作编码和成功定义。所有 task-dependent belief/action/world/decision 权重从零训练；不得在看到结果后删除失败任务，也不得把 protected exact reuse 当作新任务成绩。

## 11. 可直接复制的当前检查命令

本地：

    cd /home/jeong/zeno/wam/before-we-act
    git switch feat/model-improvements
    git pull --ff-only origin feat/model-improvements
    git status --short
    git rev-parse HEAD
    .venv/bin/python -m pytest -q tests/before_we_act
    git diff --check

远程：

    ssh -p 10328 root@69.176.92.104
    cd /workspace/fe-pc-wam
    git switch feat/model-improvements
    git pull --ff-only origin feat/model-improvements
    git status --short
    git rev-parse HEAD
    tmux list-windows -a -F '#{session_name}:#{window_index}:#{window_name}:#{window_active}'
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv

定向代码验证：

    cd /home/jeong/zeno/wam/before-we-act
    .venv/bin/python -m pytest -q tests/before_we_act/test_r15_role_query_specialist.py tests/before_we_act/test_r15_role_query_view_dedup.py tests/before_we_act/test_r15_stack_protocol.py tests/before_we_act/test_r15_portfolio_monitor.py tests/before_we_act/test_r15_runtime.py
    .venv/bin/python -m compileall -q before_we_act scripts/before_we_act
    bash -n scripts/before_we_act/launch_r15_stack_screens_tmux.sh
    bash -n scripts/before_we_act/monitor_r15_portfolio.sh
    bash -n scripts/before_we_act/stop_r15_stack_screens.sh

上述命令只验证当前代码和环境，不会创建训练任务。恢复训练前必须完成第 9 节的 checkpoint 重建和新 run manifest。

## 12. 结论与声明边界

可以声明：

- W12 的冻结正式基线是 77/100；
- R13-P0 和 exact-view-dedup 代码已合入当前唯一分支；
- R14 四路正式验收全部失败；
- role-query 在 discovery/validation 通过、formal 失败；
- exact-view-dedup 在中止前为 1/11，未完成验收；
- 当前无可恢复 checkpoint、无 GPU 任务。

禁止声明：

- R13 或 R14 已使综合闭环性能提升；
- exact-view-dedup 已通过 discovery、validation 或 formal；
- protected 四任务 74/80 是新模型的泛化成绩；
- 已删除 checkpoint 仍可直接恢复；
- 工程测试通过等价于机器人任务成功率提升。

后续任何新结果必须追加为精简的“实验身份—冻结规则—关键指标—结论”记录；长日志、逐分钟快照和淘汰分支细节继续进入 docs/archive，而不是重新膨胀本文件。
