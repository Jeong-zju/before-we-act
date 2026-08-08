# P1 多机器人模型架构与动作生成技术路线 V5.2（No-Stack 任务组合）

> 状态：CURRENT / NO-STACK RESET
> 更新日期：2026-08-08
> 当前唯一分支：feat/model-improvements
> 完整历史归档：[V2.0 全量历史原稿](../archive/20260725_P1_MULTI_ROBOT_MODEL_ARCHITECTURE_ACTION_GENERATION_ROADMAP_V2.0_FULL_HISTORY_ZH.md)
> 归档原稿 SHA-256：8458d28fe3c4ba6235348bd042ecfca560df2031878da9bdc29d48e4c882dc3c

## 0. 当前决策

新技术路线不再研究任何 Stack 任务。

永久排除：

- stack_cube；
- two_robots_stack_cube；
- three_robots_stack_cube；
- 后续名称或定义属于 cube stacking 的变体。

旧 Stack 结果只保留为历史失败证据，不进入新训练数据、模型选择、Gate、综合分母或后续优化队列。旧 R15 role-query、view-dedup、phase-balanced 和 Stack expert 工作流均转为非活动历史代码；不得作为新基线默认启用。

新的正式任务组合为：

1. lift_barrier；
2. camera_alignment；
3. long_pipeline_delivery；
4. take_photo；
5. pass_shoe；
6. place_food。

其中 pass_shoe 和 place_food 是替代 Stack 的新增任务。六任务全部从零建立新基线，不沿用历史 Stack denominator，也不把旧 W12 的 protected exact reuse 当作新成绩。

## 1. 为什么选择这六个任务

### 1.1 任务覆盖

| 任务 | 机器人数量 | 动作维度 | 合作形态 | 观测形态 |
| --- | ---: | ---: | --- | --- |
| lift_barrier | 2 | 16 | 同步搬运 | global + 2 agent views |
| camera_alignment | 3 | 24 | 多物体对齐 | global + 3 agent views，部分重复 |
| long_pipeline_delivery | 4 | 32 | 长时程协同运输 | global + 4 agent views |
| take_photo | 4 | 32 | 多机器人布局/拍摄 | global + 4 agent views |
| pass_shoe | 2 | 16 | 顺序抓取、交接、投放 | global + 2 个真实不同 agent views |
| place_food | 2 | 16 | 双物体并行抓取与放置 | pinned Hub 数据为 global-only |

该组合同时覆盖：

- 2、3、4 机器人；
- 同步协作和顺序交接；
- 短时程与长时程；
- 真异构多视角与共享 global-only 观测；
- 16、24、32 维动作；
- 多种 episode horizon。

### 1.2 新增任务的证据

#### Pass Shoe

Hugging Face：

- repo：[zeno-ai/robofactory-pass-shoe-multiview](https://huggingface.co/datasets/zeno-ai/robofactory-pass-shoe-multiview)
- revision：646bbfec792ed46c78e452acfc06b423ca1410af
- 150 个成功 episode；
- train/validation/test = 120/15/15；
- train transitions = 43,501；
- camera order = global, agent_0, agent_1；
- action codec = robofactory.2x_panda_pd_joint_pos/1；
- Hub 文件总量约 136.44 GB。

本地抽取 5 个 episode、每个起始/中段/末端共 15 帧：

| 视角对 | bit-exact | mean absolute pixel difference |
| --- | ---: | ---: |
| agent_0 / agent_1 | 0/15 | 19.0081 |
| agent_0 / global | 0/15 | 26.2432 |
| agent_1 / global | 0/15 | 26.4298 |

因此它是真多视角任务，不存在 Stack 的“四路名义不同、像素完全相同”问题。

环境成功定义为鞋子在目标区域平面距离平方小于 0.01。专家轨迹包含 robot 0 抓取并交到中间区域、robot 1 再抓取并投放到目标区，能直接测试角色分工和跨机器人交接。

#### Place Food

Hugging Face：

- repo：[zeno-ai/robofactory-place-food-multiview](https://huggingface.co/datasets/zeno-ai/robofactory-place-food-multiview)
- revision：2237d907f0b28d3f2e19fa4ea03b4048be2de27d
- 150 个成功 episode；
- train/validation/test = 120/15/15；
- train transitions = 25,975；
- pinned training manifest 的 camera order = global；
- action codec = robofactory.2x_panda_pd_joint_pos/1；
- Hub 文件总量约 30.15 GB。

任务环境配置支持 global + 2 agent cameras，但当前 pinned Hub revision 的训练 manifest 只使用 global。新路线将它定义为双机器人共享 global observation 任务，不伪造多视角结论。

本地曾存在另一份约 93 GB、含 3 个真实不同视角的旧 Place Food 数据，但其 episode_000000 SHA-256 与 pinned Hub revision 不同，且缺少 training_manifest.json。该旧缓存不得用于正式训练；如未来希望启用多视角 Place Food，必须发布并固定新的可审计 Hub revision。

环境成功定义为 meat 与 pot 的水平距离小于 0.1，且 meat 高度低于冻结阈值。它可测试共享视野下两机器人并行动作与对象关系。

### 1.3 未进入主基准的候选

| 候选 | 不选原因 |
| --- | --- |
| pick_meat | 单机器人、8 维动作、global-only，不足以检验多机器人模型 |
| strike_cube | 单机器人、8 维动作、global-only；名称含 cube 但不是 stacking，仍与本轮多机器人目标不匹配 |
| RoboCasa composites | 环境、动作 codec、数据协议与当前 RoboFactory 六任务不统一，加入会同时改变过多变量 |
| 所有 Stack | 用户明确排除，且历史已证明该任务的 observation/精度问题主导研究资源 |

pick_meat 和 strike_cube 可作为未来 single-robot transfer holdout，但不进入当前训练、选择或综合分母。

## 2. 数据审计边界

Hugging Face Dataset Viewer 对这些 HDF5 仓库返回 preview/viewer/search/statistics=false，无法通过 parquet viewer 直接统计。因此正式 receipt 必须组合：

- Hugging Face Hub repo metadata；
- exact revision；
- training_manifest.json；
- normalization.npz；
- 每个 HDF5 文件的 SHA-256；
- 本地 HDF5 schema 和像素抽样；
- evaluator task/config commit。

六任务冻结身份：

| 任务 | Hub revision | train episodes | train transitions | camera order | action dim |
| --- | --- | ---: | ---: | --- | ---: |
| lift_barrier | 6ab620091677e69370412f08cd7adecacc28c146 | 120 | 8,255 | global + agent_0 + agent_1 | 16 |
| camera_alignment | e204af13f7191dfd86dab3da529316a51558f479 | 120 | 11,764 | global + agent_0..2 | 24 |
| long_pipeline_delivery | fee628311ff52a3ae0ddfddf82379c63d28f7533 | 120 | 88,493 | global + agent_0..3 | 32 |
| take_photo | 3966385a4c688a5610d4b6cde044150f6b73d320 | 120 | 23,044 | global + agent_0..3 | 32 |
| pass_shoe | 646bbfec792ed46c78e452acfc06b423ca1410af | 120 | 43,501 | global + agent_0 + agent_1 | 16 |
| place_food | 2237d907f0b28d3f2e19fa4ea03b4048be2de27d | 120 | 25,975 | global | 16 |

已知相机边界：

- Pass Shoe 的三路像素抽样均不相同。
- Take Photo 的五路像素抽样均不相同。
- Camera Alignment 的 agent_0 与 agent_1 在抽样中 bit-exact，其他组合不同；必须去重这两路，但不能丢弃 agent_2/global。
- Place Food pinned revision 只允许 global。
- Lift Barrier 和 Long Pipeline 必须在正式训练前完成同样的像素与 calibration 审计。

## 3. Git 与模型基线

旧 R13 TD-MPC2 world model 和旧 R14 world-guided planner 已回退，不在活动代码。

当前允许复用的是：

- W11/W12 架构代码；
- 通用 RGB/qpos/action codec；
- Hugging Face 缓存、镜像、断点续传和鉴权机制；
- DINO foundation；
- 通用 tmux/status/heartbeat 基础设施。

当前不允许复用的是：

- 任何历史 checkpoint；
- Stack specialist、Stack phase、Stack expert 权重或缓存；
- 旧五任务 77/100 作为新六任务验收阈值；
- protected task 的预计算成功结果；
- 为 Stack 编写的 role-query/view-dedup 配置作为默认新模型。

所有历史 checkpoint 已删除，因此必须从零建立六任务 B6 基线。

### 3.1 R11 与 R12 到底做了什么（人话版，历史结果）

先说结论：**R11 给系统增加了一个“理解场面、预测后续”的大脑，但没有让这个大脑控制机器人，所以闭环成绩没有变化；R12 才真正更换动作生成器，旧五任务总分从 74/100 提高到 77/100，但 3 个新增成功全部来自已经退出新路线的 Stack。** 因此 R11/R12 的代码结构可以参考，旧分数不能当作当前 No-Stack 六任务已经提升的证据。

| 阶段 | 用人话说改了什么 | 实际实现 | 选出的方案 | 带来的结果 | 应该怎样理解 |
| --- | --- | --- | --- | --- | --- |
| R11 | 给机器人加一个“场面理解器”：尝试根据最近几帧画面、关节状态和已执行动作，猜接下来会发生什么、队友会怎么动、任务进度到哪里 | 四张 GPU 分别移植 V-JEPA2、LPWM、DINO-WM、LeRobot VLA-JEPA 的最小预测表征组件；它们只生成 `TeamBeliefState`，不进入最终动作链 | P0 V-JEPA2，冻结排序为 P0 > P3 > P1 > P2 | 工程验收四路全部通过；动作与 W10 逐元素完全相同，所以闭环成绩仍等价于 74/100，R11 没有另跑 Gate20 | 这是“组件接成功了”，不是“机器人变强了”。P0 的 future/action 预测还落后于直接沿用上一帧/上一动作的简单基线，胜出主要来自 progress R² 略高，P0 只比 P3 高约 0.00012 |
| R12 | 真正换“司机”：让新动作生成器读取当前高分辨率画面和 R11 的场面表示，直接输出多机器人连续动作块 | 比较 OpenPI Flow、SmolVLA Flow、ACT、Diffusion Policy；最终版本先用 DINOv3 编码完整 480×640 图像，再压成空间 token，并加入 task/agent-slot 信息；训练 10k bridge + 120k joint updates | P2 ACT action-chunk expert | 完整旧五任务 Gate20 为 77/100，严格高于 W10 的 74/100；其余三路都是 74/100 | R12 确实产生了闭环提升，但提升只来自 Stack 从 0/20 到 3/20；另外四任务通过精确回退保持原分数，并不是新模型在这些任务上学得更好 |

历史闭环数字如下。该表只用于解释 R11/R12，不进入当前六任务 B6、R14N 或 R15N 的验收分母。

| 模型/阶段 | Lift | Camera | Stack | Long Pipeline | Photo | 总分 | 相对 W10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W10 基线 | 20/20 | 14/20 | 0/20 | 20/20 | 20/20 | 74/100 | — |
| W11（R11-P0，off-path） | 与 W10 相同 | 与 W10 相同 | 与 W10 相同 | 与 W10 相同 | 与 W10 相同 | 74/100（继承；Gate20 N/A） | 0 |
| R12-P0 OpenPI | 20/20 | 14/20 | 0/20 | 20/20 | 20/20 | 74/100 | 0 |
| R12-P1 SmolVLA | 20/20 | 14/20 | 0/20 | 20/20 | 20/20 | 74/100 | 0 |
| **W12（R12-P2 ACT）** | **20/20** | **14/20** | **3/20** | **20/20** | **20/20** | **77/100** | **+3/100** |
| R12-P3 Diffusion | 20/20 | 14/20 | 0/20 | 20/20 | 20/20 | 74/100 | 0 |

R11 的关键离线事实也不能省略：P0 的 progress R² 为 0.996891，但 future-feature gain 和 partner-action gain 都因差于简单 persistence baseline 被裁剪为 -1.0；其 screen score 为 -0.500621789。换句话说，R11 建成了合法、可训练、可保存恢复的预测接口，却没有证明这个表示比简单历史外推更适合控制。

R12 的关键因果结论是：四路使用相同数据、输入和 130k 训练预算时，只有 ACT 得到 3 次完整成功；OpenPI、SmolVLA 和 Diffusion 都没有闭环增益。与此同时，SmolVLA 的离线模仿误差比 ACT 更低，却仍是 0/20，说明**离线 loss 好看不等于机器人闭环更可靠**。旧结果中真正被证明的，是 ACT 动作块对原 Stack 多阶段协调更合适；它没有证明 No-Stack 六任务会自动受益，所以新路线仍必须从零建立 B6。

## 4. 新阶段定义

| 阶段 | 名称 | 目标 |
| --- | --- | --- |
| R13N | Six-Task Data & Baseline Reset | 下载并审计六任务，训练可执行 B6 baseline |
| R14N | Six-Task Causal Tournament | 四卡独立比较数据、观测、动作头和恢复控制 |
| R15N | Generalization Confirmation | 对 winner 做独立 50-seed 六任务确认 |

新路线中不存在 Stack fallback、Stack phase metric 或 Stack 专项门。

## 5. R13N：六任务数据与基线重置

### 5.1 数据准备

远程当前已有前四个任务，但没有 Pass Shoe 和 Place Food 的正式 training manifest。部署前：

1. 以 revision-specific 目录下载 Pass Shoe 与 Place Food。
2. 复用 /workspace/.cache/huggingface。
3. 下载后逐文件校验 Hub manifest。
4. 不覆盖任何现有同名旧目录；旧 Place Food 缓存必须与 pinned revision 隔离。
5. 为六任务生成统一 dataset receipt。

当前服务器 /workspace 约 218 GB 可用；两项新数据合计约 166.59 GB，只剩约 51 GB，无法安全容纳四路训练输出。

被排除的 /workspace/datasets/robofactory_multitask/three_robots_stack_cube 当前约 165 GB。正式下载前可在确认无进程引用、生成删除 receipt 后回收该目录，或使用新的共享存储；不得边训练边被动耗尽磁盘。

### 5.2 调用链修改

R13N 实现时必须搜索并更新：

- before_we_act/benchmark.py；
- action generator 的 task allowlist；
- train/validation row-count 常量；
- task/state/action padding；
- task embedding/registry；
- evaluator task registry 和 max episode steps；
- Gate、acceptance、monitor、launch 和 stop；
- seed manifest；
- dataset/cache receipt；
- 测试中的旧五任务固定集合。

两个 Stack task 必须从活动 allowlist 和正式 Gate 中移除，但历史 archive 不修改。

### 5.3 B6 baseline

B6 是 W12-style ACT 架构在六任务上的全新从零训练：

- 不启用旧 R15 Stack patch；
- 六任务共同训练；
- task-balanced sampling；
- action/state 统一 padding 并保留真实 active-agent mask；
- view mask 由每任务 manifest 决定；
- 每个任务均由模型真实推理，不做 task-level result reuse。

训练完成后为六任务各建立：

- Discovery20 baseline；
- Validation20 baseline；
- Formal20 baseline；
- task-specific seed JSON；
- model-native rollout；
- 成功和失败视频；
- task-stage diagnostics；
- checkpoint/normalization/dataset receipt。

历史非 Stack 四任务 74/80 只用于 sanity check，不参与新 acceptance 运算。

## 6. R14N：四卡独立改进

四个分支从同一 B6 base commit 和同一数据 receipt 创建，每路只改变一个主要变量。

### 6.1 GPU0 / A：Task-Balanced Data Curriculum

- 模型架构与 B6 完全相同。
- 改变 task/episode/phase sampling。
- 对长任务不再因 transitions 多而压制短任务。
- 使用先短任务稳定、再混入长时程协作的 curriculum。
- 记录每任务有效 batch 比例和 gradient contribution。

建议分支：exp/r14n-a-six-task-curriculum。

### 6.2 GPU1 / B：Heterogeneous Observation Fusion

- 只改变 observation fusion。
- 对真多视角使用 calibration-aware view tokens 和 agent role。
- 对 bit-exact 视角在 encoder 前去重。
- 对 Place Food global-only 使用单 active view，不生成虚假 agent view。
- active-agent mask、view mask、action slice 必须一致。
- 不允许训练期 privileged state 进入部署输入。

建议分支：exp/r14n-b-heterogeneous-observation。

### 6.3 GPU2 / C：RGB Diffusion Policy Head

- observation encoder、数据和 action codec 与 B6 对齐。
- 只把 ACT/CVAE action head 替换为 RGB-conditioned action diffusion。
- horizon、执行频率、latency 上限和 seed 固定。
- 先做 tiny-batch overfit、strict shape、finite action 和 deterministic sampling preflight。

移植来源：[Diffusion Policy 官方仓库](https://github.com/real-stanford/diffusion_policy)。只移植最小 action diffusion 闭包，并固定 upstream commit、license、source map 和 parity。

建议分支：exp/r14n-c-six-task-diffusion。

### 6.4 GPU3 / D：Uncertainty-Gated Residual Recovery

- B6 生成主 action chunk。
- residual head 在预注册的不确定度或进度停滞条件触发。
- 面向六任务采集 on-policy failure/intervention 数据。
- residual 有动作幅度、持续步数、latency 和 fallback 强限制。
- 报告 intervention rate、paired wins/losses、timeout、fallback 和恢复后成功率。

建议分支：exp/r14n-d-six-task-residual。

## 7. 外部开源移植规则

每次模型尝试都先搜索论文官方开源实现，但只有满足以下条件才移植：

1. 输入模态与六任务合法 observation 一致。
2. 许可证允许。
3. exact upstream commit 可固定。
4. 最小 patch 可与本项目 action codec 对齐。
5. parity、method separation、strict load 和 action-effect 可测试。
6. RTX 5090 / CUDA 12.8 环境可运行。
7. 推理 latency 满足 task control frequency。

论文成绩只解释动机，不替代本项目闭环结果。

## 8. 新验收合同

### 8.1 统一分母

- Discovery20：6×20 = 120 episodes。
- Validation20：6×20 = 120 episodes。
- Formal20：6×20 = 120 episodes。
- Confirmation50：6×50 = 300 episodes。

各层 seeds 独立。候选在上一层冻结后才能进入下一层。

### 8.2 禁止 protected reuse

六个任务都必须由候选模型实际执行：

- 不得复制 B6 rollout JSON；
- 不得按任务路由回旧模型获得分数；
- 不得将 fallback 成功计为 candidate-native success；
- numerical safety fallback 可以保留，但对应 episode 在 candidate-native 口径计失败。

同时报告：

- model-native score；
- safety-system score；
- fallback/intervention coverage；
- 每任务 paired wins/losses。

正式 winner 以 model-native score 为主。

### 8.3 PASSED 条件

设 B6 为同 seeds 新基线，候选必须同时满足：

1. 六任务总成功数严格高于 B6。
2. 六个任务的成功数均不得低于 B6。
3. 至少两个任务严格高于 B6。
4. 至少一个新增任务 Pass Shoe 或 Place Food 严格高于 B6。
5. candidate-native coverage = 100% episodes。
6. 无 OOM、NaN、异常重启、非有限动作、无心跳或缺失 acceptance。
7. Formal20 通过后，Confirmation50 仍保持正 paired net improvement，且无单任务回退。

如果 B6 在某任务达到 20/20，该任务只要求不回退；模型选择主要由未饱和任务产生的 paired improvement 决定。

### 8.4 指标

每任务固定报告：

- Success Rate；
- Wilson interval；
- Mean Steps to Success；
- Executable Rate；
- Intervention/Fallback Rate；
- candidate-native coverage；
- paired wins/losses；
- episode latency P50/P95；
- task-specific stage completion。

任务阶段只用于诊断，不替代 success。

## 9. R15N：组合与确认

只有两个主轴分别独立通过 Formal20，才允许组合。

允许示例：

- curriculum winner + observation winner；
- observation winner + diffusion winner；
- curriculum winner + residual recovery winner。

组合必须重新运行 Discovery20、Validation20、Formal20 和 Confirmation50。失败路线不得通过组合绕过独立归因。

若四路均失败：

- 保留六任务和 B6；
- 根据 seed-level 失败分布重新提出候选；
- 不自动加入 Stack；
- single-robot pick_meat/strike_cube 只有在研究目标明确扩展到跨机器人数量泛化时才进入下一轮。

## 10. 四卡和运行约定

| GPU | 候选 | tmux 建议 |
| ---: | --- | --- |
| 0 | A Task-Balanced Curriculum | bwa-r14n-a |
| 1 | B Heterogeneous Observation | bwa-r14n-b |
| 2 | C RGB Diffusion Policy | bwa-r14n-c |
| 3 | D Residual Recovery | bwa-r14n-d |

每路必须有独立：

- branch/worktree；
- output/log/checkpoint；
- status/heartbeat；
- tmux session；
- acceptance JSON；
- seed-level results。

数据集与 Hugging Face cache 只读共享。monitor 读取 producer heartbeat 和结构化 acceptance，不从日志文件存在性猜测状态。

## 11. 当前资产和下一步

服务器：

- repo：/workspace/fe-pc-wam；
- dataset root：/workspace/datasets/robofactory_multitask；
- HF cache：/workspace/.cache/huggingface；
- 当前可用磁盘约 218 GB；
- 被排除的 ThreeRobotsStackCube 数据约 165 GB；
- Pass Shoe/Place Food 尚未以正式 manifest 部署；
- 当前无可恢复模型 checkpoint。

下一次实施顺序：

1. 为六任务实现 dataset audit 和 revision-specific download receipt。
2. 更新全部 task registry、allowlist、shape/padding、Gate 和 monitor。
3. 用 dry-run 验证六任务调用链。
4. 回收不再使用的远程 Stack 数据或扩展存储。
5. 下载 Pass Shoe/Place Food。
6. 从零训练 B6。
7. 冻结 B6 后创建 R14N 四路分支。

在 B6 checkpoint、六任务 seeds 和 acceptance 代码齐备前，不启动长期训练。

## 12. 当前验证命令

本地：

    cd /home/jeong/zeno/wam/before-we-act
    git switch feat/model-improvements
    git status --short
    git rev-parse HEAD
    .venv/bin/python -m compileall -q before_we_act scripts/before_we_act
    .venv/bin/python -m pytest -q tests/before_we_act
    git diff --check

远程状态：

    ssh -p 10328 root@69.176.92.104
    cd /workspace/fe-pc-wam
    git switch feat/model-improvements
    git pull --ff-only origin feat/model-improvements
    git status --short
    git rev-parse HEAD
    df -h /workspace

Hugging Face token 只允许通过 S0 已有环境变量或受限 secret 注入，不得写入代码、配置、argv、日志、文档或 Git。

## 13. 声明边界

可以声明：

- 新路线正式排除全部 Stack 任务；
- 新六任务包含 Pass Shoe 和 Place Food；
- Pass Shoe 已验证为真实不同多视角；
- Place Food pinned Hub revision 是双机器人 global-only；
- 新 baseline 和候选都需要从零训练；
- 新 Gate 不允许 protected result reuse。

禁止声明：

- 历史 77/100 可直接换算为六任务成绩；
- 当前已有六任务 checkpoint；
- 本地旧 Place Food 多视角缓存等同于 pinned Hub 数据；
- candidate 未真实执行某任务却获得该任务成功分；
- 任一新方法在训练前已经优于 B6；
- Stack 以诊断、fallback 或隐含分母形式重新进入路线。

后续只在本文件追加冻结阶段结论和当前有效命令。长日志、逐分钟快照和淘汰分支细节继续进入 docs/archive。
