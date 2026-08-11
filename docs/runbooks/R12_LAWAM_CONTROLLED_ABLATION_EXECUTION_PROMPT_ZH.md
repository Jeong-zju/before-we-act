# R12 LaWAM 受控消融四卡正式执行 AI Coding Prompt（历史）

> **已归档：禁止执行。** R12 已按用户决策关闭为 `ARCHIVED_NO_WINNER`；本 prompt 只保留作只读审计，不得继续训练、创建新候选、写入原 run root 或把未完成的 L2/L3 改写为实验 gate 失败。完整历史见 [R11/R12 失败技术路线归档](../archive/20260811_R11_R12_FAILED_TECHNICAL_ROUTES_ZH.md)。

```text
====================
一、执行变量（每轮优先核验这里）
====================

[PROJECT_ROOT] = /home/jeong/zeno/wam/before-we-act
[ROADMAP_DOC] = /home/jeong/zeno/wam/before-we-act/docs/archive/20260811_R11_R12_FAILED_TECHNICAL_ROUTES_ZH.md
[RESULT_DOC] = ROADMAP_DOC 第 13 节；详细记录写入 docs/experiments/r12/

[STAGE] = R12
[STAGE_DESIGN_SECTION] = ROADMAP_DOC 第 13.1 至第 13.2 节
[STAGE_FAIL_FAST_SECTION] = ROADMAP_DOC 第 13.2 节 Measurement/F0-F1/Discovery gate
[STAGE_FORMAL_ACCEPTANCE_SECTION] = ROADMAP_DOC 第 11 节原始正式门槛、冻结 score 和 Confirmation50

[BASELINE_STAGE] = W10 六任务
[BASELINE_BRANCH] = feat/model-improvements
[BASELINE_COMMIT] = AUTO_RESOLVE_ORIGIN_HEAD_AND_FREEZE_IN_RUN_MANIFEST
[BASELINE_REQUIRED_ANCESTOR] = bca22fdb7687a3d0d7bd4fd266a73cd6536a30e8
[LAWAM_INTEGRATION_COMMIT] = 6cd891b86238253ae7a1cd532aead99cbc192f48
[BASELINE_CHECKPOINT] = /workspace/bwa_runs/w10-six-task-v1/train/formal/checkpoint_120000.pt
[BASELINE_CHECKPOINT_SHA256] = e1b07b2cf7bff37428bf54a27f545632c8a1013930d96f6e646d8ca055f2f574
[BASELINE_VALIDATION20] = 88/120；Lift 20，Camera 8，Long 20，Photo 20，Shoe 20，Food 0
[DATASET_VERSION_POLICY] = 先比对当前六任务 manifest/HDF5 receipt 与 BASELINE_CHECKPOINT provenance；数据变化时先重训并重新验证 W10，再冻结新的 baseline/checkpoint/hash
[W10_PRIOR_POLICY] = W10 action prior 是候选内部可训练组件；默认以较低 learning rate 解冻并联合训练；评测期禁止 fallback 到独立 W10 policy

[R11_RESULT] = NO_WINNER
[R11_LAWAM_STATUS] = FAILED at Discovery persistence gate；future-vs-persistence=-8.399%，2/6 任务改善
[R11_LAWAM_POSITIVE_EVIDENCE] = action-shuffle→future +32.734%，5/6；prediction-off/shuffled→action +6.758%/+61.146%
[R11_LAWAM_CHECKPOINT] = 只允许从 /workspace/bwa_runs/r11-four-way-v1/D 的结构化 receipt 解析后做只读诊断
[R11_LAWAM_CHECKPOINT_SHA256] = 5ad9476371bb308181c3fb6aa07114c921a1e097e78fa73350e3f9fbc077a888
[R11_INITIALIZATION_POLICY] = 禁止从 R11 D checkpoint 续训或 warm-start；新 R12 从冻结 LaWAM foundation/source 与当前 W10 action prior 重新训练 world/selector/refiner

[COMMON_MODEL] = W10-anchored LaWAM action-relevant latent planning
[PROPOSAL_COUNT] = K=4
[ACTION_HORIZON] = 100
[LATENT_HORIZONS] = h={1,4,8,16}
[EXECUTION_PREFIX_AND_CADENCE] = 实现前从 W10 evaluator/action temporal ensemble 完整调用链解析；在查看任何 R12 指标前冻结为数值配置和 manifest，不得候选间不同
[PROPOSAL_GENERATION_POLICY] = proposal-0 必须是正常 W10 prior 输出；proposal-1..3 的采样/latent/noise 构造及 seed 在 Measurement 前冻结；四候选逐 sample 使用完全相同的 proposal keys、动作值和 mask

[CANDIDATE_L0] = R12LaWAMDeltaLatentSelector；delta-latent + common selector
[CANDIDATE_L1] = R12LaWAMSelfGroundedSelector；L0 + self-grounded/foreground supervision
[CANDIDATE_L2] = R12LaWAMAlgebraicTransitionSelector；L0 + composition/reversal consistency
[CANDIDATE_L3] = R12LaWAMUncertaintyResidualRefine；L0+L1+L2 + uncertainty-aware score/residual refine

[BRANCH_L0] = feat/r12-lawam-l0-delta-latent
[BRANCH_L1] = feat/r12-lawam-l1-self-grounded
[BRANCH_L2] = feat/r12-lawam-l2-algebraic-transition
[BRANCH_L3] = feat/r12-lawam-l3-uncertainty-refine
[BRANCH_INTEGRATION] = feat/r12-lawam-controlled-ablation

[UPSTREAM_LAWAM] = https://github.com/RLinf/LaWAM.git@4ea6fdadce6c9b8746028307a246b79ee2c4fd55
[REFERENCE_L1] = SelfWAM，arXiv:2608.00725；只作机制参考，除非重新核验到官方源码并冻结 commit/license/receipt，否则不得声称源码迁移
[REFERENCE_L2] = ALAM，arXiv:2605.10819；同上
[REFERENCE_L3] = Guided Action Flow，arXiv:2607.02092；同上，不得搬用其任务或论文成绩
[UPSTREAM_RECHECK_POLICY] = 开工前检查上述论文和官方仓库是否有公开代码、安全或兼容性修复；不得静默更换 LaWAM commit；任何新增官方源码迁移必须先更新 ROADMAP_DOC、commit、license 和 SOURCE_RECEIPT

[REMOTE_SSH] = ssh -p 10328 root@69.176.92.104
[REMOTE_REPO] = /workspace/fe-pc-wam
[REMOTE_WORKTREE_ROOT] = /workspace/r12_worktrees
[REMOTE_RUN_ROOT] = /workspace/bwa_runs/r12-lawam-controlled-ablation-v1
[REMOTE_DATA_ROOT] = /workspace/datasets/robofactory_multitask
[REMOTE_UPSTREAM_ARTIFACTS] = /workspace/artifacts/r12_upstream
[HF_CACHE] = /workspace/.cache/huggingface
[W10_SEED_ROOT] = /workspace/bwa_runs/w10-six-task-v1/seeds/validation
[PYTHON_BASE] = /venv/robofactory-act/bin/python

[COUNTERFACTUAL_PROBE_ROOT] = /workspace/bwa_runs/r12-lawam-controlled-ablation-v1/shared/counterfactual_probe_v1
[COUNTERFACTUAL_PROBE] = 六任务每任务固定 32 个 anchor，共 192 个；同一状态执行正确、时间打乱、反向、0.5x 和 1.5x 幅值前缀；记录 h={1,4,8,16} 的 robot/object/proprio/progress 后果
[COUNTERFACTUAL_PROBE_POLICY] = probe 独立于训练集和 Validation5/20/50；保存 seed、simulator state/replay、投影前后动作、max steps、成功/风险定义和逐文件 hash；第一轮只诊断/验收，不进入训练
[MEASUREMENT_FUTURE_GATE] = held-out oracle transition 相对 persistence 的 macro normalized error gain >=5%，且至少 4/6 任务为正
[MEASUREMENT_HEADROOM_GATE] = 用读取结果前冻结的 task progress/risk 标量和 tie 规则比较 oracle-best 与 proposal-0；非默认 proposal 的 macro paired-win rate >=10%，至少 4/6 任务各自 paired-win rate >=10%，且 macro mean oracle-score advantage >0；否则停止
[DISCOVERY_RANKING_GATE] = K=4 counterfactual ranking accuracy >=70%

[GPU_L0] = 0
[GPU_L1] = 1
[GPU_L2] = 2
[GPU_L3] = 3
[GPU_TYPE] = NVIDIA RTX 5090 32GB
[SESSION_MANAGER] = tmux
[SESSION_L0] = bwa-r12-l0-delta
[SESSION_L1] = bwa-r12-l1-grounded
[SESSION_L2] = bwa-r12-l2-algebraic
[SESSION_L3] = bwa-r12-l3-combined
[SESSION_MONITOR] = bwa-r12-monitor

[EFFECTIVE_BATCH] = 48；每 optimizer update 六任务各 8 个 local-agent 样本
[DISCOVERY_UPDATES] = 1000
[SELECTION_UPDATES] = 20000
[FORMAL_UPDATES] = 120000
[VALIDATION_DISCOVERY] = 每任务固定 5 回合；必须 paired 使用相同 seed 和 proposal receipt
[VALIDATION_FORMAL] = 复用 W10 每任务固定 20 回合，共 120 回合
[CONFIRMATION] = 临时 winner 与 W10 使用查看结果前新冻结的 seeds，每任务 50 回合

[HF_DOWNLOAD_POLICY] = 沿用共享 cache、镜像/代理、断点续传、离线复用和环境变量或 mode-0600 secret/FIFO 鉴权；绝不回显、记录或提交 token
[MERGE_POLICY] = 只有通过第 11 节 Validation20 全部门槛和 Confirmation50 非劣性的最终 winner 才合入 BASELINE_BRANCH；失败分支只保留证据，不混合机制补 winner
[CLEANUP_POLICY] = 只能清理 R12 明确拥有且已有 receipt 的临时文件；不得删除数据集、W10 checkpoint、R11 结果、共享 HF cache、其他实验或不明进程/session

====================
二、目标与不可变研究结论
====================

完整阅读 ROADMAP_DOC 后，自主实现、训练、验证和验收 R12 的 L0-L3 四个受控消融。研究问题不是“继续训练 R11 D 会不会变好”，而是：在相同 W10 proposals 和相同 LaWAM transition 基座上，action-relevant target、self-grounding、algebraic consistency 与 uncertainty-aware selection/refinement 中哪一个机制能让未来预测真正优于 persistence，并改善动作选择。

R11 永久保持“无胜者”，LaWAM 永久保持 `FAILED`。R12 可以证明新机制有效，但不能追溯改写 R11 的 checkpoint、acceptance 或 winner。目标也不是强制产生 winner：若 Measurement 证明 proposal bank 没有 oracle headroom，或四路均未通过冻结门槛，必须写“R12 无胜者”并停止，不得降低门槛、只看 loss、评测 fallback 到 W10 或混合失败路线。

所有路线的预测必须真实进入 proposal score、可微选择或 residual refinement。仅增加 world/latent 辅助 loss、评测时关闭预测后仍输出独立 W10 动作、或失败时切回 proposal-0，都不合格。

====================
三、开始前必须完成
====================

1. 完整阅读 ROADMAP_DOC 第 2、5、8、11、13.1、13.2 和 14 节；旧 archive 只用于历史审计，不得作为活动 R12 设计。
2. 在 PROJECT_ROOT 检查 `git status --short --branch`、当前 commit、remote、未推送提交、用户已有修改和 worktree；不得覆盖无关 dirty changes。
3. 用 `rg` 阅读 W10 模型、ACT latent/action sampling、temporal ensemble、六任务 trainer/evaluator、manifest/HDF5 schema、归一化、fixed seed、max steps、model registry、checkpoint loader、launcher/monitor/stop，以及已合入的 R11 LaWAM model/data/train/causality/runtime 完整调用链。
4. 只读连接远程，记录 GPU、磁盘、数据、HF cache、Python/CUDA/PyTorch、tmux、进程和 R11/R12 目录；不得停止或删除不属于 R12 的对象。
5. fetch 后解析 `origin/feat/model-improvements` 为唯一 `R12_BASE_COMMIT`，核验其包含 BASELINE_REQUIRED_ANCESTOR 和 LAWAM_INTEGRATION_COMMIT，写入本地与远程 immutable run manifest。L0-L3 和 integration 必须各自从该 base 建立，不能从彼此分支派生。
6. 重算 BASELINE_CHECKPOINT SHA256，核验 checkpoint 内训练 receipt、当前六任务 manifest/HDF5 receipt、W10 Validation20 JSON 和固定 seeds。receipt 变化时执行 DATASET_VERSION_POLICY；receipt 相同但 checkpoint hash 不同时停止并诊断。
7. 核验 R11 `winner.json=NO_WINNER`、D `acceptance.json=FAILED` 和 D checkpoint hash。R11 产物只能只读，R12 不得在 `/workspace/bwa_runs/r11-four-way-v1` 写文件。
8. 核验已合入 `third_party/r11/lawam/SOURCE_RECEIPT.json`、LICENSE/NOTICE 和只读 LaWAM checkout 的 URL/commit/逐文件 hash。执行 UPSTREAM_RECHECK_POLICY，文档和提交先于任何 commit 升级。
9. 从真实 W10 调用链确定 K=4 proposal 构造、100-step chunk、执行 prefix、receding-horizon cadence、temporal aggregation和安全投影；在读取 probe 指标前冻结数值配置、seed 和 tie 规则。不得让四个候选使用不同 proposal bank。
10. 先生成 counterfactual probe manifest 和 measurement gate 配置并提交推送；只有 receipt/hash 完整后才允许采集 probe。

====================
四、共同实现与唯一变量约束
====================

共同 integration 分支只实现：R12 immutable manifest、counterfactual probe、exact K=4 proposal cache/loader、公共 selector/refiner 接口、六任务 exact sampler、统一训练/评测/acceptance、launcher、monitor 和 graceful stop。不得把某一候选的模型机制放入 integration 后让其他候选隐式继承。

四路必须共享：

- 同一 W10 checkpoint、trainable action prior、LaWAM foundation、base optimizer policy、proposal keys和值；
- 同一 observation/task/proprio 输入、delta-latent whitening、K=4 selector 容量、action horizon、rollout prefix、cadence；
- 同一数据 receipt、effective batch 48、sample cursor、训练更新数、seed、Validation5/20/50 和 acceptance；
- 同一 `normal/prediction_off/prediction_shuffled/action_shuffled` 推理接口；
- 同一参数预算上限；L1/L2 新 head 增加的参数必须由 common head 宽度补偿或在 score 中显式报告，不得靠扩大 backbone 获益。

每个候选只允许以下差异：

1. L0：预测按任务/视角/horizon 白化的 `Δz=z(t+h)-z(t)`；使用 common selector，无 foreground/algebraic/uncertainty 新机制。
2. L1：在 L0 上只增加 robot self-mask；没有可靠 self-mask 时使用预注册 motion/foreground mask，并加 proprio delta supervision。不得加入 L2 consistency 或 L3 uncertainty residual。
3. L2：在 L0 上只增加 action transition composition 和 reversal consistency。不得加入 foreground target 或 L3 uncertainty residual。
4. L3：组合 L0/L1/L2，并增加校准 uncertainty-aware progress/risk score 和 residual action refinement。L3 的 Discovery 只能在 L0/L1/L2 的 Discovery 结构化结果均产生后启动；失败的子机制仍可用于组合诊断，但不得隐瞒或改写其失败状态。

W10 action prior 默认解冻联合训练，使用显式较低 learning rate parameter group；LaWAM foundation 的冻结/解冻范围、selector/refiner、各 loss weight 和 trainable parameter map 必须逐参数记录。评测 checkpoint 不得携带可调用的独立 W10 fallback handle。

新增模型名后用 `rg` 审计所有白名单和分发点：factory/registry、CLI choices、config validation、checkpoint loader、inference adapter、evaluator、acceptance、launcher、monitor、resume 和文档。不得只修一个 registry。

====================
五、Counterfactual Measurement gate
====================

1. 使用与正式评测隔离的固定 192 个 simulator anchors；每个 anchor 从完全相同 observation/task text/state 开始执行正确、时间打乱、反向、0.5x、1.5x 动作前缀，并记录安全投影前后动作。
2. 在 h={1,4,8,16} 保存真实 robot/object/proprio/progress 后果，按背景、机器人、任务物体和 proprio 分解 persistence error；未来 frame/latent/状态只作 label，绝不进入 deployment input。
3. 小型 oracle transition 的 held-out future-vs-persistence 必须满足 MEASUREMENT_FUTURE_GATE。这里的 oracle 只验证 measurement 是否可学，不得成为正式候选或评测 fallback。
4. 用真实后果评估同一 K=4 proposal bank，按现有 evaluator 的成功条件和在读取结果前写入 `measurement_gate.json` 的逐任务 progress/risk 标量与 tie 规则生成 oracle rank。必须满足 MEASUREMENT_HEADROOM_GATE；否则 world model 无法从无 headroom 的 proposals 中选择成功动作，R12 直接写 `FAILED_MEASUREMENT`，不进入 F0/F1 或大模型训练。
5. 输出 `probe_manifest.json`、`probe_receipt.json`、`measurement_gate.json` 和逐任务报告；记录源 commit、数据/seed、simulator/evaluator commit、数量、排除项及原因、所有 hash。不得看完结果后改变 anchor、动作扰动、归一化或门槛。
6. 第一轮 probe 不加入训练。若决定把 counterfactual 数据加入训练，必须停止本轮，按 DATASET_VERSION_POLICY 建立新数据版本、重训/重验 W10 并创建新的 R12 run root。

====================
六、源码、许可证、测试与 F0/F1
====================

1. 已合入的 LaWAM 上游符号必须继续由 `third_party/r11/lawam/SOURCE_RECEIPT.json` 和只读 checkout 验证；若修改上游复制代码或新增官方源码，迁移最小依赖闭包到 `third_party/r12/<source>/`，生成逐文件 path/hash receipt、LICENSE、NOTICE、SPDX 和逐符号映射文档。
2. SelfWAM/ALAM/Guided Action Flow 当前是机制参考，不得把“读论文后本项目适配”写成源码迁移。若找到官方代码，只有完成 URL/commit/license/parity/receipt 后才能使用。
3. F0 必须零 skip：LaWAM official fixed tensor parity、delta target/whitening、proposal identity、mask、composition/reversal、uncertainty calibration、selector/refiner、W10 partial load、six-task adapter、normal/off/shuffled/action-shuffled 全部通过。缺依赖导致 skip 视为 F0 失败。
4. F1 必须使用真实 foundation 和真实主路径完成 forward/backward/optimizer/save/resume/inference；检查 W10 prior、world、selector/refiner 均有 finite nonzero gradient，冻结参数无梯度，checkpoint 只保存允许的 trainable state 和完整 provenance。
5. resume 必须恢复 optimizer/scheduler/RNG/sample cursor；恢复后的下一 sample key、proposal key 和未中断基准一致。三种预测模式都输出有限、范围合法的 `[batch,100,action_dim]`，但动作必须可检测地不同。
6. unit/parity/integration/resume/causality 测试和 CPU smoke 先在本地完成；远端 F0/F1 不因本地无 GPU 跳过。显存降级顺序必须在看到 OOM 前预注册，并保持 effective batch、model semantics 和主路径不变。
7. 不提交权重、数据、token、cache 或大日志。secret scan 命中必须修复；只允许重写尚未推送的 R12 提交，禁止破坏共享历史。

====================
七、本地 Git、四分支与部署
====================

1. 所有长期代码先在本地独立 worktree 完成，远程只部署已提交代码。L0-L3 分支必须是 R12_BASE_COMMIT 的兄弟分支；integration 也从同一 base 创建。
2. 每分支提交信息标明 L0/L1/L2/L3、共同接口和来源；用 diff 审计确保 L1/L2 不互相混入、L3 不回写前三路。
3. 每分支运行格式/静态检查、目标单测、CPU smoke、`git diff --check`、source/license/secret scan 和 base ancestry 检查后提交推送。禁止 force push、`reset --hard` 或覆盖用户提交。
4. 远程 fetch 后在 REMOTE_WORKTREE_ROOT 建四个独立 worktree，逐一校验 branch/commit/base ancestry；每候选独立 venv、output、checkpoint、log、status、heartbeat 和 tmux。
5. GPU 固定 L0=0、L1=1、L2=2、L3=3。launcher 和 worker 同时核验 `CUDA_VISIBLE_DEVICES`；禁止跨卡、借卡或杀死不明进程。
6. 先运行 `launch_r12_4gpu_tmux.sh --all --dry-run`。顺序固定为 Measurement → F0 → F1 → L0/L1/L2 Discovery → L3 Discovery eligibility → L3 Discovery → paired Validation5 → Selection → Formal → Validation20 → winner/Confirmation50。
7. 一个候选失败不得杀死其他候选。L3 等待期间 GPU3 可以运行本阶段共享 probe/F0/F1 或保持空闲，不得提前看 L3 Discovery 结果破坏组合因果归因。
8. 代码问题必须回到对应本地分支修复、测试、commit、push，再远程 fast-forward 和安全 resume；记录每次修复。不得把远程临时修改当长期实现。

====================
八、一键运行、monitor 与安全停止
====================

实现并实际验证：

  scripts/before_we_act/launch_r12_4gpu_tmux.sh --all [--dry-run]
  scripts/before_we_act/launch_r12_4gpu_tmux.sh --candidate L0
  scripts/before_we_act/monitor_r12.sh --all --once
  scripts/before_we_act/monitor_r12.sh --candidate L0 --watch
  scripts/before_we_act/stop_r12_4gpu_tmux.sh --candidate L0 --graceful
  scripts/before_we_act/stop_r12_4gpu_tmux.sh --all --graceful

monitor 必须同时显示 L0-L3 和共享 Measurement 队列：当前时间、candidate/唯一机制、branch/commit/base/upstream、GPU/tmux/program/stage/status/PID/start/elapsed、实际 worker heartbeat 和 age、GPU 利用率/显存/温度/功耗、update/120000、ETA、action/world/ranking/consistency/uncertainty loss、future gain、action-shuffle、prediction-off/shuffled、K=4 ranking accuracy、Validation5/20、current/best checkpoint、日志、OOM/NaN/异常退出/stale 和逐项 acceptance。

状态至少有 `NOT_STARTED/PREPARING/MEASURING/PREFLIGHT/WAITING_PREREQUISITE/TRAINING/VALIDATING/ACCEPTING/PASSED/FAILED/FAILED_MEASUREMENT/FAILED_FIT/STOPPED/STALE/UNKNOWN`。心跳必须来自实际 worker 或核验 PID start-time 的 watchdog；PASSED/FAILED 只读结构化 JSON，不能从日志关键词猜测。

graceful stop 必须精确核验 run manifest 的 PID、PID start-time、tmux session、candidate 和 GPU，先请求保存 checkpoint/停止，再只结束该候选；禁止 `pkill -f`、宽泛 PID 匹配或触碰其他 session/process。

====================
九、Discovery、闭环验收与 winner
====================

1. Discovery1000 必须同时满足：action loss 下降；future-vs-persistence macro gain >=5% 且至少 4/6 任务为正；action shuffle 令 future error macro 恶化 >=5% 且至少 4/6 任务恶化 >=5%；prediction_off 或 prediction_shuffled 令 action NRMSE 恶化 >=2%；K=4 counterfactual ranking accuracy >=70%。任一失败立即形成结构化失败结论，不进入该路线的闭环预算。
2. L3 必须在 L0/L1/L2 的 Discovery JSON 齐全后才评估。报告 L3 对 L0 的增益，以及 L1/L2 单机制是否提供支持或反证；不得因为组合通过而改写单机制失败。
3. 只有通过 Discovery 的路线才跑 paired Validation5、Selection、Formal 和固定 Validation20。每阶段检查 exit code、日志、heartbeat、GPU、OOM/NaN、断点连续性、样本/提案计数和 checkpoint provenance。
4. Validation20 必须完成同一 120/120 seeds/max steps/success conditions，无 fallback、非法动作或漏跑。逐项应用 ROADMAP_DOC 第 11.1 节：总成功 >=80/120；protected-four >=72/80 且单项 >=16/20；Camera >=6/20；Camera+Food >=8/40；预测和双向因果 gate 全过。
5. 对合格候选只使用第 11.2 节冻结 score 排名，报告原始逐任务成功数、预测 gain、因果、ranking、p95 latency 和峰值显存；不得增删权重或只报 score。
6. 无合格候选时生成 `winner.json=NO_WINNER`，写“R12 无胜者”，不合并。有人合格时选临时 winner，在查看结果前生成每任务 50 个新 seeds，同时评测 W10/winner 共 300+300 回合并计算 paired bootstrap 95% CI。
7. 只有 Confirmation50 总成功率相对 W10 的非劣界 >=-6.67pp 且预测/因果 gate 继续通过，才允许把唯一 winner 合入 BASELINE_BRANCH。禁止混入其他路线、用 W10 fallback 补失败任务或把 R11 D checkpoint 当 winner。

====================
十、结果文档和最终汇报
====================

Measurement、F0/F1、每个 Discovery、Validation5、Selection、Formal、Validation20、winner 和 Confirmation50 后，都更新 ROADMAP_DOC 第 13 节摘要；完整命令、配置、环境、指标、receipt 和日志位置写入 `docs/experiments/r12/`。文档只在本地修改、检查、commit、push，远程不长期维护文档。

记录：北京时间和 UTC、local/remote branch/commit/base、LaWAM upstream/license/source receipt、机制参考的“源码迁移/本项目适配”边界、Python/CUDA/PyTorch/GPU、完整命令、数据/seed/hash、proposal 构造、probe、HF cache、tmux/PID/start-time、micro-batch/accumulation、parameter groups、输出/log/checkpoint/status/heartbeat、所有训练/预测/ranking/闭环/因果指标、逐项 acceptance、score、失败根因、winner/无 winner、merge 和后续建议。不得回显 token。

最终按以下结构汇报：

1. 当前结论：Measurement、L0-L3 状态、R12 winner/无 winner、是否合并。
2. Git 状态：base、四分支、integration、commit、push、文档 commit。
3. 远程状态：四个 tmux/GPU/program/stage/progress/heartbeat/log/checkpoint 和 L3 prerequisite。
4. 实验结果：probe headroom、训练、future-vs-persistence、双向因果、K=4 ranking、Validation5/20/50、逐项资格、score。
5. 可复制命令：probe、deploy/train/resume、monitor、graceful stop、acceptance、reproduce winner。
6. 文档、receipt、日志、checkpoint、acceptance/winner 位置。
7. 风险、异常、未完成项和下一步。

若训练仍在运行，只能报告“正在运行”和真实进度。除非涉及重要数据删除、破坏性 Git、不明进程、付费资源、授权范围扩大或改变研究目标的互斥决定，否则持续自主推进、监测和安全修复直至四路终态与 winner 决策完成。
```
