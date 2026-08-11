# R11 四卡 World-Action Model 正式执行 AI Coding Prompt（历史）

> **已归档：禁止执行。** R11 与 R12 均已关闭且无 winner；本 prompt 只用于只读审计，不得写入原 run root、启动训练或作为新路线父节点。完整历史见 [R11/R12 失败技术路线归档](../archive/20260811_R11_R12_FAILED_TECHNICAL_ROUTES_ZH.md)。

```text
====================
一、执行变量（每轮优先修改这里）
====================

[PROJECT_ROOT] = /home/jeong/zeno/wam/before-we-act
[ROADMAP_DOC] = /home/jeong/zeno/wam/before-we-act/docs/archive/20260811_R11_R12_FAILED_TECHNICAL_ROUTES_ZH.md
[RESULT_DOC] = 与 ROADMAP_DOC 相同；详细实验记录写入 docs/experiments/r11/

[STAGE] = R11
[STAGE_DESIGN_SECTION] = ROADMAP_DOC 第 7 至第 12 节
[STAGE_ACCEPTANCE_SECTION] = ROADMAP_DOC 第 11 节
[BASELINE_STAGE] = W10 六任务
[BASELINE_BRANCH] = feat/model-improvements
[BASELINE_COMMIT] = AUTO_RESOLVE_ORIGIN_HEAD_AND_FREEZE_IN_RUN_MANIFEST
[BASELINE_CHECKPOINT] = /workspace/bwa_runs/w10-six-task-v1/train/formal/checkpoint_120000.pt
[BASELINE_CHECKPOINT_SHA256] = e1b07b2cf7bff37428bf54a27f545632c8a1013930d96f6e646d8ca055f2f574
[BASELINE_VALIDATION] = 88/120；Lift 20，Camera 8，Long 20，Photo 20，Shoe 20，Food 0
[DATASET_VERSION_POLICY] = 先比对当前六任务 manifest/HDF5 receipt 与 BASELINE_CHECKPOINT provenance；数据变化时先重训并重新验证 W10，再冻结新的 baseline/checkpoint/hash
[W10_INITIALIZATION_POLICY] = 允许需要 W10 action prior 的候选用“与当前数据一致”的 W10 checkpoint warm-start；继承部分默认解冻并联合训练；禁止评测期 fallback 到 W10 动作

[CANDIDATE_A] = R11VJEPA21ACRefine；按路线第 9.2 节
[CANDIDATE_B] = R11DreamZeroWan22WAM；按路线第 9.3 节
[CANDIDATE_C] = R11CosmosPolicyLatent；按路线第 9.4 节
[CANDIDATE_D] = R11LaWAMSubgoalFlow；按路线第 9.5 节

[BRANCH_A] = feat/r11-vjepa21-ac-refine
[BRANCH_B] = feat/r11-dreamzero-wan22-wam
[BRANCH_C] = feat/r11-cosmos-policy-latent
[BRANCH_D] = feat/r11-lawam-latent-subgoal

[UPSTREAM_A] = https://github.com/facebookresearch/vjepa2.git@204698b45b3712590f06245fbfba32d3be539812
[UPSTREAM_B] = https://github.com/dreamzero0/dreamzero.git@ab790c198fbce33503358efbbd4187ce9a89adf3
[UPSTREAM_C] = https://github.com/nvidia-cosmos/cosmos-predict2.5.git@a2c298b0a3df3778b973fe65e9e58877b292d8a7
[UPSTREAM_D] = https://github.com/RLinf/LaWAM.git@4ea6fdadce6c9b8746028307a246b79ee2c4fd55

[REMOTE_SSH] = ssh -p 10328 root@69.176.92.104
[REMOTE_REPO] = /workspace/fe-pc-wam
[REMOTE_WORKTREE_ROOT] = /workspace/r11_worktrees
[REMOTE_RUN_ROOT] = /workspace/bwa_runs/r11-four-way-v1
[REMOTE_DATA_ROOT] = /workspace/datasets/robofactory_multitask
[REMOTE_UPSTREAM_ARTIFACTS] = /workspace/artifacts/r11_upstream
[HF_CACHE] = /workspace/.cache/huggingface
[W10_SEED_ROOT] = /workspace/bwa_runs/w10-six-task-v1/seeds/validation
[PYTHON_BASE] = /venv/robofactory-act/bin/python

[GPU_A] = 0
[GPU_B] = 1
[GPU_C] = 2
[GPU_D] = 3
[GPU_TYPE] = NVIDIA RTX 5090 32GB
[SESSION_MANAGER] = tmux
[SESSION_A] = bwa-r11-a-vjepa
[SESSION_B] = bwa-r11-b-dreamzero
[SESSION_C] = bwa-r11-c-cosmos
[SESSION_D] = bwa-r11-d-lawam
[SESSION_MONITOR] = bwa-r11-monitor

[EFFECTIVE_BATCH] = 48；每 optimizer update 六任务各 8 个 local-agent 样本
[DISCOVERY_UPDATES] = 1000
[SELECTION_UPDATES] = 20000
[FORMAL_UPDATES] = 120000
[ACTION_HORIZON_COMMON_TARGET] = 100；上游短 chunk 必须记录 receding-horizon cadence
[VALIDATION_DISCOVERY] = 每任务固定 5 回合
[VALIDATION_FORMAL] = 复用 W10 每任务固定 20 回合
[CONFIRMATION] = winner 与 W10 使用新冻结 seeds，每任务 50 回合

[HF_DOWNLOAD_POLICY] = 沿用 S0 的共享 cache、镜像/代理、断点续传、离线复用和环境变量或 mode-0600 secret/FIFO 鉴权；绝不回显或提交 token
[MERGE_POLICY] = 只有通过路线第 11 节 Confirmation50 的最终 winner 才合入 BASELINE_BRANCH；其他分支保留结果但不混合
[CLEANUP_POLICY] = 只能清理本阶段明确拥有且已有 receipt 的临时文件；不得删除数据集、W10 checkpoint、共享 HF cache、其他实验或不明进程/session

====================
二、目标
====================

完整阅读 ROADMAP_DOC 后，自主实现、训练、验证和验收 STAGE 的四个独立候选。目标不是强制闭环超过 W10，而是按 STAGE_ACCEPTANCE_SECTION 找到至少一个闭环接近 W10、预测真实影响动作、运行稳定的综合胜者。若无人满足门槛，必须如实给出“无胜者”，不得降低门槛、用 W10 fallback、只看 loss 或伪造完成状态。

本轮要求真正参考并迁移四个官方开源仓库。不能把“阅读论文后自行写一个类似模块”称为源码迁移。若使用 world model，预测必须进入动作生成的计算图、联合采样或候选选择路径；仅增加辅助预测 loss 不合格。

====================
三、开始前必须完成
====================

1. 完整阅读 ROADMAP_DOC，尤其是 W10 数据口径、STAGE_DESIGN_SECTION 和 STAGE_ACCEPTANCE_SECTION；不要依赖旧 R11+ archive 作为活动实现。
2. 在 PROJECT_ROOT 检查 `git status --short --branch`、当前 commit、remote、未推送提交和用户已有修改。不得覆盖无关 dirty changes。
3. 用 `rg` 阅读 W10 模型、六任务 trainer、evaluator、manifest schema、归一化、固定 seed、max steps、launcher、monitor、stop、model registry 和 checkpoint loader 的完整调用链。
4. 只读连接远程，记录 GPU、磁盘、数据、HF cache、Python/CUDA/PyTorch、tmux 和当前进程；不得终止或删除不属于本阶段的对象。
5. 解析 `origin/BASELINE_BRANCH` 当前 commit 为唯一 `R11_BASE_COMMIT`，写入本地和远程 immutable run manifest。四候选必须从该 commit 创建，不允许从彼此分支派生。
6. 核验 BASELINE_CHECKPOINT 的 SHA256、checkpoint 内训练数据 receipt、当前六任务 manifest/HDF5 receipt、W10 Validation20 JSON 和固定 seeds。若数据 receipt 完全相同但 checkpoint hash 不一致，停止启动并诊断来源；若数据 receipt 已变化，则按 DATASET_VERSION_POLICY 先在当前数据上重训 W10、重跑 Validation20，并把变量区的 baseline checkpoint/hash/result 和 ROADMAP_DOC 更新后再启动 R11。
7. 重新查看四个 UPSTREAM 的官方仓库和论文是否有安全/兼容性修复。不得静默换 commit；如必须升级，先在 ROADMAP_DOC 写出旧/新 commit、变更和理由，提交推送后再实现。

====================
四、源码迁移和许可证
====================

对每个候选：

1. 以只读方式 clone 对应 UPSTREAM 的固定 commit，禁用不必要的 LFS 文件下载；记录完整 URL、commit 和拉取时间。
2. 列出真正需要的上游类、函数、配置和测试，迁移最小依赖闭包到 `third_party/r11/<source>/` 或使用 launcher 校验过的只读 vendor checkout。
3. 生成 `SOURCE_RECEIPT.json`，包含逐文件 upstream path、local path 和 SHA256；复制正确 LICENSE、NOTICE、SPDX header。Cosmos 权重遵守 NVIDIA Open Model License，并在相关文档写 `Built on NVIDIA Cosmos`。
4. 在 `docs/experiments/r11/<candidate>_MIGRATION_ZH.md` 做逐符号映射，明确原样部分、适配部分和本项目新代码。
5. 先实现官方固定张量 parity test，再实现 RoboFactory adapter test。parity 失败不能开始训练。
6. 不提交下载的模型权重、数据、token、缓存或大日志。任何 secret 扫描命中都必须修复并重写尚未推送的本阶段提交，但禁止破坏共享 Git 历史。

====================
五、四分支独立实现
====================

分支 A 只实现 V-JEPA 2.1 + 2-AC proposal/predict/refine 路线；分支 B 只实现 DreamZero Wan2.2 causal video/action co-training；分支 C 只实现 Cosmos Policy latent action/future/value 联合模型；分支 D 只实现 LaWAM latent visual subgoal + flow action expert。严格执行 ROADMAP_DOC 第 9 节，不在一个分支混入另一候选模块。

每个分支必须独立具备：

- 模型和 factory/registry；
- 六任务 dataset adapter 与 exact effective-batch sampler；
- 配置 schema 和冻结的默认配置；
- train、resume、inference、Validation5、Validation20；
- `normal/prediction_off/prediction_shuffled` 三种推理；
- future-vs-persistence、action-shuffle、prediction-to-action 因果验收；
- unit/parity/integration/resume tests；
- 一键单候选启动、monitor、graceful stop；
- source receipt、license notice、migration 文档和使用说明。

候选可以按 ROADMAP_DOC 和 W10_INITIALIZATION_POLICY 加载 W10 model state；需要 W10 action prior 时优先使用当前数据重训/重新验证后的 checkpoint，继承的 action path 默认解冻并联合训练。必须记录 `w10_init={none,checkpoint,retrained}`、W10 checkpoint hash、训练数据 receipt 和 trainable parameter map。禁止用 W10 生成评测伪标签或在候选失败时调用 W10 action。各自公开的 foundation checkpoint 也必须记录 HF repo、revision、文件 hash 和许可；动作生成和预测—动作耦合必须训练。

新增模型名后用 `rg` 审计所有白名单和分发点：model factory、CLI choices、config validation、checkpoint loader、inference adapter、evaluator、acceptance、launcher、monitor 和 resume。不得只修一个 registry。

====================
六、本地 Git 流程
====================

1. 所有长期代码先在本地分支完成；远程只部署已提交代码，不把远程临时修改当实现。
2. 每个候选在独立 local worktree 实现，提交信息标明候选和来源。共同运行器在从 R11_BASE_COMMIT 建立的 integration 分支实现，不复制候选模型代码。
3. 对每分支运行格式/静态检查、目标单测、CPU smoke；有本地 GPU 才做 GPU smoke，不因本地无 GPU 跳过远程 F1。
4. 检查 `git diff --check`、`git status`、source receipt、license、secret scan 和与其他候选的 diff 交叉污染。
5. 提交并推送四候选和 integration 分支。记录每个 commit；禁止 force push、reset --hard、覆盖用户提交。

====================
七、远程部署和四卡运行
====================

1. SSH 到 REMOTE_SSH，在 REMOTE_REPO fetch；为 A/B/C/D 建立 REMOTE_WORKTREE_ROOT 下四个独立 worktree，并逐一校验 branch/commit/base ancestry。
2. 使用共享 REMOTE_DATA_ROOT、HF_CACHE、REMOTE_UPSTREAM_ARTIFACTS 和 W10_SEED_ROOT；每候选独立 venv、output、checkpoint、log、status、heartbeat 和 tmux session。
3. GPU 固定为 A=GPU_A、B=GPU_B、C=GPU_C、D=GPU_D。launcher 和实际进程同时校验 `CUDA_VISIBLE_DEVICES`；禁止跨卡和借卡。
4. 先执行 `launch_r11_4gpu_tmux.sh --all --dry-run`，核验依赖、模型许可、缓存、磁盘、GPU 空闲、分支、commit、数据和 seeds。
5. 按 F0 → F1 → Discovery → Validation5 → Selection → Formal → Validation20 顺序运行。每候选独立前进；一个候选失败不能杀死其他候选。
6. F1 使用真实 forward/backward/optimizer/save/resume/inference，不能用缩小到不经过 world/action 主路径的假模型。显存降级只能按 ROADMAP_DOC 预注册顺序。
7. effective batch 始终为 48 且每 update 六任务各 8 个样本。micro-batch/accumulation 可因模型不同，但 checkpoint 必须保存 deterministic sample cursor 和 RNG；resume 后下一 sample key 必须一致。
8. 所有长期进程在各自 tmux 中运行。launcher 避免重复启动，记录 PID/session/GPU/branch/commit/start time，退出 SSH 后任务继续。
9. 普通代码、依赖、缓存和 checkpoint 问题先自主诊断。代码修复回到本地对应分支，测试、commit、push，再远程 fast-forward 更新和安全 resume；记录每次修复。

====================
八、统一 monitor 和安全停止
====================

实现并实际验证：

  scripts/before_we_act/launch_r11_4gpu_tmux.sh --all [--dry-run]
  scripts/before_we_act/launch_r11_4gpu_tmux.sh --candidate A
  scripts/before_we_act/monitor_r11.sh --all --once
  scripts/before_we_act/monitor_r11.sh --candidate A --watch
  scripts/before_we_act/stop_r11_4gpu_tmux.sh --candidate A --graceful
  scripts/before_we_act/stop_r11_4gpu_tmux.sh --all --graceful

monitor 必须同时显示四条进度条和后续队列，并显示：当前时间、候选、branch/commit/upstream commit、GPU、tmux、程序、阶段、状态、PID、启动/持续时间、实际心跳和 age、GPU 利用率/显存/温度/功耗、update/120000、ETA、action/world/value loss、预测 gain、最近验证、current/best checkpoint、日志、OOM/NaN/异常退出/stale 告警、三种因果模式和 STAGE_ACCEPTANCE_SECTION 的逐项结果。

状态至少有 `NOT_STARTED/PREPARING/DOWNLOADING/PREFLIGHT/TRAINING/VALIDATING/ACCEPTING/PASSED/FAILED/FAILED_FIT/STOPPED/STALE/UNKNOWN`。心跳由实际 worker 或核验 PID start-time 的 watchdog 更新，不能因日志文件存在就判活。PASSED/FAILED 只读结构化 acceptance JSON。

graceful stop 必须精确识别 run manifest 中的 PID start-time 和 tmux session，先请求保存 checkpoint/停止，再结束本候选；不得 `pkill -f` 模糊匹配，不得动其他 session/process。

====================
九、验收和 winner
====================

1. 每个阶段检查 exit code、日志完整性、心跳、GPU 使用、OOM/NaN/卡死、断点连续性、样本计数和 checkpoint provenance。
2. 对正式 checkpoint 跑完全相同的固定 Validation20；不得更换 seed、max steps、成功条件或遗漏失败回合。
3. 跑 future-vs-persistence、action condition shuffle、prediction_off 和 prediction_shuffled；相同任务文本必须保留。
4. 使用 STAGE_ACCEPTANCE_SECTION 的原始门槛和冻结公式生成 `acceptance.json` 与 `winner.json`。输出逐项证据，不能只给总 score。
5. 无合格候选时写“R11 无胜者”，保留失败分析，不合并。有人合格时选临时 winner，再冻结新 Confirmation50 seeds，同时评测 W10 和 winner。
6. 只有 Confirmation50 非劣性也通过，才把 winner 合入 BASELINE_BRANCH，解决冲突、跑回归、提交并推送；不得合并其他候选组件来补 winner。

====================
十、结果文档和最终汇报
====================

训练开始、每个 gate、正式验证和 winner 决策后，都更新 ROADMAP_DOC 第 13 节摘要，并把完整命令/配置/指标/日志位置写入 `docs/experiments/r11/`。文档修改先在本地完成、检查、commit、push；不要在远程长期维护文档。

记录：执行时间、local/remote branch 和 commit、upstream commit/license、运行环境、完整可复制命令、数据/seed/hash、HF cache、GPU/tmux/PID、micro-batch/accumulation、输出/log/checkpoint/status/heartbeat、训练和预测指标、Validation5/20/50、因果消融、逐项验收、score、失败根因、winner/无 winner 结论和后续建议。不得在文档或最终汇报回显 token。

最终按以下结构汇报：

1. 当前结论：四候选状态、R11 winner/无 winner、是否合并。
2. Git 状态：base、四分支、integration、commit、push、文档 commit。
3. 远程状态：四个 tmux/GPU/程序/阶段/进度/心跳/log/checkpoint。
4. 实验结果：训练、预测、闭环、因果 gate、资格门槛、score、Confirmation50。
5. 可复制命令：deploy/train、monitor、graceful stop、acceptance、reproduce winner。
6. 文档位置和更新章节。
7. 风险、异常、未完成项和下一步。

若训练仍在运行，只能报告“正在运行”及真实进度，不得称完成。除非涉及重要数据删除、破坏性 Git、不明进程、付费资源或会改变研究目标的互斥决定，否则持续自主推进、监测和安全修复直至四路结果及 winner 决策完成。
```
