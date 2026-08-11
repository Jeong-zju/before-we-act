# P1 多机器人闭环模型技术路线 V6.2：W10 基线、R11 结论与 LaWAM 受控消融方向

> 更新日期：2026-08-11
> 活动分支：`feat/model-improvements`
> 当前状态：六任务 W10 基线和数据口径已重新核验；R11 四候选均已独立实现并推送。A/B/D 通过真实 F1 后均在 Discovery gate 按冻结公式失败，C 因冻结 Cosmos 权重授权 403 失败关闭；最终结论为“R11 无胜者”，未生成 Confirmation50 seeds，也未发生 winner/checkpoint 晋级。后续研究选择 LaWAM latent subgoal 路线作为改进起点，并于 2026-08-11 将其源码迁移、受控消融所需公共运行器和测试以 `6cd891b` 合入 `feat/model-improvements`；这是实验代码整合，不改变 LaWAM 在 R11 中仍为 `FAILED`、不是 winner 的结论
> 活动任务：Lift Barrier、Camera Alignment、Long Pipeline Delivery、Take Photo、Pass Shoe、Place Food；不包含任何 Stack Cube 任务

## 1. 当前决定

从现在起，后续模型改进只与六任务 W10 比较。除第 13 节明确回引的 LaWAM 源码迁移、受控消融所需公共运行器和测试外，原 R11、R12、R13/R13N、R14 和 R15 的代码、配置、训练入口、验收脚本及实验清单不再属于活动路线，也不能作为新候选的隐式组成部分。被回引的 R11 代码只是新阶段的实现起点，不携带 R11 checkpoint、验收状态或 winner 身份。

本次采用普通 Git 回撤提交，不改写历史。需要追溯旧实验时只读以下归档：

- [旧路线完整历史](../archive/20260725_P1_MULTI_ROBOT_MODEL_ARCHITECTURE_ACTION_GENERATION_ROADMAP_V2.0_FULL_HISTORY_ZH.md)

归档只用于审计，里面的阶段状态、命令和分支不代表当前执行计划。

## 2. W10 六任务基线

### 2.1 模型和训练口径

W10 使用 `NoWristPAIRRoute`：每个机器人读取当前全局固定相机 RGB、与自己对应的固定相机 RGB 和自己的 qpos，输出本机器人 8 维动作块。视觉主干为冻结的 DINOv3 ViT-B/16，保留完整 `30×40` token 网格；动作 horizon 为 100。模型不读取任务 ID、机器人 ID、语言、腕部图像、深度、其他机器人的状态或未来信息。

六任务训练固定为：

- 120,000 optimizer updates；batch size 48；每个 batch 每任务恰好 8 个样本。
- 共 5,760,000 个局部动作样本，六任务采样预算完全相同。
- qpos/action 归一化统计直接从训练 HDF5 的物理量重新计算。
- Place Food 没有 agent camera；训练与推理都确定性复用同一时刻的 global RGB 作为 local RGB。这是输入接口对齐，不是额外信息。
- 训练代码集成提交：`87a6153cb9d0b7899066a82c69daeea1c70996a9`。
- R11+ 回撤与新 baseline 固化提交：`6b82f8902b7bf85422b6964576c23768278b0763`。
- 远程正式训练提交：`a335a2dfdf429058f4543a15c66a1d6c6738c77b`（远程实验分支 `bwa/w10-six-task`）。远程验证时使用的 complete-resume 和 Place Food 相机回退修复已纳入当前主分支回撤提交。

### 2.2 正式 Validation20 结果

完成时间为 2026-08-10 07:48:27 UTC。六项任务使用固定 seed 文件，每项 20 回合，共 120 回合。

| 任务 | 成功数 | 成功率 | 基线含义 |
|---|---:|---:|---|
| Lift Barrier | 20/20 | 100% | 后续候选必须保护 |
| Camera Alignment | 8/20 | 40% | 第一改进目标之一 |
| Long Pipeline Delivery | 20/20 | 100% | 后续候选必须保护 |
| Take Photo | 20/20 | 100% | 后续候选必须保护 |
| Pass Shoe | 20/20 | 100% | 后续候选必须保护 |
| Place Food | 0/20 | 0% | 最高优先级改进目标 |
| **合计** | **88/120** | **73.33%** | **后续模型的唯一活动 baseline** |

这里结构化结果中的 `status=PASSED` 只表示 120 个验证回合完整执行、产物齐全，并不表示六任务都解决。模型质量结论必须读取逐任务成功数：Place Food 尚未成功，Camera Alignment 仍不稳定。

可审计结果：

- 仓库内结构化摘要：[20260810_W10_SIX_TASK_VALIDATION20.json](../reports/20260810_W10_SIX_TASK_VALIDATION20.json)
- 远程 summary：`/workspace/bwa_runs/w10-six-task-v1/evaluation/validation/summary.json`
- 远程总日志：`/workspace/bwa_runs/w10-six-task-v1/logs/validation_supervisor.log`
- 远程逐任务日志：`/workspace/bwa_runs/w10-six-task-v1/logs/validation_<task>.log`
- 本地验证产物备份：`/home/jeong/zeno/wam/remote_backups/w10_six_task_120k/validation/`

### 2.3 Checkpoint 和运行环境

| 项目 | 值 |
|---|---|
| checkpoint | `/workspace/bwa_runs/w10-six-task-v1/train/formal/checkpoint_120000.pt` |
| checkpoint SHA256 | `e1b07b2cf7bff37428bf54a27f545632c8a1013930d96f6e646d8ca055f2f574` |
| 训练状态 | `120000/120000`，`PASSED` |
| 数据根目录 | `/workspace/datasets/robofactory_multitask` |
| DINOv3 | `/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m` |
| Hugging Face 缓存 | `/workspace/.cache/huggingface`；继续沿用 S0 缓存和鉴权约定 |
| Python | `/venv/robofactory-act/bin/python` |
| GPU | 4× NVIDIA RTX 5090；训练使用 GPU0，验证分两波并行使用 GPU0–3 |
| 训练输出 | `/workspace/bwa_runs/w10-six-task-v1/train/formal` |
| 验证输出 | `/workspace/bwa_runs/w10-six-task-v1/evaluation/validation` |
| 固定 seeds | `/workspace/bwa_runs/w10-six-task-v1/seeds/validation` |

验证结束后训练和验证进程均已自然退出，没有 OOM、NaN 或残留 W10 GPU 进程。`checkpoint_120000.pt` 的哈希由正式 summary 固化；不得用同名不同哈希文件替换它。

## 3. R11 及以后回撤范围

活动代码树已撤回以下内容：

| 旧阶段 | 已撤回内容 | 当前处理 |
|---|---|---|
| R11 | team-belief / V-JEPA2 predictor、四卡训练验收和上游组件移植 | 不继承；新 R11 从 W10 重新设计 |
| R12 | action generator、ACT/P2/R4/evolution 路线、缓存与恢复诊断 | 不继承 |
| R13 / R13N | world/action baseline、六任务 R13N 训练评测链 | 不继承；其成绩不再是 baseline |
| R14 | world-guided decision 组件和验收路线 | 不继承 |
| R15 | Stack 专项、专家数据、portfolio/role-query 尝试 | 不继承；Stack 不在活动任务集 |

回撤也覆盖相应模型注册、白名单、配置、requirements、许可证副本、测试、monitor、一键启动和停止脚本。保留的只有 W10 六任务训练/验证链、W10 之前仍通用的仓库能力，以及只读历史归档。

## 4. 新 R11 的目标和硬约束

R11 不再要求闭环成功数严格超过 W10；目标是让至少一个真正使用预测的 World-Action Model 在闭环、预测可信度、因果耦合、稳定性和运行代价的综合评价中成为胜者，同时闭环表现接近 W10。这里“接近”在第 11 节量化，不能事后修改。

四个候选必须从同一个 `feat/model-improvements` commit 建立独立分支。它们可以加载各自论文公开的基础视觉/视频/VLM 权重，也可以在方案明确需要 W10 action prior 时用 W10 checkpoint 初始化，但不能把 W10 冻结成不训练的黑盒或在推理失败时 fallback 到 W10 动作。动作生成器和世界预测到动作的耦合必须训练；继承的 W10 action path 默认也要联合训练。

执行前先比较当前六任务 manifest/HDF5 receipt 与 W10 checkpoint 中冻结的训练数据哈希：

- 数据完全相同：允许使用已验证的 W10 checkpoint warm-start，但配置必须明确记录 `w10_init=checkpoint`，并默认解冻参加联合训练；也可以选择从随机动作头重新训练。
- 任一任务数据发生变化：旧的 `88/120` 和旧 checkpoint 立即降级为历史参考。先用更新后的六任务数据按 W10 口径重新训练、重新跑 Validation20 并冻结新的 checkpoint/hash/baseline，再启动所有需要 W10 初始化或与 W10 比较的 R11 实验。
- 无论数据是否变化：禁止用 W10 输出作为评测期 fallback；否则无法判断 R11 自身是否有效。

所有候选共享以下部署输入：当前 global RGB、与机器人匹配的 agent RGB、自己的 qpos，以及 manifest 中已有的规范化任务文本。Place Food 仍确定性地用 global RGB 填补缺失的 agent RGB。禁止腕部 RGB-D、其他机器人状态/动作、未来真值和评测器内部状态。由于 W10 没有任务文本，R11 与 W10 是系统级比较，不声称是单一 world-model 组件的纯净消融；四个 R11 候选之间使用完全相同的任务文本。预测因果消融也保留相同文本，避免把文本收益误记为预测收益。

训练和评测必须满足：

1. 六任务、训练 HDF5、归一化、固定 Validation20 seeds、环境 max steps 和成功判据与 W10 相同。
2. 每个 optimizer update 的有效 batch 仍为 48，并且六任务各 8 个 local-agent 样本；显存不足只能改变 micro-batch 和梯度累积，不能改变有效采样。
3. 正式预算仍为 120,000 optimizer updates，即 5,760,000 个 local-agent 样本；短程 gate 只决定是否值得继续，不能当作正式结果。
4. 预测必须进入动作生成的计算图或采样/选择路径。只加预测 loss、推理时不用预测，或只生成好看的视频，均不算 R11 实现。
5. 每个候选都要提供 `normal`、`prediction_off`、`prediction_shuffled` 三种推理模式，并输出结构化因果验收结果。
6. 每条分支独立实现、独立依赖锁、独立输出/checkpoint/log/status/heartbeat/tmux；不得混合四条方法。

## 5. 可直接复现的命令

### 5.1 训练或从 checkpoint 续跑

```bash
cd /workspace/fe-pc-wam
CUDA_VISIBLE_DEVICES=0 \
W10_SIX_DATA_ROOT=/workspace/datasets/robofactory_multitask \
W10_SIX_DINO_MODEL=/workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m \
W10_SIX_OUTPUT=/workspace/bwa_runs/w10-six-task-v1/train/formal \
scripts/before_we_act/launch_w10_six_task.sh
```

launcher 检测到 `checkpoint_latest.pt` 时会自动续跑；如果 checkpoint 已达到 120,000 updates，则只校验并保持 `PASSED`，不会重复训练。

### 5.2 验证前安全检查

```bash
cd /workspace/fe-pc-wam
W10_SIX_RUN_ROOT=/workspace/bwa_runs/w10-six-task-v1 \
W10_SIX_SEED_ROOT=/workspace/bwa_runs/w10-six-task-v1/seeds/validation \
scripts/before_we_act/validate_w10_six_task.sh --dry-run
```

### 5.3 固定种子六任务正式验证

```bash
cd /workspace/fe-pc-wam
tmux new-session -d -s bwa-w10-six-validation \
  "cd /workspace/fe-pc-wam && \
   W10_SIX_RUN_ROOT=/workspace/bwa_runs/w10-six-task-v1 \
   W10_SIX_SEED_ROOT=/workspace/bwa_runs/w10-six-task-v1/seeds/validation \
   scripts/before_we_act/validate_w10_six_task.sh \
   >> /workspace/bwa_runs/w10-six-task-v1/logs/validation_supervisor.log 2>&1"
tmux attach -t bwa-w10-six-validation
```

已完整生成的逐任务 JSON 会被保留，重跑时只补未完成任务。查看最终结果：

```bash
jq '{status,successes,episodes,tasks}' \
  /workspace/bwa_runs/w10-six-task-v1/evaluation/validation/summary.json
```

## 6. 当前里程碑

| 里程碑 | 状态 | 完成条件 |
|---|---|---|
| W10 六任务训练 | **完成** | 120,000 updates 和 checkpoint 固化 |
| W10 Validation20 | **完成** | 120/120 回合完整，结果 88/120 |
| R11+ 旧实现收口 | **完成（LaWAM 受控回引）** | 仅回引 LaWAM 及必要 R11 公共运行器；其他旧 stage 仍为非活动归档 |
| 新 R11 论文与源码调研 | **完成** | 四个来源、commit、许可证、迁移边界已冻结 |
| 新 R11 方案设计 | **完成** | 四卡候选、分段训练、因果验收和胜者规则已预注册 |
| 新 R11 训练与验收 | **已终结（无胜者）** | 四候选按预注册 gate 形成终态，失败即不进入后续预算 |

R11 已从唯一冻结 base `78471b285bc69fa8b5168fb170a3c3332efc32be` 建立四个独立候选分支和 integration 分支。A/B/D 都通过真实 F1 fresh/save/resume/inference 并完成 Discovery1000，但均未通过冻结 causal gate；C 的 official gated foundation revision 返回 403，未绕过许可 gate。2026-08-11 08:30（Asia/Shanghai）决策器冻结 `NO_WINNER`，因此当时没有 Validation20/Confirmation50，也没有 winner baseline merge。完整命令、提交、修复栈和产物路径保留在 `feat/r11-four-way-integration@678e67780e6960749410ee0649ce961b10495950` 的 `docs/experiments/r11/R11_EXECUTION_LOG_ZH.md`。同日后续将 LaWAM 实验代码合入 `feat/model-improvements` 只用于第 13 节的新受控消融；该源码合并不追溯修改冻结的 `winner.json`、`acceptance.json` 或 `merged_to_baseline=false` 实验事实。

## 7. 2026-08-10 最新论文和开源代码调研

### 7.1 采用的四个上游

| 路线 | 论文和可迁移机制 | 官方代码与冻结 commit | 许可证/权重 | 选择判断 |
|---|---|---|---|---|
| V-JEPA 2.1 + V-JEPA 2-AC | V-JEPA 2.1 的 dense predictive features；2-AC 的 action-conditioned latent predictor | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) @ [`204698b`](https://github.com/facebookresearch/vjepa2/commit/204698b45b3712590f06245fbfba32d3be539812) | 代码主体 MIT，部分文件 Apache-2.0；逐文件保留 notice | 最新 JEPA 路线，ViT-B/16 规模最有希望在单卡完成；但 2.1 与 2-AC 不是上游现成组合，必须做接口和 checkpoint parity gate |
| DreamZero | causal video/action chunk、flow-matching action head、闭环真实观测刷新 | [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) @ [`ab790c1`](https://github.com/dreamzero0/dreamzero/commit/ab790c198fbce33503358efbbd4187ce9a89adf3) | 代码 Apache-2.0；Wan2.2-TI2V-5B 权重 Apache-2.0 | 最忠实的像素/视频 WAM 路线；5B 在 32GB 单卡训练风险最高，先做真实显存 gate |
| Cosmos Policy / Predict2.5 | action、future state、value 都编码为 latent frames；联合训练并用 value 选择动作 | [NVIDIA Cosmos Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) @ [`a2c298b`](https://github.com/nvidia-cosmos/cosmos-predict2.5/commit/a2c298b0a3df3778b973fe65e9e58877b292d8a7) | 代码 Apache-2.0；2B 权重为 NVIDIA Open Model License，必须保留 attribution 和 `Built on NVIDIA Cosmos` | 预测、动作和选择天然在同一路径；官方训练是多卡配方，本轮只允许单卡 LoRA/低秩适配，不伪称原配方复现 |
| LaWAM | latent-action-conditioned latent world model；预测 latent visual subgoal 直接注入 flow action expert | [RLinf/LaWAM](https://github.com/RLinf/LaWAM) @ [`4ea6fda`](https://github.com/RLinf/LaWAM/commit/4ea6fdadce6c9b8746028307a246b79ee2c4fd55) | `pyproject.toml` 和 README 声明 MIT；NVIDIA 派生 dataloader 文件仍按 Apache-2.0；迁移时逐文件审计 | 2B VLM + latent prediction，官方公开训练/评测链；计算更轻，是四路中最可能接近 W10 的风险对冲 |

对应论文：

- [V-JEPA 2](https://arxiv.org/abs/2506.09985) 与 [V-JEPA 2.1](https://arxiv.org/abs/2603.14482)：前者公开 action-conditioned world model，后者改进稠密和时序一致特征。
- [DreamZero: World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)：视频和动作联合建模，而不是先生成视频再由旁路策略忽略它。
- [Cosmos Policy](https://arxiv.org/abs/2601.16163)：把动作、未来状态和 value 放进视频模型 latent diffusion，并支持 test-time planning。
- [LaWAM](https://arxiv.org/abs/2606.15768)：不重建像素视频，预测 latent visual subgoal 并直接条件化动作生成。

### 7.2 调研后未进入四卡首轮的方向

| 方向 | 不进入首轮的原因 | 后续位置 |
|---|---|---|
| Cosmos 3 | 当前仓库已发布推理能力，但机器人 policy post-training recipe 仍未形成比 Predict2.5 更稳定的可执行基线 | R12 再复核，不在 R11 中冒充已开放训练路线 |
| Kairos3.1 | 官方仓库当前主要是 pipeline、配置和推理权重，没有与本项目数据可直接对接的训练代码 | 只作 persistent world-state 设计参考 |
| MaskWAM | 官方 README 截至调研日仍明确写着 training/inference/evaluation code 正在准备 | 等代码发布后再候选化 |
| X-WAM / StarVLA WM4A | 已有 post-training/集成代码，但同样依赖 Wan/Cosmos 大模型，与 DreamZero/Cosmos 两卡高度重叠 | 用作数据适配和单卡工程参考，不占独立科学候选 |
| AHEAD、DSWAM、WALL-WM 等 | 论文机制有价值，但本轮未找到同时满足“官方、训练代码完整、许可证明确、单卡可迁移”的版本 | 可进入后续论文复核队列，不能只凭摘要写进模型 |

### 7.3 调研结论

事实是：高质量视频生成不等于动作更好，未来预测如果不进入动作路径也不构成 world-guided policy。工程判断是让一条轻量 latent 路线、一条 JEPA 路线和两条较重 video diffusion 路线并行，既覆盖前沿，也避免四张卡都押在 5B 级像素生成上。研究判断是 R11 更可能由 V-JEPA/LaWAM 路线成为“接近 W10 的综合胜者”，DreamZero/Cosmos 更可能提供高上限或暴露计算瓶颈；这只是预注册假设，不是结果。

## 8. 共同迁移与数据协议

### 8.1 源码迁移证据

每个候选必须在自己的分支提交：

- `third_party/r11/<source>/SOURCE_RECEIPT.json`：URL、完整 commit、拉取时间、逐文件 SHA256、上游路径和本地路径。
- `third_party/r11/<source>/LICENSE*` 与 `THIRD_PARTY_NOTICES.md`；Cosmos 额外写入权重许可和 attribution。
- `docs/experiments/r11/<candidate>_MIGRATION_ZH.md`：逐符号列出“原函数/类 → 本地函数/类 → 为 RoboFactory 做了什么修改”。
- 至少一个上游 parity test 和一个本地 adapter test。parity test 必须用固定输入比较关键张量形状、有限值和数值误差。

允许迁移上游的核心模块和最小依赖，禁止把完整上游仓库直接塞进主仓库，也禁止训练依赖 `/tmp` 或某个未记录的外部 clone。若因为许可证或依赖无法复制，可以使用固定 commit 的只读 vendor checkout，但必须在 launcher 中校验 commit，不能退化为“看过论文后自行写一个同名模块”。

### 8.2 共同样本

新增一个共享 `R11EpisodeDataset` 和确定性 `ExactSixTaskAccumulationSampler`。每个 local-agent 样本包含：

- 当前时刻 global/local RGB、own qpos、任务文本；
- 未来 RGB 索引 `t+[1,4,8,16]`，超出 episode 的位置使用 mask，不能复制后标为真值；
- 长度 100 的动作真值和 valid mask；需要较短 action chunk 的上游可取其前缀，但仍记录执行 cadence；
- episode、task、agent、time、manifest SHA256 和采样 key，便于四分支逐样本对账。

任务文本来自 manifest 已有字段并在 run manifest 冻结。四个候选使用同一文本表；禁止从评测 seed、成功状态或未来帧构造 prompt。

### 8.3 训练预算和可比性

| 阶段 | 预算 | 目的 | 继续条件 |
|---|---:|---|---|
| F0 source parity | CPU/GPU 固定张量 | 证明迁移的是上游真实模块 | receipt、license、parity test 全通过 |
| F1 fit smoke | 2 optimizer updates | 完整 forward/backward/save/resume/inference；测单卡峰值 | 32GB 内无 OOM/NaN，resume 后 next-sample key 完全一致 |
| Discovery | 1,000 updates | 排除 loss 不下降、预测不看动作、吞吐不可接受 | action loss 相对最初 100 step 中位数下降；预测优于 persistence；shuffled action 使预测变差 |
| Validation5 | 每任务 5 个固定 discovery seeds | 早期闭环和接口检查，不作正式成绩 | 30/30 回合完整；无非法动作/超时；只用于诊断，不以低成功率直接淘汰 |
| Selection | 20,000 updates | 比较学习速度和世界—动作耦合 | 完成离线因果 gate；没有持续 OOM/NaN/断点不一致 |
| Formal | 120,000 updates | 与 W10 相同 5.76M 样本预算 | 只有前述工程和因果 gate 通过的候选继续；失败候选明确记 `FAILED`，不得换成 W10 |
| Validation20 | 120 回合 | 本轮正式排名 | 第 11 节预注册规则 |
| Confirmation50 | W10 与 winner 各 300 回合 | 新 seeds 上复核非劣性 | seeds 在打开结果前冻结；报告 paired bootstrap 区间和绝对成功数 |

F1 可以通过 bf16、activation checkpointing、gradient accumulation、8-bit optimizer、CPU offload 和 LoRA 解决显存问题，但不得借用其他候选 GPU。所有节省显存的开关和峰值显存必须记录。若 5B/2B 基础模型仍不能在单卡完成真实 backward，该分支结论就是资源约束下 `FAILED_FIT`，不能临时替换成没有源码对应关系的小 MLP。

## 9. 四卡 R11 候选设计

### 9.1 总表

| ID / GPU / 分支 | 候选 | 从上游实际迁移 | 预测如何进入动作 | 主要风险 |
|---|---|---|---|---|
| A / GPU0 / `feat/r11-vjepa21-ac-refine` | `R11VJEPA21ACRefine` | V-JEPA 2.1 encoder、V-JEPA 2-AC predictor、mask/token utilities | proposal action → AC latent future → refinement action decoder；推理时必须走两段 | 2.1 encoder 与 2-AC predictor checkpoint/shape 不同构 |
| B / GPU1 / `feat/r11-dreamzero-wan22-wam` | `R11DreamZeroWan22WAM` | `CausalWanModel`、DreamZero co-train transform、Wan flow action head/scheduler、closed-loop refresh | video latent 与 action token 在 causal chunk DiT 中联合去噪；真实观测按 chunk 刷新 | Wan2.2-5B 单卡训练吞吐和显存 |
| C / GPU2 / `feat/r11-cosmos-policy-latent` | `R11CosmosPolicyLatent` | Cosmos conditioner、policy video2world rectified-flow model、hybrid SDE/sampler、trainer objective routing | action/future/value latent frames 联合生成；K 个动作样本由 value latent 排序 | 官方多卡配方缩到单卡后的稳定性；只有专家数据导致 value 监督弱 |
| D / GPU3 / `feat/r11-lawam-latent-subgoal` | `R11LaWAMSubgoalFlow` | LaWAM latent-action model、future decoder、VLM-to-LAM mapping、flowmatching expert | 预测的 `h_t1_pred` 作为 future tokens 拼入 flow action expert | Qwen3-VL/LaWM 数据接口重，且上游预训练域与本项目不同 |

### 9.2 A：V-JEPA 2.1 action-conditioned refinement

实现保留 W10 的每机器人 8 维物理动作和 ARCA 多角色输出。proposal path 优先从“与当前数据哈希一致、已重新验证”的 W10 checkpoint 初始化；如果数据已更新，必须先完成前述 W10 重训，不能继续加载旧 checkpoint。proposal、AC predictor 和 refinement decoder 在 R11 中共同训练，W10 proposal 不冻结。V-JEPA 2.1 ViT-B/16 编码两路当前帧和未来监督帧；proposal decoder 先生成 100-step 动作，再把 proposal 的前 16 步及当前 latent 送入迁移的 `ACPredictor`，预测 `t+[1,4,8,16]` latent。refinement decoder 交叉注意当前 latent、预测 latent、qpos 和 proposal，输出最终 100-step 动作。

训练使用 `L = L_final_action + 0.25 L_proposal_action + λ_jepa L_future_latent + 0.05 L_consistency`。`λ_jepa` 在前 5k updates 从 0.1 线性升到 1.0；AC predictor 前 10k 以真值/提议动作 50/50 scheduled sampling，之后只用 proposal，防止训练时看真值、推理时失配。2.1 encoder 默认只训练最后两层和 adapter；若 F1 峰值超过 31GB，可冻结 V-JEPA encoder，但 W10 proposal、predictor 和 refiner 仍须训练。

主要移植文件映射：`app/vjepa_2_1/models/{vision_transformer,predictor}.py`、`src/models/ac_predictor.py`、`app/vjepa_droid/{droid,transforms,utils}.py`。不能只复制 predictor 类名后重写内部实现。

### 9.3 B：DreamZero Wan2.2-5B causal WAM

使用官方较小的 `Wan-AI/Wan2.2-TI2V-5B`，以 LoRA 训练 video DiT，动作 head 和跨模态投影全部新初始化并全量训练。输入视频统一缩放到官方代码支持的 `160×320`；global/local 作为同一时间点的双视图 token 段，带 view embedding。动作 horizon 使用上游 24，闭环每 12 步重新读取真实观测并刷新 causal cache；环境 max steps 不变。

移植 `dreamzero_cotrain.py`、`wan_flow_matching_action_tf.py`、`wan_video_dit_action_casual_chunk.py` 和 `flow_match_scheduler.py`，保留 action/video 联合 flow loss、causal chunk attention 和 inference observation refresh。不得先离线生成未来视频，再让一个与视频 latent 断开的 W10/ACT head 出动作。

F1 默认 micro-batch 1、gradient accumulation 48、bf16、gradient checkpointing、8-bit AdamW、CPU-offload text encoder/VAE，LoRA 从 rank 16 起。F1 仍 OOM 时只能按预注册顺序降到 rank 8、减少未来视频帧但保留至少一个预测帧；仍失败则 `FAILED_FIT`。

### 9.4 C：Cosmos Policy latent-frame planning

以 `nvidia/Cosmos-Predict2-2B-Video2World` 为基础，迁移 Cosmos Policy 的 conditioner、rectified-flow policy model、hybrid SDE 和 sampler。动作 8 维、未来 qpos/RGB latent 和 value 被编码为不同类型的 latent frames，训练 objective 预注册为 policy 50%、world 25%、value 25%。由于现有 HDF5 是专家示范而不是带稠密 reward 的 rollout，value 正样本为专家 chunk，负样本只允许使用同 batch 的跨时刻/跨 agent 动作和预注册幅度的动作扰动；不能用评测结果反向标注训练集。文档必须单列这个与原论文监督的差异。

推理从同一当前观测采样 `K=4` 个 action/future/value 联合轨迹，使用模型自己的 value latent 排序并执行最高者前 25 步；若 K=4 超出时间/显存，F1 可降到 K=2，但不能降到 K=1 后继续声称做了 value-guided planning。`prediction_off` 把 future/value latent 屏蔽但保留文本，`prediction_shuffled` 在 batch 内置换 future latent。

默认 micro-batch 1、accumulation 48、bf16、activation checkpointing、8-bit optimizer、DiT LoRA rank 16。首次下载前必须非交互确认 NVIDIA Open Model License 已在服务器账户接受，并生成不含 token 的 license receipt；任何文档和模型卡保留 `Built on NVIDIA Cosmos`。

### 9.5 D：LaWAM latent visual subgoal

迁移 `latent_action_model/core`、`LatentWorldPolicyBackend`、`VLMToLAMQFormer` 和 `ConditionalFlowMatchingHead` 的最小闭包。使用 Qwen3-VL-2B 和 DINOv3 ViT-B/16 基础权重；该架构与 W10 action path 不同，默认不需要 W10 初始化。LaWM decoder、VLM-to-LAM 映射和 flow action expert 按上游训练逻辑训练。首先尝试加载公开 LaWAM pretrain 作为基础 world representation，再在六任务上重新 SFT 整个 action path；同时保存一个不加载 LaWAM task-SFT 权重的 provenance 证明。

动作 horizon 对齐到 100；`h_t1_pred` future tokens 必须像上游 `flowmatching_expert.py` 一样拼进 action DiT hidden states。训练 loss 包括 flow action、future latent perceptual 和 LAM encoder distillation；scheduled sampling 从 GT-conditioned future 逐步切到 predicted future。默认 micro-batch 2、accumulation 24，若 F1 实测允许再增大 micro-batch。

LaWAM 根目录没有独立 `LICENSE` 文件，但 `pyproject.toml` 与 README 明确声明 MIT，同时部分 NVIDIA 派生文件带 Apache-2.0 SPDX。执行时必须逐文件生成 license map，不能只放一个 MIT 文件覆盖派生文件 notice。

## 10. 分支、目录、GPU 和长期运行

共同 base branch 为 `feat/model-improvements`。执行前解析并在 `run_manifest.json` 冻结同一个 `R11_BASE_COMMIT`；四分支必须都是该 commit 的直接后代，且候选 diff 交叉审计不得包含另一候选模型。远程使用四个 worktree，禁止在一个 checkout 中来回切分支。

| ID | GPU | tmux | remote worktree | run/checkpoint/log/status |
|---|---:|---|---|---|
| A | 0 | `bwa-r11-a-vjepa` | `/workspace/r11_worktrees/a-vjepa` | `/workspace/bwa_runs/r11-four-way-v1/A/{checkpoints,logs,status}` |
| B | 1 | `bwa-r11-b-dreamzero` | `/workspace/r11_worktrees/b-dreamzero` | `/workspace/bwa_runs/r11-four-way-v1/B/{checkpoints,logs,status}` |
| C | 2 | `bwa-r11-c-cosmos` | `/workspace/r11_worktrees/c-cosmos` | `/workspace/bwa_runs/r11-four-way-v1/C/{checkpoints,logs,status}` |
| D | 3 | `bwa-r11-d-lawam` | `/workspace/r11_worktrees/d-lawam` | `/workspace/bwa_runs/r11-four-way-v1/D/{checkpoints,logs,status}` |

共享只读路径：数据 `/workspace/datasets/robofactory_multitask`，HF cache `/workspace/.cache/huggingface`，基础权重 `/workspace/artifacts/r11_upstream`，固定 seeds `/workspace/bwa_runs/w10-six-task-v1/seeds/validation`。每个候选使用独立 venv 和统一 wheel cache，解决 DreamZero/Cosmos/LaWAM 的依赖冲突。HF token 只沿用 S0 的环境变量或 mode-0600 secret/FIFO 注入，任何 command、log、status、文档和 Git diff 不得出现 token。

必须实现并验证：

```bash
scripts/before_we_act/launch_r11_4gpu_tmux.sh --all --dry-run
scripts/before_we_act/launch_r11_4gpu_tmux.sh --all
scripts/before_we_act/launch_r11_4gpu_tmux.sh --candidate A
scripts/before_we_act/monitor_r11.sh --all --once
scripts/before_we_act/monitor_r11.sh --candidate A --watch
scripts/before_we_act/stop_r11_4gpu_tmux.sh --candidate A --graceful
scripts/before_we_act/stop_r11_4gpu_tmux.sh --all --graceful
```

monitor 状态至少区分 `NOT_STARTED/PREPARING/DOWNLOADING/PREFLIGHT/TRAINING/VALIDATING/ACCEPTING/PASSED/FAILED/FAILED_FIT/STOPPED/STALE/UNKNOWN`，显示四条独立进度条、排队阶段、实际进程心跳、PID、branch/commit、GPU 指标、update/120000、ETA、loss、预测指标、闭环结果、当前/最佳 checkpoint、因果 gate 和失败原因。`PASSED/FAILED` 必须读取第 11 节验收器的结构化 JSON，不能由日志关键词推断。

## 11. 预注册验收与胜者规则

### 11.1 正式候选资格

候选必须同时满足：

1. 固定 Validation20 完整跑完 120/120 回合，无非法动作、fallback、OOM、NaN、无心跳或被自动跳过的任务。
2. 总成功数至少 `80/120`。这相当于允许比 W10 少最多 8 次成功，即绝对差不超过 6.67 个百分点，成功数达到 W10 的 90.9%。
3. W10 四个满分任务的合计至少 `72/80`，且任何单项不得低于 `16/20`。
4. Camera Alignment 至少 `6/20`，Camera Alignment + Place Food 至少 `8/40`；允许以 Place Food 的真实提升交换 Camera 的小幅回撤。
5. 在各自同一 latent/pixel 表示中，未来预测相对“复制当前状态”的 persistence baseline，macro normalized error 至少改善 5%，并在至少四个任务上改善。
6. 打乱动作条件后，future prediction error 在至少四个任务上恶化至少 5%；否则说明 world model 没有学到 action-conditioned dynamics。
7. `prediction_off` 或 `prediction_shuffled` 保持任务文本和当前观测不变时，离线 action NRMSE 至少恶化 2%，或预注册 hard-task Validation5 至少少成功 1 回合。否则预测没有实质进入动作决策。

任何一项失败都记录真实 `FAILED` 原因，不能降低门槛。R11 的“胜者”只从满足全部资格的候选中选择；若无人合格，结论就是 R11 无胜者。

### 11.2 综合分数

合格候选按以下冻结公式排名，所有比例取 `[0,1]`：

```text
score = 60 * (all_six_successes / 120)
      + 10 * (protected_four_successes / 80)
      + 10 * (camera_plus_food_successes / 40)
      +  8 * clamp(macro_prediction_gain / 0.20, 0, 1)
      +  7 * causal_gate_pass_fraction
      +  5 * min(W10_p95_action_latency / candidate_p95_action_latency, 1)
```

同分依次比较：六任务总成功数、Camera+Food、prediction-shuffled 恶化幅度、p95 latency、峰值显存。报告 score 的同时必须并列原始成功数，禁止只给复合分掩盖闭环退化。

### 11.3 Confirmation50

在查看新 seeds 结果前生成并冻结每任务 50 个 seeds。W10 与临时 winner 都跑 300 回合，使用相同环境、max steps 和成功条件。报告逐任务 paired bootstrap 95% CI。最终 winner 需满足总成功率相对 W10 的非劣界不低于 `-6.67pp`，并继续满足预测/因果 gate；否则只能称 Validation20 临时胜者，不能合入 `feat/model-improvements`。

## 12. 实施文件和白名单检查清单

每个候选分支至少包含模型、配置、训练入口、推理 adapter、Validation5/20、预测验收、单元测试、source receipt、license/notice 和候选说明。共同运行分支还要提供四卡 launcher、monitor、graceful stop、winner 决策器和 Confirmation50 入口。

实现时必须 `rg` 全调用链，检查模型名称在以下位置一致：模型 factory/registry、训练 CLI choices、checkpoint config schema、推理 loader、RoboFactory evaluator adapter、验收 allowlist、launcher candidate map、monitor status schema、resume 校验和文档示例。不得只改一个 registry。

每个 checkpoint 固化：candidate ID、model family、branch、commit、base commit、上游 commit/文件哈希、HF model revision、许可证 receipt、训练 manifest 哈希、sample cursor、optimizer/scheduler/RNG、micro-batch/accumulation、所有 loss 权重和预测推理模式。resume 后必须验证下一 sample key 与未中断基准一致。

## 13. 结果记录模板

训练开始后在本节下方追加一次性摘要，不把逐 step 日志抄进路线：

| 候选 | source parity / fit | update | 峰值显存 | Validation20 | 预测 gain | 动作→预测 gate | 预测→动作 gate | score | 结论 |
|---|---|---:|---:|---:|---:|---|---|---:|---|
| A V-JEPA | F0 29/29；真实 F1；Discovery checkpoint `6e3931…0e3e` | Discovery 1000/1000 | 7.926 GiB | 未运行 | -97.883%（仅 1/6 任务改善） | PASS：shuffled future +31.651%，4/6 任务达 5% | PASS：off +3.217%；shuffled +0.019% | — | FAILED（Discovery persistence gate） |
| B DreamZero | F0 42/42；真实 5.64B Wan + rank-16 LoRA F1；严格 Discovery resume receipt 通过；checkpoint `133a23…c78e` | Discovery 1000/1000 | 24.226 GiB | 未运行 | -308.438%（0/6 任务改善） | PASS：shuffled future +0.018%（冻结公式为宏观值 > 0） | FAIL：off -0.045%；shuffled +0.019% | — | FAILED（Discovery persistence/action-coupling gate） |
| C Cosmos | source receipt 已核验；冻结 foundation revision 下载 403 | 0/120000 | — | — | — | — | — | — | FAILED（foundation authorization） |
| D LaWAM | F0 31/31；真实 F1；Discovery checkpoint `5ad947…a888` | Discovery 1000/1000 | 16.463 GiB | 未运行 | -8.399%（仅 2/6 任务改善） | PASS：shuffled future +32.734%，5/6 任务达 5% | PASS：off +6.758%，shuffled +61.146% | — | FAILED（Discovery persistence gate） |

执行前 baseline provenance gate 于 2026-08-10 通过：`origin/feat/model-improvements` 解析为 `78471b285bc69fa8b5168fb170a3c3332efc32be`；W10 checkpoint SHA256 仍为 `e1b07b2cf7bff37428bf54a27f545632c8a1013930d96f6e646d8ca055f2f574`；六任务 900/900 个 HDF5 已逐文件重算哈希，累计验证 744,660,714,054 字节；Validation20 仍为 88/120，六组 seeds 哈希一致。数据没有变化，不触发 W10 重训。不可变远程 receipt 为 `/workspace/bwa_runs/r11-four-way-v1/preflight/baseline_provenance.json`，SHA256 `c2936ba68afbb99a9201c69894b18c9d6fc0e400d22f7c90f7f8886728208cd5`。四个官方仓库默认分支 HEAD 也仍等于第 7.1 节冻结 commit，没有升级。

训练启动记录（2026-08-10）：四个候选均从冻结 base 独立派生，官方源码 checkout 保持 detached/clean/read-only，A/B/D foundation asset receipt 已通过；C 对 `nvidia/Cosmos-Predict2-2B-Video2World@f50c09f5d8ab133a90cac3f4886a6471e9ba3f18` 的真实请求返回 403，故没有生成伪 receipt 或替换权重。A 在提交 `b69aec1f8dd9b064c6c83c668ccd8b31b0b8c18f` 完成真实 forward/backward/optimizer、partial checkpoint 保存、严格恢复和推理，随后进入 Discovery。B 在执行提交 `dc28cc5abd085542fabfaffc891c366564c0ce70` 以 rank-16 LoRA/bf16 完成相同 F1 全路径；其 update-2 checkpoint SHA256 为 `44a718fa24a9f36db1e93389998cf0f927abeb417a528be1b718c156ee77cfc2`，三种模式均输出有限 `[1,100,8]` 动作和 future prediction，随后进入 Discovery。D 在 `6d7553b1063c708ffe7384cf73d2aab04c5ab359` 完成官方 256×256 LAM 路径的 F1 fresh/resume/inference 并进入 Discovery。此处只记录真实进行态，不能据此宣称 Validation5/20、因果 gate、Confirmation50 或 winner 已完成。

Discovery gate 记录（2026-08-11 00:33，Asia/Shanghai）：D 完成连续 1,000 updates，effective batch 48、`micro_batch=2`、`accumulation=24`，端点 checkpoint SHA256 `5ad9476371bb308181c3fb6aa07114c921a1e097e78fa73350e3f9fbc077a888`，sample-cursor receipt SHA256 `08b9ce87ad66ab6e7240f735a629a0eb3d2548bddeb35e27c89d0d0267bf7b70`。action loss 最初/最后 100 updates 中位数从 2.538873 降至 0.197414；action shuffle 使未来误差宏观恶化 32.734%，5/6 任务达到 5%；prediction off/shuffled 使动作 NRMSE 分别宏观恶化 6.758%/61.146%。但 future-vs-persistence 宏观 gain 为 -8.399%，只有 2/6 任务改善，违反 Discovery 必须为正的冻结门槛，因此 pipeline 以 exit code 10 生成 `D/acceptance.json=FAILED` 并停止，未运行 Validation5、Selection 或更后阶段。该结果不重试、不降低门槛。

Discovery gate 记录（2026-08-11 00:43，Asia/Shanghai）：A 完成连续 1,000 updates，effective batch 48、`micro_batch=2`、`accumulation=24`，端点 checkpoint SHA256 `6e393139f0447564836fbb4a7039a12320ea8159db280fc0ace2e1a08fbb0e3e`，sample-cursor receipt SHA256 同为 `08b9ce87ad66ab6e7240f735a629a0eb3d2548bddeb35e27c89d0d0267bf7b70`。action loss 最初/最后 100 updates 中位数从 0.025602 降至 0.003461；action shuffle 使未来误差宏观恶化 31.651%，4/6 任务达到 5%；prediction off 使动作 NRMSE 宏观恶化 3.217%，通过 2% 替代门槛，但 prediction shuffled 只变化 0.019%。future-vs-persistence 宏观 gain 为 -97.883%，只有 1/6 任务略有改善，违反 Discovery 门槛，因此 pipeline 同样以 exit code 10 生成 `A/acceptance.json=FAILED` 并停止，未运行 Validation5、Selection 或更后阶段。该结果不重试、不降低门槛。

Discovery gate 记录（2026-08-11 08:27，Asia/Shanghai）：B 完成连续 1,000 updates，effective batch 48、六任务各 8、`micro_batch=1`、`accumulation=48`。训练执行提交为 `dc28cc5abd085542fabfaffc891c366564c0ce70`；因果 probe 修复提交 `f66b43149b57d49628bfe7fa5e6b85f55a2378e6` 只让 intervention hook 保留官方 `action=None` prefix/KV-cache 调用、在真实 action register 调用上 shuffle，并用独立脚本严格核验既有 checkpoint/cursor，不改权重、训练记录或门槛。endpoint checkpoint 为 281,476,775 bytes，SHA256 `133a239264065f0d590c07439ccc1a1b60e32fd6de3d61c785e9fedc6094c78e`；sample-cursor SHA256 `6dc8ce42f51785805a5f5ff9f5d0c49845646e9848555769d0badd85f7a9e0b8`。action loss 最初/最后 100 updates 中位数从 1.198908 降至 1.000934；但 192 样本 official Wan 因果 probe 的 future-vs-persistence 宏观 gain 为 -308.438%，0/6 任务改善，prediction off/shuffled 的动作 NRMSE 变化为 -0.045%/+0.019%。因此 B 以 exit code 10 生成 `acceptance.json=FAILED`，未运行 Validation5 或以后阶段。

最终决策记录（2026-08-11 08:30，Asia/Shanghai）：integration 运行提交 `189c7c27eb7361dc7e42ee207ece9da1ea1b13a5` 的 fail-closed 决策器只读取四份结构化终态，得到 A/B/D `failed_stage=discovery-gate`、C `failed_stage=foundation`，无资格候选、ranking 为空。不可覆盖的 `/workspace/bwa_runs/r11-four-way-v1/winner.json` 为 `status=NO_WINNER`、`complete=true`、`confirmation50.status=NOT_APPLICABLE`、`merged_to_baseline=false`，SHA256 `b81856cf2d65c8824883af7f5a5cb78251a4dcf6a3177626a3abe2628b7d94e0`。Confirmation50 seed receipt 不存在，W10/winner 300+300 回合未启动，四张 GPU 已空闲，baseline 分支未合并。共同目标测试最终为 35/35，B 修复后 F0 为 42/42；门槛、score 公式和 W10 基线均未改变。

### 13.1 R11 之后的方向选择：LaWAM 是研究起点，不是 R11 winner

R11 的正式结论保持“无胜者”。后续技术方向选择 D / LaWAM latent visual subgoal + flow action expert，含义仅为“在失败候选中选择最有信息量的起点”，不构成验收通过、临时 winner、Confirmation50 资格或 baseline policy/checkpoint 晋级。为复用已核验的上游迁移与测试，`feat/r11-lawam-latent-subgoal@6d7553b` 已通过非快进合并提交 `6cd891b` 进入 `feat/model-improvements`；合入范围是 LaWAM 实现及其必要公共运行器，不是把 R11 D 的失败模型提升为基线。任何后续实验仍必须使用新的 stage、分支、run manifest、checkpoint 和 acceptance；不得覆盖 `D/acceptance.json=FAILED`，不得把 R11 D checkpoint 改标为通过，也不得用后续结果追溯修改 R11 结论。

选择 LaWAM 的实验依据是：

- D 已完成真实 F1 和连续 Discovery1000，训练过程稳定且 action loss 从最初 100 updates 中位数 2.538873 降至最后 100 updates 的 0.197414；相比 C 的未训练和 B 的 5.64B 高成本路线，它提供了已验证的可运行 latent 路径。
- D 的 future-vs-persistence 宏观 gain 为 -8.399%，虽然明确未通过门槛，但显著接近 A 的 -97.883% 和 B 的 -308.438%；该相对接近度只用于选研究起点，不能转写为 PASS。
- D 的 action-shuffle→future 为 +32.734%，5/6 任务达到 5%；prediction off/shuffled→action 为 +6.758%/+61.146%，说明预测已真实进入动作计算路径。失败集中在“预测目标没有优于 persistence”，因此下一步应更改预测表征、监督目标和候选选择机制，而不是延长原 checkpoint 训练。

明确反面条件：若新的 action-relevant target 在冻结的 counterfactual probe 上仍不能稳定击败 persistence，或 oracle proposal ranker 不能从 W10 候选中选出更优动作，则停止 LaWAM 方向，不进入大预算训练。R11 D checkpoint 仅允许用于只读诊断；新阶段默认从冻结 LaWAM foundation/源码和 W10 action prior 初始化，重新训练 world/selector head。

### 13.2 后续受控消融路线：W10-anchored action-relevant latent planning

后续阶段不再让四张 GPU 运行四个参数规模、数据协议和推理路径均不同的 foundation model。四路共享同一 W10 action prior、LaWAM latent transition 基座、proposal selector/action refiner、六任务 sampler、数据 receipt、优化预算和评测协议，只改变一个预注册机制，以便把收益或失败归因到具体设计。

共同动作路径冻结为：

```text
observation + task text + proprio
  -> trainable W10 action prior 生成 K=4 个 100-step action proposals
  -> LaWAM action-conditioned latent transition 分别 rollout 每个 proposal 的前缀
  -> action-relevant future / progress / risk score
  -> differentiable proposal selection + residual action refinement
  -> 执行冻结 cadence 的短前缀并读取真实观测
```

W10 action prior 是候选内部的可训练组成部分，不是评测 fallback；默认解冻并以较低 learning rate 联合训练。所有合格变体的预测都必须进入 proposal score、选择或 residual refinement，禁止只有辅助 world loss。共同 action horizon 仍为 100；精确 rollout prefix 和 receding-horizon cadence 必须在新阶段配置冻结后写入 manifest。

四卡受控消融矩阵：

| 路线 | 唯一新增机制 | action-relevant target | 目的 |
|---|---|---|---|
| L0 / GPU0 | `delta-latent + common selector` | 多尺度 `Δz=z(t+h)-z(t)`，`h={1,4,8,16}`，按任务/视角/horizon 白化 | 判断去除静态背景后是否能击败 persistence |
| L1 / GPU1 | L0 + self-grounded/foreground supervision | robot self-mask 或无标注时的 motion/foreground mask、proprio delta | 判断机器人和任务相关变化监督是否提高 action sensitivity；机制参考 [SelfWAM](https://arxiv.org/abs/2608.00725) |
| L2 / GPU2 | L0 + algebraic transition consistency | action transition 的 composition/reversal consistency | 判断结构化 latent transition 是否比纯重建目标可用于动作生成；机制参考 [ALAM](https://arxiv.org/abs/2605.10819) |
| L3 / GPU3 | L0+L1+L2 + uncertainty-aware score/residual refine | 上述联合目标以及 proposal progress/risk ranking | 验证组合是否产生可闭环的综合候选；ranking/guidance 只借鉴 [Guided Action Flow](https://arxiv.org/abs/2607.02092)，不能照搬其任务或成绩 |

所有路线共用相同 `K=4` proposal 集合和 selector；L0 也是 prediction-in-action 的真实候选，不是只训练辅助 loss 的控制组。为区分各机制，L1/L2 不得互相混入，L3 只在 L0/L1/L2 的目标 gate 均有可解释结果后评估组合效应。

训练前先建立只读 counterfactual dynamics probe：保持观测和任务文本不变，对同一模拟器状态执行正确、时间打乱、反向和幅值受控扰动的动作前缀，记录 `h={1,4,8,16}` 的真实 robot/object/proprio/progress 后果；按背景、机器人、任务物体和 proprio 分解 persistence error。probe 必须有独立 manifest、seed、simulator state/replay receipt 和哈希。第一轮 probe 只用于诊断和验收，不加入训练；若后续把新 counterfactual 数据用于训练，必须按数据版本策略先冻结新数据 receipt、重训/重验 W10 并建立新 baseline，不能继续引用当前 W10 provenance。

大预算前的 fail-fast 顺序：

1. **Measurement gate**：核验 frame/action/proprio 时间对齐；小型 oracle transition 在 held-out counterfactual probe 上的宏观 future-vs-persistence gain 至少 +5%，且至少 4/6 任务为正；oracle K=4 ranker 必须证明 proposal 集合中存在优于默认 W10 proposal 的候选。任一失败即停止，不实现大模型。
2. **F0/F1**：对 LaWAM official tensor、adapter、W10 partial load、真实 forward/backward/optimizer/save/resume、三种因果模式和 K=4 selector 做零 skip parity/integration gate。不得用 R11 D checkpoint 续训冒充新 F1。
3. **Discovery1000**：action loss 必须下降；future-vs-persistence 宏观至少 +5% 且至少 4/6 任务为正；action shuffle 使 future error 宏观至少恶化 5% 且至少 4/6 任务达到 5%；prediction off 或 shuffled 使动作 NRMSE 至少恶化 2%；K=4 counterfactual ranking accuracy 至少 70%。任一项失败即停止该消融路线。
4. **闭环预算**：只有通过上述 gate 的路线才进入 paired Validation5、Selection、Formal 和 Validation20。正式闭环仍使用第 11 节的原始资格门槛和 Confirmation50 非劣界，不得因选择 LaWAM 方向而降低。

本节只冻结后续研究方向和受控消融设计，不表示新阶段已启动。开始实现前仍需建立新的 stage 变量区、四个独立分支/共同 integration 分支、上游 commit/license/source receipt、不可变 run manifest、数据和 seed receipt，并重新执行 baseline provenance gate。

详细逐任务记录保留在 `feat/r11-four-way-integration@678e67780e6960749410ee0649ce961b10495950` 的 `docs/experiments/r11/`；本路线保留执行 commit、命令、产物位置、上述表格、逐项验收和结论。论文声称的 benchmark 数字只作选型依据，不能当成本项目成绩。

## 14. 后续正式执行入口

R11 已形成终态，原正式 AI coding prompt 只用于审计和复现，不得用它在原 run root 重启训练：

- [R11 四卡 World-Action Model 改进执行 Prompt](../runbooks/R11_FOUR_GPU_WORLD_ACTION_MODEL_EXECUTION_PROMPT_ZH.md)

LaWAM 受控消融尚未分配正式 stage 名称、分支、run root 或运行 prompt。开始实现前必须从本路线第 13.1–13.2 节生成新的正式 prompt，并在变量区冻结 W10 provenance、LaWAM `FAILED` 历史、L0–L3 矩阵、counterfactual probe、上游 commit、GPU、训练预算和新 acceptance。不得沿用 R11 候选状态，不得写入 `/workspace/bwa_runs/r11-four-way-v1`，也不得把方向选择解释为对 R11 gate 的追认。
