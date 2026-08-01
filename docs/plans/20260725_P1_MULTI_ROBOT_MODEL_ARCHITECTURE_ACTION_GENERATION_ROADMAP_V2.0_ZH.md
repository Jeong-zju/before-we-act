# P1 多机器人 World-Action Flow Matching 技术路线 V3.0（ICRA Fast Track）

> 文档更新：2026-08-01
> 工程起点：当前 `feat/model-improvements` 分支
> 投稿目标：ICRA 2027，[官方 Call for Papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/) 截稿时间为 2026-09-15 11:59 PM PST
> 当前状态：M0、M1、S0、S1-R1、S2 已完成；R1 选择 `rectified_flow_cold` F1，R3 选择 own-action-conditioned W1；新 R4 hybrid 因 LiftBarrier peer-action-shuffle bootstrap 95% 下界为负而按特殊规则失败；R5 Protected Shared 与 Protected Role-MoT 均通过 protected-own/team 全部门槛，按五任务 macro peer/shared loss 选择更简单的 R5-P0 Protected Shared，且正式 P0 分支已合并回 `feat/model-improvements`；R2a 跳过、R2b 延后，下一步进入 S3-R6 world-to-Flow gated injection
> 评测原则：进入动作路径的候选按闭环成功率推进；S2 predictor 严格 off-path，因此按预测能力与因果干预门槛推进
> 相关长期方案：[Intent-Grounded Decentralized World-Action Models 多机器人协作研究方案](20260724_INTENT_GROUNDED_DECENTRALIZED_WORLD_ACTION_MODELS_MULTI_ROBOT_COLLABORATION_RESEARCH_PLAN_V2.0_ZH.md)

## 1. 本次路线调整的结论

ICRA 截稿临近，后续不再按旧版 M3–M11 的长串行路线推进。当前分支直接作为工程起点，压缩成一条可以在约七周内形成论文闭环的主线：

> 按机器人组织多模态上下文，用 Rectified Flow / Flow Matching 生成每台机器人的动作；再用动作条件的多机器人未来表示显式调制 Flow 速度场，使预测未来真正参与协作动作生成。

本次调整包含七项硬决策：

1. **当前分支就是起点。** 不重写已经验证的数据、DINOv3、按机器人视图、共享解码器、dense/MoE、时间集成、采样、checkpoint 和闭环评测基础。
2. **最终目标是 World Action Model 与 Flow Matching。** 旧的 CVAE 动作分块模型仅保留为历史基线；论文标题、方法名和主张不以 ACT 为目标。
3. **每个候选只做一个单步改进。** 相对冻结父提交，只允许改变一个研究变量、回答一个假设，并能用一个 flag 或一个 commit 完整回退；禁止同时改变数据、表示、损失、模型接口和推理协议中的多个维度。
4. **使用两卡微轮次和远程 GPU 并行验证。** 一个训练微轮次固定为 `P0=父方案复跑` 与 `P1=父方案+一个 Δ` 两个候选；两个互不依赖的微轮次可以四卡同时运行。P0/P1 使用相同训练预算与阶段对应验证：S2 比较 held-out capability，进入动作路径后比较同协议闭环成功率。新 R4 是唯一的零训练诊断例外，只组合已经完成的 checkpoint，不参与正式 winner 选择。
5. **暂时舍弃 active-agent loss weighting。** 训练目标不再根据动作幅度、active/inactive 标签或机器人活跃比例调整权重。所有 agent 使用相同损失规则，activity 最多保留为 debugging log，不参与反向传播或候选选择。
6. **进入动作路径后只用闭环成功率决定推进。** P1 与对应 P0/父方案跑相同任务；只要每个任务的闭环成功率都不低于父方案，P1 就通过并可进入下一阶段，持平也算通过。S2 是唯一例外：predictor 尚未进入动作路径，闭环输出理论上应与冻结 F1 完全相同，因此 S2 只用 held-out prediction 与 action/peer-action shuffle 证明其能力，并用 action-equivalence smoke 排除误接线；从 S3 起恢复闭环唯一选型规则。
7. **own predictor 从软约束改为硬保护。** 旧 R4 已证明 multi-head、own residual gate、分组梯度裁剪和随机数流隔离都不能保证逐任务 own no-regression。新 R5 固定从同一个合格 P0 own checkpoint 出发，own tower 以 `eval + frozen + optimizer-excluded` 方式保持函数不变；peer/shared 只能单向读取 detached own 表示，不能反向改写 own 输出。

## 2. 论文目标与边界

### 2.1 暂定论文题目

**Cross-Agent World-Conditioned Flow Matching for Multi-Robot Collaboration**

中文工作名：

**面向多机器人协作的跨智能体世界条件 Flow Matching**

最终方法类建议命名为 `CrossAgentFlowWAM`。`AgentFactorizedFlowWAM` 可以保留为 S1 纯 Flow 基类，避免把旧类名直接改包装后当作新方法。

### 2.2 核心研究问题

论文只回答一个主要问题：

> 候选联合动作所诱导的跨机器人未来后果，能否直接调制按机器人分解的 Rectified Flow 速度场，并提高真实协作任务的闭环成功率？

目标计算图为：

$$
\hat{\mathbf z}_{t+1:t+H}^{1:N,\mathrm{shared}}
=
W_\phi
\left(
\mathbf h_t^{1:N},
\mathbf x_\tau^{1:N},
\tau
\right),
$$

$$
\mathbf v_\theta^i
=
F_\theta
\left(
\mathbf x_\tau^i,
\tau,
\mathbf h_t^i,
\hat{\mathbf z}_{t+1:t+H}^{i},
\hat{\mathbf z}_{t+1:t+H}^{-i,\mathrm{shared}}
\right),
$$

其中：

- $\mathbf h_t^i$ 是第 $i$ 台机器人的视觉、状态、动作历史和任务上下文；
- $\mathbf x_\tau^i$ 是 Flow 中间状态或候选动作；
- $W_\phi$ 根据整队候选动作预测各 agent 与共享对象的联合未来 latent；
- $F_\theta$ 预测第 $i$ 台机器人的速度场，并显式读取自己的未来、其他 agent 的后果和共享对象后果；
- 推理时只能向动作路径输入**预测未来**，不能输入真实未来。

如果未来分支只作为辅助损失、没有回到速度场，它只能叫 `Flow + auxiliary future prediction`，不能作为最终 WAM 主张。

### 2.3 截至 2026-07-28 的新颖性研判

**结论：当前宽泛目标不具备足够新颖性；收紧后的核心目标具有条件性的新颖性，但尚未被实验建立。**

代码现状也支持这一判断：当前 `block_causal_transformer.py` 明确禁止 action query 读取 future query，Flow solver 又以 `include_future=False` 调用 velocity model。因此当前分支实现的是 S0/S1 工程起点和近似 R6-P0 的 `Flow + auxiliary future prediction`，**还没有实现本文拟主张的 cross-agent world-to-flow coupling**。目前能评价的是最终目标的潜在新颖性，不能把现有代码直接称为新方法。

以下组件不能单独作为论文贡献：

| 路线组件 | 最接近工作与碰撞 | 判断 |
|---|---|---|
| Flow Matching 动作生成 | [$\pi_0$](https://arxiv.org/abs/2410.24164) 等已有 Flow action expert | 非新颖基础组件 |
| previous-chunk warm start | [Streaming Flow Policy](https://arxiv.org/abs/2505.21851) 从上一动作附近的窄高斯出发并流式积分 | 只作为工程候选 |
| latent future 进入 action generation | [LaWAM](https://arxiv.org/abs/2606.15768) 已用动作条件 latent world model 预测视觉 subgoal 并条件化动作生成；[AGRA](https://arxiv.org/abs/2606.12217) 已研究 world-action 表示接口并使用因果干预诊断 | 直接碰撞，不能泛称首创 |
| 只在训练期使用未来表示 | [Being-H0.7](https://arxiv.org/abs/2605.00078) 以未来 posterior 对齐部署 prior；[Fast-WAM](https://arxiv.org/abs/2603.16666) 质疑测试时显式未来预测的必要性 | auxiliary future 不足以支撑 WAM 主张 |
| 生成候选并由 world model 评分 | [Cortex 2.0](https://arxiv.org/abs/2604.20246) 在视觉 latent 空间生成、评分并选择候选未来 | 移出 ICRA 主线 |
| 多机器人 Flow 轨迹/动作协同 | [GCo](https://arxiv.org/abs/2511.10874) 已做多机器人接触与轨迹 Flow co-generation；[Flow-Opt](https://arxiv.org/abs/2510.09204) 已做带置换不变编码的集中式多机器人 Flow 轨迹优化 | “multi-robot + Flow” 本身不新颖 |
| action-conditioned multiview world model | [A2World](https://arxiv.org/abs/2606.29501) 已建模动作驱动的多视角场景演化 | 多视角预测不是核心贡献 |

在本轮检索到的最接近工作中，尚未发现与以下完整机制相同的公开方案：

> **对联合候选 action chunk 建模其跨 agent 与共享对象的后果，再将“自己的未来 + peer 后果 + shared-object 后果”逐 Flow step 注入共享参数、按 agent 分解的速度场，并以跨 agent 因果干预证明该耦合改善闭环协作。**

因此论文贡献必须收敛为：

1. **方法贡献：** `joint candidate action → cross-agent future consequences → factorized velocity fields` 的可变 agent-slot 结构，而不是 WAM 或 Flow Matching 的简单组合；
2. **因果证据：** 对 peer action、peer future、agent slot 和共享对象 future 做 zero/shuffle/intervention，并用干预前后的闭环成功率验证这些输入是否必要；
3. **闭环证据：** 在必须同步或交接的任务中，优于 `joint Flow without world`、`local-future WAM` 和 `auxiliary-only future` 三类公平基线，并同时报告 centralized joint policy 信息上限。

新颖性与因果分析只用于最终论文表述，不再设置工程阶段门槛。投稿前不得使用 “first” 或 “首次”；最终 claim 根据已有闭环结果收缩，但不阻塞模型迭代。

### 2.4 ICRA 快线不做什么

以下内容保留为长期方向，但不进入本次主线：

- 全分辨率视频生成式 world model；
- 任意机器人数量的严格理论泛化；
- 严格去中心化通信协议和真实网络部署；
- 大规模语言意图 grounding；
- 强化学习或在线探索；
- 5B/14B 模型扩展；
- 自建大量新任务或重新采集大规模数据。

本次使用低维、可验证的未来目标：未来 proprioceptive state、未来 DINO latent、物体或团队进度。论文价值来自“world prediction 如何进入 action flow”，不是视频生成规模。

## 3. 当前分支：保留什么，替换什么

### 3.1 直接保留

| 当前能力 | 快线中的位置 |
|---|---|
| RoboFactory 原生数据、状态/动作 mask、多任务 contract | 所有候选共用的数据基础 |
| 冻结 DINOv3 与完整 spatial patch tokens | 视觉上下文与未来视觉 latent 目标 |
| 18D 状态视图、8D 动作槽等按机器人数据视图 | agent factorization 起点 |
| 共享 decoder、dense 与 top-2 MoE 两种实现 | S1 并行结构候选 |
| temporal ensemble 与 latest-chunk 路径 | 统一推理协议及消融 |
| task-balanced sampler | 多任务训练公平性 |
| checkpoint、Gate20 与成功率统计工具 | 闭环迭代基础 |
| M2 中已有的 Rectified Flow、block-causal 上下文和未来预测代码 | 新 Flow/WAM 的实现参考 |

当前静态候选的初步闭环结果可以证明这条分支适合继续改，但不能直接作为论文结果。已有不同提交间的结果变化还混合了多项改动，正式表格必须从冻结的数据、评测种子和候选父提交重新跑。

### 3.2 必须替换

- CVAE posterior、KL 目标和直接动作 MSE 不再是最终动作生成目标；
- 旧类名及 `static_act` 路径只作为 legacy baseline，不作为新方法命名空间；
- 只预测未来但不影响动作的旁路结构不能作为最终方法；
- 固定拼接整队动作的单头输出要改成按 agent slot 组织、共享参数的 Flow expert；
- 旧 M2 不再因“还没完整跑完”阻塞论文快线。

### 3.3 active-agent loss weighting 决策

从 2026-07-28 起，所有快线训练使用与 activity 无关的损失约定：

$$
\mathcal L_b =
\frac{
\sum_{t,d} m_{btd}\,q_t\,e_{btd}
}{
\sum_{t,d} m_{btd}\,q_t
},
\qquad
\mathcal L = \frac{1}{B}\sum_b \mathcal L_b.
$$

其中 $m$ 只表示有效时间步和有效维度，$q_t$ 只允许表达对所有 agent 一致的 executed-prefix 等时序权重。禁止让 $q$ 依赖动作幅度、active/inactive 判定或 agent 身份。

具体约束：

- 删除训练配置中的 `active_agent_weight`、`active_delta_threshold`、`active_agent_loss_weight` 和 `active_agent_delta_threshold`；
- Flow velocity、部署端点、平滑项、未来状态和 legacy reconstruction loss 均不做 active-agent 重加权；
- 不记录会被误认为训练目标一部分的 `active_agent_fraction` loss metric；
- teacher-context 的 active/inactive 拆分最多保留为 debugging log，不进入 evidence board；
- ICRA 截稿前不把该机制重新加回主线。若以后重启，必须作为单独、受控且多随机种子的消融。

## 4. 快线总览

```mermaid
flowchart LR
    S0["S0 冻结起点<br/>B0/B1/B2/B3"]
    S1["S1 Per-Agent Flow<br/>R1 Flow；R2 延后"]
    S2["S2 Protected Action-Conditioned World<br/>R3 Action / R4 Hybrid / R5 Role-MoT"]
    S3["S3 Safe World-to-Flow<br/>R6 Injection / R7 Unfreeze / R8 Dropout"]
    S4["S4 四种子正式评测<br/>E1/E2/E3/E4"]
    S5["S5 论文与视频<br/>冻结方法"]

    S0 --> S1 --> S2 --> S3 --> S4 --> S5
```

S0 是起点选择，不计作结构改进。S1–S3 由若干“两卡单变量微轮次”组成：

```mermaid
flowchart LR
    P["Round k<br/>冻结父提交"]
    P0["P0 父方案复跑<br/>Δ = 0"]
    P1["P1 单步候选<br/>Δ = 1"]
    T["每个候选<br/>完整约定训练预算"]
    E["每个候选<br/>阶段对应验证"]
    S["S2 capability gate<br/>或 on-path 成功率"]
    N["Round k+1<br/>选定父提交"]

    P --> P0 --> T
    P --> P1 --> T
    T --> E --> S --> N
```

两个独立微轮次可以占用四张卡并行。例如：

```text
卡 0/1：P vs P + Δdecoder
卡 2/3：P vs P + Δsource_prior
```

如果 $\Delta_{\mathrm{decoder}}$ 与 $\Delta_{\mathrm{source\_prior}}$ 都没有造成成功率退步，可以启动组合闭环；组合相对其 P0 不退步即可进入下一阶段。

“单步改进”保持轻量：

1. 只回答一个研究假设；
2. 只改变一个配置轴或一条模型接口；
3. 数据、seed、训练预算、闭环协议和其他模型路径不变；
4. 可以通过一个 flag 或一个 commit 完整回退；
5. 尽量保持改动可独立回退。

结构例外有两项。第一项是 R1 的 `legacy action generator → cold-start Rectified Flow`：head、FM loss 和 ODE solver 必须作为一个可运行的原子垂直切片共同替换，但其研究变量只有 `action_generator`；上下文、decoder、数据、action chunk、ensemble 和评测协议全部保持不变。第二项是新 R4 的 hybrid checkpoint 诊断：它不训练、不拟合统计量、不产生可晋级模型，只验证“冻结 P0 own 路径 + 旧 P1 team 路径”是否在函数组合后已经满足 R5 的目标。

所有微轮次只保留两条规则：

- P0/P1 使用相同数据 split、训练预算与阶段对应协议；
- S2 采用 prediction/shuffle capability gate；从 S3 起 P1 各任务成功率不低于 P0 就可以继续。主动停止的候选直接退出比较，不阻塞其他候选。

## 5. S0：冻结工程起点与协作任务（07-28）

### 5.1 四个并行参考方案

| 卡 | 方案 | 作用 |
|---|---|---|
| B0 | 当前 sparse MoE legacy chunk policy + temporal ensemble | 当前分支行为参考 |
| B1 | compute-matched dense legacy chunk policy + temporal ensemble | 判断 MoE 是否值得继续 |
| B2 | 现有 M2 Rectified Flow，关闭或旁路旧 future head | Flow 工程参考 |
| B3 | 当前 sparse MoE legacy chunk policy + latest chunk | 隔离 temporal ensemble 的实际贡献 |

四卡使用相同数据 manifest、DINO 权重、动作归一化、训练 update、推理频率和 Gate20 初始条件。B0 与 B3 允许复用同一公平训练 checkpoint，因为二者只改变推理聚合；其他结构不得复用 checkpoint。旧 checkpoint 只用于工程 smoke test。

S0 只建立参考坐标，不产生可外推的结构 winner：B1/B3 分别诊断 decoder 与推理聚合，B2 是旧 M2 工程参考。由于 B2 训练耗时超过 fast-track 预算，operator 决定在 Gate20 前主动终止 B2，并以已完成成功率评测的 B0 作为 R1 工程父方案。该处置不等于证明 legacy action generator 优于 Rectified Flow；正式 Flow 改进仍在 R1 中从 B0 父方案以原子垂直切片重新实现和验证。

#### 5.1.1 Vast.ai 四卡从零一键部署与运行

以下命令假设远程服务器已经自动进入唯一的永久 tmux session，且 `/workspace/fe-pc-wam` 不存在。命令不执行 `apt update`，也不允许通过 `export HF_TOKEN=...` 传递 Hugging Face token：

```bash
cd /workspace

# 1. 检查服务器必需命令；不执行 apt update。
for s0_cmd in git tmux jq python3 nvidia-smi flock df sha256sum; do
  command -v "${s0_cmd}" >/dev/null || {
    echo "缺少服务器命令：${s0_cmd}"
    exit 1
  }
done

# 2. 确认当前就在 Vast.ai 提供的永久 tmux session 中。
test -n "${TMUX:-}" || {
  echo "错误：当前终端不在 tmux session 中"
  exit 1
}

test "$(tmux list-sessions -F '#S' | wc -l)" -eq 1 || {
  echo "错误：服务器必须有且仅有一个 tmux session"
  tmux list-sessions
  exit 1
}

echo "当前 tmux session：$(tmux display-message -p '#S')"

# 3. 确认正好有四张 GPU。
nvidia-smi -L

test "$(nvidia-smi -L | wc -l)" -eq 4 || {
  echo "错误：没有检测到正好四张 GPU"
  exit 1
}

# 4. 检查磁盘空间。
df -h /workspace

# 5. 确保目标目录不存在，防止覆盖已有文件。
test ! -e /workspace/fe-pc-wam || {
  echo "错误：/workspace/fe-pc-wam 已存在，请不要覆盖"
  exit 1
}

# 6. 下载模型改进分支代码。
git clone \
  --branch feat/model-improvements \
  --single-branch \
  https://github.com/Jeong-zju/fe-pc-wam.git \
  /workspace/fe-pc-wam

cd /workspace/fe-pc-wam

# 7. 校验代码至少包含已验证的一键启动、B2 路由和安全终止能力。
git rev-parse --short HEAD

git merge-base --is-ancestor 2de5656 HEAD || {
  echo "错误：远程代码早于最低安全版本 2de5656"
  exit 1
}

test -x ./scripts/launch_s0_4gpu_tmux.sh
test -x ./scripts/stop_s0_4gpu_tmux.sh

# 8. 先检查一键部署计划；不会下载、训练或创建窗口。
./scripts/launch_s0_4gpu_tmux.sh \
  --run-id s0-round1 \
  --dry-run

# 9. 正式一键启动。
./scripts/launch_s0_4gpu_tmux.sh \
  --run-id s0-round1
```

正式启动时在隐藏提示中粘贴同时具备 DINOv3 gated 模型、两个训练数据集和 `RoboFactory_asset` 读取权限的 HF token。launcher 只在永久 session 中创建 `s0-round1-prepare`、`s0-round1-b0`、`s0-round1-b1`、`s0-round1-b2`、`s0-round1-b3` 和 `s0-round1-monitor`，不会创建、attach 或退出 tmux session。

#### 5.1.2 当前 S0 run 一键终止与窗口关闭

必须从永久 session 中不属于目标 run 的基础 `bash` window 执行。以下命令先核验工具、session、代码和 run manifest，打印只读终止计划，然后终止 `s0-round1` 的训练、验证、RoboFactory rollout server、dataloader 和 monitor 进程，最后关闭上述六个 window：

```bash
cd /workspace/fe-pc-wam

# 1. 检查终止器依赖；不执行 apt update。
for s0_stop_cmd in tmux jq grep nvidia-smi realpath sleep; do
  command -v "${s0_stop_cmd}" >/dev/null || {
    echo "缺少服务器命令：${s0_stop_cmd}"
    exit 1
  }
done

# 2. 确认仍在唯一的永久 tmux session 中。
test -n "${TMUX:-}" || {
  echo "错误：当前终端不在 tmux session 中"
  exit 1
}

test "$(tmux list-sessions -F '#S' | wc -l)" -eq 1 || {
  echo "错误：服务器必须有且仅有一个 tmux session"
  tmux list-sessions
  exit 1
}

cd /workspace/fe-pc-wam

# 3. 更新终止器并校验最低安全版本。
git switch feat/model-improvements
git pull --ff-only

git merge-base --is-ancestor 2de5656 HEAD || {
  echo "错误：代码不包含安全终止器 2de5656"
  exit 1
}

test -x ./scripts/stop_s0_4gpu_tmux.sh
test -f ./outputs/s0_runs/s0-round1/run_manifest.json

# 4. 只读检查：不发送信号、不关闭窗口。
./scripts/stop_s0_4gpu_tmux.sh \
  --run-id s0-round1 \
  --dry-run

# 5. 正式终止该 run 并关闭它创建的六个 window。
./scripts/stop_s0_4gpu_tmux.sh \
  --run-id s0-round1

# 6. 核对永久 session 仍存在，并查看是否还有 GPU 计算进程。
tmux list-windows \
  -F '#{window_index}: #{window_name} pane_dead=#{pane_dead}'

nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

终止器只匹配 manifest 中记录的 session/window 前缀，以及进程环境中与本轮绝对路径完全一致的 `S0_RUN_ROOT`。它先发送 Ctrl-C，等待 10 秒，再按需发送 SIGTERM 和 SIGKILL。它禁止从目标 run window 内自我终止，也不会调用 `tmux kill-session`，不会删除数据集、DINO/RoboFactory 权重、worktree、checkpoint、partial/resume checkpoint、日志、视频或 Gate JSON。

### 5.2 S0 Round 1 闭合决策（2026-07-28）

`s0-round1` 的 monitor 在 `2026-07-28T14:10:34.935096+00:00` 显示 B0、B1、B3 已完成 Gate20；冻结协议规定使用 seed `900–919`。B0/B1 均完成 80,000 updates，monitor 舍入后的末端 loss 均为 `0.002`；B3 按设计复用 B0 immutable checkpoint、只改变 chunk aggregation，因此其 `not started` 训练状态不是缺失实验。B2 在该快照中仅完成 `4,684/80,000` updates（5.9%）；因训练耗时过长，operator 已请求在 Gate20 前主动终止，不再等待其完成后才进入 R1。

| 候选 | LiftBarrier | LongPipelineDelivery | 相对 B0 | 阶段处置 |
|---|---:|---:|---|---|
| B0 sparse MoE + temporal ensemble | 17/20（85%） | 19/20（95%） | — | 选为 R1 工程父方案 |
| B1 compute-matched dense + temporal ensemble | 11/20（55%） | 0/20（0%） | `-30pp / -95pp` | 不替换 sparse MoE；待训练 seed 复验后再作架构级外推 |
| B2 Rectified Flow reference | — | — | 不可比较 | Gate20 前 operator stop；不作模型结论 |
| B3 B0 checkpoint + latest chunk | 6/20（30%） | 0/20（0%） | `-55pp / -95pp` | 否决 latest chunk；保留 temporal ensemble |

阶段判断如下：

- **B0 是后续 R1 的工程父方案。** 它是三条已完成分支中唯一在两个任务都有成功的坐标，因此 S0 不再等待 B2，可以进入下一阶段。该选择不是正式验收，也不构成 legacy action generator 普遍优于 Flow 的结论。
- **B2 记为主动停止，不记为模型失败。** 原因是训练 wall time 超出 fast-track 预算；它没有完成相同 update、没有 Gate20 成功率，因此不能按 0% 计分，也不能用于比较 B0 与 Flow。保留最后 progress、日志、partial/resume checkpoint 和 operator-stop 原因即可。
- **B0/B3 是本轮最干净的受控对比。** 冻结 manifest 与 launcher 要求 B3 复用 B0 checkpoint 和 paired seeds，且 B3 的 `not started` 与该协议一致；latest chunk 下 LPD 从 19/20 降至 0/20，说明 temporal ensemble 是有效策略的一部分，后续 legacy 对照保留 temporal ensemble。
- **B0/B1 明确否决当前 dense 替代方案，但结论强度低于 B0/B3。** 两者训练协议一致但 checkpoint 来自独立训练；当前结果足以作工程选型，不足以用单个训练 seed 声称 MoE 普遍优于 dense。
- **LPD 是更有区分力的回归任务。** B1/B3 在 LiftBarrier 尚有 11/20 和 6/20，却在 LPD 同时为 0/20；后续轮次不得用 LiftBarrier 单任务成功掩盖长时程协调与时序稳定性失败。
- **训练 loss 不参与闭环选型。** B0/B1 的 monitor 舍入末端 loss 同为 `0.002`，但 LiftBarrier 相差 30pp、LPD 相差 95pp；后续只按各任务闭环成功率选择候选。

monitor 中 `gate=pass` 对应 `gate_summary.passed=true`，`gate=done` 对应 gate 已完成但 `passed=false`。S0 已直接选择 B0 进入 R1，不再等待额外审计、正式 100-episode gate、checkpoint SHA 或 B2 结果。相关信息可以继续记录，但不阻塞推进。

### 5.3 S0 推进规则

S0 不再设置协作必要性审计或额外准入清单。B0 直接作为 R1 父方案；后续进入动作路径的候选只需与 B0 或各自父方案比较闭环成功率，S2 off-path predictor 使用第 7 节的 capability gate。

### 5.4 B0 进入 S1/R1

使用 `round/s0-b0-legacy-moe-ensemble` 作为 R1 工程父方案即可。除能够完成闭环并输出成功率外，不增加其他进入条件。

## 6. S1：Per-Agent Rectified Flow Action Expert（07-29）

统一 Flow 目标：

$$
\mathbf x_\tau=(1-\tau)\mathbf x_0+\tau\mathbf a^\star,
\qquad
\mathbf v^\star=\mathbf a^\star-\mathbf x_0,
$$

$$
\mathcal L_{\mathrm{FM}}
=
\operatorname{MaskedMean}
\left[
\left\|
\mathbf v_\theta-\mathbf v^\star
\right\|_2^2
\right].
$$

每个 agent 使用共享参数的 action expert，agent identity、task token 和本地/团队上下文通过显式 token 进入，输出保持按 agent slot 组织。

### 6.1 R1：只替换动作生成器（必做，两卡）

| 候选 | 相对父提交的唯一变量 | 固定不变 |
|---|---|---|
| R1-F0 | `action_generator=legacy_cvae`，父方案复跑 | 数据、context、当前 decoder、chunk、ensemble、训练预算 |
| R1-F1 | `action_generator=rectified_flow_cold` | 与 F0 相同 |

F1 的 head、FM loss 和 ODE solver 作为一个原子垂直切片共同实现；不得同时把 MoE 改成 dense、加入 warm start、改变 temporal ensemble 或接入 future latent。默认使用 4-step Euler，1-step Euler 与 2-step Heun 延后为冻结 checkpoint 上的推理消融。

R1 只比较 F0/F1 的同任务闭环成功率。若 F1 在每个任务都不低于 F0，则通过并进入后续阶段，成功率持平也算通过；若任一任务下降，则保留 F0。

#### 6.1.1 S1-R1 两分支与两卡隔离契约

S1-R1 从 `feat/model-improvements` 的同一公共基础设施提交创建两个分支：`s1/r1-f0-legacy` 只把 S0-B0 固化为 R1-F0 复跑坐标，`s1/r1-f1-flow-cold` 只完成 `legacy_cvae → rectified_flow_cold` 原子替换。两个分支分别使用 GPU 0/1、独立 worktree、独立 checkpoint/output/log，但通过符号链接只读共享模型改进分支中的 `datasets/` 与 `artifacts/`，并共享同一个 uv 环境、uv cache 和 RoboFactory 安装。F0/F1 都固定 80,000 updates、训练 seed 101、Gate20 seeds 900–919、sparse top-2 MoE、DINOv3、100-step chunk 和 temporal ensemble；F1 默认使用标准高斯 cold source、FM velocity MSE 和 4-step Euler，不开启 warm start、future path、dense decoder 或 active-agent weighting。

截至 2026-07-28，本轮公共父提交为 `65ad9de`，F0 实现提交为 `f0043ff`，F1 实现提交为 `00a29c6`。这三个提交只表示代码与运行契约已冻结，不表示 F1 已通过：只有远程训练结束后 F1 在 LiftBarrier 与 LongPipelineDelivery 的 Gate20 成功率都不低于 F0，R1 才能选择 Flow。

##### S1-R1 Round 2 结果与决策（2026-07-29）

`s1-r1-round2` 的 F0/F1 都完成 `80,000/80,000` updates；monitor 舍入后的末端 loss 分别为 `0.003` 和 `0.012`。相同 Gate20 协议下，F0 在 LiftBarrier/LongPipelineDelivery 分别为 `11/20`、`20/20`，F1 分别为 `13/20`、`20/20`。F1 在 LiftBarrier 提高 `2/20`（10 个百分点），在 LongPipelineDelivery 持平，因此满足“每个任务均不低于 F0”的 R1 推进规则，选择 F1 并将 `s1/r1-f1-flow-cold` 合入 `feat/model-improvements`。

| 候选 | 训练 | 末端 loss（monitor） | LiftBarrier | LongPipelineDelivery | R1 决策 |
|---|---:|---:|---:|---:|---|
| F0 legacy CVAE | 80,000/80,000 | 0.003 | 11/20（55%） | 20/20（100%） | 控制组 |
| F1 Rectified Flow cold | 80,000/80,000 | 0.012 | 13/20（65%） | 20/20（100%） | 通过并晋升 |

F1 monitor 的 `failed` 不是闭环失败：两个任务的 40 个 rollout 均已完成，退出码来自 `build_lpd_gate_summary.py` 只接受 `wam/static_act`、不接受 F1 的 `agent_flow` policy kind。合入后的汇总器已把 `agent_flow` 纳入文件型 checkpoint 路径；同步远程原始 rollout 后只需重建 `gate_summary.json`，不需要重新训练或重跑 40 个回合。在汇总 JSON、checkpoint 哈希与 episode-level 记录同步前，本节数字仍按 operator-reported Gate20 结果使用，不外推为正式 100 回合验收或统计显著性结论。

launcher 复用服务器已经存在的唯一永久 tmux session，只创建 `<run-id>-prepare`、`<run-id>-f0`、`<run-id>-f1` 和 `<run-id>-monitor` 四个 window，并为每个 window 设置 `remain-on-exit=on`；它不会创建、attach 或退出 tmux session。monitor 同时显示两条训练的 update/loss、两个闭环任务的成功数、候选 phase 和两张 GPU 的利用率/显存。训练或验证进程退出后 window 仍保留，便于查看日志。

#### 6.1.2 Vast.ai 两卡从零一键部署、训练、验证与监控

以下命令假设 Vast.ai 已经自动进入唯一的永久 tmux session，服务器恰好暴露两张 GPU，且 `/workspace/fe-pc-wam` 尚不存在。命令不会执行 `apt update`，HF token 只会通过隐藏输入和 mode-0600 FIFO 交给共享准备 window，不写进 shell export、tmux command 或 argv：

```bash
cd /workspace

for s1_cmd in git tmux jq python3 nvidia-smi flock df sha256sum; do
  command -v "${s1_cmd}" >/dev/null || {
    echo "缺少服务器命令：${s1_cmd}"
    exit 1
  }
done

test -n "${TMUX:-}" || {
  echo "错误：当前终端不在 Vast.ai 的永久 tmux session 中"
  exit 1
}

test "$(tmux list-sessions -F '#S' | wc -l)" -eq 1 || {
  echo "错误：服务器必须有且仅有一个 tmux session"
  tmux list-sessions
  exit 1
}

test "$(nvidia-smi -L | wc -l)" -eq 2 || {
  echo "错误：S1-R1 必须恰好暴露两张 GPU"
  nvidia-smi -L
  exit 1
}

df -h /workspace

test ! -e /workspace/fe-pc-wam || {
  echo "错误：/workspace/fe-pc-wam 已存在，请不要覆盖"
  exit 1
}

git clone \
  --branch feat/model-improvements \
  --single-branch \
  https://github.com/Jeong-zju/fe-pc-wam.git \
  /workspace/fe-pc-wam

cd /workspace/fe-pc-wam
git rev-parse --short HEAD

test -x ./scripts/launch_s1_r1_2gpu_tmux.sh
test -x ./scripts/stop_s1_r1_2gpu_tmux.sh

./scripts/launch_s1_r1_2gpu_tmux.sh \
  --run-id s1-r1-round1 \
  --dry-run

./scripts/launch_s1_r1_2gpu_tmux.sh \
  --run-id s1-r1-round1
```

正式启动时在隐藏提示中粘贴同时具备 DINOv3 gated 模型、两个训练数据集和 `RoboFactory_asset` 读取权限的 HF token。启动后 launcher 默认切到 `s1-r1-round1-monitor`；可随时从永久 session 的任意非目标 window 执行以下只读监测指令：

共享准备只调用官方基础下载命令 `hf download`：固定关闭 Xet，并用 `--max-workers 1` 串行走普通 HTTP，以避免云主机共享出口请求 `xet-read-token` 时出现 `429 Too Many Requests`。脚本不包含并发下载或重试封装；下载失败后，以新 run-id 重新启动会原地复用已完成文件并续传。

```bash
cd /workspace/fe-pc-wam

python3 scripts/s1_r1_runtime.py monitor \
  --once \
  --run-root outputs/s1_r1_runs/s1-r1-round1

tmux select-window \
  -t "$(tmux display-message -p '#S'):s1-r1-round1-monitor"
```

所有运行产物位于 `/workspace/fe-pc-wam/outputs/s1_r1_runs/s1-r1-round1/`。共享准备日志和哈希分别为 `prepare.log`、`shared_artifact_sha256.txt`；F0/F1 的训练进度、checkpoint、验证 JSON、视频和完整候选日志分别位于 `candidates/f0/` 与 `candidates/f1/`。monitor 中 Gate20 的 `lift=x/20`、`lpd=y/20` 是本轮唯一推进依据：只有 F1 两个任务都不低于 F0 才进入 R2。

共享准备完成后，两个候选还要分别完成数据 manifest/HDF5 身份校验、DINOv3 权重装载、模型与 optimizer 构建、resume 检查、DataLoader worker 启动和首批数据读取。两张 RTX 5090 的常见冷启动时间为 3–15 分钟；云盘较慢时可能达到 20–30 分钟。候选 window 在等待共享准备时每 30 秒打印一次心跳；训练器把上述子阶段写入 `candidates/<f0|f1>/train/stages.jsonl`。monitor 每 5 秒显示当前 startup 子阶段、该阶段持续时间以及 GPU 利用率，产生第一个 optimizer step 后自动切换为 `training` 并显示 update/loss。

#### 6.1.3 S1-R1 一键退出但保留永久 tmux 与全部产物

退出脚本必须从永久 session 中不属于 `s1-r1-round1-prepare/f0/f1/monitor` 的基础 `bash` window 执行。它只根据 run manifest 和进程环境中的绝对 `S1_R1_RUN_ROOT` 定位本轮进程，依次发送 Ctrl-C、SIGTERM、必要时 SIGKILL，再关闭本轮四个 window；不会调用 `tmux kill-session`，不会删除共享数据、DINO/RoboFactory 权重、worktree、checkpoint、resume、日志、视频或验证结果：

```bash
cd /workspace/fe-pc-wam

git switch feat/model-improvements
git pull --ff-only

test -f ./outputs/s1_r1_runs/s1-r1-round1/run_manifest.json
test -x ./scripts/stop_s1_r1_2gpu_tmux.sh

./scripts/stop_s1_r1_2gpu_tmux.sh \
  --run-id s1-r1-round1 \
  --dry-run

./scripts/stop_s1_r1_2gpu_tmux.sh \
  --run-id s1-r1-round1

tmux list-windows \
  -F '#{window_index}: #{window_name} pane_dead=#{pane_dead}'

nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

一键退出完成后，Vast.ai 的永久 tmux session 必须仍存在；若之后要恢复训练，保留的 `resume.pt` 会被各候选训练器读取，但为避免复用已经关闭的 window/run manifest，应使用新的 `--run-id` 启动并按需把对应 resume/checkpoint 放入新 run 的候选隔离目录。

若 `s1-r1-round1` 在共享下载阶段失败且尚未执行上述退出指令，应先切到永久 session 中任意非 S1-R1 基础 window，再执行：

```bash
cd /workspace/fe-pc-wam

./scripts/stop_s1_r1_2gpu_tmux.sh --run-id s1-r1-round1

git switch feat/model-improvements
git pull --ff-only

./scripts/launch_s1_r1_2gpu_tmux.sh --run-id s1-r1-round2
```

该流程只关闭 round1 的四个 window，不删除 `/workspace/fe-pc-wam/datasets`、`/workspace/fe-pc-wam/artifacts`、`/workspace/RoboFactory`、Hub 下载缓存或 round1 日志；round2 会继续使用这些已有内容。

若 F1 已完成 80,000 updates、仅在闭环握手、rollout 或汇总阶段失败，不得重新训练。应进入 F1 worktree，明确切换并更新 `s1/r1-f1-flow-cold`，然后用新的 retry-id 复用 round2 checkpoint，只重跑 F1 Gate20：

```bash
cd /workspace/worktrees/s1-r1-f1-flow-cold
git switch s1/r1-f1-flow-cold
git pull --ff-only origin s1/r1-f1-flow-cold

./scripts/retry_s1_r1_f1_gate.sh \
  --run-id s1-r1-round2 \
  --retry-id retry1
```

retry 输出写入 round2 的 `candidates/f1/validation/gate_s1-r1-round2_retry1/`，日志写入 `candidates/f1/logs/gate_s1-r1-round2_retry1.log`；已有 checkpoint、训练进度和首次失败的验证目录全部保留。

### 6.2 R2a/R2b：两个可选单变量微轮次（可四卡并行）

R1-F1 已通过并随合并提交 `ae7dc95` 进入 `feat/model-improvements`，冻结为后续阶段的 Flow 工程父方案 `P_flow`。以下两对候选可以同时租用四张卡：

| 微轮次 | P0 控制 | P1 单步改进 | 唯一变量 |
|---|---|---|---|
| R2a Decoder | 当前 `P_flow` decoder | 仅切换 `top-2 MoE ↔ dense FFN` | decoder family |
| R2b Source prior | Gaussian cold start | previous-chunk warm start | Flow source distribution |

R2a 不改变 source prior；R2b 不改变 decoder。每个 P1 只要各任务闭环成功率不低于对应 P0 就可以保留，持平也算通过。若两项都通过，可以直接进入组合闭环；组合方案相对组合父方案没有成功率退步即可继续。

**2026-07-30 决策：主路径不运行 R2。** R2a 在 S0 已有同方向证据：dense B1 相对 MoE B0 闭环明显退步，再花完整训练预算只会重复一个低信息量问题；R2b 的 previous-chunk warm start 改变 source distribution，既不是 S2 所需依赖，也会把跨回合分布漂移带入新的 world-model 对照。R2a 标记为跳过，R2b 移入非阻塞 backlog；以后若有空闲卡，R2b 只能作为独立 sidecar ablation，不能更换 S2 父方案或阻塞主线。

### 6.3 进入 S2

S2 固定从当前 `feat/model-improvements` 上的 F1 `rectified_flow_cold` 父方案进入；模型修改父提交为 `caa5ed3`，Flow checkpoint 优先采用已经完成 Gate20 的 R1-F1 checkpoint。S2 不再等待 R2a/R2b。正常情况下不增加 Flow 训练；若租用的新实例和持久盘都已经没有该 checkpoint，则 S2 launcher 会按已经晋升的冻结 F1 配方自动重建，而不是因缺文件永久阻塞。

## 7. S2：Agent-Factorized Action-Conditioned World Model（07-30 至 08-10）

本阶段冻结 F1 Flow 与 DINOv3，future predictor 严格保持在动作路径之外。S2 回答三个能力问题：local predictor 是否真正读取自己的候选 action chunk，team predictor 是否真正读取 peer action 并预测跨机器人/共享场景后果，以及 team capability 能否在结构上不改写合格的 own predictor。因为 predictor off-path 时不可能改变策略动作，S2 不用闭环成功率比较 W0/W1 或 P0/P1；只运行一次 predictor-disabled action-equivalence smoke，world-to-action 收益统一留到 S3。

### 7.1 S2.0：先建立 grouped trajectory contract

复用当前 manifest 与轨迹文件，不重采数据。新增 S2 专用 grouped adapter，保留 episode 内的 agent 维与共享视角；不得修改 S1 已冻结的 legacy `_local_batch` 语义。adapter 必须输出以下张量：

| 字段 | 固定 contract |
|---|---|
| current agent state | `[B, A, 18]` |
| executed/candidate action chunk | `[B, A, H, 8]` |
| agent-view observation | `[B, A, ...]` |
| global/shared-view observation | `[B, ...]`，与 agent slots 分开保留 |
| valid-agent mask | `[B, A]`，第一版 `A_max=4` |
| future targets/masks | `k ∈ {1, 25, 50, 100}`，越过 episode 边界的 target 必须 mask |

S2.0 必须先通过四类单元测试：group/agent/global shape 保持、future index 与 episode 边界 mask 正确、invalid-agent slot 对 loss 为零、predictor disabled 时 F1 输入动作与输出动作逐元素一致且 Flow/DINO checkpoint hash 不变。任一测试失败都不启动 R3。

### 7.2 Future representation 与损失

第一版不生成 RGB。每个 horizon 预测两类增量 target：归一化 proprioceptive state delta `s_{t+k}^i-s_t^i`，以及冻结 DINOv3 patch feature 的空间池化 latent delta。DINO feature 先按固定网格池化，再用仅在 train split 拟合的 PCA 从 1024 维压到 256 维；PCA basis、归一化统计、DINO checkpoint 与数据 manifest hash 都写入 checkpoint，验证/测试阶段不得重拟合。

state 使用 masked Smooth-L1，DINO delta 使用 masked cosine distance；两项先用 train split 统计量标准化，再固定等权相加：

$$
\mathcal L_{\mathrm{future}}
=
\mathcal L_{\mathrm{state}}
+
\mathcal L_{\mathrm{dino}}.
$$

真实 future 只能作为训练/验证 target，禁止作为 predictor 输入。local predictor 输出 own-state/own-view future；team predictor 额外输出所有 valid peer 的 state/view future 与 global/shared-view future。

### 7.3 Candidate-action contract

训练时使用数据中的归一化 executed action chunk 作为干净的 causal candidate。R3-W0 使用同构 action adapter，但 action 输入置零并 mask；R3-W1 输入自己的完整 action chunk。不能把单个 noisy $\mathbf x_\tau$ 作为唯一 action condition，因为早期 $\tau$ 的信号主要是噪声，容易把“模型没读动作”误判成 world model 不成立。

为 S3 预留的推理 contract 是：每个 solver step 先由冻结 base Flow 给出 provisional clean endpoint，再以 stop-gradient 方式连同 $\tau$ 送入 predictor：

$$
\hat{\mathbf a}_1^i
=
\operatorname{clip}
\left(
\mathbf x_\tau^i
+
(1-\tau)\,
\mathbf v_{\mathrm{base}}^i(\mathbf x_\tau,\tau,\mathbf h_t)
\right).
$$

local 模式只可读取本 agent 的 context 与 candidate action；team 模式可读取所有 valid agents 的 context/action 和 global slot，使用共享参数与显式 masks，不按 agent 数复制独立网络。

### 7.4 R3：Action conditioning capability（必做，两卡）

| 候选 | Predictor 输入 | Future target | 唯一变量 |
|---|---|---|---|
| R3-W0 | local context；action adapter 输入置零并 mask | own state + own DINO latent | 无候选动作信息 |
| R3-W1 | local context + own executed action chunk | 与 W0 完全相同 | `action_conditioning=on` |

W0/W1 必须从相同初始化开始，使用相同网络、target、horizon、width、参数量、训练更新、optimizer 与固定 held-out trajectory split。R3 只有同时满足以下条件才通过：

1. 每个任务上 W1 的 held-out composite future loss 都不高于 W0，且至少一个任务严格改善；
2. 每个任务分别做 paired action shuffle，`L_shuffled-L_normal>0`，episode-level paired bootstrap 的 95% 下界也必须大于 0；
3. predictor disabled 时通过 F1 action-equivalence smoke，且 Flow/DINO 无梯度、无参数或 buffer 变化。

若 action shuffle 不能稳定增大误差，结论是 predictor 没有利用候选动作；停止进入 R4，优先检查 action normalization、temporal alignment 与 adapter，再只允许一次修复重跑。不能用闭环持平把 W1 判为通过。

#### 7.4.1 S2-R3 两分支、五任务和两卡运行契约（2026-07-30）

S2.0 公共基础设施先落在 `feat/model-improvements`，再从同一个公共父提交创建 `s2/r3-w0-action-independent` 和 `s2/r3-w1-action-conditioned`。W0/W1 使用同一个 `LocalActionConditionedFuturePredictor` 类、相同参数量、seed `303`、10,000 updates、batch size `1`、optimizer、五任务 train/validation split、DINOv3/PCA 工件和 S1-R1 F1 checkpoint；分支配对检查器会删除候选 identity 与隔离输出路径后逐字段比较配置，除 `action_conditioning=false/true` 外存在任何差异都会拒绝启动。训练与验收白名单显式加入 `s2_r3_local_action_independent` 和 `s2_r3_local_action_conditioned`，未知 model kind fail closed。

五任务联合训练、联合 held-out 验证固定使用以下不可变 Hugging Face dataset revision，并通过同一个基础仓库下的 `datasets/robofactory_multitask/` 只读共享给两个 worktree：

| 任务 | Hugging Face dataset | revision | 本地目录 | 实际 RGB 相机 |
|---|---|---|---|---|
| LiftBarrier | `zeno-ai/robofactory-lift-barrier-multiview` | `6ab620091677e69370412f08cd7adecacc28c146` | `lift_barrier/` | `global, agent_0, agent_1` |
| LongPipelineDelivery | `zeno-ai/robofactory-long-pipeline-delivery-multiview` | `fee628311ff52a3ae0ddfddf82379c63d28f7533` | `long_pipeline_delivery/` | `global, agent_0..3` |
| TakePhoto | `zeno-ai/robofactory-take-photo-multiview` | `3966385a4c688a5610d4b6cde044150f6b73d320` | `take_photo/` | `global, agent_0, agent_1, agent_2, agent_3` |
| ThreeRobotsStackCube | `zeno-ai/robofactory-three-robots-stack-cube-multiview` | `d0ae346bf2ce63ec801af1f036c08a4a91faf366` | `three_robots_stack_cube/` | `global, agent_0, agent_1, agent_2` |
| CameraAlignment | `zeno-ai/robofactory-camera-alignment-multiview` | `e204af13f7191dfd86dab3da529316a51558f479` | `camera_alignment/` | `global, agent_0, agent_1, agent_2` |

补齐全部 agent 相机后，五仓库固定 revision 的 Hub `used_storage` 合计约 784 GiB。launcher 不再用不适用于原地升级的固定空闲门槛，而是按每个 revision 的目标字节数减去当前本地目录字节数计算净增长，再额外要求 32 GiB 单文件替换/续传余量；全新实例与旧 global-only 数据原地升级使用同一检查。2026-07-30 的服务器实测确认，关闭 Xet 且单 worker 时普通 HTTP 单连接只有约 `4.6 MiB/s`，而 LongPipelineDelivery 单个 HDF5 平均约 `2.37 GiB`；因此五个大型训练集恢复 S0 `9bf88ff` 的已验证传输策略：官方 `hf download` 保持 Xet 开启、不传 `--max-workers`（当前锁定 CLI 的默认值为 8）、失败后最多 5 次指数退避。DINOv3 仍关闭 Xet并固定 `--max-workers 1`。两类下载都固定不可变 revision、`HF_HUB_DOWNLOAD_TIMEOUT=600`、`HF_HUB_ETAG_TIMEOUT=60` 和最终 `--local-dir`，不调用 `snapshot_download`，也不创建第二份 snapshot。

网络抖动时官方客户端在同一本地 cache/`--local-dir` 恢复；只有 prepare 变为 `failed` 才是本轮失败。脚本每 15 秒把当前任务已完成的 episode 数写入 shared status。重启时不能只凭已经先行下载的 `training_manifest.json` 判定完成：快速完整性检查会确认该任务全部 150 个 HDF5、normalization 和 conversion manifest 均已落盘，否则继续复用已完成文件和本地 cache。S0 模式可能重新获取由普通 HTTP 留下但尚未完成的单文件 `.incomplete`，不得手工删除 cache 或已经完成的 HDF5。

grouped adapter 保留 current state `[B,4,18]`、candidate chunk `[B,4,100,8]`、五个固定相机槽位、独立 global RGB `[B,...]`、valid-agent mask `[B,4]` 和 `k={1,25,50,100}` future mask。五个正式数据集均保留 global 加全部实体 agent 相机，local predictor 的 current/future DINO target 只读取真实 agent-view；loader 的 canonical-prefix/global fallback 兼容路径仅供不完整数据诊断，正式 artifact preflight 会拒绝使用。DINOv3 patch feature 固定池化到 `2×2` 网格，再用只读取 train split 的 PCA 从 1024 维压到 256 维；PCA basis、projected std、state/DINO delta normalization、五个 manifest hash、每任务实际相机契约和 DINO hash 保存在 `artifacts/s2_r3/dino_pca_statistics.pt`，并完整嵌入候选 checkpoint。future state/RGB 只用于 target builder，不进入 predictor input。

正式五任务 R3/旧 R4 训练启动前，quick local verifier 还必须逐任务确认训练 manifest 的相机顺序严格等于 `global + 全部实体 agent`（LiftBarrier 2、LongPipelineDelivery/TakePhoto 4、ThreeRobotsStackCube/CameraAlignment 3）；只有 `global` 的过渡数据会在 DINO/PCA 之前 fail closed，不能产生正式 artifact。PCA/statistics 对 episode 边界的全 false future-view mask 作空批次跳过，绝不把零帧张量传入 DINO；若整个 horizon 最终没有任何有效 visual target，则以明确的 `empty horizon` 数据错误停止，而不是产生无效统计量。

R3 验收器不运行无区分力的成对闭环。它在每个 validation episode 固定选择 4 个时间窗，分别输出 normal 与 own-action-shuffle composite future loss，再按 episode 聚合并运行 10,000 次 paired bootstrap。`acceptance.json` 只有在五个任务上同时满足 W1 loss 不高于 W0、至少一个任务严格改善、W1 `L_shuffled-L_normal>0` 且 bootstrap 95% 下界大于 0时才通过；同时还要求 predictor-disabled F1 action output 逐元素相等、Flow/DINO 文件 hash 前后不变、predictor checkpoint 不含 Flow/DINO state。monitor 直接读取这套特殊规则，不把闭环成功率或 W0 的零 shuffle delta 误当作 R3 通过条件。

#### 7.4.2 S2-R3 正式验收结论（2026-07-31）

正式 run `s2-r3-round1-full-cameras` 已在两张 RTX 5090 上完成 W0/W1 各 10,000 updates、五任务 held-out 验证和成对 own-action shuffle 验收。两个训练进程退出码均为 0，501 个训练记录点中的 total/state/visual loss 与 gradient norm 均为有限值，无 NaN、OOM 或 Traceback。训练前 1,000 步与最后 1,000 步的均值如下：

| 候选 | 前 1,000 步 loss | 最后 1,000 步 loss | 最后 1,000 步 state loss | 最后 1,000 步 visual loss |
|---|---:|---:|---:|---:|
| R3-W0 | 1.092161 | 0.729984 | 0.146621 | 0.583363 |
| R3-W1 | 1.087506 | 0.721665 | 0.140173 | 0.581492 |

正式 `acceptance.json` 的五任务结果如下。`W1 改善` 为 `(W0-W1)/W0`；shuffle 指标为 W1 的 `L_shuffled-L_normal`，置信区间使用 15 个 episode、每 episode 4 个固定窗口和 10,000 次 episode-level paired bootstrap：

| 任务 | W0 held-out loss | W1 held-out loss | W1 改善 | W1 action-shuffle Δ | bootstrap 95% 下界 |
|---|---:|---:|---:|---:|---:|
| CameraAlignment | 0.776281 | 0.768086 | 1.06% | 0.005730 | 0.004727 |
| LiftBarrier | 1.019633 | 1.013257 | 0.63% | 0.004874 | 0.002807 |
| LongPipelineDelivery | 0.571899 | 0.565914 | 1.05% | 0.064856 | 0.060579 |
| TakePhoto | 0.749629 | 0.739450 | 1.36% | 0.033809 | 0.030487 |
| ThreeRobotsStackCube | 0.693697 | 0.683910 | 1.41% | 0.025696 | 0.023503 |
| **五任务宏平均** | **0.762228** | **0.754123** | **1.06%** | — | — |

正式验收的七项检查全部通过：

1. 任务集合严格等于预注册的五任务；
2. Flow 与 DINOv3 在训练/验证前后保持冻结且文件 hash 不变；
3. predictor disabled 时 F1 动作输出逐元素相等，最大绝对差为 `0.0`；
4. W0/W1 的初始化、模型预算、训练和验证选择契约相同，唯一研究变量是 `action_conditioning=false/true`；
5. 五任务 W1 action-shuffle 均值和 bootstrap 95% 下界全部大于 0；
6. 五任务 W1 held-out loss 均不高于 W0；
7. 至少一个任务严格改善；本次实际为五个任务全部严格改善。

额外的非门槛配对复核中，W1 held-out loss 优于 W0 的 episode 数分别为 CameraAlignment `14/15`、LiftBarrier `12/15`、LongPipelineDelivery `15/15`、TakePhoto `15/15`、ThreeRobotsStackCube `13/15`；对 W0-W1 episode 差重新 bootstrap 后，五任务探索性 95% 区间也均位于 0 以上。LiftBarrier 有 2/15 个 episode 的 action-shuffle delta 为负，但正式的任务聚合均值与预注册 episode-bootstrap 下界仍明确为正，因此不触发回退。

**正式结论：S2-R3 PASS，选择 R3-W1 作为当时旧 R4 的 local parent。** 该结论证明 local predictor 确实读取 own candidate action，并在同预算 held-out future prediction 上一致优于 action-independent W0；它不声称 off-path predictor 已提高闭环成功率，world-to-action 收益仍留到 S3 检验。当前证据来自固定 seed `303` 的一轮训练，后续可以补多 seed 作为论文稳健性分析，但不阻塞新 R4/R5 protected-own 路线。

正式产物：

- pair-level 验收：`outputs/s2_r3_runs/s2-r3-round1-full-cameras/acceptance.json`，decision=`pass_enter_r4`；
- W1 checkpoint：`outputs/s2_r3_runs/s2-r3-round1-full-cameras/candidates/w1/checkpoints/predictor.pt`；
- W1 checkpoint SHA256：`1a7fab018777b37803e4457406ed8893556e029fd331549a9f9ed51ffac524aa`；
- 五任务 PCA/statistics SHA256：`692abb2d5476091549a40c00e8653903089a3a4231da71aebe8472c833211e5e`。

candidate status 中 W0 的 detail 显示 `PASS: enter R4`、W1 显示 `pending peer evaluation` 仅由 W0 最后完成并负责写入 pair-level `acceptance.json` 导致，不表示 W0 获胜；最终选择以 `acceptance.json` 和本节的成对结果为准。

#### 7.4.3 两张 RTX 5090 一键部署、训练、验证与 monitor

以下命令假设服务器已经自动进入唯一的永久 tmux session，恰好暴露两张 RTX 5090，并能通过上述“缺失数据净增长 + 32 GiB”动态磁盘检查。S2 按以下顺序获取父 Flow：先使用有效的 `S2_R3_FLOW_CHECKPOINT`，再复用 `artifacts/s1_r1_f1/checkpoint_080000.pt`，然后搜索 `outputs/s1_r1_runs/*/candidates/f1/checkpoints/s1_r1_f1_flow_cold/checkpoint_080000.pt`；三处都不存在时，在五任务数据和 DINO 准备完成后自动用 GPU0 重训冻结的 S1-R1 F1 配方。恢复训练固定 seed `101`、batch size `4`、80,000 updates、标准高斯 cold source 和 4-step Euler；W0/W1 此时持续报告等待心跳，重训和验证完成后才分别占用 GPU0/GPU1。

自动恢复的完成 checkpoint 固定写到 `artifacts/s1_r1_f1/checkpoint_080000.pt`，每 1,000 updates 写入可跨 S2 run-id 复用的 `artifacts/s1_r1_f1/recovery/resume.pt`。中断后以新 run-id 启动会自动从该 resume 续训；训练完成后 resume 自动删除，并生成 `artifacts/s1_r1_f1/recovery/recovery_receipt.json`。receipt/验证器会 fail closed 地核对 checkpoint format、80k update、F1 method、模型/训练/DINO/generation 配置、config SHA256 以及原两任务 manifest SHA256。这里重建的是路线中已经完成 F0/F1 Gate20 并晋升的冻结 F1 配方，不重新开启 R1 模型选择，也不要求已经丢失的 F0 checkpoint。

```bash
cd /workspace

for s2_cmd in git tmux jq python3 nvidia-smi flock df sha256sum find sort grep; do
  command -v "${s2_cmd}" >/dev/null || {
    echo "缺少服务器命令：${s2_cmd}"
    exit 1
  }
done

test -n "${TMUX:-}" || {
  echo "错误：当前终端不在永久 tmux session 中"
  exit 1
}

test "$(tmux list-sessions -F '#S' | wc -l)" -eq 1 || {
  echo "错误：服务器必须有且仅有一个 tmux session"
  tmux list-sessions
  exit 1
}

test "$(nvidia-smi -L | wc -l)" -eq 2 || {
  echo "错误：S2-R3 必须恰好暴露两张 GPU"
  nvidia-smi -L
  exit 1
}

df -h /workspace

test ! -e /workspace/fe-pc-wam || {
  echo "错误：/workspace/fe-pc-wam 已存在，请不要覆盖"
  exit 1
}

git clone \
  --branch feat/model-improvements \
  --single-branch \
  https://github.com/Jeong-zju/fe-pc-wam.git \
  /workspace/fe-pc-wam

cd /workspace/fe-pc-wam
git rev-parse --short HEAD

test -x ./scripts/launch_s2_r3_2gpu_tmux.sh
test -x ./scripts/stop_s2_r3_2gpu_tmux.sh

./scripts/launch_s2_r3_2gpu_tmux.sh \
  --run-id s2-r3-round1 \
  --dry-run

./scripts/launch_s2_r3_2gpu_tmux.sh \
  --run-id s2-r3-round1
```

正式启动时只在隐藏提示中输入一次 HF token；token 通过 mode-0600 FIFO 交给 prepare window，不写入 shell export、tmux command、argv、manifest 或日志。launcher 复用当前唯一永久 session，只创建 `s2-r3-round1-prepare`、`s2-r3-round1-w0`、`s2-r3-round1-w1`、`s2-r3-round1-monitor` 四个 window，并全部设置 `remain-on-exit=on`；不会创建、attach、kill 或退出 tmux session。prepare 在需要时先使用 GPU0 恢复 S1-R1 F1，再使用 GPU0 完成 PCA/statistics；共享 ready 文件产生后两候选才开始分别占用 GPU0/GPU1。若操作员另有有效 checkpoint，仍可在启动前设置 `S2_R3_FLOW_CHECKPOINT=/absolute/path/checkpoint_080000.pt`，但它不是全新实例的一键启动前置条件。

monitor 每 5 秒显示 shared prepare 当前程序/阶段、20 秒共享心跳及其 age；触发 Flow 自动恢复时额外显示 startup 子阶段或 `S1-R1 F1 recovery update/80000、百分比、loss`。它还显示 W0/W1 当前程序、phase、各自心跳、update/total/loss、当前验证 task/batch/shuffle delta、两卡利用率/显存和 GPU process PID。两个 evaluation 都完成后还会逐任务显示 W0/W1 held-out loss、`W0-W1`、W1 shuffle delta、bootstrap 95% lower bound，以及每条特殊 gate 的 PASS/FAIL，明确给出 `PASS -> enter R4` 或 `FAIL -> stop before R4`。可随时从永久 session 的任意 window 执行只读查询：

```bash
cd /workspace/fe-pc-wam

python3 scripts/s2_r3_runtime.py monitor \
  --once \
  --run-root outputs/s2_r3_runs/s2-r3-round1

tmux select-window \
  -t "$(tmux display-message -p '#S'):s2-r3-round1-monitor"
```

若旧版本 prepare 使用关闭 Xet的单 worker，单个 episode 的预计时间持续升高，先从本轮四个 window 之外的基础 window 安全停止旧 run，再拉取 S0 加速传输修复并使用新 run-id。不得删除 `datasets/robofactory_multitask/*/.cache/huggingface/`、任何 `.incomplete` 或已经完成的 HDF5；新 run 会复用全部已完成文件与本地 cache：

```bash
cd /workspace/fe-pc-wam

./scripts/stop_s2_r3_2gpu_tmux.sh \
  --run-id s2-r3-round1

git switch feat/model-improvements
git pull --ff-only origin feat/model-improvements

./scripts/launch_s2_r3_2gpu_tmux.sh \
  --run-id s2-r3-round1-resume1
```

所有 run 产物位于 `outputs/s2_r3_runs/s2-r3-round1/`：`prepare.log`、`prepare_progress.jsonl`、自动恢复触发时的 `flow_recovery_{stages,progress}.jsonl`、`shared_artifact_sha256.txt`、`candidates/<w0|w1>/train/{stages,progress}.jsonl`、candidate checkpoint/resume、`candidates/<w0|w1>/validation/{progress.jsonl,evaluation.json}` 和最终 `acceptance.json`。跨 run 保留的 Flow checkpoint/resume/receipt 位于 `artifacts/s1_r1_f1/`。心跳超过 75 秒会显示 `STALE`；这表示当前程序没有健康回报，应先看对应 candidate/prepare log 和 GPU process，而不是把最后一个 loss 当作仍在运行。

#### 7.4.4 一键退出但永久 tmux 和全部数据/结果必须保留

从永久 session 中不属于本轮四个目标 window 的基础 `bash` window 执行：

```bash
cd /workspace/fe-pc-wam

test -f ./outputs/s2_r3_runs/s2-r3-round1/run_manifest.json

./scripts/stop_s2_r3_2gpu_tmux.sh \
  --run-id s2-r3-round1 \
  --dry-run

./scripts/stop_s2_r3_2gpu_tmux.sh \
  --run-id s2-r3-round1

tmux list-windows \
  -F '#{window_index}: #{window_name} pane_dead=#{pane_dead}'

nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

退出脚本只定位进程环境中绝对匹配 `S2_R3_RUN_ROOT` 的本轮进程，依次发送 Ctrl-C、SIGTERM、必要时 SIGKILL，再关闭本轮四个 window；禁止调用 `tmux kill-session`，也不删除共享五任务数据、Hub 原地下载缓存、DINO/PCA/Flow、worktree、checkpoint、resume、日志或验证 JSON。中断后使用新的 `--run-id` 重启；Flow 自动恢复会直接读取共享 `artifacts/s1_r1_f1/recovery/resume.pt`。W0/W1 candidate trainer 的 resume 仍属于 run 隔离目录，如需续训，应先把保留的 candidate `resume.pt` 放入新 run 对应 candidate checkpoint 目录，禁止复用已关闭 run 的 manifest/window 名。

### 7.5 新 R4：零训练 hybrid checkpoint 诊断（必做，单卡即可）

#### 7.5.1 旧 R4 的实验结论与路线变更依据

旧 R4 从同一个 R3-W1 parent 分别训练 Local P0 与 Team+shared P1。P1 已在五个任务上通过 peer/shared persistence baseline 与 peer-action-shuffle 因果门槛，说明模型已经具备跨 agent/shared future capability；但 P1 在 `LiftBarrier` 和 `ThreeRobotsStackCube` 的 own-target no-regression 门槛失败，因此不能晋级。

针对 own 回归已依次完成三项隔离诊断：

| 隔离项 | 结果 | 排除的解释 |
|---|---|---|
| 将旧 P1 checkpoint 的 `own_residual_gate` 临时置零后重评估，不训练 | 五个任务 own loss 全部变差；以 `P0 loss - P1 loss` 计，LiftBarrier/ThreeRobotsStackCube 分别恶化到 `-0.004902/-0.006259`，peer/shared 结果基本不变 | own residual 不是回归根因，当前 gate 实际在补偿 own 预测 |
| local 与 team/shared 参数分组裁剪，避免全模型 gradient norm 耦合 | 仍未满足逐任务 own no-regression | 全局梯度裁剪不是充分原因 |
| team dropout 使用独立 RNG 作用域，不再消耗 local dropout 序列 | 重新训练后仍未满足逐任务 own no-regression | 随机数流串扰不是充分原因 |

**实验事实：** 当前 shared/multi-head P1 可以学习 peer/shared consequence，但不能可靠地在逐任务层面同时保持 own predictor。该结论只否定当前结构与训练配方，不证明 own 与 team prediction 在任务本质上不可兼得。

因此旧 R4 不再继续做第四次软隔离修补，而是拆为：

1. **新 R4：** 不训练的 hybrid checkpoint 诊断，验证硬保护路径能否直接复用现有 team 能力；
2. **R5：** 从共同 protected-own parent 正式训练 Protected Role-MoT，建立可以晋级的结构与公平对照。

#### 7.5.2 Hybrid 组成与不可训练 contract

新 R4 只组合两个已经完成的 checkpoint：

- `own source`：旧 R4-P0 Local checkpoint，作为唯一 own state/view 输出来源；
- `team source`：旧 R4-P1 的 team encoder、peer decoder 与 shared decoder；旧 P1 的 local/own 输出和 `own_residual_gate` 一律丢弃。

前向依赖必须是单向的：

```text
own context/action ──> frozen P0 own tower ──> own prediction
                              |
                         detach K/V
                              v
all-agent context/action + global slot ──> old P1 team tower ──> peer/shared prediction
```

实现必须满足：

1. 全模型 `eval()`，不创建 optimizer/scheduler，不执行 backward，不更新 buffer，不重新拟合 PCA/normalization；
2. own prediction 直接返回 P0 输出，禁止经过旧 P1 own head、team residual 或可学习 gate；
3. team tower 只能读取 `detach()` 后的 P0 feature；shape/schema 不兼容时 fail closed，禁止静默退回旧 P1 local feature；
4. hybrid checkpoint 只保存组合 manifest 或轻量引用，不复制和改写两个 source checkpoint；必须记录两者 SHA256、model kind、PCA/manifest hash 与代码提交；
5. 固定 validation episode/window、normalization、persistence baseline、shuffle 配对和 bootstrap seed，完全复用旧 R4 验收协议。

#### 7.5.3 R4 诊断规则

R4 同时报告四项结果，但不产生正式 winner：

1. **protected-own 等价：** 固定窗口上 hybrid own state/view 张量与 P0 逐元素一致，`max_abs_diff == 0`，逐任务 own loss 也必须完全相等；
2. **team capability：** peer/shared loss 在每个任务上优于 persistence/context-only baseline；
3. **cross-agent causality：** own action 保持不变、只 shuffle peer action 后，peer/shared composite loss 增大，且每任务 episode-level paired bootstrap 95% 下界大于 0；
4. **off-path 安全：** predictor disabled 时 F1 action-equivalence 仍为逐元素一致，Flow/DINO/source checkpoint hash 不变。

诊断解释固定如下：

| R4 结果 | 结论 | R5 动作 |
|---|---|---|
| own 精确等价，team/shuffle 全通过 | 现有 team tower 与 protected P0 表示兼容，硬隔离足以保留两类能力 | 仍进入 R5；hybrid 只证明可行性，不作为正式训练候选 |
| own 精确等价，team 或 shuffle 失败 | own 保护已解决，但旧 team tower 依赖其原训练轨迹中的 local 表示 | R5 从 protected P0 重新训练 team 模块，不能复用旧 P1 team 权重作为正式结果 |
| own 不精确等价 | hybrid 接线或状态管理错误 | 停止 R5，先修复加载、dropout/buffer 或输出旁路 |

`outputs/s2_r4_hybrid/<run-id>/hybrid_diagnostic.json` 必须包含逐任务结果、source hash、exact-equivalence 最大差值和最终诊断；monitor 需要显示当前程序、当前 task/window、心跳与 age、已完成比例、own max-abs-diff、peer/shared loss、persistence、peer-action-shuffle delta/CI，以及上述三种结论之一。R4 不需要占用两张 GPU，也不得因为服务器有两张卡就启动任何训练。

#### 7.5.4 新 R4 远程零训练诊断结果（2026-08-01）

新 R4 已从本地分支 `s2/r4-hybrid-diagnostic` 提交 `30c1729` 实现、推送后在远程服务器更新代码，并在永久 tmux `ssh_tmux` 中运行 `s2-r4-hybrid-round1`。launcher 只创建 `prepare/evaluate/monitor` 三个 `remain-on-exit` window，评估进程仅看到物理 GPU0，GPU1 全程空闲；组合 manifest 明确记录 `training_performed=false`、`optimizer_created=false`、`statistics_fitted=false`，没有训练或重新拟合统计量。

本轮 source 与固定协议如下：

- protected P0 SHA256：`c04f8ea12c5b6d8f7c04992d7dd4a8c0a33aa7d0058987679e6553b17e410a2f`；
- old P1 team SHA256：`8edeac6a7825a7658ca9ece24b4c894236f351072e6a181cfe40d46d15ac5f2e`；
- PCA/statistics SHA256：`a0d236540b2fbe58b2771573f0d5674ac39ff4a6a65b16e2b39691de186483b9`；
- validation selection SHA256：`5cd7d23998eaba7535b7242706591a273f672b572475bef3be8565dae115285d`；
- 每个 episode 固定 4 个窗口，paired bootstrap `10,000` 次，seed `40404`；
- 完整结果：`/workspace/fe-pc-wam/outputs/s2_r4_hybrid/s2-r4-hybrid-round1/hybrid_diagnostic.json`。

| 任务 | P0/hybrid own loss | own max-abs-diff | hybrid peer/shared | persistence | peer shuffle Δ | bootstrap 95% lower | 单任务结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| CameraAlignment | `0.679914 / 0.679914` | `0` | `1.471217` | `2.123150` | `+0.009684` | `+0.007828` | 通过 |
| LiftBarrier | `0.931190 / 0.931190` | `0` | `1.757456` | `2.279464` | `+0.001628` | `-0.002375` | **失败：CI 跨零** |
| LongPipelineDelivery | `0.497391 / 0.497391` | `0` | `1.237946` | `1.485248` | `+0.125985` | `+0.120350` | 通过 |
| TakePhoto | `0.674406 / 0.674406` | `0` | `1.455406` | `1.893244` | `+0.054611` | `+0.051215` | 通过 |
| ThreeRobotsStackCube | `0.552356 / 0.552356` | `0` | `1.222832` | `2.055176` | `+0.057249` | `+0.049337` | 通过 |

额外不变量全部通过：五任务 own state/view 与 loss 均逐元素精确相等；predictor-disabled F1 action output 的 `maximum_absolute_difference=0`；Flow/DINO/P0/P1 文件 SHA256 在评估前后不变；hybrid manifest 与 predictor checkpoint 都不包含 Flow/DINO 参数。monitor 最终显示 `75/75 (100%)`、`status=complete`、独立 heartbeat、当前程序/任务、两卡利用率与完整特殊 gate。

**正式结论：新 R4 FAIL，诊断为 `fail_old_team_incompatible_with_protected_own`。** 失败不是 own 硬保护、team 绝对预测能力或 source 安全问题：它只来自 LiftBarrier 的 peer-action-shuffle episode-bootstrap 95% 下界 `-0.002375 < 0`。这说明把旧 P1 team tower 事后接到 protected P0 projections 后，LiftBarrier 的跨机器人动作因果依赖不再稳定；不能用正的均值 `+0.001628` 或优于 persistence 掩盖该特殊门槛失败。按预注册路线进入 R5，从同一个 protected P0 parent 重新训练 team modules，旧 P1 team 权重不作为 R5 初始化或正式结果。

### 7.6 R5：Protected Role-MoT team predictor（必做，两卡）

#### 7.6.1 为什么不是再加一个 multi-head 或普通 MoE

旧 P1 已经有独立 own/peer/shared 输出头；失败说明只分 decoder 不能隔离前面的表示和优化轨迹。普通 top-k MoE 依赖学习路由，仍可能让 own 与 team 共享同一可训练专家，因此也不能提供严格 no-regression。

R5 采用 role-level hard routing：借鉴 [Mixture-of-Transformers](https://arxiv.org/abs/2411.04996) 将 Attention、FFN 与 LayerNorm 按语义角色拆开，同时借鉴 [TwinBrainVLA](https://arxiv.org/abs/2601.14133) 的冻结通用分支/可训练专用分支和单向信息读取。这里按 `own/peer/shared` 角色分专家，而不是按五个任务或固定机器人编号分专家；第一版不使用 learned top-k router、load-balancing loss 或大规模稀疏专家。

#### 7.6.2 Protected Role-MoT 结构 contract

```text
                                      ┌─> exact own state/view
P0 own checkpoint ──> Protected Own ──┤
        frozen/eval                    └─> detached own tokens (K/V only)
                                                   |
all valid agent context/action ──> shared agent encoder ──> team tokens ─┬─> Peer Role Transformer ──> peer future
global/shared slot ──────────────> shared-slot encoder ───> shared token └─> Shared Role Transformer ──> shared future
```

硬约束如下：

1. **Protected Own Tower：** 两个候选都加载同一个旧 R4-P0 checkpoint；参数与 buffer 冻结、固定 `eval()`、不进入 optimizer/EMA/gradient clipping，own 输出直接旁路返回；
2. **单向读取：** Peer/Shared query 可以 cross-attend 到 detached own tokens，但 own tower 不能读取任何 team token，team loss 不允许反向进入 own tower；
3. **Role-MoT：** peer 与 shared 分别拥有私有 Attention、FFN、LayerNorm 和 decoder；可以共享输入投影与 agent encoder，但共享部分只服务 team 路径；
4. **agent 等变性：** 所有实体 agent slot 共享 encoder 参数，通过 `self/peer/shared` role embedding、valid-agent mask 与 pairwise query 保留身份，禁止为 `agent_0..3` 各复制一套网络；
5. **优化隔离：** team optimizer、gradient clipping、dropout RNG、checkpoint state 与 heartbeat 独立；protected own forward 使用确定性路径；
6. **禁止 own residual：** R5 不包含 team-to-own residual。以后若研究 team 信息改善 own，必须另开可回退轮次，不能修改本阶段 parent；
7. **checkpoint 身份：** 显式记录 `protected_own_sha256`、`protected_own_exact=true`、`team_mixer`、role blocks、trainable parameter names、PCA/manifest hash 和 team-training budget。

#### 7.6.3 两卡公平候选与训练范围

先把旧 R4 中已经验证的 grouped/team 数据、模型加载、训练/验收和运行基础设施以公共提交落回 `feat/model-improvements`，不把 P0/P1 candidate identity 或 checkpoint 写入 Git。新 R4 从该公共提交创建单独诊断分支；R4 结论写回公共文档后，再从同一个 `feat/model-improvements` 提交创建 R5 两个正式分支。两分支都通过显式 checkpoint path/hash 加载同一个 protected P0 parent，禁止从彼此分支创建：

| 候选 | protected own | 可训练 team mixer | 唯一变量 |
|---|---|---|---|
| R5-P0 Protected Shared | 同一 P0，冻结且直接输出 | peer/shared 共用一个 team Transformer，decoder 分开 | `team_mixer=shared` |
| R5-P1 Protected Role-MoT | 同一 P0，冻结且直接输出 | peer/shared 私有 Attention/FFN/LN，单向读取 own K/V | `team_mixer=role_mot` |

P0/P1 使用相同数据、固定 split、team 输入输出 contract、active depth/width、每样本激活 FLOPs、seed、updates、optimizer、batch size、normalization、validation windows 和 bootstrap。Role-MoT 因私有 block 复制允许拥有更多静态参数，但每个 role token 只走一个 hard-routed block；必须同时报告总参数、激活参数和实测吞吐，不能把它表述为严格 parameter-matched。只训练 team modules；两者的 protected own 前向和输出必须完全相同。R4 hybrid 的结果用于判断旧 team 表示是否可复用和定位风险，但旧 P1 team 权重不作为正式 R5 winner；正式 R5 必须从共同 parent 按上述配对协议训练。

训练、验证、checkpoint loader 和验收白名单必须加入并只接受以下新 model kind，未知值继续 fail closed：

- `s2_r4_protected_hybrid_diagnostic`，仅 evaluate，trainer 必须明确拒绝；
- `s2_r5_protected_shared_team`；
- `s2_r5_protected_role_mot_team`。

#### 7.6.4 R5 验收与选择

每个候选先独立满足：

1. protected own checkpoint hash 不变，固定窗口 own 输出逐元素等于 P0，`max_abs_diff == 0`；
2. 每个任务 peer/shared loss 优于 persistence/context-only baseline；
3. 每个任务 peer-action-shuffle delta 与 bootstrap 95% 下界大于 0；
4. predictor disabled 时 F1 action-equivalence 与 Flow/DINO 冻结检查通过。

选择规则：只有一个候选通过全部门槛时选择该候选；两个都通过时选择五任务 macro peer/shared held-out loss 更低者，差值相等时选择结构更简单的 R5-P0；两个都失败则停止进入 S3，优先检查 team target、agent 对齐和表示容量，不允许用 own 精确等价掩盖 team capability 失败。任何一个任务未通过都不能用 macro 平均抵消。

R5 monitor 除训练 loss、update/total、GPU/PID、当前程序和心跳外，必须把 `protected own exact` 单列为结构不变量，并按旧 R4 的特殊规则显示每任务 persistence、peer-action-shuffle delta/CI 与最终 R5 winner。数据、Hugging Face 下载、共享 cache、DINO/PCA/Flow/P0 artifact 和永久 tmux 规则继续完整复用 S0、7.4.3 与旧 R4 最近一版脚本；不得改回 `snapshot_download`，不得删除 `.cache/huggingface/`、`.incomplete` 或已完成 HDF5。

#### 7.6.5 一键部署、monitor 与退出脚本约束

R4/R5 实现时继续沿用 7.4.3 的分层：外层一键 Bash 只做依赖、唯一永久 tmux、GPU 数量、仓库 origin、磁盘和 dry-run 检查；仓库/worktree 检测、缺失项补齐、source checkpoint 定位、共享数据/artifact 链接、断点恢复、窗口修复和最终验收全部放入版本化 `.sh`。所有新 Bash/`.sh` 都禁止使用 `set -euo pipefail`，错误必须显式打印到当前终端和对应日志后再返回非零状态。

计划脚本固定为：

```text
scripts/launch_s2_r4_hybrid_tmux.sh
scripts/stop_s2_r4_hybrid_tmux.sh
scripts/launch_s2_r5_existing_server.sh
scripts/launch_s2_r5_2gpu_tmux.sh
scripts/stop_s2_r5_2gpu_tmux.sh
scripts/s2_r4_hybrid_runtime.py
scripts/s2_r5_runtime.py
```

R4 在当前唯一永久 session 中创建 `prepare/evaluate/monitor` 窗口；R5 创建 `prepare/p0/p1/monitor` 窗口，P0/P1 分别固定 GPU0/GPU1，全部 `remain-on-exit=on`。已有服务器必须先自动识别仓库、旧 R4 source、数据、Hub cache、DINO/PCA/Flow、有效 run/checkpoint/resume 和 monitor，只补齐缺失部分；数据与不可变 artifact 在 worktree 间只读共享，checkpoint/output/log 保持 run 隔离。停止脚本只能停止本 run 并关闭本轮窗口，禁止 `tmux kill-session`，不得删除数据、Hub cache、source checkpoint、resume、日志或验收 JSON。

Hugging Face 下载继续保持 S0 约定：token 只经 mode-0600 FIFO 进入 prepare，不写 export/argv/tmux/log；dataset 使用固定 revision、官方 `hf download`、Xet 开启与默认并发，DINOv3 使用 Xet 关闭和单 worker；下载中断复用原位 cache 与 `.incomplete`。R4/R5 monitor 的心跳超时继续使用 75 秒，`STALE` 必须同时提示当前程序、最后心跳 age、日志路径和 GPU PID，不能把最后一个 loss 当作仍在运行。

R4 不重新拟合视觉子空间：`artifacts/s2_r4/dino_pca_statistics.pt` 必须复用 R3 train-only artifact 的不可变 DINO `1024→256` PCA、local state/view 统计和五任务 manifest identity，只在同一批 train-only 固定窗口上新增 global shared-view delta 的独立 mean/std。P0/P1 共同记录这个扩展 artifact 的 hash；P1 的 shared target 与 persistence baseline 都使用 shared-view 统计，禁止把 local camera 分布的 mean/std 套到 global slot，也禁止用 validation 数据拟合归一化。

基础仓库只保存一份约 784 GiB 的五任务数据和一份 `artifacts/`；P1 worktree 通过只读语义的符号链接共享 `datasets/` 与 `artifacts/`，候选 checkpoint/output/log 则全部写入 run 隔离目录。P0 固定 GPU0、P1 固定 GPU1。prepare 需要 GPU0 时先恢复缺失的 S1-R1 F1 Flow、PCA/statistics 或 R3-W1 parent，两候选在此期间每 20 秒持续等待心跳；共享 ready 后才同时占用两卡训练。

#### 7.6.6 R5 实现、分支身份与两卡一键运行

R5 公共基础设施已经先在本地落到 `feat/model-improvements` 提交 `f2b8da1` 并推送。该提交包含 protected-own 模型、team-only trainer、固定窗口 evaluator、特殊验收器、fail-closed model-kind 白名单、配对配置校验、共享准备、两卡 launcher、常驻 monitor 和保留产物的 stop 脚本，但不包含 P0/P1 候选配置；随后提交 `22dd49e` 补齐受限远程 ref 的显式抓取，`1944058` 使已完成任务的 heartbeat 稳定显示 `finished` 而不是误报 `STALE`。两个正式分支都直接从同一个 `f2b8da1` 创建，不从彼此创建：

| 分支 | 提交 | model kind | 唯一变量 | GPU |
|---|---|---|---|---:|
| `s2/r5-p0-protected-shared` | `f551ceb` | `s2_r5_protected_shared_team` | `team_mixer=shared` | 0 |
| `s2/r5-p1-protected-role-mot` | `094613d` | `s2_r5_protected_role_mot_team` | `team_mixer=role_mot` | 1 |

配对校验器会在创建任何 GPU 任务前，拒绝 data/split、seed、10,000 updates、optimizer、batch size、normalization、validation windows、bootstrap 或 protected P0 路径的任何漂移。两个候选每个样本都执行两次同形状 role mixer：P0 两次复用同一 Transformer 参数，P1 分别硬路由到 peer/shared 私有 Transformer，因此 active depth/width 和 mixer 调用数一致；checkpoint 同时报告总参数、每角色激活参数和实测 updates/s，不宣称静态参数严格匹配。protected tower 永久 `eval()`、不在 optimizer/gradient clipping/checkpoint team state 中，team loss 不会回传到 P0，且没有 team-to-own residual。

已有服务器的一键更新、检查和启动如下。launcher 会自动找到最新有效 R4-P0，复用唯一永久 tmux、共享五任务数据和 `artifacts/`，创建或修复 `<run-id>-prepare/p0/p1/monitor`，并对已有 checkpoint/resume/evaluation 只补齐缺失步骤：

```bash
cd /workspace/fe-pc-wam
git fetch --no-tags origin \
  +refs/heads/feat/model-improvements:refs/remotes/origin/feat/model-improvements \
  +refs/heads/s2/r5-p0-protected-shared:refs/remotes/origin/s2/r5-p0-protected-shared \
  +refs/heads/s2/r5-p1-protected-role-mot:refs/remotes/origin/s2/r5-p1-protected-role-mot
git switch feat/model-improvements
git merge --ff-only origin/feat/model-improvements

bash scripts/launch_s2_r5_2gpu_tmux.sh \
  --run-id s2-r5-round1 --dry-run
bash scripts/launch_s2_r5_existing_server.sh \
  --run-id s2-r5-round1 --no-focus-monitor
```

若已有服务器确实缺少 HF 数据或 DINO artifact，只在最后一条启动命令追加 `--prepare-from-s0`，然后在隐藏提示中输入 token。该路径直接调用已验证的 S0 下载链：token 只经 mode-0600 FIFO；dataset 固定 revision、使用官方 `hf download`、Xet 开启和默认并发；DINO 关闭 Xet 且单 worker；原位复用 Hub cache 与 `.incomplete`。现有 asset 完整时不请求也不传 token。

任意非本轮窗口可执行以下只读 monitor；它每 5 秒显示 shared prepare、P0/P1 当前程序和 phase、20 秒心跳及 age、update/total/loss、验证 task/batch、两卡利用率/显存和 GPU PID。75 秒无心跳标为 `STALE`，同时显示最后程序、heartbeat PID 和 candidate log。两项 evaluation 完成后，它逐候选单列 `protected own exact`，逐任务显示 peer/shared、persistence、shuffle delta/CI lower 和 PASS/FAIL，最后显示 R5 winner；不会把最后一个 loss 或单纯训练完成误报为验收通过。

```bash
cd /workspace/fe-pc-wam
python3 scripts/s2_r5_runtime.py monitor --once \
  --run-root /workspace/fe-pc-wam/outputs/s2_r5_runs/s2-r5-round1
tmux select-window -t "$(tmux display-message -p '#S'):s2-r5-round1-monitor"
```

需要中止本轮但保留所有可恢复信息时，从永久 session 的非本轮窗口执行：

```bash
cd /workspace/fe-pc-wam
bash scripts/stop_s2_r5_2gpu_tmux.sh s2-r5-round1
```

stop 只终止该 run 的进程并关闭该 run 的四个窗口；不会 `tmux kill-session`，不会删除共享数据、Hub cache、DINO/PCA/Flow/P0、checkpoint、resume、日志、evaluation 或 acceptance JSON。永久 tmux session 必须继续存在。

#### 7.6.7 远程 `s2-r5-round1` 正式结果（2026-08-01）

本轮严格按“本地修改与测试 → 推送 → 远程 fast-forward → 永久 tmux 自主运行 → 结果分析 → 文档回写”执行。远程 run root 为 `/workspace/fe-pc-wam/outputs/s2_r5_runs/s2-r5-round1`；`ssh_tmux` 中的 `prepare/p0/p1/monitor` 四个 window 全部 `remain-on-exit=on`，训练结束后 session 和结果窗口仍保留。prepare 自动复用约 784 GiB 的单份五任务数据、DINO/PCA/Flow 和旧 R4-P0，没有触发 HF 下载，也没有使用或落盘 token。run 从 `2026-07-31T17:52:02Z` 到 acceptance `2026-07-31T19:21:41Z`，P0/P1 分别固定 GPU0/GPU1；两者都完成 10,000 updates、五任务 75 个 validation batches、10,000 次 episode bootstrap 和 action-equivalence 检查。

固定比较身份完全一致：training seed `505`、batch size `1`、validation selection SHA256 `5cd7d23998eaba7535b7242706591a273f672b572475bef3be8565dae115285d`、R4 train-only PCA/statistics SHA256 `a0d236540b2fbe58b2771573f0d5674ac39ff4a6a65b16e2b39691de186483b9`、protected P0 checkpoint SHA256 `c04f8ea12c5b6d8f7c04992d7dd4a8c0a33aa7d0058987679e6553b17e410a2f`。两个 candidate 在训练终点同一 update 的 own monitor loss 完全一致，固定验证上 own state/view 逐元素精确相等，`maximum_absolute_difference=0`；protected checkpoint hash 和 model hash 前后不变、P0 不在 optimizer 中。predictor-disabled F1 action output 也逐元素相等且 `maximum_absolute_difference=0`，Flow/DINO 文件 hash 稳定，checkpoint 不含 Flow/DINO state。

| 任务 | persistence | P0 peer/shared | P0 shuffle Δ | P0 CI95 lower | P1 peer/shared | P1 shuffle Δ | P1 CI95 lower |
|---|---:|---:|---:|---:|---:|---:|---:|
| CameraAlignment | 2.123150 | 1.449888 | 0.011614 | 0.009701 | 1.452008 | 0.012267 | 0.009953 |
| LiftBarrier | 2.279464 | 1.734023 | 0.008674 | 0.005269 | 1.736676 | 0.008584 | 0.005387 |
| LongPipelineDelivery | 1.485248 | 1.223233 | 0.125159 | 0.120173 | 1.229281 | 0.142079 | 0.136959 |
| TakePhoto | 1.893244 | 1.432410 | 0.069215 | 0.065682 | 1.435911 | 0.074079 | 0.069543 |
| ThreeRobotsStackCube | 2.055176 | 1.191338 | 0.063531 | 0.055491 | 1.208192 | 0.065140 | 0.057289 |

两候选在五个任务上均同时满足 `peer/shared < persistence`、shuffle mean `>0` 和 episode-bootstrap 95% lower `>0`，因此都通过独立 special gate。P0 五任务 macro peer/shared loss 为 `1.4061783383`，P1 为 `1.4124135508`；按预注册选择规则，P0 更低 `0.0062352125`，选择 `s2_r5_protected_shared_team` 进入 S3。该选择不是依赖训练 loss，也没有用 macro 掩盖单任务失败；P0 在五个单任务的绝对 peer/shared loss 上也都略低于 P1。

| 候选 | 总参数 | protected 参数 | trainable team 参数 | active peer/shared 参数 | 实测 updates/s | macro peer/shared | R5 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| P0 Protected Shared | 10,621,784 | 6,173,254 | 4,448,530 | 4,201,362 / 4,045,824 | 3.0776 | 1.406178 | **PASS / winner** |
| P1 Protected Role-MoT | 14,170,712 | 6,173,254 | 7,997,458 | 4,201,362 / 4,045,824 | 3.0981 | 1.412414 | PASS / not selected |

Role-MoT 增加约 3.55M 静态参数，但每角色激活参数、active depth/width 与每样本两次 mixer 调用和 P0 相同；本轮吞吐没有实质劣化。P1 在四个任务上得到略大的 shuffle delta/CI，但没有转化成更低的 held-out predictive loss。因而本结果支持“在当前五任务、seed `505` 和 10k team-only 预算下，共享 team Transformer 已足够，额外 role 私有化不值得作为正式 parent”，不外推为 Role-MoT 在多 seed、更大数据或进入闭环后的普遍劣势。

正式产物与哈希：P0 checkpoint `fcc0af76c2acd6805750f12e828a1249eb91e466e51f4aa77c118b6e9d330c67`、P0 evaluation `8a636942f7d96a9cb0365bad36555a51f471fc67fe2ea9d51412ecf1df8fd8a0`；P1 checkpoint `58f2997c6625a6421a07d8805054a66c75101b897fd15640080622dbe42ffc78`、P1 evaluation `7989444828dbcad2b0eb59ac70f964d25b351a895cead9c36993fd5828632cf1`；最终 `acceptance.json` SHA256 为 `2c7778ecfe7f0b53ff2ffb29ceebe0f62313850ff3dea54427f6b517049289e0`，结论为 `pass_select_p0_enter_s3`。monitor 最终显示两个 candidate `complete/finished`、`own-exact=yes`、`peer-CI+=5/5`、两套逐任务特殊 gate 和 `PASS -> select P0, enter S3`，GPU 进程为 none；永久 `ssh_tmux` 未退出。

**正式结论：S2-R5 PASS，选择 R5-P0 Protected Shared 作为 S3 的 team parent。** 新 R4 暴露的 LiftBarrier 因果不稳定已经被“从 protected P0 重新训练 team modules”修复：LiftBarrier 的 P0 shuffle mean/CI lower 从 hybrid 的 `0.001628/-0.002375` 提升到 `0.008674/0.005269`，同时保持 own 严格不变。S2 的结论仍只证明 off-path future prediction 与跨机器人动作因果依赖，不声称闭环动作收益；下一阶段必须用 gate 初始为零、可关闭且可回退的 world-to-Flow residual 在闭环中验证价值。

工程晋级已完成：正式 winner 分支 `s2/r5-p0-protected-shared` 通过 merge commit `b59cc9e` 合并回 `feat/model-improvements`，保留其独立分支历史；P0 的 config、candidate env 与 candidate card 现已成为模型改进主线的一部分。S3 必须从该主线创建分支并固定本节记录的 P0 checkpoint/hash，不允许从未入选的 P1 分支继续派生。

### 7.7 S2 产物与进入 S3 的硬门槛

S2 必须产出 R3-W1、旧 R4-P0 protected-own、R4 hybrid 诊断和 R5 protected team predictor，对应配置、固定 validation split、normal/action-shuffle/peer-action-shuffle episode-level JSON、target normalization/PCA artifact 及其 hash。R3 与 R5 的全部门槛通过，且 protected own 精确等价成立后，才能把 protected-own/R5 team predictor 作为 S3 的 local/team parents。R4 hybrid 是诊断，不是可晋级 checkpoint。

以下任一情况直接判 S2 无效：predictor disabled 后动作不再与 F1 等价；Flow/DINO/protected-own 任一参数或 buffer 改变；future target 泄漏进输入；action shuffle 不增大 local error；peer-action shuffle 不增大 peer/shared error；R5 own 输出不能逐元素复现 P0。S2 不声称闭环提升，也不因 off-path 闭环持平而晋升模型；S3 才检验预测未来是否改善动作。

## 8. S3：让受保护的联合未来真正调制 Flow（08-11 至 08-21）

本阶段固定数据、Flow、world target、future representation、R5 protected-own 与 R5 team predictor，先只增加一个可关闭的 world-to-flow 接口。S3 可以改变动作生成，但永远不能解冻或旁路 protected own predictor。注入必须是基础 Flow 的受控残差，而不是替换原有动作路径：

$$
\mathbf v_{\mathrm{new}}^i
=
\mathbf v_{\mathrm{base}}^i
+
g^i
\Delta \mathbf v^i
\left(
\mathbf h_t^i,
\hat{\mathbf z}_{\mathrm{own}},
\hat{\mathbf z}_{\mathrm{peer}},
\hat{\mathbf z}_{\mathrm{shared}}
\right),
\qquad
g_{\mathrm{init}}=0.
$$

实现时使用有界 gate，例如 $g=g_{\max}\tanh(\alpha)$ 且 $\alpha_{\mathrm{init}}=0$；future 无效或全部被 mask 时强制 $g=0$。这里的 gate 只控制 **world-to-Flow velocity residual**，与旧 R4 已废弃的 `own_residual_gate` 不是同一个参数。`gate=0` 时必须退化为冻结的 S1 Flow。第一版不允许用直接 cross-attention 覆盖所有 action layers，不做 proposal scoring 或 energy guidance；这些高跨度方案移到 ICRA 后。

### 8.1 R6L/R6J：只增加 gated residual injection（四卡并行）

使用 S2 冻结的 protected-own parent 与 R5 team parent，各启动一个两卡微轮次：

| 微轮次 | P0 控制 | P1 单步改进 | 固定范围 |
|---|---|---|---|
| R6L Protected Local | protected own predictor off-path，`injection=off` | 只用 own future 加入 residual adapter，gate 初始化为 0 | Flow、protected own 与 team predictor 均冻结 |
| R6J Protected Team | R5 predictor off-path，`injection=off` | 用 own + peer + shared future 加入同构 residual adapter，gate 初始化为 0 | Flow、protected own 与 team predictor 均冻结 |

P1 只训练 adapter 与 velocity gate。两组使用相同 adapter 宽度、初始化、优化器、训练更新、solver 和闭环协议，因此 `R6J-P1 vs R6L-P1` 只反映 future scope，`P1 vs P0` 只反映 injection。R6J 中的 own latent 必须来自 protected P0 路径，peer/shared latent 来自 R5 team tower；不得恢复旧 P1 own head 或 team-to-own residual。

#### 8.1.1 四分支实现、双卡两两排程与白名单（2026-08-01）

S3-R6 公共基础设施已先在本地写入 `feat/model-improvements` 提交 `50d64bd`、完成相关回归测试并推送。公共提交包含 `CrossAgentWorldConditionedFlow`、同构 local/team residual adapter、有界 `max_gate*tanh(alpha)` velocity gate、adapter/gate-only trainer、闭环 inference、四分支矩阵校验器、S3 特殊验收器、常驻 monitor、S0 下载复用、双卡两两 launcher 和保留产物的 stop 脚本；不包含候选身份配置。随后四个分支全部直接从同一个 `50d64bd` 创建，不从彼此派生：

| 执行批次 | GPU | 分支 | 候选身份提交 / 当前 head | model kind | 训练 |
|---|---:|---|---|---|---|
| 1 | 0 | `s3/r6l-p0-protected-local-aux` | `b61ee77` / `691ea94` | `s3_r6l_protected_local_aux` | 0 update，off-path 控制 |
| 1 | 1 | `s3/r6l-p1-protected-local-gated` | `1479aa3` / `76d36ab` | `s3_r6l_protected_local_gated` | 10,000 update，仅 adapter/gate |
| 2 | 0 | `s3/r6j-p0-protected-team-offpath` | `21e36fa` / `964fe3a` | `s3_r6j_protected_team_offpath` | 0 update，off-path 控制 |
| 2 | 1 | `s3/r6j-p1-protected-team-gated` | `84db555` / `05a5cdc` | `s3_r6j_protected_team_gated` | 10,000 update，仅 adapter/gate |

训练、checkpoint loader、闭环服务端和验收器的 fail-closed 白名单只增加上表四个 kind；未知 kind、kind 与 `micro_round/candidate_id/future_scope/injection` 不一致、R6J 不是 accepted R5-P0 Shared team parent、protected-own/R5-P0 hash 漂移时均在创建有效结果前失败。四个 config 使用相同五任务数据、Flow、protected-own、R5-P0、PCA、adapter shape、adapter seed `60606`、训练 seed `606`、optimizer、P1 10,000 updates、4-step Euler、Gate20 seed `900` 与 temporal ensemble；矩阵校验器只允许 R6L/R6J future scope、P0/P1 injection 和隔离输出路径不同。

实现按每次 velocity evaluation 计算 `clean_action = x_tau + (1-tau)*v_base`，以 stop-gradient clean action 调用冻结 future predictor；Euler 的每一步都重新预测，Heun 若以后启用则 predictor/corrector 两次 evaluation 都重新预测。`injection=false` 完全不执行 future predictor；`injection=true` 时只 adapter 与 gate 进入 optimizer、gradient clipping、resume 和 S3 checkpoint，Flow、DINO、protected-own 与 team predictor 参数不写入 S3 trainable state。gate 精确为零时动作与 base Flow 逐元素相等会记录为诊断，但按 8.2 不被错误提升为额外候选门槛。

双卡 launcher 会在永久 tmux 中创建 `<run-id>-prepare`、四个 candidate 和 `<run-id>-monitor` 六个 `remain-on-exit` window。R6L-P0/P1 先分别占用 GPU0/GPU1；R6J 两个 window 在不占 GPU 的 queued 状态持续报告 20 秒心跳，拿到完整 R6L pair acceptance 后才自动分别使用 GPU0/GPU1。数据集、Hub cache、DINO/PCA/Flow/R4-P0/R5-P0 只在基础仓库保存一份并由四个 worktree 共享，checkpoint、resume、日志、闭环视频和结果按 candidate 隔离。

已有双 5090 服务器从零检查、更新和一键启动如下；正式 launcher 会自动发现现有约 784 GiB 五任务数据及最新 accepted R4-P0/R5-P0，只补齐缺失 worktree、parent link、run、resume、window 或 monitor：

```bash
cd /workspace/fe-pc-wam
git fetch --no-tags origin \
  +refs/heads/feat/model-improvements:refs/remotes/origin/feat/model-improvements
git switch feat/model-improvements
git merge --ff-only origin/feat/model-improvements

bash scripts/launch_s3_r6_2gpu_tmux.sh \
  --run-id s3-r6-round1 --dry-run
bash scripts/launch_s3_r6_existing_server.sh \
  --run-id s3-r6-round1 --no-focus-monitor
```

`launch_s3_r6_existing_server.sh` 会同时检查五任务数据、DINO 与 `/workspace/RoboFactory` 的 Python/scene asset；RoboFactory 缺失时自动追加 `--prepare-from-s0` 并在当前终端做一次隐藏 token 提示，手动调用底层 launcher 时也可显式追加该参数。提交 `ea93741` 将 RoboFactory 纳入 shared-ready 条件，并让同一次隐藏输入在进程内依次经两个 mode-0600 FIFO 复用 S0 环境准备和必要的五任务/PCA 补齐；token 不进入 export、argv、tmux command、manifest、普通文件或日志。dataset 仍使用固定 revision、官方 `hf download`、Xet 开启与默认并发，DINOv3/RoboFactory asset 使用 Xet 关闭和单 worker，中断后原位复用 Hub cache 与 `.incomplete`；已有完整五任务/PCA 时不重算也不复制。accepted S2 parent checkpoint 不是 HF 数据，缺失时必须显式提供 `--protected-own PATH --protected-team PATH`，不能静默重训或换 parent。

monitor 每 5 秒显示 shared prepare 与四个 candidate 的当前程序、queued/waiting/startup/training/validating/accepting/complete 状态、20 秒心跳及 age、update/10,000、loss、gate、当前闭环 task/episode/success、两卡利用率/显存和 GPU process PID。提交 `8dd88e0` 进一步让 rollout 从环境初始化、等待 inference、连接到每 25 step 都原子更新 `task/episode/step/success/stage`，因此第 0 个 episode 也不会回退成旧训练阶段；四分支已同步到表中的 current head。75 秒没有新心跳标记 `STALE`，同时显示最后程序和 candidate log；最后一个 loss 绝不被当作仍在运行。R6L/R6J 结果产生后，monitor 只按本阶段特殊规则逐任务显示 `P0 success、P1 success、delta、P1>=P0 PASS/FAIL`，并单列 protected-own 结构不变量；zero/noise/shuffle/fallback 和 gate-zero 诊断不会变成额外准入 gate。只读查看：

```bash
cd /workspace/fe-pc-wam
python3 scripts/s3_r6_runtime.py monitor --once \
  --run-root /workspace/fe-pc-wam/outputs/s3_r6_runs/s3-r6-round1
tmux select-window -t "$(tmux display-message -p '#S'):s3-r6-round1-monitor"
```

需要停止本轮时只能从永久 session 的非本轮窗口执行：

```bash
cd /workspace/fe-pc-wam
bash scripts/stop_s3_r6_2gpu_tmux.sh s3-r6-round1
```

stop 只终止本 run 的进程并关闭上述六个 window；禁止 `tmux kill-session`，不会删除共享数据、Hub cache、父 checkpoint、candidate checkpoint/resume、日志、视频、Gate summary 或 acceptance JSON。永久 tmux session 始终保留。

#### 8.1.2 正式远程结果（运行后回写）

正式 run、四分支训练/闭环成功数、特殊验收结论、checkpoint/acceptance hash 和失败分析在远程两批实验完成后回写本节；在完整结果产生前不得把 training loss、gate 非零或单任务改善表述为 R6 通过。

每个 solver step 必须重新执行：

1. 用冻结 base Flow 从当前 $\mathbf x_\tau^{1:N}$ 计算 base velocity 与 provisional clean action $\hat{\mathbf a}_1^{1:N}$；
2. 按 S2 的 candidate-action contract，用 stop-gradient 的 $\hat{\mathbf a}_1^{1:N}$、$\tau$ 与上下文预测 future latent；
3. 计算 gated residual correction；
4. 更新 $\mathbf x_\tau$。

不能直接用 raw $\mathbf x_\tau$ 代替 clean action contract，也不能缓存一个与 $\mathbf x_\tau$ 无关的 future summary，却声称 world model 正在评估候选动作。

### 8.2 闭环保持规则

R6L/R6J 的 P1 分别与对应 P0 比较。只要 P1 在每个任务的闭环成功率都不低于 P0，就可以继续，持平也算通过。`gate=0` 等价性、zero/noise、mask、fallback 和数值诊断不再作为额外准入门槛；protected own hash/输出等价属于模型加载不变量，不是可以被闭环持平豁免的候选指标。

### 8.3 实现说明

真实未来只用于训练 target，部署动作路径使用模型预测的 future latent。zero/shuffle intervention 可以作为论文分析，但不决定候选能否继续。R6J-P1 只要相对 R6J-P0 没有闭环成功率退步，就可以进入 R7；与 R6L-P1 的比较只用于结果说明。

### 8.4 R7a/R7b：逐模块解冻（可选，四卡并行）

R6J-P1 通过后，将其冻结为 `P_inject`，再运行两个独立微轮次。protected own tower 在两个微轮次中都保持冻结，旧版“解冻整个 world predictor”被取消：

| 微轮次 | P0 控制 | P1 单步改进 | 唯一变量 |
|---|---|---|---|
| R7a Team adaptation | R5 team tower 冻结 | 仅以小学习率解冻 peer/shared Role-MoT team modules | team gradient scope |
| R7b Flow adaptation | Flow 冻结 | 仅以小学习率解冻 Flow | Flow gradient scope |

两轮都保留同一 gated residual，protected own 始终不进入 optimizer、EMA 或 gradient clipping。每个 P1 只要相对 P0 的各任务闭环成功率没有退步即可保留；若两个都通过，可以直接进行组合闭环，组合没有成功率退步即可继续。

### 8.5 R8：Future dropout（可选，两卡）

只有 R6/R7 已冻结且仍有时间时，比较 `future_dropout=off` 与 `future_dropout=on`。dropout 只作用于送往 Flow adapter 的预测 future，不作用于 protected own tower；P1 闭环成功率不低于 P0 即可保留，R8 不阻塞主路径。

## 9. S4：正式训练、评测与统计（08-22 至 08-31）

### 9.1 四卡并行方式

冻结模型后，四张卡不再训练新结构，而是并行训练同一正式方案的四个随机种子：

| 卡 | 训练随机种子 | 作用 |
|---|---:|---|
| E1 | 101 | 正式复现 1 |
| E2 | 202 | 正式复现 2 |
| E3 | 303 | 正式复现 3 |
| E4 | 404 | 正式复现 4 |

S4 使用 S3 中最近一个闭环成功率没有退步的方案。存在组合分支时先跑一次组合闭环；不退步就使用组合，否则使用其 P0。随后四张卡用于该方案的四个正式种子。

### 9.2 主表

1. 当前分支最佳 legacy per-agent chunk baseline；
2. R1/R2 冻结的 Per-Agent Flow；
3. Joint/team-context Flow without world prediction，隔离“多机器人联合建模”本身；
4. R5 winner：Protected own + Team/Role-MoT world prediction，不注入 velocity；
5. R6L-P1：Protected local-future gated residual injection，隔离单机器人 latent WAM；
6. R6J-P1 或 R7/R8 verified winner：Protected Team+Shared World-Conditioned Action Flow；
7. centralized joint policy，作为信息上限而不是最终方法。

### 9.3 核心消融

- dense vs top-2 MoE；
- local future vs joint/peer-conditioned future；
- shared team Transformer vs peer/shared Role-MoT；
- post-hoc R4 hybrid vs 从共同 protected parent 正式训练的 R5；
- joint/team-context Flow without world vs cross-agent world-conditioned Flow；
- auxiliary-only vs world-to-flow coupling；
- zero-init gate 的 residual injection：`gate=0` 等价性；
- frozen base vs 仅解冻 team Role-MoT vs 仅解冻 Flow；protected own 不参与解冻消融；
- normal vs zero vs shuffled predicted future；
- own action/future 不变时，normal vs zero/shuffled peer action 和 peer future；
- temporal ensemble on/off；
- 1-step Euler、4-step Euler、2-step Heun。

active-agent loss weighting 不进入主表和消融表。

上述主表和消融按时间选择执行，不阻塞阶段推进。

### 9.4 唯一评测指标：闭环成功率

至少覆盖两类协作关系，例如同步搬运与顺序交接。每个任务只记录：

- 成功 episode 数、总 episode 数和闭环成功率；
- paired initial conditions 下的逐回合结果。

推进时直接比较同任务成功率。候选相对父方案在所有任务都没有下降即可通过；Wilson 区间、paired test、多随机种子和其他统计均为可选报告项，不构成准入条件。

## 10. 远程 GPU 多分支闭环迭代协议

### 10.1 Round 定义

每个训练微轮次固定包含 `P0=父方案复跑` 与 `P1=父方案+一个 Δ`。新 R4 是不训练、不选 winner 的单分支 checkpoint 诊断，不适用该配对约束。round 只需记录：

- round ID、P0/P1 分支和 P1 改动；
- 训练预算、闭环任务与 seeds；
- P0/P1 各任务成功率。

其他环境、hash、provenance 和审计信息按需记录，不作为闭环推进条件。

### 10.2 远程运行

1. 每个微轮次从同一个父提交创建 P0/P1 两个本地 worktree/分支；并行两个微轮次时共四个分支；
2. P0/P1 尽量使用相同训练预算与闭环协议；
3. 回传 checkpoint 和成功率结果即可，其他运行信息不阻塞选择。

### 10.3 On-path 候选只需闭环；S2 使用 capability gate

从 S3 起，候选完成训练后跑与父方案相同的闭环任务并输出成功率。主动早停或没有闭环结果的候选退出本轮，不阻塞其他候选；不再要求额外 smoke、reload、provenance 或 artifact 审计才能进入选择。S2 predictor 严格 off-path，是此规则的唯一例外：R3 按 7.4 的 local capability gate 选择，R4 按 7.5 只做 hybrid 诊断，R5 按 7.6 的 protected-own/team capability gate 选择；三者不进行没有区分力的成对闭环选型。

### 10.4 选择一个或多个 winner

唯一规则是：

$$
\forall\,\text{task},\quad
\operatorname{SuccessRate}(P1,\text{task})
\ge
\operatorname{SuccessRate}(P0,\text{task}).
$$

对于进入动作路径的候选，满足即通过，持平也通过；任一任务下降则保留 P0。无需显著性、置信区间、正式 episode 数、其他指标或额外审计。两个并行 P1 都通过时可以进入组合闭环。S2 不适用该公式，按第 7 节 capability gate 执行。

### 10.5 多分支组合不是直接 Git 合并

多个通过候选可以建立组合分支。组合分支相对其 P0 在所有任务的成功率都不下降即可继续，持平也通过；无需兼容性表、Pareto 条件或额外 artifact 审计。

### 10.6 分支与产物命名

候选命名建议：

```text
round/s0-b0-legacy-moe-ensemble
round/s0-b1-legacy-dense-ensemble
round/s0-b2-flow-reference
round/s0-b3-legacy-moe-latest
s1/r1-f0-legacy
s1/r1-f1-flow-cold
s1/r2a-p0-decoder-current
s1/r2a-p1-decoder-alternative
s1/r2b-p0-cold
s1/r2b-p1-warm
s1/r2m-verified-merge
s2/r3-w0-action-independent-local
s2/r3-w1-action-conditioned-local
s2/r4-p0-local                         # 历史旧 R4 source
s2/r4-p1-team-shared                   # 历史旧 R4 source
s2/r4-hybrid-diagnostic
s2/r5-p0-protected-shared
s2/r5-p1-protected-role-mot
s3/r6l-p0-protected-local-aux
s3/r6l-p1-protected-local-gated
s3/r6j-p0-protected-team-offpath
s3/r6j-p1-protected-team-gated
s3/r7a-p1-unfreeze-team
s3/r7b-p1-unfreeze-flow
s3/r7m-verified-merge
s3/r8-p1-future-dropout
```

每轮保留选定 parent、checkpoint、配置和成功率摘要即可；其他信息按需记录，不作为推进条件。

## 11. 代码落地顺序

当前分支保留为可运行参考，新主线不要继续堆进 legacy 类：

```text
models/wam_multimodal/
  agent_factorized_flow_wam.py
  action_conditioned_world_model.py
  protected_role_mot_world_model.py
  cross_agent_world_conditioned_flow.py

train/
  agent_factorized_flow_training.py
  grouped_future_dataset.py
  action_conditioned_world_training.py
  world_action_flow_training.py

scripts/
  train_action_conditioned_world_model.py
  evaluate_action_conditioning.py
  compose_s2_r4_hybrid_checkpoint.py
  evaluate_s2_r4_hybrid_checkpoint.py
  train_s2_r5_protected_role_mot.py

tests/
  test_s2_grouped_future_dataset.py
  test_s2_action_conditioned_world_model.py
  test_s2_r4_hybrid_checkpoint.py
  test_s2_r5_protected_role_mot.py

experiments/wam_flow/
  round_manifest.schema.yaml
  candidate_card.schema.yaml
  evidence_board.schema.yaml

configs/wam_flow/
  s1_r1_f0_legacy.yaml
  s1_r1_f1_flow_cold.yaml
  s1_r2a_decoder_current.yaml
  s1_r2a_decoder_alternative.yaml
  s1_r2b_cold.yaml
  s1_r2b_warm.yaml
  s1_r2m_verified_merge.yaml
  s2_r3_action_independent_local.yaml
  s2_r3_action_conditioned_local.yaml
  s2_r4_hybrid_diagnostic.yaml
  s2_r5_protected_shared.yaml
  s2_r5_protected_role_mot.yaml
  s3_r6l_protected_local_aux.yaml
  s3_r6l_protected_local_gated.yaml
  s3_r6j_protected_team_offpath.yaml
  s3_r6j_protected_team_gated.yaml
  s3_r7a_unfreeze_team.yaml
  s3_r7b_unfreeze_flow.yaml
  s3_r7m_unfreeze_team_flow.yaml
  s3_r8_future_dropout.yaml
```

实现顺序：

1. 抽取当前 per-agent token、DINO、decoder 和 inference contract；
2. 完成 R1 原子垂直切片：保持 rollout API 与其他路径不变，只把 action generator 替换为 cold-start Rectified Flow；
3. R1 通过后跳过 R2a，将 R2b 移入非阻塞 backlog，并冻结 `caa5ed3` 与 F1 checkpoint 作为 S2 父方案；
4. 新增 grouped trajectory adapter 与 future target builder，保留 `[B,A,...]`、global slot、future masks，并先完成 S2.0 contract tests；
5. 建立 off-path local future predictor；R3 只打开 own candidate-action adapter，以 held-out error 与 action shuffle 选择 W1；
6. 记录旧 R4 team capability 通过但 own no-regression 在 gate、gradient clipping、RNG 三项隔离后仍失败；不再继续软修补；
7. 新 R4 组合旧 P0 own 与旧 P1 team source，只做 exact-own、persistence 和 peer-action-shuffle 诊断，禁止训练和晋级；
8. R5 从共同 protected P0 parent 建立 Protected Shared/Protected Role-MoT 两卡候选；own 硬旁路，peer/shared 单向读取 detached own K/V；
9. 训练、验证、加载与验收白名单加入 R4 evaluate-only 和两个 R5 model kind；trainer 必须拒绝 hybrid kind；
10. 建立 `CrossAgentFlowWAM` residual adapter，并只将 velocity gate 初始化为 0；R6 只训练 adapter/gate，Flow 与全部 world predictor 冻结；
11. R6 通过后才允许 R7 分别解冻 team Role-MoT 或 Flow，protected own 永不解冻；future dropout 单独放在 R8；
12. checkpoint schema 显式记录 `action_generator`、`future_scope`、`protected_own_sha256`、`protected_own_exact`、`team_mixer`、`injection`、`trainable_modules`、gate、solver、target normalization/PCA 与 manifest hash；
13. 加入 peer-action/future zero/shuffle intervention 和 joint-Flow-without-world baseline；
14. legacy checkpoint 只通过 legacy loader 读取，禁止静默加载到新方法。

## 12. 时间表与论文并行

| 日期 | 工程主线 | 论文主线 |
|---|---|---|
| 07-28 | S0 起点/任务冻结；远程 round 基础设施 | 写问题、近邻碰撞图、实验协议 |
| 07-29 | S1 R1：legacy vs cold Flow 两卡完整闭环 | 写方法 1：agent factorization + Flow |
| 07-30 | S2.0：grouped adapter、future target/PCA、contract tests | 写方法 2：future representation 与 causal action contract |
| 07-30–07-31 | S2 R3 已完成：action-independent vs action-conditioned local future | 写 local action-conditioned dynamics 与干预协议 |
| 07-31–08-01 | 旧 R4 已完成且未晋级：team capability 通过、own no-regression 失败并完成三项隔离诊断 | 固化负结果和结构转向依据 |
| 08-01–08-02 | 新 R4 已完成：own 精确等价，但 LiftBarrier peer-shuffle CI 跨零，按特殊规则失败并进入 R5 | 记录旧 R4 三项隔离反证、hybrid 负结果与 protected-own 动机 |
| 08-01 | R5 已完成：Protected Shared 与 Protected Role-MoT 均通过，按 macro peer/shared loss 选择 P0 | 写单向 role routing、exact-own contract 与 cross-agent/shared future |
| 08-11–08-17 | S3 R6L/R6J：protected local/team 注入四卡并行 | 完成方法图与首轮闭环结果 |
| 08-18–08-21 | S3 R7 可选逐模块解冻；R8 不得阻塞；冻结模型 | 根据成功率整理主张 |
| 08-22–08-31 | S4 四种子正式训练与闭环 | 成功率主表与统计脚本 |
| 09-01–09-07 | 必要消融与补跑 | 完整初稿、图表和附录 |
| 09-08–09-09 | 只修关键缺口 | 完成 supplementary video |
| 09-10–09-14 | 禁止新增方法 | 压缩到 8 页、内部审稿、最终检查 |
| 09-15 | 只做提交检查 | 提交 |

写作从 S0 同时开始，不能等实验全部结束再写。

## 13. 简化推进与回退规则

1. S2 off-path predictor 按第 7 节推进：R3 验证 own-action dependence，R4 只做零训练 hybrid 诊断，R5 同时要求 protected-own 精确等价和 team capability；action/peer-action shuffle 无效时停止，不能用闭环持平替代。
2. 从 S3 起，P1 在所有任务的闭环成功率都不低于 P0：P1 通过，持平也通过。
3. 从 S3 起，P1 任一任务成功率低于 P0：该轮保留 P0，后续阶段仍可从 P0 继续。
4. On-path 候选主动早停或没有闭环结果：跳过该候选，不阻塞其他分支。
5. 可选轮次来不及完成：直接跳过，不阻塞主路径。
6. protected own 的 checkpoint hash、冻结范围和逐元素输出等价是 R5 以后始终成立的结构不变量；除该不变量和 S2 已定义的 capability/equivalence 检查外，不再增加其他审计或额外准入清单阻塞推进。

## 14. 从现在开始的执行清单

1. **已完成：** 结束 B2，使用 B0 作为 R1 父方案。
2. **已完成：** 建立 R1-F0/F1，完成训练并运行相同闭环任务。
3. **已完成：** F1 在两个任务上均不低于 F0，已晋升为 `P_flow`。
4. **已决策：** 跳过 R2a，将 R2b 延后为非阻塞 sidecar；S2 固定使用 `caa5ed3` 与 R1-F1 checkpoint。
5. **已完成：** 实现 S2.0 grouped adapter、future target builder 与四类 contract tests，完成五任务 PCA/statistics。
6. **已完成：** R3 用 own-action shuffle 验证 action dependence，五任务 gate 全部通过并选择 W1。
7. **已完成但未晋级：** 旧 R4-P1 通过五任务 peer/shared persistence 与 peer-action-shuffle 门槛，但 own no-regression 失败；gate 置零、分组梯度裁剪、team dropout RNG 隔离三项诊断均未改变结论。
8. **已完成但未通过：** 新 R4 零训练 hybrid 在五任务保持 protected-own 精确等价、team loss 优于 persistence、source/action-equivalence 不变；仅 LiftBarrier peer-action-shuffle bootstrap 95% 下界为 `-0.002375`，按特殊规则判定旧 team tower 与 protected P0 表示不兼容。
9. **已完成并通过：** R5 从共同 protected P0 parent 建立 `s2/r5-p0-protected-shared` 与 `s2/r5-p1-protected-role-mot`；两者 own 精确等价、五任务 persistence/shuffle CI、action-equivalence 与 frozen-parent gate 全部通过，按 macro peer/shared loss `1.406178 < 1.412414` 选择 P0。
10. **进行中：** S3-R6 world-to-Flow gated residual injection 已完成四分支实现、白名单、双卡两两 launcher、常驻 monitor 与特殊闭环验收器；下一步在远程完成 R6L/R6J 两批正式训练和 Gate20，按逐任务 P1 不退步规则决定是否进入 R7。
