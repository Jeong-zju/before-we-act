# FE-PC WAM 多模态扩模技术路线 V1.0

> 调研截止：2026-07-18
>
> 目标：在保留现有 proprioceptive Joint WAM 可复现基线的前提下，逐步扩展到至少使用机器人 state 与 RGB 视觉、可联合预测未来世界与动作、最终能够对齐 DreamZero 一类开源 World Action Model 的多任务模型。
>
> 本文中的参数量、显存和周期，除明确链接到开源项目的事实外，均是用于排期的工程预算，不是论文复现保证。

## 1. 结论先行

**[事实]** 当前系统已经有一个可工作的 Joint WAM，但它仍是小规模、单任务、纯 proprioception 模型：本地实例化统计为 world model `954,172` 参数、action flow `4,608,592` 参数，合计 `5,562,764` 个可训练参数。它在 cooperative-stop 的 standard/challenge 上已与 action prior 同为 100%，说明 pipeline 可执行，也说明当前任务已经无法提供继续扩模所需的学习信号。

**[判断]** 推荐的主路线不是直接把 5.6M 模型替换为 14B DreamZero，而是：

```text
5.6M state-only Joint WAM
  → 20–60M trainable 的视觉 latent WAM
  → 150–350M block-causal 多模态 WAM
  → 0.8–1.5B action-centered video WAM
  → 4–5B 开源 WAM 适配与蒸馏
  → 14B DreamZero 级联合 video-action 模型
```

主线采用三个设计判断：

1. **先证明视觉有因果价值，再增加参数。** 新 benchmark 必须包含 state 无法恢复、但视觉可观测的目标或事件；否则模型即使接收图像，也可能完全忽略图像。
2. **训练时学习未来世界，部署时默认允许 action-only fast path。** Fast-WAM 与 GigaWorld-Policy 的结果都表明，video co-training 的价值可能主要来自训练表征，不一定要求每次控制都生成完整未来视频。
3. **DreamZero 是最终对齐目标，不是第一版实现模板。** DreamZero 的 14B 版本需要多 GPU，模型刷新与动作执行采用 chunk/cache/异步系统；直接迁入当前单任务环境会首先遇到数据与评测失真，而不是模型容量不足。

### 1.1 最低输入契约

从 Phase M0 起，所有被称为“多模态 WAM”的主方法必须同时接收：

- 当前及短历史机器人 state/proprioception；
- 至少一路与 state 时间对齐的 RGB 图像；
- 已执行动作历史；
- 当前任务条件。Phase M1 可使用固定 task ID，Phase M2 起使用自然语言或结构化 goal token。

不允许把图像只用于录像、离线可视化或辅助标签，却继续让 policy 只读取 state。验收报告必须记录 policy 实际消费的 observation keys，并运行视觉反事实消融。

## 2. 截至 2026-07-18 的开源模型调研

### 2.1 第一梯队：可直接借鉴或适配

| 模型 | 规模/范式 | 已公开资产 | 关键工程事实 | 本项目定位 |
|---|---|---|---|---|
| [DreamZero](https://github.com/dreamzero0/dreamzero) | 14B Wan2.1；另有 5B Wan2.2 路径；block-causal joint video-action diffusion | Apache-2.0；DROID/AgiBot 权重；完整训练、LoRA/full fine-tune、推理和新 embodiment 指南；预处理 DROID 数据 | 默认 33 帧、24-step action chunk、320×176、3 视角；仓库称分布式推理至少 2 GPU，14B checkpoint 约 45GB；支持 5B Wan2.2 backbone | **零样本与跨 embodiment 北极星；Phase M4/M5 适配对象** |
| [GigaWorld-Policy-0.5](https://github.com/open-gigaai/giga-world-policy) | action-centered WAM；video dynamics 训练、action-only 部署；Mixture-of-Transformers | 训练/推理代码、HF 权重、LeRobot v3 数据入口 | 官方报告 RTX 4090 约 85 ms；README 暂以 open-loop server/client 推理为主；默认示例 action horizon 48 | **近期主工程模板；优先复刻 causal fast path** |
| [Fast-WAM](https://github.com/yuantianyuan01/FastWAM) | video co-training，test-time 跳过 future imagination | LIBERO/RoboTwin 训练与评测代码、模型、预处理数据 | 论文报告约 190 ms，且去掉 video co-training 比去掉 test-time video generation 伤害更大 | **Phase M2/M3 的最重要对照与初始化参考** |
| [DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT) | video DiT + action DiT 的双 flow/cascaded VAM | MIT；训练、仿真、部署、G1 代码；LeRobot loader | 官方建议训练使用 8 张以上 GPU；显式区分 video/action timestep 与 noise scale | **Phase M3 双专家与 loss 设计参考** |
| [LingBot-VA](https://github.com/Robbyant/lingbot-va) | causal autoregressive video-action；dual-stream MoT、异步执行、KV cache | Apache-2.0；基础模型、RoboTwin/LIBERO 权重、post-train 和自定义 LeRobot 数据流程 | 将动作映射到带 mask 的统一 30 维表示，视频预先抽 Wan2.2 VAE latent；代码要求独立 Python 3.10/PyTorch 2.9 环境 | **Phase M3/M4 的 causal/asynchronous 参考** |
| [Kairos 3.1](https://github.com/kairos-agi/kairos-sensenova) | 4B native cross-embodiment WAM；hybrid linear temporal memory | Apache-2.0；推理代码；4B 通用、RoboTwin 2.0、LIBERO-Plus 权重 | 当前公开入口重点是推理和 benchmark；README 未给出完整自定义 post-training 流程；视频生成延迟不能直接当作 policy 延迟 | **4B 紧凑 WAM 参考与推理基线，不作为近期训练底座** |

### 2.2 最新但需要谨慎解释的工作

- [LingBot-VA 2.0](https://arxiv.org/abs/2607.08639) 于 2026-07-09 提出 semantic visual-action tokenizer、原生 causal pretraining、sparse MoE 与异步 re-grounding，是当前最值得跟踪的“具身原生”路线。[公开仓库](https://github.com/Robbyant/lingbot-va)已经放入 VA2 论文，但截至调研日，README 列出的权重与训练说明仍主要对应初代 LingBot-VA，未清楚列出独立的 VA 2.0 checkpoint/config。因此可借鉴设计，暂不把它作为锁版本依赖。
- [Flash-WAM](https://flashwam.github.io/) 对 video/action 两个不同噪声区间分别做 consistency distillation，把 LingBot-VA 的推理从多步压到 1–2 步。项目报告单 L40S 从 8.1 s 降至 348 ms，但在 RoboTwin 和真机上仍存在相对 teacher 的成功率回归。它适合 Phase M4 的部署蒸馏，不适合在模型尚未学会任务时提前使用。
- [Efficient-WAM](https://arxiv.org/abs/2606.10040) 把 5B video teacher 压到 1B、使用 sparse video latent 和 asymmetric denoising，论文报告约 100 ms/chunk。它非常接近本项目的中间规模目标，但截至调研日没有确认到与 DreamZero/Fast-WAM 同等完整的官方训练仓库，因此当前只作为 M3 设计目标。
- [DreamZero-SO101](https://vizuara-ai-lab.github.io/dreamzero-so101/paper.html) 展示了 715 episodes、LoRA rank 4、108M 可训练参数适配 14B DreamZero 的低成本路径，但作者明确说明尚不适合物理闭环，H100 上约 7.6 s/chunk。它证明“小数据完成接口打通”可行，不证明小数据能获得可部署泛化。

### 2.3 必须保留的非 WAM 基线

| 基线 | 用途 |
|---|---|
| 当前 5.6M proprioceptive Joint WAM / action prior | 检验视觉模型是否真的带来新能力，而非只增加计算量 |
| [SmolVLA 450M](https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx) | 单卡可训练、原生使用多相机 + state + language 的小型 VLA 基线 |
| [OpenPI π0/π0.5](https://github.com/Physical-Intelligence/openpi) | 3B 级 flow-matching VLA 强基线；有 base checkpoint、LeRobot 适配与 PyTorch/JAX 实现 |
| [GR00T N1.7 3B](https://github.com/NVIDIA/Isaac-GR00T) | 开放的多模态 cross-embodiment VLA；支持 state、图像、语言、自定义数据微调 |

这些模型不是本项目最终的 WAM 目标，但如果同数据、同相机、同动作接口下，WAM 长期不优于它们，就不能把“能生成未来”当成控制收益。

## 3. 目标系统结构

### 3.1 模态与因果边界

```text
RGB history ── visual tokenizer ──┐
                                  ├─ block-causal temporal trunk ─┬─ action flow/chunk head
state history ─ state adapter ────┤                              ├─ future state/risk heads
past actions ─ action adapter ────┤                              └─ future visual latent/video head
task text/goal ─ text adapter ────┘
```

必须遵守以下 attention 约束：

- action token 可以看当前/过去的 image、state、action 与 task；
- future video target 不得泄漏给同时间块的 action token；
- future video branch 可以由生成动作条件化；
- 部署 fast path 可以跳过 video decoder，但不可以跳过当前视觉 encoder；
- state 走独立低延迟 token path，不能先渲染为图像再让模型间接恢复 state。

### 3.2 统一输出

每个模型等级至少输出：

1. `action_chunk[t:t+K]`：连续动作与有效维度 mask；
2. `future_state[t+1:t+H]`：state delta、置信度和终止/成功/失败；
3. `future_visual_latent`：M1/M2 为语义 latent，M3 起可解码为视频；
4. `risk/progress/OOD`：用于 shadow 评测和部署 veto；
5. `action_source`、模型刷新时间、chunk age：用于证明动作确实来自目标模型。

### 3.3 联合目标

统一训练目标建议为：

\[
\mathcal{L}=
\lambda_a\mathcal{L}_{FM/action}
+\lambda_s\mathcal{L}_{state}
+\lambda_v\mathcal{L}_{visual\ future}
+\lambda_c\mathcal{L}_{world-action\ consistency}
+\lambda_r\mathcal{L}_{risk/progress}
+\lambda_m\mathcal{L}_{modality\ alignment}.
\]

其中有两条不可违反的 target 规则：

- 专家动作可使用同一 demonstration 的真实 future state/video；
- 生成动作与 demonstration 动作不同时，不得把 demonstration 的 future state/video 冒充生成动作的 ground truth。此时只能使用同一生成动作上的冻结 teacher、仿真 relabel、真实执行 relabel 或无 target 的可微约束。

## 4. ~~Phase M0：多模态数据与非饱和 benchmark~~（已验收，2026-07-19）

**周期预算：2–4 周。模型规模：保持 5.6M，不扩模。**

这是整条路线最重要的一步。M0 不通过，后续所有阶段停止。

### 4.1 新数据 schema

M0-v2 使用 `wam.multimodal/1.1`。`camera_order` 固定为 `[fixed, robot_0_camera, robot_1_camera]`，每条 transition 至少包含：

```text
timestamp / frame_index / episode_index / seed
task.text / task.id
observation.state                         float32 [Ds]
action.commanded / action.executed        float32 [Da]
next_observation.state                    float32 [Ds]

for <camera> in camera_order:
  observation.images.<camera>             uint8   [H,W,3]
  observation.image_timestamp.<camera>    float64
  observation.image_state_timestamp.<camera>
  observation.image_frame_index.<camera>  int64
  next_observation.images.<camera>         uint8   [H,W,3]
  next_observation.image_timestamp.<camera>
  next_observation.image_state_timestamp.<camera>
  next_observation.image_frame_index.<camera>
  camera.{intrinsics,extrinsics,resolution}.<camera>
  next_camera.{intrinsics,extrinsics,resolution}.<camera>

event.visual_signal_active / visual_signal_onset_step
event.visual_signal_kind / rendered_cue_variant
reward / terminated / truncated / done / success / failure / failure_reason
schema_version / behavior_id / environment_config / randomization_config
```

相机矩阵与对应 RGB 在同一次 capture 中采样并做 sample-hold，而不是把 episode 首帧外参复制到全程。HDF5 中 K/E/resolution 的形状分别为 `[T,3,3]`、`[T,4,4]`、`[T,2]`；外参约定为 OpenCV optical camera pose in world。四个 `event.*` 字段只用于离线审计，不由 `MultimodalTrajectoryDataset` 返回给 policy。

M0-v2 canonical 资产采用仓库内同一份 MuJoCo XML、`mujoco.Renderer`、HDF5 与 MP4。LeRobot 是后续适配格式，不是本 Gate 的证据来源。

### 4.2 采样规范

- M0 同时采集全局 `fixed` 与两路随机器人机体运动的 `robot_0_camera`、`robot_1_camera`，统一为 256×256、10 Hz；state/action 为 20 Hz，所有流保存显式时间戳和 frame index。共享 XML 保留 standard 环境的 legacy robot-camera pose；M0 chase pose/FOV 只覆盖 `VisualRequiredEnv` 的独立 `MjModel` instance，并由源代码 SHA-256 绑定。
- 每个 episode、每个 camera 生成一份 `videos/episode_XXXXXX/<camera>.mp4`，只编码 10 Hz 的新 capture；20 Hz HDF5 transition 行通过 frame index 明确表示预期 sample-hold。HDF5 `uint8` 是 canonical raw-unannotated RGB，`mp4v` 视频是有损可视资产，不能作为唯一真值来源。
- 三相机在同一 state snapshot 上顺序渲染，frame index/timestamp 必须跨相机一致；`fixed` 外参应静态，两路机载相机外参必须随各自机器人运动。
- camera order、MuJoCo 版本、XML 路径及 SHA-256、camera ID、parent body、FOV 与代码 SHA-256 写入 manifest，不依赖字典遍历顺序或文档常量。
- train/validation/test 以完整 physical-seed cue pair 为最小单元，并隔离 seed、场景、对象组合和随机化模板。

### 4.3 构造真正需要视觉的任务

当前 22 维 centralized proprioception 足以解决 cooperative-stop。M0 应新增至少三类任务：

1. **视觉事件停止**：刹车/通行信号只通过专用 MuJoCo `visual_event_signal` 几何体进入 RGB，state 中不包含事件真值。onset 前信号保持中性、机器人刹车灯关闭且 paired cue 的 state/RGB 完全相同；onset capture 出现后，policy 才能在下一次动作决策分支。机器人刹车灯不得复制 stop/pass cue 或出现绿色，只能在对应 agent 已请求减速后各自点红；
2. **视觉目标选择**：目标由颜色、形状或屏幕标记指定，state 只含机器人自身 proprioception；
3. **视觉障碍/遮挡**：障碍位置、临时禁区或另一主体意图只从图像可见。

不能把 visual cue 同时编码进 task state、reward shaping input 或 privileged state 再意外传给 policy。原 cooperative-stop 环境另保留每个机器人自己的刹车灯：事件前全部关闭，事件激活后只有真实 braking agent 点亮；人类 annotation/viewer 在事件前不得显示未来 braking agent、启动时刻或倒计时。所有训练/benchmark RGB 均禁用 annotation。

### 4.4 Gate M0

- 数据审计：逐相机 capture skew P99 小于半个控制周期、action frame age 不超过一个视频周期；跨相机 frame index/timestamp 同步；episode 边界零穿越；每相机 MP4 帧数、损坏视频、空帧、预期 sample-hold 和意外重复捕获全部报告；
- 标定审计：current/next K/E/resolution 与 RGB 同 capture，sample-hold 一致；矩阵有限且旋转合法；`fixed` 静态、两路机载外参动态；
- 视觉信号审计：event paired cue 在 onset 前三路 raw RGB 序列逐帧完全一致，首次 active capture 对齐且三路均产生 cue-dependent 像素分叉；目标和障碍 cue 也必须在三路相机中可见；
- 来源审计：只接受 `renderer_backend=mujoco.Renderer` 与 `geometry_source=mujoco_xml`，并复核 MuJoCo 版本、XML/camera rig/source SHA-256；
- state-only policy 在 visual-required suite 上不得超过 70%，否则任务仍可被 state 或环境先验投机；
- scripted oracle 和读取正确视觉 cue 的 oracle 应达到 95% 以上，证明任务可解；
- RGB 随机打乱应使视觉 oracle 显著失败；
- 原有 proprioceptive suite 的行为和 checkpoint 不回归。

**交付物**：schema 1.1、三相机 HDF5/MP4、逐 capture 标定、采集配置、dataset card、MuJoCo/XML provenance、signal-onset evidence、visual-required suite、state-only/vision-oracle 报告、正式 acceptance 报告。

### 4.5 Legacy analytical 预验收（2026-07-18，非 M0-v2 Gate）

旧 `outputs/phase_m0` 使用 schema 1.0、单路 `fixed` 和 analytical OpenCV renderer。它完成了 2,400 episode 的接口/因果任务预验收，但不包含 MuJoCo 原始相机、机载视角或动态外参，因此只保留为历史 preflight，不能据此划去 M0-v2 标题，也不能冒充本阶段正式视频证据。

### 4.6 MuJoCo M0-v2 正式验收

2026-07-19 首个候选正式集在机器 Gate 通过后，被三路 MP4 人工复核否决：`visual_event_stop` 错误地把 stop/pass cue 同步复制到两盏名为 `brake_light` 的机器人灯，导致 PASS 时出现绿色刹车灯，且 onset capture 尚未执行响应动作时灯已变化。旧 manifest/audit/benchmark/acceptance 哈希与结论全部作废，候选资产仅保留为 rejected 证据，不得用于划去 M0。

修复后的正式 Gate 必须额外包含 `visual_event_signal_semantic_isolation` 与 `brake_lights_action_causal_and_red_only`：onset cue-dependent geom 只能是专用信号；两盏灯在 onset capture 仍关闭，之后只按各 agent 的实际减速/制动保持命令独立点红，任何路径不得变绿，truth/opposite RGB 干预不得改变同动作历史下的刹车灯状态。本轮已据此重新完成 2,400 episode、1,200 回合 benchmark、全量审计与 acceptance。

**验收结论：[判断] Phase M0 通过。** Canonical `outputs/phase_m0_mujoco/phase_m0_acceptance.json` 记录 `formal_protocol=true` 且 `passed=true`，所有聚合检查均通过：

- **数据与来源**：2,400 episodes、124,769 transitions、7,200 段三相机 MP4；capture skew P99 为 `0 s`，action frame age 最大为 `0.05 s`，坏视频、空帧、episode 跨界、跨相机不同步、动态外参与视觉信号失败均为 `0`；MuJoCo/XML/camera rig/源码哈希与 manifest 一致。
- **视觉因果 Gate**：1,200 回合中，三个任务及 macro 均为 state-only `50%`、scripted oracle `100%`、vision oracle `100%`、opposite-cue RGB `0%`，视觉反事实下降 `100pp`；policy 输入审计未发现 privileged leakage。
- **语义、人工与回归复核**：两项新门禁通过；三任务×三视角代表性 MP4 复核确认 onset 信号隔离、刹车灯按动作延后独立点红且无绿灯；400 个 event cue pair 的 5,600 个决策前行在 state/action/时序上零差异。旧 checkpoint 锚点 `1c5fc531…d482` 严格重载且树哈希不变，standard/challenge 各 `20/20`；仓库全量测试 `176 passed`。
- **Canonical 证据哈希**：manifest `d0e1289035286db2bf64a7aca63cf767e04b92eeda582877723f8fc1ba5d1c08`，audit `d2e09ed494fe221c1caf4de27242f93887ab7c21ee0549ce1dd1d03e3b0042d2`，benchmark `cd27cb7787e8abfaa91ef60ffb602ce071b0dec4c83a9967dead87e36f2a9e19`，acceptance `7a54e067eaf7580434000e2be4ed3ecf777dcfc3037db114b7eca2a6b7825c76`；`rejected_brake_cue_mirror_20260719` 目录不参与本结论。

**判断边界**：M0 证明的是多模态数据、时序/标定契约与视觉因果 benchmark 闭环，不是已训练的多模态 WAM 或对象泛化结论。本 Gate 的 `object_combination_id` 是 template-scoped split identity；其零重叠不代表 XML 几何/材质的 unseen-object 隔离，物理对象多样性与 unseen-object split 留待 M1/M2 验证。

## 5. ~~Phase M1：视觉条件 latent WAM（DINOv3 重构）~~（已验收，2026-07-21；修订统计协议）

**周期预算：3–6 周。规模目标：20–60M trainable，另加冻结视觉 backbone。算力预算：1×24–48GB GPU。**

### 5.1 模型改造

- 保留现有 state feature encoder、world heads、stateful action flow 和 prior anchor；
- 默认冻结项目别名 `dinov3_vitl16_lvd` 对应的 DINOv3 ViT-L/16 encoder；该别名严格映射官方 Hugging Face 模型 `facebook/dinov3-vitl16-pretrain-lvd1689m`，其他 encoder 必须通过独立配置显式切换并绑定自身 revision、配置与权重哈希；
- 保留 policy 的 raw RGB 契约 `96×96@10Hz`，encoder 内部按官方预处理放大到 `256×256`；原生 `1024` 维 DINOv3 patch token 经可训练投影进入 `512` 维、16-token、3-layer Perceiver resampler；
- 将 visual token 与 recurrent belief/state feature 融合，生成新的 planning feature；
- `future_visual_latent_head` 预测未来 1/2/4/8 帧冻结 DINOv3 teacher 的 CLS feature，而不是像素；
- runtime 对每个新 10Hz RGB frame 只编码一次，并在相邻 20Hz 控制步复用 detached feature cache；frame index 不变但 RGB 改变时 fail closed；
- action flow 仍为单 expert、8-step chunk、执行 2 步，保持与当前 baseline 的控制语义一致。

### 5.2 训练顺序

1. 冻结旧 world/action 和视觉 backbone，仅训练 visual adapter 与 fusion；
2. 冻结视觉 backbone，联合训练 fusion、future latent head 与 action residual；
3. 以低 10–20 倍学习率解冻旧 recurrent belief/world heads；
4. 当前 M1 canonical 配置始终冻结视觉 backbone；局部解冻只能作为后续独立研究配置，不能与本阶段验收证据混用。

### 5.3 必做对照

- state-only current Joint WAM；
- vision-only action policy；
- state + vision、无 future latent loss；
- state + vision、带 future latent loss；
- 同参数量但用额外 MLP 代替 future latent head。

### 5.4 Gate M1

- visual-required suite 上，state+vision 相对 state-only 的总体配对成功率至少提升 10pp，且按 evaluation seed 聚类的总体 95% CI 下界严格为正；3 个训练 seed 中至少 2 个的配对均值方向为正，逐 seed CI 必须报告但不再拥有单 seed 否决权；
- 原 cooperative-stop suite 相对当前 direct policy 回归不超过 5pp；
- shuffle RGB、冻结首帧、遮住 cue 三种测试至少一种造成 15pp 以上成功率下降，证明模型实际使用视觉；该项保留所选干预在全部训练 seed 上 CI 下界为正的严格要求；
- state shuffle 的总体配对成功率也必须下降至少 5pp、总体 95% CI 下界严格为正，且至少 2/3 训练 seed 的配对均值方向为正，证明模型没有退化成纯视觉策略；
- H=8 future latent 线性 probe 对目标位置/事件状态显著优于 current-frame-only baseline；
- 单卡 P95 sensor-to-action 小于 50 ms，或明确把视觉编码降频并报告 action age。

M1 失败时先修数据与 fusion，不允许仅把 hidden dim 加倍后继续扩模。

### 5.5 历史 ResNet-18 pilot 验收（2026-07-20；不适用于当前 DINOv3 版本）

**历史结论：[判断] ResNet-18 pilot 通过当时的 Gate。** `outputs/phase_m1/phase_m1_acceptance.json` 记录 `formal_protocol=true`、`passed=true` 且 `claim_allowed=true`，其训练、视觉因果、future probe、实时性与 legacy 回归检查全部通过。由于当前 canonical M1 已将视觉 teacher 重构为 DINOv3，这份报告仅作为旧实现的回归锚点，不能据此划去当前 Phase M1 标题或声称 DINOv3 版本通过：

- **训练与可复现性**：5 个必做变体各训练 3 个 seed，共 15 个 checkpoint；全部通过哈希校验与严格重载，最大输出差异为 `0`。冻结 prior anchor、预训练视觉 backbone、M0 数据 manifest、上游 M0 acceptance 与 legacy checkpoint 树均保持不变。
- **视觉因果与运行时 Gate**：正式矩阵为 3 个任务 × 5 个模型条件/4 个干预条件 × 3 个训练 seed × 100 个评测 seed × 2 个 cue，共 `16,200` 条唯一 episode 记录。主模型 clean 成功率为 `79.9444%`，相对 state-only 的 `50.0000%` 提升 `29.9444pp`（95% CI `[28.2778, 31.6111]pp`）；shuffle RGB、冻结首帧、遮住 cue 分别下降 `77.2222pp`、`63.2778pp`、`57.4444pp`，state shuffle 下降 `49.0556pp`（95% CI `[47.3333, 50.8333]pp`）。所有动作均来自被测策略，fallback、privileged observation 与非有限动作均为 `0`；`447,616` 次重规划全部为 cold replan。主模型 clean 运行时 deadline miss 为 `0`，sensor-to-action P95 为 `30.7151 ms`，降频路径 action age P95 为 `50.1528 ms`。
- **未来表征与旧任务回归**：H=8 object probe 的 RMSE 为 `0.011745`，优于 current-frame-only baseline 的 `0.034293`，改善 `0.022548`（95% CI `[0.018879, 0.025371]`）；event balanced accuracy 为 `1.000`，相对 baseline `0.965` 提升 `0.035`（95% CI `[0.010, 0.060]`，McNemar `p=0.015625`）。legacy standard/challenge 两个 suite 中，M1 与旧策略均为 `500/500`，回归 `0pp`；M1 的 `73,053` 个控制步全部满足 20Hz control → 10Hz raw RGB 的 `0,0,1,1,…` 帧序列，旧策略全程未读取 RGB。
- **Canonical 证据哈希**：配置 `f6a9a5896ca3b80b37756d02f4fac06d9c5c6403e5476a6562e892c173e050d9`，M0 manifest `d0e1289035286db2bf64a7aca63cf767e04b92eeda582877723f8fc1ba5d1c08`，training summary `b7a6d17b71ec46b6061c265a87b2197e9955a946167c62947240ba60856c1332`，episode JSONL `132dfddf26a99fbe228395a7c88bfdd8034a6351606f6300b49e1b651df4b9a9`，visual evaluation `00b18aca4ac64e779fb53d89c2e9ac89f5e1decb6ac616fee1981d705f45e3fe`，future probe `daed3a5f005ab6e34009fa488c41d746bca3a0675470c9bd13a5727d82576482`，legacy regression `398cfbe597911c339450ec1cb886d351f2bdd78cf4c6fb57185aa4a3dd12174d`，acceptance `9cbe6c8a86e8a91511bda18e3e97c6cf1456b8f87ec5f3cde6fa3e7a2e480d09`。

**判断边界**：本历史结论限定于 ResNet-18 teacher、M0-scale 成功示教、chunk=8/execute=2 与 cold-replan 正式协议。future probe 是同示教 expert action chunk 条件下的离线 teacher-forced 可预测性证据，不构成部署闭环 future-head 因果结论；`gap1_not_runtime_replan_and_cold_only` 的 state causal pair 仅为诊断项，未进入正式训练或 Gate。state 依赖在总体及各训练 seed 聚合上成立，不应外推为每个任务分层都同向；本阶段也不声明 warm-start/异步运行、失败风险学习、unseen-object 泛化或真实机器人有效性。

### 5.6 DINOv3 重构状态与验收边界

- 新 canonical 配置为 `configs/wam_multimodal/m1_latent_wam_dinov3.yaml`；旧 `configs/wam_multimodal/m1_latent_wam.yaml` 与 `outputs/phase_m1/` 保留用于历史 ResNet-18 复现，不覆盖、不重解释。
- DINOv3 模型配置与 `model.safetensors` 必须先由 `scripts/prepare_dinov3_encoder.py` 显式下载到本地并完成身份、revision 与 SHA-256 校验；训练与评测入口只接受已校验的本地文件，禁止在训练中静默联网或回退到随机/其他权重。
- 官方权重属于 gated model，使用者必须先在模型页接受 DINOv3 License 并登录 Hugging Face；模型与许可证来源分别为 [官方模型页](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) 与 [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)。
- checkpoint v2 自包含冻结 teacher 的配置与权重身份；encoder 参数始终 `requires_grad=False`、保持 eval，并与可训练 adapter/resampler 的严格重载证据分开审计。
- **本地 smoke（2026-07-20）**：设置 `MUJOCO_GL=egl` 后全仓 `346 passed`。覆盖 tiny DINO 的冻结与官方预处理、CLS/register/patch token 语义、`1024→512` 可训练投影、动态 future latent、训练反向路径、checkpoint v2 自包含严格重载与篡改拒绝、10Hz feature cache、权重准备入口的离线幂等/fail-closed，以及 evaluation/rollout/acceptance 契约；真实历史 ResNet-18 `state_vision_future/seed_101` v1 checkpoint 也已在 CPU 严格重载。本次触达文件的 Ruff 检查通过。
- **真实 DINOv3 工件加载回归（2026-07-20）**：首轮正式训练在 preflight 暴露官方 safetensors 的 `layer.*` 与直接构造 wrapper 所需 `model.layer.*` 的 key-space 差异；实现现按目标 key 逐项、无歧义地补齐 base-model 前缀，保持 `embeddings.*`/`norm.*` 原名，并继续使用 SHA-256 身份校验与 `strict=True`，缺 key、多 key、shape 篡改和前缀冲突均 fail-closed。对 pinned `model.safetensors`（`dcb2e451…`，415 keys）完成真实 CPU 严格加载与单帧前向：encoder 全冻结，`96×96 RGB → [1,256,1024]` patch tokens 与 `[1,1024]` CLS latent，输出全为有限值。
- **正式训练吞吐优化与性能 smoke（2026-07-20）**：中止首轮训练后确认瓶颈为单进程 HDF5/gzip 解码与过小的 DINO microbatch，而非磁盘等待或 GPU 显存；现场表现为约 2 秒满载、约 3 秒低载，进程平均仅使用约 1 个 CPU core。canonical 配置现采用 train/validation/causal-pair 分离的 `4/2/2` worker、pinned memory、persistent workers 与 `prefetch_factor=2`；数据集按变体和 stage 投影实际必需字段，state-only 不再读取 RGB，无 future 变体不再读取 future RGB，零权重 world/future head 也不再执行。真实 M0 HDF5 上，同一组 512 个 weighted samples 的 batch tensor 逐元素一致且 CUDA loader 全部 pinned；8-batch 输入流由约 `23.96 s` 降至 `3.36 s`（约 `7.1×`，仅代表该数据流 smoke，不外推为完整训练加速比）。RTX 5090 上用 pinned 官方 DINOv3 权重完成等价前后向 microbenchmark：microbatch `8/16/32/64` 分别为 `231.1/242.1/242.0/234.9 frames/s`，总 loss 完全一致，因此 canonical 取 `16`；峰值 allocated 显存约 `3.35 GiB`。上述优化已包含在 `346 passed` 全仓回归中。
- **DINOv3 正式训练与诊断 rollout（2026-07-20 至 2026-07-21）**：5 个必做变体 × 3 个训练 seed 的 15 个 checkpoint 已完成正式训练，全部严格重载一致，最大输出差异为 `0`。另完成主模型 180 回合带仿真相机 MP4 的诊断 rollout，成功 `149/180`（`82.7778%`），其中 seed 101/202/303 分别为 `48.3333%/100%/100%`；18 个 MP4 及 sidecar 均通过哈希、帧数、分辨率、FPS 与 JSONL 链接校验。该 rollout 明确标记 `formal_protocol=false`、`diagnostic_only=true`，只用于定位跨 seed 波动，不替代下述正式 Gate。

### 5.7 DINOv3 正式验收（2026-07-21；按修订统计协议通过）

**验收结论：[判断] Phase M1 DINOv3 通过。** Canonical `outputs/phase_m1_dinov3/phase_m1_acceptance.json` 使用 core `wam.multimodal.m1.acceptance/2` 与 bundle `wam.multimodal.m1.acceptance_bundle/2`，记录 `formal_protocol=true`、`technical_checks_passed=true`、`core_gate_passed=true`、`bundle_checks_passed=true`、`passed=true` 且 `claim_allowed=true`，所有检查均通过，故划去 Phase M1 标题。

- **统计协议修订与审计边界**：原 `/1` 协议在看到结果后曾因 seed 101 两项反向而失败，报告 SHA-256 为 `18270f2299d7c7b78d37c4aed81b5f95131399d4e07448695541ff309189c5b3`。按本次明确放宽的 `/2` 协议，视觉价值与 state-shuffle 仍须满足原总体效应阈值和总体 clustered 95% CI 下界严格为正，同时由 3 个训练 seed 中至少 2 个配对均值同向；逐 seed CI 保留为必报诊断，不再由单个 seed 一票否决。视觉干预仍要求所选干预在全部训练 seed 上 CI 下界为正；future probe 仍要求每个 seed 对两个 baseline 均显著，不随本次修订放宽。训练配置与全部上游证据字节未变，仅重算验收判定。
- **正式矩阵与价值/state Gate**：闭环矩阵包含 3 个任务 × 5 个模型条件/4 个干预条件 × 3 个训练 seed × 100 个评测 seed × 2 个 cue，共 `16,200` 条唯一记录，无缺失、重复、额外或 schema 非法记录。主模型 clean 成功率为 `81.6667%`，state-only 为 `49.5556%`，总体提升 `32.1111pp`（95% CI `[31.4444, 32.7778]pp`），seed 101/202/303 的差值为 `-3.6667/+50.0000/+50.0000pp`，满足 2/3 同向。主模型 state shuffle 后为 `50.8333%`，总体下降 `30.8333pp`（95% CI `[29.7778, 31.8889]pp`），三 seed 的下降为 `-5.0000/+53.6667/+43.8333pp`，同样满足 2/3 同向。
- **视觉因果与输入契约**：shuffle RGB、冻结首帧、遮住 cue 后成功率分别为 `1.5556%`、`16.6667%`、`43.5556%`，相对 clean 分别下降 `80.1111pp`、`65.0000pp`、`38.1111pp`。shuffle RGB 与冻结首帧在全部 3 个训练 seed 上通过正 CI，足以通过“至少一种干预”的 Gate；遮住 cue 仅 2/3 seed 同向，按保留的严格规则不单独计为通过。policy action source、raw RGB 刷新、无 privileged observation、无 fallback 与有限有界动作契约全部通过。
- **future、实时性与回归项**：H=8 object probe RMSE 为 `0.010119`，优于 current-frame baseline 的 `0.062890`，改善 `0.052771`（95% CI `[0.050944, 0.054582]`）；event balanced accuracy 为 `1.000`，相对 baseline `0.965` 提升 `0.035`（95% CI `[0.010, 0.060]`，McNemar `p=0.015625`）。sensor-to-action P95 为 `16.5244 ms`，action age P95 为 `50.0907 ms`，deadline miss 为 `0`；legacy standard/challenge 中新旧策略均为 `500/500`，回归 `0pp`。架构、20–60M trainable budget、15 个 checkpoint 哈希与严格重载、正式训练、上游 M0 和证据 bundle 契约均通过。
- **Canonical 证据哈希**：配置 `e085476c0d0af0f0e60c49c95d2ae063ff855411f768190e4b670e67210f5da3`，training summary `4bf8572db9749655ebc19e0aac5e354ae41a50c9eb009c8f22ab33b585fd1322`，smoke gate `5526d931ff81f2bd08b02f6090b8dca6040d530b09e8987ab1c8558403fb8a98`，episode JSONL `303c1da38ff4dc73a237b407f79d85526f9fb89f965d2a7319cd06eeb6e063a0`，visual evaluation `8844dfa19d75cd0f0692c0c1029b655782d41191ac5403ef4510afde59e39eca`，future probe `5a1271a8626a3a0283f7a356015b5d85f19c1ca35f9d8f67d5e57bb2084975f3`，legacy regression `de79cd0b5542e7f4ced10d01283d417844e8b0a9c98b2b91d359fd70c29cafad`，acceptance `/2` `10577968749ecb07499383467783eccb17911357f9c9ba37f8d8630f068a5214`。

**判断边界**：本结论支持“总体证据与多数初始化 seed 证明视觉价值和 state 依赖”，不支持“所有初始化都稳定”——seed 101 的视觉增益 `-3.6667pp`（95% CI `[-5.6667, -1.6667]pp`）和 state-shuffle 下降 `-5.0000pp`（95% CI `[-6.6667, -3.3333]pp`）仍是明确的训练稳定性技术债，必须继续报告，不得删除失败 seed 或挑选 checkpoint。本阶段仍不声明 warm-start/异步运行、失败风险学习、unseen-object 泛化或真实机器人有效性。

## 6. Phase M2：150–350M block-causal 多模态 WAM

**周期预算：6–10 周。算力预算：2–4×48–80GB GPU。**

### 6.1 架构升级

将 GRU 主干替换为 block-causal Transformer：

- `d_model=768–1024`，12–18 层；
- history 4–8 个视觉时刻、16–32 个 state/action 时刻；
- 每个时间块含 visual、state、past-action、task token；
- future block 同时生成 visual latent、state latent 与 16-step action chunk；
- action head 继续用 flow matching，保留 warm start 和 observation re-ground；
- 增加 embodiment/task adapter，但当前 8 维动作不做无意义 padding。

M2 起自然语言进入正式输入。语言不是用户基础要求，但没有任务条件就无法做多任务与 DreamZero 类 zero-shot 测试。

### 6.2 数据规模

- 内部 visual-required 任务至少 20k–50k episodes；
- 至少 20 个任务/目标组合、3 类视觉随机化，并沿用 M0 的 3 个 canonical 相机视角；
- 失败、恢复、扰动轨迹占 20–40%，不能只保留成功 episode；
- 引入 RoboTwin 2.0 或 RoboCasa 的一个小分片，验证 action/state adapter 不绑定当前环境。

### 6.3 Gate M2

- seen task、unseen object、unseen scene、unseen camera 四个 split 分开报告；
- 在 matched-data 条件下优于 M1，且不是仅靠参数量：去掉 future latent loss 的同规模模型必须作为对照；
- `state+vision` 在 visual-required suite 相对两种单模态模型均有统计显著优势；
- H=16 state NRMSE、视觉对象跟踪误差、action recoverability 随 horizon 的退化曲线可控；
- action-only fast path 与显式 future-latent path 的成功率差距不超过 5pp，延迟至少降低 2 倍；若不满足，保留 future-latent online path 并继续分析；
- 至少 3 个训练 seeds，正式评测每个 suite 100–500 episodes，报告 paired confidence interval。

## 7. Phase M3：0.8–1.5B action-centered Video WAM

**周期预算：2–4 个月。算力预算：约 8×80GB GPU；实际取决于分辨率、帧数和是否冻结 video backbone。**

### 7.1 推荐结构

这一阶段从“预测视觉特征”进入“可解码的未来视频 latent”：

- 使用开源 video VAE 将 RGB 压到时空 latent；
- 使用 0.8–1.5B causal Video DiT 学习 action-conditioned future；
- 单独的 Action DiT 从当前/过去 video latent、state 和 task 生成动作；
- 两个 expert 通过 cross-attention 或共享前若干层耦合；
- attention mask 保证 action 不读取 future video target；
- 训练可同时运行 AC-WM（给定动作预测世界）与 WAM（联合动作/世界）batch；
- 部署默认只激活 action expert，video expert 在 shadow、困难状态或离线评测时运行。

可直接借鉴 Fast-WAM 的 `ActionDiT from Wan2.2 interpolation`、GigaWorld 的 action-centered causal mask，以及 DiT4DiT 的 video/action 分离噪声与 timestep。

### 7.2 数据课程

1. **视觉动力学预训练**：内部仿真 + 公开机器人视频，允许 action-free video；
2. **action-conditioned world modeling**：只用有 state/action 对齐的数据；
3. **joint WAM post-training**：多任务成功、失败、恢复轨迹；
4. **on-policy repair**：收集当前模型失败点，优先 relabel 接触、遮挡、恢复阶段。

建议达到 2–10M 有动作 transitions，再逐步加入更多公开数据。数据小时数、task diversity 和 embodiment diversity 必须与参数量一起记录，禁止只报告“用了更多数据”。

### 7.3 评测世界模型的正确方法

FVD/LPIPS/肉眼好看不是 WAM 的充分指标。M3 必须增加：

- future frame 中目标/机器人 keypoint、mask、深度或光流误差；
- 用 inverse dynamics 从生成视频恢复真实动作的误差；
- 同一 action 的 state/video 动态兼容性；
- contact/event timing error；
- imagined success 与真实 closed-loop success 的 rank correlation；
- policy 在真实 observation re-ground 后是否能从 rollout 偏差恢复。

### 7.4 Gate M3

- matched-data 下同时优于 M2、SmolVLA 和 standalone Action DiT；
- video co-training 相对无 video loss 的控制收益至少 5pp，或数据效率提升至少 2 倍；
- action-only inference 保留 teacher 95% 以上的成功率；
- P95 latency 满足目标控制循环，或通过异步 chunk 保证 action stale time 小于安全阈值；
- 生成视频的 action recoverability、state consistency 与真实成功率相关，不能只有视觉质量提升；
- 通过视觉/状态反事实、privileged leakage、train-eval scene overlap 全部审计。

## 8. Phase M4：适配 4–5B 开源 WAM

**周期预算：1–2 个月。算力预算：4–8×80GB GPU 做 LoRA/post-training；先以小规模 smoke 验证。**

建议并行做两个短实验，再选一条进入正式训练：

### 路线 A：GigaWorld-Policy-0.5

适合目标是本地部署、现有 GPU 较有限、优先 action-only fast path 的情况。

执行项：

- 把 `wam.multimodal/1.1` 转为 LeRobot v3；
- state/action 适配到其 `model_dim=32` 与 48-step action schema，使用显式 mask；
- 只用 1–5% 数据做 open-loop overfit 与 strict reload；
- 再运行 visual-required closed-loop 50-seed gate；
- 对比原版 joint、action-only、关闭 AC-WM loss 三种配置。

### 路线 B：DreamZero Wan2.2 5B

适合目标是最终迁往 14B DreamZero、需要保留 joint video-action generation 和跨 embodiment 能力的情况。

执行项：

- 使用 DreamZero 的 LeRobot v2 → GEAR 转换器与 modality config；
- 第一轮 adapter 可仅使用一路相机、160×320、LoRA、短 action horizon 做 pipeline 验证，但源数据仍保留三视角；
- 第二轮恢复三相机、33 帧/24 action block 语义；
- 验证 state token、KV cache、block index 与实际执行 chunk 的严格时间对齐；
- 分别报告 model refresh rate、action execution rate 和环境 control rate。

### Gate M4

- 外部模型在 matched internal data 上至少优于 M3 一个关键泛化 split，或在同成功率下将训练成本/部署延迟降低 30%；
- 预训练权重确实带来收益：从随机或 video-only initialization 的对照必须存在；
- 运行时同时消费 RGB 与 state；任何单模态丢失都能被日志和 gate 检出；
- 依赖环境与主项目隔离，通过 WebSocket/Policy API 连接，不能为一个外部模型重写当前核心依赖；
- 若 5B 模型不优于 M3，不进入 14B。

## 9. Phase M5：14B DreamZero 级模型

**周期预算：3–6 个月以上。适配算力预算：8–32×80GB；从头预训练需要远高于此的资源与数据。**

只有满足以下前置条件才启动：

- M4 已证明 5B 预训练 WAM 相对 1B 内部模型有明确收益；
- 至少 50 个任务、多个 scene/object split、至少两个 embodiment 或可公开外部 benchmark；
- action-labeled robot 数据达到百小时级并有持续收集能力；
- 具备至少 2 GPU 的在线推理服务器、训练集群与数据存储预算；
- 已建立失败恢复、OOD、latency 与安全 stop 机制。

### 9.1 训练顺序

1. 先严格复现官方 DreamZero-DROID inference/eval；
2. 使用官方 14B checkpoint 对本项目做 LoRA embodiment adaptation；
3. 冻结 video trunk，训练 state/action adapters 与 action head；
4. 解冻后部 DiT blocks 做 joint video-action post-training；
5. 只有跨任务验证明确受限于 backbone 时才考虑 full fine-tune；
6. 通过 caching、量化、异步 execution 或 Flash-WAM 类蒸馏做部署优化。

官方仓库的 4-GPU 配置是训练脚本默认，不等于 4 张 GPU 可以从头复现论文级预训练。正式预算必须先用 100/1,000 steps 的吞吐和显存实测外推。

### 9.2 最终验收

- seen task 与 unseen task/object/environment 分开，不能只报平均成功率；
- 进行 video-only demonstration transfer 与 new embodiment few-shot adaptation；
- 与 π0.5、GR00T N1.7、M4 5B、M3 1B 在同一硬件/数据条件下对比；
- success、progress、latency、GPU memory、模型刷新率、action age、fallback/safe-stop 率全部报告；
- 明确区分 joint WAM direct、action-only distilled path、runtime fallback 的成功来源；
- 不以生成视频“看起来完成任务”替代物理闭环结果。

## 10. Benchmark 与统计协议

### 10.1 三层 benchmark

| 层级 | 内容 | 目的 |
|---|---|---|
| Internal-A | 原 cooperative-stop standard/challenge | 防止 state/control 基础能力回归 |
| Internal-B | visual event/target/obstacle + scene randomization | 证明视觉使用与 world-action coupling |
| External | RoboTwin 2.0、LIBERO-Plus、RoboCasa GR1 中逐步选择 | 与开源 WAM/VLA 形成可比较结果 |

不要一开始同时接三个外部 benchmark。推荐先 RoboTwin 2.0，因为 Fast-WAM、LingBot-VA、GigaWorld、Kairos 都提供相关入口；然后选 LIBERO-Plus 测泛化，最后用 RoboCasa/whole-body 测真实复杂度。

### 10.2 两张主表

每个阶段都必须维护两张表：

1. **Matched-data table**：所有方法同数据、同图像、同 state、同 action horizon、尽量同训练 updates；回答架构是否有效。
2. **Native-pretraining table**：允许各模型使用官方预训练权重；回答最佳可用系统是什么。

混合两张表会把“数据规模收益”错误归因给 WAM 架构。

### 10.3 必报指标

- closed-loop success、progress、return、失败原因；
- state NRMSE、visual latent/keypoint error、action recoverability、world-real consistency；
- state-only / vision-only / state+vision 与 modality shuffle/drop/counterfactual；
- seen/unseen task、object、scene、camera、embodiment；
- OOD AUROC、uncertainty-error correlation、risk calibration；
- sensor-to-action P50/P95/P99、model refresh Hz、control Hz、action chunk age、deadline miss；
- peak VRAM、tokens/s、GPU-hours、数据小时数、参数量与 active parameters；
- direct/fallback/safe-stop action-source coverage。

正式比较至少 3 个训练 seeds；闭环用 paired evaluation seeds；成功率报告 Wilson interval，配对方法用 bootstrap 或 McNemar 检验；在 20-seed smoke 失败时停止 500-seed 正式运行。M1 `/2` 的视觉价值与 state-shuffle Gate 使用“总体固定阈值 + 按 evaluation seed 聚类的总体正 CI + 严格多数训练 seed 的配对均值同向”，逐 seed CI 必报但不单独否决；视觉干预保留所选干预全部训练 seed 正 CI，future probe 保留每个训练 seed 对两个 baseline 均显著的要求。

## 11. 仓库落地结构

保留现有 proprioceptive pipeline 不动，新增并行模块：

```text
configs/wam_multimodal/
  m0_data.yaml
  m1_latent_wam.yaml                  # 历史 ResNet-18 pilot
  m1_latent_wam_dinov3.yaml          # 当前 Phase M1 canonical
  m2_causal_wam.yaml
  m3_video_wam.yaml

models/wam_multimodal/
  vision_encoder.py
  token_resampler.py
  state_action_adapters.py
  block_causal_transformer.py
  latent_world_head.py
  video_action_model.py

train/
  multimodal_trajectory_dataset.py
  multimodal_losses.py

policies/
  multimodal_joint_wam.py

scripts/
  collect_wam_multimodal_dataset.py
  audit_wam_multimodal_dataset.py
  prepare_dinov3_encoder.py
  train_multimodal_wam.py
  evaluate_multimodal_wam.py

external_adapters/
  fastwam/
  gigaworld/
  dreamzero/
```

具体改造点：

- `data/trajectory.py`：提供 `wam.multimodal/1.1` profile、三相机 current/next calibration/timestamp 与事件审计字段；
- `scripts/collect_wam_multimodal_dataset.py`：采集 MuJoCo 三相机 HDF5/MP4，并对 image/state 同步、camera rig 和 provenance 做 fail-closed 绑定；
- `train/multimodal_trajectory_dataset.py`：保持 state-only loader 不变，严格读取三相机窗口、时间戳和逐 capture 标定，避免破坏当前 state-only 测试基线；
- `policies/joint_wam.py`：不直接塞入大模型逻辑，新增 multimodal policy；
- checkpoint manifest 增加 vision backbone、tokenizer、camera order、frame sampling、task vocabulary 和 external base SHA-256；
- DreamZero/GigaWorld/LingBot 分别依赖不同 PyTorch/CUDA/Python 组合，使用独立环境或容器，通过 Policy API/WebSocket 连接主评测器。

## 12. 前 90 天执行计划

### 第 1–2 周：M0 接口

- 冻结当前 state-only baseline 与正式结果；
- 定义 `wam.multimodal/1.1`、三相机 camera manifest 与逐 capture 标定契约；
- 完成 state/image/action 时间同步测试；
- 新增 visual-required 任务原型和 visual oracle。

### 第 3–4 周：M0 数据

- 采集 2,400 episode 正式 M0 Gate 数据；
- 跑 state-only、RGB shuffle、oracle gate；
- 任务仍可被 state 投机时，修改 observation contract，而不是继续采集；
- M0 通过后再采集 10k–20k episode 的 M1/M2 扩展数据。

### 第 5–8 周：M1

- 接入冻结视觉 encoder 与 token resampler；
- 先做 256-sample overfit，再做 1% 数据训练；
- 完成 state-only/vision-only/fusion/future-loss 四组消融；
- 通过 20-seed direct/no-fallback gate 后跑 100–500 seeds。

### 第 9–12 周：M2 起步

- 实现 block-causal mask 单元测试；
- 将 task text、state、image、past action 统一成时间块；
- 训练 50–100M 缩小版验证 loss 与吞吐，再扩到 150–350M；
- 同时用独立环境跑一次 Fast-WAM released checkpoint，打通外部 benchmark 与 Policy API。

90 天的成功定义是 **M1 正式通过 + M2 可训练 smoke + Fast-WAM 外部基线可运行**，不是完成 14B。

## 13. 资源决策

| 可用资源 | 建议上限 | 不建议做的事 |
|---|---|---|
| 单张 24GB | M1；SmolVLA baseline；低分辨率 latent cache | DreamZero 14B 训练；online video generation |
| 单张 48GB/80GB | 完整 M1、缩小 M2、部分 LoRA smoke | 声称复现 5B/14B full training |
| 4×80GB | M2、M3 小模型、DreamZero 5B/14B LoRA smoke | 未做吞吐外推就承诺 full pretraining |
| 8×80GB | M3 正式训练、5B post-training、DiT4DiT/Fast-WAM 复现 | 跳过 M0/M1 直接做 14B 单任务微调 |
| 16×80GB 以上 + 数据团队 | M4/M5 与多 embodiment 数据飞轮 | 只扩参数、不扩任务/数据/评测 |

## 14. 停止条件与可证伪判断

1. **M0 visual-required task 中 state-only 仍接近饱和**：停止模型扩容，重做任务与 observation contract。
2. **M1 state+vision 不优于 state-only，且 shuffle 图像无影响**：判断视觉被忽略；停止进入 Transformer。
3. **M2 同规模 future-loss WAM 不优于纯 action model**：判断当前数据上的 world objective 没有控制价值；保留多模态 VLA，不宣传 WAM 收益。
4. **M3 显式视频生成不优于 action-only fast path**：部署采用 action-only，视频分支仅训练/shadow；这不是失败，而是 Fast-WAM/GigaWorld 路线得到本项目验证。
5. **M4 5B 预训练模型不优于 M3 1B**：不进入 14B，优先改善数据多样性和 adapter。
6. **14B 的性能/延迟曲线不优于 5B**：停止按参数扩展，转向 distillation、sparse MoE、memory 或数据课程。

## 15. 风险与判断边界

### 15.1 最可能的失败原因

- **视觉伪使用**：图像进入 forward，但 state 已包含全部答案；
- **future leakage**：训练 action token 偷看真实 future video token；
- **数据规模错配**：5B/14B 只在单任务几千 episode 上微调，得到高训练集成功和低泛化；
- **视频漂亮但不可执行**：FVD 改善，动作 recoverability 与 closed-loop 不变；
- **延迟统计混淆**：把 action chunk 内的执行频率称为模型推理频率；
- **开源完整性误判**：有权重和 inference 不等于可以复现 pretraining；
- **外部依赖污染**：不同项目强制的 PyTorch/CUDA 版本破坏现有环境。

### 15.2 这个路线可能错的理由

- LingBot-VA 2.0 的“具身原生预训练”若迅速完整开源，直接适配 native tokenizer 可能比沿 video generator 逐级扩展更有效；
- 当前 cooperative-stop 与真实接触操控差异很大，M1/M2 上的收益未必能预测 DreamZero 级操作泛化；
- Fast-WAM/GigaWorld 的 action-only 结论可能依赖其 benchmark，接触密集或强反事实任务仍可能需要 online imagination；
- DreamZero 的零样本收益来自预训练数据、video backbone 与系统优化的组合，单独复制架构未必复现。

因此本文不是“一定扩到 14B”的承诺，而是一条每个阶段都能被数据推翻、也能在较小模型处停止的路线。

## 16. 最终技术决策

近期正式目标锁定为：

> 先完成一个同时读取 state、RGB、past action 和 task condition 的 20–60M latent Joint WAM，在视觉必需且 prior 未饱和的任务上证明视觉和 future modeling 的独立收益；随后迁移到 150–350M block-causal Transformer，并以 Fast-WAM/GigaWorld-Policy 的 action-centered 结构为 1B 级主线。只有 5B 开源 WAM 的 matched-data 实验优于内部 1B 模型，才进入 DreamZero 14B。

这个顺序既保留当前工程资产，又与 2026 年开源 WAM 的主流收敛方向一致：causal video-action modeling、state/action 显式 token、训练时 world supervision、部署时异步 chunk 与可选 action-only fast path。
