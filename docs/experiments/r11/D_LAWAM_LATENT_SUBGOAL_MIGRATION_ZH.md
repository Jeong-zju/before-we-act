# R11 D：LaWAM latent visual subgoal 源码迁移

## 结论与上游身份

候选 D 从固定的官方 `RLinf/LaWAM@4ea6fdadce6c9b8746028307a246b79ee2c4fd55`
只读 checkout 直接导入 `LatentWorldPolicyBackend`、`VLMToLAMQFormer`、
`ConditionalFlowMatchingHead`、LAM decoder、Qwen processor/collator 和 freeze policy。
不是按论文另写一个同名动作头。逐文件路径、SHA256 和 license 见
`third_party/r11/lawam/SOURCE_RECEIPT.json` 与 `LICENSE_MAP.md`。
独立 venv 只安装下列运行依赖；源码由 builder 在 receipt 校验后加入 import path，
不执行 editable install，也不向只读 vendor checkout 写入 `egg-info`。

## 逐符号映射

| 上游符号 | 本地入口 | 适配内容 |
|---|---|---|
| `LatentWorldPolicyBackend` | `_build_official_backend` | Qwen3-VL → latent action → LaWM future decoder → flow action 的完整官方主路径 |
| `VLMToLAMQFormer` | backend 原类 | 8 个 VLM action query 映射为单个 LAM code；用于 decoder 和 distillation |
| `LatentLAMModel/load_latent_action_model` | backend 原 loader | 加载公开 `lawam_lam`；DINOv3 ViT-B/16 encoder 保持冻结，LAM decoder 按官方 SFT 配置解冻 |
| `ConditionalFlowMatchingHead` | backend `flow` | 100×8 动作 flow matching；`h_t`、`h_t1_pred`、VLM hidden 全部进入 action DiT cross-attention |
| `_build_flow_future_condition` | `set_flow_train_step` | 0→1、10k update 的 GT/predicted future scheduled sampling，保留上游 straight-through bridge |
| `LatentWorldTrainCollator` | `LaWAMRoboFactoryAdapter.training_batch` | global 当前/最远合法未来组成两帧 primary video；local 当前帧作为 wrist view；精确保留尾部 action mask |
| `LatentWorldPolicyInferBatchBuilder` | `LaWAMRoboFactoryAdapter.inference_batch` | 同一任务文本、global/local 当前观测、9D state、agent-specific embodiment ID |
| `apply_policy_freeze` | builder | 沿用官方 LIBERO 配方：保留前 16 个 LLM layers、冻结 embedding/末层、解冻 vision merger 与 LAM decoder |

## 预测如何真实影响动作

官方 backend 先用 `vlm_to_lam` 得到 latent action，再由 `lam.decoder(h_t,
pred_action_emb)` 生成 `h_t1_pred`。flow head 的 encoder condition 是
`concat(h_t, h_t1_pred, h_vlm)`；因此每个 action velocity step 都直接 cross-attend
预测 future tokens。这里没有 W10/ACT fallback。

- `prediction_off`：在官方 LAM decoder forward hook 上把 `h_t1_pred` 清零，文本和当前观测不变。
- `prediction_shuffled`：batch 内置换 decoder 输出；单样本时反转 vision-token 序列。
- action-shuffle gate：在官方 `VLMToLAMQFormer` 输出处置换 latent action，再测 future error；
  这是 LaWAM 中动作条件的真实 latent-action 表示，不是改任务文本。

`causal_probe` 直接调用官方 backend 的 shared training encoder，读取其真实
`h_t1_pred`、`h_t1_gt` 与 `h_t`，分别作为 prediction、target 和 persistence；因此
future-vs-persistence 与 action-shuffle 都核验实际送入 flow expert 的 latent subgoal，
没有另造辅助预测头。

## 100-step 与训练目标

动作 horizon 从官方 LIBERO 的 50 对齐到 100，action/state 维度改为本项目 8/9。
flow 的 `horizon_sec=1`，训练时 `action_hz` 等于真实有效尾部长度，从而官方 time-grid
mask 与 HDF5 action mask 完全一致；推理固定 100，输出形状始终 `B×100×8`。

总 loss 使用官方
`loss_flow + 0.1 loss_perceptual + 0.1 loss_distill`。DINO/LAM encoder 是冻结 teacher；
Q-Former、LaWM decoder、flow head 和官方配方允许的 VLM 路径联合训练。effective batch
固定 48，默认 micro-batch 2、accumulation 24。
优化器冻结默认值为 AdamW8bit、lr `1e-5`、weight decay `1e-4`。

## Foundation 与无 task-SFT 证明

冻结资产如下；revision 与逐文件 hash 必须由 immutable asset receipt 核验：

- `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`
- `facebook/dinov3-vitb16-pretrain-lvd1689m@5931719e67bbdb9737e363e781fb0c67687896bc`
- `jialei02/lawam_lam@bd993da2a0861afaac5a95ac86d2555b1313ab8c`
- `jialei02/lawam_pretrain@62b14a8e8990050ec8aeb1e1b8c8694d2bf60e84`

builder 要求 receipt 明写 `task_sft_checkpoint=none`，不会下载或加载
`lawam_libero_sft_release`、`lawam_robotwin_sft_release`。pretrain 只做 shape-compatible
初始化；由于本项目 action/state/horizon 不同，所有跳过 shape 都进入 checkpoint provenance。

## 许可证

上游根目录无独立 LICENSE；README 与 `pyproject.toml` 声明 MIT。NVIDIA 派生 DiT 和
Hugging Face flow head 是 Apache-2.0，Meta 派生 LAM utilities 保留 MIT copyright。
DINOv3 权重是独立的 DINOv3 License gated 资产，必须由账号已接受许可后下载并保存
模型仓库的 `LICENSE.md` hash；不能用项目 MIT 声明覆盖它。
