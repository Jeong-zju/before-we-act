# FE-PC-WAM V1 数据采集与一键训练

## 1. 为什么必须重新采集

正式数据必须满足当前去中心化 schema。loader 要求：

- `T+1` 个本地 observations 对齐 `T` 个 actions；
- 每台机器人独立的 deployable stream；
- 物体只以带 `valid/confidence/age` 的本地估计出现；
- 队友状态与仿真真值只能位于 `/privileged`；
- action 使用发送方 ego/base frame。

不满足该契约的数据不会被一键训练脚本接受；不要把其他 schema 的 HDF5 复制到
`train/val/test` 目录。

## 2. 先检查 GPU

```bash
/home/jeong/miniconda3/envs/wam-py311/bin/python -c \
  "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

必须看到 `True` 和至少 `1` 个 device，才建议运行正式训练。当前 Codex 执行环境未看到
CUDA device；如果你的交互式 shell 也是 `False`，正式五阶段训练会退化到 CPU，耗时很长。

## 3. 首先运行端到端 smoke

Smoke 只验证采集、五阶段反向传播、checkpoint lineage 和 runtime，不代表模型质量。

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

/home/jeong/miniconda3/envs/wam-py311/bin/python \
  scripts/collect_fe_pc_wam_dataset.py \
  --out-dir datasets/smoke \
  --smoke

/home/jeong/miniconda3/envs/wam-py311/bin/python \
  scripts/train_fe_pc_wam_pipeline.py \
  --dataset-root datasets/smoke \
  --out-dir checkpoints/smoke \
  --history 8 \
  --horizon 16 \
  --num-workers 0 \
  --min-active-codes 1 \
  --min-usage-ratio 0.01 \
  --smoke
```

不要把 `checkpoints/smoke` 用于评估或部署。

## 4. 正式数据一键采集

下面是建议的正式数据规模。采集器会自动生成互不重叠 seed 的
`train/val/test`，并混合：

- `scripted / noisy / recovery` 行为；
- `nominal / narrow / occlusion / asymmetric_obstacle / blocked_passage /
  false_belief / hard_comm` 场景；
- 与场景匹配的物体估计 dropout、噪声和陈旧观测。

使用 `datasets/carry` 作为正式目录。追加 `--resume` 时，采集器会审计已有 episode
的 schema 与传感 provenance，不兼容的数据会被直接拒绝。

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

/home/jeong/miniconda3/envs/wam-py311/bin/python \
  scripts/collect_fe_pc_wam_dataset.py \
  --out-dir datasets/carry \
  --train-episodes 1400 \
  --val-episodes 160 \
  --test-episodes 80 \
  --episode-len 500 \
  --seed 20260710
```

采集中断后，用完全相同的参数追加 `--resume`：

```bash
/home/jeong/miniconda3/envs/wam-py311/bin/python \
  scripts/collect_fe_pc_wam_dataset.py \
  --out-dir datasets/carry \
  --train-episodes 1400 \
  --val-episodes 160 \
  --test-episodes 80 \
  --episode-len 500 \
  --seed 20260710 \
  --resume
```

脚本不会静默覆盖已有 episode。结果写入：

```text
datasets/carry/
├── dataset_manifest.json
├── train/episode_*.hdf5
├── val/episode_*.hdf5
└── test/episode_*.hdf5
```

## 5. 正式五阶段一键训练

以下是建议的第一轮基线，不是已经调优的最终超参数。`stride=4` 用于减少高度重叠的
窗口和训练时间。进度条写到 stderr 并留在终端原地刷新，最终 pipeline manifest JSON
写到 stdout；下面将最终 JSON 单独保存，避免训练结束时大段 JSON 在终端滚屏。

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
mkdir -p checkpoints/carry

/home/jeong/miniconda3/envs/wam-py311/bin/python \
  scripts/train_fe_pc_wam_pipeline.py \
  --dataset-root datasets/carry \
  --out-dir checkpoints/carry \
  --device cuda \
  --history 8 \
  --horizon 16 \
  --stride 4 \
  --batch-size 32 \
  --num-workers 4 \
  --plan-epochs 30 \
  --belief-epochs 30 \
  --wam-epochs 50 \
  --intention-epochs 30 \
  --wam-robust-epochs 30 \
  --strict-codebook-health \
  --log-every 50 \
  > checkpoints/carry/pipeline_result.json
```

执行顺序固定为：

```text
plan → belief → wam → intention → wam_robust
```

如果显存不足，首先把 `--batch-size 32` 改为 `16` 或 `8`。不要通过减小 codebook
健康门槛来掩盖 code collapse。

训练中断后，使用原参数并追加 `--resume`。只有配置、数据路径、contract tag 和上游 SHA256
全部一致的阶段才会复用。已经完整保存的阶段显示 `reused`；中断时尚未保存的当前阶段
会从头开始：

```bash
/home/jeong/miniconda3/envs/wam-py311/bin/python \
  scripts/train_fe_pc_wam_pipeline.py \
  --dataset-root datasets/carry \
  --out-dir checkpoints/carry \
  --device cuda \
  --history 8 --horizon 16 --stride 4 \
  --batch-size 32 --num-workers 4 \
  --plan-epochs 30 --belief-epochs 30 --wam-epochs 50 \
  --intention-epochs 30 --wam-robust-epochs 30 \
  --strict-codebook-health \
  --log-every 50 \
  --resume \
  > checkpoints/carry/pipeline_result.json
```

如需明确丢弃已有阶段并重训，使用 `--force-retrain`；该选项不能与 `--resume` 同时使用。

训练、`normalization` 和 `codebook scan` 使用终端原地刷新的进度条，不会为每次更新
新增一行。训练条显示当前 epoch、loss、吞吐率、耗时和 ETA；阶段边界只输出少量
`start/reused/completed/failed` 信息。在 CUDA 上，每个完成的训练阶段会输出
`peak_cuda_memory`。

进度条会按时间自动原地刷新；`--log-every N` 控制 loss 和吞吐率后缀每经过多少个优化
step 更新，默认值为 `50`。如需让 stdout 保持机器可读 JSON 且不显示任何过程信息，
可使用 `--quiet`；显示频率和静默选项不会写入 checkpoint 的训练配置，因此不会改变
`--resume` 兼容性。

## 6. 输出与必要检查

```text
checkpoints/carry/
├── plan.pt
├── belief.pt
├── wam.pt
├── intention.pt
├── wam_robust.pt
└── pipeline_manifest.json
```

正式闭环使用 `wam_robust.pt`。`wam.pt` 是用真实 teammate plan teacher 训练的中间阶段。

`--strict-codebook-health` 会在 active code 数或 usage ratio 不足时停止后续训练。重点检查
`plan.pt` / manifest 中的：

```text
used_codes
dead_codes
usage_ratio
entropy
perplexity
codebook_health_passed
```

当前 pipeline 使用 `train/` 完成参数学习，并把 `val/`、`test/` 路径写入 manifest；
正式成功率、安全性和通信模式对比需要随后在相同 seeds 上运行配对评估，不能以训练 loss
或 smoke loss 代替。

## 7. Held-out 组件评估

> 组件评估要求数据与 checkpoint 都使用严格的 per-agent contact/force provenance；
> 不兼容的输入会被 CLI 直接拒绝。

先在 validation split 检查泛化，不要先看 test。评估全程使用冻结权重，并在结束时再次
验证 checkpoint SHA256：

```bash
/home/jeong/miniconda3/envs/wam-py311/bin/python \
  scripts/evaluate_fe_pc_wam_components.py \
  --data-dir datasets/carry/val \
  --plan-checkpoint checkpoints/carry/plan.pt \
  --belief-checkpoint checkpoints/carry/belief.pt \
  --wam-checkpoint checkpoints/carry/wam_robust.pt \
  --intention-checkpoint checkpoints/carry/intention.pt \
  --split validation --device cuda --batch-size 64 --num-workers 4 \
  --output outputs/acceptance/components_val.json
```

报告包含 held-out plan code usage/perplexity 与动作重构、belief/WAM 各项 loss，以及
intention accuracy、macro-F1、Brier、ECE 和 active-support coverage。

## 8. CPU 接线 smoke

当前无 CUDA 的 shell 只跑截断 smoke，不把结果解释为性能：

```bash
/home/jeong/miniconda3/envs/wam-py311/bin/python \
  scripts/evaluate_fe_pc_wam_rollouts.py \
  --dataset-root datasets/carry \
  --checkpoint-dir checkpoints/carry \
  --split val --output-dir outputs/acceptance/smoke \
  --device cpu --max-episodes 2 --max-steps 2 \
  --num-candidates 2 --num-teammate-hypotheses 1 \
  --residual-sigma-points 1 \
  --no-save-failure-videos
```

如果 `wam_robust.pt` 尚未训练完成，可显式加入 `--use-base-wam`，临时使用
`wam.pt` 检查闭环接线和定性行为。该选项即使在目录中存在 `wam_robust.pt` 也会强制
选择基础 WAM，并在 summary/snapshot 中记录 `diagnostic_base_wam=true`；这类运行永远
不能用于正式验收或冻结 test 配置：

```bash
MUJOCO_GL=egl /home/jeong/miniconda3/envs/wam-py311/bin/python \
  scripts/evaluate_fe_pc_wam_rollouts.py \
  --dataset-root datasets/private_gates_v1 \
  --checkpoint-dir checkpoints/private_gates_v1 \
  --use-base-wam \
  --split val --output-dir outputs/acceptance/base_wam_probe \
  --device cuda --max-episodes 4
```

输出目录会包含可恢复的完整逐 episode records、紧凑的 `records.json`、`summary.json`、
`candidate_codes.json`、带 candidate 证据的 `artifact_audit.json`，以及记录 Git/source、
依赖、所选 HDF5 集合、数据 manifest 和 checkpoint 哈希的 `experiment_snapshot.json`。
`COMPLETED.json` 标记一次运行已经完成；非空输出目录不会被静默覆盖。CPU smoke 的
`frozen_config.json` 会明确标记 `validation_freeze_eligible=false`。

默认显示总 episode 进度条，并在每个实际执行的 episode 下显示 step 进度条；`--quiet`
会关闭两个进度条。闭环默认保存所有失败 episode 的完整 MP4：首跑不渲染，失败后以
同一 seed/runtime 配置确定性重放并录像，只有结局、步数和 return 校验一致才挂到原始
record。可用 `--no-save-failure-videos` 关闭。若还要为每种通信模式的前 2 个 episode
（无论成功失败）输出 MP4：

```bash
MUJOCO_GL=egl /home/jeong/miniconda3/envs/wam-py311/bin/python \
  scripts/evaluate_fe_pc_wam_rollouts.py \
  --dataset-root datasets/carry \
  --checkpoint-dir checkpoints/carry \
  --split val --output-dir outputs/acceptance/val_video_smoke \
  --device cuda --max-episodes 2 --max-steps 100 \
  --render-video --video-episodes 2 \
  --video-fps 20 --video-width 640 --video-height 480
```

视频按 `videos/<mode>/episode_XXXXXX.mp4` 保存，`videos.json` 记录选择原因、相对路径、
SHA256、帧数、FPS、分辨率和 codec。录像采用流式编码，不会把整个 episode 的帧保存在
内存中。无显示器的 NVIDIA/EGL 环境使用 `MUJOCO_GL=egl`；纯 CPU 且安装了 OSMesa 时
可改为 `MUJOCO_GL=osmesa`。失败重放会额外执行一次失败 episode；显式 `--render-video`
则会给指定的前 N 个 episode 增加实时渲染开销。

## 9. Validation 通信校准

在能看到 CUDA 的环境中，以不同 `--delta-margin` / `--lambda-bits` / `--cooldown-steps`
重复 validation 运行。每组参数使用独立输出目录；中断后以相同参数追加 `--resume`：

```bash
/home/jeong/miniconda3/envs/wam-py311/bin/python \
  scripts/evaluate_fe_pc_wam_rollouts.py \
  --dataset-root datasets/carry \
  --checkpoint-dir checkpoints/carry \
  --split val --output-dir outputs/acceptance/val_margin_0 \
  --device cuda --delta-margin 0.0 \
  --match-baselines-to-selective
```

`--match-baselines-to-selective` 会先运行 selective，再用确定性的 cumulative-quota
periodic schedule 和 random probability 匹配 validation 上实际观察到的 selective
request rate（绝对误差门槛 2 个百分点）。只有完整 160 episode、CUDA、严格传感契约、
五模式配对、预算匹配和 artifact audit 全部通过时，才会生成可用于 test 的有效冻结配置。

首次运行不要加 `--resume`。只有同一输出目录已经生成 snapshot 且代码、依赖、数据、
checkpoint 和全部评估参数均未变化时，才用原命令追加 `--resume` 续跑。

## 10. 冻结后的一次性 test

只在 validation 配置冻结后，把完全相同的策略/成本参数用于 test：

```bash
/home/jeong/miniconda3/envs/wam-py311/bin/python \
  scripts/evaluate_fe_pc_wam_rollouts.py \
  --dataset-root datasets/carry \
  --checkpoint-dir checkpoints/carry \
  --split test --output-dir outputs/acceptance/test_frozen \
  --device cuda \
  --frozen-config-from outputs/acceptance/val_margin_0/frozen_config.json
```

test 禁止重新计算 selective rate 或调整任何通信参数；CLI 会从完整、无截断的五模式
validation `frozen_config.json` 验证 160 条/模式的紧凑 record-set、配对 input digest、
candidate audit、snapshot、代码/依赖/数据/checkpoint 哈希后，导入每个模式的 runtime
配置。test 首次运行同样不要加 `--resume`，中断后才能以完全相同的命令追加该选项。

正式汇总比较 `no_comm / always_reply / selective_vpi / periodic / random`，提供 paired
bootstrap 95% CI、按 scenario 的描述性结果和默认验收判定。同步仿真 transport 的
realized delay 固定记录为 0；VPI 所收取的 expected delay cost 单独记录，不能据此声称
已经验证真实网络时延。通信 bits 是按配置的 residual quantization 计算的协议预算，
不是 serializer 或真实链路上的 wire-byte 测量。
