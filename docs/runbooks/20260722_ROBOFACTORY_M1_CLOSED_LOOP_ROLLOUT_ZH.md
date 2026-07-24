# RoboFactory LiftBarrier M1 闭环 Rollout

本入口用于评测 `checkpoints/m1_liftbarrier_scratch_seed101` 的真实 RoboFactory 闭环成功率，并为每个 episode 保存 MP4。RoboFactory 与 WAM 的 Python/Torch 版本不兼容，因此必须用两个进程、两个虚拟环境运行：

```text
RoboFactory Python 3.9                         WAM Python 3.11
┌──────────────────────────────┐              ┌──────────────────────────┐
│ LiftBarrier reset/step       │ raw RGB+36D  │ frozen DINO + scratch M1 │
│ success + sensors MP4        ├─────────────►│ stateful policy.act()    │
│ seed/episode/result evidence │◄─────────────┤ raw 16D pd_joint_pos     │
└──────────────────────────────┘  TCP 8765    └──────────────────────────┘
```

通信默认只绑定 `127.0.0.1`，使用版本化长度帧、JSON 元数据和原始 ndarray 字节。RGB 保持无损 `uint8`，不经过 JPEG；协议不接受 pickle。每次推理请求都绑定 `(episode_index, step)`，乱序或重复请求、错误 shape/dtype、非有限动作、legacy/fallback policy 诊断都会立即终止运行，不会下发随机动作兜底。

## 快速 smoke（3 episodes）

先在终端 1 启动环境端：

```bash
cd /home/jeong/zeno/wam/RoboFactory
source ./activate_uv.sh

python ../fe_pc_wam/scripts/serve_robofactory_m1_rollout.py \
  --robofactory-root . \
  --host 127.0.0.1 \
  --port 8765 \
  --episodes 3 \
  --seed-start 900 \
  --max-steps 500 \
  --sim-backend cpu \
  --shader default \
  --video-fps 20 \
  --output-dir ../fe_pc_wam/outputs/robofactory_m1_closed_loop_smoke_seed900_n3
```

看到 `waiting for the M1 inference client` 后，在终端 2 启动推理端：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

uv run --frozen python scripts/run_robofactory_m1_inference.py \
  --checkpoint checkpoints/m1_liftbarrier_scratch_seed101 \
  --config configs/wam_multimodal/m1_liftbarrier_scratch.yaml \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8765
```

推理端会先严格校验本地 DINO 配置/权重哈希及整个 scratch checkpoint，再连接环境端。入口还会锁定正式训练 YAML 的完整 SHA-256；配置内容有任何漂移都会拒绝运行。smoke 使用 900–902，不会提前预览正式评测的 1000–1099 seeds。首次加载期间环境端保持等待是正常行为。

## 正式 benchmark（100 episodes）

smoke 完成后使用新的输出目录重新启动两个终端。环境端命令为：

```bash
cd /home/jeong/zeno/wam/RoboFactory
source ./activate_uv.sh

python ../fe_pc_wam/scripts/serve_robofactory_m1_rollout.py \
  --robofactory-root . \
  --host 127.0.0.1 \
  --port 8765 \
  --episodes 100 \
  --seed-start 1000 \
  --max-steps 500 \
  --sim-backend cpu \
  --shader default \
  --video-fps 20 \
  --output-dir ../fe_pc_wam/outputs/robofactory_m1_closed_loop_seed1000_n100
```

推理端命令与 smoke 完全相同。默认 runtime 每次生成 8-step action chunk 后先执行 2 步再重规划，并对上一 chunk 做 shift-2/warm-start；这与正式训练的 `execution_steps=2` 合成 warm-start 契约一致。每个 control step 的新 RGB 仍会进入两帧视觉历史，但不会提前触发 shift-1。正式入口不暴露改变 replan 或 warm-start 语义的参数，避免把消融结果误记为 canonical benchmark。

需要注意一个已显式记录的训练/在线分布差异：episode reset 的第一次决策只有一帧 RGB、一份 state 且没有历史 action；policy 会使用带 mask 的零填充补齐历史，而训练窗口没有覆盖这个完全相同的单帧 reset 情形。该限制会写入 summary，不能把它解释成接口错误，也不能在报告成功率时隐去。

## 输出

环境端输出目录结构如下：

```text
robofactory_m1_closed_loop_seed1000_n100/
├── rollout_summary.json
├── rollout_episodes.jsonl
└── videos/
    ├── episode_0000_seed_1000_success.mp4
    ├── episode_0001_seed_1001_failure.mp4
    └── ...
```

- `rollout_summary.json`：最终成功数、成功率、Wilson 95% 区间、checkpoint/DINO/action codec 身份、正式配置哈希、WAM 与 RoboFactory 的 Git/关键源码哈希、RoboFactory 环境配置哈希、两端依赖版本、seed 协议、已知限制、动作范围和完整 episode 索引。每集 reset 前会同时设置全局 NumPy seed 与 `env.reset(seed=...)`，覆盖 RoboFactory 场景构建器对全局 `np.random` 的直接调用，保证 barrier 随机位置可复现。
- `rollout_episodes.jsonl`：每完成一集即 `fsync` 落盘，包含 success、步数、stop reason、视频相对路径、推理及 RPC latency。
- `videos/*.mp4`：`sensors` 模式下 agent0、agent1、global 三相机横向拼接，20 FPS；policy 只消费其中无损的 `head_camera_global`，不会看到其他相机。

输出目录实行 fail-closed：只要已包含文件就拒绝复用，防止显式视频名被覆盖。调试时可在环境端添加 `--no-video`，但正式成功率运行应保留视频证据。

只有 `rollout_summary.json` 同时满足 `formal_benchmark.reportable=true`、`completed=true`、`fatal_error=null`，并完成规定的 100 个 seed 和 100 个 MP4，才可对外报告正式成功率。可在运行结束后执行：

```bash
jq -e '
  .formal_benchmark.reportable == true and
  .completed == true and
  .fatal_error == null and
  (.episodes_completed == .benchmark_protocol.episodes_requested)
' ../fe_pc_wam/outputs/robofactory_m1_closed_loop_seed1000_n100/rollout_summary.json

jq '{successes, episodes_completed, success_rate, success_rate_wilson_95}' \
  ../fe_pc_wam/outputs/robofactory_m1_closed_loop_seed1000_n100/rollout_summary.json
```

发生异常时仍会原子写入 partial summary，并尽可能保存 `_aborted.mp4`，用于定位故障；其中的部分成功率和置信区间不是正式 benchmark 结果。

## 常见故障

- `pkg_resources`、`pynvml`、Gymnasium wrapper 属性转发 deprecation，以及 barrier default-pose warning：这是当前 SAPIEN/ManiSkill/RoboFactory 依赖发出的非阻断警告；只要随后出现 `Simulator ready` 即可继续。
- `failed to find device "cuda"`：SAPIEN/Vulkan 未看到 NVIDIA renderer；请在有 GPU/Vulkan 权限的外部终端运行，不要在隔离 sandbox 内启动仿真。
- `address already in use`：换一个端口，并在两个终端同时修改 `--port`。
- DINO/checkpoint hash mismatch：不要继续 rollout；确认使用正式权重工件和 `m1_liftbarrier_scratch.yaml`，入口不会联网下载或回退随机视觉权重。
- 输出目录已有文件：换一个新的 run 名称。入口不会覆盖既有结果或视频。
- 跨机器运行：协议没有身份认证；默认禁止非 loopback bind。只有在受信任、隔离网络中才可显式加 `--allow-remote`，并配置对应 `--host`。
