# S4-R8 并行双卡部署、监控与验收手册

状态：2026-08-03 预注册执行版。R8 与 R7 并行；组合的是胜出方法配方，不是 checkpoint 权重。

## 1. 固定实验身份

| GPU | 分支 | model kind | 唯一候选轴 |
|---:|---|---|---|
| 0 | `s4/r8-p0-horizon-prefix-mean` | `s4_r8_horizon_prefix_mean` | `action_prefix_aggregator=prefix_mean` |
| 1 | `s4/r8-p1-causal-prefix-attention` | `s4_r8_causal_prefix_attention` | `action_prefix_aggregator=causal_prefix_attention` |

两边共同固定 `utility_coupling_weight=0`、rank 32、30k fast selection、全 750 episodes 和同一个 float32 projected-next-view cache。R8 只读取共同 R6L/R5/R4 ancestors；不允许读取 R7 P0/P1 checkpoint。

## 2. 一键启动

服务器首次创建后先保证永久会话存在；此命令幂等且不会关闭已有会话：

```bash
tmux has-session -t ssh_tmux 2>/dev/null || tmux new-session -d -s ssh_tmux -n shell
```

从空服务器首次部署时，在 `feat/model-improvements` 干净 worktree 中运行：

```bash
cd /workspace/fe-pc-wam
bash scripts/launch_s4_r8_existing_server.sh --run-id s4-r8-parallel-fast30k-round1 --focus-monitor
```

若 datasets、RoboFactory 或 DINO 缺失，入口自动追加 `--prepare-from-s0`，并通过 S0 的 mode-0600 FIFO 在终端隐藏读取 Hugging Face token。dataset 固定 revision、官方 `hf download`、Xet 开启且使用默认 8 workers；DINO/RoboFactory 关闭 Xet 且单 worker；失败后复用原位 `.incomplete`。token 不进入 argv、持久 environment、tmux command、manifest 或日志。已有共享资产时不会再次索取 token。这个 asset-only bootstrap 不重训任何旧阶段模型；R6L-P1、R5-P0、R4-P0 和五任务 Flow 必须来自已验收服务器或显式路径，并继续由固定 SHA256 fail closed。

底层一键入口等价于：

```bash
bash scripts/launch_s4_r8_2gpu_tmux.sh \
  --run-id s4-r8-parallel-fast30k-round1 \
  --prepare-from-s0 \
  --focus-monitor
```

launcher 复用永久 `ssh_tmux`，创建 `<run>-prepare`、`<run>-p0`、`<run>-p1`、`<run>-monitor` 四个 `remain-on-exit=on` 窗口。P0/P1 是各占一张 GPU 的独立训练，不使用 DDP；datasets、artifacts、DINO cache 和 future feature cache 只保留一份，candidate output/checkpoint 完全隔离。

## 3. 查看程序、状态、进度与心跳

进入永久监控窗口：

```bash
tmux attach -t ssh_tmux
tmux select-window -t ssh_tmux:s4-r8-parallel-fast30k-round1-monitor
```

非交互一次性状态：

```bash
cd /workspace/fe-pc-wam
python3 scripts/s4_r8_runtime.py monitor \
  --once \
  --run-root outputs/s4_r8_runs/s4-r8-parallel-fast30k-round1
```

monitor 每 5 秒刷新，并明确显示：

- 当前 phase、实际 program、wrapper PID、child PID、GPU PID、GPU 利用率和显存；
- 20 秒 producer heartbeat，超过 75 秒显示 `STALE`；
- shared prepare、双 GPU future-cache worker 的 task/episode 进度；
- 跨服务器新下载数据会先显示一次 `HDF5 import SHA256=x/750` 与字节进度；只有逐文件匹配 accepted manifest 后才生成 stat-bound receipt，绝不通过回拨 mtime 绕过 proof；
- preflight、update/30000、百分比、team/agent exposure、下一 milestone、Flow 冻结/解冻状态、loss、gradient norm、各组 LR；
- Gate20 的 condition/task/episode/step，四个 core condition 优先完成，四个 diagnostic condition 后置；
- `p0_p1_step0_exact`、12/12 suffix/prefix-sensitive、12/12 prefix bootstrap lower>0、common causal gates 和最终 winner。

直接查看候选窗口与日志：

```bash
tmux select-window -t ssh_tmux:s4-r8-parallel-fast30k-round1-p0
tmux select-window -t ssh_tmux:s4-r8-parallel-fast30k-round1-p1
tail -f outputs/s4_r8_runs/s4-r8-parallel-fast30k-round1/candidates/p0/logs/candidate.log
tail -f outputs/s4_r8_runs/s4-r8-parallel-fast30k-round1/candidates/p1/logs/candidate.log
```

## 4. 安全退出与恢复

只停止本轮窗口和已核对 PID，不销毁永久 `ssh_tmux`：

```bash
cd /workspace/fe-pc-wam
bash scripts/stop_s4_r8_2gpu_tmux.sh \
  --run-id s4-r8-parallel-fast30k-round1
```

重新执行同一个 launch 命令会核验 manifest、branch、config、checkpoint/resume hash 后修复 dead pane 并从候选隔离的 resume 继续。若 shared prepare 在 ready 前被精确 stop，launcher 只会在同一 run 且 prepare pane 确实需要重启时清除 `shared.failed`；已有 `shared_hdf5_verification_receipt.json` 与 SHA256 sidecar 必须成对存在并重新通过 750 文件 stat identity、manifest、proof hash 和 receipt hash 校验后才复用，因此不会重扫跨服务器导入的约 707 GiB 内容，也不会用未验证收据绕过门槛。不要手工 `killall python`、删除 run root 或重建同名 run。需要改变 micro/accum、cache 或验收协议时，必须两分支同步修改并使用新 run-id。

## 5. 验收产物与 winner 处理

关键产物位于：

```text
outputs/s4_r8_runs/<run-id>/pairs/pair_exact.json
outputs/s4_r8_runs/<run-id>/pairs/p0_p1_step0_exact.json
outputs/s4_r8_runs/<run-id>/candidates/{p0,p1}/checkpoints/policy.pt
outputs/s4_r8_runs/<run-id>/candidates/{p0,p1}/validation/candidate_report.json
outputs/s4_r8_runs/<run-id>/candidates/{p0,p1}/validation/prefix_suffix_exact.json
outputs/s4_r8_runs/<run-id>/candidates/{p0,p1}/validation/prefix_shuffle_by_source_horizon.json
outputs/s4_r8_runs/<run-id>/acceptance.json
```

正式规则先淘汰任一 common structural/causal Gate20 或 R8 special gate 失败者，再比较 normal Gate20 五任务 macro；精确持平选 P0。acceptance 只决定方法分支。胜出分支 fast-forward/merge 到 `feat/model-improvements` 时保留 checkpoint hash、配置、全部报告与本手册的结果回填；不会把 P0/P1 checkpoint 参数相加或平均。

## 6. 结果回填（训练完成后更新）

待本轮 `acceptance.json` 生成后填写：run-id、公共 commit、P0/P1 branch commit、preflight 吞吐/峰值显存、30k checkpoint SHA256、12 组 prefix bootstrap 下界、八路 Gate20 macro、eligible candidates、winner、合并 commit 与结果同步位置。未生成 acceptance 前不得先写 winner。
