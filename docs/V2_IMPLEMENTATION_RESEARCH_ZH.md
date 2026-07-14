# FE-PC-WAM Research-v2 实现与阶段 A 验收

Research-v2 与 `private_gates_v1` 完全隔离。V1 数据和 checkpoint 只能通过旧入口复现，不能作为 V2 上游。冻结基线及 SHA256 见 `docs/V1_MANIFEST_BASELINE.json`。

## 已实现信号流

```text
ego-local history
→ BeliefEncoderV2 [4,256]
→ state-conditioned PlanProposalV2 top-K
→ action-only tokenizer decode
→ ego actions + local teammate-hypothesis actions
→ DirectParallelWorldModelV2 / BlockTransitionWorldModelV2
→ return quantiles + constraint probability + ensemble variance
→ G[k,m] → E_q aggregation → VPI
→ content-blind arbitration → execute one ego action
```

正式单 world 配置为 57,341,420 个部署参数，其中 belief 为 5,015,040、单个 block world 为 39,600,393。RTX 5090 profile 默认训练三个独立 world seed；完整 ensemble 共 136,542,206 个参数，但三个成员顺序训练，不会同时保留三份优化器状态。Block WAM 使用 `H=16、L=4`，块内并行、块间递推、transition cell 共享；训练 self-rollout 比例按 `0 → 50%` 调度。

WAM 的部署 forward 只接受 `belief`、`ego_actions[16,4]` 和 `teammate_hypothesis_actions[16,4]`。posterior 权重只在外部计算 `G_no/G_reveal/VPI`。训练期 teacher belief 只能通过独立的 `forward_train()` 使用，导出的 `forward()` 不接受 target。

## 可直接运行的数据与训练指令

当前机器的 `python` 不在默认 PATH。先从仓库根目录激活已安装环境，后续命令才可直接复制：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
source /home/jeong/miniconda3/etc/profile.d/conda.sh
conda activate wam-py311
python --version
```

然后执行：

```bash
# 先做一次低成本连通性检查；这不是质量训练
python scripts/collect_research_v2_dataset.py \
  --out-dir datasets/research_v2_smoke --smoke

python scripts/train_research_v2_pipeline.py \
  --dataset-root datasets/research_v2_smoke \
  --out-dir checkpoints/research_v2_smoke \
  --smoke --device cpu
```

正式 D1 默认规模为 6400/800/800，另有 100 个 pilot episode 和 0.80 成功率 gate。Collector 固定混合 scripted、noisy、recovery、exploratory 和 near-miss 策略；每个 episode 从相同环境/传感器 snapshot 采集 matched branch groups。目标主机为 32 CPU 线程/64 GB RAM 时直接运行：

```bash
python scripts/collect_research_v2_dataset.py \
  --out-dir datasets/research_v2 \
  --workers 16
```

若采集被中断，原命令增加 `--resume`：

```bash
python scripts/collect_research_v2_dataset.py \
  --out-dir datasets/research_v2 \
  --workers 16 \
  --resume
```

Collector 为每个 worker 分配独立 episode/HDF5，完成后原子改名；resume 会核对 seed、schema、采集配置和代码指纹，清理自身临时文件并补齐编号空洞。默认保留 20 GiB 磁盘余量。配置或采集语义发生变化时会拒绝混合续采。

RTX 5090 正式训练命令：

```bash
python scripts/train_research_v2_pipeline.py \
  --dataset-root datasets/research_v2 \
  --out-dir checkpoints/research_v2 \
  --profile rtx5090 \
  --device cuda
```

中断后恢复：

```bash
python scripts/train_research_v2_pipeline.py \
  --dataset-root datasets/research_v2 \
  --out-dir checkpoints/research_v2 \
  --profile rtx5090 \
  --device cuda \
  --resume
```

`rtx5090` profile 的关键设置如下：

| 阶段 | microbatch | epochs | 每 epoch 最大 step | 累积 |
|---|---:|---:|---:|---:|
| plan | 2048 | 10 | 1000 | 1 |
| belief | 1024 | 12 | 1250 | 1 |
| world_direct | 256 | 5 | 1000 | 2 |
| world_block | 192 | 15 | 2000 | 2 |
| proposal | 1024 | 10 | 1000 | 1 |
| intention | 1024 | 12 | 1000 | 1 |
| calibration | 1024 | 1 | 100 | 1 |

该 profile 使用 `stride=2`、8 个 DataLoader worker、BF16 自动混合精度、TF32、fused AdamW、pinned memory、persistent workers、validation early stopping 和三个独立 world seed。Dataset 会按阶段投影 HDF5 字段，避免 plan/belief 大 batch 读取未使用的 matched-future payload。默认通信价格为 `0.05`（VPI 风险单位），避免未定价时退化成几乎每个 cooldown 都请求；正式通信消融可用 `--communication-price` 覆盖并冻结。首次正式训练建议先用上述命令；确认本机 PyTorch/Triton 组合稳定后，可额外加 `--compile` 尝试进一步加速。显存不足时优先降低 `--world-batch-size`，例如降到 128，而不要改变数据或 horizon 契约。

DAG 为：

```text
plan → belief → world_direct → world_block
→ proposal → intention → calibration
```

每个可训练阶段保存 `last.pt`、validation 选择的 `best.pt` 和 `trainer_state.pt`。RTX 5090 profile 的 patience=5、相对 min-delta=0.1%。resume 会恢复 optimizer、混合精度 scaler、early-stop、epoch 和 RNG 状态。World 优先使用 branch regret，proposal 优先使用 branch-oracle top-K coverage。Belief checkpoint 声明 EMA 为统一部署表示；world/proposal/intention/calibration/runtime 都使用同一权重。校准基于完整 world ensemble 的聚合 validation 预测，bundle 会校验所有上游和成员 SHA256。训练结束会写出 `pipeline_manifest.json` 与 `runtime_bundle/runtime_bundle.json`，训练和校准只读 train/validation split，不读取 test。

## 通信与独立部署

`MessageCodecV2` 支持 code-only、8D、16D int8 residual payload，round-trip reply 大小分别为 78、142、206 bits。

双 request 时，双方独立计算：

```text
requester = (episode_sequence + step) mod 2
```

每步只投递一个 reply。Responder 必须执行 codec-canonicalized provisional plan；coordinator 强制检查 delivered latent 与发送方实际执行 latent 完全一致。当前消息只承诺同一步动作（TTL=0），不会把未执行 shift/commit 的旧 H-step plan 错当作下一步真值。

`load_independent_local_runtime_v2()` 每次调用都会重新构造 belief/proposal/intention/world 实例，并验证 bundle/artifact SHA256。共享无状态对象只是仿真优化，不是部署依赖。

## 审计与评估

```bash
python scripts/audit_research_v2.py \
  --dataset-root datasets/research_v2 \
  --checkpoint \
    checkpoints/research_v2/plan/best.pt \
    checkpoints/research_v2/belief/best.pt \
    checkpoints/research_v2/world_direct/best.pt \
    checkpoints/research_v2/world_block/member_00_seed_7/best.pt \
    checkpoints/research_v2/proposal/best.pt \
    checkpoints/research_v2/intention/best.pt \
    checkpoints/research_v2/calibration/best.pt \
  --bundle checkpoints/research_v2/runtime_bundle/runtime_bundle.json \
  --output checkpoints/research_v2/audit.json
```

审计检查 V2 schema、matched branches、forward signatures、checkpoint contract、artifact hashes 和空 privileged runtime input 列表。

`eval/research_v2.py` 提供 return quantile coverage/pinball、constraint Brier/ECE、grouped branch regret、proposal top-K coverage、VPI calibration 和 D1→D2 scaling gate。测试 split 的组件评估必须提供 frozen validation config。

## 阶段边界

阶段 A 只证明 contract、训练 wiring、runtime 和审计正确，不把 smoke loss 当作模型质量证据。D1/D2、三个训练 seed、正式 ensemble、闭环/OOD 和一次性 test 运行属于阶段 B，需要 GPU 与单独审批。Flow Matching/DiT 仍处于 Go/No-Go 之后。
