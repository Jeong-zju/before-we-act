# FE-PC-WAM Research-v2 阶段 A 执行报告

> 执行日期：2026-07-13
> 结论：工程 contract、CPU smoke、独立 runtime 与审计通过；尚无正式模型质量结论。

## 完成项

- V1 基线已按 git SHA 与 artifact SHA256 冻结，V2 schema/checkpoint/bundle 均拒绝 V1 artifact。
- 新增 matched intervention HDF5 contract，保存强制 joint action、未来 ego-local observation、reward/progress/contact/force/success/constraint/terminal 和 valid mask。
- agent 0/1 均在 dataset 唯一入口转换为 ego-first；group/snapshot ID 不属于 `INPUT_KEYS`。
- belief 辅助 target 使用当前决策时刻 `t`；EMA belief 只生成 future target。
- V2-Base 部署配置参数量为 57,077,740：

| 模块 | 参数量 |
|---|---:|
| Plan tokenizer | 327,523 |
| Belief | 4,751,360 |
| Proposal | 6,187,840 |
| Intention | 6,210,624 |
| Block world model | 39,600,393 |

- WAM 已改为逐步动作条件；direct/block 对照拥有相同 I/O，block transition 参数跨四块共享。
- `wam_robust` 不在 V2 DAG 中；错误/推断 plan 不再与真实 outcome 错配训练。
- 状态条件 proposal、teammate posterior、ensemble risk、外部 `E_q[G]`、VPI 和 int8 message codec 已接通。
- 双 request 每步最多投递一个 reply；发送方执行 codec-canonicalized plan，并强制执行 delivered/executed latent 等值检查。
- 两次 bundle load 产生完全独立的 belief/world/controller 对象，仅 artifact hash 相同。

## 实际 smoke 结果

数据目录：`datasets/research_v2_smoke`，共 2/1/1 train/val/test episodes，1.6 MiB；采集 9 个 matched branch groups。

训练目录：`checkpoints/research_v2_smoke`，1.3 MiB。以下阶段均完成一个 CPU train step、一个 validation step并生成 `best.pt`、`last.pt`：

```text
plan → belief → world_direct → world_block
→ proposal → intention → calibration
```

Pipeline manifest 明确记录：

```text
test_split_read = false
wam_robust_present = false
```

Runtime bundle file SHA256：`d386cdc2d88ac143ffd2ed47f365dd2b78301f03570a2c7481a86c74713cfe92`。

独立 runtime 探针结果：

```text
joint_action_shape = [8]
routed_messages = 0
model_instances_independent = true
artifact_hash_equal = true
```

## 验证与审计

- 回归测试：85 passed。
- Research-v2 audit：passed。
- 审计 episode files：4。
- 审计 matched branch groups：9。
- 审计 checkpoint stages：plan、belief、world_direct、world_block、proposal、intention、calibration。
- `privileged_runtime_inputs = []`。
- `git diff --check` 与 staged diff whitespace check 通过。

## 未执行项

以下属于已约定的阶段 B，当前未启动：

- 100-episode pilot 与 D1 6400/800/800 正式采集；
- D2、layout/dynamics/sensor OOD packs；
- Small/Base 三 seed 训练和三成员 world ensemble；
- selective/no-comm/always/random/periodic 正式闭环；
- validation calibration、冻结通信价格、一次性 test；
- Large、Flow Matching 或 DiT Go/No-Go。

当前环境未检测到可用 NVIDIA GPU。Smoke loss 仅证明 wiring 正确，不能用于论文结论或阶段 B 验收。
