# M2 双任务多视角模型闭环失败记录

## 结论

`feat/model-improvements` 分支在 commit
`4dae9e79bbfdd2d36a0381da2179a97390766d83` 上训练的 M2 双任务多视角模型，
虽然通过训练报告、严格重载和 teacher-context 离线验证，但未通过 RoboFactory
闭环 Gate。本次训练按失败处理，不得作为可接受 checkpoint 使用。

失败 checkpoint 的身份如下：

- 训练 seed：`101`
- 格式：`wam.robofactory.m2.checkpoint/5`
- checkpoint tree SHA-256：
  `2900d054ea917c26812bb729af7d5c8d601daa1864dd57445903f057170d0126`
- `model.safetensors` SHA-256：
  `32b8412655b292ed2b3aebbeca5fd77eefff4cd48a6ecd3d47434b48bf28e476`
- 任务：`lift_barrier`、`long_pipeline_delivery`

## 2026-07-27 闭环结果

正式 Gate 使用训练 seed `3000`、validation seed `3099` 和未见 seed
`900–902`。验收要求每个任务分别达到 `1/1`、`1/1` 和至少 `2/3` 成功。

| 任务 | train 3000 | validation 3099 | unseen 900–902 | Gate |
|---|---:|---:|---:|---|
| LiftBarrier | 0/1 | 1/1 | 1/3 | 失败 |
| LongPipelineDelivery | 0/1 | 0/1 | 0/3 | 失败 |

所有 rollout 均正常完成，`fatal_error=null`、
`direct_model_action_coverage=1.0` 且 `engineering_smoke_passed=true`。因此这不是
fallback、推理链路中断或评测未完成造成的假失败；模型闭环任务能力未达到门槛，
其中 LongPipelineDelivery 在 5 个评测 episode 中全部失败。

被本次 v5 checkpoint 取代的 v4 joint checkpoint
`d4817f30000876d8af82808889060641b5fa0a736bc01e52c1f64befd5768217`
也呈现相同结果：LiftBarrier 为 `0/1`、`1/1`、`1/3`，
LongPipelineDelivery 为 `0/1`、`0/1`、`0/3`。该结果进一步说明升级到
多视角 v5 后没有解决 LongPipelineDelivery 的闭环失败。

## 离线通过不改变闭环判定

本次训练报告为 `wam.robofactory.m2.training_report/5`，其中
`passed=true`、`strict_reload_max_abs_difference=0.0`；teacher-context
验证也报告 `passed=true`。这些结果只证明训练产物、严格重载和离线数据路径
满足工程契约，不能替代闭环任务成功率。最终判定以闭环 Gate 为准。

## 后续与产物处置

失败 v5 checkpoint、resume 快照、训练输出、闭环视频、同系列旧归档，以及已被
取代且同样闭环失败的 v4 joint checkpoint 和输出，均在记录上述不可变身份和
指标后清理，不再占用正式候选路径。

等待以下三个独立改进分支完成训练，并使用相同任务、seed 和成功率门槛重新进行
闭环横向比较：

- `exp/lpd-agent-factorized-m2`
- `exp/lpd-static-dino-act-moe`
- `exp/lpd-temporal-ensemble`
