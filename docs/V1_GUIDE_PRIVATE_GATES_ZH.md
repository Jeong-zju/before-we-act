# FE-PC-WAM V1 Private Gates：重新采集与验收

该版本与旧数据、旧归一化统计和旧 checkpoint 不兼容。每个 episode 包含三个
协作决策门，事件按 `decisive-private / locally-inferable / redundant` 平衡生成。
决定性事件只有一台机器人能在 deployable packet 中看到有效 cue；另一台机器人的
cue 必须为全零且 `valid=0`。消息仍然只有 plan code、residual 和 envelope metadata。

## 1. 采集

先只跑 pilot：

```bash
python scripts/collect_fe_pc_wam_dataset.py \
  --out-dir datasets/private_gates_v1 \
  --pilot-only
```

pilot 专家成功率低于 95% 时命令失败，不会继续正式采集。修复任务或专家后，应使用
新的空目录重新采集。正式默认规模为 train 2400、val 400、test 400；三个 split 的
seed offset 在采集前固定，test 不参与训练、校准或通信成本选择。

pilot 通过后以 `--resume` 复用并重新审计 pilot，再继续正式 split：

```bash
python scripts/collect_fe_pc_wam_dataset.py \
  --out-dir datasets/private_gates_v1 \
  --resume
```

每个事件第一次进入决策区时，从同一 simulator snapshot 执行六组 plan pair 分支，
保存动作段、return、success 和 constraint violation。主轨迹随后从原 snapshot 继续，
分支执行不会推进主轨迹 RNG 或动力学状态。

## 2. 从零训练

```bash
python scripts/train_fe_pc_wam_pipeline.py \
  --dataset-root datasets/private_gates_v1 \
  --out-dir checkpoints/private_gates_v1
```

顺序仍为 `plan → belief → wam → intention → wam_robust`。Plan Tokenizer 增加
left/hold/right 辅助监督；WAM 增加 step reward、return quantiles、success/failure、
collision/force risk 和 completion time heads，并用分支数据训练候选排序。

## 3. 验证集通信成本冻结

将不同 bit/delay 价格的 validation 结果整理为：

```json
{"split":"validation","points":[
  {"bit_price":0.0001,"delay_price":0.05,"success_rate":0.9,"bits_per_episode":100}
]}
```

然后运行：

```bash
python scripts/select_communication_cost.py \
  --validation-sweep outputs/private_gates/validation_sweep.json \
  --output outputs/private_gates/pareto_selection.json
```

选择器删除被支配点，并选择归一化成功损失—通信量空间中距理想点最近的工作点。
测试集只能使用冻结结果，不允许按测试结果回调成本。

## 4. 关键证据

- `task/private_event_*` 只包含本机实际观察到的 cue；无效 cue 必须为零。
- `/privileged/*private_event*` 只作为标签与审计信息。
- `G` 默认来自校准后的 return/risk heads；旧手工 G 以 `G_legacy` 保留。
- paired acceptance 要求 Selective 对 no-comm 的成功率差 95% CI 下界大于零、
  相对 always-reply 非劣 5 个百分点、通信量不超过 always-reply 的 50%，并且必要
  事件请求率高于冗余事件。
