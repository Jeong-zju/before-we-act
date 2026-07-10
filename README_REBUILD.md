# FE-PC-WAM 数据集与模型重建说明

本文档说明如何从干净状态重新生成 Stage 2 数据集，并按阶段重新训练 FE-PC-WAM 模型。

## 1. 归档旧生成文件

先预览将会移动哪些文件：

```bash
PYTHONPATH=. python scripts/archive_generated.py
```

确认无误后，把旧生成文件移动到 `archive/<timestamp>/`：

```bash
PYTHONPATH=. python scripts/archive_generated.py --execute
```

归档脚本默认处理 `datasets/`、`checkpoints/`、`artifacts/`、`outputs/` 和 `logs/`。Python 与 pytest 缓存属于可再生成文件，会直接清理。

## 2. 重建 Stage 2 数据集

完整数据集重建：

```bash
PYTHONPATH=. python scripts/build_stage2_dataset.py --recipe full --archive-existing --cleanup-build
```

小规模烟测数据集重建：

```bash
PYTHONPATH=. python scripts/build_stage2_dataset.py --recipe smoke
```

完整 recipe 会采集：

- `scripted`: 1000 episodes, `seed_start=0`, `noise_std=0.0`
- `noisy`: 500 episodes, `seed_start=100000`, `noise_std=10.0`
- `recovery`: 100 episodes, `seed_start=200000`, `noise_std=0.0`

最终 active 数据集只保留：

- `datasets/stage2/train`
- `datasets/stage2/val`
- `datasets/stage2/test`
- `datasets/stage2/split_manifest.json`
- `datasets/stage2/dataset_manifest.json`
- `datasets/stage2/README.md`
- `datasets/stage2/diagnostics/`

最终 split 文件是真实复制出的 HDF5 文件，不是符号链接，因此中间采集目录可以安全归档。

## 3. 从零分阶段训练

完整分阶段训练：

```bash
PYTHONPATH=. python scripts/train_fe_pc_wam_pipeline.py --profile full --resume
```

小规模烟测训练：

```bash
PYTHONPATH=. python scripts/train_fe_pc_wam_pipeline.py --profile smoke --resume
```

训练总控会依次运行：

1. 数据集验证
2. Plan tokenizer 训练与导出
3. Slot encoder 训练与导出
4. 冻结 tokenizer 和 slot encoder 后训练 WAM
5. 冻结 tokenizer、slot encoder 和 WAM 后训练 intention module
6. Free-energy 评估
7. Communication 评估
8. 闭环 `no_comm`、`always_comm`、`selective_comm` policy rollout 评估

稳定推理 artifacts 会导出到：

- `artifacts/plan_tokenizer/plan_tokenizer.pt`
- `artifacts/slot_encoder/slot_encoder.pt`
- `artifacts/wam/wam.pt`
- `artifacts/intention/intention.pt`

使用 `--resume` 时，已完成阶段会跳过。如果某个阶段未完成但存在 `checkpoints/<stage>/last.pt`，总控脚本会自动把该 checkpoint 传给底层训练脚本继续训练。

## 4. 运行最终闭环推理

```bash
PYTHONPATH=. python eval/evaluate_policy.py \
  --mode selective_comm \
  --num_episodes 20 \
  --render_video 1
```

## 5. 兼容性说明

Plan tokenizer 是 `codebook_size`、`latent_dim` 和 `horizon` 的唯一基准。Slot encoder 与 WAM 训练会先读取 frozen tokenizer / slot checkpoint，并在创建新模型前自动同步依赖维度，避免 codebook 或 slot 形状不一致。
