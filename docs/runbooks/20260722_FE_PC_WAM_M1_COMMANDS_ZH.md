# FE-PC WAM M1 可复制指令

本文只收录 M1 命令，不包含 M0 的数据采集、审计、benchmark 或 acceptance。

当前仓库有两条可独立使用的 M1 路线：

1. **仓库内 canonical Phase M1**：在 `fe_pc_wam` 单一 Python 3.11 环境中，使用 `VisualRequiredEnv` 的三个视觉任务训练和闭环评测 DINOv3 latent WAM；
2. **RoboFactory LiftBarrier scratch M1**：先在 RoboFactory Python 3.9 环境中生成/运行仿真，再在 `fe_pc_wam` Python 3.11 环境中转换、训练和推理。

命令按当前工作区 `/home/jeong/zeno/wam` 编写。数据集、checkpoint 和正式评测产物都应采用新目录；不要覆盖已经通过验收的证据。

## 1. 共用环境和 DINOv3 工件

进入 FE-PC WAM 环境：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
uv sync --frozen
uv run --frozen python --version
```

两条 M1 路线使用同一份固定 DINOv3 工件。先校验本地配置和权重：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

printf '%s  %s\n' \
  ce962b0c8ca4f2deb48c6fdfd6035257e3769f1d4d9154c92aba51991e46e290 \
  artifacts/vision/dinov3_vitl16_lvd/config.json \
  dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179 \
  artifacts/vision/dinov3_vitl16_lvd/model.safetensors \
  | sha256sum -c -
```

若工件尚未准备，在已获得 gated model 权限并完成 Hugging Face 登录后执行：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

uv run --frozen python scripts/prepare_dinov3_encoder.py \
  --encoder dinov3_vitl16_lvd \
  --output-dir artifacts/vision/dinov3_vitl16_lvd
```

---

# 路线 A：仓库内 canonical Phase M1

这条路线只使用 `fe_pc_wam` 环境，不跨 Python 环境。三个 M1 任务为：

- `visual_event_stop`
- `visual_target_select`
- `visual_obstacle_avoid`

策略输入是 22D proprioception、8D action 和 `fixed` 相机 RGB。canonical 配置为 `configs/wam_multimodal/m1_latent_wam_dinov3.yaml`。

## A1. M1 输入数据与初始化检查

M1 直接读取已经固定的多模态 manifest，不执行数据转换。这里只检查 M1 的输入依赖，不运行任何 M0 命令：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

test -f datasets/mujoco_visual_required_wam_multimodal_m0/manifest.json
test -d checkpoints/joint_wam

printf '%s  %s\n' \
  d0e1289035286db2bf64a7aca63cf767e04b92eeda582877723f8fc1ba5d1c08 \
  datasets/mujoco_visual_required_wam_multimodal_m0/manifest.json \
  | sha256sum -c -
```

运行 M1 专用 preflight。它会校验 manifest/HDF5、legacy Joint WAM、DINOv3、数据窗口、因果 pair、256-sample overfit 和 1% 训练链路。使用时间戳隔离诊断产物：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

RUN_ID="$(date +%Y%m%d_%H%M%S)"

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run --frozen python scripts/train_multimodal_wam.py \
  --config configs/wam_multimodal/m1_latent_wam_dinov3.yaml \
  --device cuda:0 \
  --torch-threads 16 \
  --preflight-only \
  --checkpoint-root "checkpoints/preflight/phase_m1_dinov3_${RUN_ID}" \
  --output-root "outputs/preflight/phase_m1_dinov3_${RUN_ID}"
```

该命令是 M1 preflight，不是正式训练证据。

## A2. 仓库内 M1 训练 smoke

以下命令只训练 `state_vision_future/seed_101`，并将三个 stage 缩放到 1%，用于验证完整 M1 训练和 checkpoint 保存链路：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

RUN_ID="$(date +%Y%m%d_%H%M%S)"

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run --frozen python scripts/train_multimodal_wam.py \
  --config configs/wam_multimodal/m1_latent_wam_dinov3.yaml \
  --device cuda:0 \
  --torch-threads 16 \
  --variants state_vision_future \
  --seeds 101 \
  --steps-scale 0.01 \
  --checkpoint-root "checkpoints/preflight/phase_m1_smoke_${RUN_ID}" \
  --output-root "outputs/preflight/phase_m1_smoke_${RUN_ID}"
```

## A3. 仓库内 M1 正式训练

正式训练运行 5 个 variant × 3 个训练 seed：

- `state_only`
- `vision_only`
- `state_vision_no_future`
- `state_vision_future`
- `state_vision_param_matched_mlp`

canonical 输出为 `checkpoints/phase_m1_dinov3` 和 `outputs/phase_m1_dinov3/training`。全新正式运行前执行：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

test ! -e checkpoints/phase_m1_dinov3 || {
  echo "canonical M1 checkpoint 已存在；拒绝覆盖" >&2
  exit 1
}

test ! -e outputs/phase_m1_dinov3 || {
  echo "canonical M1 输出已存在；拒绝覆盖" >&2
  exit 1
}

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run --frozen python scripts/train_multimodal_wam.py \
  --config configs/wam_multimodal/m1_latent_wam_dinov3.yaml \
  --device cuda:0 \
  --torch-threads 16
```

如果 canonical 正式训练曾正常写出部分 checkpoint/report 后被中断，只使用入口自带的严格 resume，不手工拼接产物：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run --frozen python scripts/train_multimodal_wam.py \
  --config configs/wam_multimodal/m1_latent_wam_dinov3.yaml \
  --device cuda:0 \
  --torch-threads 16 \
  --resume
```

检查正式训练摘要：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

jq -e '
  .formal_protocol == true and
  .passed == true
' outputs/phase_m1_dinov3/training/training_summary.json
```

## A4. 仓库内 M1 正式闭环评测

该入口在仓库内 `VisualRequiredEnv` 上运行正式 clean/intervention 闭环评测。它会自动先跑 smoke gate，再覆盖 100 个 physical seeds、2 个 cue、5 个 variant 和 3 个训练 seed：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

test ! -e outputs/phase_m1_dinov3/evaluation/visual_required_episodes.jsonl || {
  echo "正式 M1 episode 记录已存在；拒绝混入旧证据" >&2
  exit 1
}

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run --frozen python scripts/evaluate_multimodal_wam.py \
  --config configs/wam_multimodal/m1_latent_wam_dinov3.yaml \
  --device cuda:0 \
  --torch-threads 16
```

检查闭环评测结果：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

jq -e '
  .formal_protocol == true and
  .passed == true
' outputs/phase_m1_dinov3/evaluation/visual_required_evaluation.json

jq '{
  records,
  expected_records,
  aggregation,
  runtime
}' outputs/phase_m1_dinov3/evaluation/visual_required_evaluation.json
```

## A5. 仓库内 M1 future probe

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

test ! -e outputs/phase_m1_dinov3/evaluation/future_probe.json || {
  echo "正式 future probe 已存在；拒绝覆盖" >&2
  exit 1
}

CUDA_VISIBLE_DEVICES=0 \
uv run --frozen python scripts/evaluate_m1_future_probe.py \
  --config configs/wam_multimodal/m1_latent_wam_dinov3.yaml \
  --device cuda:0 \
  --batch-size 64 \
  --torch-threads 16

jq -e '
  .formal_protocol == true and
  .passed == true
' outputs/phase_m1_dinov3/evaluation/future_probe.json
```

## A6. 仓库内 M1 legacy regression

该命令对 standard/challenge 各运行 500 个 seed，验证 M1 没有破坏原 proprioceptive cooperative-stop 能力：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

test ! -e outputs/phase_m1_dinov3/evaluation/legacy_regression.json || {
  echo "正式 legacy regression 已存在；拒绝覆盖" >&2
  exit 1
}

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run --frozen python scripts/evaluate_m1_legacy.py \
  --config configs/wam_multimodal/m1_latent_wam_dinov3.yaml \
  --device cuda:0 \
  --torch-threads 16

jq -e '
  .formal_protocol == true and
  .passed == true
' outputs/phase_m1_dinov3/evaluation/legacy_regression.json
```

## A7. 仓库内 Gate M1 最终验收

只有训练、visual-required 闭环、future probe 和 legacy regression 全部完成后，才运行最终 acceptance：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

test ! -e outputs/phase_m1_dinov3/phase_m1_acceptance.json || {
  echo "正式 M1 acceptance 已存在；拒绝覆盖" >&2
  exit 1
}

uv run --frozen python scripts/accept_phase_m1.py \
  --config configs/wam_multimodal/m1_latent_wam_dinov3.yaml \
  --device cpu \
  --torch-threads 16

jq -e '
  .formal_protocol == true and
  .technical_checks_passed == true and
  .core_gate_passed == true and
  .bundle_checks_passed == true and
  .passed == true and
  .claim_allowed == true
' outputs/phase_m1_dinov3/phase_m1_acceptance.json
```

## A8. 仓库内 M1 诊断 rollout 与 MP4

该命令不改写正式 acceptance，使用正式 DINOv3 `state_vision_future` checkpoint 运行新的诊断 seeds 并录制 MP4：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

RUN_ID="$(date +%Y%m%d_%H%M%S)"

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run --frozen python scripts/rollout_multimodal_wam.py \
  --config configs/wam_multimodal/m1_latent_wam_dinov3.yaml \
  --device cuda:0 \
  --torch-threads 16 \
  --train-seeds 101 202 303 \
  --tasks visual_event_stop visual_target_select visual_obstacle_avoid \
  --physical-seeds 10 \
  --cue-variants 0 1 \
  --video-episodes-per-task 2 \
  --video-width 640 \
  --video-height 360 \
  --output-dir "outputs/phase_m1_dinov3/rollouts/diagnostic_${RUN_ID}"
```

查看诊断成功率：

```bash
jq '.aggregation | {overall, by_task, by_train_seed, by_cue}' \
  /home/jeong/zeno/wam/fe_pc_wam/outputs/phase_m1_dinov3/rollouts/diagnostic_*/rollout_summary.json
```

---

# 路线 B：RoboFactory LiftBarrier scratch M1

这条路线使用 36D state、16D 双 Panda `pd_joint_pos` action 和 global RGB。数据生成/环境端在 RoboFactory Python 3.9 中运行；转换、训练和推理端在 FE-PC WAM Python 3.11 中运行。

## B1. 生成 RoboFactory 数据集

```bash
cd /home/jeong/zeno/wam/RoboFactory
source ./activate_uv.sh

python robofactory/script/generate_data.py \
  --config robofactory/configs/table/lift_barrier.yaml \
  --num 150 \
  --seed 3 \
  --traj-name lift_barrier_seed3_n150 \
  --record-dir demos/lift_barrier_rgb_video_seed3_n150 \
  --max-attempts 750 \
  --video-fps 20 \
  --save-video
```

确认生成批次完整，再安装到 canonical 源路径。命令不会覆盖已有源数据：

```bash
cd /home/jeong/zeno/wam/RoboFactory

SOURCE_DIR="demos/lift_barrier_rgb_video_seed3_n150/LiftBarrier-rf/motionplanning"

test -f "${SOURCE_DIR}/lift_barrier_seed3_n150.h5" && \
test -f "${SOURCE_DIR}/lift_barrier_seed3_n150.json" && \
jq -e '
  (.episodes | length) == 150 and
  .episodes[0].episode_seed == 3 and
  ([.episodes[].success] | all)
' "${SOURCE_DIR}/lift_barrier_seed3_n150.json" || {
  echo "生成批次不完整" >&2
  exit 1
}

mkdir -p data/h5_data

test ! -e data/h5_data/LiftBarrier-rf.h5 && \
test ! -e data/h5_data/LiftBarrier-rf.json || {
  echo "canonical 源数据已存在；拒绝覆盖" >&2
  exit 1
}

cp "${SOURCE_DIR}/lift_barrier_seed3_n150.h5" \
  data/h5_data/LiftBarrier-rf.h5
cp "${SOURCE_DIR}/lift_barrier_seed3_n150.json" \
  data/h5_data/LiftBarrier-rf.json
```

## B2. 转换并准备 LiftBarrier M1 数据

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

test ! -e datasets/robofactory_lift_barrier_m1_v1 || {
  echo "目标数据集已存在；请保留现有产物或改用新的版本化目录" >&2
  exit 1
}

uv run --frozen python scripts/convert_robofactory_dataset.py \
  --input ../RoboFactory/data/h5_data/LiftBarrier-rf.h5 \
  --metadata-json ../RoboFactory/data/h5_data/LiftBarrier-rf.json \
  --out-dir datasets/robofactory_lift_barrier_m1_v1 \
  --profile m1-scratch \
  --format hdf5 \
  --camera global \
  --fps 20 \
  --task-id lift_barrier \
  --task "Lift the barrier together" \
  --executed-action-source command-echo \
  --canonical-only \
  --compression gzip

uv run --frozen python scripts/prepare_robofactory_m1_training_artifacts.py \
  --dataset-dir datasets/robofactory_lift_barrier_m1_v1 \
  --transition-selection through-first-done-inclusive
```

如果只重建已有转换数据的 manifest/normalization，增加 `--overwrite`：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

uv run --frozen python scripts/prepare_robofactory_m1_training_artifacts.py \
  --dataset-dir datasets/robofactory_lift_barrier_m1_v1 \
  --transition-selection through-first-done-inclusive \
  --overwrite
```

全量校验 train/validation/test 和首尾 M1 window：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

uv run --frozen python scripts/smoke_m1_data_protocol.py \
  --manifest datasets/robofactory_lift_barrier_m1_v1/training_manifest.json \
  --splits train validation test \
  --state-history 32 \
  --action-chunk 8 \
  --visual-history 2 \
  --future-horizons 1 2 4 8 \
  --camera global
```

## B3. LiftBarrier M1 训练 smoke

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

SMOKE_OUTPUT="checkpoints/preflight/m1_liftbarrier_$(date +%Y%m%d_%H%M%S)"

uv run --frozen python scripts/train_liftbarrier_m1_scratch.py \
  --config configs/wam_multimodal/m1_liftbarrier_scratch.yaml \
  --device cuda:0 \
  --steps-scale 0.001 \
  --output "${SMOKE_OUTPUT}"
```

## B4. LiftBarrier M1 正式训练

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

test ! -e checkpoints/m1_liftbarrier_scratch_seed101 || {
  echo "正式 LiftBarrier checkpoint 已存在；拒绝覆盖" >&2
  exit 1
}

uv run --frozen python scripts/train_liftbarrier_m1_scratch.py \
  --config configs/wam_multimodal/m1_liftbarrier_scratch.yaml \
  --device cuda:0
```

## B5. LiftBarrier M1 不跨环境验证

这里的“不跨环境”只验证数据、代码和 checkpoint 契约，不启动 RoboFactory，因此不能给出真实闭环成功率。

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

uv run --frozen pytest -q \
  tests/test_m1_scratch_path.py \
  tests/test_m1_manifest_dataset.py \
  tests/test_robofactory_m1_training_artifacts.py \
  tests/test_robofactory_m1_closed_loop.py
```

校验 checkpoint schema 和所有内含工件 hash：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam/checkpoints/m1_liftbarrier_scratch_seed101

jq -e '
  .format_version == "wam.multimodal.m1.scratch_checkpoint/1" and
  .initialization_mode == "scratch" and
  .train_seed == 101 and
  .state_dim == 36 and
  .action_dim == 16 and
  .action_domain == "canonical_unit_action" and
  .frozen_visual_backbone == true and
  .action_anchor_mode == "none" and
  .legacy_weight_files == []
' schema.json

jq -r '
  .artifact_sha256
  | to_entries[]
  | "\(.value)  \(.key)"
' schema.json | sha256sum -c -
```

严格重建并重载 DINO、latent WAM、action-flow、normalization 和 action codec：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

uv run --frozen python - <<'PY'
from pathlib import Path
import json
import yaml

from scripts.run_robofactory_m1_inference import _build_vision_encoder
from train.m1_scratch_checkpointing import (
    load_scratch_m1_checkpoint,
    scratch_checkpoint_tree_sha256,
)

config_path = Path("configs/wam_multimodal/m1_liftbarrier_scratch.yaml")
checkpoint_path = Path("checkpoints/m1_liftbarrier_scratch_seed101")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
vision = _build_vision_encoder(config["vision"])
_, metadata = load_scratch_m1_checkpoint(
    checkpoint_path,
    vision_encoder=vision,
    device="cpu",
)
print(json.dumps({
    "passed": True,
    "checkpoint_tree_sha256": scratch_checkpoint_tree_sha256(checkpoint_path),
    "schema": metadata["schema"],
    "stage_state": metadata["stage_state"],
}, indent=2, sort_keys=True))
PY
```

## B6. LiftBarrier 跨环境闭环 smoke

终端 1 使用 RoboFactory Python 3.9，运行 seeds 900–902：

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

等待终端 1 显示 `waiting for the M1 inference client`。终端 2 使用 FE-PC WAM Python 3.11：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

uv run --frozen python scripts/run_robofactory_m1_inference.py \
  --checkpoint checkpoints/m1_liftbarrier_scratch_seed101 \
  --config configs/wam_multimodal/m1_liftbarrier_scratch.yaml \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8765
```

检查 smoke：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

jq -e '
  .completed == true and
  .fatal_error == null and
  .episodes_completed == 3
' outputs/robofactory_m1_closed_loop_smoke_seed900_n3/rollout_summary.json

jq '{successes, episodes_completed, success_rate, success_rate_wilson_95}' \
  outputs/robofactory_m1_closed_loop_smoke_seed900_n3/rollout_summary.json
```

## B7. LiftBarrier 跨环境正式 benchmark

终端 1 使用 100 个未见 seeds（1000–1099），并保存 100 个 MP4：

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

终端 2：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

uv run --frozen python scripts/run_robofactory_m1_inference.py \
  --checkpoint checkpoints/m1_liftbarrier_scratch_seed101 \
  --config configs/wam_multimodal/m1_liftbarrier_scratch.yaml \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8765
```

只有下列门禁通过，成功率才可作为正式结果报告：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

jq -e '
  .formal_benchmark.reportable == true and
  .completed == true and
  .fatal_error == null and
  .episodes_completed == 100 and
  .episodes_completed == .benchmark_protocol.episodes_requested
' outputs/robofactory_m1_closed_loop_seed1000_n100/rollout_summary.json

jq '{successes, episodes_completed, success_rate, success_rate_wilson_95}' \
  outputs/robofactory_m1_closed_loop_seed1000_n100/rollout_summary.json
```

partial summary 和 `_aborted.mp4` 只用于排障，不能报告其中的部分成功率。
