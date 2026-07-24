# FE-PC WAM Phase M2 外部终端指令

本页只使用 RoboFactory 原生 `table` 场景和 11 个上游任务，不采集或读取 `fe_pc_wam` 自建的 `VisualRequiredEnv` 场景。所有输出目录都要求不存在或为空，避免覆盖 M1/M2 证据。

## 2026-07-24：Lift + Long 多视角 640×480 v5

本节优先于下方历史命令。v5 固定模型相机槽为
`global, agent_0, agent_1, agent_2, agent_3`。Lift 使用前三槽，Long
使用全部五槽；不存在的槽位由 validity mask 屏蔽。每个视角都有独立的
camera-slot identity 和 agent/global identity，不做跨相机平均。RoboFactory
原始帧仍是无损 `240×320`，冻结 DINOv3 在模型入口保持 4:3 比例缩放至
`480×640`。

重新采集两个任务。Lift 和 Long 仍按顺序执行；每个任务开始前根据进程
CPU affinity/cgroup quota、`MemAvailable`/cgroup memory 余量自动选择 worker。
Lift 和 Long 分别采用 2.5 GiB、4 GiB 的单 worker 内存预算，默认最多
16 个 worker、使用 80% 可用内存。worker 使用互不重叠的交错 seed lane，
并在完成后合并为单个 HDF5/JSON：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
bash scripts/collect_m2_liftbarrier_longpipeline_multiview_data.sh
```

采集脚本支持安全重跑：默认复用已有且正好包含 150 条 episode 的任务数据；
若某任务只有部分 episode，则把该任务现有的 HDF5/JSON（包括残留的 worker
分片）移入
`RoboFactory/data/m2_raw/_collection_archive/<时间戳>/<任务>/motionplanning/`
后，只重采这个未完成任务。若要强制归档并重采两个任务，设置
`M2_COLLECTION_EXISTING=archive`；若希望发现任何已有文件就停止，设置
`M2_COLLECTION_EXISTING=error`。

在归档或启动 worker 之前，脚本会先检查 `nvidia-smi` 和 SAPIEN
`cuda:0` RenderSystem。CPU 物理仿真仍需要可用的 NVIDIA/Vulkan 设备来生成
RGB；若提示 `Driver/library version mismatch`，先重启主机使内核模块与
用户态 NVIDIA 库版本一致，再原样重跑采集命令。

脚本启动时会打印 CPU limit、memory limit、最终 worker 和每 worker
线程数。通常无需覆盖；需要限制最大并行度时：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
M2_COLLECTION_MAX_PROCS=8 \
  bash scripts/collect_m2_liftbarrier_longpipeline_multiview_data.sh
```

只有需要强制指定时才使用手动覆盖：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
M2_LIFT_COLLECTION_NUM_PROCS=12 \
M2_LONG_COLLECTION_NUM_PROCS=8 \
  bash scripts/collect_m2_liftbarrier_longpipeline_multiview_data.sh
```

归档旧的两个转换目录，然后分别转换并生成训练 artifact：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
bash scripts/convert_m2_liftbarrier_longpipeline_multiview_data.sh
```

转换默认以 episode 为并行单元，自动根据 CPU affinity/cgroup quota、当前可用
内存、单 worker 内存预算和 16 worker 上限选择进程数。每个 worker 写唯一的
`episode_XXXXXX.hdf5`，父进程确定性合并 manifest 并校验文件集合。Rich
进度条按已完成 episode 实时刷新。需要限制或手动指定时：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
M2_CONVERSION_MAX_WORKERS=8 M2_LIFT_CONVERSION_NUM_WORKERS=8 \
M2_LONG_CONVERSION_NUM_WORKERS=8 \
  bash scripts/convert_m2_liftbarrier_longpipeline_multiview_data.sh
```

脚本内执行的完整转换命令如下：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
RUN_ID="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="datasets/archive/pre_multiview_${RUN_ID}"
mkdir -p "${ARCHIVE}"

if [[ -e datasets/robofactory_multitask/lift_barrier ]]; then
  mv datasets/robofactory_multitask/lift_barrier "${ARCHIVE}/"
fi
if [[ -e datasets/robofactory_multitask/long_pipeline_delivery ]]; then
  mv datasets/robofactory_multitask/long_pipeline_delivery "${ARCHIVE}/"
fi

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/convert_robofactory_dataset.py \
  --input ../RoboFactory/data/m2_raw/LiftBarrier-rf/motionplanning/LiftBarrier-rf_m2_multiview_150.h5 \
  --metadata-json ../RoboFactory/data/m2_raw/LiftBarrier-rf/motionplanning/LiftBarrier-rf_m2_multiview_150.json \
  --out-dir datasets/robofactory_multitask/lift_barrier \
  --profile m1-scratch \
  --format hdf5 \
  --fps 20 \
  --task "Lift the barrier together" \
  --task-id lift_barrier \
  --camera global \
  --camera agent_0 \
  --camera agent_1 \
  --executed-action-source command-echo \
  --episodes 150 \
  --success-only \
  --num-workers auto \
  --max-workers 16 \
  --worker-memory-mib 1024 \
  --memory-fraction 0.75 \
  --compression gzip

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/prepare_robofactory_m1_training_artifacts.py \
  --dataset-dir datasets/robofactory_multitask/lift_barrier \
  --transition-selection through-first-done-inclusive \
  --split-seed 7 \
  --expected-episodes 150 \
  --expected-state-dim 36 \
  --expected-action-dim 16 \
  --expected-task-id lift_barrier \
  --expected-camera global \
  --expected-camera agent_0 \
  --expected-camera agent_1 \
  --expected-fps 20 \
  --action-codec configs/action_codecs/robofactory_2panda_pd_joint_pos_16d.json

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/convert_robofactory_dataset.py \
  --input ../RoboFactory/data/m2_raw/LongPipelineDelivery-rf/motionplanning/LongPipelineDelivery-rf_m2_multiview_150.h5 \
  --metadata-json ../RoboFactory/data/m2_raw/LongPipelineDelivery-rf/motionplanning/LongPipelineDelivery-rf_m2_multiview_150.json \
  --out-dir datasets/robofactory_multitask/long_pipeline_delivery \
  --profile m1-scratch \
  --format hdf5 \
  --fps 20 \
  --task "Deliver the long pipeline together" \
  --task-id long_pipeline_delivery \
  --camera global \
  --camera agent_0 \
  --camera agent_1 \
  --camera agent_2 \
  --camera agent_3 \
  --executed-action-source command-echo \
  --episodes 150 \
  --success-only \
  --num-workers auto \
  --max-workers 16 \
  --worker-memory-mib 1536 \
  --memory-fraction 0.75 \
  --compression gzip

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/prepare_robofactory_m1_training_artifacts.py \
  --dataset-dir datasets/robofactory_multitask/long_pipeline_delivery \
  --transition-selection through-first-done-inclusive \
  --split-seed 7 \
  --expected-episodes 150 \
  --expected-state-dim 72 \
  --expected-action-dim 32 \
  --expected-task-id long_pipeline_delivery \
  --expected-camera global \
  --expected-camera agent_0 \
  --expected-camera agent_1 \
  --expected-camera agent_2 \
  --expected-camera agent_3 \
  --expected-fps 20 \
  --action-codec configs/action_codecs/robofactory_4panda_pd_joint_pos_32d.json
```

正式训练。`script` 保留 PTY，因此 Rich 进度条实时刷新，同时保存日志：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
bash scripts/train_m2_liftbarrier_longpipeline_multiview_640x480.sh
```

脚本内执行的训练命令如下：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
RUN_ID="$(date +%Y%m%d_%H%M%S)"
mkdir -p checkpoints/archive outputs/archive
if [[ -e checkpoints/phase_m2_liftbarrier_longpipeline_multiview_640x480_seed101 ]]; then
  mv checkpoints/phase_m2_liftbarrier_longpipeline_multiview_640x480_seed101 \
    "checkpoints/archive/phase_m2_multiview_640x480_seed101_${RUN_ID}"
fi
if [[ -e outputs/phase_m2_liftbarrier_longpipeline_multiview_640x480 ]]; then
  mv outputs/phase_m2_liftbarrier_longpipeline_multiview_640x480 \
    "outputs/archive/phase_m2_multiview_640x480_${RUN_ID}"
fi
mkdir -p outputs/phase_m2_liftbarrier_longpipeline_multiview_640x480

script -qefc \
  'CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/train_robofactory_m2.py --config configs/wam_multimodal/m2_liftbarrier_longpipeline_joint.yaml --device cuda:0 --torch-threads 16 --seed 101' \
  outputs/phase_m2_liftbarrier_longpipeline_multiview_640x480/seed101_training.log
```

训练后运行双任务闭环门控。脚本依次验证训练 seed `3000`、validation
seed `3003`、未见 seed `900–902`；每个任务要求 `1/1`、`1/1`、
至少 `2/3` 成功并保留视频：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
bash scripts/run_m2_liftbarrier_longpipeline_multiview_gate.sh
```

## 0. 当前门控：只训练和验证 LiftBarrier

多任务训练暂停。必须先让 checkpoint v2 在 LiftBarrier 上同时通过训练
seed `3000`、validation seed `3003` 和未见 seed `900–902` 的闭环门控。
v2 修复包含：

- past/target/generated action 全部使用每任务 train-only z-score；
- endpoint loss 穿过与部署共用的可微生成路径，不再使用 `x_tau` 单点外推；
- 生成 solver、clip、execution steps 和 warm-start 状态写入 checkpoint 并在推理时严格核对；
- 首轮门控固定 cold start、1-step Euler、`execution_steps=2`，先排除迭代 solver 和 warm-start 两个变量。

首次 v2 闭环在五个 seed 上均完成协同抓取、但无法持续 lift。根因是
`action_horizon=16` 的完整未来窗口约束删除了每条示范最后 15 个决策状态；
120 条训练轨迹共丢失 1800 个尾段窗口，而持续 lift 正好发生在该区间。
v3 保留这些尾段状态，用 repeat-last 只作张量填充，并用 validity mask 保证
padding 不进入 loss；实际会执行的前两步动作额外使用 `4.0` 权重。

先运行不训练模型的轻量回归测试：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
UV_CACHE_DIR=.uv-cache uv run --frozen pytest -q \
  tests/test_phase_m2_robofactory.py \
  tests/test_m1_training_checkpoint_policy.py
```

运行三步 CPU 工程 smoke（只验证数据、loss、checkpoint v2 和严格重载）：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
test ! -e checkpoints/smoke/phase_m2_v3_tail_contract
test ! -e outputs/smoke/phase_m2_v3_tail_contract.json

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/train_robofactory_m2.py \
  --config configs/wam_multimodal/m2_liftbarrier_single.yaml \
  --device cpu \
  --torch-threads 16 \
  --seed 101 \
  --smoke \
  --checkpoint checkpoints/smoke/phase_m2_v3_tail_contract \
  --report outputs/smoke/phase_m2_v3_tail_contract.json
```

正式训练 LiftBarrier 单任务 180M M2。`script -qefc` 为训练进程保留 PTY，
终端能实时刷新 Rich 进度条，同时把完整输出保存到日志：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
test ! -e checkpoints/phase_m2_liftbarrier_tailfixed_seed101
test ! -e outputs/phase_m2_liftbarrier_tailfixed/seed101_training.json
mkdir -p outputs/phase_m2_liftbarrier_tailfixed

script -qefc \
  'CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=.uv-cache uv run --frozen python scripts/train_robofactory_m2.py --config configs/wam_multimodal/m2_liftbarrier_single.yaml --device cuda:0 --torch-threads 16 --seed 101 --checkpoint checkpoints/phase_m2_liftbarrier_tailfixed_seed101 --report outputs/phase_m2_liftbarrier_tailfixed/seed101_training.json' \
  outputs/phase_m2_liftbarrier_tailfixed/seed101_training.log
```

训练结束后先检查 artifact 契约：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
jq -e '
  .passed == true and
  .strict_reload_max_abs_difference == 0 and
  .action_space == "per_task_zscore_canonical_unit_action" and
  .dataset.windows == 7904 and
  .action_generation == {
    "execution_steps": 2,
    "normalized_action_clip": 10.0,
    "solver": "euler",
    "solver_steps": 1,
    "warm_start": false
  } and
  .action_objective == {
    "executed_prefix_weight": 4.0,
    "tail_windows": "repeat_last_with_validity_masks"
  }
' outputs/phase_m2_liftbarrier_tailfixed/seed101_training.json

jq -e '
  .format_version == "wam.robofactory.m2.checkpoint/3" and
  .task_vocabulary == ["lift_barrier"] and
  .action_space == "per_task_zscore_canonical_unit_action" and
  .action_objective.executed_prefix_weight == 4.0
' checkpoints/phase_m2_liftbarrier_tailfixed_seed101/schema.json

jq -e '
  length == 1 and
  .[0].task_id == "lift_barrier" and
  (.[0].action_mean | length) == 16 and
  (.[0].action_std | length) == 16 and
  all(.[0].action_std[]; . > 0)
' checkpoints/phase_m2_liftbarrier_tailfixed_seed101/task_runtime.json
```

然后执行唯一的 LiftBarrier 闭环门控命令。它依次运行 seed `3000`、
`3003`、`900–902`，保留视频，要求 `1/1`、`1/1`、至少 `2/3` 成功且
direct-model-action coverage 为 `1`：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
bash scripts/run_m2_liftbarrier_gate.sh
```

只有上面的 gate 返回零退出码后才清理旧 M2 失败产物：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
bash scripts/cleanup_obsolete_m2_artifacts.sh
```

清理脚本会保留全部 M1 产物、LiftBarrier 数据、expert replay、
seed-contract 证据、新 checkpoint v2 和通过门控的输出；删除不可恢复。
若 gate 失败，不运行清理脚本，把最新
`outputs/phase_m2_liftbarrier_tailfixed_gate_*/gate_summary.json` 和失败视频用于下一轮诊断。

## 1. 生成 RoboFactory 原生多任务数据

```bash
cd /home/jeong/zeno/wam/RoboFactory
source ./activate_uv.sh

TASKS=(
  CameraAlignment-rf LiftBarrier-rf LongPipelineDelivery-rf PassShoe-rf
  PickMeat-rf PlaceFood-rf StackCube-rf StrikeCube-rf TakePhoto-rf
  ThreeRobotsStackCube-rf TwoRobotsStackCube-rf
)

for TASK in "${TASKS[@]}"; do
  python -m robofactory.script.generate_data \
    --scene table \
    --task "${TASK}" \
    --num 150 \
    --seed 3000 \
    --max-attempts 1500 \
    --record-dir data/m2_raw \
    --traj-name "${TASK}_m2_150"
done
```

这一步调用的是 RoboFactory 自带 planner、任务 YAML、场景 builder 和成功判据。没有 `fe_pc_wam` 自建环境参与。

上游 table YAML 的原生 agent 数为 1–4；`LongPipelineDelivery`、`TakePhoto` 各有 4 台 Panda。单臂任务的 `head_camera` 与多臂任务的 `head_camera_global` 都只做无损别名规范化为训练字段 `global`，实际视角不变。

## 2. 转换、动作 codec 和训练 manifest

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/create_robofactory_panda_action_codec.py \
  --agents 1 \
  --output configs/action_codecs/robofactory_1panda_pd_joint_pos_8d.json

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/create_robofactory_panda_action_codec.py \
  --agents 2 \
  --output configs/action_codecs/robofactory_2panda_pd_joint_pos_16d.json

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/create_robofactory_panda_action_codec.py \
  --agents 3 \
  --output configs/action_codecs/robofactory_3panda_pd_joint_pos_24d.json

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/create_robofactory_panda_action_codec.py \
  --agents 4 \
  --output configs/action_codecs/robofactory_4panda_pd_joint_pos_32d.json

SPECS=(
  'CameraAlignment-rf|camera_alignment|Align the cameras together|3'
  'LiftBarrier-rf|lift_barrier|Lift the barrier together|2'
  'LongPipelineDelivery-rf|long_pipeline_delivery|Deliver the long pipeline together|4'
  'PassShoe-rf|pass_shoe|Pass the shoe between robots|2'
  'PickMeat-rf|pick_meat|Pick the meat together|1'
  'PlaceFood-rf|place_food|Place the food together|2'
  'StackCube-rf|stack_cube|Stack the cubes together|1'
  'StrikeCube-rf|strike_cube|Strike the cube together|1'
  'TakePhoto-rf|take_photo|Take a photo together|4'
  'ThreeRobotsStackCube-rf|three_robots_stack_cube|Stack the cubes with three robots|3'
  'TwoRobotsStackCube-rf|two_robots_stack_cube|Stack the cubes with two robots|2'
)

for SPEC in "${SPECS[@]}"; do
  IFS='|' read -r ENV_ID TASK_ID TASK_TEXT AGENTS <<<"${SPEC}"
  SOURCE="../RoboFactory/data/m2_raw/${ENV_ID}/motionplanning/${ENV_ID}_m2_150"
  TARGET="datasets/robofactory_multitask/${TASK_ID}"
  STATE_DIM=$((AGENTS * 18))
  ACTION_DIM=$((AGENTS * 8))
  CODEC="configs/action_codecs/robofactory_${AGENTS}panda_pd_joint_pos_${ACTION_DIM}d.json"

  UV_CACHE_DIR=.uv-cache uv run --frozen python \
    scripts/convert_robofactory_dataset.py \
    --input "${SOURCE}.h5" \
    --metadata-json "${SOURCE}.json" \
    --out-dir "${TARGET}" \
    --profile m1-scratch \
    --format hdf5 \
    --fps 20 \
    --task "${TASK_TEXT}" \
    --task-id "${TASK_ID}" \
    --camera global \
    --executed-action-source command-echo \
    --episodes 150 \
    --success-only

  UV_CACHE_DIR=.uv-cache uv run --frozen python \
    scripts/prepare_robofactory_m1_training_artifacts.py \
    --dataset-dir "${TARGET}" \
    --transition-selection through-first-done-inclusive \
    --split-seed 7 \
    --expected-episodes 150 \
    --expected-state-dim "${STATE_DIM}" \
    --expected-action-dim "${ACTION_DIM}" \
    --expected-task-id "${TASK_ID}" \
    --expected-camera global \
    --expected-fps 20 \
    --action-codec "${CODEC}"
done
```

## 3. 训练 smoke

先用已经存在的 LiftBarrier manifest 运行 CPU 小模型。该命令验证真实 HDF5、三阶段 loss、Rich 阶段计数、保存与严格重载，不是模型质量证据：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam
RUN_ID="$(date +%Y%m%d_%H%M%S)"

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  scripts/train_robofactory_m2.py \
  --config configs/wam_multimodal/m2_causal_wam.yaml \
  --manifests datasets/robofactory_lift_barrier_m1_v1/training_manifest.json \
  --device cpu \
  --torch-threads 16 \
  --seed 101 \
  --smoke \
  --checkpoint "checkpoints/smoke/phase_m2_${RUN_ID}" \
  --report "outputs/smoke/phase_m2_${RUN_ID}.json"
```

## 4. 180M 多任务正式训练

以下依次训练 3 个初始化 seed。训练入口默认启用 BF16/TF32、task-balanced replacement sampler、16 个 DataLoader worker 上限、4 倍预取、pinned memory、异步 H2D 与持久 worker。

当前 11-task train split 的 HDF5 字段解压后总计 `148204394176` bytes（约 `138.03 GiB`），其中 current/next RGB 各约 `68.94 GiB`；因此 64 GiB 主机上的正式配置显式使用 disk-backed HDF5，并利用 worker 预取和 OS page cache，不做共享 RAM 预载。若将 `training.preload_to_ram` 改回 `true`，入口仍会在训练前按 RAM 和 `/dev/shm` 的精确 estimate/budget fail closed。

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

for SEED in 101 202 303; do
  CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=.uv-cache \
  uv run --frozen python scripts/train_robofactory_m2.py \
    --config configs/wam_multimodal/m2_causal_wam.yaml \
    --device cuda:0 \
    --torch-threads 16 \
    --seed "${SEED}" \
    --checkpoint "checkpoints/phase_m2_robofactory_multitask_seed${SEED}" \
    --report "outputs/phase_m2_robofactory_multitask/seed${SEED}_training.json"
done
```

多 GPU 单 seed 可改用（disk-backed 模式不会为每个 rank 复制完整解压数据）：

```bash
cd /home/jeong/zeno/wam/fe_pc_wam

CUDA_VISIBLE_DEVICES=0,1,2,3 UV_CACHE_DIR=.uv-cache \
uv run --frozen torchrun --standalone --nproc-per-node=4 \
  scripts/train_robofactory_m2.py \
  --config configs/wam_multimodal/m2_causal_wam.yaml \
  --torch-threads 16 \
  --seed 101 \
  --checkpoint checkpoints/phase_m2_robofactory_multitask_seed101 \
  --report outputs/phase_m2_robofactory_multitask/seed101_training.json
```

## 5. 单终端闭环 smoke

训练完成后先用未进入正式区间的 seed 900–902 跑 fast path。环境端与推理端会使用各自的 Python 环境，通过 loopback lossless RPC 连接：

```bash
cd /home/jeong/zeno/wam
RUN_ID="$(date +%Y%m%d_%H%M%S)"

(
  cd RoboFactory
  source ./activate_uv.sh
  python ../fe_pc_wam/scripts/serve_robofactory_m2_rollout.py \
    --robofactory-root . \
    --task LiftBarrier-rf \
    --scene table \
    --host 127.0.0.1 \
    --port 8872 \
    --episodes 3 \
    --seed-start 900 \
    --max-steps 500 \
    --sim-backend cpu \
    --shader default \
    --video-fps 20 \
    --output-dir "../fe_pc_wam/outputs/phase_m2_rollout_smoke_${RUN_ID}"
) &
SERVER_PID=$!

cd fe_pc_wam
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=.uv-cache \
uv run --frozen python scripts/run_robofactory_m2_inference.py \
  --checkpoint checkpoints/phase_m2_robofactory_multitask_seed101 \
  --config configs/wam_multimodal/m2_causal_wam.yaml \
  --device cuda:0 \
  --precision bf16 \
  --host 127.0.0.1 \
  --port 8872

wait "${SERVER_PID}"
```

检查 smoke 完整性：

```bash
jq -e '
  .completed == true and
  .fatal_error == null and
  .closed_loop_smoke_passed == true and
  .direct_model_action_coverage == 1
' "outputs/phase_m2_rollout_smoke_${RUN_ID}/rollout_summary.json"
```

## 6. 20-seed Gate、正式闭环与 future path 对照

下面先定义一个可直接复用的单终端函数。它会启动原生环境、运行 checkpoint 推理、等待双方正常退出，并检查 direct/no-fallback 汇总；client 若提前失败会终止等待中的 server。fast/future 只改变是否执行 future shadow branch，task、checkpoint 和 evaluation seeds 保持一致。

```bash
cd /home/jeong/zeno/wam
set -euo pipefail
RUN_ID="$(date +%Y%m%d_%H%M%S)"

run_m2_rollout() {
  local TASK="$1"
  local TRAIN_SEED="$2"
  local MODE="$3"
  local EPISODES="$4"
  local EVAL_SEED_START="$5"
  local LABEL="$6"
  local PORT=8872
  local OUTPUT_REL="outputs/phase_m2_${LABEL}_${RUN_ID}/${TASK}_train${TRAIN_SEED}_${MODE}"
  local MODE_ARGS=()
  if [[ "${MODE}" == "future" ]]; then
    MODE_ARGS+=(--future-path)
  fi

  (
    cd RoboFactory
    source ./activate_uv.sh
    python ../fe_pc_wam/scripts/serve_robofactory_m2_rollout.py \
      --robofactory-root . \
      --task "${TASK}" \
      --scene table \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --episodes "${EPISODES}" \
      --seed-start "${EVAL_SEED_START}" \
      --max-steps 500 \
      --sim-backend cpu \
      --shader default \
      --video-fps 20 \
      --output-dir "../fe_pc_wam/${OUTPUT_REL}" \
      "${MODE_ARGS[@]}"
  ) &
  local SERVER_PID=$!

  if ! (
    cd fe_pc_wam
    CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=.uv-cache \
    uv run --frozen python scripts/run_robofactory_m2_inference.py \
      --checkpoint "checkpoints/phase_m2_robofactory_multitask_seed${TRAIN_SEED}" \
      --config configs/wam_multimodal/m2_causal_wam.yaml \
      --device cuda:0 \
      --precision bf16 \
      --host 127.0.0.1 \
      --port "${PORT}"
  ); then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    return 1
  fi
  wait "${SERVER_PID}"

  jq -e --argjson episodes "${EPISODES}" '
    .completed == true and
    .fatal_error == null and
    .episodes_completed == $episodes and
    .direct_model_action_coverage == 1
  ' "fe_pc_wam/${OUTPUT_REL}/rollout_summary.json"
  jq '{task, future_path, successes, episodes_completed, success_rate,
       success_rate_wilson_95, episode_return, direct_model_action_coverage,
       formal_benchmark}' \
    "fe_pc_wam/${OUTPUT_REL}/rollout_summary.json"
}
```

先对全部 11 个原生任务、3 个训练 seed 跑未预览的 20-seed fast/future Gate：

```bash
TASKS=(
  CameraAlignment-rf LiftBarrier-rf LongPipelineDelivery-rf PassShoe-rf
  PickMeat-rf PlaceFood-rf StackCube-rf StrikeCube-rf TakePhoto-rf
  ThreeRobotsStackCube-rf TwoRobotsStackCube-rf
)

for TRAIN_SEED in 101 202 303; do
  for TASK in "${TASKS[@]}"; do
    run_m2_rollout "${TASK}" "${TRAIN_SEED}" fast 20 900 gate20
    run_m2_rollout "${TASK}" "${TRAIN_SEED}" future 20 900 gate20
  done
done
```

只有对应 task/训练 seed 的两组 20-seed Gate 完整且动作来源合规后，才对同一矩阵跑 100-seed 正式配对闭环：

```bash
for TRAIN_SEED in 101 202 303; do
  for TASK in "${TASKS[@]}"; do
    run_m2_rollout "${TASK}" "${TRAIN_SEED}" fast 100 2000 formal100
    run_m2_rollout "${TASK}" "${TRAIN_SEED}" future 100 2000 formal100
  done
done
```

正式判断以闭环 success、失败视频和 fast/future 配对结果为主；训练 loss 只用于排错。若 20-seed Gate 已明显失败，应先看 MP4 和 action-source/latency，不要继续堆到 100/500 seeds。
