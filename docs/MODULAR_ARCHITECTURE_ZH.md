# 模块化仿真与数据管线

## 1. 已落地的边界

本次重构把核心依赖固定为单向关系：

```text
models/                         envs/
  纯张量计算                      环境、runner、实时节拍、RGB/MP4
  不读取数据集/仿真状态             不导入 data/models
       ▲                            │
       │                            ▼
       └── application adapters   data/
                                   字段契约、采集适配、HDF5/LeRobot 导出
                                   允许依赖 envs 的公共 runtime 契约
```

边界由 `tests/test_modular_architecture.py` 的 AST import guard 自动检查：

- `models/` 禁止导入 `data`、`envs`；
- `envs/` 禁止导入 `data`、`models`；
- `data/` 可以导入 `envs.runtime`，但 exporter 只消费
  `SimulationTransition`，不绑定具体 MuJoCo 类；
- `scripts/` 是应用组合层，可以组合环境与数据导出；模型适配器也应放在这一层。

历史版本的模型、训练、评测、采集脚本和文档已经删除。当前仓库只保留这套模块化
契约，不提供旧接口兼容层。

## 2. 仿真环境独立运行

环境公共契约位于 `envs/runtime.py`：

- `SimulationEnvironment`：`reset / step / render / close / control_dt`；
- `Policy`：只接收当前 observation；
- `SimulationTransition`：严格表示
  `observation[t] -> action[t] -> observation[t+1]`；
- `SimulationRunner`：批量运行和 wall-clock 实时运行共用同一个执行器；
- `RolloutObserver`：viewer、视频、数据集都是旁路消费者，环境不反向依赖它们。

独立批量运行：

```bash
python -m envs.run --scenario nominal --episodes 3
```

按环境 `control_dt` 实时运行，并打开 MuJoCo viewer：

```bash
python -m envs.run --scenario private_gates --realtime --viewer
```

离屏 RGB 流式写 MP4（逐帧编码，不缓存整条 episode）：

```bash
MUJOCO_GL=egl python -m envs.run \
  --realtime --camera fixed --camera topdown \
  --video outputs/live_fixed.mp4
```

`TwoRobotCarryNarrowPassageEnv.render()` 使用按分辨率懒创建的 MuJoCo renderer；
`close()` 会释放全部 renderer context。Gym wrapper 同时支持
`render_mode="rgb_array"`。

## 3. 格式无关的数据契约

`data/trajectory.py` 将“采什么”与“存成什么”拆开：

- `FieldSpec(name, source, dtype, required)` 描述一个字段；
- `TrajectorySchema` 解析字段但不关心 HDF5 / Parquet / MP4；
- source 使用稳定的 dotted path，例如：
  `observation.global_state`、`next_observation.object`、`info.progress`、
  `images.fixed`；
- exporter 接收同一个 transition stream，可以同时输出多个格式。

默认 profile：

| Profile | 默认用途 | 关键默认字段 |
|---|---|---|
| `vla` | 通用 VLA / LeRobot | `observation.state`、`observation.images.*`、`action`、`task`、`timestamp`、reward/done |
| `wam` | World Action Model | 双 agent 当前/下一状态、object/global state、action、reward/done、progress/force |
| `robocasa` | RoboCasa 风格 VLA | 使用当前 RoboCasa 的 LeRobot 主结构，保留 task、state、action、多相机视频 |
| `rmbench` | RMBench / RoboTwin 风格 | 单 episode 轨迹、双 agent state、next state、action、instruction、success/progress |

[事实] RoboCasa 1.0.1 的训练数据主体使用 LeRobot 结构，并另外保存
MuJoCo replay extras；RMBench 基于 RoboTwin 2.0，后者按 episode 保存 HDF5，图像可保存为
bit stream，同时分开保存 instructions 和视频。这里借鉴的是稳定字段语义，不声称逐字节兼容
所有上游私有版本。

## 4. 导出后端

### 4.1 HDF5

`HDF5TrajectoryExporter` 每个 episode 写一个文件：

```text
hdf5/
├── episode_000000.hdf5
│   ├── attrs: format_version, profile, fps, task, seed, num_steps, ...
│   ├── schema/...              # 每个字段的 source / dtype / required
│   └── data/...
└── videos/episode_000000/*.mp4 # --stream-video 时生成
```

特性：

- extendable/chunked dataset，逐 transition append；
- `.partial.hdf5` 完成后原子改名，异常中断不会伪装成完整 episode；
- dtype 显式保真，shape 在第一帧确定，后续变化会立即失败；
- RGB 可同时保存原始数组并实时编码 MP4；
- 一个文件一个 episode，便于 RMBench/RoboTwin 式并行采集和坏样本隔离。

### 4.2 LeRobotDataset v3

`LeRobotTrajectoryExporter` 不重写 LeRobot 的 Parquet/MP4 细节，而是调用官方
`LeRobotDataset.create -> add_frame -> save_episode -> finalize` API。启用
`streaming_encoding=True` 时视频在采集过程中编码；`finalize()` 确保 Parquet footer
与元数据落盘。

LeRobot 是可选依赖；未安装时 HDF5 和仿真仍可独立工作：

```bash
pip install -r requirements-dataset-export.txt
```

## 5. 采集命令

同一 rollout 同时导出 HDF5 与 LeRobot：

```bash
MUJOCO_GL=egl python scripts/collect_modular_dataset.py \
  --out-dir datasets/modular_carry \
  --format hdf5 --format lerobot \
  --profile vla \
  --episodes 100 \
  --camera fixed --camera topdown \
  --stream-video \
  --repo-id local/modular-carry
```

WAM 默认字段、无视频的轻量采集：

```bash
python scripts/collect_modular_dataset.py \
  --out-dir datasets/wam_stream \
  --format hdf5 --profile wam --episodes 100
```

命令结束后根目录会写 `manifest.json`，记录实际格式、fps、字段映射和每个 episode
结果。

## 6. 自定义字段

命令行覆盖：

```bash
python scripts/collect_modular_dataset.py \
  --out-dir datasets/custom --format hdf5 --profile wam \
  --drop-field force \
  --field diagnostics.robot_distance=info.robot_distance::float32 \
  --field labels.communication_required=info.communication_required::bool
```

同名 `--field` 会替换 profile 默认映射。完整自定义可通过 `--schema-json`：

```json
{
  "profile": "custom_vla",
  "fields": [
    {"name": "observation.state", "source": "observation.global_state", "dtype": "float32"},
    {"name": "action", "source": "action", "dtype": "float32"},
    {"name": "task", "source": "task"},
    {"name": "done", "source": "done", "dtype": "bool"}
  ]
}
```

## 7. 取舍与证伪条件

Bull：单向 import guard + format-neutral transition stream 让新增环境、新相机或新 exporter
不再改模型；同一 rollout 多格式 fan-out 也避免重复仿真造成的数据漂移。

Bear：LeRobot 的磁盘格式和 writer API 仍由上游演进；本环境尚未安装可选 LeRobot
依赖，因此仓库内验证覆盖 API contract fake 和 HDF5 真实端到端，不能冒充已完成本机
LeRobot loader round-trip。当前参考模型是低维 WAM，RGB 已能采集但尚未接入视觉 encoder。

Arbiter：当前采集统一使用模块化入口。安装可选依赖后，必须把
“官方 LeRobot loader 能读取实际输出、episode/frame/video 数严格一致”作为发布门禁。

可证伪条件：

- 若在 **2026-07-28** 前发现模型或环境包需要反向导入 dataset 才能实现新功能，说明
  当前公共契约不足，应先扩展 protocol，不能放松依赖方向；
- 若在 **2026-08-14** 前 LeRobot 官方 loader round-trip 仍不能通过，则
  `LeRobotTrajectoryExporter` 应标记 experimental，不能宣称格式支持完成；
- 若 RGB 帧数、transition 行数、video frame count 任一不一致，流式采集设计立即判失败，
  不允许靠训练端截断来掩盖。

## 8. 上游格式依据

- LeRobotDataset v3：<https://huggingface.co/docs/lerobot/lerobot-dataset-v3>
- RoboCasa Using Datasets：<https://robocasa.ai/docs/build/html/datasets/using_datasets.html>
- RoboTwin 2.0 Collect Data：<https://robotwin-platform.github.io/doc/usage/collect-data.html>
- RMBench：<https://github.com/RoboTwin-Platform/RMBench>
