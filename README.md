# WAM Modular Runtime

面向 VLA / World Action Model 实验的模块化仿真与数据基础设施。模型、仿真环境和
数据集是三个独立模块，依赖方向由测试自动约束。

```text
models/   只接收显式张量输入；不导入环境或数据集
envs/     MuJoCo 环境、独立 runner、实时节拍、RGB/MP4 流
data/     字段契约、HDF5/LeRobot 流式导出；可消费 envs 的 rollout
scripts/  组合环境与数据导出的应用入口
```

## 仿真独立运行

批量运行：

```bash
python -m envs.run --scenario nominal --episodes 3
```

按环境控制周期实时运行：

```bash
python -m envs.run --scenario private_gates --realtime --viewer
```

流式写视频，不在内存中缓存整条 episode：

```bash
MUJOCO_GL=egl python -m envs.run \
  --realtime --camera fixed --video outputs/live_fixed.mp4
```

## 数据集采集与导出

HDF5 每个 episode 一个文件，采用可扩展 chunk 逐 transition 写入：

```bash
python scripts/collect_modular_dataset.py \
  --out-dir datasets/wam_stream \
  --format hdf5 --profile wam --episodes 100
```

同一次 rollout 可以同时导出 HDF5 和 LeRobot：

```bash
MUJOCO_GL=egl python scripts/collect_modular_dataset.py \
  --out-dir datasets/modular_carry \
  --format hdf5 --format lerobot \
  --profile vla \
  --camera fixed --camera topdown \
  --stream-video
```

默认字段 profile 为 `vla`、`wam`、`robocasa`、`rmbench`。通过
`--field NAME=SOURCE::DTYPE` 添加或替换字段，通过 `--drop-field NAME` 删除字段，
也可以使用 `--schema-json` 定义完整 schema。

LeRobot 是可选依赖；HDF5 与仿真不依赖它：

```bash
pip install -r requirements-dataset-export.txt
```

## 模型边界

`WorldActionModel` 只执行显式输入的一步预测，不访问环境、采集器或磁盘：

```python
import torch

from models import WorldActionModel, WorldActionModelConfig, WorldModelInputs

model = WorldActionModel(WorldActionModelConfig(state_dim=32, action_dim=8))
prediction = model(
    WorldModelInputs(
        state=torch.zeros(4, 32),
        action=torch.zeros(4, 8),
    )
)
```

VLA policy 可实现 `PolicyInputs -> PolicyOutput` 契约；图像、语言 token、mask 和上下文
都必须由调用方显式传入。

## 验证

```bash
pytest -q
ruff check .
black --check .
```

架构边界、字段布局、上游格式依据和证伪条件见
[模块化架构设计](docs/MODULAR_ARCHITECTURE_ZH.md)。
