# S10 `core` 用户数据复现与接入记录

> 状态：已完成复现并接入 `feat/model-improvements`
> 复现日期：2026-08-03
> 接入日期：2026-08-04
> 方法名：`core`（Stereo-CoRE 的无腕部相机适配）

## 1. 接入结论

S10 不再从原 R7/R8/R9 结构继续搭建起点，而是直接使用已经在用户自有服务器完成训练和五任务冻结评测的 `core`。[官方 Stereo-CoRE release](https://github.com/YananZHOU5555/Stereo-CoRE) 以 subtree 形式完整保存在 `vendor/stereo-core/`，固定上游提交为 `f60995c082a18cc849fcf3537ac4b89f1ac9b19f`，许可证为 MIT；用户数据适配的模型、训练器、评测器和部署脚本也保存在同一目录。

本次只建立一个可审计的 S10 起点：没有创建或推送额外 S10 分支，也没有在接入时选择新结构。后续多分支渐进改动由用户从这次提交自行开始。

## 2. 用户数据与模型契约

训练使用用户自己的五任务 HDF5 数据，而不是同事的数据：

| 任务 | training manifest SHA256 |
|---|---|
| LiftBarrier | `a4180b2730c1ca5bbe8f28359f91bad9575a60197698c7ae650a54b14612aeb0` |
| CameraAlignment | `909379b070286e7b1eda90623fef5cafbc712007fd89f071b5b579ae1415a420` |
| ThreeRobotsStackCube | `89052e2746cc6e0cea4ff32013b3c10d8104624cfe285e620b82e48b13b4c99c` |
| LongPipelineDelivery | `35ce94c4cbe121c2fbc1013d8a5366a2c4799c3ce78f46baf3e12fc00b2b7114` |
| TakePhoto | `d382dd6f99964770a600d8b13da9c3cf55a69a20a54a15b6e78d6b9f6108003f` |

每任务使用 120 条训练 episode，共 600 条。每个 batch 固定为 40 个本地样本，即每任务精确 8 个；正式预算为 120,000 optimizer updates、4.8M local action chunks，随机种子为 `20260803`。动作与状态归一化只由上述用户训练集重新计算。

策略输入严格为当前 640×480 global fixed RGB、对应机器人的 640×480 fixed RGB 与 own qpos。它不读取腕部 RGB/深度、task/agent ID、语言、peer state/action 或真实未来。模型保留冻结 DINOv3-B/16、30×40 cross-relative-bias 双视图融合、ACT 4 层 posterior/7 层 decoder、4 个 rank-32 role adapter，以及 capability-only top-2 CoRE routing。

## 3. 已完成结果

冻结 `frozen100` 协议对每任务评测 100 个 seed，单回合最多 1500 steps：

| 任务 | 成功数 | SR@1 |
|---|---:|---:|
| LiftBarrier | 100/100 | 100% |
| CameraAlignment | 60/100 | 60% |
| ThreeRobotsStackCube | 0/100 | 0% |
| LongPipelineDelivery | 100/100 | 100% |
| TakePhoto | 97/100 | 97% |
| 五任务宏平均 | 357/500 | **71.4%** |

机器可读结果保存在 [20260804_S10_CORE_USER_DATA_FROZEN100.json](20260804_S10_CORE_USER_DATA_FROZEN100.json)。`ThreeRobotsStackCube=0%` 是后续 S10 的首要失败点，但本次接入不据此预选任何改进结构。由于这里的固定第三人称 RGB 和用户数据协议与同事发布结果的腕部 RGB-D/数据协议不同，两组成功率不能作为同条件方法优劣比较。

## 4. 产物与来源绑定

| 产物 | SHA256 |
|---|---|
| `checkpoint_120000.pt` | `061b7a4acea8fa10f146779e7a1206822179920dfe573db536d237df81eb541d` |
| `config.json` | `cb330d494a3a20e4108f1e68859d0ef96805d8afd9392ae5a06c81efde3a4f96` |
| `frozen100/summary.json` | `2e44e2fbf54c86b7884c2234de86a0095e27651fdcf7bf8c65529d6aa46458af` |

checkpoint 大小为 734,197,493 bytes，仍保存在用户自有复现服务器的 `/workspace/runs/no_wrist_stereo_core_120k/`，不提交到 Git。仓库中的三份关键适配源文件与该服务器逐字节一致：

| 文件 | SHA256 |
|---|---|
| `stereo_core/no_wrist_pair_model.py` | `056fae41f2da17767c3b6af54fc0373324fec4972fc8a7ffa0fae07a95ae8673` |
| `stereo_core/train_no_wrist_pair.py` | `ba9d07fa5c3a69ca2deb344b43dcd6788ef4f0a5c15cb77086e54aef33a99b20` |
| `stereo_core/evaluate_no_wrist_pair.py` | `be474a410bb40bd116942997592e279942a2f8f200347ee4b5c48fdc418519b6` |

## 5. 从当前分支复跑

仓库入口会检查五份用户 manifest、DINOv3 目录、固定 batch 和正式更新数，并在 `checkpoint_latest.pt` 存在时精确续训：

```bash
S10_CORE_DATA_ROOT=/path/to/robofactory_multitask \
S10_CORE_DINO_MODEL=/path/to/dinov3-vitb16-pretrain-lvd1689m \
S10_CORE_PYTHON=/venv/robofactory-act/bin/python \
scripts/train_s10_core_user_data.sh
```

服务器原始部署流程见 `vendor/stereo-core/deployment/`，训练/评测说明见 `vendor/stereo-core/docs/NO_WRIST_DEPLOYMENT.md`。原评测器显式绑定 `/workspace/RoboFactory`；复用已经完成的服务器环境时保持该路径即可。
