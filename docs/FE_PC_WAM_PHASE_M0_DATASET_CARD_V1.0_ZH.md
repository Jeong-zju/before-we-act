# FE-PC WAM Phase M0 多模态数据集卡 V1.0

> Schema：`wam.multimodal/1.1`
>
> 配置：`configs/wam_multimodal/m0_data.yaml`
>
> Canonical 数据：`datasets/mujoco_visual_required_wam_multimodal_m0`
>
> 状态：M0-v2 验收资产；是否通过只以 `outputs/phase_m0_mujoco/phase_m0_acceptance.json` 的 `passed` 与 `formal_protocol` 为准，本文不预先声明 Gate 已通过。

## 1. 用途与边界

该数据集用于验证机器人 proprioception 无法恢复、但 MuJoCo RGB 能够观测的任务条件，并为 Phase M1 的视觉条件 latent WAM 提供训练入口。它不是通用机器人基础模型数据集，也不用于证明真实机器人安全性。

数据覆盖视觉事件停止、视觉目标选择、视觉障碍/禁区三类 visual-required 任务。每个 physical seed 均以两个相反 cue 成对采集；cue parity 不参与物理随机化，改变 cue 不得改变初始 state、场景/对象 identity 或 cue 出现前的动力学。采集 policy 使用明确标记为 privileged 的 scripted oracle 生成可解轨迹；正式 state-only、vision oracle 和 opposite-cue RGB benchmark 在独立 held-out seeds 上运行。

Canonical 配置为每任务 800 episodes、总计 2,400 episodes。train/validation/test 每任务分别为 600/100/100 episodes，并保持 task、cue 与 physical-seed pair 平衡。低于 2,000 episodes 的独立配置只能用于 diagnostic，必须标记 `formal_protocol=false`，不能给出 M0 正式通过结论。

旧 `datasets/visual_required_wam_multimodal_m0` 与 `outputs/phase_m0` 是 schema 1.0、单 `fixed`、analytical OpenCV renderer 的历史 preflight，不是本数据集的替代品，也不参与 M0-v2 Gate。

## 2. 仿真器、相机与刹车信号

- 场景来自 `envs/assets/two_robot_carry.xml`，RGB 由 `mujoco.Renderer` 直接从 `MjModel`/`MjData` 渲染。Manifest 绑定 MuJoCo 版本、XML 路径/SHA-256、camera rig 与正式协议代码 SHA-256。
- Canonical `camera_order` 为 `[fixed, robot_0_camera, robot_1_camera]`。`fixed` 是 world-body 全局视角；另外两台相机分别挂载在 `robot_a`、`robot_b` 机体上。共享 XML 保留 standard 环境的 legacy local pose，`VisualRequiredEnv` 只在自己的独立 `MjModel` instance 中覆盖 chase pose/FOV；该 override 由 `visual_required_env.py` 的 source SHA-256 绑定，不会改写 standard 相机。
- 所有正式 RGB 均为未标注帧。OpenCV 只用于 MP4 编码与人类可视化，不参与场景绘制；annotation 不得进入 policy、HDF5 图像或正式 MP4。
- `visual_event_stop` 在 step 14 前显示相同的中性信号，机器人刹车灯关闭。step 14 的 next capture 首次只由专用 `visual_event_signal` 显示 stop/pass cue；产生该 capture 的动作仍基于中性 current image，policy 只能在下一次 `act` 时分支。两盏机器人刹车灯不得复制 cue、不得变绿，只能在后续动作已经让对应 agent 请求减速时各自点红，并在零速制动命令保持期间维持红色。正式审计要求同一 physical-seed pair 的三路 pre-onset raw RGB 序列逐帧完全一致，active capture 的 cue-dependent MuJoCo geom 严格只有专用信号。
- 原 cooperative-stop 环境的物理刹车语义独立保留：事件前两盏灯均关闭，事件激活后只点亮真实 braking agent。人类 annotation/viewer 在事件前不得显示未来 braking agent、启动时刻或倒计时。

## 3. 观测、视频与时间语义

- 控制频率：20 Hz；三台相机捕获频率：10 Hz；分辨率：256×256 RGB。
- `timestamp` 是 current state/action 的仿真时间。`observation.image_timestamp.<camera>` 是所选 current RGB 的 capture 时间，`image_state_timestamp` 是与该 capture 严格配对的 state snapshot 时间；`next_observation.*` 使用相同定义。
- 同一 10 Hz capture 可被相邻两个 20 Hz transition 引用；必须依据显式 `image_frame_index` 识别 sample-hold，不能用行号或隐式“每两步一帧”推断。
- 三相机在同一个 state snapshot 上捕获并共享 frame index/timestamp。Capture sync skew 定义为 `abs(image_timestamp-image_state_timestamp)`；它与 `state timestamp-image_timestamp` 的 action frame age 分开审计。
- HDF5 中的 `[H,W,3] uint8` 是 canonical raw-unannotated RGB。每 episode、每 camera 的 MP4 只编码唯一 10 Hz capture，是 `mp4v` 有损可视资产，不是唯一真值来源。
- 轨迹窗口不得跨 episode；episode 视频也不得包含相邻 episode 的帧。

## 4. 相机标定

每次新 capture 与 RGB 一起读取相机标定；sample-hold 时 K/E/resolution 必须与所引用 frame index 的 snapshot 逐值一致：

```text
data/camera/intrinsics/<camera>          float32 [T,3,3]
data/camera/extrinsics/<camera>          float32 [T,4,4]
data/camera/resolution/<camera>          int64   [T,2]
data/next_camera/intrinsics/<camera>     float32 [T,3,3]
data/next_camera/extrinsics/<camera>     float32 [T,4,4]
data/next_camera/resolution/<camera>     int64   [T,2]
```

标定模型为 pinhole；外参约定为 `opencv_optical_camera_pose_in_world`，即 OpenCV optical frame 的 camera-to-world pose。`fixed` 外参必须静态；`robot_0_camera`、`robot_1_camera` 外参必须随各自 parent body 运动。任何缺字段、NaN/Inf、错误 shape、非整数或不匹配的 resolution 都 fail closed。

## 5. Schema 1.1

每个 HDF5 episode 的 `data/` 下至少包含：

```text
timestamp / frame_index / episode_index / seed
task/text / task/id
observation/state
action/commanded / action/executed
next_observation/state

for <camera> in [fixed, robot_0_camera, robot_1_camera]:
  observation/images/<camera>
  observation/image_timestamp/<camera>
  observation/image_state_timestamp/<camera>
  observation/image_frame_index/<camera>
  next_observation/images/<camera>
  next_observation/image_timestamp/<camera>
  next_observation/image_state_timestamp/<camera>
  next_observation/image_frame_index/<camera>
  camera/intrinsics|extrinsics|resolution/<camera>
  next_camera/intrinsics|extrinsics|resolution/<camera>

event/visual_signal_active
event/visual_signal_onset_step
event/visual_signal_kind
event/rendered_cue_variant

reward / terminated / truncated / done
success / failure / failure_reason
schema_version / behavior_id
environment_config / randomization_config
```

`event.*` 来源于 transition 的 post-action `info`，描述对应 `next_observation` 的信号状态，只用于数据审计。`MultimodalTrajectoryDataset` 验证这些字段存在，但不把它们、cue truth 或其他 privileged state 返回给训练 sample。Policy 输入中只允许显式 allowlist 的 proprioception、past executed actions、cue-independent task condition，以及配置指定的原始 RGB。

## 6. Split 与防泄漏

train/validation/test 以完整 physical-seed pair 为最小单元；同一 seed 的两个 cue 不拆分。三个 split 隔离 seed、randomization template、scene identity 和 object-combination identity，并要求：

- episode、seed、template、scene、object-combination overlap 全为 0；
- 每个 split 覆盖全部三种任务和两个 cue；
- 同一 physical-seed pair 的 scene/object/template identity 完全一致；
- event pair 在 onset 前的 state、动作历史、capture timestamp/frame index 与三路 raw RGB 序列一致，active capture 的 timestamp/frame index 对齐但 cue 像素发生分叉。

## 7. Gate M0

数据与相机 Gate：

- 每相机 capture sync skew P99 `< 0.025 s`；action frame age max `<= 0.1 s`；
- 跨相机 frame index/timestamp 同步，episode boundary crossing 为 0；
- 损坏视频、空帧为 0；预期 10/20 Hz frame reuse 与意外 duplicate capture 分开报告；
- current/next calibration 连续且与 capture sample-hold 一致，矩阵有限、旋转合法、resolution 匹配；
- `fixed` 外参静态、两路 robot camera 外参动态；每个 episode 三份 MP4 均可解码且帧数正确；
- 三路 signal-onset/pair 像素证据、MuJoCo/XML/camera/source provenance 与三相机 loader smoke 全部通过。

任务 Gate 对每个 task 及 macro 同时执行：

- state-only success rate `<= 0.70`；
- privileged scripted oracle success rate `>= 0.95`；
- 只消费 `images.fixed` 的 vision oracle success rate `>= 0.95`；
- clean vision oracle 相对 opposite-cue MuJoCo RGB 的成功率下降 `>= 0.20`。

Benchmark policy 可以只消费 `fixed`，但三相机仍必须同步渲染、记录并提供 cue/onset 和动态外参证据。Opposite-cue 测试只改变 renderer-visible cue；physical seed、truth、scene/object identity、初始 state 与物理动力学保持不变。报告必须记录 presented/consumed observation leaf paths，禁止 annotation 或 privileged truth。

旧 proprioceptive Joint WAM 还必须严格重载：先匹配冻结的 pre-M0 checkpoint-tree SHA-256 anchor，再验证验收前后 tree 不变，并在 standard/challenge held-out seeds 上保持配置要求的成功率。

## 8. 产物与哈希边界

Canonical 数据目录：

```text
datasets/mujoco_visual_required_wam_multimodal_m0/
  manifest.json
  hdf5/episode_XXXXXX.hdf5
  videos/episode_XXXXXX/fixed.mp4
  videos/episode_XXXXXX/robot_0_camera.mp4
  videos/episode_XXXXXX/robot_1_camera.mp4
```

Canonical 验收目录：

```text
outputs/phase_m0_mujoco/
  multimodal_dataset_audit.json
  visual_required/visual_required_benchmark.json
  visual_required/visual_required_episodes.jsonl
  phase_m0_acceptance.json
```

配置、dataset card、manifest、HDF5/MP4、episode records 与报告均以 SHA-256 绑定。Dataset card 在采集前被 manifest 哈希；采集后修改本卡会使 provenance 审计失败，必须重新采集。CLI 覆盖数据量、seed 或输出目录时只能生成 diagnostic 资产，不得覆盖 canonical 正式证据。

## 9. 已知限制

- 图像虽来自真实 `mujoco.Renderer` 与机体挂载相机，仍是仿真 RGB，不包含真实相机曝光、rolling shutter、镜头畸变、硬件时钟漂移或网络抖动。
- 环境沿用仓库现有 geometric controller：机器人 pose 由 acceleration-limited kinematics 更新，MuJoCo 用于 XML geometry、contact query、effort 与 RGB rendering；它不是 `mj_step` 电机动力学或真实机器人闭环。
- MP4 使用有损 `mp4v`；逐像素审计以 HDF5 raw RGB 为准。
- 三类任务专门用于检测视觉因果价值，不能外推到开放词汇、真实安全或跨 embodiment 泛化。
- 20 Hz HDF5 行会 sample-hold 10 Hz RGB；训练和统计必须依据 frame index，不能把预期复用误判为采集故障。
- 本次 M0 验收不作 GPU latency 或真实部署实时性声明。
