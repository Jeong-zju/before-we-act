# FE-PC WAM 远程自动化运行手册

`scripts/wam_automation.sh` 把代码、环境、数据、训练和真实 RoboFactory
闭环验证组织成可组合的有序动作。任一动作失败都会立即停止，后续动作不会
执行；成功位置会写入状态文件，可用 `--resume` 从失败处继续。

默认 profile 是 LiftBarrier M1 scratch：

- FE-PC WAM：Python 3.11、仓库 `uv.lock`、冻结 DINOv3；
- RoboFactory：独立 Python 3.9 环境，绝不复用 WAM 的 `.venv`；
- 数据：Hugging Face dataset repo 下载到
  `datasets/robofactory_lift_barrier_m1_v1`；
- 训练：`train_liftbarrier_m1_scratch.py` 的四阶段正式训练；
- 验证：RoboFactory 环境进程与 WAM 推理进程通过本机 TCP 编排。

## 1. 准备配置和认证

在已有检出中：

```bash
cd /path/to/fe_pc_wam
cp configs/automation.example.env configs/automation.env
```

编辑 `configs/automation.env`，至少填写：

- `WORKSPACE_ROOT`：放置两个同级仓库的绝对目录；
- `FE_REF`：FE-PC WAM 分支、tag 或 commit；
- `ROBOFACTORY_REPO_URL` / `ROBOFACTORY_REF`：与当前闭环契约兼容的
  RoboFactory fork 和 revision；
- `HF_DATASET_REPO`：已经转换并生成训练 manifest 的 dataset repo。

Token 只放在进程环境中：

```bash
export HF_TOKEN='hf_...'
```

不要把 token 写入命令行、Git 配置或 `automation.env`。脚本调用新版
`hf` CLI，并通过继承环境读取 token；日志不会展开 token。也可以事先使用
`hf auth login` 的本机凭据，`hf-auth` 以 `hf auth whoami` 的实际结果为准。

Hugging Face 数据集根目录应直接包含：

```text
training_manifest.json
training_manifest.json.sha256
normalization.npz
hdf5/
```

manifest 内的 HDF5 相对路径必须在上传后仍然成立。默认 DINOv3 是 gated
模型；运行前还要在模型页面接受许可，并保证 token 有 gated model 读取权限。

## 2. 多动作顺序执行

动作以空格或逗号分隔，严格从左到右执行：

```bash
./scripts/wam_automation.sh \
  --config configs/automation.env \
  code robofactory env robofactory-env hf-auth hf-download vision data-check
```

也可以写为：

```bash
./scripts/wam_automation.sh \
  --config configs/automation.env \
  code,robofactory,env,robofactory-env,hf-auth,hf-download,vision,data-check
```

常用动作：

| 动作 | 作用 |
|---|---|
| `code` | clone/fetch FE-PC WAM，安全切换 `FE_REF`，只允许 fast-forward |
| `robofactory` | clone/fetch RoboFactory，切换 `ROBOFACTORY_REF` |
| `env` | 安装 uv/CPython 3.11，并按 WAM `uv.lock` 创建 `.venv` |
| `robofactory-env` | 创建隔离的 Python 3.9 环境 |
| `assets` | 下载 RoboFactory 基础资产 |
| `hf-download` / `hf-upload` | 下载或可恢复地上传 Hugging Face 数据集 |
| `vision` | 下载固定 revision 的 DINOv3 并校验架构与 SHA-256 |
| `doctor` | 检查空间、GPU/Vulkan、Python 版本和目录 |
| `data-check` | 校验 manifest、hash、三个 split 和 M1 window |
| `test` | 运行与自动链路相关的契约测试 |
| `train-smoke` / `train` | 短训练预检 / 正式训练 |
| `validate-smoke` / `validate` | 3 集闭环 smoke / 100 seeds 正式 benchmark |
| `snapshot` | 保存两个 Git HEAD 和数据/checkpoint 来源 |

`code` 和 `robofactory` 在已有仓库存在 tracked 修改时会拒绝切换分支，不会
reset 或覆盖本地工作。正式 checkpoint 和闭环输出同样实行 fail-closed。

## 3. 从零到训练闭环的一条指令

配置和 `HF_TOKEN` 已就绪后：

```bash
./scripts/wam_automation.sh --config configs/automation.env full
```

`full` 展开为：

```text
code → robofactory → env → robofactory-env → assets → hf-auth
→ hf-download → vision → doctor → data-check → test
→ train → validate → snapshot
```

先验证远程机器、权限和路径时使用：

```bash
./scripts/wam_automation.sh \
  --config configs/automation.env \
  --dry-run full
```

`--dry-run` 不 clone、不建目录、不联网、不启动训练，只打印实际顺序和参数。
正式运行前建议先跑资源较小的完整 smoke：

```bash
./scripts/wam_automation.sh --config configs/automation.env full-smoke
```

当同一动作序列中 `train-smoke` 位于 `validate-smoke` 之前时，闭环 smoke
会自动加载 `WAM_SMOKE_CHECKPOINT`；否则 `validate-smoke` 默认加载正式
`WAM_CHECKPOINT`。两种 checkpoint 都不会被自动覆盖。

如果远程服务器尚无仓库，可先把脚本和配置通过运维系统、`scp` 或可信的 raw
URL 放到服务器，再从任意目录执行。脚本不依赖仓库内 Python 即可完成首次
clone；`WORKSPACE_ROOT` 决定两个仓库的落点。

## 4. 中断恢复与日志

默认状态和日志位于：

```text
${WORKSPACE_ROOT}/.wam-automation/
├── lock
├── state
├── logs/<RUN_ID>.log
└── runs/<RUN_ID>/provenance.json
```

失败后用相同配置和相同动作序列恢复：

```bash
./scripts/wam_automation.sh \
  --config configs/automation.env \
  --resume full
```

状态文件记录动作位置，并绑定 repo/ref、数据 revision、配置、checkpoint 和
完整动作序列的 fingerprint。上述内容变化时，脚本拒绝错误续跑。运行锁防止
同一 workspace 中两个 pipeline 同时写 checkpoint 或占用同一闭环端口。

需要保留多个独立状态时：

```bash
./scripts/wam_automation.sh \
  --config configs/automation.env \
  --state-file /path/to/state.m1 \
  code env hf-download data-check
```

## 5. 数据集上传

大目录默认使用可恢复的 `hf upload-large-folder`：

```bash
./scripts/wam_automation.sh \
  --config configs/automation.env \
  hf-auth hf-upload
```

通过 `HF_UPLOAD_DIR`、`HF_UPLOAD_REPO` 和 `HF_UPLOAD_REVISION` 选择来源与
目标。`HF_UPLOAD_MODE=single` 改为单 commit 上传；`.cache/**` 不会上传。
目标 repo 默认 private，并使用 `--exist-ok` 幂等创建。

## 6. RoboFactory 兼容性

上游 RoboFactory 没有根目录 `uv.lock` 时，`robofactory-env` 会按其 pinned
requirements 创建 Python 3.9 环境；提供根 `pyproject.toml + uv.lock` 的 fork
则自动使用 locked 模式。RTX 5090 等新 GPU 应使用项目已适配 CUDA 12.8 的
RoboFactory fork/lock，不能依赖上游旧 Torch wheel。

正式 `validate` 启动 100 集之前会校验 RoboFactory 环境 YAML 和关键源码的
SHA-256。ref 不符合当前正式协议时立即失败，不会生成一个看似成功但不可报告
的结果。`validate-smoke` 用于工程联通检查，不能替代正式 benchmark。

## 7. 常见失败

- `hf auth whoami` 失败：在当前终端/作业调度器中导出 `HF_TOKEN`，或先完成
  `hf auth login`。
- DINO `Access denied`：token 对应账号尚未接受 gated model 许可。
- `tracked local changes`：手工处理对应仓库修改；脚本不会 reset。
- `checkpoint already exists`：保留正式产物并换配置/output，或确认后手工归档；
  脚本不会删除 checkpoint。
- 正式 RoboFactory contract mismatch：切换到配置指定的兼容 fork/ref，不能
  绕过 hash 门禁。
- 空间不足：调整挂载点或清理可再生产物；默认要求至少 30 GiB 空闲。
