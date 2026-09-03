# before-we-act (CARE)

CARE 是一个部署期的多机器人动作选择框架：在一个冻结的参考策略之上，
用团队信念模型（B-core / team belief）对候选动作分支进行校准打分，只在
校准置信下界支持时才释放替代动作。本仓库包含 CARE 的核心库，以及在两个
基准上的完整封存复现流程。

## 结果与封存复现流程

| 基准 | 结果 | 复现文档 |
|---|---|---|
| RoboFactory（6 任务 × 20 局） | **103/120 = 85.83%** | [docs/reproducibility/care_robofactory/README.md](docs/reproducibility/care_robofactory/README.md) |
| MARS-Control（4 任务 × 20 局） | **28/80 = 35.00%** | [docs/reproducibility/care_mars/README.md](docs/reproducibility/care_mars/README.md) |

两份文档各自固定了 checkpoint 身份（sha256）、随机种子、逐阶段命令与
预期输出。两个基准的 CARE 阶段超参一致；差异在于参考策略的来源：
RoboFactory 使用随代码发布的冻结参考 checkpoint，MARS-Control 在流水线内
从零训练参考策略（B0-H + team belief）。

## 环境

```bash
uv sync            # 依 pyproject.toml + uv.lock 创建 .venv
uv run python -m pytest tests   # 全部测试应通过
```

详见 [UV_ENVIRONMENT.md](UV_ENVIRONMENT.md)。仿真依赖 RoboFactory
（ManiSkill/SAPIEN）的独立 checkout，各复现文档中固定了其 commit。

## 目录结构

- `before_we_act/` — CARE 核心库：信念模型与打分器、候选分支采集、
  两个基准的训练与闭环评测入口。
- `scripts/before_we_act/` — 两条封存流水线（supervisor conf + shell 入口）
  及其数据准备 / 选择 / 校准 / 汇总脚本。
- `deployment/mars_care/` — MARS-Control 官方专家数据采集与本地部署工具。
- `deployment/<baseline>/` — 论文对比基线（ACT、DP、GauDP、LatentToM、
  ManiFlow、OpenVLA-OFT、π0.5、RDT-1B）的可复现适配，逐目录含 README。
- `benchmarks/robofactory_baselines.py` — RoboFactory 六任务统一评测协议。
- `stereo_core/` + `vendor/stereo-core/` — 上游 Stereo-CoRE（MIT）的适配
  子树与原始 checkout；`scripts/before_we_act/validate_upstream_core.py`
  按 `UPSTREAM_CORE_MANIFEST.json` 校验其完整性。
- `configs/` — 各流程的冻结配置（含 sha 与版本钉定）。
- `docs/reproducibility/` — 封存结果的复现记录（CARE 与各基线）。

## Checkpoints

发布的 checkpoint（参考策略、CARE 打分器、分支家族数据）以复现文档中的
sha256 为准；下载后可直接进入对应文档的评测阶段，无需重新训练。

## 测试

```bash
uv run python -m pytest tests
```

测试覆盖动作契约、缓存契约、冻结配置、监督器入口与核心模型的位精确
前向。任何配置漂移都会在测试或各流水线的前置校验中报错。
