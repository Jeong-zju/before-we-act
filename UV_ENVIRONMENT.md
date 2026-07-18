# FE-PC WAM 的 uv 环境

`fe_pc_wam` 目录内的 `pyproject.toml`、`.python-version` 和 `uv.lock` 是本项目唯一的 Python 环境定义。项目只支持 Python 3.11；Linux/Windows 上的 PyTorch 使用 CUDA 12.8 wheel。

## 初始化或更新

在 `fe_pc_wam` 目录运行：

```bash
uv sync
```

这会创建或更新当前目录的 `.venv`。日常命令优先通过 `uv run` 执行，不依赖当前 shell 中是否激活了 conda/venv：

```bash
uv run pytest
uv run python scripts/train_action_prior.py --help
```

不要在 `.venv` 中直接执行 `pip install`。新增运行依赖使用 `uv add <package>`，新增测试或开发依赖使用 `uv add --dev <package>`，并提交更新后的 `pyproject.toml` 与 `uv.lock`。
