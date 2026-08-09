# EV-Flying 跨模型 Runtime 调度

`run_all_ev_flying_runtime.py` 会统一调度多个模型仓库中的 Runtime 工具，
逻辑上属于工作区级调度器。由于工作区根目录未纳入 Git，该工具
存放在 Grid_Mamba 仓库内，以便进行版本控制。

在 Grid_Mamba 仓库根目录下运行：

```bash
python -m tools.runtime.ev_flying.run_all_ev_flying_runtime --help
```

运行前需要配置各模型对应的 Python 环境变量。若未分别设置
`GRID_MAMBA_PYTHON` 等变量，可通过 `CONDA_ENVS_ROOT` 指定统一的 Conda
环境根目录。

该调度器不包含各模型的具体 profiler。Grid_Mamba 自身的 profiler 位于
`tools/runtime/dataset/`。
