# Grid_Mamba 工具目录

本目录按照工具职责和生命周期进行组织。所有命令均应在 Grid_Mamba
仓库根目录下，以 Python 模块形式运行，例如：

```bash
python -m tools.experiments.evuav.run_hlc2_paper_ablation --list
python -m tools.runtime.evuav_window.run_evuav_window_size_runtime preflight all
```

## 活动工具分组

- `experiments/`：训练与消融实验入口。其中
  `core/run_ablation.py` 是各数据集实验注册表共用的基础执行器。
- `runtime/`：数据集级运行时间评测、EVUAV 因果窗口 Runtime、FRED
  适配器及复杂度分析工具。
- `evaluation/tracking/`：EVUAV 轨迹构建、参数搜索和实例级指标评测。
- `analysis/`：不定义正式训练结果的诊断与可视化分析工具。
- `archive/`：历史开发消融、旧版 Runtime 协议和阶段特征候选可视化。

Runtime 测试位于 `tests/tools/runtime/`。自动生成的 `__pycache__` 不属于
实验资产，不应保留。

## 实验资产路径

活动工具统一使用 `experiments/` 作为规范实验资产根目录。已完成实验的
配置、日志及不可变清单中记录的历史路径作为来源证据保留，但不作为当前
可执行路径使用。

## 跨模型调度工具

EV-Flying 多模型 Runtime 调度器位于
`runtime/ev_flying/run_all_ev_flying_runtime.py`。它会调用工作区中多个
模型仓库的 profiler，但代码保存在 Grid_Mamba 仓库中，避免工作区
根目录未纳入 Git 时丢失版本记录。
