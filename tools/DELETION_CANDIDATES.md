# 工具删除候选清单

本文件只记录删除候选，不代表已经授权删除。Grid_Mamba 内部没有与这些
脚本完全相同的源文件，因此删除属于实验来源保留策略，而不是简单的重复
文件清理。

## 经确认后可优先删除

### `archive/ablation_development/`

依赖废弃模型分支的 4 个执行器已经随核心代码收口删除。目前剩余 2 个
窗口实验执行器：

- 早期窗口缩放变体；
- 未被选中的 50 ms 无重叠变体。

删除该目录不会删除正式 checkpoint，也不会影响正式 HLC2、窗口大小、
EV-Flying 或 FRED 执行器。但删除后将无法按原注册表精确复现这些探索性
实验。

### `archive/stage_feature_candidates/`

其中 14 个脚本用于前期特征提取、校准、验证和多种候选图渲染。论文最终
阶段特征图已经在 `../latex/figures/stage_feature_analysis/` 下拥有独立
流水线。

这些脚本互相导入，若决定删除，应整目录删除。已有分析结果仍保存在
`experiments/analysis/stage_features/`。

## 条件删除候选

### `archive/runtime_legacy/run_window_size_runtime.py`

旧版整样本 Runtime 资产已经保存在
`experiments/archive_pending/legacy_runtime/`。删除脚本不会影响已记录数值，
但会失去按论文时期协议重新执行 Runtime 的入口。当前因果窗口 Runtime
不依赖该脚本。

由于此前已经明确要求保留论文 Runtime 协议，默认建议继续保留该脚本；
只有决定采用“仅保留结果资产”的归档方式时再删除。

## 暂时不要删除

- `experiments/`：正式训练与消融实验入口。
- `runtime/evuav_window/`：当前锁定的 Runtime 协议及 W25 补充实验。
- `runtime/dataset/`：Grid_Mamba 数据集级 Runtime 工具。
- `runtime/complexity/`：当前模型的复杂度分析工具。
- `evaluation/tracking/`：唯一的轨迹与实例评测链。
- `analysis/`：仍可复用的诊断和可视化工具。
- `tests/tools/runtime/`：用于保护 Runtime 协议和迁移路径。
- `_paths.py`：在不修改历史清单原文的情况下解析当前规范资产路径。

工作区中有多个模型仓库保存了 `ev_flying_runtime_common.py` 的副本，但
Grid_Mamba profiler 目前仍依赖本地版本。必须先统一所有使用方，才能删除
任意一个副本。
