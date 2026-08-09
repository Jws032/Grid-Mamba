# 工具删除候选清单

本文件只记录删除候选，不代表已经授权删除。Grid_Mamba 内部没有与这些
脚本完全相同的源文件，因此删除属于实验来源保留策略，而不是简单的重复
文件清理。

## 已执行删除

- `archive/`：论文时期旧版 Runtime 入口已于 2026-08-09 按确认整目录删除；
  对应 `experiments/archive_pending/` 旧协议资产也已删除。当前锁定 Runtime
  入口和论文已记录数值不受影响。
- `archive/ablation_development/`：剩余 2 个未被正式采用的窗口实验执行器，
  已于 2026-08-09 按确认整目录删除；正式消融入口和结果资产不受影响。
- `archive/stage_feature_candidates/`：前期阶段特征提取、校准和候选渲染
  流水线，已于 2026-08-09 按确认整目录删除；论文最终自包含流水线保留。
- `runtime/complexity/`：未形成正式结果资产，也未被活动入口、测试或论文引用；
  已于 2026-08-09 按确认删除。
- `analysis/ev_flying/analyze_ev_flying_denoise.py`：去噪数据质量分析已经完成且
  不再使用；对应 `experiments/analysis/denoise/` 已于 2026-08-09 一并删除。
- `analysis/evuav/run_inference_strategy_ablation.py`：开发期推理策略消融未进入
  论文；对应 `FULL_SC12/inference_strategy/` 已于 2026-08-09 一并删除。
- `analysis/ev_flying/visualize_ev_flying_fail_cases.py`：未形成已保留的生成资产，
  已于 2026-08-09 按确认删除。

## 暂时不要删除

- `experiments/`：正式训练与消融实验入口。
- `runtime/evuav_window/`：当前锁定的 Runtime 协议及 W25 补充实验。
- `runtime/dataset/`：Grid_Mamba 数据集级 Runtime 工具。
- `evaluation/tracking/`：唯一的轨迹与实例评测链。
- `analysis/evuav/visualize_evuav_swc_temporal.py`：保留论文 SWC 数据生成能力，
  默认生成正式样例 `test_020_w09_15` 的第 9 至 15 个窗口。
- `tests/tools/runtime/`：用于保护 Runtime 协议和迁移路径。
- `_paths.py`：在不修改历史清单原文的情况下解析当前规范资产路径。

工作区中有多个模型仓库保存了 `ev_flying_runtime_common.py` 的副本，但
Grid_Mamba profiler 目前仍依赖本地版本。必须先统一所有使用方，才能删除
任意一个副本。
