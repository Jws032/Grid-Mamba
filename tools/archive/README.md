# 历史归档工具

本目录保存历史实验代码，不作为新实验的默认入口。

- `ablation_development/`：探索性或未被正式选中的消融实验执行器。
- `runtime_legacy/`：论文时期使用的整样本窗口 Runtime 实现；对应输出
  保存在 `experiments/archive_pending/legacy_runtime/`。
- `stage_feature_candidates/`：前期阶段特征校准和候选可视化工具。论文
  最终图的独立流水线位于工作区
  `latex/figures/stage_feature_analysis/`。

归档工具默认不得向正式实验目录写入结果。只有在明确决定不再保留历史
复现能力后，才可以删除相应工具。
