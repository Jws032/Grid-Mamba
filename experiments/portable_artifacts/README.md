# 可移植实验资产

原本分散在 `artifacts/` 下的 Runtime 锁文件统一保存在
`runtime_locks/` 中。为了保持既有 Runtime 入口兼容，`artifacts/` 下
继续保留指向这些受版本控制锁文件的相对路径软链接。

仓库级 `artifacts/` 继续保留两份活动 Runtime 清单：

- `artifacts/ev_flying.yaml`
- `artifacts/evuav.yaml`

EVUAV 跨模型实例对比执行包没有形成正式结果，已删除。轨迹可视化候选包
也已删除；论文采用的 `test_014` 输入、渲染工具和正式图仍完整保存在
工作区 `latex/figures/instance_segmentation_discussion/`，其输入哈希与删除前
的仓库候选包一致。

EF53 和 EF54 清单已随其诊断资产于 2026-08-09 删除，避免保留指向不存在
archive 的失效索引。
