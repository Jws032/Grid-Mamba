# 可移植实验资产

原本分散在 `artifacts/` 下的 Runtime 锁文件统一保存在
`runtime_locks/` 中。为了保持既有 Runtime 入口兼容，`artifacts/` 下
继续保留指向这些受版本控制锁文件的相对路径软链接。

以下已纳入版本控制且可独立使用的资产包继续保留在仓库级
`artifacts/` 目录，避免改写其 Git 历史：

- `artifacts/evuav_track_visualization/`
- `artifacts/evuav_instance_comparison/`
- `artifacts/ev_flying.yaml`
- `artifacts/evuav.yaml`
- `artifacts/ev_flying_ef53_50ms.yaml`
- `artifacts/ev_flying_ef54_25ms_runtime.yaml`

EF53 和 EF54 清单为受版本控制的普通 YAML 文件，其中的活动路径
指向 `experiments/archive_pending/diagnostic_runs/ev_flying/`。该目录中的
大型诊断资产保持本地存储，不纳入 Git。
