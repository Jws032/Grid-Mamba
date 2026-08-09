# FRED 分析工具

本目录保存 FRED 数据集的只读审查与可视化工具，不参与模型训练和正式评测。
生成结果统一写入 `experiments/analysis/fred/`，不会重新创建顶层
`diagnostics/` 或 `visualizations/` 目录。

- `visualize_fred_point_labels_3d.py`：渲染处理后 NPZ 的 x-y-t 点云；
- `visualize_fred_rgb_event_compare.py`：对比 RGB、bbox 与多个事件时间窗；
- `visualize_fred_segmentation_xy_bins.py`：按事件数量区间抽样并渲染 XY 投影。

`_rgb_common.py` 只保存 `rgb_event_compare` 需要的 bbox、RGB 帧解析和绘制公共
函数，不提供独立命令入口。原独立 label audit 与 RGB/bbox audit 工具已随对应
分析产物删除。

其中 RGB、bbox 和原始事件审查需要工作区级 `../datasets/FRED/`。读取原始
`events.hdf5` 还需要 HDF5 ECF 解码插件；默认从
`tools/hdf5_ecf/plugin/` 查找，也可以通过命令行参数指定其他位置。

所有脚本都可先运行 `--help` 检查入口。建议将大规模生成结果保留为本地实验
资产，不提交到公开代码仓库。
