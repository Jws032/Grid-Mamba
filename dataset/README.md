# 数据集代码说明

本目录只保存 Grid_Mamba 使用的数据读取代码和离线处理工具。数据实体统一
位于工作区级 `datasets/`，不放回本目录。

## 运行期读取模块

以下文件保留在原路径，供训练、测试和 Runtime 工具直接导入：

- `basedataset.py`：数据集基类和批次拼接逻辑；
- `ev_uav.py`：EV-UAV 数据读取；
- `ev_flying.py`：EV-Flying 数据读取；
- `fred_segmentation.py`：FRED 分割数据读取。

这些模块的路径保持不变，现有的 `dataset.ev_uav`、`dataset.ev_flying` 和
`dataset.fred_segmentation` 导入无需调整。

## 离线预处理工具

`preprocessing/` 保存不参与训练期读取的离线脚本：

- `filter_ev_flying_hot_pixels.py`：对 EV-Flying 原始事件执行热像素过滤；
- `process_ev_flying_offline_dem.py`：生成 Grid_Mamba 使用的 EV-Flying NPZ
  数据及统计清单；
- `visualize_ev_flying_processed.py`：检查处理后数据并生成可视化结果。

在 Grid_Mamba 根目录下以模块方式运行，例如：

```bash
python -m dataset.preprocessing.filter_ev_flying_hot_pixels --help
python -m dataset.preprocessing.process_ev_flying_offline_dem --help
python -m dataset.preprocessing.visualize_ev_flying_processed --help
```

离线工具默认读写工作区级 `../datasets/`。实际执行覆盖式处理前，应明确指定
输入、输出路径并检查 `--overwrite` 等参数。处理结果检查图默认写入
`experiments/figures/generated/ev_flying_processed/`，不会重新创建旧的
`visualization/` 入口。
