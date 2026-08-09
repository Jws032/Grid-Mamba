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

`preprocessing/` 保存不参与训练期读取的离线脚本，并按数据集分开组织：

```text
preprocessing/
├── ev_flying/
│   ├── filter_hot_pixels.py       # 热像素过滤
│   ├── process_offline_dem.py     # 生成训练 NPZ 与统计清单
│   └── visualize_processed.py     # 检查处理结果
└── fred/
    ├── download_raw_fred.sh       # 下载官方原始 FRED 数据
    ├── bbox_to_point_labels.py    # bbox 转点级弱标注核心
    ├── build_point_label_dataset.py # 批量生成 FRED_segmentation
    ├── extract_event_window_npz.py # 从 HDF5 提取指定事件时间窗
    ├── split_train_val.py         # 序列级 train/val 划分
    └── filter_by_bbox_area.py     # 生成 area1250 派生数据集
```

`ev_flying/` 中的脚本负责 EV-Flying 原始事件去噪、训练数据生成和结果检查；
`fred/` 中的脚本负责从原始 FRED bbox 构建点级弱标注、划分验证集，并生成
按目标框面积筛选的派生数据集。

`download_raw_fred.sh` 默认将官方数据下载到工作区级 `../datasets/FRED/`，
不会在项目目录内创建数据副本。代理、并发数和目标目录可分别通过
`FRED_PROXY`、`FRED_CONCURRENCY` 和 `FRED_TARGET_DIR` 环境变量覆盖；脚本
默认不启用代理。`extract_event_window_npz.py` 是诊断工具使用的轻量 HDF5
事件窗口提取入口。

在 Grid_Mamba 根目录下以模块方式运行，例如：

```bash
python -m dataset.preprocessing.ev_flying.filter_hot_pixels --help
python -m dataset.preprocessing.ev_flying.process_offline_dem --help
python -m dataset.preprocessing.ev_flying.visualize_processed --help
```

### FRED bbox 转点标注链路

基础数据集 `FRED_segmentation` 来自共享原始数据集 `FRED/`。每个原始序列
ZIP 中至少需要包含 `coordinates.txt` 和 `Event/events.hdf5`，正式生成过程为：

1. `fred/build_point_label_dataset.py` 从 `FRED/train/*.zip` 和
   `FRED/test/*.zip` 中提取目标框与事件文件；
2. `fred/bbox_to_point_labels.py` 将目标框裁剪到 `1280×720` 传感器范围；
3. 对时间戳为 `t_bbox` 的目标框，将满足
   `t_bbox - 33333 < t_event <= t_bbox` 且空间坐标位于闭区间目标框内的事件
   标记为前景，并写入对应 `instance_id`；其余事件标记为背景；
4. 将事件划分为 8 秒分块，默认删除首分块、无前景分块和不完整分块；
5. 输出紧凑 NPZ，包含 `x`、`y`、`t_us`、`p`、`label`、
   `instance_id` 和 `meta`，并生成 manifest、summary 与断点记录；
6. 再使用 `fred/split_train_val.py` 生成序列级验证集。

这些标签是从 bbox 派生的点级弱标注，不是人工逐事件标注。可先运行内置测试
和只读任务预览：

```bash
python -m dataset.preprocessing.fred.bbox_to_point_labels \
  --run-synthetic-test

python -m dataset.preprocessing.fred.build_point_label_dataset \
  --fred-root ../datasets/FRED \
  --output-root ../datasets/FRED_segmentation \
  --splits train test \
  --window-us 33333 \
  --chunk-ms 8000 \
  --dry-run
```

原始 FRED 的 `events.hdf5` 使用 HDF5 filter `36559`。读取真实事件需要与
运行平台匹配的 `libH5Zecf.so`，可通过 `--hdf5-plugin-path` 指定；该二进制
插件属于运行环境依赖，二进制文件没有纳入 Git 版本控制。未提供插件时，任务
预览与内置合成测试仍可运行，但不能读取和生成真实事件数据。

### FRED area1250 生成链路

`FRED_segmentation_area1250` 由共享数据集 `FRED_segmentation` 派生，实际采用
的处理顺序如下：

1. 使用 `fred/split_train_val.py`，以随机种子 `20260615` 和
   `density-stratified` 方法，从原始训练序列中划出验证集；
2. 使用 `fred/filter_by_bbox_area.py`，保留每个 8 秒分块中
   `mean clipped bbox area < 1250 px²` 的样本；传感器尺寸为 `1280×720`，
   输出采用独立复制模式 `copy`。

复现前先执行只读预览：

```bash
python -m dataset.preprocessing.fred.split_train_val \
  --root ../datasets/FRED_segmentation \
  --seed 20260615 \
  --method density-stratified \
  --dry-run

python -m dataset.preprocessing.fred.filter_by_bbox_area \
  --source-root ../datasets/FRED_segmentation \
  --fred-root ../datasets/FRED \
  --output-root ../datasets/FRED_segmentation_area1250 \
  --max-mean-area-px2 1250 \
  --copy-mode copy \
  --dry-run
```

第一步会在正式执行时移动基础数据集中的验证文件，第二步在指定
`--overwrite` 时会重建整个输出目录。现有共享数据集已经完成上述处理，
不要直接对其重复执行；需要复现时应使用单独副本，并在核对 dry-run 结果后
再移除 `--dry-run`。`filter_by_bbox_area.py` 会优先读取 manifest 中仍有效的
`source_zip`，旧服务器绝对路径失效时则按 `--fred-root`、原始划分和序列编号
定位对应 ZIP。

离线工具默认读写工作区级 `../datasets/`。实际执行覆盖式处理前，应明确指定
输入、输出路径并检查 `--overwrite` 等参数。处理结果检查图默认写入
`experiments/figures/generated/ev_flying_processed/`，不会重新创建旧的
`visualization/` 入口。
