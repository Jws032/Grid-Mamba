# Grid Mamba

Grid Mamba 是面向事件点云分割的实验代码库，当前整理版本支持 EV-UAV、
EV-Flying 和 FRED 三套数据。论文内容和已报告指标已经冻结；本仓库的整理目标
是保留核心模型、正式实验入口、可复现工具和实验资产登记，不再扩展论文结果。

## 项目状态

- 核心模型代码已经移除未采用的开发分支，保留正式模型与论文消融所需路径。
- 数据实体统一放在仓库同级的 `datasets/`，不提交到 Git。
- 正式 checkpoint、日志、评测和 Runtime 资产统一放在 `experiments/`，大型实体
  默认不提交到 Git；路径、选择规则和校验和记录在 `experiments/registry/`。
- CP04 正式版本采用 `rerun1`，W300 采用 `rerun2`，EV-Flying 正式实验只保留
  EF45；W25 仅作为实验资产保留。
- FRED 正式训练结果和 EF45→FRED 跨数据集结果已经集中到主服务器管理。

更完整的整理决策和进度见
[项目整理记录](docs/项目整理_20260808.md) 与
[实验资产说明](experiments/README.md)。

## 目录结构

```text
Grid_Mamba/
├── configs/                    # EV-UAV、EV-Flying、FRED 配置
├── dataset/                    # 运行期数据读取与离线预处理代码
│   └── preprocessing/          # 按数据集分类的离线处理工具
├── model/Grid_Mamba/           # Grid Mamba 核心模型
├── tools/
│   ├── experiments/            # 正式训练与消融入口
│   ├── runtime/                # Runtime 评测工具
│   ├── evaluation/             # 轨迹构建与实例级评测
│   └── analysis/               # SWC、FRED 与数据统计工具
├── experiments/
│   ├── runs/                   # 正式实验资产，本地保存且默认被 Git 忽略
│   ├── analysis/               # 筛选后的分析与数据统计资产
│   └── registry/               # 受版本控制的登记、迁移记录和校验和
├── tests/                      # CPU 可执行的回归测试
├── train_grid_mamba.py         # 底层训练入口
└── test_grid_mamba_cetus_style.py # 底层测试与 Runtime 入口
```

## 运行环境

当前经过验证的核心兼容组合为：

| 组件 | 版本或属性 |
|---|---|
| Python | 3.9.25 |
| PyTorch | 2.2.0 |
| CUDA | 11.8（PyTorch 编译版本） |
| PyTorch C++ ABI | `_GLIBCXX_USE_CXX11_ABI=False` |
| `mamba-ssm` | 2.2.0 |
| `causal-conv1d` | 1.4.0 |
| `spconv-cu118` | 2.3.8 |
| NumPy | 1.26.4 |
| PyYAML | 6.0.2 |

建议创建 Python 3.9 的独立 Conda 环境，先安装 PyTorch 2.2.0 + CUDA 11.8，
再安装与 `cu118`、`torch2.2`、`cxx11abiFALSE` 和 `cp39` 完全匹配的
`mamba-ssm`、`causal-conv1d` 与 `spconv`。训练和分析工具还使用 MLflow、
tqdm、pandas、SciPy、scikit-learn、Matplotlib、Seaborn、Pillow 和 OpenCV。

本仓库没有保留本机 wheel 缓存，也尚未提供可覆盖全部 CUDA 二进制依赖的
一键环境文件。不要混用不同 PyTorch、CUDA、Python 或 C++ ABI 的预编译包。
详细版本、原 wheel 标签与 SHA-256 见
[环境依赖版本记录](docs/环境依赖版本记录.md)。

## 数据集布局

代码默认从仓库上一级的共享目录读取数据：

```text
../datasets/
├── EV-UAV/
├── EV-Flying/
├── EV-Flying-raw/
├── FRED_segmentation/
└── FRED_segmentation_area1250/
```

- `EV-UAV`、`EV-Flying` 和 `FRED_segmentation` 是训练与评测使用的数据。
- `EV-Flying-raw` 供离线预处理和原始数据统计使用。
- `FRED_segmentation_area1250` 是从 `FRED_segmentation` 派生的小目标子集。
- 原始 FRED 不属于普通代码同步范围，目前保留在专用实验服务器。

`configs/evisseg_fred*.yaml` 中保留了实验时的源服务器绝对路径，作为原始配置
证据。跨服务器运行 FRED 时，优先使用正式入口的 `--data-root` 参数覆盖数据
路径，不要直接修改已经冻结的正式配置和实验资产。数据生成、bbox 转点标注、
验证集划分与 area1250 筛选过程见 [数据集代码说明](dataset/README.md)。

## 快速检查

进入仓库并激活环境：

```bash
cd Grid_Mamba
conda activate grid_mamba
```

列出当前保留的正式实验：

```bash
python -m tools.experiments.evuav.run_hlc2_paper_ablation --list
python -m tools.experiments.evuav.run_window_size_curve --list
python -m tools.experiments.ev_flying.run_ev_flying_ablation --list
python -m tools.experiments.fred.run_fred_ablation --help
```

在独立临时目录中执行一个 HLC2 冒烟训练：

```bash
python -m tools.experiments.evuav.run_hlc2_paper_ablation \
  --experiment MC01 \
  --stage train \
  --smoke \
  --output-root /tmp/grid_mamba_hlc2_smoke \
  --cuda-visible-devices 0
```

对 FRED 执行单批次冒烟检查：

```bash
python -m tools.experiments.fred.run_fred_ablation \
  --stage smoke \
  --data-root ../datasets/FRED_segmentation \
  --output-root /tmp/grid_mamba_fred_smoke \
  --cuda-visible-devices 0
```

`train_grid_mamba.py` 和 `test_grid_mamba_cetus_style.py` 是各正式调度器调用的
底层入口。若直接调用，应先复制 YAML，并把 `model_save_root`、`model_path`、
`output_path` 和数据路径改到新的实验目录，避免覆盖已冻结资产。

## 正式实验资产

主要规范路径如下：

| 内容 | 规范路径 |
|---|---|
| EV-UAV FULL | `experiments/runs/evuav/baseline/FULL_SC12` |
| EV-UAV HLC2 消融 | `experiments/runs/evuav/ablation/hlc2` |
| EV-UAV 窗口实验 | `experiments/runs/evuav/window_size/formal` |
| EV-Flying EF45 | `experiments/runs/ev_flying/baseline/EF45` |
| FRED 正式实验 | `experiments/runs/fred/ablation/FRED_SC12_GS_SCALED` |
| EF45→FRED | `experiments/runs/cross_dataset/EF45_to_FRED` |
| 数据集统计 | `experiments/analysis/dataset_stats` |

这些目录中的大型文件不会随 GitHub 克隆自动下载。获得独立保存的实验资产后，
可在仓库根目录执行：

```bash
sha256sum -c experiments/registry/checksums.sha256
```

如果只克隆了公开代码而没有实验资产，上述校验会报告文件缺失，这是预期行为。

## 测试

在 `grid_mamba` 环境中运行：

```bash
python -m unittest discover -s tests -t . -v
```

当前整理版本共有 34 项单元测试，覆盖正式实验注册表、路径迁移、FRED
预处理定位逻辑以及 EV-UAV 窗口 Runtime 的资产锁和调度行为。

## FRED HDF5 插件

原始 FRED 的 `events.hdf5` 使用 HDF5 filter `36559`。真实读取需要匹配平台的
`libH5Zecf.so` 与 `libhdf5_ecf_codec.so`，并通过 `--hdf5-plugin-path` 或
`HDF5_PLUGIN_PATH` 指定插件目录。两个动态库必须放在同一目录。

由于当前没有找到明确的再发布许可，插件二进制不会提交到公开 GitHub 仓库；
目标实验服务器继续保留经过校验的 `tools/hdf5_ecf` 环境。没有插件时仍可运行
预处理任务预览和合成测试，但不能读取真实 FRED HDF5 事件。

## 资产与版本管理约定

- GitHub `main` 是唯一维护主线，不再使用原 Gitee 仓库。
- 数据集、checkpoint、训练日志和大部分分析输出不进入 Git 历史。
- 正式实验内部文件不做选择性删减；迁移完成后删除旧源目录，避免多份混杂。
- 论文采用的指标名称和 Runtime 数值保持不变。
- 新实验应使用新的输出目录，不要对现有规范路径使用 `--overwrite`。

迁移来源、删除依据与正式版本选择可查阅
[`experiments/registry/migration_manifest.tsv`](experiments/registry/migration_manifest.tsv)
和 [`experiments/registry/experiments.yaml`](experiments/registry/experiments.yaml)。
