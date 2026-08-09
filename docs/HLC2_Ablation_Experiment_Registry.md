# HLC² 正式消融实验编号与执行清单

本清单对应正式论文消融实验。所有新实验只以
`Grid_Mamba/experiments/runs/evuav/baseline/FULL_SC12/train_config.yaml`
为配置源，不参考开发阶段的其他消融目录。

正式输出根目录：

```text
Grid_Mamba/experiments/runs/evuav/ablation/hlc2/
```

统一训练协议为 seed 37、50 epochs、相同 EVUAV 数据划分和 evaluation
script。每个 variant 分别评估 `best_iou_seed37.pt` 与
`best_loss_seed37.pt`，最终报告所有 threshold 中 IoU 最高的一行。

## Local ST scale 定义

| 名称 | `scale_strides` |
|---|---|
| Fine | `[24.0, 24.0, 100.0]` |
| Medium | `[48.0, 48.0, 200.0]` |
| Coarse | `[128.0, 128.0, 400.0]` |

## MC：Main Components

| ID | 论文 variant | 关键代码设置 | 执行方式 |
|---|---|---|---|
| `MC01` | SubMConv only | `use_local_mamba=false`, grid position off, SWC off | 新训练 |
| `MC02` | Local ST sequence modeling, single scale | 只使用 **Medium** scale，grid position off，SWC off | 新训练 |
| `MC03` | Multi-scale local ST modeling | Fine+Medium+Coarse，grid position off，SWC off | 新训练 |
| `MC04` | + Grid-relative position encoding | Full multi-scale，grid position on，SWC off | 新训练 |
| `MC05` | + Historical context propagation | SWC on，`prev_context` temporal-cell diffusion on，aligned injection on，后置 context-map spatial conv off | 新训练 |
| `MC06` | Full HLC² | 指定的 `SC12_GS_G4_FINE_LOW_MID` | 只读引用，不重训 |

`MC02` 的精确覆盖为：

```yaml
GRID_MAMBA:
  use_local_mamba: true
  scale_strides:
    - [48.0, 48.0, 200.0]
  use_grid_pos_encoding: false
  use_spatial_window_context: false
  use_temporal_cell_diffusion: false
  spatial_context_use_conv: false
```

## LS：Local Spatiotemporal Scale

除 `scale_strides` 外，其余设置均保持 FULL。

| ID | Scale 组合 | Canonical ID | 执行方式 |
|---|---|---|---|
| `LS01` | Fine | `LS01` | 新训练 |
| `LS02` | Medium | `LS02` | 新训练 |
| `LS03` | Coarse | `LS03` | 新训练 |
| `LS04` | Fine + Medium | `LS04` | 新训练 |
| `LS05` | Fine + Coarse | `LS05` | 新训练 |
| `LS06` | Medium + Coarse | `LS06` | 新训练 |
| `LS07` | Fine + Medium + Coarse | `MC06` | FULL 别名 |

## CS：Context Cell Stride

除 `spatial_context_stride` 外，其余设置均保持 FULL。

| ID | Context stride | Canonical ID | 执行方式 |
|---|---:|---|---|
| `CS01` | 4 | `CS01` | 新训练 |
| `CS02` | 8 | `MC06` | FULL 别名 |
| `CS03` | 16 | `CS03` | 新训练 |
| `CS04` | 32 | `CS04` | 新训练 |
| `CS05` | 64 | `CS05` | 新训练 |

## CP：Context Propagation Design

Position-aligned context injection 没有独立开关；只要启用 SWC，context
就会按 spatial cell 索引残差注入 event-level features。

| ID | Context 设计 | Canonical ID | 执行方式 |
|---|---|---|---|
| `CP01` | No context | `MC04` | `MC04` 别名 |
| `CP02` | Cell-wise Mamba + aligned injection；两种 diffusion 均关闭 | `CP02` | 新训练 |
| `CP03` | 仅启用 `prev_context` temporal-cell diffusion | `MC05` | `MC05` 别名 |
| `CP04` | 仅启用后置 context-map spatial conv | `CP04` | `rerun1`（论文采用） |
| `CP05` | 两种 propagation mechanism 同时启用 | `MC06` | FULL 别名 |

CP04 的论文正式结果固定来自
`Grid_Mamba/experiments/runs/evuav/ablation/hlc2/CP04/summary.json`：
`best_loss`、threshold `0.72`、IoU `89.96%`、ACC `94.35%`、
$P_d$ `94.63%`、$F_a$ `43.01\times10^{-4}`。原 CP04 结果仅作为历史记录保留，
不再用于论文报告。

## 执行接口

执行器：`Grid_Mamba/tools/experiments/evuav/run_hlc2_paper_ablation.py`

```bash
# 查看全部 23 个逻辑编号
python -m tools.experiments.evuav.run_hlc2_paper_ablation --list

# 只生成某一组配置，不训练
python -m tools.experiments.evuav.run_hlc2_paper_ablation \
  --group MC --stage config

# 执行单个实验的完整精度流程
python -m tools.experiments.evuav.run_hlc2_paper_ablation \
  --experiment MC01 --stage all

# 执行全部正式消融的精度流程；canonical alias 会自动去重
python -m tools.experiments.evuav.run_hlc2_paper_ablation \
  --all --stage all --continue-on-error

# 只准备 LS Runtime 清单（核对 checkpoint 并写入哈希，不执行 profiling）
python -m tools.experiments.evuav.run_hlc2_paper_ablation \
  --group LS --stage runtime_prepare

# 精度实验完成后，单独测量 LS Runtime
python -m tools.experiments.evuav.run_hlc2_paper_ablation \
  --group LS --stage runtime
```

`--stage all` 只包含 config、train、test、eval 和 summarize，不会自动执行
Runtime。Runtime 固定采用完整 EVUAV test split、FP32、warmup 1、repeat 3。
CS 组不统计 Runtime。

## 不在本执行器中的实验

- Window Size：已经在“Grid Mamba 时间窗口调整”中完成。本执行器不注册
  `WSxx`，不读取、不修改、不迁移原精度结果。Runtime 由独立的
  `tools/archive/runtime_legacy/run_window_size_runtime.py`
  按论文最终选定的七个模型来源统计。该协议现作为论文时期的历史实现保留。

```bash
# 只准备七个 Window Size Runtime 清单，不执行 profiling
python -m tools.archive.runtime_legacy.run_window_size_runtime \
  --all --stage prepare

# 后续真正执行 Runtime
python -m tools.archive.runtime_legacy.run_window_size_runtime \
  --all --stage runtime --continue-on-error
```

- Multi-scale 具体尺度值敏感性：暂时只保留 `SVxx` 编号前缀，待确定具体
  stride 组合后再建立独立 registry。

逻辑编号总数为 23；去除 FULL 和所有 alias 后，需要新训练的 canonical
experiments 总数为 17。
