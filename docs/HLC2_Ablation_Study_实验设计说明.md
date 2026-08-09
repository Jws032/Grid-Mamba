# HLC2 Ablation Study 实验设计说明

本文档用于指导 HLC$^2$ 论文的消融实验设计。实验设计应从论文 Method 的建模逻辑出发，而不是从代码开关出发。核心目标是回答：为什么仅依赖微观事件结构不够，为什么需要局部时空线索建模，为什么需要历史上下文，以及尺度选择是否合理。

## 总体评价指标

所有消融实验建议统一报告以下指标：

- IoU：主指标，衡量 event-level foreground/background segmentation 的整体质量。
- ACC：事件级分类准确率。
- $P_d$：目标事件检出能力，反映 foreground recall。
- $F_a$：虚警率，反映 background suppression。
- Runtime：仅在 Local ST scale 和 window size 两组实验中报告；Context Cell Size 组不统计 Runtime。

阈值选择、测试集划分、训练 schedule 和 evaluation protocol 应与主 SOTA 对比保持一致。除被消融的因素外，其余设置固定为最终模型设置。

## Table 1: Main Component Ablation

### 目的

这张表是最重要的主消融表。它从最弱的微观局部结构 baseline 开始，逐步加入 HLC$^2$ 的关键设计，用于证明：

1. 仅依赖 SubMConv 捕获的微观邻域事件结构不足以支持 event-level tiny object segmentation。
2. 当前窗口内的 local spatiotemporal sequence modeling 可以形成更可靠的局部运动线索。
3. Multi-scale local ST modeling 可以覆盖不同速度和不同运动范围的小目标。
4. Grid-relative position encoding 有助于保留 local ST region 内的细粒度事件位置。
5. Historical context propagation 可以利用 trajectory-level temporal coherence。
6. Full HLC$^2$ 通过 spatial diffusion 和 position-aligned context injection 进一步稳定事件级预测。

### 推荐表格形式

| Variant | Description | IoU | ACC | $P_d$ | $F_a$ |
|---|---|---:|---:|---:|---:|
| SubMConv only | Only aggregates microscopic neighboring events. |  |  |  |  |
| + Local ST sequence modeling (single scale) | Models event dependencies within local spatiotemporal regions. |  |  |  |  |
| + Multi-scale local ST modeling (Section III-C 的 encoder, without Grid-relative position encoding) | Uses multiple local ST region sizes to capture different motion extents. |  |  |  |  |
| + Grid-relative position encoding | Preserves fine event positions inside each local ST region. |  |  |  |  |
| + Historical context propagation | Propagates recurrent context across consecutive windows. |  |  |  |  |
| Full HLC$^2$ (+spatial diffusion) | Adds spatial diffusion and position-aligned context injection. |  |  |  |  |

### 术语说明

`Local ST sequence modeling` 指：在当前 event window 内，按照 event 的 $(x,y,t)$ 坐标将事件划分到局部 spatiotemporal regions 中，只在同一个局部区域内部进行 sequence modeling。这里的 grid 是 Method 中的 local spatiotemporal grid，不是 context cell。

`Full HLC$^2$` 相比 `+ Historical context propagation` 的区别是：历史上下文不只在同一 spatial cell 上递推，还通过 spatial diffusion 扩散到邻近 cells，并通过 position-aligned context injection 注入回 event-level features。

## Table 2: Local Spatiotemporal Scale Ablation

### 目的

这张表分析 local ST grid 的尺度选择。它回答：当前窗口内的局部建模单元应该多大？尺度太小可能把目标轨迹切得过碎；尺度太大可能把目标事件和背景事件混在一起。Multi-scale 设计是否优于单一尺度？

### 推荐表格形式

| Variant | Fine ST grid | Medium ST grid | Coarse ST grid | IoU | ACC | $P_d$ | $F_a$ | Runtime |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| Fine only | yes |  |  |  |  |  |  |  |
| Medium only |  | yes |  |  |  |  |  |  |
| Coarse only |  |  | yes |  |  |  |  |  |
| Fine + Medium | yes | yes |  |  |  |  |  |  |
| Fine + Coarse | yes |  | yes |  |  |  |  |  |
| Medium + Coarse |  | yes | yes |  |  |  |  |  |
| Full multi-scale | yes | yes | yes |  |  |  |  |  |

### 写作解释重点

- Fine scale 更能保留细粒度 event-level positions，但局部运动上下文有限。
- Coarse scale 能覆盖更长的局部轨迹片段，但更容易引入背景或其他目标事件。
- Multi-scale local ST modeling 同时覆盖 fine-grained event evidence 和不同运动范围的小目标，因此应当更稳定。

表格 caption 中可以说明：fine, medium, and coarse denote increasingly larger local spatiotemporal regions. 具体 stride 数值可放在 Implementation Details 或 caption 中。

## Table 3: Context Cell Size Ablation

### 目的

这张表分析 Window-to-Cell Pooling 中 context cell 的空间分辨率。它回答：跨窗口历史上下文应该维护在多粗的空间网格上？

Local ST grid 和 context cell 是两个不同概念：

- Local ST grid：当前窗口内的三维 $(x,y,t)$ 局部序列建模单元，用于 local cue extraction。
- Context cell：跨窗口历史上下文传播的二维 spatial cell，用于 Window-to-Cell Pooling、Cell-wise Mamba 和 context injection。

### 推荐表格形式

| Cell stride / cell size | Interpretation | IoU | ACC | $P_d$ | $F_a$ |
|---|---|---:|---:|---:|---:|
| Small | high-resolution context map |  |  |  |  |
| Medium | balanced context map |  |  |  |  |
| Large | low-resolution context map |  |  |  |  |

如果实验中使用明确的 stride 数值，可以写成：

| Cell stride | IoU | ACC | $P_d$ | $F_a$ |
|---:|---:|---:|---:|---:|
| 4 |  |  |  |  |
| 8 |  |  |  |  |
| 16 |  |  |  |  |
| 32 |  |  |  |  |
| 64 |  |  |  |  |

### 写作解释重点

- Cell size 太小：每个 cell 内 foreground events 更稀疏，历史状态可能不稳定，计算成本也更高。
- Cell size 太大：目标和背景更容易被聚合到同一个 context cell，削弱 event-level discrimination。
- 中等 cell size 通常在 context stability、spatial precision 和 efficiency 之间取得更好平衡。

## Table 4: multi-scale 不同scale选择的影响

### 目的

这张表是可选细节消融，分析multi-scale grid，具体的scale选择的影响，这块先做一些消融实验看看结果吧，这块如果要做的很细，工作量会很大，且结果容易不好解释。


## Figure 1: Temporal Window Size Sensitivity

### 目的

这张图分析输入 event window 的时间长度对结果的影响。它与 Table 2 和 Table 3 互补：

- Window size：输入窗口的时间跨度。
- Local ST scale：窗口内部的局部时空建模尺度。
- Context cell size：历史上下文传播的空间分辨率。

### 推荐图内容

x-axis: window size，例如 50, 100, 200, 300, 400, 800, 1600 ms。

left y-axis: IoU, ACC, $P_d$。

right y-axis: $F_a$。

### 写作解释重点

- HLC$^2$ 在较宽 window-size 范围内保持稳定，说明方法不是依赖某个固定 temporal aggregation scale。
- 过短窗口可能导致局部目标事件不足，局部线索不完整。
- 过长窗口可能引入更多背景事件，从而增大 false alarm。
- 中等窗口通常更好地平衡 foreground evidence accumulation 和 background suppression。

## 建议的 Ablation Studies 小节结构

推荐在论文中按以下顺序组织：

```latex
\subsubsection{Effect of Main Components}
\subsubsection{Effect of Local Spatiotemporal Scale}
\subsubsection{Effect of Context Cell Size}
\subsubsection{Effect of Context Propagation Design}
\subsubsection{Sensitivity to Temporal Window Size}
```

如果版面紧张，优先保留：

1. Main Component Ablation。
2. Local Spatiotemporal Scale Ablation。
3. Context Cell Size Ablation。
4. Temporal Window Size Sensitivity。

Context Propagation Design 可以作为补充实验，或在已有结果充分时放入主文。

## 给实验执行者的注意事项

- 每组实验只改变一个因素，其余设置保持 Full HLC$^2$ 一致。
- 所有结果尽量使用同一随机种子、同一训练轮数、同一数据划分和同一 evaluation script。
- 每个表格至少报告 IoU、ACC、$P_d$、$F_a$。
- Runtime 只填入 Local ST scale 和 window size 两组表格，Context Cell Size 组不报告。
- 如果某个 variant 难以严格实现，应在记录中说明实际替代方式，避免论文中写成不准确的机制消融。
- 表格中的 variant 名称应尽量使用 Method 术语，例如 local ST sequence modeling、historical context propagation、spatial diffusion、position-aligned context injection，而不是代码变量名。
