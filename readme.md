# Grid Mamba Stream 模型

代码是基于 EVUAV 的开源代码上开发，有些冗余文件暂时保留。核心文件如下：

## 训练与测试

训练代码：  train_grid_mamba.py

测试采用两阶段测试
- 测试代码：    test_grid_mamba_cetus_style.py。输出一个predictions.txt文件。
- 指标评估：    evaluation/pixel_based_eval.py。根据predictions.txt文件，计算并输出评估指标。

## 数据输入处理

路径：dataset/



## 核心模型

模型文件夹：model/Grid_Mamba

1. tsgraph_embedding.py：
-  输入：原始点云
-  输出：结合 event_score.py 计算的event score，得到点云的嵌入向量。

2. grid_mamba.py: 构建 Grid Mamba 模型

3. local_mamba_block.py：每个Grid的点序列，输入该模块处理。

4. point_head.py：分类头


