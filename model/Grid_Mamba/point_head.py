import torch.nn as nn


class PointHead(nn.Module):
    def __init__(self, in_dim, num_classes=1):
        super().__init__()
        # 对于二分类任务，num_classes应为1，输出单个目标分数
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, combined_feat):
        """
        Args:
            combined_feat: [N, in_dim] 组合特征
        Returns:
            logits: [N] 预测分数，表示属于目标的概率分数
                输出形状为[N]，可以直接与标签[seg_label]进行比较
                可以通过sigmoid激活函数转换为概率值
        """
        output = self.mlp(combined_feat)
        # 将输出从[N, 1] squeeze为[N]，以匹配标签的形状
        if self.num_classes == 1:
            output = output.squeeze(-1)
        return output