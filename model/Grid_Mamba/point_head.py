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
            logits: [N, 1] 预测分数，表示属于目标的概率分数
                如果num_classes=1，则输出形状为[N, 1]
                可以通过sigmoid激活函数转换为概率值
        """
        return self.mlp(combined_feat)