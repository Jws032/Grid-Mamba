import torch.nn as nn


class PointHead(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
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
            logits: [N, num_classes] 预测分数
        """
        return self.mlp(combined_feat)