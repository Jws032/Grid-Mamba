import torch
import torch.nn as nn

class PointHead(nn.Module):
    def __init__(self, in_dim, num_classes=1):
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("Grid Mamba only supports binary event segmentation")
        
        # 1. 尺度自适应注意力：动态学习三个尺度的权重
        # 维度收缩率设为 8，减少参数量
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 8, in_dim),
            nn.Sigmoid()
        )
        
        # 2. 深度 MLP
        self.fc1 = nn.Linear(in_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        
        self.fc3 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, combined_feat):
        """
        combined_feat: [N, in_dim] 多尺度特征拼接
        """
        # --- A. 尺度自适应分配 ---
        # 计算全局通道统计量，用于给不同尺度分配重要度
        global_context = combined_feat.mean(dim=0, keepdim=True) # [1, in_dim]
        attn_weights = self.attention(global_context) # [1, in_dim]
        x = combined_feat * attn_weights
        
        # --- B. 特征提炼层 ---
        # 第一层 (256维)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 第二层 (128维)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # --- C. 最终映射 ---
        output = self.fc3(x)
        
        return output.squeeze(-1)
