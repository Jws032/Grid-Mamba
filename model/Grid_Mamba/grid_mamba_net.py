import torch.nn as nn
from local_mamba_block import LocalMambaBlock
from global_vim_block import GlobalVimBlock
from point_head import PointHead
from event_score import temporal_peak_filter_fast_v3


class GridMambaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.ts_encoder = TSGraphEmbedding(...)
        self.local_mamba = LocalMambaBlock(...)
        self.global_vim = GlobalVimBlock(...)
        self.head = PointHead(...)

    def forward(self, points, prev_state=None):
        # points: [N, 3] (x, y, t)

        # 1. TS 图特征嵌入
        feat = self.ts_encoder(points)                      # [N, C]
        
        # 2. 局部 Mamba 模块
        grid_feat, point2grid = self.local_stage(points, feat)
        
        # 3. 全局 VIM 模块
        F, new_state = self.global_stage(grid_feat, prev_state)
        
        # 4. 分类头
        out = self.head(points, feat, grid_feat, F, point2grid)

        return out, new_state