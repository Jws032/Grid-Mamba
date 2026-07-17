import os
from configs.configs import cfg
import torch
import numpy as np
from dataset.basedataset import BaseDataLoader

class EvUAV(BaseDataLoader):
    def __init__(self, configs, mode='train'):
        super().__init__(configs)

        self.mode = mode
        self.root = os.path.join(self.root,mode)
        self.file_list = sorted(
            file_name
            for file_name in os.listdir(self.root)
            if file_name.endswith('.npz')
        )

    def __getitem__(self, sample_idx):
        file_name = self.file_list[sample_idx]
        events = np.load(os.path.join(self.root,file_name))
        evs_norm,ev_loc,seg_label,idx= events['evs_norm'][:,0:4],events['ev_loc'],events['evs_norm'][:,4],events['evs_norm'][:,5]
        knn_cache_key = f"{self.mode}/{os.path.splitext(file_name)[0]}"

        if self.mode=='train':
            num_events = ev_loc.shape[0]
            if num_events >= cfg.max_events_num:
                dowmsample_idx = np.random.choice(num_events,cfg.max_events_num,replace=False)
                ev_loc = ev_loc[dowmsample_idx]
                evs_norm=evs_norm[dowmsample_idx]
                seg_label = seg_label[dowmsample_idx]
                idx = idx[dowmsample_idx]
                knn_cache_key = None
                print('downsample')

        out={}
        # 修改：使用原始坐标 ev_loc 而不是归一化坐标 evs_norm
        # ev_loc 格式应该是 [x, y, t] 的原始坐标
        out['points'] = ev_loc[:, 0:3]  # [x,y,t] 原始坐标，将在模型内进行归一化处理
        out['seg_label'] = seg_label  # [0,1]
        out['idx'] = idx
        out['file_name'] = file_name
        out['split'] = self.mode
        out['knn_cache_key'] = knn_cache_key

        return out


    def __len__(self):
        return len(self.file_list)
