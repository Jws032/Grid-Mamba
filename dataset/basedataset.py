import torch
import numpy as np

class BaseDataLoader(torch.utils.data.Dataset):
    """
    Base class for dataloader.
    """

    def __init__(self, configs):
        self.configs = configs
        self.root = configs.root
        self.whole_t = configs.whole_t
        self.res = configs.res

    @staticmethod
    def custom_collate(batch):
        batch_size = len(batch)
        points_batches = []
        seg_label_batches = []
        idx_label_batches = []
        file_names = []
        splits = []
        knn_cache_keys = []

        for i, ev in enumerate(batch):
            # 直接使用归一化的[x, y, t]作为点云坐标
            # evs_norm[:, 0:3] 包含 [x, y, t] 归一化坐标
            points = ev['points'][:, 0:3]
            points_batches.append(points)

            seg_label = ev['seg_label']
            seg_label_batches.append(seg_label)

            idx_label = ev['idx']
            idx_label_batches.append(idx_label)

            file_names.append(ev.get('file_name'))
            splits.append(ev.get('split'))
            knn_cache_keys.append(ev.get('knn_cache_key'))

        # 合并所有批次的数据
        points_batches = np.concatenate(points_batches, axis=0)
        seg_label_batches = np.concatenate(seg_label_batches, axis=0)
        idx_label_batches = np.concatenate(idx_label_batches, axis=0)

        # 转换为PyTorch张量
        points_tensor = torch.from_numpy(points_batches).float().contiguous()
        seg_label_tensor = torch.from_numpy(seg_label_batches).float().contiguous()
        idx_label_tensor = torch.from_numpy(idx_label_batches).float().contiguous()

        output = {}
        output['points'] = points_tensor  # [N, 3] 归一化的[x, y, t]
        output['seg_label'] = seg_label_tensor  # [N]
        output['idx_label'] = idx_label_tensor  # [N]
        output['file_name'] = file_names
        output['split'] = splits
        output['knn_cache_key'] = knn_cache_keys

        return output
