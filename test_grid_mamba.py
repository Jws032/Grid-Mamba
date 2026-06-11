import os
from configs.configs import cfg
import torch
import torch.nn as nn
import numpy as np
from dataset.ev_uav import EvUAV
import random
from model.Grid_Mamba.grid_mamba_net import GridMambaNet
import mlflow
import tqdm
from utils.eval import evalute
import time


def setup(seed):
    """与train_grid_mamba.py保持一致的设置"""
    seed_n = seed
    print('random seed:' + str(seed_n))
    g = torch.Generator()
    g.manual_seed(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # 修改：使用warn_only=True允许非确定性操作
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed_n)


def get_single_knn_cache_key(batch):
    keys = batch.get("knn_cache_key")
    if isinstance(keys, (list, tuple)) and len(keys) == 1:
        return keys[0]
    return None


if __name__ == '__main__':
    seed = 37
    setup(seed)
    device = "cuda:0"

    net = GridMambaNet(cfg).eval()
    net.cuda()

    dataset = EvUAV(cfg, mode='test')
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=dataset.custom_collate
    )

    # 加载模型
    net.load_state_dict(torch.load(cfg.model_path))
    print('dict load: ', cfg.model_path)

    pbar = tqdm.tqdm(
        total=len(test_dataloader),
        unit="Batch",
        unit_scale=True,
        desc="Test",
        position=0,
        leave=True
    )

    evaluter = evalute(cfg)

    # 记录列表
    record_list = []

    for sample, ev in enumerate(test_dataloader):
        # 记录开始时间（同步 CUDA 确保准确计时）
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            # 直接使用points字段，已经是归一化的[x, y, t]格式
            points = ev['points'].float().cuda()  # [N, 3]
            label = ev['seg_label'].float().cuda()
            idx = ev['idx_label']
            knn_cache_key = get_single_knn_cache_key(ev)

            # GridMambaNet 前向传播
            preds, _ = net(points, knn_cache_key=knn_cache_key)  # preds: [N, 1]
            
            point_count = preds.shape[0]  # 点数量
            
            # 记录结束时间
            torch.cuda.synchronize()
            end_time = time.time()
            duration = end_time - start_time  # 测试时长（秒）
            
            # 新增：记录当前 sample 的信息
            record_list.append({
                'sample': sample,
                'duration': duration,
                'point_count': point_count
            })
            
            if cfg.eval:
                evaluter.matches[str(sample)] = {}
                evaluter.matches[str(sample)]['seg_pred'] = preds.cpu()
                evaluter.matches[str(sample)]['seg_gt'] = label.cpu()
                if cfg.roc:
                    # 注意：GridMambaNet没有直接提供时间戳，需要从points中提取
                    ts = points[:, 2].cpu()  # 提取时间戳
                    ev_locs = points.cpu()   # 使用points作为位置信息
                    # 确保所有张量都在CPU上，并处理坐标边界问题
                    try:
                        evaluter.roc_update(ts, preds.cpu(), idx, label.cpu(), ev_locs)
                    except IndexError as e:
                        print(f"Warning: IndexError in roc_update for sample {sample}: {e}")
                        # 跳过ROC计算，但继续其他评估

        pbar.update(1)
        torch.cuda.empty_cache()

    pbar.close()


    if cfg.eval:
        iou = evaluter.evaluate_semantic_segmantation_miou()
        seg_acc = evaluter.evaluate_semantic_segmantation_accuracy()
        pd, fa = None, None
        if cfg.roc:
            try:
                pd, fa = evaluter.cal_roc()
            except Exception as e:
                print(f"Warning: ROC calculation failed: {e}")
                pd, fa = None, None
        
        print('iou:{},seg_acc:{},pd:{},fa:{}'.format(iou, seg_acc, pd, fa))
