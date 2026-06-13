import os
from configs.configs import cfg
import torch
import torch.nn as nn
import numpy as np
from dataset.ev_uav import EvUAV
from dataset.ev_flying import EvFlying
import random
from model.Grid_Mamba.grid_mamba_net import GridMambaNet
import mlflow
import tqdm
from utils.eval import evalute
import time
from pathlib import Path


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


def build_dataset(cfg, mode):
    dataset_name = str(getattr(cfg, "dataset_name", "ev_uav")).lower()
    if dataset_name == "ev_uav":
        return EvUAV(cfg, mode=mode)
    if dataset_name == "ev_flying":
        return EvFlying(cfg, mode=mode)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


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

    dataset = build_dataset(cfg, mode='test')
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=dataset.custom_collate
    )

    # 加载模型
    net.load_state_dict(torch.load(cfg.model_path))
    print('dict load: ', cfg.model_path)

    # 添加输出文件路径
    output_path = getattr(cfg, 'output_path', 'predictions.txt')
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm.tqdm(
        total=len(test_dataloader),
        unit="Batch",
        unit_scale=True,
        desc="Test",
        position=0,
        leave=True
    )

    evaluter = evalute(cfg)

    # 打开输出文件
    total_points = 0
    successful_batches = 0

    with open(output_path, 'w') as f_out:
        # 写入表头（参考 inference.py 格式）
        f_out.write("file_idx point_idx x y t gt pred prob\n")

        for sample, ev in enumerate(test_dataloader):
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
                
                # 计算概率和二值预测
                probs = torch.sigmoid(preds.reshape(-1)).cpu()  # 概率值 [N]
                pred_binary = (probs >= 0.9).long()  # 预测标签(0或1) [N]
                
                point_count = preds.shape[0]  # 点数量
                
                # 记录结束时间
                torch.cuda.synchronize()
                end_time = time.time()
                duration = end_time - start_time  # 测试时长（秒）
                
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
                
                # 保存结果到文件（参考 inference.py）
                try:
                    # 获取原始坐标（points已经是[x, y, t]格式）
                    valid_points = points.cpu().numpy()
                    valid_labels = label.cpu().numpy()
                    valid_probs = probs.cpu().numpy()
                    valid_preds = pred_binary.cpu().numpy()
                    
                    for point_idx, (point, gt, pred, prob) in enumerate(zip(
                        valid_points, valid_labels, valid_preds, valid_probs
                    )):
                        x, y, t = point[0], point[1], point[2]
                        f_out.write(f"{sample} {point_idx} {x:.6f} {y:.6f} {t:.6f} "
                                    f"{int(gt)} {int(pred)} {prob:.6f}\n")
                    
                    total_points += len(valid_points)
                    successful_batches += 1
                except Exception as e:
                    print(f"Error saving batch {sample}: {e}")
                    continue

            pbar.update(1)
            torch.cuda.empty_cache()

    pbar.close()

    # 输出统计信息
    print("\n=== 测试统计信息 ===")
    print(f"总样本数: {successful_batches}")
    print(f"总点数: {total_points}")

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
