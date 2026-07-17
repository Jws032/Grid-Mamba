import os
import json
import sys
from configs.configs import cfg
import json
import torch
import torch.nn as nn
import numpy as np
from dataset.ev_uav import EvUAV
from dataset.ev_flying import EvFlying
from dataset.fred_segmentation import FredSegmentation
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
    if dataset_name == "fred_segmentation":
        return FredSegmentation(cfg, mode=mode)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def get_single_knn_cache_key(batch):
    keys = batch.get("knn_cache_key")
    if isinstance(keys, (list, tuple)) and len(keys) == 1:
        return keys[0]
    return None


def get_single_value(batch, key):
    values = batch.get(key)
    if isinstance(values, (list, tuple)) and len(values) == 1:
        return values[0]
    return values


def cuda_synchronize_if_available():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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

    runtime_only = bool(getattr(cfg, "runtime_only", False))
    limit_test = getattr(cfg, "limit_test", None)
    if limit_test is not None:
        limit_test = int(limit_test)
        if limit_test <= 0:
            raise ValueError("--limit-test must be positive")

    if runtime_only:
        runtime_json = getattr(cfg, "runtime_json", None)
        if runtime_json is None:
            output_path = getattr(cfg, 'output_path', 'predictions.txt')
            runtime_json = str(Path(output_path).parent / "runtime_summary.json")
        runtime_json = Path(runtime_json)
        runtime_json.parent.mkdir(parents=True, exist_ok=True)
        per_sample_path = runtime_json.with_name("runtime_per_sample.jsonl")

        max_samples = len(test_dataloader)
        if limit_test is not None:
            max_samples = min(max_samples, limit_test)

        pbar = tqdm.tqdm(
            total=max_samples,
            unit="sample",
            desc="Runtime",
            position=0,
            leave=True,
        )

        total_points = 0
        successful_batches = 0
        iterator = iter(test_dataloader)
        per_sample_records = []

        cuda_synchronize_if_available()
        total_start = time.perf_counter()
        with per_sample_path.open("w", encoding="utf-8") as f_runtime:
            for sample in range(max_samples):
                cuda_synchronize_if_available()
                sample_start = time.perf_counter()
                ev = next(iterator)

                with torch.no_grad():
                    points = ev['points'].float().cuda()
                    knn_cache_key = get_single_knn_cache_key(ev)
                    preds, _ = net(points, knn_cache_key=knn_cache_key)
                    probs = torch.sigmoid(preds.reshape(-1))
                    pred_binary = probs >= 0.9
                    positive_points = int(pred_binary.sum().detach().cpu())
                    prob_mean = float(probs.mean().detach().cpu())
                    prob_max = float(probs.max().detach().cpu())
                    point_count = int(preds.shape[0])

                cuda_synchronize_if_available()
                sample_seconds = time.perf_counter() - sample_start

                total_points += point_count
                successful_batches += 1
                record = {
                    "file_idx": sample,
                    "file_name": get_single_value(ev, "file_name"),
                    "points": point_count,
                    "seconds": sample_seconds,
                    "points_per_sec": point_count / sample_seconds if sample_seconds > 0 else None,
                    "positive_points": positive_points,
                    "prob_mean": prob_mean,
                    "prob_max": prob_max,
                }
                per_sample_records.append(record)
                f_runtime.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_runtime.flush()
                pbar.update(1)
                torch.cuda.empty_cache()

        cuda_synchronize_if_available()
        total_inference_sec = time.perf_counter() - total_start
        pbar.close()

        summary = {
            "model": "GridMamba",
            "dataset": "FRED_segmentation",
            "split": "test",
            "num_samples": successful_batches,
            "num_points": total_points,
            "total_inference_sec": total_inference_sec,
            "runtime_sec_per_sample": (
                total_inference_sec / successful_batches if successful_batches else None
            ),
            "points_per_sec": total_points / total_inference_sec if total_inference_sec > 0 else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "device": device,
            "checkpoint": str(cfg.model_path),
            "config": getattr(cfg, "config", None),
            "runtime_per_sample": str(per_sample_path),
            "limit_test": limit_test,
        }
        with runtime_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        sys.exit(0)

    # 添加输出文件路径
    output_path = getattr(cfg, 'output_path', 'predictions.txt')
    prediction_threshold = float(getattr(cfg, 'prediction_threshold', 0.9))
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_only = bool(getattr(cfg, 'runtime_only', False))
    limit_test = getattr(cfg, 'limit_test', None)
    runtime_json_path, runtime_sample_path = runtime_output_paths(output_dir)

    pbar = tqdm.tqdm(
        total=len(test_dataloader),
        unit="Batch",
        unit_scale=True,
        desc="Test",
        position=0,
        leave=True
    )

    evaluter = None if runtime_only else evalute(cfg)

    # 打开输出文件
    total_points = 0
    successful_batches = 0
    runtime_rows = []
    f_out = None if runtime_only else open(output_path, 'w')
    try:
        if f_out is not None:
            # 写入表头（参考 inference.py 格式）
            f_out.write("file_idx point_idx x y t gt pred prob file_name\n")

        cuda_synchronize()
        runtime_start = time.perf_counter()
        for sample, ev in enumerate(test_dataloader):
            if limit_test is not None and sample >= int(limit_test):
                break
            cuda_synchronize()
            sample_start = time.perf_counter()
            
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
                pred_binary = (probs >= prediction_threshold).long()
                
                point_count = preds.shape[0]  # 点数量

                if cfg.eval and evaluter is not None:
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
                file_name = ev['file_name'][0]
                if f_out is not None:
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
                                        f"{int(gt)} {int(pred)} {prob:.6f} {file_name}\n")
                    except Exception as e:
                        print(f"Error saving batch {sample}: {e}")
                        continue

                cuda_synchronize()
                duration = time.perf_counter() - sample_start
                runtime_rows.append({
                    "file_idx": int(sample),
                    "file_name": str(file_name),
                    "points": int(point_count),
                    "seconds": float(duration),
                })
                total_points += int(point_count)
                successful_batches += 1

            pbar.update(1)
            if not runtime_only:
                torch.cuda.empty_cache()

        cuda_synchronize()
        total_runtime_sec = time.perf_counter() - runtime_start
    finally:
        if f_out is not None:
            f_out.close()

    pbar.close()

    # 输出统计信息
    print("\n=== 测试统计信息 ===")
    print(f"总样本数: {successful_batches}")
    print(f"总点数: {total_points}")
    if runtime_only:
        write_runtime_outputs(
            runtime_json_path,
            runtime_sample_path,
            runtime_rows,
            total_runtime_sec,
            total_points,
            cfg.model_path,
        )

    if cfg.eval and evaluter is not None:
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
