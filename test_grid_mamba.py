import torch
from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.Grid_Mamba.grid_mamba_net import GridMambaNet
from utils.eval import evalute
import tqdm
import time

if __name__ == '__main__':
    device = "cuda:0"

    net = GridMambaNet(cfg).eval()
    net.cuda()

    dataset = EvUAV(cfg, mode='test')

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.custom_collate)

    net.load_state_dict(torch.load(cfg.model_path))
    print('dict load: ', cfg.model_path)

    pbar = tqdm.tqdm(total=len(test_dataloader), desc='video', unit='video', unit_scale=True, position=0, leave=True)

    evaluter = evalute(cfg)

    # 新增：记录列表
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

            # GridMambaNet 前向传播
            preds, _ = net(points)  # preds: [N, 1]
            
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
                    evaluter.roc_update(ts, preds, idx, label.cpu(), ev_locs)

        pbar.update(1)

    # 新增：输出统计信息
    print("\n=== 测试统计信息 ===")
    for record in record_list:
        print(f"Sample {record['sample']}: 时长={record['duration']:.4f}s, 点数={record['point_count']}")

    if cfg.eval:
        iou = evaluter.evaluate_semantic_segmantation_miou()
        seg_acc = evaluter.evaluate_semantic_segmantation_accuracy()
        if cfg.roc:
            pd, fa = evaluter.cal_roc()
        print('iou:{},seg_acc:{},pd:{},fa:{}'.format(iou, seg_acc, pd, fa))