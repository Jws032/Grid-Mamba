import os
from configs.configs import cfg
import torch
import torch.nn as nn
import numpy as np
from dataset.ev_uav import EvUAV
import random
# 替换模型导入
from model.Grid_Mamba.grid_mamba_net import GridMambaNet
from utils.stcloss import STCLoss

import torch.optim as optim
import mlflow
import tqdm
from utils.eval import evalute

def setup(seed):
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
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed_n)

if __name__ == '__main__':
    seed = 37
    setup(seed)
    device = "cuda:0"

    net = GridMambaNet(cfg).train()
    net.cuda()

    dataset = EvUAV(cfg, mode='train')
    train_sampler = torch.utils.data.sampler.RandomSampler(list(range(len(dataset))))
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.custom_collate, sampler=train_sampler)  # batch_size=1 for point cloud

    stc_criterion = STCLoss(k=cfg.k, t=cfg.t, cfg=cfg).cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = 1e5
    best_iou = 0

    # for val
    val_dataset = EvUAV(cfg, mode='val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, collate_fn=val_dataset.custom_collate)  # 注意这里应该是val_dataset
    evaluter = evalute(cfg)

    # mlflow
    mlflow.set_experiment('train')
    mlflow.start_run(run_name='train')

    for epoch in range(cfg.epochs):
        pbar = tqdm.tqdm(total=len(train_dataloader), unit="Batch", unit_scale=True,
                         desc="Epoch: {}".format(epoch), position=0, leave=True)

        for ev in train_dataloader:
            # GridMambaNet 期望的输入格式
            # 假设 ev['locs'] 包含 [x, y, t] 坐标，形状为 [N, 4] 其中第4维是时间戳
            ev_locs = ev['locs'].float().cuda()  # [N, 4]
            
            # 提取 x, y, t 坐标 (归一化到 [0,1])
            # 假设坐标已经是归一化的，如果不是需要进行归一化
            if ev_locs.shape[1] >= 4:
                # 假设前3维是 x, y, z，第4维是时间
                # 对于2D事件相机，z可能为0，我们使用 x, y, t
                points = ev_locs[:, [0, 1, 3]].clone()  # [N, 3] -> [x, y, t]
            else:
                points = ev_locs[:, :3].clone()  # [N, 3]
            
            # 确保坐标在 [0,1] 范围内（如果需要归一化）
            # 这里假设数据已经归一化，如果没有，请取消注释以下代码：
            # points[:, 0] = points[:, 0] / cfg.sensor_width  # x 归一化
            # points[:, 1] = points[:, 1] / cfg.sensor_height  # y 归一化  
            # points[:, 2] = points[:, 2] / cfg.max_time  # t 归一化
            
            label = ev['seg_label'].float().cuda()  # [N]

            # GridMambaNet 前向传播
            preds, _ = net(points)  # preds: [N, num_classes]

            # 计算损失 - 需要调整STCLoss以适应新的输出格式
            # 如果STCLoss期望特定格式，可能需要修改
            loss = nn.CrossEntropyLoss()(preds, label.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

            with torch.no_grad():
                mlflow.log_metric('loss', loss.item())
                if loss.item() < best_loss:
                    torch.save(net.state_dict(), cfg.model_save_root + '/best_loss_seed{}.pt'.format(seed))
                    best_loss = loss.item()
            torch.cuda.empty_cache()

        scheduler.step()

        with torch.no_grad():
            if epoch >= 40:
                for sample, ev in enumerate(val_dataloader):
                    ev_locs = ev['locs'].float().cuda()
                    
                    if ev_locs.shape[1] >= 4:
                        points = ev_locs[:, [0, 1, 3]].clone()
                    else:
                        points = ev_locs[:, :3].clone()
                    
                    label = ev['seg_label'].float().cuda()
                    idx = ev['idx_label']

                    preds, _ = net(points)
                    
                    # 确保预测结果格式正确
                    if preds.shape[0] != label.shape[0]:
                        # 如果预测和标签长度不匹配，可能需要插值或采样
                        # 这里假设它们应该匹配
                        continue
                    
                    evaluter.matches[str(sample)] = {}
                    evaluter.matches[str(sample)]['seg_pred'] = preds.cpu()
                    evaluter.matches[str(sample)]['seg_gt'] = label.cpu()
                
                iou = evaluter.evaluate_semantic_segmantation_miou()

                if iou.item() > best_iou:
                    torch.save(net.state_dict(), cfg.model_save_root + '/best_iou_seed{}.pt'.format(seed))
                    best_iou = iou.item()  # 修正：应该是best_iou而不是best_loss