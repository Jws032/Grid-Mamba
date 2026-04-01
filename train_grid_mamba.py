import os
from configs.configs import cfg
import torch
import torch.nn as nn
import numpy as np
from dataset.ev_uav import EvUAV
import random
# 替换模型导入
from model.Grid_Mamba.grid_mamba_net import GridMambaNet
# 删除不必要的STCLoss导入，因为实际使用CrossEntropyLoss
# from utils.stcloss import STCLoss

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
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.custom_collate)  # batch_size=1 for point cloud

    # 删除STCLoss的初始化，因为不使用
    # stc_criterion = STCLoss(k=cfg.k, t=cfg.t, cfg=cfg).cuda()

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
            # 直接使用points字段，已经是归一化的[x, y, t]格式
            points = ev['points'].float().cuda()  # [N, 3]
            
            label = ev['seg_label'].float().cuda()  # [N]

            # 调试：检查输入数据是否包含NaN或异常值
            if torch.isnan(points).any() or torch.isinf(points).any():
                print(f"Warning: points contains NaN/Inf! Sample shape: {points.shape}")
                continue
            if torch.isnan(label).any() or torch.isinf(label).any():
                print(f"Warning: label contains NaN/Inf! Sample shape: {label.shape}")
                continue
            
            # 严格验证标签值必须是0或1
            if not torch.all((label == 0) | (label == 1)):
                unique_labels = torch.unique(label)
                print(f"Warning: label contains invalid values! Unique values: {unique_labels}")
                # 强制转换为二进制标签
                label = (label > 0.5).float()

            # GridMambaNet 前向传播
            preds, _ = net(points)  # preds: [N, 1]

            # 调试：检查模型输出
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                print(f"Warning: model output contains NaN/Inf! Shape: {preds.shape}")
                continue

            # 计算损失 - 使用BCEWithLogitsLoss进行二分类
            # 使用reduction='none'进行安全计算
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            element_loss = loss_fn(preds.squeeze(1), label)
            
            # 过滤掉异常损失值
            valid_loss_mask = ~torch.isnan(element_loss) & ~torch.isinf(element_loss)
            if valid_loss_mask.sum() == 0:
                print("Warning: all loss elements are NaN/Inf!")
                continue
                
            loss = element_loss[valid_loss_mask].mean()

            # 调试：检查损失值
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: loss is NaN/Inf! Preds range: [{preds.min():.4f}, {preds.max():.4f}]")
                continue

            optimizer.zero_grad()
            loss.backward()

            # 关键修复：添加梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            # 调试：检查梯度是否包含NaN
            has_nan_grad = False
            for param in net.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                print("Warning: gradients contain NaN/Inf! Skipping update.")
                optimizer.zero_grad()  # 清除梯度，跳过这次更新
                continue
                
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
            if epoch >= 1:
                for sample, ev in enumerate(val_dataloader):
                    points = ev['points'].float().cuda()
                    
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
                    best_iou = iou.item()