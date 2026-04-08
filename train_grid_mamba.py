import os
from configs.configs import cfg
import torch
import torch.nn as nn
import numpy as np
from dataset.ev_uav import EvUAV
import random
from model.Grid_Mamba.grid_mamba_net import GridMambaNet

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
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed_n)

if __name__ == '__main__':
    seed = 37
    setup(seed)
    device = "cuda:0"

    net = GridMambaNet(cfg).train().cuda()

    dataset = EvUAV(cfg, mode='train')
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=dataset.custom_collate
    )

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=cfg.lr
    )
    
    # 替换掉原来的 StepLR
    # T_max 设为总 Epoch 数，eta_min 设为最小学习率 (建议 1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    best_loss = 1e5
    best_iou = 0

    # ===== 新增：early stopping =====
    best_val_loss = 1e5
    patience = 20
    no_improve_epoch = 0

    # ===== val =====
    val_dataset = EvUAV(cfg, mode='val')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=val_dataset.custom_collate
    )
    evaluter = evalute(cfg)

    # mlflow
    mlflow.set_experiment('train')
    mlflow.start_run(run_name='train')

    for epoch in range(cfg.epochs):
        net.train()

        pbar = tqdm.tqdm(
            total=len(train_dataloader),
            unit="Batch",
            unit_scale=True,
            desc=f"Epoch: {epoch}",
            position=0,
            leave=True
        )

        for batch_idx, ev in enumerate(train_dataloader):
            points = ev['points'].float().cuda()
            label = ev['seg_label'].float().cuda()

            preds, _ = net(points)

            # ===== NaN 检查 =====
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                print("Warning: model output contains NaN/Inf!")
                continue

            # ===== loss =====
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            element_loss = loss_fn(preds, label)

            valid_mask = ~torch.isnan(element_loss) & ~torch.isinf(element_loss)
            if valid_mask.sum() == 0:
                continue

            loss = element_loss[valid_mask].mean()

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            # ===== 梯度检查 =====
            has_nan_grad = False
            for param in net.parameters():
                if param.grad is not None and (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                ):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                optimizer.zero_grad()
                continue

            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

            with torch.no_grad():
                mlflow.log_metric('train_loss', loss.item(), step=epoch)

                if loss.item() < best_loss:
                    torch.save(
                        net.state_dict(),
                        cfg.model_save_root + f'/best_loss_seed{seed}.pt'
                    )
                    best_loss = loss.item()

            torch.cuda.empty_cache()

        scheduler.step()

        # =========================
        # ===== 验证（每个epoch）=====
        # =========================
        net.eval()

        with torch.no_grad():
            val_loss_total = 0
            val_count = 0
            loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

            evaluter.matches = {}  # 清空

            for sample, ev in enumerate(val_dataloader):
                points = ev['points'].float().cuda()
                label = ev['seg_label'].float().cuda()

                preds, _ = net(points)

                if preds.shape[0] != label.shape[0]:
                    continue

                # ===== val loss =====
                loss = loss_fn(preds, label)
                val_loss_total += loss.item()
                val_count += 1

                # ===== eval =====
                evaluter.matches[str(sample)] = {}
                evaluter.matches[str(sample)]['seg_pred'] = preds.cpu()
                evaluter.matches[str(sample)]['seg_gt'] = label.cpu()

            if val_count > 0:
                val_loss = val_loss_total / val_count
            else:
                val_loss = 0

            mlflow.log_metric('val_loss', val_loss, step=epoch)

            iou = evaluter.evaluate_semantic_segmantation_miou()
            mlflow.log_metric('val_iou', iou.item(), step=epoch)

            print(f"\nEpoch {epoch} | Val Loss: {val_loss:.6f} | IoU: {iou.item():.6f}")

            # ===== 保存 best iou =====
            if iou.item() > best_iou:
                torch.save(
                    net.state_dict(),
                    cfg.model_save_root + f'/best_iou_seed{seed}.pt'
                )
                best_iou = iou.item()

            # ===== early stopping =====
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epoch = 0
            else:
                no_improve_epoch += 1

            if no_improve_epoch >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break