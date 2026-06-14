import os
import shutil
from configs.configs import cfg
import torch
import torch.nn as nn
import numpy as np
from dataset.ev_uav import EvUAV
from dataset.ev_flying import EvFlying

import random
from model.Grid_Mamba.grid_mamba_net import GridMambaNet

import torch.optim as optim
import mlflow
import tqdm
from utils.eval import evalute


def setup(seed):
    seed_n = seed
    print('random seed:' + str(seed_n))
    deterministic = bool(getattr(cfg, 'deterministic', True))
    cudnn_benchmark = bool(getattr(cfg, 'cudnn_benchmark', False))

    g = torch.Generator()
    g.manual_seed(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed_n)

def get_amp_dtype():
    amp_dtype = str(getattr(cfg, 'amp_dtype', 'bf16')).lower()
    if amp_dtype in {'bf16', 'bfloat16'}:
        return torch.bfloat16
    if amp_dtype in {'fp16', 'float16'}:
        return torch.float16
    raise ValueError("amp_dtype must be 'bf16' or 'fp16'")


def save_train_config_snapshot():
    config_path = getattr(cfg, "config", None)
    if config_path is None:
        return

    if not os.path.exists(config_path):
        print(f"Warning: config file not found, skip snapshot: {config_path}")
        return

    target_path = os.path.join(cfg.model_save_root, "train_config.yaml")
    shutil.copy2(config_path, target_path)


def build_dataset(mode):
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
    device = torch.device("cuda:0")
    use_amp = bool(getattr(cfg, 'use_amp', False)) and device.type == 'cuda'
    amp_dtype = get_amp_dtype()
    empty_cache_every_batch = bool(getattr(cfg, 'empty_cache_every_batch', False))
    num_workers = int(getattr(cfg, 'train_workers', 0))
    train_shuffle = bool(getattr(cfg, 'train_shuffle', True))
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed)

    os.makedirs(cfg.model_save_root, exist_ok=True)
    save_train_config_snapshot()

    net = GridMambaNet(cfg).train().to(device)

    dataset = build_dataset(mode='train')
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=dataset.custom_collate,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=dataloader_generator,
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

    # ===== early stopping =====
    use_early_stopping = bool(getattr(cfg, 'early_stopping', True))
    best_val_loss = 1e5
    patience = int(getattr(cfg, 'early_stopping_patience', 20))
    no_improve_epoch = 0

    # ===== val =====
    val_dataset = build_dataset(mode='val')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=val_dataset.custom_collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    evaluter = evalute(cfg)
    train_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    val_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    # mlflow
    mlflow.set_experiment('train')
    mlflow.start_run(run_name='train')

    for epoch in range(cfg.epochs):
        net.train()
        train_loss_total = 0
        train_count = 0

        pbar = tqdm.tqdm(
            total=len(train_dataloader),
            unit="Batch",
            unit_scale=True,
            desc=f"Epoch: {epoch}",
            position=0,
            leave=True
        )

        for batch_idx, ev in enumerate(train_dataloader):
            points = ev['points'].float().to(device, non_blocking=True)
            label = ev['seg_label'].float().to(device, non_blocking=True)
            knn_cache_key = get_single_knn_cache_key(ev)

            with torch.autocast(
                device_type='cuda',
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                preds, _ = net(points, knn_cache_key=knn_cache_key)

            # ===== NaN 检查 =====
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                print(f"Warning: train output contains NaN/Inf at epoch={epoch}, batch={batch_idx}!")
                continue

            # ===== loss =====
            element_loss = train_loss_fn(preds.float(), label)

            valid_mask = ~torch.isnan(element_loss) & ~torch.isinf(element_loss)
            if valid_mask.sum() == 0:
                continue

            loss = element_loss[valid_mask].mean()

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
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
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

            loss_value = loss.item()
            train_loss_total += loss_value
            train_count += 1
            train_loss = train_loss_total / train_count

            pbar.set_postfix(batch_loss=loss_value, train_loss=train_loss)
            pbar.update(1)

            with torch.no_grad():
                global_step = epoch * len(train_dataloader) + batch_idx
                mlflow.log_metric('train_batch_loss', loss_value, step=global_step)

            if empty_cache_every_batch:
                torch.cuda.empty_cache()

        scheduler.step()
        train_loss = train_loss_total / train_count if train_count > 0 else 0
        mlflow.log_metric('train_loss', train_loss, step=epoch)

        if train_count > 0 and train_loss < best_loss:
            torch.save(
                net.state_dict(),
                cfg.model_save_root + f'/best_loss_seed{seed}.pt'
            )
            best_loss = train_loss

        # =========================
        # ===== 验证（每个epoch）=====
        # =========================
        net.eval()

        with torch.no_grad():
            val_loss_total = 0
            val_count = 0
            evaluter.matches = {}  # 清空

            for sample, ev in enumerate(val_dataloader):
                points = ev['points'].float().to(device, non_blocking=True)
                label = ev['seg_label'].float().to(device, non_blocking=True)
                knn_cache_key = get_single_knn_cache_key(ev)

                with torch.autocast(
                    device_type='cuda',
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    preds, _ = net(points, knn_cache_key=knn_cache_key)

                if preds.shape[0] != label.shape[0]:
                    continue
                if torch.isnan(preds).any() or torch.isinf(preds).any():
                    print(f"Warning: val output contains NaN/Inf at epoch={epoch}, sample={sample}!")
                    continue

                # ===== val loss =====
                loss = val_loss_fn(preds.float(), label)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: val loss is NaN/Inf at epoch={epoch}, sample={sample}!")
                    continue
                val_loss_total += loss.item()
                val_count += 1

                # ===== eval =====
                evaluter.matches[str(sample)] = {}
                evaluter.matches[str(sample)]['seg_pred'] = preds.float().cpu()
                evaluter.matches[str(sample)]['seg_gt'] = label.cpu()

            if val_count > 0:
                val_loss = val_loss_total / val_count
            else:
                val_loss = 0

            mlflow.log_metric('val_loss', val_loss, step=epoch)

            iou = evaluter.evaluate_semantic_segmantation_miou()
            mlflow.log_metric('val_iou', iou.item(), step=epoch)

            print(
                f"\nEpoch {epoch} | Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | IoU: {iou.item():.6f}"
            )

            # ===== 保存 best iou =====
            if iou.item() > best_iou:
                torch.save(
                    net.state_dict(),
                    cfg.model_save_root + f'/best_iou_seed{seed}.pt'
                )
                best_iou = iou.item()

            # ===== early stopping =====
            if use_early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epoch = 0
                else:
                    no_improve_epoch += 1

                if no_improve_epoch >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
