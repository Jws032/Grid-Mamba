import os
import shutil
import itertools
from configs.configs import cfg
import torch
import torch.nn as nn
import numpy as np
from dataset.ev_uav import EvUAV
from dataset.ev_flying import EvFlying
from dataset.fred_segmentation import FredSegmentation

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
    if os.path.abspath(config_path) == os.path.abspath(target_path):
        return

    shutil.copy2(config_path, target_path)


def build_dataset(mode):
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


def latest_train_state_path(seed):
    return os.path.join(cfg.model_save_root, f"latest_train_state_seed{seed}.pt")


def configured_resume_path(seed):
    path = str(getattr(cfg, "resume_path", "") or "").strip()
    if not path:
        path = latest_train_state_path(seed)
    return os.path.expanduser(path)


def save_train_state(
    path,
    *,
    epoch,
    net,
    optimizer,
    scheduler,
    best_loss,
    best_iou,
    best_val_loss,
    no_improve_epoch,
    dataloader_generator,
    train_loss,
    val_loss,
    val_iou,
    seed,
):
    state = {
        "epoch": int(epoch),
        "seed": int(seed),
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": float(best_loss),
        "best_iou": float(best_iou),
        "best_val_loss": float(best_val_loss),
        "no_improve_epoch": int(no_improve_epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_iou": float(val_iou),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "dataloader_generator_state": dataloader_generator.get_state(),
        "cfg_epochs": int(cfg.epochs),
    }
    if torch.cuda.is_available():
        state["cuda_random_state_all"] = torch.cuda.get_rng_state_all()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_train_state(
    path,
    *,
    net,
    optimizer,
    scheduler,
    dataloader_generator,
    device,
):
    if not os.path.exists(path):
        raise FileNotFoundError(f"resume checkpoint not found: {path}")

    state = torch.load(path, map_location=device)
    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }
    missing = sorted(required - set(state.keys()))
    if missing:
        raise ValueError(
            f"resume checkpoint is missing {missing}; "
            "expected an epoch-level training state, not a model-only checkpoint."
        )

    net.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])

    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "numpy_random_state" in state:
        np.random.set_state(state["numpy_random_state"])
    if "torch_random_state" in state:
        torch.set_rng_state(state["torch_random_state"].cpu())
    if "cuda_random_state_all" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [rng_state.cpu() for rng_state in state["cuda_random_state_all"]]
        )
    if "dataloader_generator_state" in state:
        dataloader_generator.set_state(state["dataloader_generator_state"].cpu())

    start_epoch = int(state["epoch"]) + 1
    return {
        "start_epoch": start_epoch,
        "best_loss": float(state.get("best_loss", 1e5)),
        "best_iou": float(state.get("best_iou", 0)),
        "best_val_loss": float(state.get("best_val_loss", 1e5)),
        "no_improve_epoch": int(state.get("no_improve_epoch", 0)),
    }


def build_optimizer(net):
    optim_name = str(getattr(cfg, "optim", "Adam")).lower()
    lr = float(cfg.lr)
    weight_decay = float(getattr(cfg, "weight_decay", 0.0))
    params = filter(lambda p: p.requires_grad, net.parameters())

    if optim_name == "adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim_name == "adamw":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optim_name == "sgd":
        momentum = float(getattr(cfg, "momentum", 0.9))
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {getattr(cfg, 'optim', None)}. "
            "Expected Adam, AdamW, or SGD."
        )

    print(
        f"optimizer: {optimizer.__class__.__name__}, "
        f"lr={lr}, weight_decay={weight_decay}"
    )
    return optimizer


def build_bce_loss(device):
    loss_pos_weight = getattr(cfg, "loss_pos_weight", None)
    if loss_pos_weight is None:
        print("loss: BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss(reduction='none')

    loss_pos_weight = float(loss_pos_weight)
    if loss_pos_weight <= 0:
        raise ValueError("loss_pos_weight must be positive when set")

    pos_weight = torch.tensor([loss_pos_weight], device=device, dtype=torch.float32)
    print(f"loss: BCEWithLogitsLoss, pos_weight={loss_pos_weight}")
    return nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)


def compute_loss(preds, label, loss_fn):
    element_loss = loss_fn(preds.float(), label)

    valid_mask = ~torch.isnan(element_loss) & ~torch.isinf(element_loss)
    if valid_mask.sum() == 0:
        return None

    loss = element_loss[valid_mask].mean()
    if torch.isnan(loss) or torch.isinf(loss):
        return None
    return loss


def build_scheduler(optimizer):
    scheduler_t_max = int(getattr(cfg, "scheduler_t_max", 100))
    scheduler_eta_min = float(getattr(cfg, "scheduler_eta_min", 1e-6))
    if scheduler_t_max <= 0:
        raise ValueError("scheduler_t_max must be positive")

    print(
        f"scheduler: CosineAnnealingLR, "
        f"T_max={scheduler_t_max}, eta_min={scheduler_eta_min}"
    )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_t_max,
        eta_min=scheduler_eta_min,
    )


if __name__ == '__main__':
    seed = 37
    setup(seed)
    device = torch.device("cuda:0")
    use_amp = bool(getattr(cfg, 'use_amp', False)) and device.type == 'cuda'
    amp_dtype = get_amp_dtype()
    empty_cache_every_batch = bool(getattr(cfg, 'empty_cache_every_batch', False))
    num_workers = int(getattr(cfg, 'train_workers', 0))
    train_shuffle = bool(getattr(cfg, 'train_shuffle', True))
    train_limit_batches = int(getattr(cfg, 'train_limit_batches', 0))
    val_limit_batches = int(getattr(cfg, 'val_limit_batches', 0))
    train_window_backward_chunk_size = int(
        getattr(cfg, 'train_window_backward_chunk_size', 0)
    )
    if train_window_backward_chunk_size < 0:
        raise ValueError("train_window_backward_chunk_size must be >= 0")
    use_persistent_workers = (
        num_workers > 0
        and train_limit_batches <= 0
        and val_limit_batches <= 0
    )
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
        persistent_workers=use_persistent_workers,
        generator=dataloader_generator,
    )

    optimizer = build_optimizer(net)
    
    scheduler = build_scheduler(optimizer)

    best_loss = 1e5
    best_iou = 0

    # ===== early stopping =====
    use_early_stopping = bool(getattr(cfg, 'early_stopping', True))
    best_val_loss = 1e5
    patience = int(getattr(cfg, 'early_stopping_patience', 20))
    no_improve_epoch = 0
    start_epoch = 0

    resume_enabled = bool(getattr(cfg, 'resume', False))
    if resume_enabled:
        resume_path = configured_resume_path(seed)
        resume_state = load_train_state(
            resume_path,
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader_generator=dataloader_generator,
            device=device,
        )
        start_epoch = resume_state["start_epoch"]
        best_loss = resume_state["best_loss"]
        best_iou = resume_state["best_iou"]
        best_val_loss = resume_state["best_val_loss"]
        no_improve_epoch = resume_state["no_improve_epoch"]
        print(
            f"Resumed training from {resume_path}: "
            f"next_epoch={start_epoch}, best_loss={best_loss:.6f}, "
            f"best_iou={best_iou:.6f}, best_val_loss={best_val_loss:.6f}, "
            f"no_improve_epoch={no_improve_epoch}"
        )

    # ===== val =====
    val_dataset = build_dataset(mode='val')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=val_dataset.custom_collate,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
    )
    evaluter = evalute(cfg)
    loss_fn = build_bce_loss(device)

    # mlflow
    mlflow.set_experiment('train')
    mlflow.start_run(run_name='train')

    if start_epoch >= cfg.epochs:
        print(
            f"Resume checkpoint already reached epoch {start_epoch - 1}; "
            f"cfg.epochs={cfg.epochs}. No training epochs remain."
        )

    for epoch in range(start_epoch, cfg.epochs):
        net.train()
        train_loss_total = 0
        train_count = 0

        train_iter = train_dataloader
        train_total_batches = len(train_dataloader)
        if train_limit_batches > 0:
            train_iter = itertools.islice(train_dataloader, train_limit_batches)
            train_total_batches = min(train_total_batches, train_limit_batches)

        pbar = tqdm.tqdm(
            total=train_total_batches,
            unit="Batch",
            unit_scale=True,
            desc=f"Epoch: {epoch}",
            position=0,
            leave=True
        )

        for batch_idx, ev in enumerate(train_iter):
            points = ev['points'].float().to(device, non_blocking=True)
            label = ev['seg_label'].float().to(device, non_blocking=True)
            knn_cache_key = get_single_knn_cache_key(ev)

            optimizer.zero_grad(set_to_none=True)

            if train_window_backward_chunk_size > 0:
                loss_value = 0.0
                valid_count = 0
                batch_failed = False
                chunk_iter = net.iter_forward_window_chunks(
                    points,
                    chunk_size=train_window_backward_chunk_size,
                    knn_cache_key=knn_cache_key,
                )

                while True:
                    with torch.autocast(
                        device_type='cuda',
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        try:
                            preds, pred_indices, _ = next(chunk_iter)
                        except StopIteration:
                            break

                    # ===== NaN 检查 =====
                    if torch.isnan(preds).any() or torch.isinf(preds).any():
                        print(f"Warning: train output contains NaN/Inf at epoch={epoch}, batch={batch_idx}!")
                        batch_failed = True
                        break

                    # ===== loss =====
                    chunk_label = label[pred_indices]
                    element_loss = loss_fn(preds.float(), chunk_label)
                    valid_mask = ~torch.isnan(element_loss) & ~torch.isinf(element_loss)
                    chunk_valid_count = int(valid_mask.sum().item())
                    if chunk_valid_count == 0:
                        continue

                    loss = element_loss[valid_mask].sum() / max(int(label.numel()), 1)
                    if torch.isnan(loss) or torch.isinf(loss):
                        batch_failed = True
                        break

                    loss.backward()
                    loss_value += float(loss.detach().item())
                    valid_count += chunk_valid_count

                if batch_failed or valid_count == 0:
                    optimizer.zero_grad(set_to_none=True)
                    continue
            else:
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
                loss = compute_loss(preds, label, loss_fn)
                if loss is None:
                    continue

                loss.backward()
                loss_value = loss.item()

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

            val_iter = val_dataloader
            if val_limit_batches > 0:
                val_iter = itertools.islice(val_dataloader, val_limit_batches)

            for sample, ev in enumerate(val_iter):
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
                loss = compute_loss(preds, label, loss_fn)
                if loss is None:
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
            should_stop = False
            if use_early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epoch = 0
                else:
                    no_improve_epoch += 1

                if no_improve_epoch >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    should_stop = True

            save_train_state(
                latest_train_state_path(seed),
                epoch=epoch,
                net=net,
                optimizer=optimizer,
                scheduler=scheduler,
                best_loss=best_loss,
                best_iou=best_iou,
                best_val_loss=best_val_loss,
                no_improve_epoch=no_improve_epoch,
                dataloader_generator=dataloader_generator,
                train_loss=train_loss,
                val_loss=val_loss,
                val_iou=iou.item(),
                seed=seed,
            )

            if should_stop:
                break
