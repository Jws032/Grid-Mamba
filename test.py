import torch
from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from utils.eval import evalute
import tqdm
import time

if __name__ == '__main__':
    device = "cuda:0"

    net = evspsegnet(cfg).eval()
    net.cuda()

    dataset = EvUAV(cfg, mode='test')

    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size,collate_fn=dataset.custom_collate)

    net.load_state_dict(torch.load(cfg.model_path))
    print('dict load: ',cfg.model_path)


    pbar = tqdm.tqdm(total=len(test_dataloader), desc='video', unit='video',unit_scale=True,position=0, leave=True)

    evaluter = evalute(cfg)

    # 新增：记录列表
    record_list = []

    for sample,ev in enumerate(test_dataloader):
        # 记录开始时间（同步 CUDA 确保准确计时）
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            x = ev['voxel_ev']  #[x,y,t,p]
            label = ev['seg_label'].float().cuda()
            p2v_map = ev['p2v_map'].long().cuda()
            ev_locs = ev['locs'].float().requires_grad_()
            idx = ev['idx_label']
            ts = ev_locs[:,3]

            # 体素级预测
            preds, voxel = net(x)  
            voxel_count = preds.shape[0]  # 体素数量
            
            # 点级别预测:将体素预测映射回点
            preds = preds[p2v_map].squeeze().cpu()
            point_count = preds.shape[0]  # 点数量
            
            
            # 记录结束时间
            torch.cuda.synchronize()
            end_time = time.time()
            duration = end_time - start_time  # 测试时长（秒）
            
            # 新增：记录当前 sample 的信息
            record_list.append({
                'sample': sample,
                'duration': duration,
                'voxel_count': voxel_count,
                'point_count': point_count
            })
            
            
            if cfg.eval:
                evaluter.matches[str(sample)] = {}
                evaluter.matches[str(sample)]['seg_pred']= preds
                evaluter.matches[str(sample)]['seg_gt'] = label
                if cfg.roc:
                    evaluter.roc_update(ts,preds,idx,label.cpu(),ev_locs)

        pbar.update(1)

    # 新增：输出统计信息
    print("\n=== 测试统计信息 ===")
    for record in record_list:
        print(f"Sample {record['sample']}: 时长={record['duration']:.4f}s, 体素数={record['voxel_count']}, 点数={record['point_count']}")

    if cfg.eval:
        iou = evaluter.evaluate_semantic_segmantation_miou()
        seg_acc = evaluter.evaluate_semantic_segmantation_accuracy()
        if cfg.roc:
            pd, fa= evaluter.cal_roc()
        print('iou:{},seg_acc:{},pd:{},fa:{}'.format(iou,seg_acc,pd,fa))
