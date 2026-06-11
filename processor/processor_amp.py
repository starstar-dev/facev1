import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
from model.coen_lite import quality_gate
import torch.distributed as dist
from utils.visualize_coen import save_batch_qmap_visualization


def kl_div(x, y):
    return F.kl_div(x.log_softmax(dim=1), y.softmax(dim=1), reduction='sum')


def KL_loss(a, b):
    return min(kl_div(a.softmax(dim=-1).log(), b.softmax(dim=-1)),
               kl_div(b.softmax(dim=-1).log(), a.softmax(dim=-1)))


def MC_loss(scoreR, scoreN, scoreT):
    return KL_loss(scoreR, scoreN) + KL_loss(scoreT, scoreN)


log_period = 20
eval_period = 1
scaler = amp.GradScaler()


def do_train_amp(cfg, model, center_criterion, train_loader, val_loader, optimizer,
                 optimizer_center, scheduler, loss_fn, num_query, local_rank):
    logger = logging.getLogger("transreid.image_train")
    
    has_parts = getattr(model, 'num_parts', 0) > 0
    if has_parts:
        logger.info(f'PATA: {model.num_parts} parts, ID_w=0.3, Align_w=0.2')
    
    device = "cuda"
    if device:
        model.to(local_rank)
    if hasattr(loss_fn, "circle_loss_fn") and loss_fn.circle_loss_fn is not None:
        loss_fn.circle_loss_fn.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
    best_index = {'mAP': 0.0, 'Rank-1': 0.0, 'Rank-5': 0.0, 'Rank-10': 0.0}

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter1 = AverageMeter()
    acc_meter2 = AverageMeter()

    start_time = time.time()
    
    epochs = cfg.SOLVER.MAX_EPOCHS
    accum_steps = cfg.SOLVER.GRAD_ACCUM_STEPS if hasattr(cfg.SOLVER, 'GRAD_ACCUM_STEPS') else 1

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        loss_meter.reset()
        acc_meter.reset()
        acc_meter1.reset()
        acc_meter2.reset()
        scheduler.step(epoch)
        evaluator.reset()

        for n_iter, (img1, img2, img3, vid, target_cam, _, flare_label) in enumerate(train_loader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            flare_label = flare_label.to(device)

            with amp.autocast(enabled=True):
                mode1, mode2, mode3, RFeat, scoreR_forlabel, NFeat, scoreN_forlabel, fce_feats = \
                    model(img1, img2, img3, target, flare_label=flare_label)
                
                loss1 = loss_fn(mode1[0], mode1[1], target, target_cam)
                loss2 = loss_fn(mode2[0], mode2[1], target, target_cam)
                loss3 = loss_fn(mode3[0], mode3[1], target, target_cam)

                mcloss = 0
                kl_loss = 0
                part_loss = 0
                part_align = 0
                fce_loss = 0

                loss = loss1 + loss2 + loss3

                # CoEN-lite: dynamic quality gating + feature repair
                # Quality scores are computed by the model (also used for fusion repair)
                raw_model = model.module if hasattr(model, "module") else model

                if raw_model.use_coen_lite:
                    coen_qR = raw_model._coen_qR
                    coen_qN = raw_model._coen_qN

                    # patch-level q：局部坏不代表整图废掉，所以 loss gate 要温和
                    coen_gR = quality_gate(coen_qR, floor=0.7)
                    coen_gN = quality_gate(coen_qN, floor=0.7)
                    coen_g_pair = (coen_gR + coen_gN) / 2.0

                    loss1 = loss1 * coen_gR
                    loss2 = loss2 * coen_gN
                    loss = loss1 + loss2 + loss3

                    coen_qR_log = getattr(raw_model, "_coen_qR_log", float(coen_qR.detach().item()))
                    coen_qN_log = getattr(raw_model, "_coen_qN_log", float(coen_qN.detach().item()))
                else:
                    coen_qR_log = 1.0
                    coen_qN_log = 1.0
                    coen_gR = 1.0
                    coen_gN = 1.0
                    coen_g_pair = 1.0

                if model.use_mfmp and scoreR_forlabel is not None:
                    kl_loss = 0.8 * KL_loss(scoreR_forlabel, scoreN_forlabel)
                    if model.use_coen_lite:
                        kl_loss = kl_loss * coen_g_pair
                    loss += kl_loss
                if model.use_mcloss:
                    mcloss = MC_loss(mode1[0][0], mode2[0][0], mode3[0][0])
                    if model.use_coen_lite:
                        mcloss = mcloss * coen_g_pair
                    loss += mcloss
                if model.use_fce and fce_feats is not None:
                    bn_R, bn_N, bn_T = fce_feats
                    fce_loss = 0.5 * (F.mse_loss(bn_R, bn_T) + F.mse_loss(bn_N, bn_T))
                    loss += fce_loss

                # Part losses (v3 PATA)
                if has_parts and len(mode1[0]) > 1:
                    PART_ID_WEIGHT = 0.3
                    PART_ALIGN_WEIGHT = 0.2
                    for i in range(1, len(mode1[0])):
                        part_loss += PART_ID_WEIGHT * loss_fn(mode1[0][i], mode1[1][i], target, target_cam)
                        part_loss += PART_ID_WEIGHT * loss_fn(mode2[0][i], mode2[1][i], target, target_cam)
                        part_loss += PART_ID_WEIGHT * loss_fn(mode3[0][i], mode3[1][i], target, target_cam)
                    loss += part_loss
                    for i in range(1, len(mode1[0])):
                        part_align += PART_ALIGN_WEIGHT * KL_loss(mode1[0][i], mode2[0][i])
                        part_align += PART_ALIGN_WEIGHT * KL_loss(mode3[0][i], mode2[0][i])
                    loss += part_align

            # 每 5 个 epoch 的第 1 个 batch 保存一次 q_map 可视化
            raw_model = model.module if hasattr(model, "module") else model
            if raw_model.use_coen_lite and n_iter == 0 and epoch % 5 == 0:
                vis_dir = os.path.join(cfg.SAVE_DIR, "coen_vis")
                print(f"[QMAP SAVE] epoch={epoch}, n_iter={n_iter}, save_dir={vis_dir}")

                save_batch_qmap_visualization(
                    model,
                    save_dir=vis_dir,
                    epoch=epoch,
                    max_samples=4
                )

            scaler.scale(loss / accum_steps).backward()
            if (n_iter + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                    scaler.step(optimizer_center)
                    scaler.update()

                optimizer.zero_grad()
                optimizer_center.zero_grad()

            if isinstance(mode1[0][0], list):
                acc = (mode1[0][0].max(1)[1] == target).float().mean()
                acc1 = (mode2[0][0].max(1)[1] == target).float().mean()
                acc2 = (mode3[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (mode1[0][0].max(1)[1] == target).float().mean()
                acc1 = (mode2[0][0].max(1)[1] == target).float().mean()
                acc2 = (mode3[0][0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img1.shape[0])
            acc_meter.update(acc, 1)
            acc_meter1.update(acc1, 1)
            acc_meter2.update(acc2, 1)
            torch.cuda.synchronize()

            peer_R_open = getattr(raw_model, "_peer_R_open_ratio", torch.tensor(0.0)).detach().item()
            peer_N_open = getattr(raw_model, "_peer_N_open_ratio", torch.tensor(0.0)).detach().item()
            peer_R_mean = getattr(raw_model, "_peer_R_gate_mean", torch.tensor(0.0)).detach().item()
            peer_N_mean = getattr(raw_model, "_peer_N_gate_mean", torch.tensor(0.0)).detach().item()

            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] "
                    "CoEN(qR={:.2f},qN={:.2f}) "
                    "Peer(R_open={:.2f},N_open={:.2f},R_mean={:.3f},N_mean={:.3f}) "
                    "Loss: {:.3f}, Acc: {:.3f},Acc1: {:.3f},Acc2: {:.3f}, "
                    "Base Lr: {:.2e},biloss:{:.3f},icloss:{:.3f},"
                    "loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, "
                    "ploss:{:.3f},palign:{:.3f},total_loss:{:.3f},fceloss:{:.4f}"
                    .format(
                        epoch, (n_iter + 1), len(train_loader),
                        coen_qR_log, coen_qN_log,
                        peer_R_open, peer_N_open, peer_R_mean, peer_N_mean,
                        loss_meter.avg, acc_meter.avg, acc_meter1.avg, acc_meter2.avg,
                        scheduler._get_lr(epoch)[0], kl_loss, mcloss,
                        loss1, loss2, loss3, part_loss, part_align, loss, fce_loss
                    )
                )
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if not cfg.MODEL.DIST_TRAIN:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % eval_period == 0:
            model.eval()
            evaluator.reset()
            for n_iter, (img1, img2, img3, vid, camid, camids, viewids, img_paths, flare_label) in enumerate(val_loader):
                with torch.no_grad():
                    img1 = img1.to(device)
                    img2 = img2.to(device)
                    img3 = img3.to(device)
                    camids = camids.to(device)
                    flare_label = flare_label.to(device)
                    target = vid.to(device)
                    feat = model(img1, img2, img3, target, flare_label=flare_label)
                    if cfg.DATASETS.NAMES == "MSVR310":
                        evaluator.update((feat, vid, camid, viewids, img_paths))
                    else:
                        evaluator.update((feat, vid, camid, img_paths))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            if mAP >= best_index['mAP']:
                best_index['mAP'] = mAP
                best_index['Rank-1'] = cmc[0]
                best_index['Rank-5'] = cmc[4]
                best_index['Rank-10'] = cmc[9]
                torch.save(model.state_dict(),
                           os.path.join(cfg.SAVE_DIR, cfg.MODEL.NAME + 'best.pth'))
            logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
            logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
            logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
            logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
            torch.cuda.empty_cache()


def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
    model.eval()
    for n_iter, (img1, img2, img3, vid, camid, camids, viewids, img_paths, flare_label) in enumerate(val_loader):
        with torch.no_grad():
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            target = vid.to(device)
            camids = camids.to(device)
            flare_label = flare_label.to(device)
            feat = model(img1, img2, img3, target, flare_label=flare_label)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, vid, camid, viewids, img_paths))
            else:
                evaluator.update((feat, vid, camid, img_paths))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
