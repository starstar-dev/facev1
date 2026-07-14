import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from datasets import make_dataloader
from model.clip_facenet import CLIPFACENet
from utils.ablation import ABLATION_CONFIGS, apply_ablation_config, format_ablation_flags
from utils.logger import setup_logger
from utils.metrics import euclidean_distance, eval_func, eval_func_msrv


def image_quality_score(img_r, img_n, mean, std):
    """Lower score means stronger RGB/NI over-exposure."""
    device = img_r.device
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    r = (img_r * std_t + mean_t).clamp(0, 1)
    n = (img_n * std_t + mean_t).clamp(0, 1)
    r_over = (r >= 0.95).float().mean(dim=(1, 2, 3))
    n_over = (n >= 0.95).float().mean(dim=(1, 2, 3))
    damage = 0.5 * (r_over + n_over)
    return (1.0 - damage).detach().cpu().numpy()


def split_query_groups(scores, flare_labels, num_query, mode):
    if mode == "flare":
        labels = np.asarray(flare_labels[:num_query])
        return {
            "clean": np.where(labels == 0)[0],
            "flare": np.where(labels == 1)[0],
        }

    q_scores = np.asarray(scores[:num_query], dtype=np.float64)
    order = np.argsort(q_scores)
    groups = np.array_split(order, 3)
    return {
        "low_quality": groups[0],
        "mid_quality": groups[1],
        "high_quality": groups[2],
    }


def evaluate_group(distmat, pids, camids, sceneids, num_query, group_indices, dataset_name):
    if len(group_indices) == 0:
        return None

    q_idx = np.asarray(group_indices, dtype=np.int64)
    q_pids = np.asarray(pids[:num_query])[q_idx]
    q_camids = np.asarray(camids[:num_query])[q_idx]
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])
    sub_dist = distmat[q_idx, :]

    if dataset_name == "MSVR310":
        q_sceneids = np.asarray(sceneids[:num_query])[q_idx]
        g_sceneids = np.asarray(sceneids[num_query:])
        cmc, mAP = eval_func_msrv(
            sub_dist, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids
        )
    else:
        cmc, mAP, _ = eval_func(sub_dist, q_pids, g_pids, q_camids, g_camids)

    return mAP, cmc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", "--ablation", default="full", choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--config_file", default="configs/WMVeID863/clip_facenet_wmveid863.yml")
    parser.add_argument("--stratify", default="quality", choices=["quality", "flare"])
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    logger = setup_logger("transreid", cfg.OUTPUT_DIR, if_train=False)
    logger.info("Running quality-stratified CLIP-FACENet inference")

    _, _, val_loader, num_query, num_classes, _ = make_dataloader(cfg)
    model = CLIPFACENet(num_classes=num_classes, cfg=cfg)
    model.use_mamba_enhance = False
    model.use_fce = False
    exp_name, exp_cfg = apply_ablation_config(model, args.exp)
    logger.info(f"Ablation {exp_name}: {exp_cfg['desc']}")
    logger.info(f"Ablation flags: {format_ablation_flags(model)}")
    model.load_param(cfg.TEST.WEIGHT)

    device = "cuda"
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    feats, pids, camids, sceneids = [], [], [], []
    quality_scores, flare_labels = [], []

    with torch.no_grad():
        for img1, img2, img3, vid, camid, camids_batch, viewids, img_paths, flare_label in val_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            target = vid.to(device)
            flare_label_gpu = flare_label.to(device)

            feat = model(img1, img2, img3, target, flare_label=flare_label_gpu)
            feats.append(feat.cpu())
            pids.extend(np.asarray(vid))
            camids.extend(np.asarray(camid))
            sceneids.extend(np.asarray(viewids.cpu()))
            flare_labels.extend(np.asarray(flare_label))
            quality_scores.extend(
                image_quality_score(img1, img2, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
            )

    feats = torch.cat(feats, dim=0)
    if cfg.TEST.FEAT_NORM == "yes":
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)

    qf = feats[:num_query]
    gf = feats[num_query:]
    distmat = euclidean_distance(qf, gf)

    groups = split_query_groups(quality_scores, flare_labels, num_query, args.stratify)
    for name, indices in groups.items():
        result = evaluate_group(distmat, pids, camids, sceneids, num_query, indices, cfg.DATASETS.NAMES)
        if result is None:
            logger.info(f"{name}: empty group")
            continue
        mAP, cmc = result
        logger.info(
            f"{name}: n={len(indices)}, mAP={mAP * 100:.1f}, "
            f"Rank-1={cmc[0] * 100:.1f}, Rank-5={cmc[4] * 100:.1f}, Rank-10={cmc[9] * 100:.1f}"
        )


if __name__ == "__main__":
    main()
