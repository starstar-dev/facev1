"""
E4: Subset-specific evaluation from a trained checkpoint.

Evaluates a trained model on both "hard" and "normal" query subsets
in a SINGLE inference pass, and verifies that weighted average equals full mAP.

Usage:
  python scripts/test_subset.py \
      --config_file configs/WMVeID863/clip_facenet_wmveid863.yml \
      --weight ./logs/xxx/clip_facenetbest.pth \
      --subset_split ./hard_normal_splits/WMVeID863/
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from model import FACENet
from datasets import make_dataloader
from utils.metrics import eval_func
from utils.reranking import re_ranking


def load_subset_indices(subset_dir, subset_name):
    path = os.path.join(subset_dir, f"{subset_name}_queries.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Subset file not found: {path}")
    with open(path, "r") as f:
        indices = [int(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(indices)} {subset_name} query indices from {path}")
    return indices


def euclidean_distance(qf, gf):
    """Squared Euclidean distance (same as metrics.py)."""
    m, n = qf.size(0), gf.size(0)
    distmat = (
        torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return distmat.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Subset evaluation for E4 Hard/Normal analysis")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--weight", type=str, required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--subset", type=str, default=None,
                        choices=["hard", "normal"])
    parser.add_argument("--subset_split", type=str, required=True,
                        help="Directory containing hard_queries.txt and normal_queries.txt")
    parser.add_argument("--no_reranking", action="store_true",
                        help="Disable re-ranking")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Override config options")
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    logger = logging.getLogger("test_subset")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Checkpoint: {args.weight}")

    hard_indices = load_subset_indices(args.subset_split, "hard")
    normal_indices = load_subset_indices(args.subset_split, "normal")

    # Create dataloader — need camera_num for FACENet constructor
    _, _, val_loader, num_query, num_classes, camera_num = make_dataloader(cfg)

    logger.info(f"Total queries: {num_query}")
    logger.info(f"Hard queries:  {len(hard_indices)} "
                f"({100 * len(hard_indices) / num_query:.1f}%)")
    logger.info(f"Normal queries:{len(normal_indices)} "
                f"({100 * len(normal_indices) / num_query:.1f}%)")

    # Build model (same as test.py)
    model = FACENet(cfg, num_class=num_classes, camera_num=camera_num, view_num=0)
    model.to(device)
    model.load_param(args.weight)
    logger.info("Model loaded via load_param (same as test.py)")

    # ====== Run inference ONCE ======
    model.eval()
    all_feats = []
    all_pids = []
    all_camids = []

    logger.info("Running inference...")
    for n_iter, (img1, img2, img3, vid, camid, camids, viewids,
                 img_paths, flare_label) in enumerate(val_loader):
        with torch.no_grad():
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            target = vid.to(device)
            flare_label = flare_label.to(device)
            feat = model(img1, img2, img3, target, flare_label=flare_label)
            all_feats.append(feat.cpu())
            all_pids.extend(vid.tolist())
            all_camids.extend(camids.tolist())

    all_feats = torch.cat(all_feats, dim=0)
    logger.info(f"Total features: {all_feats.shape[0]}")

    if cfg.TEST.FEAT_NORM == "yes":
        all_feats = torch.nn.functional.normalize(all_feats, dim=1, p=2)
        logger.info("Features normalized")

    qf = all_feats[:num_query]
    gf = all_feats[num_query:]
    q_pids = np.array(all_pids[:num_query])
    g_pids = np.array(all_pids[num_query:])
    q_camids = np.array(all_camids[:num_query])
    g_camids = np.array(all_camids[num_query:])

    logger.info(f"Query features: {qf.shape}, Gallery features: {gf.shape}")

    # Compute distmat ONCE
    if cfg.TEST.RE_RANKING and not args.no_reranking:
        logger.info("Using re-ranking")
        distmat_full = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
    else:
        logger.info("Using euclidean distance (squared)")
        distmat_full = euclidean_distance(qf, gf)

    # Evaluate all three splits from the SAME distmat
    cmc_full, mAP_full, _ = eval_func(
        distmat_full, q_pids, g_pids, q_camids, g_camids)

    cmc_hard, mAP_hard, _ = eval_func(
        distmat_full[hard_indices], q_pids[hard_indices], g_pids,
        q_camids[hard_indices], g_camids)

    cmc_normal, mAP_normal, _ = eval_func(
        distmat_full[normal_indices], q_pids[normal_indices], g_pids,
        q_camids[normal_indices], g_camids)

    N_hard, N_normal = len(hard_indices), len(normal_indices)
    N_total = N_hard + N_normal
    mAP_weighted = (N_hard * mAP_hard + N_normal * mAP_normal) / N_total

    print()
    print("=" * 72)
    print("E4 Evaluation: Full vs Hard vs Normal (SAME inference pass)")
    print("=" * 72)
    print(f"  {'':20s} {'Queries':>8s} {'mAP':>8s} {'R1':>8s} {'R5':>8s} {'R10':>8s}")
    print(f"  {'-' * 56}")
    print(f"  {'Full':20s} {N_total:8d} {mAP_full:7.1%} {cmc_full[0]:7.1%} "
          f"{cmc_full[4]:7.1%} {cmc_full[9]:7.1%}")
    print(f"  {'Hard':20s} {N_hard:8d} {mAP_hard:7.1%} {cmc_hard[0]:7.1%} "
          f"{cmc_hard[4]:7.1%} {cmc_hard[9]:7.1%}")
    print(f"  {'Normal':20s} {N_normal:8d} {mAP_normal:7.1%} {cmc_normal[0]:7.1%} "
          f"{cmc_normal[4]:7.1%} {cmc_normal[9]:7.1%}")
    print(f"  {'-' * 56}")
    print(f"  {'Weighted Avg':20s} {N_total:8d} {mAP_weighted:7.1%}")
    print(f"  {'Full - Weighted':20s} {'':8s} {mAP_full - mAP_weighted:+.4%}")
    print("=" * 72)
    if abs(mAP_full - mAP_weighted) < 1e-6:
        print("  ✓ mAP_full == weighted average (as expected)")
    else:
        print("  ✗ MISMATCH — this should not happen with same features!")
    print("=" * 72)

    if args.subset is not None:
        print()
        print("=" * 60)
        print(f"E4 Subset Evaluation: {args.subset.upper()} queries")
        print("=" * 60)
        mAP_sub = mAP_hard if args.subset == "hard" else mAP_normal
        cmc_sub = cmc_hard if args.subset == "hard" else cmc_normal
        N_sub = N_hard if args.subset == "hard" else N_normal
        print(f"  Query count:  {N_sub} / {N_total}")
        print(f"  mAP:          {mAP_sub:.1%}")
        for r in [1, 5, 10, 20]:
            if r <= len(cmc_sub):
                print(f"  Rank-{r:<7}: {cmc_sub[r - 1]:.1%}")
        print("=" * 60)


if __name__ == "__main__":
    main()