import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import cfg
from datasets import make_dataloader
from model.clip_facenet import CLIPFACENet
from utils.ablation import ABLATION_CONFIGS, apply_ablation_config


@torch.no_grad()
def extract_modal_cls_features(model, img1, img2, img3, flare_label):
    """
    Return modality-specific features used for similarity visualization:
    cls_R, cls_N, cls_T after CoEN fusion.
    """
    featR = model.backbone_rgb(img1)
    featN = model.backbone_ni(img2)
    featT = model.backbone_ti(img3)

    if model.use_mfmp:
        featR_mfmp, featN_mfmp = model.get_cls_feat_mfmp(featR, featN)
    else:
        featR_mfmp, featN_mfmp = featR, featN

    q_R = None
    q_N = None

    if model.use_coen_lite:
        # Let model compute and store q maps.
        _ = model(img1, img2, img3, label=None, flare_label=flare_label)
        q_R = getattr(model, "_last_q_R_map", None)
        q_N = getattr(model, "_last_q_N_map", None)

    featR_in = featR
    featN_in = featN

    if model.use_fusion:
        featR = model.fusion_R(featR_in, featT, featN_in, q_R, q_N)
        featN = model.fusion_N(featN_in, featT, featR_in, q_N, q_R)

    cls_R = featR[:, 0]
    cls_N = featN[:, 0]
    cls_T = featT[:, 0]

    cls_R = F.normalize(cls_R, dim=1)
    cls_N = F.normalize(cls_N, dim=1)
    cls_T = F.normalize(cls_T, dim=1)

    return cls_R, cls_N, cls_T


def collect_modal_features(cfg, weight_path, exp_name, max_batches=80):
    _, _, val_loader, num_query, num_classes, _ = make_dataloader(cfg)

    model = CLIPFACENet(num_classes=num_classes, cfg=cfg)
    model.use_mamba_enhance = False
    model.use_mcloss = True
    model.use_fce = False
    model.use_mfmp = True

    apply_ablation_config(model, exp_name)
    model.load_param(weight_path)

    model.cuda()
    model.eval()

    feats_R = []
    feats_N = []
    feats_T = []
    labels = []

    with torch.no_grad():
        for batch_idx, (img1, img2, img3, vid, camid, camids_batch, viewids, img_paths, flare_label) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            img1 = img1.cuda()
            img2 = img2.cuda()
            img3 = img3.cuda()
            flare_label = flare_label.cuda()

            cls_R, cls_N, cls_T = extract_modal_cls_features(
                model, img1, img2, img3, flare_label
            )

            feats_R.append(cls_R.cpu())
            feats_N.append(cls_N.cpu())
            feats_T.append(cls_T.cpu())
            labels.extend(np.asarray(vid))

    feats_R = torch.cat(feats_R, dim=0)
    feats_N = torch.cat(feats_N, dim=0)
    feats_T = torch.cat(feats_T, dim=0)
    labels = np.asarray(labels)

    return feats_R, feats_N, feats_T, labels


def compute_similarity_distribution(feats_R, feats_N, feats_T, labels, num_negative=20000):
    """
    Positive:
        same sample cross-modal pairs: R-N, R-T, N-T

    Negative:
        random different-ID cross-modal pairs.
    """
    n = len(labels)

    pos = []

    pos.extend((feats_R * feats_N).sum(dim=1).numpy().tolist())
    pos.extend((feats_R * feats_T).sum(dim=1).numpy().tolist())
    pos.extend((feats_N * feats_T).sum(dim=1).numpy().tolist())

    neg = []

    modality_feats = [feats_R, feats_N, feats_T]

    tries = 0
    while len(neg) < num_negative and tries < num_negative * 20:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        tries += 1

        if labels[i] == labels[j]:
            continue

        m1, m2 = random.sample([0, 1, 2], 2)

        sim = (modality_feats[m1][i] * modality_feats[m2][j]).sum().item()
        neg.append(sim)

    return np.asarray(pos), np.asarray(neg)


def draw_similarity_hist(pos_a, neg_a, pos_b, neg_b, label_a, label_b, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))

    bins = np.linspace(-0.2, 1.0, 60)

    plt.hist(
        pos_a,
        bins=bins,
        density=True,
        alpha=0.35,
        color="#1f77b4",
        label=f"{label_a} positive",
    )
    plt.hist(
        neg_a,
        bins=bins,
        density=True,
        alpha=0.25,
        color="#ff7f0e",
        label=f"{label_a} negative",
    )

    plt.hist(
        pos_b,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        color="#1f77b4",
        label=f"{label_b} positive",
    )
    plt.hist(
        neg_b,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        color="#ff7f0e",
        label=f"{label_b} negative",
    )

    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title("Cross-modal similarity distribution")
    plt.legend(fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved similarity distribution to {save_path}")


def print_stats(name, pos, neg):
    print(f"[{name}] positive mean={pos.mean():.4f}, std={pos.std():.4f}")
    print(f"[{name}] negative mean={neg.mean():.4f}, std={neg.std():.4f}")
    print(f"[{name}] gap={(pos.mean() - neg.mean()):.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", default="configs/WMVeID863/clip_facenet_wmveid863.yml")

    parser.add_argument("--weight_a", required=True, help="Checkpoint of w/o SupCon model")
    parser.add_argument("--weight_b", required=True, help="Checkpoint of full model")

    parser.add_argument("--exp_a", default="wo_supcon", choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--exp_b", default="original_G", choices=list(ABLATION_CONFIGS.keys()))

    parser.add_argument("--label_a", default="w/o SupCon")
    parser.add_argument("--label_b", default="Full")

    parser.add_argument("--max_batches", type=int, default=80)
    parser.add_argument("--num_negative", type=int, default=20000)
    parser.add_argument("--save_path", default="./vis/supcon_similarity/similarity_distribution.png")

    parser.add_argument("opts", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print(f"Collecting features for {args.label_a}...")
    R_a, N_a, T_a, y_a = collect_modal_features(
        cfg=cfg,
        weight_path=args.weight_a,
        exp_name=args.exp_a,
        max_batches=args.max_batches,
    )
    pos_a, neg_a = compute_similarity_distribution(
        R_a, N_a, T_a, y_a, num_negative=args.num_negative
    )

    print(f"Collecting features for {args.label_b}...")
    R_b, N_b, T_b, y_b = collect_modal_features(
        cfg=cfg,
        weight_path=args.weight_b,
        exp_name=args.exp_b,
        max_batches=args.max_batches,
    )
    pos_b, neg_b = compute_similarity_distribution(
        R_b, N_b, T_b, y_b, num_negative=args.num_negative
    )

    print_stats(args.label_a, pos_a, neg_a)
    print_stats(args.label_b, pos_b, neg_b)

    draw_similarity_hist(
        pos_a=pos_a,
        neg_a=neg_a,
        pos_b=pos_b,
        neg_b=neg_b,
        label_a=args.label_a,
        label_b=args.label_b,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()