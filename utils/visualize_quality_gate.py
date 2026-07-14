import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import cfg
from datasets import make_dataloader
from model.clip_facenet import CLIPFACENet
from model.coen_lite import patch_quality_to_scalar, quality_gate
from utils.ablation import ABLATION_CONFIGS, apply_ablation_config


def collect_gate_values(cfg, weight_path, exp_name, floor=0.85, max_batches=50):
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

    gate_R_flare = []
    gate_R_clean = []
    gate_N_flare = []
    gate_N_clean = []

    q_R_flare = []
    q_R_clean = []
    q_N_flare = []
    q_N_clean = []

    with torch.no_grad():
        for batch_idx, (img1, img2, img3, vid, camid, camids_batch, viewids, img_paths, flare_label) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            img1 = img1.cuda()
            img2 = img2.cuda()
            img3 = img3.cuda()
            flare_label = flare_label.cuda()

            _ = model(img1, img2, img3, vid.cuda(), flare_label=flare_label)

            q_R_map = getattr(model, "_last_q_R_map", None)
            q_N_map = getattr(model, "_last_q_N_map", None)

            if q_R_map is None or q_N_map is None:
                continue

            q_R = patch_quality_to_scalar(q_R_map).detach()
            q_N = patch_quality_to_scalar(q_N_map).detach()

            g_R = quality_gate(q_R, floor=floor).detach()
            g_N = quality_gate(q_N, floor=floor).detach()

            flare = flare_label.detach().bool()

            gate_R_flare.extend(g_R[flare].cpu().numpy().tolist())
            gate_R_clean.extend(g_R[~flare].cpu().numpy().tolist())
            gate_N_flare.extend(g_N[flare].cpu().numpy().tolist())
            gate_N_clean.extend(g_N[~flare].cpu().numpy().tolist())

            q_R_flare.extend(q_R[flare].cpu().numpy().tolist())
            q_R_clean.extend(q_R[~flare].cpu().numpy().tolist())
            q_N_flare.extend(q_N[flare].cpu().numpy().tolist())
            q_N_clean.extend(q_N[~flare].cpu().numpy().tolist())

    return {
        "gate_R_flare": gate_R_flare,
        "gate_R_clean": gate_R_clean,
        "gate_N_flare": gate_N_flare,
        "gate_N_clean": gate_N_clean,
        "q_R_flare": q_R_flare,
        "q_R_clean": q_R_clean,
        "q_N_flare": q_N_flare,
        "q_N_clean": q_N_clean,
    }


def draw_boxplot(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = [
        "RGB\nflare",
        "RGB\nclean",
        "NI\nflare",
        "NI\nclean",
    ]

    values = [
        data["gate_R_flare"],
        data["gate_R_clean"],
        data["gate_N_flare"],
        data["gate_N_clean"],
    ]

    plt.figure(figsize=(7, 5))
    box = plt.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        showmeans=True,
    )

    colors = ["#ff9999", "#99ccff", "#ff9999", "#99ccff"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    plt.ylabel("Quality-aware loss weight")
    plt.title("Distribution of quality-aware loss gates")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved quality gate boxplot to {save_path}")


def draw_quality_boxplot(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = [
        "RGB\nflare",
        "RGB\nclean",
        "NI\nflare",
        "NI\nclean",
    ]

    values = [
        data["q_R_flare"],
        data["q_R_clean"],
        data["q_N_flare"],
        data["q_N_clean"],
    ]

    plt.figure(figsize=(7, 5))
    box = plt.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        showmeans=True,
    )

    colors = ["#ff9999", "#99ccff", "#ff9999", "#99ccff"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    plt.ylabel("Quality score")
    plt.title("Distribution of estimated quality scores")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved quality score boxplot to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="configs/WMVeID863/clip_facenet_wmveid863.yml")
    parser.add_argument("--weight", required=True)
    parser.add_argument("--exp", default="original_G", choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--floor", type=float, default=0.85)
    parser.add_argument("--max_batches", type=int, default=80)
    parser.add_argument("--save_dir", default="./vis/quality_gate")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    data = collect_gate_values(
        cfg=cfg,
        weight_path=args.weight,
        exp_name=args.exp,
        floor=args.floor,
        max_batches=args.max_batches,
    )

    draw_boxplot(data, os.path.join(args.save_dir, "quality_gate_boxplot.png"))
    draw_quality_boxplot(data, os.path.join(args.save_dir, "quality_score_boxplot.png"))


if __name__ == "__main__":
    main()