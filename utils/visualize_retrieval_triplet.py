import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# ICASSP uses a two-column page. 7.16 in is the usable width of a full-width
# figure in the official template. Keeping the Matplotlib canvas at the final
# insertion size prevents fonts from being scaled down by LaTeX.
PAPER_WIDTH_IN = 7.16
PAPER_HEIGHT_IN = 3.25
PAPER_DPI = 600
CORRECT_COLOR = "#009E73"    # color-blind-safe green
INCORRECT_COLOR = "#D55E00"  # color-blind-safe vermilion

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stix",
    "savefig.facecolor": "white",
}


def save_figure_pdf_and_png(fig, save_path):
    """Save vector text/lines for LaTeX and a 600-dpi raster fallback."""
    stem, _ = os.path.splitext(save_path)
    pdf_path = f"{stem}.pdf"
    png_path = f"{stem}.png"

    parent = os.path.dirname(os.path.abspath(stem))
    os.makedirs(parent, exist_ok=True)

    save_kwargs = {
        "bbox_inches": "tight",
        "pad_inches": 0.02,
        "facecolor": "white",
    }
    fig.savefig(pdf_path, **save_kwargs)
    fig.savefig(png_path, dpi=PAPER_DPI, **save_kwargs)
    return pdf_path, png_path


def load_triplet_images(path_item):
    """
    path_item = (vis_path, ni_path, th_path)
    """
    vis = Image.open(path_item[0]).convert("RGB")
    ni = Image.open(path_item[1]).convert("RGB")
    th = Image.open(path_item[2]).convert("RGB")
    return vis, ni, th


def collect_features(cfg, weight_path, exp_name):
    """
    Run inference and collect:
    - features
    - pids
    - camids
    - image paths
    """
    import torch

    from datasets import make_dataloader
    from model.clip_facenet import CLIPFACENet
    from utils.ablation import apply_ablation_config
    from utils.metrics import euclidean_distance

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

    feats = []
    pids = []
    camids = []
    paths = []

    with torch.no_grad():
        for img1, img2, img3, vid, camid, camids_batch, viewids, img_paths, flare_label in val_loader:
            img1 = img1.cuda()
            img2 = img2.cuda()
            img3 = img3.cuda()
            vid_cuda = vid.cuda()
            flare_label = flare_label.cuda()

            feat = model(img1, img2, img3, vid_cuda, flare_label=flare_label)

            feats.append(feat.cpu())
            pids.extend(np.asarray(vid))
            camids.extend(np.asarray(camid))
            paths.extend(list(img_paths))

    feats = torch.cat(feats, dim=0)

    if cfg.TEST.FEAT_NORM == "yes" or cfg.TEST.FEAT_NORM is True:
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)

    qf = feats[:num_query]
    gf = feats[num_query:]

    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])

    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])

    q_paths = paths[:num_query]
    g_paths = paths[num_query:]

    distmat = euclidean_distance(qf, gf)

    return {
        "distmat": distmat,
        "q_pids": q_pids,
        "g_pids": g_pids,
        "q_camids": q_camids,
        "g_camids": g_camids,
        "q_paths": q_paths,
        "g_paths": g_paths,
    }


def get_valid_gallery_order(dist_row, q_pid, q_camid, g_pids, g_camids):
    """
    Remove same ID + same camera gallery images, following standard ReID protocol.
    """
    order = np.argsort(dist_row)
    remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
    keep = np.invert(remove)
    return order[keep]


def draw_triplet_retrieval_case(
    save_path,
    q_path,
    gallery_paths,
    gallery_labels,
    title=None,
    topk=5,
):
    """
    Draw:
        Query | Rank-1 | Rank-2 | ... | Rank-k
    rows:
        RGB
        NIR
        TIR
    """
    cols = topk + 1
    rows = 3

    if len(gallery_paths) < topk or len(gallery_labels) < topk:
        raise ValueError("gallery_paths and gallery_labels must contain topk items")

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(PAPER_WIDTH_IN, PAPER_HEIGHT_IN),
            squeeze=False,
        )

        # Use the terminology used in the paper instead of dataset-specific
        # abbreviations. The caption should explain the border colors.
        modal_names = ["RGB", "NIR", "TIR"]
        q_imgs = load_triplet_images(q_path)

        for r in range(rows):
            ax = axes[r, 0]
            ax.imshow(q_imgs[r], interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_anchor("C")

            if r == 0:
                ax.set_title("Query", pad=3)

            ax.set_ylabel(modal_names[r], labelpad=2)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#222222")
                spine.set_linewidth(0.9)

        for c in range(topk):
            g_imgs = load_triplet_images(gallery_paths[c])
            is_correct = bool(gallery_labels[c])
            color = CORRECT_COLOR if is_correct else INCORRECT_COLOR

            for r in range(rows):
                ax = axes[r, c + 1]
                ax.imshow(g_imgs[r], interpolation="lanczos")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_anchor("C")

                if r == 0:
                    ax.set_title(f"Rank-{c + 1}", color=color, pad=3)

                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(color)
                    spine.set_linewidth(1.2)

        # A suptitle repeats the paper caption and makes the panels smaller.
        # It is still available for multi-panel figures, but disabled by the
        # callers below for the standalone paper figure.
        top = 0.90
        if title:
            fig.suptitle(title, fontsize=9, y=0.995)
            top = 0.86

        fig.subplots_adjust(
            left=0.055,
            right=0.995,
            bottom=0.025,
            top=top,
            wspace=0.075,
            hspace=0.075,
        )

        output_paths = save_figure_pdf_and_png(fig, save_path)
        plt.close(fig)
        return output_paths


def visualize_ours_only(result, save_dir, topk=5, num_cases=10):
    """
    Draw retrieval ranking cases for one model.
    """
    os.makedirs(save_dir, exist_ok=True)

    distmat = result["distmat"]
    q_pids = result["q_pids"]
    g_pids = result["g_pids"]
    q_camids = result["q_camids"]
    g_camids = result["g_camids"]
    q_paths = result["q_paths"]
    g_paths = result["g_paths"]

    saved = 0

    for qi in range(len(q_pids)):
        order = get_valid_gallery_order(
            distmat[qi],
            q_pids[qi],
            q_camids[qi],
            g_pids,
            g_camids,
        )

        labels = [g_pids[idx] == q_pids[qi] for idx in order[:topk]]
        gallery_paths = [g_paths[idx] for idx in order[:topk]]

        save_path = os.path.join(
            save_dir,
            f"ours_case_{saved:02d}_pid_{q_pids[qi]}.png",
        )

        draw_triplet_retrieval_case(
            save_path=save_path,
            q_path=q_paths[qi],
            gallery_paths=gallery_paths,
            gallery_labels=labels,
            title=None,
            topk=topk,
        )

        saved += 1
        if saved >= num_cases:
            break

    print(f"Saved {saved} ours-only retrieval cases to {save_dir}")


def visualize_baseline_wrong_ours_correct(
    baseline,
    ours,
    save_dir,
    topk=5,
    num_cases=10,
):
    """
    Select cases where:
        baseline Top-1 is wrong
        ours Top-1 is correct
    """
    os.makedirs(save_dir, exist_ok=True)

    b_dist = baseline["distmat"]
    o_dist = ours["distmat"]

    q_pids = ours["q_pids"]
    g_pids = ours["g_pids"]
    q_camids = ours["q_camids"]
    g_camids = ours["g_camids"]
    q_paths = ours["q_paths"]
    g_paths = ours["g_paths"]

    saved = 0

    for qi in range(len(q_pids)):
        b_order = get_valid_gallery_order(
            b_dist[qi],
            q_pids[qi],
            q_camids[qi],
            g_pids,
            g_camids,
        )

        o_order = get_valid_gallery_order(
            o_dist[qi],
            q_pids[qi],
            q_camids[qi],
            g_pids,
            g_camids,
        )

        baseline_top1_correct = g_pids[b_order[0]] == q_pids[qi]
        ours_top1_correct = g_pids[o_order[0]] == q_pids[qi]

        if baseline_top1_correct:
            continue

        if not ours_top1_correct:
            continue

        labels = [g_pids[idx] == q_pids[qi] for idx in o_order[:topk]]
        gallery_paths = [g_paths[idx] for idx in o_order[:topk]]

        save_path = os.path.join(
            save_dir,
            f"baseline_wrong_ours_correct_{saved:02d}_pid_{q_pids[qi]}.png",
        )

        draw_triplet_retrieval_case(
            save_path=save_path,
            q_path=q_paths[qi],
            gallery_paths=gallery_paths,
            gallery_labels=labels,
            title=None,
            topk=topk,
        )

        saved += 1
        if saved >= num_cases:
            break

    print(f"Saved {saved} baseline-wrong-ours-correct cases to {save_dir}")


def main():
    from config import cfg
    from utils.ablation import ABLATION_CONFIGS

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        default="configs/WMVeID863/clip_facenet_wmveid863.yml",
    )

    parser.add_argument(
        "--ours_weight",
        required=True,
        help="Path to your best model checkpoint.",
    )

    parser.add_argument(
        "--baseline_weight",
        default="",
        help="Optional baseline checkpoint. If provided, script selects baseline-wrong/ours-correct cases.",
    )

    parser.add_argument(
        "--ours_exp",
        default="original_G",
        choices=list(ABLATION_CONFIGS.keys()),
    )

    parser.add_argument(
        "--baseline_exp",
        default="backbone",
        choices=list(ABLATION_CONFIGS.keys()),
    )

    parser.add_argument("--save_dir", default="./vis/retrieval_triplet")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num_cases", type=int, default=10)

    parser.add_argument("opts", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Collecting ours features...")
    ours = collect_features(
        cfg=cfg,
        weight_path=args.ours_weight,
        exp_name=args.ours_exp,
    )

    if args.baseline_weight:
        print("Collecting baseline features...")
        baseline = collect_features(
            cfg=cfg,
            weight_path=args.baseline_weight,
            exp_name=args.baseline_exp,
        )

        visualize_baseline_wrong_ours_correct(
            baseline=baseline,
            ours=ours,
            save_dir=args.save_dir,
            topk=args.topk,
            num_cases=args.num_cases,
        )
    else:
        visualize_ours_only(
            result=ours,
            save_dir=args.save_dir,
            topk=args.topk,
            num_cases=args.num_cases,
        )


if __name__ == "__main__":
    main()