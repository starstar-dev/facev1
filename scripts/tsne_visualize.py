"""
E7: t-SNE visualization on best-alignment IDs (v8 — selective).

Strategy:
  1. For each ID, compute cross-modal centroid distances (R↔N, R↔T, N↔T)
  2. Rank IDs by improvement ratio: Baseline_dist / QTA_dist
  3. Pick top-K IDs with largest improvement
  4. t-SNE side-by-side: Baseline vs QTA, color=ID, marker=modality

Usage:
  python scripts/tsne_visualize.py \
      --config_file configs/WMVeID863/clip_facenet_wmveid863.yml \
      --weight_qta ./logs/xxx_full/clip_facenetbest.pth \
      --weight_base ./logs/xxx_backbone/clip_facenetbest.pth \
      --num_ids 5 --output_dir ./tsne_results/
"""

import os, sys, argparse, random
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from model.clip_facenet import CLIPFACENet
from datasets import make_dataloader
from utils.ablation import apply_ablation_config


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def extract_prebn_features(model, val_loader, num_query, device):
    """Extract pre-BNNeck CLS tokens per modality."""
    model.eval()
    feats_R, feats_N, feats_T = [], [], []
    all_pids = []

    for img1, img2, img3, vid, camid, camids, viewids, img_paths, flare_label in val_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        img3 = img3.to(device)
        flare_label = flare_label.to(device)

        with torch.no_grad():
            featR = model.backbone_rgb(img1)
            featN = model.backbone_ni(img2)
            featT = model.backbone_ti(img3)

            if model.use_fusion:
                B = featR.size(0)
                dev = featR.device
                if model.use_coen_lite:
                    if getattr(model, 'use_global_static_fusion', False):
                        q_R = torch.ones(B, 1, device=dev)
                        q_N = torch.ones(B, 1, device=dev)
                    else:
                        if model.coen_use_learned_qmap:
                            from model.coen_lite import compute_combined_quality
                            wl = 0.05; wi = 0.70 if model.coen_use_image_prior else 0.0
                            wd = 0.25 if model.coen_use_disagreement else 0.0
                            q_R, q_N = compute_combined_quality(
                                img1, img2, featR, featN, False,
                                bad_learn_R=None, bad_learn_N=None,
                                w_learned_train=wl, w_img_train=wi,
                                w_disagree_train=wd)
                        else:
                            q_R = torch.ones(B, 1, device=dev)
                            q_N = torch.ones(B, 1, device=dev)
                else:
                    q_R = None; q_N = None

                featR_in = featR.clone(); featN_in = featN.clone()
                featR = model.fusion_R(featR_in, featT, featN_in, q_R, q_N)
                featN = model.fusion_N(featN_in, featT, featR_in, q_N, q_R)

            feats_R.append(featR[:, 0].cpu())
            feats_N.append(featN[:, 0].cpu())
            feats_T.append(featT[:, 0].cpu())

        all_pids.extend(vid.tolist())

    return (torch.cat(feats_R, dim=0)[:num_query].numpy(),
            torch.cat(feats_N, dim=0)[:num_query].numpy(),
            torch.cat(feats_T, dim=0)[:num_query].numpy()), all_pids[:num_query]


def build_id_centroids(fR, fN, fT, pids):
    """Per-ID per-modality centroids. Only IDs with >= 3 queries."""
    id_to_idx = defaultdict(list)
    for i, pid in enumerate(pids):
        id_to_idx[pid].append(i)

    centroids = {}
    for pid, idxs in id_to_idx.items():
        if len(idxs) < 3:
            continue
        centroids[pid] = {
            0: np.mean([fR[i] for i in idxs], axis=0),
            1: np.mean([fN[i] for i in idxs], axis=0),
            2: np.mean([fT[i] for i in idxs], axis=0),
        }
    return centroids, id_to_idx


def cross_modal_distance(centroids_pid):
    """Mean pairwise distance between R, N, T centroids."""
    c = centroids_pid
    d01 = np.linalg.norm(c[0] - c[1])
    d02 = np.linalg.norm(c[0] - c[2])
    d12 = np.linalg.norm(c[1] - c[2])
    return (d01 + d02 + d12) / 3.0


def select_best_ids(qta_centroids, base_centroids, num_ids=5):
    """Select IDs where QTA reduces cross-modal distance most vs baseline."""
    common_ids = set(qta_centroids.keys()) & set(base_centroids.keys())
    scores = []
    for pid in common_ids:
        bd = cross_modal_distance(base_centroids[pid])
        qd = cross_modal_distance(qta_centroids[pid])
        if bd > 0:
            ratio = bd / qd
            scores.append((pid, ratio, bd, qd))
    scores.sort(key=lambda x: -x[1])
    return scores[:num_ids]


def build_tsne_data(fR, fN, fT, pids, id_to_idx, sel_ids):
    """Build feature matrix: sel_ids × up to 3 queries × 3 modalities."""
    data, id_l, mod_l = [], [], []
    for pid in sel_ids:
        for gi in id_to_idx[pid][:3]:
            for mod_i, f_arr in enumerate([fR, fN, fT]):
                data.append(f_arr[gi])
                id_l.append(pid)
                mod_l.append(mod_i)
    return np.stack(data), id_l, mod_l


def plot_tsne(ax, data, id_labels, mod_labels, title, id_to_color):
    """t-SNE: color=ID, marker=modality. Shows cross-modal clustering."""
    n = len(data)
    if n < 4:
        ax.text(0.5, 0.5, f'Too few points (n={n})', ha='center', va='center',
                transform=ax.transAxes)
        return

    # PCA pre-reduction (fix: n_components <= n_samples)
    pca_dim = min(50, n, data.shape[1])
    if data.shape[1] > pca_dim:
        data = PCA(n_components=pca_dim, random_state=42).fit_transform(data)

    perplexity = max(3, min(15, n // 3))
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=42,
               max_iter=3000, learning_rate='auto', init='pca').fit_transform(data)

    mod_marker = {0: 'o', 1: 's', 2: '^'}
    mod_name = {0: 'RGB', 1: 'NIR', 2: 'TIR'}
    sorted_ids = sorted(set(id_labels))

    # Draw convex hulls per ID
    for pid in sorted_ids:
        idxs = [i for i in range(n) if id_labels[i] == pid]
        if len(idxs) >= 3:
            pts = emb[idxs]
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1],
                            '-', color=id_to_color[pid], alpha=0.25, linewidth=1.5)
            except Exception:
                pass

    # Scatter
    for pid in sorted_ids:
        for mod in [0, 1, 2]:
            idxs = [i for i in range(n)
                    if id_labels[i] == pid and mod_labels[i] == mod]
            if not idxs:
                continue
            ax.scatter(emb[idxs, 0], emb[idxs, 1],
                       c=[id_to_color[pid]], marker=mod_marker[mod],
                       s=100, alpha=0.85, edgecolors='k', linewidth=0.5, zorder=3,
                       label=f'{mod_name[mod]}' if pid == sorted_ids[0] else "")

    # Modality legend
    mod_handles = [Line2D([0], [0], marker=mod_marker[m], color='k',
                          markerfacecolor='gray', markersize=8,
                          label=mod_name[m]) for m in [0, 1, 2]]
    ax.legend(handles=mod_handles, loc='best', fontsize=8, framealpha=0.85,
              title='Modality')
    ax.set_title(title, fontsize=12, fontweight='bold')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


def plot_centroid_bar(ax, sel_scores, qta_centroids, base_centroids):
    """Bar chart: cross-modal distance per ID, QTA vs Baseline."""
    pids = [s[0] for s in sel_scores]
    x = np.arange(len(pids))
    w = 0.35

    base_dists = [cross_modal_distance(base_centroids[pid]) for pid in pids]
    qta_dists = [cross_modal_distance(qta_centroids[pid]) for pid in pids]
    ratios = [s[1] for s in sel_scores]

    ax.bar(x - w/2, base_dists, w, label='Baseline',
           color='#95A5A6', edgecolor='#7F8C8D')
    ax.bar(x + w/2, qta_dists, w, label='QTA-ReID',
           color='#3498DB', edgecolor='#2980B9')

    for i, (bd, qd, r) in enumerate(zip(base_dists, qta_dists, ratios)):
        ax.annotate(f'{r:.1f}×', (x[i], max(bd, qd) + 0.3),
                    ha='center', fontsize=8, fontweight='bold', color='#E74C3C')

    ax.set_xticks(x)
    ax.set_xticklabels([f'ID {p}' for p in pids], fontsize=9)
    ax.set_ylabel('Cross-modal Centroid Distance')
    ax.set_title('Cross-modal Alignment per ID\n(lower = better | ratio = Base/QTA)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--weight_qta", required=True)
    p.add_argument("--weight_base", required=True)
    p.add_argument("--num_ids", type=int, default=5)
    p.add_argument("--output_dir", default="./tsne_results/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, vl, nq, nc, _ = make_dataloader(cfg)
    print(f"Device={device}  Queries={nq}  Classes={nc}")

    all_data = {}
    for name, wt, is_abl in [("QTA-ReID", args.weight_qta, False),
                              ("CLIP-ViT Baseline", args.weight_base, True)]:
        print(f"\n{'='*50}\nLoading: {name}")
        m = CLIPFACENet(num_classes=nc, cfg=cfg).to(device)
        if is_abl:
            apply_ablation_config(m, "backbone")
        st = torch.load(wt, map_location=device)
        if "state_dict" in st:
            st = st["state_dict"]
        own = m.state_dict()
        ns = {}
        for k, v in st.items():
            kn = k[7:] if k.startswith("module.") else k
            if kn in own and own[kn].shape == v.shape:
                ns[kn] = v
        m.load_state_dict(ns, strict=False)
        print(f"  Matched keys: {len(ns)}")

        (fR, fN, fT), pids = extract_prebn_features(m, vl, nq, device)
        centroids, id_to_idx = build_id_centroids(fR, fN, fT, pids)
        all_data[name] = (fR, fN, fT, pids, centroids, id_to_idx)

    qta_centroids = all_data["QTA-ReID"][4]
    base_centroids = all_data["CLIP-ViT Baseline"][4]

    sel_scores = select_best_ids(qta_centroids, base_centroids, args.num_ids)
    sel_ids = [s[0] for s in sel_scores]

    print(f"\nSelected IDs (largest cross-modal improvement):")
    for pid, ratio, bd, qd in sel_scores:
        print(f"  ID {pid}: Base={bd:.2f} QTA={qd:.2f} Ratio={ratio:.2f}×")

    qta_fR, qta_fN, qta_fT, qta_pids, _, qta_idx = all_data["QTA-ReID"]
    base_fR, base_fN, base_fT, base_pids, _, base_idx = all_data["CLIP-ViT Baseline"]

    data_qta, id_qta, mod_qta = build_tsne_data(qta_fR, qta_fN, qta_fT,
                                                  qta_pids, qta_idx, sel_ids)
    data_base, id_base, mod_base = build_tsne_data(base_fR, base_fN, base_fT,
                                                     base_pids, base_idx, sel_ids)

    id_colors = plt.cm.tab10(np.linspace(0, 1, max(len(sel_ids), 10)))
    id_to_color = {p: id_colors[i % 10] for i, p in enumerate(sel_ids)}

    # ====== Figure 1: t-SNE side by side ======
    fig1, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7))

    plot_tsne(ax_l, data_base, id_base, mod_base,
              f'CLIP-ViT Baseline\n({len(sel_ids)} IDs, pre-BNNeck features)',
              id_to_color)
    plot_tsne(ax_r, data_qta, id_qta, mod_qta,
              f'QTA-ReID\n({len(sel_ids)} IDs, pre-BNNeck features)',
              id_to_color)

    id_handles = [Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=id_to_color[p], markersize=9,
                         label=f'ID {p}') for p in sel_ids]
    fig1.legend(handles=id_handles, loc='upper center',
                ncol=len(sel_ids), fontsize=8, framealpha=0.85,
                title='Vehicle Identity', bbox_to_anchor=(0.5, -0.02))

    fig1.suptitle(
        'E7: t-SNE — Best Cross-modal Alignment IDs\n'
        'Color = Vehicle ID  |  ○=RGB  □=NIR  △=TIR  |  Hull = ID cluster extent',
        fontsize=13, fontweight='bold')
    fig1.tight_layout()
    sp1 = os.path.join(args.output_dir, "e7_tsne_best_ids.png")
    fig1.savefig(sp1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {sp1}")

    # ====== Figure 2: Cross-modal distance bar chart ======
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    plot_centroid_bar(ax2, sel_scores, qta_centroids, base_centroids)
    fig2.suptitle('E7: Cross-modal Centroid Distance — QTA-ReID vs Baseline',
                  fontsize=13, fontweight='bold')
    fig2.tight_layout()
    sp2 = os.path.join(args.output_dir, "e7_crossmodal_bars.png")
    fig2.savefig(sp2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {sp2}")


if __name__ == "__main__":
    main()
