import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from config import cfg
from datasets import make_dataloader
from model.clip_facenet import CLIPFACENet
from utils.ablation import ABLATION_CONFIGS, apply_ablation_config
from utils.metrics import euclidean_distance


def compute_ap(order, good_mask):
    """
    order: sorted gallery indices
    good_mask: bool array, shape [num_gallery], True means same ID valid match
    """
    matches = good_mask[order].astype(np.int32)

    if matches.sum() == 0:
        return 0.0

    cmc = matches.cumsum()
    precision = cmc / (np.arange(len(matches)) + 1.0)
    ap = (precision * matches).sum() / matches.sum()
    return float(ap)


def collect_all_features(cfg, weight_path, exp_name):
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

    with torch.no_grad():
        for img1, img2, img3, vid, camid, camids_batch, viewids, img_paths, flare_label in val_loader:
            img1 = img1.cuda()
            img2 = img2.cuda()
            img3 = img3.cuda()
            flare_label = flare_label.cuda()

            feat = model(img1, img2, img3, vid.cuda(), flare_label=flare_label)
            feat = torch.nn.functional.normalize(feat, dim=1, p=2)

            feats.append(feat.cpu())
            pids.extend(np.asarray(vid))
            camids.extend(np.asarray(camid))

    feats = torch.cat(feats, dim=0)
    pids = np.asarray(pids)
    camids = np.asarray(camids)

    return feats, pids, camids, num_query


def select_good_ids(feats, pids, camids, num_query, max_ids=8, min_ap=0.8):
    qf = feats[:num_query]
    gf = feats[num_query:]

    q_pids = pids[:num_query]
    g_pids = pids[num_query:]

    q_camids = camids[:num_query]
    g_camids = camids[num_query:]

    distmat = euclidean_distance(qf, gf)

    candidates = []

    for qi in range(num_query):
        q_pid = q_pids[qi]
        q_camid = q_camids[qi]

        order = np.argsort(distmat[qi])

        # ReID 协议：去掉同 ID 同 camera
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        order = order[~remove]

        if len(order) == 0:
            continue

        top1_correct = g_pids[order[0]] == q_pid

        good_mask = (g_pids == q_pid)
        good_mask = good_mask & ~(g_camids == q_camid)

        ap = compute_ap(order, good_mask)

        if top1_correct and ap >= min_ap:
            candidates.append((q_pid, ap))

    # 同一个 ID 可能有多个 query，只保留最高 AP
    id_best_ap = {}
    for pid, ap in candidates:
        if pid not in id_best_ap:
            id_best_ap[pid] = ap
        else:
            id_best_ap[pid] = max(id_best_ap[pid], ap)

    sorted_ids = sorted(id_best_ap.items(), key=lambda x: x[1], reverse=True)
    selected_ids = [pid for pid, ap in sorted_ids[:max_ids]]

    print("Selected IDs:")
    for pid, ap in sorted_ids[:max_ids]:
        print(f"  PID {pid}, AP={ap:.3f}")

    return selected_ids


def sample_features_by_ids(feats, pids, selected_ids, max_per_id=8):
    selected_features = []
    selected_labels = []

    id_count = {pid: 0 for pid in selected_ids}

    for idx, pid in enumerate(pids):
        if pid not in id_count:
            continue

        if id_count[pid] >= max_per_id:
            continue

        selected_features.append(feats[idx].numpy())
        selected_labels.append(pid)
        id_count[pid] += 1

        if all(v >= max_per_id for v in id_count.values()):
            break

    selected_features = np.asarray(selected_features)
    selected_labels = np.asarray(selected_labels)

    return selected_features, selected_labels


def plot_tsne(features, labels, save_path, title):
    n_samples = len(features)

    perplexity = min(20, max(5, (n_samples - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )

    emb = tsne.fit_transform(features)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 7))

    unique_ids = list(dict.fromkeys(labels.tolist()))
    cmap = plt.get_cmap("tab10")

    for idx, pid in enumerate(unique_ids):
        mask = labels == pid
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=55,
            color=cmap(idx % 10),
            edgecolors="black",
            linewidths=0.35,
            alpha=0.88,
            label=str(pid),
        )

    plt.title(title, fontsize=14)
    plt.xticks([])
    plt.yticks([])

    plt.legend(
        title="ID",
        fontsize=8,
        title_fontsize=9,
        loc="best",
        ncol=2,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved t-SNE figure to {save_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", default="configs/WMVeID863/clip_facenet_wmveid863.yml")
    parser.add_argument("--weight", required=True)
    parser.add_argument("--exp", default="original_G", choices=list(ABLATION_CONFIGS.keys()))

    parser.add_argument("--save_path", default="./vis/tsne_final_good_ids.png")
    parser.add_argument("--max_ids", type=int, default=8)
    parser.add_argument("--max_per_id", type=int, default=8)
    parser.add_argument("--min_ap", type=float, default=0.8)
    parser.add_argument("--title", default="t-SNE of final retrieval features")

    parser.add_argument("opts", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    feats, pids, camids, num_query = collect_all_features(
        cfg=cfg,
        weight_path=args.weight,
        exp_name=args.exp,
    )

    selected_ids = select_good_ids(
        feats=feats,
        pids=pids,
        camids=camids,
        num_query=num_query,
        max_ids=args.max_ids,
        min_ap=args.min_ap,
    )

    features, labels = sample_features_by_ids(
        feats=feats,
        pids=pids,
        selected_ids=selected_ids,
        max_per_id=args.max_per_id,
    )

    plot_tsne(
        features=features,
        labels=labels,
        save_path=args.save_path,
        title=args.title,
    )


if __name__ == "__main__":
    main()