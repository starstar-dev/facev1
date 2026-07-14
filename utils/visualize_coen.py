import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def denorm_clip(img):
    """
    img: [3,H,W], CLIP normalized
    """
    mean = CLIP_MEAN.to(img.device)
    std = CLIP_STD.to(img.device)
    img = img * std + mean
    return img.clamp(0, 1)


def save_qmap_overlay(img, q_map, save_path, title="q_map"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = denorm_clip(img).detach().cpu()
    img_np = img.permute(1, 2, 0).numpy()

    q = q_map.detach().cpu()
    if q.dim() == 2:
        q = q.squeeze(-1)

    q = q.view(1, 1, 14, 14)

    bad_map = 1.0 - q
    bad_map = F.interpolate(
        bad_map,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )[0, 0].numpy()

    bad_min = float(bad_map.min())
    bad_mean = float(bad_map.mean())
    bad_max = float(bad_map.max())

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img_np)
    im = axes[1].imshow(
        bad_map,
        cmap="jet",
        alpha=0.45,
        vmin=0.0,
        vmax=1.0
    )
    axes[1].set_title(
        f"{title}\n"
        f"bad_mean={bad_mean:.3f}, bad_max={bad_max:.3f}"
    )
    axes[1].axis("off")

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_batch_qmap_visualization(model, save_dir, epoch, max_samples=4):
    """
    Save one paper-style visualization figure:
    each row = RGB image | RGB low-quality map | NIR image | NIR low-quality map.
    Top-k samples are selected by low-quality score.
    """
    raw_model = model.module if hasattr(model, "module") else model

    if not hasattr(raw_model, "_last_q_R_map"):
        return

    os.makedirs(save_dir, exist_ok=True)

    img_R = raw_model._last_img_R
    img_N = raw_model._last_img_N
    q_R = raw_model._last_q_R_map
    q_N = raw_model._last_q_N_map

    # Select samples with stronger degradation response.
    bad_score_R = (1.0 - q_R).mean(dim=(1, 2))
    bad_score_N = (1.0 - q_N).mean(dim=(1, 2))
    bad_score = bad_score_R + bad_score_N

    k = min(max_samples, img_R.size(0))
    topk_idx = torch.topk(bad_score, k=k).indices.tolist()

    save_paper_qmap_grid(
        img_R=img_R,
        img_N=img_N,
        q_R=q_R,
        q_N=q_N,
        indices=topk_idx,
        save_path=os.path.join(save_dir, f"epoch{epoch:03d}_paper_qmap_grid.png")
    )

def qmap_to_badmap(q_map, size=(224, 224)):
    q = q_map.detach().cpu()
    if q.dim() == 2:
        q = q.squeeze(-1)

    q = q.view(1, 1, 14, 14)
    bad_map = 1.0 - q

    bad_map = F.interpolate(
        bad_map,
        size=size,
        mode="bilinear",
        align_corners=False
    )[0, 0].numpy()

    return bad_map


def tensor_to_img_np(img):
    img = denorm_clip(img).detach().cpu()
    return img.permute(1, 2, 0).numpy()


def save_paper_qmap_grid(img_R, img_N, q_R, q_N, indices, save_path):
    """
    Paper-style visualization:
    rows are selected samples;
    columns are RGB image, RGB low-quality map, NIR image, NIR low-quality map.
    """
    num_rows = len(indices)
    num_cols = 4

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 3.0, num_rows * 2.6)
    )

    if num_rows == 1:
        axes = axes[None, :]

    col_titles = [
        "RGB",
        "RGB low-quality map",
        "NIR",
        "NIR low-quality map"
    ]

    for c in range(num_cols):
        axes[0, c].set_title(col_titles[c], fontsize=10)

    last_im = None

    for row, i in enumerate(indices):
        rgb_np = tensor_to_img_np(img_R[i])
        nir_np = tensor_to_img_np(img_N[i])

        bad_R = qmap_to_badmap(q_R[i])
        bad_N = qmap_to_badmap(q_N[i])

        axes[row, 0].imshow(rgb_np)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(rgb_np)
        last_im = axes[row, 1].imshow(
            bad_R,
            cmap="jet",
            alpha=0.45,
            vmin=0.0,
            vmax=1.0
        )
        axes[row, 1].axis("off")

        axes[row, 2].imshow(nir_np)
        axes[row, 2].axis("off")

        axes[row, 3].imshow(nir_np)
        last_im = axes[row, 3].imshow(
            bad_N,
            cmap="jet",
            alpha=0.45,
            vmin=0.0,
            vmax=1.0
        )
        axes[row, 3].axis("off")

    # One shared colorbar for the whole figure.
    cbar = fig.colorbar(
        last_im,
        ax=axes,
        fraction=0.018,
        pad=0.01
    )
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Low-quality score", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()