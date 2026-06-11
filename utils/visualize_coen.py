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
    """
    img:   [3,224,224]
    q_map: [196,1] or [196]
    低 q = 低质量 patch，用热图显示。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = denorm_clip(img).detach().cpu()
    img_np = img.permute(1, 2, 0).numpy()

    q = q_map.detach().cpu()
    if q.dim() == 2:
        q = q.squeeze(-1)

    q = q.view(1, 1, 14, 14)

    # 低质量区域应该更亮：bad_map = 1 - q
    bad_map = 1.0 - q
    bad_map = F.interpolate(
        bad_map,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )[0, 0].numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(img_np)
    plt.imshow(bad_map, cmap="jet", alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_batch_qmap_visualization(model, save_dir, epoch, max_samples=4):
    """
    保存 RGB / NIR 的 q_map 可视化。
    """
    raw_model = model.module if hasattr(model, "module") else model

    if not hasattr(raw_model, "_last_q_R_map"):
        return

    img_R = raw_model._last_img_R
    img_N = raw_model._last_img_N
    q_R = raw_model._last_q_R_map
    q_N = raw_model._last_q_N_map

    bsz = min(max_samples, img_R.size(0))

    for i in range(bsz):
        save_qmap_overlay(
            img_R[i],
            q_R[i],
            os.path.join(save_dir, f"epoch{epoch:03d}_sample{i}_RGB_qmap.png"),
            title=f"Epoch {epoch} RGB low-quality map"
        )
        save_qmap_overlay(
            img_N[i],
            q_N[i],
            os.path.join(save_dir, f"epoch{epoch:03d}_sample{i}_NIR_qmap.png"),
            title=f"Epoch {epoch} NIR low-quality map"
        )