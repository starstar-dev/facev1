"""
CoEN-Lite: Quality detection + feature repair for flare-robust ReID.

Quality detection (two levels):
1. Image-level: saturation + texture from raw pixels (always available)
2. Cross-modal: R↔N patch similarity after MFMP alignment (training only)
   - Follows original FACENet's FCE dependency on MFMP:
     FCE used MFMP features to detect damaged patches → hard replacement
     CoEN uses MFMP features to compute soft quality → soft gating + repair

v2 (token-level): per-patch quality instead of per-sample scalar.
  - L2 cross-modal quality now returns (B, 196) per-patch
  - Training combined quality: (B, 196) per-patch
  - Inference: (B,) scalar (image-level only, no MFMP)
  - CrossModalFusion gates per-patch → local flare, local repair
  - Loss gate aggregates to scalar: q_sample = q_patches.mean(dim=1)
"""

import torch
import torch.nn.functional as F


def compute_image_quality(img, saturation_thresh=1.8):
    """
    Pixel-level quality from raw images.
    
    Signals:
    1. Saturation: fraction of near-white pixels (flare signature)
    2. Texture: gradient magnitude (flare washes out edges)
    
    Args:
        img: (B, 3, H, W) CLIP-preprocessed images
    Returns:
        quality: (B,) in [0.15, 1.0]
    """
    B, C, H, W = img.shape
    
    saturated = (img > saturation_thresh).float().reshape(B, -1).mean(dim=1)
    
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    grad_mag = (dy[:, :, :, :-1] + dx[:, :, :-1, :]).reshape(B, -1).mean(dim=1)
    texture = (grad_mag - 0.1) / 0.7
    
    raw_q = (1.0 - 0.6 * saturated) * (0.2 + 0.8 * texture.clamp(0.0, 1.0))
    return 0.15 + 0.85 * raw_q.clamp(0.0, 1.0)

def compute_image_quality_map(img, patch_size=16):
    """
    Patch-level image quality from CLIP-normalized image.

    Args:
        img: [B,3,224,224]
    Returns:
        q_img:   [B,196,1], high = clean
        bad_img: [B,196,1], high = damaged / flare-like
    """
    B, C, H, W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0

    # CLIP denorm，避免直接在 normalized 空间判断亮度
    mean = img.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = img.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    x = (img * std + mean).clamp(0, 1)

    gray = x.mean(dim=1, keepdim=True)
    max_rgb = x.max(dim=1, keepdim=True).values
    min_rgb = x.min(dim=1, keepdim=True).values

    # 1) 高亮 / 过曝
    over = (max_rgb > 0.88).float()

    # 2) 近白区域，车灯/反光常见
    near_white = ((max_rgb > 0.82) & ((max_rgb - min_rgb) < 0.18)).float()

    # 3) 纹理损失：梯度太低但亮度高，更像耀斑/过曝
    dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
    dy = F.pad(dy, (0, 0, 0, 1))
    grad = dx + dy

    over_patch = F.avg_pool2d(over, patch_size, patch_size).flatten(1)
    white_patch = F.avg_pool2d(near_white, patch_size, patch_size).flatten(1)
    grad_patch = F.avg_pool2d(grad, patch_size, patch_size).flatten(1)

    low_texture = (1.0 - (grad_patch / 0.18).clamp(0, 1))

    bad = 0.45 * over_patch + 0.35 * white_patch + 0.20 * (over_patch * low_texture)
    bad = bad.clamp(0, 1)

    q = 1.0 - bad
    q = 0.15 + 0.85 * q

    return q.unsqueeze(-1), bad.unsqueeze(-1)

def build_center_foreground_prior(batch_size, device, dtype, h=14, w=14):
    """
    Simple foreground prior for cropped vehicle ReID images.
    Suppress border/background responses.
    Return: [B,196,1]
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing="ij"
    )

    # 车辆一般在 crop 中心，边缘/底部背景适当压低
    dist2 = (xx / 0.95) ** 2 + (yy / 0.85) ** 2
    prior = torch.exp(-dist2)

    # 不完全屏蔽边缘，只是降权
    prior = 0.35 + 0.65 * prior
    prior = prior.reshape(1, h * w, 1).expand(batch_size, -1, -1)

    return prior

def compute_cross_modal_quality(featR, featN, per_patch=True):
    """
    Feature-level quality from MFMP-aligned R↔N features.
    
    After MFMP aligns R and N patches in feature space:
    - High R↔N patch similarity → both modalities clean
    - Low similarity → one is damaged by flare
    
    v2: returns per-patch quality (B, 196) to enable token-level repair.
    
    Args:
        featR: (B, N, D) MFMP-aligned R features (CLS + patches)
        featN: (B, N, D) MFMP-aligned N features
        per_patch: if True, return (B, 196); if False, return (B,)
    Returns:
        quality: (B, 196) or (B,) in [0.15, 1.0]
    """
    # Exclude CLS token, compare patch tokens
    patches_R = featR[:, 1:, :]  # (B, 196, 768)
    patches_N = featN[:, 1:, :]
    
    # Per-patch cosine similarity after MFMP alignment
    sim = F.cosine_similarity(patches_R, patches_N, dim=-1)  # (B, 196)
    
    if per_patch:
        # Token-level: each patch has its own quality score
        quality = (sim - 0.2) / 0.7
        return 0.15 + 0.85 * quality.clamp(0.0, 1.0)  # (B, 196)
    else:
        mean_sim = sim.mean(dim=1)  # (B,)
        quality = (mean_sim - 0.2) / 0.7
        return (0.15 + 0.85 * quality.clamp(0.0, 1.0)).unsqueeze(-1)


def _smooth_patch_map(x):
    """
    Smooth [B,196,1] patch map with 3x3 avg pooling.
    This reduces noisy isolated learned qmap responses at inference.
    """
    if x is None:
        return None

    B, L, C = x.shape
    h = int(L ** 0.5)
    w = h

    if h * w != L:
        return x

    x2d = x.view(B, h, w, C).permute(0, 3, 1, 2)  # [B,1,14,14]
    x2d = F.avg_pool2d(x2d, kernel_size=3, stride=1, padding=1)
    x = x2d.permute(0, 2, 3, 1).reshape(B, L, C)

    return x


def compute_combined_quality(
    img_R,
    img_N,
    featR,
    featN,
    training,
    bad_learn_R=None,
    bad_learn_N=None
):
    """
    H2 Hybrid q_map.

    Training:
        full H2 = 0.15 learned + 0.55 image_prior + 0.30 disagreement attribution

    Inference:
        lightweight H2 = image_prior-dominant + conservative learned correction
        learned qmap only works where image prior or disagreement supports damage.
    """

    # 1. Image prior must be computed first
    q_img_R, bad_img_R = compute_image_quality_map(img_R)  # [B,196,1]
    q_img_N, bad_img_N = compute_image_quality_map(img_N)  # [B,196,1]

    # 2. Now bad_img_R/N exist, so zeros_like is safe
    if bad_learn_R is None:
        bad_learn_R = torch.zeros_like(bad_img_R)
    if bad_learn_N is None:
        bad_learn_N = torch.zeros_like(bad_img_N)

    # 3. Foreground prior: suppress border/background false positives
    B = img_R.size(0)
    fg_prior = build_center_foreground_prior(
        batch_size=B,
        device=img_R.device,
        dtype=img_R.dtype
    )

    bad_img_R = bad_img_R * fg_prior
    bad_img_N = bad_img_N * fg_prior

    use_feature_quality = featR is not None and featN is not None

    if use_feature_quality:
        patches_R = featR[:, 1:, :]  # [B,196,768]
        patches_N = featN[:, 1:, :]  # [B,196,768]

        sim = F.cosine_similarity(patches_R, patches_N, dim=-1).unsqueeze(-1)

        # R/N disagreement. high = R/N token mismatch
        disagree = (1.0 - ((sim - 0.2) / 0.7).clamp(0, 1)).clamp(0, 1)

        # Attribute disagreement to the modality whose image prior looks worse
        bad_sum = bad_img_R + bad_img_N + 1e-6
        assign_R = bad_img_R / bad_sum
        assign_N = bad_img_N / bad_sum

        attrib_R = disagree * assign_R
        attrib_N = disagree * assign_N

        if training:
            # 原 H2：训练阶段保留完整 learned qmap 参与
            bad_R = (
                0.15 * bad_learn_R
                + 0.55 * bad_img_R
                + 0.30 * attrib_R
            )
            bad_N = (
                0.15 * bad_learn_N
                + 0.55 * bad_img_N
                + 0.30 * attrib_N
            )

        else:
            # 推理轻量 H2：
            # 1) smooth learned qmap，去掉孤立噪声点
            # 2) leaned 只在有 image prior 或 disagreement 支持的位置生效
            # 3) learned 权重从 0.15 降到 0.05
            bad_learn_R_s = _smooth_patch_map(bad_learn_R)
            bad_learn_N_s = _smooth_patch_map(bad_learn_N)

            support_R = ((bad_img_R > 0.06) | (attrib_R > 0.08)).float()
            support_N = ((bad_img_N > 0.06) | (attrib_N > 0.08)).float()

            bad_learn_R_eff = bad_learn_R_s * support_R
            bad_learn_N_eff = bad_learn_N_s * support_N

            bad_R = (
                0.0* bad_learn_R_eff
                + 0.80 * bad_img_R
                + 0.20 * attrib_R
            )
            bad_N = (
                0.0 * bad_learn_N_eff
                + 0.80 * bad_img_N
                + 0.20 * attrib_N
            )

    else:
        # 一般不会走到这里，因为 forward 会传 featR_mfmp/featN_mfmp
        if training:
            bad_R = 0.15 * bad_learn_R + 0.85 * bad_img_R
            bad_N = 0.15 * bad_learn_N + 0.85 * bad_img_N
        else:
            bad_learn_R_s = _smooth_patch_map(bad_learn_R)
            bad_learn_N_s = _smooth_patch_map(bad_learn_N)

            support_R = (bad_img_R > 0.06).float()
            support_N = (bad_img_N > 0.06).float()

            bad_R = 0.05 * bad_learn_R_s * support_R + 0.95 * bad_img_R
            bad_N = 0.05 * bad_learn_N_s * support_N + 0.95 * bad_img_N

    # bad map -> quality map
    q_R = 1.0 - bad_R.clamp(0, 1)
    q_N = 1.0 - bad_N.clamp(0, 1)

    q_R = 0.15 + 0.85 * q_R
    q_N = 0.15 + 0.85 * q_N

    return q_R.clamp(0.15, 1.0), q_N.clamp(0.15, 1.0)


def patch_quality_to_scalar(q, worst_ratio=0.3):
    """
    Aggregate patch-level quality to sample-level quality.

    Args:
        q: [B, 196, 1] or [B, 196] or [B]
    Returns:
        q_sample: [B]
    """
    if q is None:
        return None

    if q.dim() == 1:
        return q

    if q.dim() == 3:
        q = q.squeeze(-1)

    k = max(1, int(q.size(1) * worst_ratio))
    return q.topk(k, dim=1, largest=False).values.mean(dim=1)


def compute_modality_quality(img_R, img_N, saturation_thresh=1.8):
    """Batch-mean quality (backward compat for processor)."""
    q_R = compute_image_quality(img_R, saturation_thresh).mean()
    q_N = compute_image_quality(img_N, saturation_thresh).mean()
    return q_R, q_N


def quality_gate(quality, floor=0.3):
    """Map quality to loss weight: [floor, 1.0]."""
    return floor + (1.0 - floor) * quality
