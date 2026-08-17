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

    # CLIP denorm
    mean = img.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = img.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    x = (img * std + mean).clamp(0, 1)

    gray = x.mean(dim=1, keepdim=True)
    max_rgb = x.max(dim=1, keepdim=True).values
    min_rgb = x.min(dim=1, keepdim=True).values

    over = (max_rgb > 0.88).float()
    near_white = ((max_rgb > 0.82) & ((max_rgb - min_rgb) < 0.18)).float()

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

    dist2 = (xx / 0.95) ** 2 + (yy / 0.85) ** 2
    prior = torch.exp(-dist2)

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
    patches_R = featR[:, 1:, :]
    patches_N = featN[:, 1:, :]
    
    sim = F.cosine_similarity(patches_R, patches_N, dim=-1)
    
    if per_patch:
        quality = (sim - 0.2) / 0.7
        return 0.15 + 0.85 * quality.clamp(0.0, 1.0)
    else:
        mean_sim = sim.mean(dim=1)
        quality = (mean_sim - 0.2) / 0.7
        return (0.15 + 0.85 * quality.clamp(0.0, 1.0)).unsqueeze(-1)


def _smooth_patch_map(x):
    """
    Smooth [B,L,1] patch map with 3x3 avg pooling.
    """
    if x is None:
        return None

    B, L, C = x.shape
    h = int(L ** 0.5)
    w = h

    if h * w != L:
        return x

    x2d = x.view(B, h, w, C).permute(0, 3, 1, 2)
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
    bad_learn_N=None,
    w_learned_train=0.15,
    w_img_train=0.55,
    w_disagree_train=0.30,
    w_learned_eval=0.05,
    w_img_eval=0.70,
    w_disagree_eval=0.25
):
    """
    H2 Hybrid q_map.

    Training:
        full H2 = 0.15 learned + 0.55 image_prior + 0.30 disagreement attribution

    Inference:
        lightweight H2 = image_prior-dominant + conservative learned correction
        learned qmap only works where image prior or disagreement supports damage.
    """

    # 1. Image prior (always 14×14 = 196 for 224×224 input)
    q_img_R, bad_img_R = compute_image_quality_map(img_R)
    q_img_N, bad_img_N = compute_image_quality_map(img_N)

    # 2. Learned qmap (if absent, init as zeros)
    if bad_learn_R is None:
        bad_learn_R = torch.zeros_like(bad_img_R)
    if bad_learn_N is None:
        bad_learn_N = torch.zeros_like(bad_img_N)

    # 3. Foreground prior
    B = img_R.size(0)
    fg_prior = build_center_foreground_prior(B, img_R.device, img_R.dtype)

    bad_img_R = bad_img_R * fg_prior
    bad_img_N = bad_img_N * fg_prior

    use_feature_quality = featR is not None and featN is not None

    # --- Resize image priors to match feature patch count (for CNN backbones) ---
    if use_feature_quality:
        num_feat_patches = featR[:, 1:, :].size(1)  # 196 ViT, 49 CNN (7×7)
        if num_feat_patches != 196:
            h_new = int(num_feat_patches ** 0.5)
            def _resize(p):
                p2d = p.reshape(B, 14, 14, 1).permute(0, 3, 1, 2)
                p2d = F.interpolate(p2d, size=(h_new, h_new),
                                    mode='bilinear', align_corners=False)
                return p2d.permute(0, 2, 3, 1).reshape(B, num_feat_patches, 1)
            bad_img_R = _resize(bad_img_R)
            bad_img_N = _resize(bad_img_N)
            q_img_R = _resize(q_img_R)
            q_img_N = _resize(q_img_N)
            bad_learn_R = _resize(bad_learn_R)
            bad_learn_N = _resize(bad_learn_N)

    if use_feature_quality:
        patches_R = featR[:, 1:, :]
        patches_N = featN[:, 1:, :]

        sim = F.cosine_similarity(patches_R, patches_N, dim=-1).unsqueeze(-1)

        disagree = (1.0 - ((sim - 0.2) / 0.7).clamp(0, 1)).clamp(0, 1)

        bad_sum = bad_img_R + bad_img_N + 1e-6
        assign_R = bad_img_R / bad_sum
        assign_N = bad_img_N / bad_sum

        attrib_R = disagree * assign_R
        attrib_N = disagree * assign_N

        if training:
            bad_R = (
                w_learned_train * bad_learn_R
                + w_img_train * bad_img_R
                + w_disagree_train * attrib_R
            )
            bad_N = (
                w_learned_train * bad_learn_N
                + w_img_train * bad_img_N
                + w_disagree_train * attrib_N
            )
        else:
            bad_learn_R_s = _smooth_patch_map(bad_learn_R)
            bad_learn_N_s = _smooth_patch_map(bad_learn_N)

            support_R = ((bad_img_R > 0.06) | (attrib_R > 0.08)).float()
            support_N = ((bad_img_N > 0.06) | (attrib_N > 0.08)).float()

            bad_learn_R_eff = bad_learn_R_s * support_R
            bad_learn_N_eff = bad_learn_N_s * support_N

            bad_R = (
                w_learned_eval * bad_learn_R_eff
                + w_img_eval * bad_img_R
                + w_disagree_eval * attrib_R
            )
            bad_N = (
                w_learned_eval * bad_learn_N_eff
                + w_img_eval * bad_img_N
                + w_disagree_eval * attrib_N
            )
    else:
        if training:
            bad_R = w_learned_train * bad_learn_R + w_img_train * bad_img_R
            bad_N = w_learned_train * bad_learn_N + w_img_train * bad_img_N
        else:
            bad_learn_R_s = _smooth_patch_map(bad_learn_R)
            bad_learn_N_s = _smooth_patch_map(bad_learn_N)
            support_R = (bad_img_R > 0.06).float()
            support_N = (bad_img_N > 0.06).float()
            bad_R = w_learned_eval * bad_learn_R_s * support_R + w_img_eval * bad_img_R
            bad_N = w_learned_eval * bad_learn_N_s * support_N + w_img_eval * bad_img_N

    q_R = 1.0 - bad_R.clamp(0, 1)
    q_N = 1.0 - bad_N.clamp(0, 1)
    q_R = 0.15 + 0.85 * q_R
    q_N = 0.15 + 0.85 * q_N

    return q_R.clamp(0.15, 1.0), q_N.clamp(0.15, 1.0)


def patch_quality_to_scalar(q, worst_ratio=0.3):
    """
    Aggregate patch-level quality to sample-level quality.
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
