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
        return 0.15 + 0.85 * quality.clamp(0.0, 1.0)  # (B,)


def compute_combined_quality(img_R, img_N, featR, featN, training):
    """
    Combine image-level and cross-modal quality.
    
    Training: 0.3 × image + 0.7 × cross_modal → per-patch (B, 196)
              image quality (B,1) broadcasts to per-patch
    Inference: 1.0 × image → per-sample (B,)
              (MFMP off, fallback to pixel-level only)
    
    Returns:
        q_R, q_N: (B, 196) during training, (B,) during inference
    """
    q_img_R = compute_image_quality(img_R)  # (B,)
    q_img_N = compute_image_quality(img_N)  # (B,)
    
    if training and featR is not None and featN is not None:
        q_cross = compute_cross_modal_quality(featR, featN, per_patch=True)  # (B, 196)
        # Broadcast image quality to per-patch for weighted combination
        q_R = 0.3 * q_img_R.unsqueeze(1) + 0.7 * q_cross  # (B, 196)
        q_N = 0.3 * q_img_N.unsqueeze(1) + 0.7 * q_cross  # (B, 196)
    else:
        q_R = q_img_R  # (B,)
        q_N = q_img_N  # (B,)
    
    return q_R, q_N


def patch_quality_to_scalar(q):
    """
    Aggregate per-patch quality to per-sample scalar.
    
    Used for:
    - Logging (single number per sample)
    - Loss gating (loss is per-sample)
    
    Args:
        q: (B, 196) per-patch or (B,) scalar
    Returns:
        (B,) per-sample scalar
    """
    if q.dim() == 2:  # (B, 196)
        return q.mean(dim=1)  # (B,)
    return q  # already (B,)


def compute_modality_quality(img_R, img_N, saturation_thresh=1.8):
    """Batch-mean quality (backward compat for processor)."""
    q_R = compute_image_quality(img_R, saturation_thresh).mean()
    q_N = compute_image_quality(img_N, saturation_thresh).mean()
    return q_R, q_N


def quality_gate(quality, floor=0.3):
    """Map quality to loss weight: [floor, 1.0]."""
    return floor + (1.0 - floor) * quality
