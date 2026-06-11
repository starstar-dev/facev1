"""
CoEN-Lite: Quality detection + feature repair for flare-robust ReID.

Quality detection (two levels):
1. Image-level: saturation + texture from raw pixels (always available)
2. Cross-modal: R↔N patch similarity after MFMP alignment (training only)
   - Follows original FACENet's FCE dependency on MFMP:
     FCE used MFMP features to detect damaged patches → hard replacement
     CoEN uses MFMP features to compute soft quality → soft gating + repair

Combined quality = 0.3 × image + 0.7 × cross_modal (training)
                 = 1.0 × image                      (inference, no MFMP)
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


def compute_cross_modal_quality(featR, featN):
    """
    Feature-level quality from MFMP-aligned R↔N features.
    
    After MFMP aligns R and N patches in feature space:
    - High R↔N patch similarity → both modalities clean
    - Low similarity → one is damaged by flare
    
    This mirrors original FACENet's FCE which used MFMP output:
      (featN_label > threshold) → detect damaged R patches
    
    Args:
        featR: (B, N, D) MFMP-aligned R features (CLS + patches)
        featN: (B, N, D) MFMP-aligned N features
    Returns:
        quality: (B,) in [0.15, 1.0], shared for both R and N
    """
    # Exclude CLS token, compare patch tokens
    patches_R = featR[:, 1:, :]  # (B, 196, 768)
    patches_N = featN[:, 1:, :]
    
    # Per-patch cosine similarity after MFMP alignment
    # After alignment, differences = damage, not content
    sim = F.cosine_similarity(patches_R, patches_N, dim=-1)  # (B, 196)
    mean_sim = sim.mean(dim=1)  # (B,) per-sample mean similarity
    
    # Similarity → quality
    # sim ~0.3 (damaged) to ~0.9 (clean)
    quality = (mean_sim - 0.2) / 0.7
    return 0.15 + 0.85 * quality.clamp(0.0, 1.0)


def compute_combined_quality(img_R, img_N, featR, featN, training):
    """
    Combine image-level and cross-modal quality.
    
    Training: 0.3 × image + 0.7 × cross_modal (MFMP available)
    Inference: 1.0 × image (MFMP off, fallback to pixels)
    
    Returns:
        q_R, q_N: (B,) quality scores in [0.15, 1.0]
    """
    q_img_R = compute_image_quality(img_R)
    q_img_N = compute_image_quality(img_N)
    
    if training and featR is not None and featN is not None:
        q_cross = compute_cross_modal_quality(featR, featN)  # (B,)
        q_R = 0.3 * q_img_R + 0.7 * q_cross
        q_N = 0.3 * q_img_N + 0.7 * q_cross
    else:
        q_R = q_img_R
        q_N = q_img_N
    
    return q_R, q_N


def compute_modality_quality(img_R, img_N, saturation_thresh=1.8):
    """Batch-mean quality (backward compat for processor)."""
    q_R = compute_image_quality(img_R, saturation_thresh).mean()
    q_N = compute_image_quality(img_N, saturation_thresh).mean()
    return q_R, q_N


def quality_gate(quality, floor=0.3):
    """Map quality to loss weight: [floor, 1.0]."""
    return floor + (1.0 - floor) * quality
