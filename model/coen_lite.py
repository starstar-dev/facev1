"""
CoEN-Lite: Lightweight Corruption-aware quality gating + feature repair.

Two mechanisms:
1. Per-sample quality scoring from image statistics (saturation + texture)
2. Quality-aware CrossModalFusion repair (poor R/N → borrow more from TI)

No extra parameters, no extra forward passes.
"""

import torch


def compute_per_sample_quality(img, saturation_thresh=1.8):
    """
    Compute per-sample image quality for flare detection.
    
    Uses two complementary signals:
    1. Saturation ratio: fraction of pixels near white (flare signature)
       - More saturation = more flare = lower quality
    2. Local texture variance: patch-wise standard deviation
       - Less texture = washed out by flare = lower quality
    
    Args:
        img: (B, 3, H, W) CLIP-preprocessed images
        saturation_thresh: CLIP-norm value above which pixel is "saturated"
    Returns:
        quality: (B,) scores in [0.15, 1.0], higher = better quality
    """
    B, C, H, W = img.shape
    
    # Signal 1: Saturation ratio
    saturated = (img > saturation_thresh).float().reshape(B, -1).mean(dim=1)
    
    # Signal 2: Texture via local gradient magnitude
    # High gradient = edges/texture present = good
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    grad_mag = (dy[:, :, :, :-1] + dx[:, :, :-1, :]).reshape(B, -1).mean(dim=1)
    # Normalize: typical grad_mag ranges 0.1-0.8
    texture = (grad_mag - 0.1) / 0.7
    
    # Combine: high texture + low saturation = good
    raw_q = (1.0 - 0.6 * saturated) * (0.2 + 0.8 * texture.clamp(0.0, 1.0))
    
    # Soft floor: even worst images get 15%
    quality = 0.15 + 0.85 * raw_q.clamp(0.0, 1.0)
    
    return quality


def compute_modality_quality(img_R, img_N, saturation_thresh=1.8):
    """
    Compute batch-mean quality scores (for backward compatibility with processor).
    
    Returns:
        q_R, q_N: scalar quality scores in [0.2, 1.0]
    """
    q_R = compute_per_sample_quality(img_R, saturation_thresh).mean()
    q_N = compute_per_sample_quality(img_N, saturation_thresh).mean()
    return q_R, q_N


def quality_gate(quality, floor=0.3):
    """
    Map quality score to loss weight via soft gate.
    
    Args:
        quality: scalar in [0.15, 1.0]
        floor: minimum gate value (default 0.3)
    Returns:
        gate value in [floor, 1.0]
    """
    return floor + (1.0 - floor) * quality
