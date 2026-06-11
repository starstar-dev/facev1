"""
CoEN-Lite: Lightweight Corruption-aware quality gating.
Based on CoEN (Corruption-aware Embedding Network) but simplified:
- Uses image statistics (saturation) instead of learned quality predictor
- Per-batch modality-level gating (not per-sample, avoids loss function changes)
- No extra parameters, no extra forward passes
"""

import torch


def compute_modality_quality(img_R, img_N, saturation_thresh=1.8):
    """
    Compute batch-level quality scores for RGB and NIR modalities.
    
    For flare detection: saturated pixels (near-white due to overexposure)
    indicate corrupted regions. More saturation = lower quality.
    
    Args:
        img_R: (B, 3, H, W) CLIP-preprocessed RGB images
        img_N: (B, 3, H, W) CLIP-preprocessed NIR images  
        saturation_thresh: CLIP-norm value above which pixel is "saturated"
                           (1.8 ≈ original > 0.95 after CLIP normalization)
    Returns:
        q_R, q_N: scalar quality scores in [0.2, 1.0], higher = better
    """
    
    def _quality(x):
        B = x.size(0)
        # Fraction of saturated pixels per image
        saturated = (x > saturation_thresh).float().reshape(B, -1).mean(dim=1)
        # Quality: inverse of saturation
        q = 1.0 - saturated
        # Soft floor at 0.2: even fully saturated gets 20% weight
        return (0.2 + 0.8 * q.clamp(0.0, 1.0)).mean()
    
    q_R = _quality(img_R)
    q_N = _quality(img_N)
    return q_R, q_N


def quality_gate(quality, floor=0.3):
    """
    Map quality score to loss weight via soft gate.
    
    Args:
        quality: scalar in [0.2, 1.0]
        floor: minimum gate value (default 0.3)
    Returns:
        gate value in [floor, 1.0]
    """
    return floor + (1.0 - floor) * quality
