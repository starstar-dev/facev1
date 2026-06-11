"""Pre-compute per-ID flare ratios for the training set."""
import os
import numpy as np
from PIL import Image
from collections import defaultdict
from torchvision.transforms.functional import to_tensor
import torch


def compute_flare_ratio_per_id(dataset_root, train_ids=None):
    """Compute per-ID flare ratio by scanning training images.
    Uses the same histogram method as get_exposure_fake_label.
    
    Returns:
        pid_flare_ratio: dict pid -> flare_ratio (0.0 to 1.0)
    """
    train_dir = os.path.join(dataset_root, 'train')
    vids = os.listdir(train_dir)
    
    pid_flare = defaultdict(lambda: [0, 0])  # [flare_count, total]
    
    for vid in vids:
        pid = int(vid)
        vis_dir = os.path.join(train_dir, vid, 'vis')
        
        if not os.path.isdir(vis_dir):
            continue
        
        for img_name in os.listdir(vis_dir):
            # Load RGB and NIR images
            vpath = os.path.join(vis_dir, img_name)
            npath = os.path.join(train_dir, vid, 'ni', img_name)
            
            if not os.path.exists(npath):
                continue
            
            try:
                r_img = Image.open(vpath)
                n_img = Image.open(npath)
                
                r_tensor = to_tensor(r_img)
                n_tensor = to_tensor(n_img)
                
                r_hist = torch.histc(r_tensor, bins=100, min=0, max=1)
                n_hist = torch.histc(n_tensor, bins=100, min=0, max=1)
                
                ratio_r = torch.sum(r_hist[95:]) / torch.sum(r_hist)
                ratio_n = torch.sum(n_hist[95:]) / torch.sum(n_hist)
                
                is_flare = (ratio_r >= 0.095 and ratio_n >= 0.095)
                pid_flare[pid][1 if is_flare else 0] += 1
                
            except Exception:
                continue
    
    # Compute ratios
    pid_flare_ratio = {}
    for pid, (non_flare, flare) in pid_flare.items():
        total = non_flare + flare
        pid_flare_ratio[pid] = flare / total if total > 0 else 0.0
    
    return pid_flare_ratio
