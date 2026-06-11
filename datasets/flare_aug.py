"""
Flare Augmentation for Robust Vehicle ReID.
Simulates strong flare effects on RGB and NIR images to force the model
to learn flare-invariant features. Thermal images are NOT augmented.

The SAME flare pattern is applied to both RGB and NIR (paired) because
in real scenarios, the same physical flare affects both cameras.
"""
import random
import math
import numpy as np
from PIL import Image


class FlareAugmentation:
    """Apply synthetic flare effects to paired RGB+NIR images.
    
    Flare parameters are pre-determined per call to ensure identical
    augmentation on both modalities.
    
    Args:
        prob: probability of applying ANY flare augmentation
        num_flares: (min, max) number of flare sources
        flare_intensity: (min, max) brightness in [0, 1]
        flare_size: (min, max) radius as fraction of image diagonal
    """
    def __init__(
        self,
        prob=0.6,
        num_flares=(1, 5),
        flare_intensity=(0.35, 0.9),
        flare_size=(0.04, 0.22),
    ):
        self.prob = prob
        self.num_flares = num_flares
        self.flare_intensity = flare_intensity
        self.flare_size = flare_size

    def _generate_flare_params(self, h, w):
        """Generate a list of flare parameters for one call.
        Returns None if no flare should be applied (prob check failed).
        """
        if random.random() > self.prob:
            return None
        
        n = random.randint(*self.num_flares)
        diag = math.sqrt(h**2 + w**2)
        
        flares = []
        for _ in range(n):
            flares.append({
                'cx': random.randint(0, w - 1),
                'cy': random.randint(0, h - 1),
                'radius': max(5, int(random.uniform(*self.flare_size) * diag)),
                'intensity': random.uniform(*self.flare_intensity),
                'has_bloom': random.random() < 0.4,
                'n_streaks': random.randint(3, 8),
            })
        return flares

    def _apply_flares(self, img_np, flares):
        """Apply pre-determined flares to a numpy image."""
        h, w = img_np.shape[:2]
        
        for f in flares:
            cx, cy, radius, intensity = f['cx'], f['cy'], f['radius'], f['intensity']
            
            # --- Circular overexposure ---
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            sigma = radius / 2.5
            gaussian = np.exp(-0.5 * (dist / sigma)**2)
            flare = gaussian * intensity * 255
            
            for c in range(min(img_np.shape[2], 3)):
                img_np[:, :, c] = np.clip(img_np[:, :, c] + flare, 0, 255)
            
            # --- Local saturation boost ---
            # Pixels within the core of the flare get extra saturation push
            core_mask = dist < (radius * 0.6)
            if core_mask.any():
                for c in range(min(img_np.shape[2], 3)):
                    img_np[core_mask, c] = np.clip(
                        img_np[core_mask, c] + intensity * 100, 0, 255
                    )
            
            # --- Bloom streaks ---
            if f['has_bloom']:
                streak_len = radius * random.uniform(2.0, 4.0)
                n_streaks = f['n_streaks']
                
                for _ in range(n_streaks):
                    angle = random.uniform(0, 2 * math.pi)
                    length = random.uniform(radius, streak_len)
                    sw = max(1, int(radius * random.uniform(0.03, 0.12)))
                    
                    ex = int(cx + length * math.cos(angle))
                    ey = int(cy + length * math.sin(angle))
                    
                    for t in np.linspace(0, 1, max(int(length), 10)):
                        px = int(cx + t * (ex - cx))
                        py = int(cy + t * (ey - cy))
                        falloff = (1 - t) * intensity * 0.5
                        
                        if 0 <= px < w and 0 <= py < h:
                            y0, y1 = max(0, py-sw), min(h, py+sw+1)
                            x0, x1 = max(0, px-sw), min(w, px+sw+1)
                            for c in range(min(img_np.shape[2], 3)):
                                img_np[y0:y1, x0:x1, c] = np.clip(
                                    img_np[y0:y1, x0:x1, c] + falloff * 255, 0, 255
                                )

    def __call__(self, img_rgb, img_nir):
        """Apply paired flare augmentation to RGB and NIR PIL images.
        Returns (augmented_rgb, augmented_nir) — same flares on both.
        """
        img_rgb_np = np.array(img_rgb).astype(np.float32)
        h, w = img_rgb_np.shape[:2]
        
        # Generate flare parameters ONCE
        flares = self._generate_flare_params(h, w)
        
        if flares is None:
            return img_rgb, img_nir
        
        # Apply same flares to RGB
        self._apply_flares(img_rgb_np, flares)
        img_rgb_out = Image.fromarray(np.clip(img_rgb_np, 0, 255).astype(np.uint8))
        
        # Apply same flares to NIR
        img_nir_np = np.array(img_nir).astype(np.float32)
        if len(img_nir_np.shape) == 2:  # grayscale
            img_nir_np = np.stack([img_nir_np]*3, axis=-1)
        self._apply_flares(img_nir_np, flares)
        img_nir_out = Image.fromarray(np.clip(img_nir_np, 0, 255).astype(np.uint8))
        
        return img_rgb_out, img_nir_out
