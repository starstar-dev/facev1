"""
E4: Hard/Normal Subset Split for Query Set.

Supports 3 dataset formats:
  - WMVeID863:  ((vis, ni, th), pid, cam, trackid)
  - MSVR310:    ((vis, ni, th), vid, camid, sceneid)
  - RGBNT100:   (concat_img, pid, camid, trackid)  — 768×128 concat, RGB=first 256px

Usage:
  python scripts/split_hard_normal.py --dataset WMVeID863 --data_root .../WMVEID863/
  python scripts/split_hard_normal.py --dataset RGBNT100  --data_root .../data/
  python scripts/split_hard_normal.py --dataset MSVR310   --data_root .../data/
"""

import os, sys, argparse
import numpy as np
import cv2
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ====== Detection helpers ======

def compute_overexposure_ratio(gray_img, top_percentile=98, threshold_ratio=0.095):
    gray = gray_img.astype(np.float32)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    top_count = hist[int(256 * top_percentile / 100):].sum()
    ratio = top_count / total if total > 0 else 0.0
    return ratio > threshold_ratio, ratio


def compute_laplacian_var(gray_img):
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()


def is_hard(rgb_img, overexposure_area_threshold=0.15, laplacian_threshold=50):
    reasons = []
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    _, oe_ratio = compute_overexposure_ratio(gray)
    overexposure_area = (gray > 250).mean()
    if overexposure_area > overexposure_area_threshold:
        reasons.append(f"overexposure({overexposure_area:.2f})")

    lap_var = compute_laplacian_var(gray)
    if lap_var < laplacian_threshold:
        reasons.append(f"low_texture({lap_var:.1f})")

    label = "hard" if len(reasons) > 0 else "normal"
    metrics = {
        "oe_ratio": float(oe_ratio),
        "overexposure_area": float(overexposure_area),
        "laplacian_var": float(lap_var),
    }
    return label, reasons, metrics


# ====== Format detection ======

def detect_format(first_entry):
    """Return 'three_paths' or 'concat'."""
    if isinstance(first_entry[0], (tuple, list)) and len(first_entry[0]) == 3:
        return "three_paths"
    return "concat"


def read_rgb(query_entry, fmt):
    """Extract RGB image (BGR numpy array) from a query entry."""
    if fmt == "three_paths":
        # ((vis, ni, th), pid, cam, ...)  — vis = RGB
        vpath = query_entry[0][0]
        img = cv2.imread(vpath)
    else:
        # RGBNT100: (concat_path, pid, cam, trackid)
        concat_path = query_entry[0]
        img = cv2.imread(concat_path)
        if img is not None:
            strip_w = img.shape[1] // 3
            img = img[:, :strip_w]  # first strip = RGB (256×128)
    return img


# ====== Main ======

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="WMVeID863",
                        choices=["WMVeID863", "RGBNT100", "MSVR310"])
    parser.add_argument("--data_root", default="/root/autodl-tmp/projects/data/")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--overexposure_area", type=float, default=0.15)
    parser.add_argument("--laplacian_threshold", type=float, default=50)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"./hard_normal_splits/{args.dataset}/"
    os.makedirs(args.output_dir, exist_ok=True)

    # Import dataset
    if args.dataset == "WMVeID863":
        from datasets.WMVEID863 import WMVEID863 as DS
    elif args.dataset == "RGBNT100":
        from datasets.RGBNT100 import RGBNT100 as DS
    elif args.dataset == "MSVR310":
        from datasets.msvr310 import MSVR310 as DS
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    ds = DS(root=args.data_root, print_info=False)
    queries = ds.query
    fmt = detect_format(queries[0])

    print(f"Dataset: {args.dataset}  Format: {fmt}")
    print(f"Total queries: {len(queries)}")
    print(f"Thresholds: overexposure_area>{args.overexposure_area}, "
          f"laplacian_var<{args.laplacian_threshold}\n")

    hard_indices, normal_indices = [], []
    reason_counter = Counter()
    all_metrics = {"oe_ratio": [], "overexposure_area": [], "laplacian_var": []}

    for idx, entry in enumerate(queries):
        rgb_bgr = read_rgb(entry, fmt)
        if rgb_bgr is None:
            normal_indices.append(idx)
            continue
        rgb_img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        label, reasons, metrics = is_hard(
            rgb_img,
            overexposure_area_threshold=args.overexposure_area,
            laplacian_threshold=args.laplacian_threshold,
        )

        for r in reasons:
            reason_counter[r.split("(")[0]] += 1
        for k in all_metrics:
            all_metrics[k].append(metrics[k])

        (hard_indices if label == "hard" else normal_indices).append(idx)

    # Write
    for subset, indices in [("hard", hard_indices), ("normal", normal_indices)]:
        path = os.path.join(args.output_dir, f"{subset}_queries.txt")
        with open(path, "w") as f:
            for i in indices:
                f.write(f"{i}\n")

    # Report
    total = len(queries)
    print(f"=== Split Results ===")
    print(f"Hard:   {len(hard_indices)} ({100*len(hard_indices)/total:.1f}%)")
    print(f"Normal: {len(normal_indices)} ({100*len(normal_indices)/total:.1f}%)")
    print(f"\nHard reasons: {dict(reason_counter)}")
    for k, vals in all_metrics.items():
        arr = np.array(vals)
        print(f"  {k}: mean={arr.mean():.4f} std={arr.std():.4f} "
              f"min={arr.min():.4f} max={arr.max():.4f}")
    print(f"\nSaved: {args.output_dir}")


if __name__ == "__main__":
    main()