"""Generate the final CoEN low-quality-map figure from a fixed checkpoint.

Unlike the training-time preview in processor/processor_amp.py, this script
runs the frozen model in eval mode on the deterministic validation loader.
It therefore produces a reproducible figure suitable for the paper.
"""

import argparse
import heapq
import json
import os
import sys

import torch


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import cfg
from datasets import make_dataloader
from model.clip_facenet import CLIPFACENet
from utils.ablation import ABLATION_CONFIGS, apply_ablation_config
from utils.visualize_coen import save_paper_qmap_grid


def _visibility_score(img):
    """Estimate whether the vehicle remains visually interpretable.

    This score is used only to select representative paper examples. The
    model output and reported metrics are untouched. A central crop is used
    because the vehicle is centered in the ReID crops.
    """
    mean = img.new_tensor(CLIP_MEAN).view(3, 1, 1)
    std = img.new_tensor(CLIP_STD).view(3, 1, 1)
    x = (img.detach() * std + mean).clamp(0.0, 1.0)
    gray = x.mean(dim=0)

    height, width = gray.shape
    y0, y1 = height // 10, height - height // 10
    x0, x1 = width // 10, width - width // 10
    center = gray[y0:y1, x0:x1]

    clipped = (center > 0.97).float().mean()
    crushed = (center < 0.03).float().mean()
    contrast = (center.std(unbiased=False) / 0.22).clamp(0.0, 1.0)
    dx = torch.abs(center[:, 1:] - center[:, :-1]).mean()
    dy = torch.abs(center[1:, :] - center[:-1, :]).mean()
    detail = ((dx + dy) / 0.12).clamp(0.0, 1.0)

    visible_range = (1.0 - clipped) * (1.0 - crushed)
    score = visible_range * (0.55 * contrast + 0.45 * detail)
    return float(score.item())


def _make_record(
    raw_model,
    sample_index,
    batch_index,
    sample_in_batch,
    paths,
    pid,
    score,
):
    """Copy one sample to CPU before the next inference batch overwrites it."""
    visibility_rgb = _visibility_score(raw_model._last_img_R[sample_in_batch])
    visibility_ni = _visibility_score(raw_model._last_img_N[sample_in_batch])
    return {
        "sample_index": int(sample_index),
        "batch_index": int(batch_index),
        "sample_in_batch": int(sample_in_batch),
        "pid": int(pid),
        "score": float(score),
        "visibility_rgb": visibility_rgb,
        "visibility_ni": visibility_ni,
        "visibility": 0.5 * (visibility_rgb + visibility_ni),
        "paths": [str(path) for path in paths],
        "img_R": raw_model._last_img_R[sample_in_batch].detach().cpu().clone(),
        "img_N": raw_model._last_img_N[sample_in_batch].detach().cpu().clone(),
        "q_R": raw_model._last_q_R_map[sample_in_batch].detach().cpu().clone(),
        "q_N": raw_model._last_q_N_map[sample_in_batch].detach().cpu().clone(),
    }


def _select_representative_records(pool, num_samples):
    """Choose clear, diverse examples from a strongly degraded candidate pool."""
    records = [item[2] for item in pool]
    if not records:
        return []

    score_min = min(record["score"] for record in records)
    score_max = max(record["score"] for record in records)
    score_range = max(score_max - score_min, 1e-12)

    for record in records:
        degradation = (record["score"] - score_min) / score_range
        # All records already belong to the high-degradation pool. Visibility
        # therefore receives more weight so fully saturated crops do not win.
        record["paper_score"] = 0.70 * record["visibility"] + 0.30 * degradation

    ranked = sorted(records, key=lambda record: record["paper_score"], reverse=True)
    selected = []
    selected_indices = set()
    used_pids = set()

    # Prefer different vehicle identities so near-duplicate query/gallery
    # frames do not consume several rows in the final figure.
    for record in ranked:
        if record["pid"] in used_pids:
            continue
        selected.append(record)
        selected_indices.add(record["sample_index"])
        used_pids.add(record["pid"])
        if len(selected) == num_samples:
            return selected

    # Small validation subsets may not contain enough distinct identities.
    for record in ranked:
        if record["sample_index"] in selected_indices:
            continue
        selected.append(record)
        if len(selected) == num_samples:
            break

    return selected


def collect_paper_samples(
    model,
    val_loader,
    device,
    num_samples,
    selection="representative",
    sample_indices=None,
    candidate_pool=80,
):
    """Collect fixed, worst-case, or clear representative validation samples."""
    wanted = None if sample_indices is None else set(sample_indices)
    fixed_records = {}
    top_records = []
    pool_size = num_samples if selection == "topk" else max(candidate_pool, num_samples)
    global_index = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            (
                img_R,
                img_N,
                img_T,
                pids,
                _,
                _,
                _,
                img_paths,
                flare_label,
            ) = batch

            img_R = img_R.to(device, non_blocking=True)
            img_N = img_N.to(device, non_blocking=True)
            img_T = img_T.to(device, non_blocking=True)
            pids = pids.to(device, non_blocking=True)
            flare_label = flare_label.to(device, non_blocking=True)

            model(
                img_R,
                img_N,
                img_T,
                pids,
                flare_label=flare_label,
            )

            raw_model = model.module if hasattr(model, "module") else model
            if not hasattr(raw_model, "_last_q_R_map"):
                raise RuntimeError(
                    "The selected experiment does not expose CoEN quality maps. "
                    "Use --exp full/original_G or another CoEN-enabled setting."
                )

            bad_R = 1.0 - raw_model._last_q_R_map
            bad_N = 1.0 - raw_model._last_q_N_map
            reduce_dims_R = tuple(range(1, bad_R.dim()))
            reduce_dims_N = tuple(range(1, bad_N.dim()))
            scores = bad_R.mean(dim=reduce_dims_R) + bad_N.mean(dim=reduce_dims_N)

            for sample_in_batch in range(img_R.size(0)):
                sample_index = global_index + sample_in_batch
                score = scores[sample_in_batch].item()

                if wanted is not None and sample_index not in wanted:
                    continue

                # Avoid copying four tensors to CPU when the sample cannot enter
                # the current top-k heap.
                if (
                    wanted is None
                    and len(top_records) >= pool_size
                    and score <= top_records[0][0]
                ):
                    continue

                record = _make_record(
                    raw_model=raw_model,
                    sample_index=sample_index,
                    batch_index=batch_index,
                    sample_in_batch=sample_in_batch,
                    paths=img_paths[sample_in_batch],
                    pid=pids[sample_in_batch].item(),
                    score=score,
                )

                if wanted is not None:
                    fixed_records[sample_index] = record
                elif len(top_records) < pool_size:
                    heapq.heappush(
                        top_records,
                        (record["score"], record["sample_index"], record),
                    )
                elif record["score"] > top_records[0][0]:
                    heapq.heapreplace(
                        top_records,
                        (record["score"], record["sample_index"], record),
                    )

            global_index += img_R.size(0)

            if wanted is not None and wanted.issubset(fixed_records):
                break

    if wanted is not None:
        missing = sorted(wanted.difference(fixed_records))
        if missing:
            raise IndexError(f"Validation sample indices not found: {missing}")
        return [fixed_records[index] for index in sample_indices]

    if selection == "representative":
        return _select_representative_records(top_records, num_samples)

    return [item[2] for item in sorted(top_records, key=lambda x: x[0], reverse=True)]


def save_records(records, save_path, selection):
    """Render the grid and save the exact sample provenance beside it."""
    if not records:
        raise RuntimeError("No validation samples were collected")

    img_R = torch.stack([record["img_R"] for record in records])
    img_N = torch.stack([record["img_N"] for record in records])
    q_R = torch.stack([record["q_R"] for record in records])
    q_N = torch.stack([record["q_N"] for record in records])

    pdf_path, png_path = save_paper_qmap_grid(
        img_R=img_R,
        img_N=img_N,
        q_R=q_R,
        q_N=q_N,
        indices=list(range(len(records))),
        save_path=save_path,
    )

    stem, _ = os.path.splitext(save_path)
    metadata_path = f"{stem}.json"
    metadata = {
        "selection": selection,
        "samples": [
            {
                "sample_index": record["sample_index"],
                "batch_index": record["batch_index"],
                "sample_in_batch": record["sample_in_batch"],
                "pid": record["pid"],
                "low_quality_score_sum": record["score"],
                "visibility_rgb": record["visibility_rgb"],
                "visibility_ni": record["visibility_ni"],
                "visibility_mean": record["visibility"],
                "paper_selection_score": record.get("paper_score"),
                "paths": record["paths"],
            }
            for record in records
        ],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return pdf_path, png_path, metadata_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a reproducible, paper-ready CoEN quality-map grid."
    )
    parser.add_argument(
        "--config_file",
        default="configs/WMVeID863/clip_facenet_wmveid863.yml",
    )
    parser.add_argument(
        "--weight",
        required=True,
        help="Best/final model checkpoint used for the reported experiment.",
    )
    parser.add_argument(
        "--exp",
        default="original_G",
        choices=list(ABLATION_CONFIGS.keys()),
    )
    parser.add_argument(
        "--selection",
        choices=["representative", "topk", "indices"],
        default="representative",
        help=(
            "representative: select clear, identity-diverse examples from a "
            "high-degradation candidate pool; topk: select the absolute "
            "worst cases; indices: use fixed --indices."
        ),
    )
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument(
        "--candidate_pool",
        type=int,
        default=80,
        help=(
            "Number of highest-degradation samples considered by "
            "--selection representative."
        ),
    )
    parser.add_argument(
        "--save_path",
        default="./vis/paper/coen_low_quality_maps.png",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.selection == "indices" and not args.indices:
        raise ValueError("--selection indices requires one or more --indices")
    if args.num_samples < 1:
        raise ValueError("--num_samples must be at least 1")
    if args.candidate_pool < args.num_samples:
        raise ValueError("--candidate_pool must be at least --num_samples")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    _, _, val_loader, _, num_classes, _ = make_dataloader(cfg)
    model = CLIPFACENet(num_classes=num_classes, cfg=cfg)
    apply_ablation_config(model, args.exp)
    model.load_param(args.weight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    sample_indices = args.indices if args.selection == "indices" else None
    records = collect_paper_samples(
        model=model,
        val_loader=val_loader,
        device=device,
        num_samples=args.num_samples,
        selection=args.selection,
        sample_indices=sample_indices,
        candidate_pool=args.candidate_pool,
    )
    outputs = save_records(records, args.save_path, args.selection)

    print("Selected validation indices:", [r["sample_index"] for r in records])
    print("Saved paper figure:")
    for path in outputs:
        print(" ", path)


if __name__ == "__main__":
    main()
