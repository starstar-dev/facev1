"""
Hyperparameter sensitivity sweep for QTA-ReID.
Sweeps peer_margin and ti_strength, records mAP/R1 at each point.
Runs inference only (no retraining) on a trained checkpoint.
"""
import os, sys, argparse, json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import cfg
from datasets import make_dataloader
from model.clip_facenet import CLIPFACENet
from utils.ablation import ABLATION_CONFIGS, apply_ablation_config
from utils.metrics import R1_mAP_eval
from utils.logger import setup_logger


def evaluate(model, val_loader, num_query, device='cuda'):
    """Run inference and return (mAP, R1, R5, R10)."""
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm='yes', cfg=cfg)
    evaluator.reset()
    model.to(device)
    model.eval()
    for n_iter, (img1, img2, img3, vid, camid, camids, viewids, img_paths, flare_label) in enumerate(val_loader):
        with torch.no_grad():
            img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)
            camids = camids.to(device)
            target = vid.to(device)
            flare_label = flare_label.to(device)
            feat = model(img1, img2, img3, target, flare_label=flare_label)
            evaluator.update((feat, vid, camid, img_paths))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    return mAP, cmc[0], cmc[4], cmc[9]


def sweep_param(model, val_loader, num_query, param_name, values):
    """Sweep a single parameter, return list of (value, mAP, R1)."""
    results = []
    for v in values:
        # Set parameter on both fusion modules
        setattr(model.fusion_R, param_name, v)
        setattr(model.fusion_N, param_name, v)
        mAP, r1, r5, r10 = evaluate(model, val_loader, num_query)
        results.append({'value': v, 'mAP': mAP, 'R1': r1, 'R5': r5, 'R10': r10})
        print(f"  {param_name}={v:.3f}  mAP={mAP:.4f}  R1={r1:.4f}")
    return results


def plot_sensitivity(results, param_name, output_path):
    """Plot sensitivity curve."""
    values = [r['value'] for r in results]
    maps = [r['mAP'] for r in results]
    r1s = [r['R1'] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(values, maps, 'b-o', linewidth=2, markersize=8, label='mAP')
    ax2.plot(values, r1s, 'r-s', linewidth=2, markersize=8, label='Rank-1')
    ax1.set_xlabel(param_name, fontsize=14)
    ax1.set_ylabel('mAP', color='b', fontsize=14)
    ax2.set_ylabel('Rank-1', color='r', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')

    # Mark default value
    default = {'peer_margin': 0.03, 'ti_strength': 1.0}[param_name]
    if default in values:
        idx = values.index(default)
        ax1.axvline(x=default, color='gray', linestyle='--', alpha=0.5, label=f'Default={default}')
        ax1.plot(default, maps[idx], 'b*', markersize=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title(f'Hyperparameter Sensitivity: {param_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/WMVeID863/clip_facenet_wmveid863.yml')
    parser.add_argument('--weight', type=str, required=True, help='Path to trained QTA-ReID checkpoint')
    parser.add_argument('--exp', default='full')
    parser.add_argument('--output_dir', default='./sensitivity_results')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    train_loader, _, val_loader, num_query, num_classes, _ = make_dataloader(cfg)

    # Load model
    model = CLIPFACENet(num_classes=num_classes, cfg=cfg)
    model.use_mamba_enhance = False
    model.use_fce = False
    apply_ablation_config(model, args.exp)
    model.load_param(args.weight)

    # Sweep peer_margin
    print("\n" + "="*60)
    print("Sweeping peer_margin (TI compensation threshold)")
    print("="*60)
    peer_margin_values = [0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.15]
    results_margin = sweep_param(model, val_loader, num_query, 'peer_margin', peer_margin_values)
    plot_sensitivity(results_margin, 'peer_margin',
                     os.path.join(args.output_dir, 'sensitivity_peer_margin.png'))

    # Reset to default
    model.fusion_R.peer_margin = 0.03
    model.fusion_N.peer_margin = 0.03

    # Sweep ti_strength
    print("\n" + "="*60)
    print("Sweeping ti_strength (TI anchor compensation strength)")
    print("="*60)
    ti_strength_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    results_ti = sweep_param(model, val_loader, num_query, 'ti_strength', ti_strength_values)
    plot_sensitivity(results_ti, 'ti_strength',
                     os.path.join(args.output_dir, 'sensitivity_ti_strength.png'))

    # Save JSON
    all_results = {
        'peer_margin': results_margin,
        'ti_strength': results_ti,
    }
    with open(os.path.join(args.output_dir, 'sensitivity_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {args.output_dir}")