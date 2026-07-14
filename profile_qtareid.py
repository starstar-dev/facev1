import argparse
import os
import time

import torch
import torch.nn as nn

from config import cfg
from datasets import make_dataloader
from model.clip_facenet import CLIPFACENet
from utils.ablation import ABLATION_CONFIGS, apply_ablation_config, format_ablation_flags


class ForwardWrapper(nn.Module):
    def __init__(self, model, target, flare_label):
        super().__init__()
        self.model = model
        self.target = target
        self.flare_label = flare_label

    def forward(self, img_r, img_n, img_t):
        return self.model(img_r, img_n, img_t, self.target, flare_label=self.flare_label)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description="Profile QTA-ReID params, optional FLOPs, and FPS.")
    parser.add_argument("--exp", "--ablation", default="full", choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--config_file", default="configs/WMVeID863/clip_facenet_wmveid863.yml")
    parser.add_argument("--weight", default="", help="Optional checkpoint path.")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--warmup", default=30, type=int)
    parser.add_argument("--iters", default=100, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _, _, _, _, num_classes, _ = make_dataloader(cfg)
    model = CLIPFACENet(num_classes=num_classes, cfg=cfg)
    model.use_mamba_enhance = False
    model.use_fce = False
    exp_name, _ = apply_ablation_config(model, args.exp)

    if args.weight:
        model.load_param(args.weight)

    model.to(device).eval()
    total, trainable = count_params(model)
    print(f"Experiment: {exp_name}")
    print(f"Ablation flags: {format_ablation_flags(model)}")
    print(f"Params: total={total / 1e6:.2f}M, trainable={trainable / 1e6:.2f}M")

    h, w = cfg.INPUT.SIZE_TEST
    img_r = torch.randn(args.batch_size, 3, h, w, device=device)
    img_n = torch.randn(args.batch_size, 3, h, w, device=device)
    img_t = torch.randn(args.batch_size, 3, h, w, device=device)
    target = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    flare_label = torch.zeros(args.batch_size, dtype=torch.long, device=device)

    try:
        from thop import profile

        wrapper = ForwardWrapper(model, target, flare_label).to(device).eval()
        macs, params = profile(wrapper, inputs=(img_r, img_n, img_t), verbose=False)
        print(f"FLOPs: {2 * macs / 1e9:.2f}G (THOP convention: FLOPs=2*MACs)")
        print(f"THOP params: {params / 1e6:.2f}M")
    except Exception as exc:
        print(f"FLOPs: skipped ({exc.__class__.__name__}: {exc})")

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(img_r, img_n, img_t, target, flare_label=flare_label)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(args.iters):
            _ = model(img_r, img_n, img_t, target, flare_label=flare_label)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

    samples = args.batch_size * args.iters
    fps = samples / elapsed
    ms_per_sample = elapsed * 1000.0 / samples
    print(f"Inference speed: {fps:.2f} samples/s, {ms_per_sample:.2f} ms/sample")
    print(f"Batch setting: batch={args.batch_size}, warmup={args.warmup}, iters={args.iters}, input={h}x{w}")


if __name__ == "__main__":
    main()
