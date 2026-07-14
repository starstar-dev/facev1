import argparse
import time
import torch
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from config import cfg
from model.clip_facenet import CLIPFACENet
from utils.ablation import apply_ablation_config


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_speed(model, batch_size=1, size=224, warmup=20, repeat=100):
    model.eval().cuda()

    x1 = torch.randn(batch_size, 3, size, size).cuda()
    x2 = torch.randn(batch_size, 3, size, size).cuda()
    x3 = torch.randn(batch_size, 3, size, size).cuda()

    for _ in range(warmup):
        _ = model(x1, x2, x3)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(repeat):
        _ = model(x1, x2, x3)

    torch.cuda.synchronize()
    total = time.time() - start

    ms_per_batch = total / repeat * 1000
    ms_per_img = ms_per_batch / batch_size

    return ms_per_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="configs/WMVeID863/clip_facenet_wmveid863.yml")
    parser.add_argument("--exp", default="original_G")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # num_classes 只影响分类头参数量。可以填 WMVeID863 train ids=603。
    model = CLIPFACENet(num_classes=603, cfg=cfg)
    apply_ablation_config(model, args.exp)

    total, trainable = count_params(model)

    print(f"Params total: {total / 1e6:.2f} M")
    print(f"Params trainable: {trainable / 1e6:.2f} M")

    ms = measure_speed(model, batch_size=args.batch_size, size=224)
    print(f"Inference time: {ms:.2f} ms/image")


if __name__ == "__main__":
    main()