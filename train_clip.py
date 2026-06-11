"""
CLIP-FACENet v2 Training Script.
Supports MFMP and MC Loss.
"""
from utils.logger import setup_logger
from datasets import make_dataloader
from model.clip_facenet import CLIPFACENet
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train_amp
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import datetime


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLIP-FACENet v2 Training")
    parser.add_argument("--config_file", default="configs/WMVeID863/clip_facenet_wmveid863.yml",
                        help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--gpu", default='0', type=int)
    parser.add_argument("--IC_param", default='0.8', type=float)
    parser.add_argument("--use_mfmp", default=1, type=int, help="Enable MFMP")
    parser.add_argument("--use_mcloss", default=1, type=int, help="Enable MC Loss")
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    now = datetime.datetime.now()
    strtime = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = cfg.OUTPUT_DIR + '_v2_mfmp_mc_' + strtime
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cfg.SAVE_DIR = output_dir
    cfg.IC_param = args.IC_param
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    setup_logger('transreid', output_dir, 'train_log')
    
    import logging
    logger = logging.getLogger("transreid.image_train")
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'CLIP-FACENet v2: MFMP={bool(args.use_mfmp)}, MCLOSS={bool(args.use_mcloss)}')

    train_loader, train_loader_normal, val_loader, num_query, num_classes, num_cameras = make_dataloader(cfg)

    logger.info('Building CLIP-FACENet v2...')
    model = CLIPFACENet(num_classes=num_classes, cfg=cfg)
    model.use_mamba_enhance = False
    model.use_mcloss = bool(args.use_mcloss)
    model.use_fce = False
    model.use_mfmp = bool(args.use_mfmp)
    logger.info(f'MFMP enabled: {model.use_mfmp}')
    logger.info(f'MC Loss enabled: {model.use_mcloss}')
    logger.info(f'Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    do_train_amp(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query,
        args.local_rank,
    )
