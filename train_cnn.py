"""
Training script for QTA-ReID with CNN backbone.
Usage:
    python train_cnn.py --backbone resnet50 --config_file configs/WMVeID863/clip_facenet_wmveid863.yml --epochs 150
"""
import os, sys, argparse, datetime, random, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from config import cfg
from datasets import make_dataloader
from model.qta_reid_cnn import QTAReIDCNN
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train_amp
from utils.logger import setup_logger


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50', choices=['resnet50', 'osnet'])
    parser.add_argument('--config_file', default='configs/WMVeID863/clip_facenet_wmveid863.yml')
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.MAX_EPOCHS = args.epochs
    
    # Adjust config for CNN backbone
    cfg.INPUT.SIZE_TRAIN = [224, 224]
    cfg.INPUT.SIZE_TEST = [224, 224]
    # CNN needs ImageNet normalization, not CLIP
    cfg.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
    cfg.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
    
    now = datetime.datetime.now()
    strtime = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = f"{cfg.OUTPUT_DIR}_cnn_{args.backbone}_{strtime}"
    os.makedirs(output_dir, exist_ok=True)
    cfg.SAVE_DIR = output_dir
    cfg.freeze()
    
    set_seed(cfg.SOLVER.SEED)
    setup_logger("transreid", output_dir, "train_log")
    
    train_loader, train_loader_normal, val_loader, num_query, num_classes, _ = \
        make_dataloader(cfg, use_flare_sampler=True, modality_dropout=0.0)
    
    model = QTAReIDCNN(num_classes=num_classes, backbone_name=args.backbone)
    
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    
    do_train_amp(cfg, model, center_criterion, train_loader, val_loader,
                 optimizer, optimizer_center, scheduler, loss_func, num_query, 0)