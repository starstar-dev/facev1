"""
CLIP-FACENet v3.1: v2 + Token Scorer + FCE.
Warm-started from v3 best checkpoint.
"""
from utils.logger import setup_logger
from datasets import make_dataloader
from model.clip_facenet_v3 import CLIPFACENetV3
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train_amp
import random, torch, numpy as np, os, argparse, datetime
from config import cfg


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLIP-FACENet v3.1 TokenScorer+FCE Training")
    parser.add_argument("--config_file", default="configs/WMVeID863/clip_facenet_wmveid863.yml", type=str)
    parser.add_argument("--warmstart", default="/root/autodl-tmp/FACENet/logs/clip_facenet_v3_tokenscorer_2026_06_11_09_27_40/clip_facenetbest.pth", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--gpu", default='0', type=int)
    parser.add_argument("--IC_param", default='0.8', type=float)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    now = datetime.datetime.now()
    strtime = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = cfg.OUTPUT_DIR + '_v31_fce_' + strtime
    os.makedirs(output_dir, exist_ok=True)

    cfg.SAVE_DIR = output_dir
    cfg.IC_param = args.IC_param
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    setup_logger('transreid', output_dir, 'train_log')
    
    import logging
    logger = logging.getLogger("transreid.image_train")
    logger.info(f'CLIP-FACENet v3.1 TokenScorer+FCE | Output: {output_dir}')
    logger.info(f'Warm-start from: {args.warmstart}')

    train_loader, train_loader_normal, val_loader, num_query, num_classes, num_cameras = make_dataloader(cfg)

    logger.info('Building CLIP-FACENet v3.1...')
    model = CLIPFACENetV3(num_classes=num_classes, cfg=cfg)
    
    if os.path.exists(args.warmstart):
        state_dict = torch.load(args.warmstart, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f'Warm-start loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}')
        if len(missing) > 0:
            logger.info(f'Missing: {missing[:5]}...')
    else:
        logger.warning(f'Warm-start checkpoint not found at {args.warmstart}, training from scratch!')

    logger.info(f'Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    do_train_amp(cfg, model, center_criterion, train_loader, val_loader,
                 optimizer, optimizer_center, scheduler, loss_func, num_query, args.local_rank)
