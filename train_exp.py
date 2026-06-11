"""
Unified experiment runner for v2 grid: A (baseline) through E (flare+dropout).
"""
from utils.logger import setup_logger
from datasets import make_dataloader
from model.clip_facenet import CLIPFACENet
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


EXP_CONFIGS = {
    'A': {'flare_sampler': False, 'dropout': 0.0, 'desc': 'v2 baseline'},
    'B': {'flare_sampler': True,  'dropout': 0.0, 'desc': 'v2 + flare-balanced sampler'},
    'C': {'flare_sampler': False, 'dropout': 0.1, 'desc': 'v2 + modality dropout p=0.1'},
    'D': {'flare_sampler': False, 'dropout': 0.2, 'desc': 'v2 + modality dropout p=0.2'},
    'E': {'flare_sampler': True,  'dropout': 0.1, 'desc': 'v2 + flare sampler + dropout p=0.1'},
    'F': {'flare_sampler': False, 'dropout': 0.0, 'coen_lite': True,  'desc': 'v2 + CoEN-lite quality gate'},
    'G': {'flare_sampler': True,  'dropout': 0.0, 'coen_lite': True,  'desc': 'v2 + CoEN-lite + flare sampler'},
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="v2 Experiment Grid")
    parser.add_argument("--exp", default="A", choices=['A','B','C','D','E','F','G'])
    parser.add_argument("--config_file", default="configs/WMVeID863/clip_facenet_wmveid863.yml", type=str)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--IC_param", default='0.8', type=float)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    exp_cfg = EXP_CONFIGS[args.exp]
    use_flare = exp_cfg['flare_sampler']
    mdrop = exp_cfg['dropout']
    use_coen = exp_cfg.get('coen_lite', False)

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.MAX_EPOCHS = args.epochs
    cfg.IC_param = args.IC_param

    now = datetime.datetime.now()
    strtime = now.strftime('%Y_%m_%d_%H_%M_%S')
    label = f'exp{args.exp}'
    if use_flare: label += '_flare'
    if mdrop > 0: label += f'_md{int(mdrop*100)}'
    if use_coen: label += '_coenlite'
    output_dir = cfg.OUTPUT_DIR + f'_grid_{label}_' + strtime
    os.makedirs(output_dir, exist_ok=True)
    cfg.SAVE_DIR = output_dir
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    setup_logger('transreid', output_dir, 'train_log')
    
    import logging
    logger = logging.getLogger("transreid.image_train")
    logger.info(f'Experiment {args.exp}: {exp_cfg["desc"]}')
    logger.info(f'Flare sampler: {use_flare}, Modality dropout: {mdrop}, CoEN-lite: {use_coen}')

    train_loader, train_loader_normal, val_loader, num_query, num_classes, num_cameras = \
        make_dataloader(cfg, use_flare_sampler=use_flare, modality_dropout=mdrop)

    model = CLIPFACENet(num_classes=num_classes, cfg=cfg)
    model.use_mamba_enhance = False
    model.use_mcloss = True
    model.use_fce = False
    model.use_mfmp = True
    model.use_coen_lite = use_coen

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    do_train_amp(cfg, model, center_criterion, train_loader, val_loader,
                 optimizer, optimizer_center, scheduler, loss_func, num_query, args.local_rank)
