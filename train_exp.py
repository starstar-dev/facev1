"""
Removal ablation runner for CLIP-FACENet.
Use --exp/--ablation names matching the paper table.
"""
from utils.logger import setup_logger
from utils.ablation import ABLATION_CONFIGS, apply_ablation_config, format_ablation_flags
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
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP-FACENet removal ablation runner")
    parser.add_argument("--exp", "--ablation", default="full", choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--config_file", default="configs/WMVeID863/clip_facenet_wmveid863.yml", type=str)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--resume_weight", default="", type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--IC_param", default="0.8", type=float)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.MAX_EPOCHS = args.epochs
    cfg.SOLVER.START_EPOCH = args.start_epoch
    cfg.IC_param = args.IC_param

    now = datetime.datetime.now()
    strtime = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = cfg.OUTPUT_DIR + f"_abl_{args.exp}_" + strtime
    os.makedirs(output_dir, exist_ok=True)
    cfg.SAVE_DIR = output_dir
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    setup_logger("transreid", output_dir, "train_log")

    import logging
    logger = logging.getLogger("transreid.image_train")
    exp_name = args.exp
    exp_cfg = ABLATION_CONFIGS[exp_name]

    train_loader, train_loader_normal, val_loader, num_query, num_classes, num_cameras = \
        make_dataloader(
            cfg,
            use_flare_sampler=exp_cfg["flare_sampler"],
            modality_dropout=exp_cfg["modality_dropout"],
        )

    model = CLIPFACENet(num_classes=num_classes, cfg=cfg)
    model.use_mamba_enhance = False
    model.use_fce = False

    exp_name, exp_cfg = apply_ablation_config(model, args.exp)
    if args.resume_weight:
        model.load_param(args.resume_weight)
    logger.info(f"Ablation {exp_name}: {exp_cfg['desc']}")
    logger.info(
        f"Data flags: flare_sampler={exp_cfg['flare_sampler']}, "
        f"modality_dropout={exp_cfg['modality_dropout']}"
    )
    logger.info(f"Ablation flags: {format_ablation_flags(model)}")

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    do_train_amp(cfg, model, center_criterion, train_loader, val_loader,
                 optimizer, optimizer_center, scheduler, loss_func, num_query, args.local_rank)
