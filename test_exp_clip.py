import os
import argparse
from config import cfg
from datasets import make_dataloader
from model.clip_facenet import CLIPFACENet
from processor import do_inference_amp as do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="G", choices=["F", "G"])
    parser.add_argument("--config_file", default="configs/WMVeID863/clip_facenet_wmveid863.yml")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    logger = setup_logger("transreid", cfg.OUTPUT_DIR, if_train=False)
    logger.info("Running CLIP-FACENet inference")

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num = make_dataloader(cfg)

    model = CLIPFACENet(num_classes=num_classes, cfg=cfg)
    model.use_mamba_enhance = False
    model.use_mcloss = True
    model.use_fce = False
    model.use_mfmp = True
    model.use_coen_lite = True

    model.load_param(cfg.TEST.WEIGHT)

    do_inference(cfg, model, val_loader, num_query)