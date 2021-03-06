import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data.dataset import AVADataset, AVADatasetNpy
from models.base import AVAModel
from slowfast.utils.def_config import assert_and_infer_cfg
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter
from slowfast.utils.parser import load_config, parse_args


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, ori_boxes, metadata) in enumerate(tqdm(test_loader)):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
        test_meter.data_toc()

        # Compute the predictions.
        preds = model(inputs)

        preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
        ori_boxes = ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
        metadata = metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(preds, ori_boxes, metadata)
        test_meter.log_iter_stats(None, cur_iter)
        test_meter.iter_tic()

    test_meter.finalize_metrics()
    return test_meter

def main():
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    print("Constucting Validation DataLoader")
    test_dataset = AVADatasetNpy(cfg.DATA.FEATURE_DIR, "val")
    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)
    print("Dataloaders constructed")
    
    print("Constructing Model")
    model = AVAModel(dim_in=2304, dim_out=80)
    model = model.cuda()
    print("Model Construction Complete")
    
    print("Loading model state dict")
    state_dict = torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH)
    model.load_state_dict(state_dict)
    print("Model state dict loaded")

    print("Constructing AVA Meter")
    test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    test_meter = perform_test(test_loader, model, test_meter, cfg)
    print("Sucessfully ended")

if __name__ == "__main__":
    main()
