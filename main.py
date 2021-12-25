import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from slowfast.utils.def_config import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
from slowfast.utils.meters import AVAMeter

import numpy as np
from tqdm import tqdm
import os 

from data.dataset import AVADataset
from models.base import AVAModel

import wandb
from utils.wandb import init_wandb

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
        test_meter.update_stats(preds[0], ori_boxes[0], metadata[0])
        test_meter.log_iter_stats(None, cur_iter)
        test_meter.iter_tic()

    test_meter.finalize_metrics()
    return test_meter

def train(data_loader, model, cfg):
    # pretrained = (torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH))
    # model.load_state_dict(pretrained, strict=False)
    
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    gamma = cfg.SOLVER.GAMMA
    momentum = cfg.SOLVER.MOMENTUM
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          weight_decay=wd,
                          momentum=momentum)
    
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    for cur_epoch in tqdm(range(cfg.SOLVER.MAX_EPOCH)):
        epoch_loss = 0.0
        for cur_iter, (inputs, labels, ori_boxes, metadata) in enumerate(tqdm(data_loader)):
            model.train()
            if cfg.NUM_GPUS:
                # Transfer the data to the current GPU device.
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Forward pass
            preds = model(inputs)
            loss = criterion(preds, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_loss = loss.item()
            epoch_loss += iter_loss
        
        scheduler.step()
        epoch_loss /= len(data_loader)
        wandb.log({
            "Epoch Loss": epoch_loss,
            "Learning Rate": optimizer.param_groups[0]["lr"],
            "Epoch": cur_epoch
        })
    

def main():
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    
    # Initialize wandb
    init_wandb(cfg)

    print("Constucting DataLoader")
    train_dataset = AVADataset(cfg.DATA.FEATURE_DIR, "train")
    data_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE)

    print("Constructing Model")
    model = AVAModel(dim_in=2304, dim_out=80)
    model = model.cuda()
    print("Model Construction Complete")
    
    print("Training loop called")
    train(data_loader, model, cfg)
    print("Training completed")

    print("Constructing AVA Meter")
    test_meter = AVAMeter(len(data_loader), cfg, mode="test")
    test_meter = perform_test(data_loader, model, test_meter, cfg)

if __name__ == "__main__":
    main()