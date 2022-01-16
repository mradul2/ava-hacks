import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from slowfast.utils.def_config import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter

import numpy as np
from tqdm import tqdm
import os 

from data.dataset import AVADataset
from models.base import AVAModel

import wandb
from utils.wandb import init_wandb

def train_epoch(train_loader, model, criterion, optimizer, train_meter, cur_epoch, cfg):
    # Enable train mode.
    model.train()
    data_size = len(train_loader)
    epoch_loss = 0.0
    
    for cur_iter, (inputs, labels, ori_boxes, metadata) in enumerate(tqdm(train_loader)):
        # Transfer the data to the current GPU device.
        inputs = inputs.cuda()
        labels = labels.cuda()
        labels = labels[0]
        inputs = inputs[0]
        ori_boxes = ori_boxes.cuda()
        metadata = metadata.cuda()
        
        # Forward pass
        preds = model(inputs)
        labels = torch.argmax(labels, dim=1)
        loss = criterion(preds, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_loss = loss.item()
        iter_loss /= inputs.shape[0]
        epoch_loss += iter_loss
        
        train_meter.update_stats(None, None, None, iter_loss, 0)
        
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        
        wandb.log({
            "Iteration loss": iter_loss,
        })
        
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    epoch_loss /= len(train_loader)
    wandb.log({
        "Epoch Loss": epoch_loss,
    })
    
@torch.no_grad()
def eval_epoch(valid_loader, model, val_meter, cur_epoch, cfg):
    model.eval()
    for cur_iter, (inputs, labels, ori_boxes, metadata) in enumerate(tqdm(valid_loader)):
        # Transfer the data to the current GPU device.
        inputs = inputs.cuda()
        labels = labels.cuda()
        labels = labels[0]
        inputs = inputs[0]
        ori_boxes = ori_boxes.cuda()
        metadata = metadata.cuda()
        
        # Forward pass
        preds = model(inputs)
        labels = torch.argmax(labels, dim=1)
        
        # Update and log stats.
        val_meter.update_stats(preds, ori_boxes[0], metadata[0])
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        
    val_meter.log_epoch_stats(cur_epoch)
    print(val_meter.full_map)
    val_meter.reset()
    
def train(train_loader, valid_loader, model, train_meter, valid_meter, cfg):
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
    
    for cur_epoch in tqdm(range(0, cfg.SOLVER.MAX_EPOCH)):
        train_epoch(train_loader, model, criterion, optimizer, train_meter, cur_epoch, cfg)
        eval_epoch(valid_loader, model, valid_meter, cur_epoch, cfg)
        scheduler.step()
        
def main():
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    
    # Initialize wandb
    init_wandb(cfg)

    print("Constructing Training and Validation DataLoader")
    train_dataset = AVADataset(cfg.DATA.FEATURE_DIR, "train")
    valid_dataset = AVADataset(cfg.DATA.FEATURE_DIR, "val")
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)
    print("Dataloaders constructed")
    
    print("Constructing Model")
    model = AVAModel(dim_in=2304, dim_out=80)
    model = model.cuda()
    print("Model Construction Complete")
    
    print("Constructing Train and Validation Meter")
    train_meter = AVAMeter(len(train_loader), cfg, mode="train")
    valid_meter = AVAMeter(len(valid_loader), cfg, mode="val")
    print("Meters constructed")
    
    print("Training loop called")
    train(train_loader, valid_loader, model, train_meter, valid_meter, cfg)
    print("Training completed")

if __name__ == "__main__":
    main()