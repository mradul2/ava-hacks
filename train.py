import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data.dataset import AVADataset, AVADatasetNpy
from models.base import AVAModel
from slowfast.utils.def_config import assert_and_infer_cfg
from slowfast.utils.meters import AVAMeter
from slowfast.utils.parser import load_config, parse_args
from utils.wandb import init_wandb


def train_epoch(train_loader, model, criterion, optimizer, train_meter, cur_epoch, cfg):
    # Enable train mode.
    model.train()
    epoch_loss = 0.0
    
    for cur_iter, (inputs, labels, ori_boxes, metadata) in enumerate(train_loader):
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
        
        train_meter.update_stats(None, None, None, iter_loss, 0)
        
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        
        wandb.log({
            "Iteration loss": iter_loss,
            "Learning rate": optimizer.param_groups[0]['lr']
        })
        
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    epoch_loss /= len(train_loader)
    wandb.log({
        "Epoch Loss": epoch_loss,
    })
    
@torch.no_grad()
def eval_epoch(valid_loader, model, criterion, val_meter, cur_epoch, cfg):
    model.eval()
    valid_loss = 0.0
    for cur_iter, (inputs, labels, ori_boxes, metadata) in enumerate(valid_loader):
        # Transfer the data to the current GPU device.
        inputs = inputs.cuda()
        ori_boxes = ori_boxes.cuda()
        metadata = metadata.cuda()
        labels = labels.cuda()
        
        # Forward pass
        preds = model(inputs)

        # Backward pass
        loss = criterion(preds, labels)
        valid_loss += loss.item()

    valid_loss /= len(valid_loader)

    wandb.log({
        "Validation Loss": valid_loss,
    })

@torch.no_grad()
def calculate_metrics(model, valid_loader, val_meter, cfg):
    model.eval()
    for cur_iter, (inputs, labels, ori_boxes, metadata) in enumerate(valid_loader):
        # Transfer the data to the current GPU device.
        inputs = inputs.cuda()
        ori_boxes = ori_boxes.cuda()
        metadata = metadata.cuda()
        labels = labels.cuda()
        
        # Forward pass
        preds = model(inputs)
        
        # Update and log stats.
        head = None
        if cfg.MODEL.HEAD_ACT == "sigmoid":
            head = nn.Sigmoid()
        elif cfg.MODEL.HEAD_ACT == "softmax":
            head = nn.Softmax(dim=1)
        if head is not None:
            preds = head(preds) 
            
        val_meter.update_stats(preds, ori_boxes, metadata)
        
    validation_results = val_meter.full_map
    print(validation_results)
    val_meter.reset()

    
def train(train_loader, valid_loader, model, train_meter, valid_meter, cfg):
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    gamma = cfg.SOLVER.GAMMA
    momentum = cfg.SOLVER.MOMENTUM
    if cfg.MODEL.LOSS_FUNC == "bce_logit":
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          weight_decay=wd,
                          momentum=momentum)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    for cur_epoch in tqdm(range(1, cfg.SOLVER.MAX_EPOCH+1)):
        train_epoch(train_loader, model, criterion, optimizer, train_meter, cur_epoch, cfg)
        eval_epoch(valid_loader, model, criterion, valid_meter, cur_epoch, cfg)
        scheduler.step()
    calculate_metrics(model, valid_loader, valid_meter, cfg)

    # Save the model
    torch.save(model.state_dict(), "model.pth")
    wandb.save("model.pth")
        
def main():
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    
    # Initialize wandb
    init_wandb(cfg)

    print("Constructing Training and Validation DataLoader")
    train_dataset = AVADatasetNpy(cfg.DATA.FEATURE_DIR, "train")
    valid_dataset = AVADatasetNpy(cfg.DATA.FEATURE_DIR, "val")
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
