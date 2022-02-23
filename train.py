import os
from xml.dom.expatbuilder import theDOMImplementation

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
from zmq import CONFLATE

from data.dataset import AVADataset, AVADatasetNpy
from models.base import AVAModel, AVAModelMLP
from slowfast.utils.def_config import assert_and_infer_cfg
from slowfast.utils.meters import AVAMeter
from slowfast.utils.parser import load_config, parse_args
from utils.wandb import init_wandb

from utils.defs import *


@torch.no_grad()
def evaluation(valid_loader, model, criterion):
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
    return valid_loss

@torch.no_grad()
def calculate_metrics(model, valid_loader, val_meter, cfg):
    model.eval()
    for inputs, labels, ori_boxes, metadata in valid_loader:
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
        
    val_meter.log_epoch_stats(-1)
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
    elif cfg.MODEL.LOSS_FUNC == "bce_logit_weighted":
        pos_weight = torch.log(1 + 1 / CLASS_FREQ)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          weight_decay=wd,
                          momentum=momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg.SOLVER.STEPS, gamma)

    model.train()
    iterator = iter(train_loader)
    for cur_iter in tqdm(range(cfg.SOLVER.MAX_ITERATIONS)):
        batch = next(iterator)
        inputs, labels, ori_boxes, metadata = batch

        # Convert to cuda tensors
        inputs = inputs.cuda()
        labels = labels.cuda()
        ori_boxes = ori_boxes.cuda()
        metadata = metadata.cuda()

        # Forward pass
        preds = model(inputs)
        loss = criterion(preds, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        iteration_loss = loss.item()
        wandb.log({
            "Iteration": cur_iter,
            "Train Iteration loss": iteration_loss,
            "Learning rate": optimizer.param_groups[0]["lr"],
        })

        if cur_iter % cfg.TRAIN.EVAL_PERIOD == 0:
            valid_loss = evaluation(valid_loader, model, criterion)
            wandb.log(
                {"Validation loss": valid_loss}
            )

    calculate_metrics(model, valid_loader, valid_meter, cfg)
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
    dim_in = 2304
    dim_mid = 512
    dim_out = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.MODEL_NAME == "Base":
        model = AVAModel(dim_in, dim_out)
    elif cfg.MODEL.MODEL_NAME == "MLP":
        model = AVAModelMLP(dim_in, dim_mid, dim_out)
    else:
        raise ValueError("Unknown model name: {}".format(cfg.MODEL.MODEL_NAME))
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