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

from data.dataset import AVADataset, AVADatasetNpy
from models.base import AVAModel, AVAModelMLP
from slowfast.utils.def_config import assert_and_infer_cfg
from slowfast.utils.meters import AVAMeter
from slowfast.utils.parser import load_config, parse_args
from utils.wandb import init_wandb
        
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
    
    train_frequency = torch.zeros(cfg.MODEL.NUM_CLASSES)
    valid_frequency = torch.zeros(cfg.MODEL.NUM_CLASSES)

    # Iterate over the training and validation loop
    for iter, (batch) in enumerate(train_loader):
        inputs, labels, ori_boxes, metadata = batch
        for i in range(labels.shape[0]):
            train_frequency += labels[i]

    for iter, (batch) in enumerate(valid_loader):
        inputs, labels, ori_boxes, metadata = batch
        for i in range(labels.shape[0]):
            valid_frequency += labels[i]

    print("Training class frequency:")
    print(train_frequency)
    print("Validation class frequency:")
    print(valid_frequency)

if __name__ == "__main__":
    main()