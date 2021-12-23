import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import os
import numpy as np

class AVA(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.preds_dir = os.path.join(self.root_dir, "preds")
        self.labels_dir = os.path.join(self.root_dir, "labels")
        self.ori_boxes_dir = os.path.join(self.root_dir, "ori_boxes")
        self.meta_data_dir = os.path.join(self.root_dir, "meta_data")
        self.transform = transform
        
    def __len__(self):
        
    def __getitem__(self, index):
        