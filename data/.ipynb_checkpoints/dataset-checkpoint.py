import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import os
import glob
import numpy as np

from os.path import join
from os.path import isfile


class AVADataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = root_dir
        if mode == "train" or mode == "val":
            self.root_dir = join(self.root_dir, mode)
        else:
            print("Incorrect mode provided")
        
        self.transform = transform
        
        # Create list of all files present in the corresponding directories
        self.file_list = []
        for f in os.listdir(self.root_dir):
            file_path = join(self.root_dir, f)
            if isfile(file_path):
                self.file_list.append(file_path)
        self.file_list = sorted(self.file_list)
        
        print("Total KeyFrame Tensors Loaded: ", len(self.file_list))
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, index):
        file = np.load((self.file_list)[index])
        
        preds = file['preds'] 
        labels = file['labels']
        ori_boxes = file['ori_boxes']
        metadata = file['metadata']
        
        preds = torch.from_numpy(preds)
        labels = torch.from_numpy(labels)
        ori_boxes = torch.from_numpy(ori_boxes)
        metadata = torch.from_numpy(metadata)
        
        return preds, labels, ori_boxes, metadata