import glob
import os
from os.path import isfile, join

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


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

class AVADatasetNpy(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = root_dir
        if mode == "train" or mode == "val":
            self.root_dir = join(self.root_dir, mode)
        else:
            print("Incorrect mode provided")

        self.transform = transform

        # Read the npy files
        self.preds = np.load(join(self.root_dir, mode + "_preds.npy"))
        self.labels = np.load(join(self.root_dir, mode + "_labels.npy"))
        self.ori_boxes = np.load(join(self.root_dir, mode + "_ori_boxes.npy"))
        self.metadata = np.load(join(self.root_dir, mode + "_metadata.npy"))

        print("Total Box Tensors Loaded: ",len(self))

    def __len__(self):
        return self.preds.shape[0]

    def __getitem__(self, index):
        preds = self.preds[index]
        labels = self.labels[index]
        ori_boxes = self.ori_boxes[index]
        metadata = self.metadata[index]

        preds = torch.from_numpy(preds)
        labels = torch.from_numpy(labels)
        ori_boxes = torch.from_numpy(ori_boxes)
        metadata = torch.from_numpy(metadata)

        return preds, labels, ori_boxes, metadata
