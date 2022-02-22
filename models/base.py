import torch
import torch.nn as nn
import torchvision

class SlowFastHead(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SlowFastHead, self).__init__()   
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.projection = nn.Linear(dim_in, dim_out, bias=True)
    def forward(self, preds):
        preds = self.projection(preds)
        return preds 

class MLP(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out):
        super(MLP, self).__init__()   
        self.dim_in = dim_in
        self.din_mid = dim_mid
        self.dim_out = dim_out
        self.fc1 = nn.Linear(dim_in, dim_mid)
        self.fc2 = nn.Linear(dim_mid, dim_out)
        self.relu = nn.ReLU()
    def forward(self, preds):
        x = self.fc1(preds)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
class AVAModel(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(AVAModel, self).__init__()
        self.head = SlowFastHead(dim_in, dim_out)
    def forward(self, preds):
        preds = self.head(preds)
        return preds

class AVAModelMLP(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out):
        super(AVAModel, self).__init__()
        self.head = MLP(dim_in, dim_mid, dim_out)
    def forward(self, preds):
        preds = self.head(preds)
        return preds