import atexit
import torch
import torch.nn as nn
import torchvision

class SlowFastHead(nn.Module):
    def __init__(self, dim_in, dim_out, act):
        super(SlowFastHead, self).__init__()   
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.projection = nn.Linear(dim_in, dim_out, bias=True)
        if act == "none":
            self.act = None
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        if act == "softmax":
            self.act = nn.Softmax(dim=1)     
    def forward(self, preds):
        preds = self.projection(preds)
        if self.act is not None:
            preds = self.act(preds)
        return preds 
        
class AVAModel(nn.Module):
    def __init__(self, dim_in, dim_out, act):
        super(AVAModel, self).__init__()
        self.head = SlowFastHead(dim_in, dim_out, act)
    def forward(self, preds):
        preds = self.head(preds)
        return preds