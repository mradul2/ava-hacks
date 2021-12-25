import torch
import torch.nn as nn
import torchvision

class SlowFastHead(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SlowFastHead, self).__init__()   
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.projection = nn.Linear(dim_in, dim_out, bias=True)
        self.act = nn.Sigmoid()
    def forward(self, preds):
        preds = self.projection(preds)
        preds = self.act(preds)
        return preds 
        
class AVAModel(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(AVAModel, self).__init__()
        self.head = SlowFastHead(dim_in, dim_out)
    def forward(self, preds):
        preds = self.head(preds)
        return preds