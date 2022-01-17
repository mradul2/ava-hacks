import torch
import torch.nn as nn

import numpy as np
import os 

from models.base import AVAModel

def main():
    model = AVAModel(dim_in=2304, dim_out=80)
    slowfast_state_dict = torch.load("/content/slowfast.pkl", map_location=torch.device('cpu'))
    model.load_state_dict(slowfast_state_dict["model_state"], strict=False)
    torch.save(model.state_dict(), "/content/slowfast_head.pth")

    for key in slowfast_state_dict["model_state"].keys():
        if key in model.state_dict().keys():
            print(key)

if __name__ == "__main__":
    main()