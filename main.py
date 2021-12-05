from utils.def_config import assert_and_infer_cfg
from utils.parser import load_config, parse_args
import torch
import torch.nn as nn
import torchvision

# Trained SlowFast R101 on AVA 2.2, pre-trained on Kinetics 600 (mAP: 29.4)
path = '/content/drive/MyDrive/SLOWFAST_64x2_R101_50_50A.pkl'

args = parse_args()
cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)

from models import build_model
from datasets import loader

myloader = loader.construct_loader(cfg, "train")


mymodel = build_model(cfg)
mymodel.load_state_dict((torch.load(path))['model_state'])

rand_input = torch.rand(1, 3, 64, 224, 224)
frames = rand_input

fast_pathway = torch.index_select(
            frames,
            2,
            torch.linspace(
                0, frames.shape[2] - 1, frames.shape[2] // 2
            ).long(),
        ).to('cuda')
slow_pathway = torch.index_select(
            frames,
            2,
            torch.linspace(
                0, frames.shape[2] - 1, frames.shape[2] // 8
            ).long(),
        ).to('cuda')

frame_list = [slow_pathway, fast_pathway]       

rand_input = frame_list
rand_bbox = torch.tensor([[0, 2.1, 3, 4, 5],[0, 2.1, 2, 4, 5]]).to('cuda')

rand_output = mymodel(rand_input, rand_bbox)
print(rand_output.shape)
