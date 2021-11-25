from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
import torch

# Trained SlowFast R101 on AVA 2.2, pre-trained on Kinetics 600 (mAP: 29.4)
path = '/content/drive/MyDrive/SLOWFAST_64x2_R101_50_50A.pkl'

args = parse_args()
cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)
# 

from models import build_model
mymodel = build_model(cfg)
mymodel.load_state_dict((torch.load(path, map_location=torch.device('cpu')))['model_state'])

rand_input = [torch.rand(1, 3, 16, 224, 224), torch.rand(1, 3, 64, 224, 224)]
rand_output = mymodel(rand_input)
print(rand_output.shape)