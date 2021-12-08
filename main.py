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


for batch in myloader:
  batch = batch
  inputs, labels, _, meta = batch

  if cfg.NUM_GPUS:
      # Transferthe data to the current GPU device.
      if isinstance(inputs, (list,)):
          for i in range(len(inputs)):
              inputs[i] = inputs[i].cuda(non_blocking=True)
      else:
          inputs = inputs.cuda(non_blocking=True)
      labels = labels.cuda()
      for key, val in meta.items():
          if isinstance(val, (list,)):
              for i in range(len(val)):
                  val[i] = val[i].cuda(non_blocking=True)
          else:
              meta[key] = val.cuda(non_blocking=True)

  preds = model(inputs, meta["boxes"])
  break


