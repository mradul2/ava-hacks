import torch
import torch.nn as nn
import torchvision

from slowfast.utils.def_config import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
from slowfast.models import build_model
from slowfast.datasets import loader

from numpy import savez_compressed
import numpy as np
from tqdm import tqdm
import os 

@torch.no_grad()
def feature_extraction(cfg, data_loader, model, root_dir):  
    model.eval()
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(tqdm(data_loader)):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Compute the predictions.
        preds = model(inputs, meta["boxes"])
        ori_boxes = meta["ori_boxes"]
        metadata = meta["metadata"]

        preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
        ori_boxes = ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
        metadata = metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
        labels = labels.detach().cpu() if cfg.NUM_GPUS else labels.detach()
        
        file_name = os.path.join(root_dir, str(f'{cur_iter}.npz'))
        
        savez_compressed(file_name,
                        preds=preds,
                        ori_boxes=ori_boxes,
                        metadata=metadata,
                        labels=labels)

def main():
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    print("Constucting DataLoader")
    train_data_loader = loader.construct_loader(cfg, "train")
    valid_data_loader = loader.construct_loader(cfg, "val")

    print("Constructing Model")
    model = build_model(cfg)
    model.load_state_dict((torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH))['model_state'])
    
    
    # Root directory for feature storage: ./features/
    if not os.path.isdir(os.path.join(cfg.DATA.FEATURE_DIR)): 
        os.mkdir(os.path.join(cfg.DATA.FEATURE_DIR))
    
    # Directory for train feature storage: ./features/train/
    # Directory for train feature storage: ./features/val/
    train_root_dir = os.path.join(cfg.DATA.FEATURE_DIR, "train")
    val_root_dir = os.path.join(cfg.DATA.FEATURE_DIR, "val")
    if not os.path.isdir(train_root_dir): 
        os.mkdir(train_root_dir)
    if not os.path.isdir(val_root_dir): 
        os.mkdir(val_root_dir)
    
    print("Extracting Features for Training Set")
    feature_extraction(cfg, train_data_loader, model, train_root_dir)
    print("Extracting Features for Validation Set")
    feature_extraction(cfg, valid_data_loader, model, val_root_dir)
    print("Feature Extraction and Saving Completed")

if __name__ == "__main__":
    main()