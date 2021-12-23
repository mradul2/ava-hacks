import torch
import torch.nn as nn
import torchvision

from slowfast.utils.def_config import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
from slowfast.utils.meters import AVAMeter

from slowfast.models import build_model
from slowfast.datasets import loader

from numpy import savez_compressed
import numpy as np
from tqdm import tqdm
import os 


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(tqdm(test_loader)):
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
        test_meter.data_toc()

        # Compute the predictions.
        preds = model(inputs, meta["boxes"])
        ori_boxes = meta["ori_boxes"]
        metadata = meta["metadata"]

        preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
        ori_boxes = (
            ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
        )
        metadata = (
            metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
        )

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(preds, ori_boxes, metadata)
        test_meter.log_iter_stats(None, cur_iter)
        test_meter.iter_tic()

    test_meter.finalize_metrics()
    return test_meter

@torch.no_grad()
def feature_extraction(cfg, data_loader, model):
    # Create directory for feature storage
    directory_names = ["preds", "ori_boxes", "labels", "metadata"]
    if os.path.join(os.getcwd() + "features") is None: 
        os.mkdir("features")
    for directory_name in directory_names:
         if not os.path.isdir(os.path.join(os.getcwd(), "features", directory_name)): 
            os.mkdir(os.path.join("features", directory_name))   
        
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
        
        savez_compressed('./features/preds/' + str(cur_iter) + '.npz', preds)
        savez_compressed('./features/ori_boxes/' + str(cur_iter) + '.npz', ori_boxes)
        savez_compressed('./features/metadata/' + str(cur_iter) + '.npz', metadata)
        savez_compressed('./features/labels/' + str(cur_iter) + '.npz', labels)

def main():
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    print("Constucting DataLoader")
    data_loader = loader.construct_loader(cfg, "val")

    print("Constructing Model")
    model = build_model(cfg)
    model.load_state_dict((torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH))['model_state'])
    
    feature_extraction(cfg, data_loader, model)

    # print("Constructing AVA Meter")
    # test_meter = AVAMeter(len(myloader), cfg, mode="test")
    # test_meter = perform_test(myloader, mymodel, test_meter, cfg)
        

if __name__ == "__main__":
    main()