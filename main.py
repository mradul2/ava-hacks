from utils.def_config import assert_and_infer_cfg
from utils.parser import load_config, parse_args
import torch
import torch.nn as nn
import torchvision

from utils.meters import AVAMeter


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
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


def main():
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

    test_meter = AVAMeter(len(myloader), cfg, mode="test")
    test_meter = perform_test(myloader, mymodel, test_meter, cfg)
        


if __name__ == "__main__":
  main()