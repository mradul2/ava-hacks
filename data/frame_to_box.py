from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import os 

from dataset import AVADataset


def perform(train_loader, val_loader, cfg):

    train_idx = 0
    train_pred_dir = "./data/train/preds"
    train_ori_box_dir = "./data/train/ori_boxes"
    train_metadata_dir = "./data/train/metadata"
    train_label_dir = "./data/train/labels"
    if not os.path.exists(train_pred_dir):
        os.makedirs(train_pred_dir)
    if not os.path.exists(train_ori_box_dir):
        os.makedirs(train_ori_box_dir)
    if not os.path.exists(train_metadata_dir):
        os.makedirs(train_metadata_dir)
    if not os.path.exists(train_label_dir):
        os.makedirs(train_label_dir)

    for cur_iter, (inputs, labels, ori_boxes, metadata) in enumerate(tqdm(train_loader)):
        preds = inputs
        preds.detach()
        labels.detach()
        ori_boxes.detach()
        metadata.detach()

        preds = preds[0]
        ori_boxes = ori_boxes[0]
        metadata = metadata[0]
        labels = labels[0]

        for i in range(preds.shape[0]):
            pred_box = preds[i]
            ori_box = ori_boxes[i]
            meta = metadata[i]
            label = labels[i]

            # save the tensors as numpy files
            np.save(os.path.join(train_pred_dir, str(train_idx)), pred_box)
            np.save(os.path.join(train_ori_box_dir, str(train_idx)), ori_box)
            np.save(os.path.join(train_metadata_dir, str(train_idx)), meta)
            np.save(os.path.join(train_label_dir, str(train_idx)), label)

            train_idx += 1

    val_idx = 0
    val_pred_dir = "./data/val/preds"
    val_ori_box_dir = "./data/val/ori_boxes"
    val_metadata_dir = "./data/val/metadata"
    val_label_dir = "./data/val/labels"
    if not os.path.exists(val_pred_dir):
        os.makedirs(val_pred_dir)
    if not os.path.exists(val_ori_box_dir):
        os.makedirs(val_ori_box_dir)
    if not os.path.exists(val_metadata_dir):
        os.makedirs(val_metadata_dir)
    if not os.path.exists(val_label_dir):
        os.makedirs(val_label_dir)

    for cur_iter, (inputs, labels, ori_boxes, metadata) in enumerate(tqdm(val_loader)):
        preds = inputs

        preds.detach()
        labels.detach()
        ori_boxes.detach()
        metadata.detach()

        preds = preds[0]
        ori_boxes = ori_boxes[0]
        metadata = metadata[0]
        labels = labels[0]

        for i in range(preds.shape[0]):
            pred_box = preds[i]
            ori_box = ori_boxes[i]
            meta = metadata[i]
            label = labels[i]

            # save the tensors as numpy files
            np.save(os.path.join(val_pred_dir, str(val_idx)), pred_box)
            np.save(os.path.join(val_ori_box_dir, str(val_idx)), ori_box)
            np.save(os.path.join(val_metadata_dir, str(val_idx)), meta)
            np.save(os.path.join(val_label_dir, str(val_idx)), label)

            val_idx += 1


def main():

    print("Constucting Training and Validation DataLoader")
    train_dataset = AVADataset("data/ava_features", "train")
    val_dataset = AVADataset("data/ava_features", "val")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print("Dataloaders constructed")

    cfg = None
    perform(train_loader, val_loader, cfg)

if __name__ == "__main__":
    main()