from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import os 

from dataset import AVADataset


def perform(train_loader, val_loader, cfg):

    pred_array = []
    ori_box_array = []
    meta_array = []
    label_array = []

    for cur_iter, (inputs, labels, ori_boxes, metadata) in enumerate(tqdm(train_loader)):
        preds = inputs
        preds = preds.detach().numpy()
        labels = labels.detach().numpy()
        ori_boxes = ori_boxes.detach().numpy()
        metadata = metadata.detach().numpy()

        preds = preds[0]
        ori_boxes = ori_boxes[0]
        metadata = metadata[0]
        labels = labels[0]

        for i in range(preds.shape[0]):
            pred_box = preds[i]
            ori_box = ori_boxes[i]
            meta = metadata[i]
            label = labels[i]

            pred_array.append(pred_box)
            ori_box_array.append(ori_box)
            meta_array.append(meta)
            label_array.append(label)


    np.save("data/train_preds.npy", np.array(pred_array))
    np.save("data/train_ori_boxes.npy", np.array(ori_box_array))
    np.save("data/train_metadata.npy", np.array(meta_array))
    np.save("data/train_labels.npy", np.array(label_array))


    for cur_iter, (inputs, labels, ori_boxes, metadata) in enumerate(tqdm(val_loader)):
        preds = inputs

        preds.detach().numpy()
        labels.detach().numpy()
        ori_boxes.detach().numpy()
        metadata.detach().numpy()

        preds = preds[0]
        ori_boxes = ori_boxes[0]
        metadata = metadata[0]
        labels = labels[0]

        for i in range(preds.shape[0]):
            pred_box = preds[i]
            ori_box = ori_boxes[i]
            meta = metadata[i]
            label = labels[i]

            pred_array.append(pred_box)
            ori_box_array.append(ori_box)
            meta_array.append(meta)
            label_array.append(label)

    np.save("data/val_preds.npy", np.array(pred_array))
    np.save("data/val_ori_boxes.npy", np.array(ori_box_array))
    np.save("data/val_metadata.npy", np.array(meta_array))
    np.save("data/val_labels.npy", np.array(label_array))
    


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