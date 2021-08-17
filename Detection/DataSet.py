from torch.utils.data import Dataset
import numpy as np
import random
from AnchorGenerator import generator_anchor_area, apply_bboxReg_area
import Config as config
from Transform import myTransform
import matplotlib.pyplot as plt
from matplotlib import patches
from utils import (read_image_2D, get_ImgPath, generator_bbox, randomIdx, bbox2show)


def data_split_2D(img_lst, ratio, shuffle=False):
    n_total = len(img_lst)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], img_lst
    if shuffle:
        random.shuffle(img_lst)
    train_lst = img_lst[:offset]
    val_lst = img_lst[offset:]
    return train_lst, val_lst


class RetinaDataSet(Dataset):
    def __init__(self, img_lst, isTrain=True):
        super(RetinaDataSet, self).__init__()
        self.img_lst = img_lst
        self.expFactor = config.ExpBox
        self.img_shape = config.Image_shape
        self.isTrain = isTrain

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, index):
        img, mask = read_image_2D(self.img_lst[index])
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask,axis=0)
        if self.isTrain:
            img, mask = myTransform(img, mask)
        gt_boxes = []
        expF = self.expFactor
        for i in range(config.BoxScale):
            gt_boxes.append(generator_bbox(mask, mode='box', expFactor=expF))
            expF += 1
        gt_boxes = np.array(gt_boxes)
        gt_boxes = gt_boxes.reshape((-1, 4))
        anchors, bbox_labels, LossIdx, bboxIdx = generator_anchor_area(img_shape=self.img_shape, gtBoxes=gt_boxes)
        return img / 255, mask / 255, anchors, bbox_labels, LossIdx, bboxIdx


def test():
    img_lst = get_ImgPath()
    train_lst, val_lst = data_split_2D(img_lst, 3, shuffle=False)
    dataset = RetinaDataSet(train_lst, isTrain=False)
    idx = randomIdx(train_lst)
    # idx = 207
    img, mask, anchors, bbox_labels, LossIdx, RegIdx = dataset[idx]

    output_anchcors, output_bboxs = apply_bboxReg_area(bboxMat=bbox_labels, img_shape=config.Image_shape,
                                                       anchors=anchors)
    img = img.squeeze(0)
    print("------------------------Show Image-------------------------------")
    print(idx)
    gt_boxes = []
    expFactor = config.ExpBox
    for i in range(config.BoxScale):
        gt_boxes.append(generator_bbox(mask, mode='box', expFactor=expFactor))
        expFactor += 1
    gt_boxes = np.array(gt_boxes)
    gt_boxes = gt_boxes.reshape((-1, 4))
    plt.figure(idx)
    ax = plt.axes()
    plt.imshow(img)
    counter = 0
    for i in range(len(output_bboxs)):
        print(output_bboxs[i].shape, output_anchcors[i].shape)
        bbox = bbox2show(output_bboxs[i], mode='center')
        for idx, box in enumerate(bbox):
            counter += 1
            topright = (box[0], box[1])
            height = box[2]
            width = box[3]
            rect = patches.Rectangle(topright, height, width, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
    gt_boxes = bbox2show(gt_boxes, mode='box')
    for box in gt_boxes:
        topright = (box[0], box[1])
        height = box[2]
        width = box[3]
        print('gt_height:', height, 'gt_width:', width)
        rect = patches.Rectangle(topright, height, width, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    print('Total bounding box: ', counter)
    plt.show()


if __name__ == "__main__":
    test()
