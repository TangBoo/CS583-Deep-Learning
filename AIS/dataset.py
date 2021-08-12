import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import config
from utils import iou_width_height
from pydicom import dcmread
import numpy as np
from utils import cells_to_bboxes, convert_box
import Meta as meta

anchors = meta.anchors
table = meta.meta_table


class YoloDataset(Dataset):
    def __init__(self, table, anchors, GridSize=config.GridSize,
                 C=config.NUM_CLASSES, transform=None, isTrain=True, shuffle=True):
        '''
        Args:
            1. csv_file : meta data
            2. img_dir : the paht of img file
            3. anchors : put all anchors for all scale prediction together
            4. receptive_field
            5. C : the number of classes in dataset
            6. Transform : do some augmentation if you need
        '''

        if shuffle:
            table = table.sample(frac=1.0).reset_index(drop=True)
        self.annotations = table
        self.transform = transform
        self.label = table.values[:, 5:8].tolist()
        self.bbox = table["boxes"]
        self.img_path = table["img_path"]
        self.S = GridSize
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        '''
        Args:
            1. anchors : the size of all bouding box for all scales,
                        each scale get three bouding box with different size.
            2. bbox : the ground truth is for input image
            3. Image : Input Image
        Describe:
            At first, we assign each ground truth box with one anchor,
            so we go through all bbox for finding out which scale is the each one of bboxes belonging to.
            As for looking for scale bbox belonging to, we only care about the area.
            We are looking for which bounding box of anchor has most similar area with ground truth.
            Beside, when we has assigned a ground truth with a bounding box,
            how about other anchor whose overlap area with ground truth is bigger than 0.5. Ignore!!!
        '''

        labels = torch.tensor(self.label[index])  # [prob_obj, T, I, A]
        bboxes = np.array(self.bbox[index])  # [num_box, x, y, w, h]
        img_path = self.img_path[index]
        image = dcmread(img_path).pixel_array.astype('float64')

        # box_show = [convert_box(box, image.shape) for box in bboxes]
        # img_show = image.copy()
        # h, w = img_show.shape
        # for box in box_show:
        #     print(box.shape, (box[0] * w, box[1] * h), (box[2] * w, box[3] * h))
        #     cv2.rectangle(img_show, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)), (255, 255, 0), 1)
        # plt.figure()
        # plt.imshow(img_show)
        # plt.show()

        if self.transform:
            image = self.transform(image)

        # create target for 3 prediction scales with 3 anchors for each one
        # 6 : [prob_obj, x, y, scale_w, scale_h, T, I, A]
        c, h, w = image.shape
        targets = [torch.zeros((self.num_anchors // 3, h // S, w // S, 5 + 1)) for S in self.S]
        for idx, box in enumerate(bboxes):
            iou_anchors = iou_width_height(torch.tensor(box[2:]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height = box # normalized coordination, ground truth
            has_anchor = [False] * 3

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx] # receptive field
                i, j = int(S * y), int(S * x) # h//S receptive Field
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i # 偏离感受野中心点大小。 原图中每一个点都被某一个feature map中的点的感受野包含，int(S * y), 将原图坐标映射到feature map中，int(S*y)则对应感受野中心， 若 S*y小数点后有数，则代表偏离感受野中心距离。
                    width_cell, height_cell = (S * width, S * height) # 将原图中bounding box的长和宽映射到feature map中，它中每一个点都代表者一个感受野，则映射后的宽和长即为感受野的伸缩倍数。
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5:] = np.argmax(labels)
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1
        return image, tuple(targets)


def test(image, target):
    gt_box = []
    for y, anchor, recep_field in zip(target, anchors, config.GridSize):
        # [batch_size, 3 * h * w, (score, x, y, w, h, num_classes)]
        boxes = cells_to_bboxes(y, anchor, recep_field, is_preds=False)
        index = np.where(boxes[..., 1:2] == 1)
        boxes = boxes[index[:-1]]
        gt_box += boxes[...,2:]

    # for box in gt_box:
    #     box = convert_box(box)
    #     print(box)
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 1)

    # plt.figure(figsize=(15, 7))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.show()


def main():
    assert table is not None
    dataset = YoloDataset(table, anchors, shuffle=True)
    idx = np.random.randint(0, len(dataset))
    image, target = dataset[idx]
    test(image, target)


if __name__ == "__main__":
    table = table.dropna()
    main()
