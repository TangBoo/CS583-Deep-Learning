import torch
import torch.nn as nn

import config
from utils import bbox_overlaps_ciou
from dataset import YoloDataset
import Meta as meta
import numpy as np
import copy
import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F

anchors = meta.anchors
table = meta.meta_table
recep_field = config.RECEP_FIELD


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, prediction, target, anchors):
        '''
        Input:
            1. prediction:[batch_size, anchors, height, width, 5 + num_classes] 5 is [pred_class, y, x, h, w, num_class]
            2. target : [batch_size, anchors, height, width, 5 + class_label]
            3. anchor : the size of feature map in three scales.
               anchor box on the particular scale of shape (anchors on scale, 2), [1, 3, 1, 1, 2].
        '''
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # Loss for no object
        assert prediction[..., 0][noobj].shape == target[..., 0][noobj].shape, (
                prediction[..., 0][noobj].shape == target[..., 0][noobj].shape)
        no_object_loss = self.bce(prediction[..., 0:1][noobj], target[..., 0:1][noobj])

        # Loss for object
        # Due to computation of score corresponding to iou in original Image,
        # so the width, height in iou computations are correspond to the size of original Image.
        h, w = prediction.shape[2:4]
        anchors = tensor(anchors).reshape((1, 3, 1, 1, 2)).to(config.DEVICE, dtype=torch.float) # [3, 2]->[1, 3, 1, 1, 2]
        anchors[..., 0] *= w
        anchors[..., 1] *= h

        box_preds = torch.cat([self.sigmoid(prediction[..., 1:3]), torch.exp(prediction[..., 3:5]) * anchors], dim=-1)

        # the gradient in ious do not back propagate
        ious = bbox_overlaps_ciou(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(prediction[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # Loss for box coordinates, refer to bounding box regression
        prediction[..., 1:3] = self.sigmoid(prediction[..., 1:3])
        target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))
        box_loss = self.mse(prediction[..., 1:5][obj], target[..., 1:5][obj])

        # CLass Loss
        class_loss = self.entropy((prediction[..., 5:][obj]), (target[..., 5][obj].long()))

        # print('sigmoid',F.logsigmoid(prediction[..., 5:][obj]))
        return self.lambda_box * box_loss \
               + self.lambda_obj * object_loss \
               + self.lambda_noobj * no_object_loss \
               + self.lambda_class * class_loss


def test() -> None:
    dataset = YoloDataset(table, anchors, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    myloss = YoloLoss()
    itr = train_loader.__iter__()
    image, targets = next(itr)
    b_, c_, h_, w_ = image.shape

    fakes = targets.copy()
    for idx, f in enumerate(fakes):
        b, c, h, w, m = f.shape
        index = f.reshape(-1, 6)[..., 5:].type(torch.int64)
        y = torch.zeros((b * c * h * w, 3)).scatter_(-1, index, 1)
        y = y.reshape((b, c, h, w, -1))
        pred = torch.cat([f[..., :5], y], dim=-1)
        pred[..., 3:4] /= w_
        pred[..., 4:5] /= h_
        loss = myloss.forward(pred, targets[idx], anchors[idx])


if __name__ == "__main__":
    test()
