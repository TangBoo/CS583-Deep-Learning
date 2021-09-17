import torch
import torch as t
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable as V
import numpy as np
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

import Config as config


def Iou_loss(pred, mask, eps=1e-6):
    pred_ = F.sigmoid(pred)
    output = torch.zeros_like(pred).to(pred.device)
    output[pred_ > 0.5] = 1
    output = output.reshape(-1, 1)
    mask = mask.reshape(-1, 1)
    intersect = sum(output * mask)
    area = output.sum() + mask.sum() - intersect
    return ((intersect + eps) / (area + eps)).item()


def Dice_coef_loss(y_pred, y_true, index=None):
    if index is None:
        index = list()

    if len(index) != 0:
        if all(index) == 0:
            return 0
        y_pred = y_pred[t.where(index == 1)]
        y_true = y_true[t.where(index == 1)]

    y_pred = y_pred.contiguous().view(-1, 1)
    y_true = y_true.contiguous().view(-1, 1)
    assert y_pred.shape == y_true.shape, "Input shape {}, Label shape {}, Not match".format(y_pred.shape, y_true.shape)

    intersection = (y_pred * y_true).sum()
    A_sum = t.sum(y_pred * y_pred)
    B_sum = t.sum(y_true * y_true)
    # print("A_sum shape:", A_sum.shape, "B_sum shape:", B_sum.shape, "intersection shape:", intersection.shape)
    loss = 1 - ((2 * intersection + config.eps) / (A_sum + B_sum + config.eps))
    return loss


def Focal_loss(input, anchors, index=None, gamma=2.0, eps=1e-6, alpha=None, reduction='mean'):
    if reduction not in ['mean', 'sum']:
        raise ValueError

    if index is None:
        index = list()

    if alpha is None:
        alpha = t.tensor([1, 1]).cuda()
    else:
        alpha = t.tensor([alpha, 1 - alpha]).cuda()

    if len(index) != 0:
        loss_idx = t.where(index == 1)
        input = input[loss_idx].reshape(-1, 1)
        anchors = anchors[loss_idx].reshape(-1, 1)
    else:
        input = input.contiguous().view(-1, 1)
        anchors = anchors.contiguous().view(-1, 1)

    prob_ = input.clone()
    target = anchors.clone()
    prob = t.sigmoid(prob_)
    target = target.type_as(prob)
    pt = (1 - prob) * target + prob * (1 - target)
    focal_weight = (alpha[0] * target + alpha[1] * (1 - target)) * t.pow(pt + eps, gamma)

    assert input.shape == anchors.shape, "The shape of Input: {} doesn't match Anchor :{}".format(input.shape,
                                                                                                  anchors.shape)
    loss = F.binary_cross_entropy_with_logits(input, anchors, reduction='none') * focal_weight
    if reduction == "mean":
        ans = loss.mean()
        return ans
    elif reduction == "sum":
        return loss.sum()
    return loss


def SmoothLoss(input, bbox_label, index, reduction='sum'):
    # if all(index.flatten()) == 0:
    #     return 0

    val_idx = t.where(index == 1)
    loss = F.smooth_l1_loss(input[val_idx], bbox_label[val_idx], reduction='none')
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class _AbstractLoss(nn.Module):
    def __init__(self, name='Abstract loss', normalizer='sigmoid'):
        super(_AbstractLoss, self).__init__()
        assert normalizer in ['sigmoid', 'softmax', 'none']
        if normalizer == 'sigmoid':
            self.normalizer = nn.Sigmoid()
        elif normalizer == 'softmax':
            self.normalizer = nn.Softmax(dim=1)
        else:
            self.normalizer = lambda x: x
        self.name = name

    def custom_loss(self, input, target, *kwargs):
        raise NotImplementedError

    def forward(self, input, target, *kwargs):
        input = self.normalizer(input)
        return self.custom_loss(input, target, *kwargs)


class BBoxRegLoss(_AbstractLoss):
    def __init__(self):
        super(BBoxRegLoss, self).__init__(name="Bounding Box Regression", normalizer='none')

    def custom_loss(self, input, target, *kwargs):
        return SmoothLoss(input, target, *kwargs)


class AnchorLoss(_AbstractLoss):
    def __init__(self):
        super(AnchorLoss, self).__init__(name="Anchor SubNet Loss", normalizer='none')

    def custom_loss(self, input, target, *kwargs):
        # ans = 0.7 * Focal_loss(input, target, *kwargs) + 0.3 * Dice_coef_loss(t.sigmoid(input), target, None)
        ans = Focal_loss(input, target, *kwargs)
        # ans = Dice_coef_loss(F.sigmoid(input), target, None)
        return ans


class SegLoss(_AbstractLoss):
    def __init__(self):
        super(SegLoss, self).__init__(name="Segmentation Loss", normalizer='sigmoid')

    def custom_loss(self, input, target, *kwargs):
        return Dice_coef_loss(input, target, *kwargs)


def test():

    image = t.randn(2, 1, 128, 128)
    mask = t.randn(2, 1, 128, 128)
    idx = t.randint(0, 2, size=(2,))
    loss = Dice_coef_loss(image, mask, idx)
    print(loss)


if __name__ == "__main__":
    test()
