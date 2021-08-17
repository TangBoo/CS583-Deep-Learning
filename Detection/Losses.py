import torch
import torch as t
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable as V
import numpy as np
from torch.nn import MSELoss, SmoothL1Loss, L1Loss


def Dice_coef_loss(y_pred, y_true, smooth=1e-6):
    y_pred_f = y_pred.contiguous().view(-1)
    y_true_f = y_true.contiguous().view(-1)

    intersection = (y_pred_f * y_true_f).sum()

    A_sum = t.sum(y_pred_f * y_pred_f)
    B_sum = t.sum(y_true_f * y_true_f)

    loss = 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))
    return loss


def Focal_loss(input, anchors, index, gamma=2.0, eps=1e-6, alpha=None, reduction='mean'):
    if reduction not in ['mean', 'sum']:
        raise ValueError

    if alpha is None:
        alpha = t.tensor([1, 1]).cuda()
    else:
        alpha = t.tensor([alpha, 1 - alpha]).cuda()

    prob_ = input.contiguous().view(-1, 1)
    target = anchors.contiguous().view(-1, 1)
    index = index.contiguous().view(-1, 1)
    prob = t.sigmoid(prob_)
    target = target.type_as(prob)
    pt = (1 - prob) * target + prob * (1 - target)
    focal_weight = (alpha[0] * target + alpha[1] * (1 - target)) * t.pow(pt + eps, gamma)

    assert input.shape == anchors.shape, "The shape of Input: {} doesn't match Anchor :{}".format(input.shape, anchors.shape)

    loss = F.binary_cross_entropy_with_logits(input, anchors) * focal_weight * index

    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    return loss


def SmoothLoss(input, bbox_label, index, reduction='mean'):
    input = input.contiguous().view(-1, 4)
    bbox_label = bbox_label.contiguous().view(-1, 4)
    index = index.contiguous().view(-1, 1)
    loss = F.smooth_l1_loss(input, bbox_label) * index

    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
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
        super(AnchorLoss, self).__init__(name="Anchor SubNet Loss", normalizer='sigmoid')

    def custom_loss(self, input, target, *kwargs):
        return Focal_loss(input, target, *kwargs)


class SegLoss(_AbstractLoss):
    def __init__(self):
        super(SegLoss, self).__init__(name="Segmentation Loss", normalizer='sigmoid')

    def custom_loss(self, input, target, *kwargs):
        return Dice_coef_loss(input, target)


def test():
    input = t.rand(1, 9, 128, 128, 4).cuda()
    target = t.rand(1, 9, 128, 128, 4).cuda()
    index = t.randint(0, 2, size=(1, 9, 128, 128, 1)).float().cuda()
    loss = BBoxRegLoss()
    res = loss(input, target, index)
    print(res)


if __name__ == "__main__":
    test()