import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
import Config as config
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import random
from utils import introIndex


def flatten(tensor):
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # (B, C, D, H, W) -> (C, B, D, H, W)
    transposed = tensor.permute(axis_order)
    # (C, B * D * H * W)
    return transposed.contiguous().view(C, -1)


def Iou(pred, mask):
    if pred.is_cuda:
        pred = pred.cpu().numpy()
    if mask.is_cuda:
        mask = mask.cpu().numpy()
    output = np.zeros_like(pred)
    output[pred > 0.5] = 1
    output = output.reshape(-1, 1)
    mask = mask.reshape(-1, 1)
    intersect = sum(output.transpose(1, 0) @ mask)
    area = sum(output + mask) - intersect
    return ((intersect + config.eps) / (area + config.eps)).item()


def dice(input, target, eps=1e-4):
    assert input.size() == target.size()
    # input: (N, C, D, H, W) -> (C, N * D * H * W)
    input = flatten(input)
    target = flatten(target)
    intersect = (input * target).sum(-1)
    denominator = input.sum(-1) + target.sum(-1)
    return (2 * intersect + eps) / (denominator + eps)


def dice_coef_loss(y_pred, y_true, smooth=1e-6):
    y_pred_f = y_pred.contiguous().view(-1)
    y_true_f = y_true.contiguous().view(-1)

    intersection = (y_pred_f * y_true_f).sum()

    A_sum = torch.sum(y_pred_f * y_pred_f)
    B_sum = torch.sum(y_true_f * y_true_f)

    loss = 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))
    return loss


def bal_bce(input, target, ratio=3, reduction='mean'):
    if ratio is not None:
        idx_mat = introIndex(target, ratio)
    else:
        idx_mat = torch.ones_like(target).float()

    # weight =
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float().mul(idx_mat)
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    total_num = num_pos + num_neg
    pos_w = num_neg / total_num
    pos_n = 1.1 * num_pos / total_num
    weight = pos_w * pos + pos_n * neg

    ans = F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=reduction).mul(idx_mat)
    if reduction == 'mean':
        return ans.sum() / idx_mat.sum() if ratio is not None else ans.mean()
    if reduction == 'sum':
        return ans.sum()
    return ans


def dice_loss(input, target, eps=1e-4):
    assert input.size() == target.size()
    # ans = 1 - dice(input, target, eps)
    ans = dice_coef_loss(input, target, eps)
    return ans.squeeze(0)


def bce_focal_loss(input, mask, gamma=2.0, eps=1e-6, alpha=None, reduction="mean"):
    if alpha is None:
        alpha = torch.tensor([1, 1]).cuda()
    else:
        alpha = torch.tensor([alpha, 1 - alpha]).cuda()

    prods = flatten(input)
    target = flatten(mask)
    prods = torch.sigmoid(prods)
    target = target.type_as(prods)
    pt = (1 - prods) * target + prods * (1 - target)
    focal_weight = (alpha[0] * target + alpha[1] * (1 - target)) * torch.pow(pt + eps, gamma)

    assert input.shape == mask.shape, "The shape of input doesn't match mask"
    loss = F.binary_cross_entropy_with_logits(input, mask) * focal_weight

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

    def forward(self, input, target):
        input = self.normalizer(input)
        return self.custom_loss(input, target)


class DiceLoss(_AbstractLoss):
    def __init__(self, normalizer='sigmoid'):
        super(DiceLoss, self).__init__(name='Dice Loss', normalizer=normalizer)

    def custom_loss(self, input, target, *kwargs):
        return dice_loss(input, target)


class FocalLoss(_AbstractLoss):
    def __init__(self, alpha=None, gamma=2, eps=1e-6, reduction='mean', normalizer='none'):
        super(FocalLoss, self).__init__(name='Focal Loss', normalizer=normalizer)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction

    def custom_loss(self, input, target, *kwargs):
        return bce_focal_loss(input, target, alpha=None, gamma=self.gamma, reduction=self.reduction, eps=self.eps)


class BCELoss(_AbstractLoss):
    def __init__(self):
        super(BCELoss, self).__init__(name='Balance CrossEntropy', normalizer='sigmoid')

    def custom_loss(self, input, target, ratio=3, reduction='mean'):
        return bal_bce(input, target, ratio=ratio, reduction=reduction)


class BCEDice(_AbstractLoss):
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6, reduction='mean', normalizer='sigmoid'):
        super(BCEDice, self).__init__(name='BCEDice',normalizer=normalizer)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.bce = bal_bce
        self.dice = DiceLoss()

    def custom_loss(self, input, target, *kwargs):
        bceloss = self.bce(input, target, ratio=None)
        diceloss = self.dice(input, target)
        self.name = 'Bce:' + str(bceloss.item()) + '-dice:' + str(diceloss.item())
        return self.alpha * bceloss + self.beta * diceloss


class FocalDice(_AbstractLoss):
    def __init__(self, alpha=1, beta=1, f_alpha=None, f_gamma=2, eps=1e-6, reduction='mean', normalizer='sigmoid'):
        super(FocalDice, self).__init__(normalizer=normalizer)
        self.alpha = alpha
        self.beta = beta
        self.f_alpha = f_alpha
        self.f_gamma = f_gamma
        self.eps = eps
        self.reduction = reduction
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.bal_bce = BCELoss()

    def custom_loss(self, input, target, *kwargs):
        f_ans = self.focal(input, target)
        d_ans = self.dice(input, target)
        if torch.isnan(f_ans + d_ans):
            self.name = self.bal_bce.name
            return bal_bce(input, target)
        else:
            self.name = self.focal.name + ":" + str(f_ans.item())+" & " + self.dice.name + ":" + str(d_ans.item())
        return self.alpha * f_ans + self.beta * d_ans


def test():
    input = torch.rand(2, 1, 256, 256).cuda()
    target = torch.randint(0, 2, (2, 1, 256, 256)).float().cuda()
    loss = bal_bce(input, target, ratio=3, reduction='mean')
    print(loss)


if __name__ == "__main__":
    test()
