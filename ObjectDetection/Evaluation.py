import torch as t
from torchnet import meter
import numpy as np
import Config as config
import torch.nn as nn
from utils import iou, iou_mat, nms


def getAUC(anchor, anchorLabel):
    y_axis = []
    x_axis = []
    anchor = t.sigmoid(anchor)
    N = len(np.where(anchorLabel == 0)[0])
    P = np.sum(anchorLabel)
    for threshold in np.arange(0.5, 0.95, 0.05):
        temp = anchor.clone()
        temp[temp >= threshold] = 1
        temp[temp < threshold] = 0
        TP = np.sum(temp[np.where(anchorLabel == 1)])
        FP = np.sum(temp[np.where(anchorLabel == 0)])
        y_axis.append(TP / P)
        x_axis.append(FP / N)
    auc = np.trapz(y_axis, x_axis)
    return x_axis, y_axis, auc


def getPR(gtBox, bbox):
    """
    :param gtBox: ground truth box
    :param bbox: have processed by nms
    :return: x_axis, y_axis, ap
    """
    y_axis = []
    x_axis = []
    for threshold in np.arange(0.1, 0.95, 0.05):
        TP, FP, FN = countTF(gtBox, bbox, threshold)
        x_axis.append(TP/(TP + FN)) # Recall
        y_axis.append(TP/(TP + FP)) # Precision
    ap = np.trapz(y_axis, x_axis)
    return x_axis, y_axis, ap


def PrecisionRecall(gtBox, predBox, threshold=config.PosThreshold):
    TP, FP, FN = countTF(gtBox, predBox, threshold)
    return TP / (TP + FP + config.eps), TP / (TP + FN + config.eps)


def countTF(gtBox, predBox, threshold):
    TP, FP, FN = 0, 0, 0
    if t.is_tensor(predBox):
        where = t.where
    else:
        where = np.where
    for gt in gtBox:
        res = iou(gt, predBox)
        temp = len(where(res > threshold)[0])
        TP += temp
        FP += len(res) - temp
        if temp == 0:
            FN += 1
    return TP, FP, FN


class EvalMatrix(object):
    def __init__(self, normalizer='sigmoid'):
        # TN, FP
        # FN, TP
        normalizer = normalizer.lower()
        assert normalizer in ['sigmoid', 'softmax', 'none']
        if normalizer == 'sigmoid':
            self.normalizer = nn.Sigmoid()
        elif normalizer == 'softmax':
            self.normalizer = nn.Softmax(dim=1)
        else:
            self.normalizer = lambda x: x
        self.numClass = config.Num_class + 1
        self.mtr = meter.ConfusionMeter(self.numClass)
        self.eps = 1e-6

    def pixelAcc(self):
        tn_tp = np.diag(self.mtr.value()).sum()
        total = self.mtr.value().sum()
        return tn_tp / total

    def precision(self):
        prec_speci = np.diag(self.mtr.value()) / self.mtr.value().sum(axis=1)
        return prec_speci[1]

    def sensitive(self):
        #tp/p
        p_num = np.sum(self.mtr.value(), axis=1)
        tp = np.diag(self.mtr.value())
        sensi = tp / p_num
        if self.numClass == 2:
            return sensi[1]
        return sensi

    def mAP(self):
        acc = self.pixelAcc()
        meanAcc = np.nanmean(acc)
        return meanAcc

    def IntersectOverUnion(self):
        intersect = np.diag(self.mtr.value())
        # (TN + FN, FP + TP) + (TN +FP, FN + TP) - (TN, TP)
        union = np.sum(self.mtr.value(), axis=1) + np.sum(self.mtr.value(), axis=0) - np.diag(self.mtr.value())
        IoU = intersect / union
        if self.numClass == 2:
            return IoU[1]
        return IoU

    def mIoU(self):
        miou = np.nanmean(self.IntersectOverUnion())
        return miou

    def genConfusionMat(self, output, label):
        # output:[batch, C, D, H, W]
        output = self.normalizer(output)
        pred = t.zeros_like(output)
        if self.numClass == 2:
            pred[output > 0.5] = 1
        else:
            pred = t.max(output, dim=1)[1]

        pred = pred.flatten()
        label = label.flatten()
        assert pred.shape == label.shape
        self.mtr.add(pred, label)

    def reset(self):
        self.mtr = meter.ConfusionMeter(self.numClass)


if __name__ == "__main__":
    x = t.rand(16, 4).cuda()
    y = t.rand(5, 4).cuda()
    res = PrecisionRecall(x, y)
    print(res)