import torch
from torchnet import meter
import numpy as np
import Config as config
from torch import nn


class SegmentationMetrix(object):
    def __init__(self, normalizer='sigmoid'):
        # TN, FP
        # FN, TP
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
        #(TN + FN, FP + TP) + (TN +FP, FN + TP) - (TN, TP)
        union = np.sum(self.mtr.value(), axis=1) + np.sum(self.mtr.value(), axis=0) - np.diag(self.mtr.value())
        IoU = intersect / union
        if self.numClass == 2:
            return IoU[1]
        return IoU

    def mIoU(self):
        miou = np.nanmean(self.IntersectOverUnion())
        return miou

    def genConfusionMat(self, output, label):
        #output:[batch, C, D, H, W]
        output = self.normalizer(output)

        pred = torch.zeros_like(output)
        if self.numClass == 2:
            pred[output > 0.5] = 1
        else:
            pred = torch.max(output, dim=1)[1]

        pred = pred.flatten()
        label = label.flatten()
        assert pred.shape == label.shape
        self.mtr.add(pred, label)

    def reset(self):
        self.mtr = meter.ConfusionMeter(self.numClass)


def test():
    pred = torch.rand((2, 1, 256, 256)).to(config.Device)
    label = torch.randint(0, 2, size=(2, 1,256, 256))
    confusion_matrix = SegmentationMetrix()
    confusion_matrix.genConfusionMat(pred, label)
    print(confusion_matrix.mIoU())
    print(confusion_matrix.sensitive())
    print(confusion_matrix.precision())
    print(confusion_matrix.pixelAcc())


if __name__ == "__main__":
    test()








