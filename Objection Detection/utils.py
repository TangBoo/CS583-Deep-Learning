import pdb

import torch as t
from torch import nn as nn
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
import os
import cv2
import Config as config
from DataProcess import get_ImgPath
from einops import repeat, rearrange
import random
import visdom
import time
import glob

features = {}


def vis_image(img, ax=None):
    if t.is_tensor(img):
        img = img.numpy()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype('np.uint8'))
    return ax


class Visualizer(object):
    def __init__(self, env="Detection", port=8098, **kwargs):
        self.vis = visdom.Visdom(env=env, port=port, **kwargs)
        self.index = {}
        self.train_log_text = ''
        self.val_log_text = ''

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def show_anchor(self, name, imgs):
        # img: [b, c, h, w]
        for i in range(imgs.shape[0]):
            img = imgs[i]
            # print(img.shape)
            self.vis.images(img * 255, win=name, opts=dict(title=name))

    def imgs(self, name, img_, **kwargs):
        # [batch, channel, d, h, w]
        size = img_.shape
        if len(size) == 5:
            idx0 = np.random.randint(0, size[0], size=(1,))
            idx1 = np.random.randint(0, size[1], size=(8,))
            idx2 = np.random.randint(0, size[2], size=(8,))
            img = np.expand_dims(img_.cpu().numpy()[idx0, idx1, idx2, ...], axis=1)
            self.vis.images(img, win=name, nrow=4, opts=dict(title=name))
        else:
            idx0 = np.random.randint(0, size[0], size=(1,))
            idx1 = np.random.randint(0, size[1], size=(8,))
            img = np.expand_dims(img_.cpu().numpy()[idx0, idx1, ...], axis=1)
            self.vis.images(img, win=name, nrow=4, opts=dict(title=name))

    def videos(self, name, imgs_, **kwargs):
        size = imgs_.shape
        idx0, idx1 = 0, 0
        if size[0] > 1:
            idx0 = np.random.randint(0, size[0], size=(1,))
        if size[1] > 1:
            idx1 = np.random.randint(0, size[1], size=(1,))
        imgs_ = imgs_.cpu().numpy()[idx0, idx1, ...].swapaxes(1, 0)
        self.vis.video(tensor=imgs_, win=name)

    def img(self, name, img_, **kwargs):
        size = img_.shape
        idx0, idx1, idx2 = 0, 0, 0
        if size[0] > 1:
            idx0 = np.random.randint(0, size[0], size=(1,))
        if size[1] > 1:
            idx1 = np.random.randint(0, size[1], size=(8,))
        if size[2] > 1 and size == 5:
            idx2 = np.random.randint(0, size[2], size=(8,))
        if size == 5:
            self.vis.image(img_.cpu().numpy()[idx0, idx1, idx2, ...], win=name, opts=dict(title=name), **kwargs)
        else:
            self.vis.image(img_.cpu().numpy()[idx0, idx1, ...], win=name, opts=dict(title=name), **kwargs)

    def show_boxes(self, name, img_, box):
        ax = vis_image(img_)
        if len(box) == 0:
            return ax

        for i, bb in enumerate(box):
            xy = (bb[1], bb[0])
            height = bb[2] - bb[0]
            width = bb[3] - bb[1]
            ax.add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor='red', linewidth=1))
        self.img(name, ax)

    def log(self, info, isTrain=True):
        if isTrain:
            self.train_log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
            self.vis.text(self.train_log_text, win="Train log")
        else:
            self.val_log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
            self.vis.text(self.val_log_text, win="Validation Log")

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=name
                      , opts=dict(title=name), update=None if x == 0 else 'append', **kwargs)
        self.index[name] = x + 1

    def img_many(self, d):
        for k, v in d.items():
            self.imgs(k + ': ' + str(v.shape), v * 256)

    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)

    def __getattr__(self, name):
        return getattr(self.vis, name)


def generator_bbox(mask, mode='box', expFactor=1):
    """
    :param mask:
    :param mode: 'center' or 'box'
    :param expFactor: expand the height,width of the anchor box for easier semantic segmentation
    :return: [left_bottom_y, left_bottom_x, right_top_y, right_top_x] or [cnt_y, cnt_x, height, width]
    """
    if mode not in ['center', 'box']:
        raise ValueError

    if mask.shape[0] == 1:
        mask = mask.squeeze(0)

    if t.is_tensor(mask):
        mask = mask.numpy()

    bbox = find_bbox(mask)
    ans = np.zeros((len(bbox), 4))
    for idx, box in enumerate(bbox):
        ymin = box[0] - 1
        xmin = box[1] - 1
        height = box[2] + 1
        width = box[3] + 1

        if expFactor != 1:
            cnt_y = ymin + height / 2
            cnt_x = xmin + width / 2
            height = expFactor * height
            width = expFactor * width
            ymin = cnt_y - height / 2
            xmin = cnt_x - width / 2

        if mode == 'center':
            ans[idx, :] = [ymin + height / 2, xmin + width / 2, height, width]
        else:
            ans[idx, :] = [ymin + height, xmin, ymin, xmin + width]
    return ans


def cnt2box(bbox):
    """
    :param bbox:[num, 4]
    :return: [cnt_y, cnt_x, height, width] -> [ymax, xmin, ymin, xmax]
    """
    if isinstance(bbox, list):
        raise ValueError

    if t.is_tensor(bbox):
        ans = t.zeros(len(bbox), 4).to(bbox.device)
        typef = t.tensor
    else:
        ans = np.zeros((len(bbox), 4))
        typef = np.array

    for idx, box in enumerate(bbox):
        cnt_y, cnt_x = box[:2]
        height, width = box[2:]
        ans[idx, :] = typef([cnt_y + height / 2, cnt_x - width / 2, cnt_y - height / 2, cnt_x + width / 2])
    return ans


def box2cnt(bbox):
    """
    :param bbox:[num, 4]
    :return: [leftTop_y, leftTop_x, rightBottom_y, rightBottom_x] -> [cnt_y, cnt_x, height, width]
    """
    if isinstance(bbox, list):
        raise ValueError

    if t.is_tensor(bbox):
        ans = t.zeros((len(bbox), 4)).to(bbox.device)
        typef = t.tensor
    else:
        ans = np.zeros((len(bbox), 4))
        typef = np.array

    for idx, box in enumerate(bbox):
        ymax, xmin = box[:2]
        ymin, xmax = box[2:]
        height = ymax - ymin
        width = xmax - xmin
        ans[idx, :] = typef([ymax - height / 2, xmin + width / 2, height, width])
    return ans


def boxMat2cntMat(bbox_label):
    """
    :param bbox_label: [ymax, xmin, ymin, xmax]
    :return: bbox label in center type coordination
    Describ:
        it is used to convert to bbox label matrix in box type coordinate to center type coordination
    """
    bbox_label_ = bbox_label.copy()
    ymax = bbox_label[..., 0:1]
    xmin = bbox_label[..., 1:2]
    ymin = bbox_label[..., 2:3]
    xmax = bbox_label[..., 3:]

    height = ymax - ymin
    width = xmax - xmin

    bbox_label_[..., 0:1] = ymax - height / 2
    bbox_label_[..., 1:2] = xmin + width / 2
    bbox_label_[..., 2:3] = height
    bbox_label_[..., 3:] = width

    return bbox_label_.clip(0, config.Image_shape[0])


def cntMat2boxMat(bbox_label):
    """
    :param bbox_label: [9, h, w, 4], 4: [cnt_y, cnt_x, h, w]
    :return: bbox label in box type coordination, [cnt_y, cnt_x, h, w] -> [ymax, xmin, ymin, xmax]
    Describ:
        it is used to convert bbox label matrix in center type coordination to box type coordination
    """
    bbox_label_ = bbox_label.copy()
    cnt_y = bbox_label[..., 0:1]
    cnt_x = bbox_label[..., 1:2]
    height = bbox_label[..., 2:3]
    width = bbox_label[..., 3:]
    bbox_label_[..., 0:1] = cnt_y + height / 2
    bbox_label_[..., 1:2] = cnt_x - width / 2
    bbox_label_[..., 2:3] = cnt_y - height / 2
    bbox_label_[..., 3:] = cnt_x + width / 2

    return bbox_label_.clip(0, config.Image_shape[0])


def bbox2show(bbox, mode='center'):
    """
    :param bbox:[num, 4]
    :param mode: 'center' or 'box'
    :return: center:[cnt_y, cnt_x, height, width] -> [ymin, xmin, height, width] or box:[ymax, xmin, ymin, xmax] -> [ymin, xmin, height, width]
    """
    if mode not in ['center', 'box']:
        raise ValueError

    if isinstance(bbox, list):
        bbox = np.array(bbox)

    if t.is_tensor(bbox):
        bbox = bbox.numpy()

    ans = np.zeros((len(bbox), 4))
    for idx, box in enumerate(bbox):
        if mode == 'center':
            cnt_y, cnt_x = box[:2]
            height, width = box[2:]
            ans[idx, :] = [int(cnt_y - height / 2), int(cnt_x - width / 2), height, width]
        else:
            ymax, xmin = box[:2]
            ymin, xmax = box[2:]
            height = ymax - ymin
            width = xmax - xmin
            ans[idx, :] = [ymin, xmin, height, width]
    return ans


def iou(gtBox, predBox):
    """
    :param gtBox: [4,]
    :param predBox: eg. [c, h, w, 4] 4:[ymax, xmin, ymin, xmax]
    :return: [c, h, w, 1], being careful about both (ymax - ymin) and (xmax - xmin) less than 0. In that case, interoverlape should be zero
    """
    if type(gtBox) != type(predBox):
        raise ValueError(type(gtBox).__name__, type(predBox).__name__, "Not Match")

    maximum = t.maximum if t.is_tensor(gtBox) else np.maximum
    minimum = t.minimum if t.is_tensor(gtBox) else np.minimum

    # overlap:
    delta_h = (minimum(gtBox[..., 0:1], predBox[..., 0:1]) - maximum(gtBox[..., 2:3], predBox[..., 2:3])).clip(min=0)
    delta_w = (minimum(gtBox[..., 3:], predBox[..., 3:]) - maximum(gtBox[..., 1:2], predBox[..., 1:2])).clip(min=0)
    interoverlap = delta_h * delta_w

    # area:
    area1 = (predBox[..., 0:1] - predBox[..., 2:3]) * (predBox[..., 3:] - predBox[..., 1:2])
    area2 = (gtBox[..., 0:1] - gtBox[..., 2:3]) * (gtBox[..., 3:] - gtBox[..., 1:2])
    union = area1 + area2 - interoverlap
    ans = interoverlap / (union + config.eps)
    return ans


def iou_mat(gtBox, predBox, predmode='center', gtmode='box'):
    """
    :param mode:
    :param gtmode: it should be box
    :param predmode: it should be box.
    :param gtBox: ground truth:[ymax, xmin, ymin, xmax], size: [num, 4]
    :param predBox: eg. [9, h, w, (cnt_y, cnt_x, h, w)]
    :return: iou
    """
    if predmode not in ['center', 'box']:
        raise ValueError

    if isinstance(gtBox, list) or isinstance(predBox, list):
        raise ValueError

    if predmode == 'center':
        predBox = cntMat2boxMat(predBox)  # mode = 'box'

    if gtmode == 'center':
        gtBox = cntMat2boxMat(gtBox)

    f_c, f_h, f_w, _ = predBox.shape
    num_box = gtBox.shape[0]

    if t.is_tensor(predBox):
        ans = t.zeros((num_box, f_c, f_h, f_w, 1)).to(predBox.device)
    else:
        ans = np.zeros((num_box, f_c, f_h, f_w, 1))

    for ii in range(num_box):
        ans[ii, :] = iou(gtBox[ii], predBox)
    return ans


def point2box(img_shape, feature_shape, bbox):
    """
    :param bbox: eg. [3, 3, 2]
    :param img_shape: [h, w]
    :param feature_shape: [f_h, f_w]
    :return: eg. bbox_label[9, f_h, f_w, 4]
    """
    h, w = img_shape
    if len(bbox.shape) > 3:
        raise ValueError('error: no allow of dimension of bbox bigger than 3')

    if len(bbox.shape) > 2:
        bbox = rearrange(bbox, 'h w c->(h w) c')

    bbox_shape = (len(bbox),) + feature_shape + (4,)  # eg. [9, h, w, 4]
    bbox_label = np.zeros(bbox_shape)
    f_h, f_w = feature_shape

    # ---------------Mapping-----------------
    anchor_stride_h = h / f_h
    anchor_stride_w = w / f_w
    y = np.arange(0 + 0.5, feature_shape[0] + 0.5, 1) * anchor_stride_h - 0.5  # [1,shape[0]//anchor_stride]
    x = np.arange(0 + 0.5, feature_shape[1] + 0.5, 1) * anchor_stride_w - 0.5

    # y = np.arange(0.5, h + 0.5, anchor_stride_h) - 0.5
    # x = np.arange(0.5, w + 0.5, anchor_stride_w) - 0.5
    x, y = np.meshgrid(x, y)

    coor = np.dstack((y, x))
    bbox_label[..., :2] = repeat(coor, 'h w c-> n h w c', n=len(bbox))
    bbox_label[..., 2:] = repeat(bbox, 'n c -> n h w c', h=f_h, w=f_w)
    return bbox_label


def bbox2label(feature_shape, img_shape, gtbox, bbox, pos_idx, iouIndex, gtmode='box', predmode='center',
               Normalizer=False):
    """
    :param predmode: its mode should be center
    :param gtmode: its mode should be center
    :param Normalizer: not useful in bounding box regression
    :param pos_idx: point out all box will join in bounding box regression
    :param iouIndex: the index of ground truth box for each box with biggest iou
    :param img_shape: [h, w]
    :param feature_shape: [f_h, f_w]
    :param gtbox:[num, 4], 4:[cnt_y, cnt_x, h, w], not normalized
    :param bbox: eg.[9, f_h, f_w, 4], not normalized
    :return: label in bounding box regression, [9, h, w, 4] 4:[t_y, t_x, t_h, t_w]
    x = (gt_x - box_x)/f_w, y: pass
    h: log(gt_h / box_h), w: pass
    """
    if gtmode not in ['box', 'center']:
        raise ValueError

    if len(pos_idx) != len(bbox.shape):
        raise ValueError
    else:
        pos_idx_ = pos_idx[:-1]

    h, w = img_shape if len(img_shape) == 2 else img_shape[1:]
    f_h, f_w = feature_shape
    # ------------Normalize box-------------------
    if Normalizer:  # do not do normalization
        bbox = normalizer(bbox, (h, w))
        gtbox = normalizer(gtbox, (h, w))

    if gtmode == 'box':
        gtbox = box2cnt(gtbox)

    if predmode == 'box':
        bbox = boxMat2cntMat(bbox)

    b_c, b_h, b_w, _ = bbox.shape
    gtboxMat = repeat(gtbox, 'n v -> n c h w v', c=b_c, h=b_h, w=b_w)
    if len(iouIndex.shape) != len(gtboxMat.shape):
        iouIndex = np.expand_dims(iouIndex, axis=0)

    gtboxMat_ = np.take_along_axis(gtboxMat, iouIndex, axis=0).squeeze(0)
    ans = np.zeros_like(bbox)
    pos_box = bbox[pos_idx_]
    pos_gt = gtboxMat_[pos_idx_]
    pos_box[..., :2] = (pos_gt[..., :2] - pos_box[..., :2]) / (f_h, f_w)
    pos_box[..., 2:] = np.log(pos_gt[..., 2:] / pos_box[..., 2:])
    ans[pos_idx_] = pos_box

    return ans


def create_box(area_size, area_ratios, hw_ratios):
    """
    :param area_size: list: get from kmeans with ncluster = 3
    :param area_ratios: [2**0, 2**(1/2), 2**(2/3)]
    :param hw_ratios: [1:2, 1:1, 2:1]
    :return: The height and width of box, [len(area_size), 3, 3, 2]
    """
    if isinstance(area_size, list):
        area_size = np.array(area_size)
    if isinstance(area_ratios, list):
        area_ratios = np.array(area_ratios)
    if isinstance(hw_ratios, list):
        hw_ratios = np.array(hw_ratios)

    areas = repeat(area_size, 'h -> c h', c=len(area_ratios))
    area_ratios_ = repeat(area_ratios, 'h->h c', c=len(area_size))
    areas_ = areas * area_ratios_
    boards = np.sqrt(areas_)
    boards_ = repeat(boards.T, 'h w->h w c', c=len(hw_ratios))
    hw_ratios_ = repeat(hw_ratios, 'h->c h', c=len(area_ratios))
    box_w = boards_ * hw_ratios_
    box_h = boards_
    boxes_shape = (len(area_size),) + (len(area_ratios), len(hw_ratios)) + (2,)
    boxes = np.dstack((box_h, box_w)).reshape(boxes_shape)
    return boxes


def create_idx(iouMat, neg_threshold=0.3, pos_threshold=0.5, ratio=2):
    """
    :param iouMat: [9, f_h, f_w, 1]
    :param neg_threshold: those are smaller than this will be negative
    :param pos_threshold: those are bigger than this will be positive
    :param ratio: Positive / Negative
    :return:
        idx_max: [9, f_h, f_w, 1], point out which cases will be contribute to backward signal
        pos_idx: the index of positive cases
    """

    iouIdx = np.expand_dims(np.argmax(iouMat, axis=0), axis=0)
    iouMat_MaxIou = np.take_along_axis(iouMat, iouIdx, 0).squeeze(0)

    pos_idx = np.where(iouMat_MaxIou >= pos_threshold)
    neg_idx = np.where(iouMat_MaxIou < neg_threshold)
    num_pos = len(pos_idx[0])
    num_neg = len(neg_idx[0])
    random_idx = random.sample(range(num_neg), min(num_pos * ratio, num_neg))
    neg_idx_ = tuple(tp[random_idx] for tp in neg_idx)
    idx_mat = np.zeros_like(iouMat_MaxIou)
    idx_mat[pos_idx] = 1

    # ------------filter Omit Anchors-------------
    iouIdx = iouIdx.squeeze(0)
    iouIdx_ = iouIdx.copy()
    iouIdx += 1
    iouIdx[neg_idx] = 0
    satAnchor = [np.sum(iouIdx == i + 1) for i in range(len(iouMat))]
    for i in range(len(iouMat)):
        if satAnchor[i] >= config.LowestNum_Anc:
            continue
        for threshold in np.arange(pos_threshold, neg_threshold, -0.1):
            locSecBest = np.where(idx_mat != 1) and np.where(iouMat[i] > threshold)
            totalNum = len(locSecBest[0])
            if totalNum == 0:
                continue
            random_secIdx = random.sample(range(totalNum), min(config.LowestNum_Anc, totalNum))
            secBestIdx = tuple(tp[random_secIdx] for tp in locSecBest)
            idx_mat[secBestIdx] = 1
            iouIdx_[secBestIdx] = i
            if totalNum >= config.LowestNum_Anc:
                break
    pos_idx_ = np.where(idx_mat == 1)
    idx_mat[neg_idx_] = 1
    return idx_mat, pos_idx_, iouIdx_


def mapAnc2Img(ancMats, image, threshold=0.5):
    # ancMat = t.sigmoid(ancMat).numpy()
    if t.is_tensor(image):
        image = image.cpu().numpy()
    else:
        image = image.copy()
    b, c, h, w = image.shape
    for j in range(config.Output_features):
        if t.is_tensor(ancMats[j]):
            ancMat = t.sigmoid(ancMats[j]).cpu().numpy()
        else:
            ancMat = ancMats[j]

        for i in range(b):
            anc_c, anc_h, anc_w, _ = ancMat[i].shape
            stride_h = h / anc_h
            stride_w = w / anc_w
            y = np.arange(0 + 0.5, anc_h + 0.5, 1) * stride_h - 0.5
            x = np.arange(0 + 0.5, anc_w + 0.5, 1) * stride_w - 0.5
            y = y.clip(0, h)
            x = x.clip(0, w)
            x, y = np.meshgrid(x, y)
            coor = np.dstack((y, x))
            coor_ = repeat(coor, 'h w c->n h w c', n=anc_c)
            loc = np.where(ancMat[i] >= threshold)[:-1]
            coor_ = coor_[loc]
            y = np.array([int(i) for i in coor_[..., 0]], dtype='int')
            x = np.array([int(j) for j in coor_[..., 1]], dtype='int')
            for k in range(c):
                image[i][k][x, y] = 255 / (k + 1)
    return image


def TypeConverter(mat, mode):
    if mode not in ['numpy', 'tensor', 'ndarray']:
        raise ValueError

    if mode == 'numpy':
        mode = 'ndarray'

    if type(mat).__name__.lower() != mode:
        if mode == 'tensor':
            mat = t.from_numpy(mat)
        else:
            mat = mat.numpy()
    return mat


def myStr(tensor):
    return str(t2i(tensor))


def t2i(tensor):
    if t.is_tensor(tensor):
        return tensor.item()
    else:
        return tensor


def normalizer(box, divisor):
    if isinstance(box, list):
        box = np.array(box)

    if box.shape[-1] != len(divisor):
        raise ValueError("Divisor is not match for box in last dimension")

    return box / divisor


def map2Image(box, multiplier):
    if isinstance(box, list):
        box = np.array(box)

    if box.shape[-1] != len(multiplier):
        raise ValueError("Divisor is not match for box in last dimension")

    return box * multiplier


def find_bbox(mask):
    """
    :param mask:Ground Truth
    :return: [left_bottom_y, left_bottom_x, height, width]
    """
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    stats = stats[stats[:, 4].argsort()]
    return stats[:-1]


def read_image_2D(case_path):
    name = case_path.split('/')[-1]
    patient, _, case, slc = name.split('/')[-1].split('_')[0:4]
    mask_path = os.path.join(config.Mask_root, "{}_case_{}_{}_AifMask.png".format(patient, case, slc))
    image = cv2.imread(case_path, 0)
    mask = cv2.imread(mask_path, 0)
    return image, mask, None


def randomIdx(lst):
    return np.random.randint(len(lst))


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


def throw_exception(bboxes, gt_boxes, anchors=None):
    """
    :param anchors: [9, h, w, 1]
    :param bboxes: [9, h,, w, (x, y, bh, bw)]
    :param gt_boxes: [num, (ymax, xmin, ymin, xmax)]
    :return: exceptional bounding box
    """
    if anchors is not None:
        posIdx = np.where(anchors == 1)
        bboxes = bboxes[posIdx]

    shape = bboxes.shape[:-1] + (1,)
    throw_out = np.zeros(shape)
    for idx, gtBox in enumerate(gt_boxes):
        throw_out += iou(gtBox, bboxes)
    excepIndex = np.where(throw_out == 0)
    print(len(excepIndex[0]))
    return


def get_output(name):
    def hook(model, input, output):
        features[name] = output.data

    return hook


def get_pth(model, checkpoint):
    def func(path):
        data_time = path.split('/')[-1].split('_')[1:]
        key = "{}_{}".format(data_time[-2], data_time[-1])
        return key

    if isinstance(model, nn.DataParallel):
        model = model.module
    model_name = model.__str__().split("\n")[0][:-1]
    pth_lst = glob.glob(checkpoint + '/*.pth')
    coll_pth = []
    for item in pth_lst:
        sub_name = item.split('/')[-1].split('_')[0][1:-1].split(' ')[-1][1:-1].split('.')[-1]
        if sub_name == model_name:
            coll_pth.append(item)
    coll_pth.sort(key=func)
    return coll_pth[-1]


def nms(bboxes, anchor, threshold=0.5, mode='box'):
    mode = mode.lower()
    if mode not in ['center', 'box']:
        raise ValueError

    if mode == 'center':
        bboxes = cnt2box(bboxes)

    if not len(anchor):
        return
    ans = []
    sortIdx = t.argsort(anchor, descending=True, dim=0)
    priQueue = t.index_select(bboxes, index=sortIdx.view(-1), dim=0)

    while len(priQueue) != 0:
        top = priQueue[0]
        priQueue = priQueue[1:]
        if len(priQueue) == 0:
            break
        priQueue = priQueue[t.where(iou(top, priQueue) < threshold)[:-1]]
        ans.append(top)

    return t.stack(ans)


def main(mode='center'):
    if mode not in ['center', 'box']:
        raise ValueError

    img_lst = get_ImgPath()
    idx = randomIdx(img_lst)
    img, mask, _ = read_image_2D(img_lst[idx])

    plt.figure()
    ax = plt.axes()
    plt.imshow(mask, cmap='bone')
    bbox = generator_bbox(mask, mode=mode, expFactor=1)

    bbox = bbox2show(bbox, mode=mode)
    for box in bbox:
        topright = (box[0], box[1])
        height = box[2]
        width = box[3]
        rect = patches.Rectangle(topright, height, width, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    predbox = np.random.rand(16, 16, 4)
    gtbox = np.random.rand(4, 4)
    ans = iou_mat(gtbox, predbox)
    print(ans.shape)
    return


def test():
    image = t.rand(2, 3, 256, 256).cuda()
    ancMats = [t.rand(2, 9, 128, 128, 1).cuda() for i in range(config.Output_features)]
    res = mapAnc2Img(ancMats, image)
    print(res.shape)


if __name__ == "__main__":
    test()
