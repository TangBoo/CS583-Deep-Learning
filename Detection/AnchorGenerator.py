import torch

import Config as config
import numpy as np
import torch as t
import matplotlib.pyplot as plt
from matplotlib import patches
from utils import (create_box, create_idx, point2box, iou_mat,
                   bbox2label, get_ImgPath, randomIdx, read_image_2D,
                   generator_bbox, bbox2show, iou, cnt2box, TypeConverter)


def generator_anchor_area(img_shape, gtBoxes, depth=config.Depth, area_size=config.Areas,
                          area_ratios=config.Area_ratios,
                          hw_ratios=config.HW_ratio, Pos_threshold=config.PosThreshold,
                          Neg_threshold=config.NegThreshold,
                          Num_features=config.Output_features):
    """
    :param Num_features: How many feature map will be outputted by backbone network
    :param hw_ratios: h/w [2, 1, 0.5]
    :param gtBoxes: ground truth bounding box, [num, 4]
    :param area_size: list:[area1, area2, area4]
    :param img_shape: [h, w] or [c, h, w]
    :param depth: config.depth, deep of network
    :param area_ratios: [3, h/w]
    :param Pos_threshold: positive threshold
    :param Neg_threshold: negative threshold
    :return:
        anchors: positive and negative labels for all anchor subnetwork,list[[9, f_h, f_w, num_classes],...]
        bbox_labels: bbox reg label, list[[9, h, w, 4],...], 4:[t_y, t_x, t_h, t_w]
        indexes: Which locations will contribute to backward signal, including negative and positive points, list[[9, f_h, f_w, 1]]
    """
    if Num_features >= depth - 1:  # don't need the last output proposal that is for segmentation
        raise ValueError("Depth is not enough for output {} features map".format(Num_features))

    if area_size is not None and Num_features != len(area_size):
        raise ValueError("There are no enough areas size for output features map")

    h, w = img_shape if len(img_shape) == 2 else img_shape[1:]
    feature_size = [(h // j, w // j) for j in config.RecepField]  # eg.[3, 5, 9]

    if isinstance(area_size, list):
        area_size = np.array(area_size)
    if isinstance(area_ratios, list):
        area_ratios = np.array(area_ratios)
    if isinstance(hw_ratios, list):
        hw_ratios = np.array(hw_ratios)

    # ----Create bounding box-----------
    if area_size is None:
        area_size = [config.RecepField[j] ** 2 for j in config.OutputIndex]
    feature_size = [feature_size[i] for i in config.OutputIndex]
    boxes = create_box(area_size, area_ratios, hw_ratios)  # eg. [3, 3, 3, (h, w)]
    bbox_labels = []
    anchors = []
    indexes = []
    bboxIdxes = []
    for i in range(len(area_size)):
        bbox_label = point2box(img_shape, feature_size[i], boxes[i])  # mode = 'center'
        iouMat = iou_mat(gtBoxes, bbox_label, predmode='center', gtmode='box')  # output: [gtbox_num, 9, f_h, f_w, 1]

        # ---------------Create Anchor Label--------------
        index, pos_idx, iouIdx = create_idx(iouMat, neg_threshold=Neg_threshold, pos_threshold=Pos_threshold, ratio=3)
        anchor_shape = (index.shape[0:-1]) + (config.Num_class,)  # 0:pos, 1:neg
        bboxIdx_shape = (index.shape[0:-1]) + (1, )
        anchor = np.zeros(anchor_shape)
        bboxIdx = np.zeros(bboxIdx_shape)
        anchor[pos_idx] = 1
        bboxIdx[pos_idx] = 1
        # ---------------Create box Label------------------
        bbox_label_ = bbox2label(feature_shape=feature_size[i], img_shape=img_shape, gtbox=gtBoxes, bbox=bbox_label,
                                 pos_idx=pos_idx, iouIndex=iouIdx, gtmode='box', predmode='center')

        anchors.append(anchor.astype(float))
        indexes.append(index.astype(float))
        bbox_labels.append(bbox_label_.astype(float))
        bboxIdxes.append(bboxIdx.astype(float))

    return anchors, bbox_labels, indexes, bboxIdxes


def apply_bboxReg_area(bboxMat, anchors, img_shape=config.Image_shape, threshold=config.PosThreshold, area_size=config.Areas,
                       area_ratios=config.Area_ratios, hw_ratios=config.HW_ratio):
    """
    :param normalizer: sigmoid or softmax
    :param mode: pass
    :param hw_ratios: Pass
    :param area_ratios: Pass
    :param area_size: Pass
    :param threshold: threshold for positive
    :param anchors: output anchor subnetwork
    :param bboxMat: list: eg. [9, f_h, f_w, 4] 4:[dy, dx, dh, dw]
    :param img_shape: [h, w]
    :return: the outputs of submodules for bounding box coordination and positive anchor
    Describe:
        dy = p_h(dy) + p_x, dx:pass
        dh = p_h*exp(dh), dw:pass
        North that: There is no dimension for batch, handle that outside this function.
    """
    if not isinstance(bboxMat[0], type(anchors[0])):
        raise ValueError(type(bboxMat[0]), type(anchors[0]), " Type Not Match")

    if len(bboxMat) != len(anchors):
        raise ValueError('ValidIndex is not match for bounding box output')

    if area_size is None:
        area_size = [config.RecepField[j] ** 2 for j in config.OutputIndex]

    proposals = create_box(area_size, area_ratios, hw_ratios)  # [4, 3, 3, 2]
    output_bboxs = []
    output_anchcors = []

    for i in range(len(proposals)):
        output_mat = bboxMat[i]  # eg.[9, f_h, f_w, (delta_y, delta_x, delta_h, delta_w)]
        anchor_ = anchors[i]
        f_h, f_w = output_mat.shape[-3:-1]
        boxCoord = point2box(img_shape, (f_h, f_w), proposals[i])  # eg.[9, f_h, f_w, (y, x, bh, hw)], mode = 'center'

        if torch.is_tensor(anchor_):
            anchor_ = t.sigmoid(anchor_)
            boxCoord = torch.from_numpy(boxCoord).to(output_mat.device)
            exp = torch.exp
            where = torch.where
        else:
            exp = np.exp
            where = np.where

        if output_mat.shape[0] == 1:
            output_mat = output_mat.squeeze(0)

        if anchor_.shape[0] == 1:
            anchor_ = anchor_.squeeze(0)

        if output_mat.shape != boxCoord.shape or len(output_mat.shape) != len(boxCoord.shape):
            raise ValueError('Output feature map is not match for label size')

        # y, x, h, w
        output_mat[..., :1] = output_mat[..., :1] * boxCoord[..., 2:3] + boxCoord[..., :1]
        output_mat[..., 1:2] = output_mat[..., 1:2] * boxCoord[..., 3:] + boxCoord[..., 1:2]
        output_mat[..., 2:3] = boxCoord[..., 2:3] * exp(output_mat[..., 2:3])
        output_mat[..., 3:] = boxCoord[..., 3:] * exp(output_mat[..., 3:])
        taken_index = where(anchor_ > threshold)[:-1]
        output_bboxs.append(output_mat[taken_index])
        output_anchcors.append(anchor_[taken_index])

    return output_anchcors, output_bboxs


def test():
    img_lst = get_ImgPath()
    idx = randomIdx(img_lst)
    # idx = 197
    print('Case Index:', idx)
    img, mask = read_image_2D(img_lst[idx])
    expFactor = config.ExpBox
    gt_boxes = []
    for i in range(config.BoxScale):
        gt_boxes.append(generator_bbox(mask, mode='box', expFactor=expFactor))
        expFactor += 1
    gt_boxes = np.array(gt_boxes)
    gt_boxes = gt_boxes.reshape((-1, 4))

    img_shape = config.Image_shape if len(config.Image_shape) == 2 else config.Image_shape[1:]
    anchors, bbox_labels, indexes, bboxIdx = generator_anchor_area(img_shape=img_shape, gtBoxes=gt_boxes)
    output_anchcors, output_bboxs = apply_bboxReg_area(bboxMat=bbox_labels, img_shape=img_shape, anchors=anchors)
    # output boxes repeat num for gt_box
    plt.figure(idx)
    ax = plt.axes()
    plt.imshow(mask)
    counter = 0
    for i in range(len(output_bboxs)):
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
