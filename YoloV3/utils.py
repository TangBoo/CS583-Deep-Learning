import torch
import numpy as np
import math
from tqdm import tqdm
import Meta
from collections import Counter
import warnings

warnings.filterwarnings('ignore')
anchors = Meta.anchors


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
            boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def convert_box(bbox):
    ans = torch.zeros_like(bbox)
    x = bbox[..., 0]
    y = bbox[..., 1]
    w = bbox[..., 2]
    h = bbox[..., 3]
    ans[..., 0] = torch.maximum(x - w / 2, torch.zeros_like(x))
    ans[..., 1] = torch.maximum(y - h / 2, torch.zeros_like(y))
    ans[..., 2] = x + w / 2
    ans[..., 3] = y + h / 2
    return ans


def bbox_overlaps_ciou(bboxes1, bboxes2):
    bboxes1 = convert_box(bboxes1)
    bboxes2 = convert_box(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2
    inter_max_xy = torch.minimum(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.maximum(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.minimum(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    if exchange:
        cious = cious.T
    return cious


def non_max_suppression(bboxes, iou_threshold, threshold=0.5, box_format="corners"):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes\
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (confident score)
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes if
            intersection_over_union(chosen_box[2:], box[2:]) < iou_threshold or box[0] != chosen_box
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def cells_to_bboxes(predictions, anchors, recep_field, is_preds=True):
    '''
    Args:
        1. predictions : [batch_size, anchors, h, w, 5 + num_classes]
        2. anchors : the anchors used for the predictions, [1, 3, 1, 1, 2].
        3. the number of cells the image is divided in on the width and height.
        4. is_preds: whether the input is predictions or the true bounding box.
    '''
    if len(predictions.shape) == 4:
        predictions = predictions.unsqueeze(0)

    # predictions:[batch, anchors, h, w, 5 + num_classes]
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]  # [score, x, y, w, h, num_classes:3]

    # scale size
    h, w = predictions.shape[2:4]
    # [1, 3, 1, 1, 2]
    anchors = np.array(anchors).reshape((1, 3, 1, 1, 2))
    anchors[..., 0] *= w
    anchors[..., 1] *= h

    if is_preds:
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])  # x, y
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors  # width, height
        scores = torch.sigmoid(predictions[..., 0])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)

    # [batch_size, 3, h, w, 1]
    x_indices = torch.arange(w).repeat(batch_size, num_anchors, h, 1).unsqueeze(-1).to(predictions.device)
    y_indices = torch.arange(h).repeat(batch_size, num_anchors, w, 1).unsqueeze(-1).to(predictions.device)

    x = box_predictions[..., 0:1] + x_indices  # (0, 1, 2, 3, 4)
    y = box_predictions[..., 1:2] + y_indices.permute(0, 1, 3, 2, 4)  # [batch_size, 3, w, h, 1]
    w_h = box_predictions[..., 2:4]

    # [batch_size, 3 * h * w, (best_class, score, x, y, w, h)]
    converted_boxers = torch.cat((best_class, scores, x * recep_field, y * recep_field, w_h), dim=-1).reshape(
        batch_size, num_anchors * h * w, 6)
    return converted_boxers


def get_evaluation_bboxes(loader, model, iou_threshold, anchors, threshold, box_format="midpoint", device="cuda"):
    '''
    Args:
        anchors: [3, 3, 2]
        :param anchors:
        :param threshold:
        :param device:
        :param box_format:
    '''
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []

    # Image:[batch, channel, h, w, 5 + classes], y:[batch, 3, h, w, 5 + num_classes]
    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to("cuda")
        with torch.no_grad():
            predictions = model(x)  # (predictions(list):[[batch, h, w, 5 + num_classes],...])

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        gtboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            anchor = torch.tensor([*anchors[i]]).to("cuda")
            # [batch, num_anchors * h * w, (class, score, x, y, w, h)]
            boxes_scale_i = cells_to_bboxes(predictions[i], anchor, is_preds=True)
            gt_scale_i = cells_to_bboxes(y[i], anchor, is_preds=False)
            for idx, box, gt_box in enumerate(zip(boxes_scale_i, gt_scale_i)):
                # idx = batch
                bboxes[idx] += box
                gtboxes[idx] += gt_box

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(bboxes[idx], iou_threshold=iou_threshold,
                                            threshold=threshold, box_format=box_format)
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in gtboxes[idx]:
                if box[1] < threshold:
                    continue
                all_true_boxes.append([train_idx] + box)

            train_idx += 1

    return all_pred_boxes, all_true_boxes


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def mAP(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    '''
    Args:
        pred_boxes(list) : prediction bounding box,
        [train_idx, class_pred, prob_score, x1, y1, x2, y2]
        true_boxes(list) : similar as pred_boxes except all the correct ones
        iou_threshold(float) : threshold where predicted bboxes is correct
        box_format(str) : "midpoint" or "corners" used to specify bboxes
        num_classes(int) : number of classes

    Returns:
        float：mAP value across all classes given a specific Iou threshold

    Describ:
        precision = tp / (tp + fp)
    '''
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
