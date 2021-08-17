import torch
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
import Config as config
from Models import RetinaUNet
from Losses import BBoxRegLoss, AnchorLoss, SegLoss
from DataSet import RetinaDataSet
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import warnings
from torch.autograd import Variable as V
from tqdm import tqdm
from utils import Visualizer, features, get_pth, get_ImgPath, data_split_2D, nms
from Evaluation import getPR, getAUC, EvalMatrix
from AnchorGenerator import apply_bboxReg_area
import time

try:
    import ipdb as pdb
except:
    import pdb
warnings.filterwarnings('ignore')


def val(model, dataloader, vis, boxShow=False):
    model.eval()
    segCFM = EvalMatrix()
    ancCFM = EvalMatrix()
    SEGLoss = 0
    ANCLoss = 0
    REGLoss = 0
    dice = 0
    precision = 0
    sensitive = 0
    ancPreci = 0
    ancSensi = 0
    RegCriterion = BBoxRegLoss()
    AncCriterion = AnchorLoss()
    SegCriterion = SegLoss()
    totalNum = len(dataloader)
    with torch.no_grad():
        for ii, (img, mask, anchors_labels, bbox_labels, LossIdx, bboxIdx) in enumerate(tqdm(dataloader)):
            segCFM.reset()
            ancCFM.reset()
            img = V(img).float().cuda()
            mask = V(mask).float().cuda()
            seg, anchors, boxes = model(img)
            SEGLoss += SegCriterion(img, mask)
            dice += 1 - SEGLoss
            curReg = 0
            curAnc = 0

            for i in range(config.Output_features):  # [features, batch, 9, h, w, ...]
                curReg += RegCriterion(V(boxes[i]).cuda(), V(bbox_labels[i]).cuda(), V(bboxIdx[i]).cuda())
                curAnc += AncCriterion(V(anchors[i]).cuda(), V(anchors_labels[i]).cuda(), V(LossIdx[i]).cuda())

            if boxShow and curAnc / config.Output_features < 0.5:
                for j in range(config.ValBatchSize):
                    boxes_input = [boxes[i][j:j + 1] for i in range(config.Output_features)]
                    anchors_input = [anchors[i][j:j + 1] for i in range(config.Output_features)]
                    output_anchcors, output_bboxs = apply_bboxReg_area(bboxMat=boxes_input,
                                                                       img_shape=config.Image_shape,
                                                                       anchors=anchors_input)  # bbox:[batch, 9, h, w, 4], anchor:[batch, 9, h, w, 1]
                    box = []
                    for i in range(config.Output_features):
                        finalBox = nms(output_bboxs[i], output_anchcors[i])
                        if not finalBox:
                            continue
                        box.append(finalBox)
                    if len(box) != 0:
                        box = t.stack(box, dim=0)
                        vis.show_boxes(mask[j], box)
                    else:
                        continue

            ANCLoss += curAnc / config.Output_features
            REGLoss += curReg / config.Output_features
            segCFM.genConfusionMat(seg.clone(), mask.clone())
            for i in range(config.Output_features):
                ancCFM.genConfusionMat(anchors[i].clone(), anchors_labels[i].clone())
                ancPreci += ancCFM.precision()
                ancSensi += ancCFM.sensitive()
                precision += segCFM.precision()
                sensitive += segCFM.sensitive()
    model.train()
    return SEGLoss / totalNum, REGLoss / totalNum, ANCLoss / totalNum, \
           precision / (config.Output_features * totalNum), sensitive / (config.Output_features * totalNum), ancPreci / (config.Output_features * totalNum), \
           ancSensi / (config.Output_features * totalNum), dice / totalNum


def train(**kwargs):
    vis = Visualizer(env=config.Env, port=config.vis_port)
    model = nn.DataParallel(RetinaUNet(base=64, InChannel=1, OutChannel=1, BackBone='fpn')).cuda()
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    train_matrix = EvalMatrix()
    if config.checkpoint:
        try:
            if isinstance(model, nn.DataParallel):
                model.module.load(get_pth(model, config.checkpoint))
            else:
                model.load(get_pth(model, config.checkpoint))
            print("Load Model Successfully")
        except:
            pass

    img_lst = get_ImgPath()
    train_lst, val_lst = data_split_2D(img_lst, ratio=(1 - config.Val_percent), shuffle=False)
    train_data = RetinaDataSet(train_lst)
    train_dataloader = DataLoader(train_data, batch_size=config.TrainBatchSize, shuffle=True,
                                  num_workers=config.Num_workers)

    val_data = RetinaDataSet(val_lst, isTrain=False)
    val_dataloader = DataLoader(val_data, batch_size=config.ValBatchSize, shuffle=False)

    lr = config.lr
    RegCriterion = BBoxRegLoss()
    AncCriterion = AnchorLoss()
    SegCriterion = SegLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=config.Weight_decay)
    previousLoss = 0
    previousValLoss = 100
    debug_patient = config.Debug_Patient
    patient = config.Lr_Patient
    for epoch in range(config.Max_epoch):
        train_matrix.reset()
        start_time = time.time()
        train_precision = 0
        train_sensi = 0
        loss_counter = 0
        for ii, (img, mask, anchors_labels, bbox_labels, LossIdx, bboxIdx) in enumerate(tqdm(train_dataloader)):
            input = V(img).float().cuda()
            mask = V(mask).float().cuda()
            with torch.cuda.amp.autocast():
                seg, ancOutput, boxOutput = model(input)
                seg_loss = SegCriterion(seg, mask)
                box_loss = 0
                anc_loss = 0
                for i in range(config.Output_features):
                    box_loss += RegCriterion(V(boxOutput[i]).cuda(), V(bbox_labels[i]).cuda(), V(bboxIdx[i]).cuda())
                    anc_loss += AncCriterion(V(ancOutput[i]).cuda(), V(anchors_labels[i]).cuda(), V(LossIdx[i]).cuda())
                loss = seg_loss + box_loss + anc_loss
            loss_counter += loss.item()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # train valuation:
            train_matrix.genConfusionMat(seg.clone(), mask.clone())
            train_precision += train_matrix.precision()
            train_sensi += train_matrix.sensitive()

            if ii % config.Print_freq == config.Print_freq - 1:
                vis.log('train loss in Segmentation: ' + str(seg_loss.item()))
                vis.log('train loss in bounding box regression: ' + str(box_loss))
                vis.log('train loss in Anchor Net: ' + str(anc_loss))
                vis.plot('Segmentation loss:', seg_loss.item())
                vis.plot('Bounding box regression loss', box_loss.item())
                vis.plot('Anchor Net Loss: ', anc_loss.item())
                vis.plot('Training Loss: ', loss.item())
        end_time = time.time()

        avgSeg, avgReg, ancAvg, avgPrec, avgSensi, avgAncPreci, avgAncSensi, avgDice = val(model, val_dataloader, vis,
                                                                                           boxShow=True)
        avg_loss = loss_counter / len(train_dataloader)
        vis.log("Validation Dice:", str(avgDice))
        vis.plot("Avg Train Loss: ", avg_loss)
        vis.plot("Avg Val Loss for Seg: ", avgSeg.item())
        vis.plot("Avg Val Loss for Regression:", avgReg.item())
        vis.plot("Avg Val Loss for Anchor Net:", ancAvg.item())
        vis.plot("Avg Val Precision for Seg: ", avgPrec.item())
        vis.plot("Avg Val Sensitive for Seg: ", avgSensi.item())
        vis.plot("Avg Val Precision for Anchor Net:", avgAncPreci.item())
        vis.plot("Avg Val Sensitive for Anchor Net:", avgAncSensi.item())
        val_loss = avgSeg + avgReg + ancAvg
        epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Mean Precision: {:.5f}, \
                Valid Loss: {:.5f}'.format(
            epoch, avg_loss, train_precision / len(train_dataloader), val_loss.item()))
        print(epoch_str + str(end_time - start_time) + ' lr: {}'.format(lr))

        if avg_loss >= previousLoss != 0:
            debug_patient -= 1
            if patient == 0:
                patient = config.Lr_Patient
                lr = lr * config.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                patient -= 1

            if debug_patient == 0:
                pdb.set_trace()
                debug_patient = config.Debug_Patient

        previousLoss = avg_loss
        if val_loss < previousValLoss / 2 and previousLoss != 100:
            model.module.save()
        previousValLoss = val_loss


def test():
    vis = Visualizer(port=config.vis_port)
    model = nn.DataParallel(RetinaUNet(base=64, InChannel=1, OutChannel=1, BackBone='fpn')).cuda()
    img_lst = get_ImgPath()
    train_lst, val_lst = data_split_2D(img_lst, ratio=(1 - config.Val_percent), shuffle=False)
    train_data = RetinaDataSet(train_lst)
    train_dataloader = DataLoader(train_data, batch_size=config.TrainBatchSize, shuffle=True,
                                  num_workers=config.Num_workers)
    val_data = RetinaDataSet(val_lst, isTrain=False)
    val_dataloader = DataLoader(val_data, batch_size=config.ValBatchSize, shuffle=False)
    avgSeg, avgReg, ancAvg, avgPrec, avgSensi, avgAncPreci, avgAncSensi, avgDice = val(model, val_dataloader, vis,
                                                                                       boxShow=True)


if __name__ == "__main__":
    test()
