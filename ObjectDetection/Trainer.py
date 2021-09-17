import traceback

import numpy as np
import torch
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
import Config as config
import utils
from Models import RetinaUNet
from Losses import BBoxRegLoss, AnchorLoss, SegLoss, Iou_loss
from DataSet import RetinaDataSet
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import warnings
from torch.autograd import Variable as V
from tqdm import tqdm
from utils import features, get_pth, get_ImgPath, data_split_2D, nms, myStr, t2i, read_image_2D, mapAnc2Img
from Visualization import Visualizer
from Evaluation import getPR, getAUC, EvalMatrix, PrecisionRecall
from AnchorGenerator import apply_bboxReg_area
import time
from DataProcess import read_2dPlus, get_Path_2dPlus

try:
    import ipdb as pdb
except:
    import pdb
warnings.filterwarnings('ignore')


def val(model, dataloader, vis, boxShow=False):
    model.eval()
    segCFM = EvalMatrix()
    ancCFM = EvalMatrix()
    seg_Loss, anc_Loss, reg_Loss, dice = 0, 0, 0, 0
    seg_Precision, seg_Sensitive, anc_Precision, anc_Sensitive = 0, 0, 0, 0
    RegCriterion = BBoxRegLoss()
    AncCriterion = AnchorLoss()
    SegCriterion = SegLoss()
    totalNum = len(dataloader)
    with torch.no_grad():
        for ii, (img, mask, anchors_labels, bbox_labels, ancLossIdx, bboxLossIdx, segIdx, GtBoxes) in enumerate(
                tqdm(dataloader)):
            segCFM.reset()
            img = V(img).float().cuda()
            mask = V(mask).float().cuda()
            seg, ancOutput, boxOutput = model(img)
            vis.show_hist(name="Val Anchor Output", tensor=ancOutput)
            # ----------------------Show Anchor Location--------
            for threshold in np.arange(0.4, 0.6, 0.1):
                image_ = mapAnc2Img(ancOutput, img, threshold=threshold)
                vis.show_anchor(name="Anc Loc {}".format(threshold), imgs=image_)

            # ----------------------Segmentation-------------
            temp_seg = 0
            if seg is not None:
                temp_seg = SegCriterion(seg, mask, segIdx)
                if temp_seg != 0:
                    dice += Iou_loss(seg, mask)
                    vis.plot("Dice in Iteration:", t2i(Iou_loss(seg, mask)))
            # --------------------Loss Computation--------------
            if temp_seg != 0 and not t.isnan(temp_seg):
                seg_Loss += temp_seg
                vis.plot("Follow Seg Loss:", t2i(temp_seg))
            for i in range(config.Output_features):
                temp_reg = RegCriterion(V(boxOutput[i]).cuda(), V(bbox_labels[i]).cuda(), V(bboxLossIdx[i]).cuda())
                if temp_reg != 0 and not t.isnan(temp_reg):
                    reg_Loss += temp_reg
                temp_anc = AncCriterion(V(ancOutput[i]).cuda(), V(anchors_labels[i]).cuda())
                if temp_anc != 0 and not t.isnan(temp_anc):
                    anc_Loss += temp_anc

            # ---------------Show Bounding Box----------------
            if boxShow and temp_anc / config.Output_features < 1.0 and temp_anc != 0:
                for j in range(config.ValBatchSize):
                    boxes_input = [boxOutput[i][j:j + 1] for i in range(
                        config.Output_features)]  # bbox:[features, batch, 9, h, w, 4], bbox:[batch, features, 9, h, w, 1]
                    anchors_input = [ancOutput[i][j:j + 1] for i in range(
                        config.Output_features)]  # anchor:[features, batch, 9, h, w, 4], anchor:[batch, features, 9, h, w, 1]
                    output_anchcors, output_bboxs = apply_bboxReg_area(bboxMat=boxes_input,
                                                                       img_shape=config.Image_shape,
                                                                       anchors=anchors_input)
                    box = []
                    box_Precision = 0
                    box_Recall = 0
                    for i in range(config.Output_features):
                        finalBox = nms(output_bboxs[i], output_anchcors[i])
                        if finalBox is None:
                            continue
                        box.append(finalBox)
                    if len(box) != 0:
                        box = t.stack(box, dim=0)
                        vis.show_boxes(mask[j], box)

                        if len(GtBoxes) != 0:
                            tmp_p, tmp_r = PrecisionRecall(GtBoxes, finalBox)
                            box_Precision += tmp_p
                            box_Recall += tmp_r
                            vis.log("Box Precision:" + myStr(box_Precision))
                            vis.log("Box Recall:" + myStr(box_Recall))
                    else:
                        continue
            if seg is not None:
                segCFM.genConfusionMat(seg.clone(), mask.clone())
                seg_Precision += segCFM.precision()
                seg_Sensitive += segCFM.sensitive()
            for i in range(config.Output_features):
                ancCFM.genConfusionMat(ancOutput[i].clone(), anchors_labels[i].clone())
                anc_Precision += ancCFM.precision() / config.Output_features
                anc_Sensitive += ancCFM.sensitive() / config.Output_features

    Val_avgLoss = (seg_Loss + anc_Loss + reg_Loss) / totalNum
    vis.log("Validation Dice:" + myStr(dice / totalNum), isTrain=False)
    vis.log("Validation Seg Loss:" + myStr(seg_Loss / totalNum), isTrain=False)
    vis.log("Validation BoxReg Loss:" + myStr(reg_Loss / totalNum), isTrain=False)
    vis.log("Validation Anchor Loss:" + myStr(anc_Loss / totalNum), isTrain=False)
    vis.plot("Avg Val Iou:", t2i(dice / totalNum))
    vis.plot("Avg Val Loss for Seg: ", t2i(seg_Loss / totalNum))
    vis.plot("Avg Val Loss for Regression:", t2i(reg_Loss / totalNum))
    vis.plot("Avg Val Loss for Anchor Net:", t2i(anc_Loss / totalNum))
    vis.plot("Avg Val Precision for Seg: ", t2i(seg_Precision / totalNum))
    vis.plot("Avg Val Sensitive for Seg: ", t2i(seg_Sensitive / totalNum))
    vis.plot("Avg Val Precision for Anchor Net:", t2i(anc_Precision / totalNum))
    vis.plot("Avg Val Sensitive for Anchor Net:", t2i(anc_Sensitive / totalNum))
    # -------------Weight Statistics----------------
    for name in model.module.ANCNets.state_dict():
        if name.endswith("weight"):
            vis.show_hist(name=name, tensor=model.module.ANCNets.state_dict()[name])

    model.train()
    return Val_avgLoss


def train(**kwargs):
    vis = Visualizer(env=config.Env, port=config.vis_port)
    model = nn.DataParallel(
        RetinaUNet(base=config.Base, InChannel=config.InputChannel, OutChannel=1, BackBone='fpn', IncludeTop=False,
                   Godown=False, IncludeSeg=False)).to(config.Device)

    model.train()
    scaler = torch.cuda.amp.GradScaler()
    Seg_Matrix = EvalMatrix()
    Anc_Matrix = EvalMatrix()
    if config.checkpoint:
        try:
            if isinstance(model, nn.DataParallel):
                model.module.load(get_pth(model, config.checkpoint))
            else:
                model.load(get_pth(model, config.checkpoint))
            print("Load Model Successfully")
        except:
            pass

    if config.Data_type.lower() == '2dplus':
        img_read = read_2dPlus
        img_lst = get_Path_2dPlus()
    elif config.Data_type.lower() == '2d':
        img_read = read_image_2D
        img_lst = get_ImgPath()
    else:
        raise ValueError

    train_lst, val_lst = data_split_2D(img_lst, ratio=(1 - config.Val_percent), shuffle=True)
    train_data = RetinaDataSet(train_lst, img_read)
    train_dataloader = DataLoader(train_data, batch_size=config.TrainBatchSize, shuffle=True,
                                  num_workers=config.Num_workers)

    val_data = RetinaDataSet(val_lst, img_read, isTrain=False)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=config.Num_workers)

    lr = config.lr

    RegCriterion = BBoxRegLoss().to(config.Device)
    AncCriterion = AnchorLoss().to(config.Device)
    SegCriterion = SegLoss().to(config.Device)
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=config.Weight_decay)
    # scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    # scheduler_cosin = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5, last_epoch=-1)
    scheduler_monitor = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                                                   threshold_mode='rel', threshold=0.0001, cooldown=5,
                                                                   min_lr=1e-5, eps=1e-8)

    previousLoss = 0
    previousValLoss = 100
    debug_patient = config.Debug_Patient
    patient = config.Lr_Patient
    for epoch in range(config.Max_epoch):
        Seg_Matrix.reset()
        Anc_Matrix.reset()
        start_time = time.time()
        Seg_precision, Seg_sensi, loss_counter, AvgAncLoss, AvgAncPrec, AvgAncSensi = 0, 0, 0, 0, 0, 0
        # if epoch != 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr * config.lr_decay

        for ii, (img, mask, anchors_labels, bbox_labels, ancLossIdx, bboxLossIdx, segIdx, _) in enumerate(
                tqdm(train_dataloader)):
            input = V(img).float().to(config.Device, dtype=torch.float)
            mask = V(mask).float().to(config.Device, dtype=torch.float)
            optimizer.zero_grad()
            seg, ancOutput, boxOutput = model(input)
            # ---------------------Anchor Output Statistics----------------
            # vis.show_hist(name="Train Anchor Output", tensor=ancOutput)
            Anc_precision, Anc_sensi, Anc_iou = 0, 0, 0
            box_loss, anc_loss, seg_loss, loss = 0., 0., 0., 0.
            temp_seg = 0
            if seg is not None:
                temp_seg = SegCriterion(seg, mask, segIdx)
            if temp_seg != 0 and not t.isnan(temp_seg):
                seg_loss += temp_seg

            for i in range(config.Output_features):
                # ----------------------Bounding Box Regression-----------------
                temp_reg = RegCriterion(boxOutput[i], V(bbox_labels[i]).to(config.Device, dtype=torch.float),
                                        V(bboxLossIdx[i]).to(config.Device, dtype=torch.float))
                if temp_reg != 0 and not t.isnan(temp_reg):
                    box_loss += temp_reg / config.Output_features
                # ----------------Anchor Loss Computation----------------
                temp_anc = AncCriterion(ancOutput[i], V(anchors_labels[i]).to(config.Device, dtype=torch.float),
                                        V(ancLossIdx[i]).to(config.Device, dtype=torch.float))
                # handle = ancOutput[i].register_hook(utils.get_grad)
                if temp_anc != 0 and not t.isnan(temp_anc):
                    anc_loss += temp_anc

                Anc_Matrix.genConfusionMat(ancOutput[i], anchors_labels[i])
                Anc_precision += (Anc_Matrix.precision() / config.Output_features)
                Anc_sensi += (Anc_Matrix.sensitive() / config.Output_features)
                Anc_iou += (Anc_Matrix.mIoU())

            if seg_loss != 0:
                loss += seg_loss
            if anc_loss != 0:
                loss += anc_loss
            if box_loss != 0:
                loss += box_loss

            if loss == 0:
                # vis.log("Nan Loss: Seg:{}, Box:{}, Anc:{}".format(myStr(seg_loss), myStr(box_loss), myStr(anc_loss)))
                continue

            loss.backward()
            # ---------------------Anchor Gradient Statistics----------------
            # vis.show_hist(name="Train Anchor Gradient", tensor=t.tensor([k.mean() for k in utils.features['loss']]))
            loss_counter += t2i(loss)
            optimizer.step()
            # scheduler_cosin.step()
            # if anc_loss != 0:
            #     scheduler_monitor.step(anc_loss)
            # ---------------------Train Evaluation-------------------------
            if seg is not None:
                Seg_Matrix.genConfusionMat(seg.clone(), mask.clone())
                Seg_precision += Seg_Matrix.precision()
                Seg_sensi += Seg_Matrix.sensitive()
            AvgAncLoss += anc_loss
            AvgAncPrec += Anc_precision
            AvgAncSensi += Anc_sensi
            if seg_loss != 0:
                vis.log('train loss in Segmentation: ' + myStr(seg_loss))
                vis.plot('Segmentation loss:', t2i(seg_loss))

            if ii % config.Print_freq == config.Print_freq - 1:
                vis.log('train loss in bounding box regression: ' + myStr(box_loss))
                vis.log('train loss in Anchor Net: ' + myStr(anc_loss))
                vis.plot("Anchor Precision:", t2i(Anc_precision))
                vis.plot("Anchor Sensitive:", t2i(Anc_sensi))
                vis.plot('Bounding box regression loss', t2i(box_loss))
                vis.plot('Anchor Net Loss: ', t2i(anc_loss))
                vis.plot('Training Loss: ', t2i(loss))
        end_time = time.time()

        # -----------------Validation--------------------
        val_loss = val(model, val_dataloader, vis, boxShow=False)
        avg_loss = loss_counter / len(train_dataloader)
        vis.plot("Avg Train Anchor Loss:", t2i(AvgAncLoss / len(train_dataloader)))
        vis.plot("Avg Train Anchor Precision:", t2i(AvgAncPrec / len(train_dataloader)))
        vis.plot("Avg Train Anchor Sensitive:", t2i(AvgAncSensi / len(train_dataloader)))
        epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Seg Mean Precision: {:.5f}, Train Seg Mean Sensitive:{:.5f},\
                Valid Loss: {:.5f}'.format(
            epoch, avg_loss, Seg_precision / len(train_dataloader), Seg_sensi / len(train_dataloader), val_loss.item()))
        print(epoch_str + " Time:" + str(end_time - start_time) + ' lr: {}'.format(
            optimizer.state_dict()['param_groups'][0]['lr']))

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
    # handle.remove()


def test():
    model = nn.DataParallel(
        RetinaUNet(base=config.Base, InChannel=config.InputChannel, OutChannel=1, BackBone='fpn', IncludeTop=False,
                   Godown=True, IncludeSeg=False)).cuda()
    vis = Visualizer(env=config.Env, port=config.vis_port)

    if config.Data_type.lower() == '2dplus':
        img_read = read_2dPlus
        img_lst = get_Path_2dPlus()
    elif config.Data_type.lower() == '2d':
        img_read = read_image_2D
        img_lst = get_ImgPath()
    else:
        raise ValueError
    train_lst, val_lst = data_split_2D(img_lst, ratio=(1 - config.Val_percent), shuffle=False)
    train_data = RetinaDataSet(train_lst, img_read)
    train_dataloader = DataLoader(train_data, batch_size=config.TrainBatchSize, shuffle=True,
                                  num_workers=config.Num_workers)

    val_data = RetinaDataSet(val_lst, img_read, isTrain=False)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=config.Num_workers)
    val(model, val_dataloader, vis=vis)


if __name__ == "__main__":
    train()
