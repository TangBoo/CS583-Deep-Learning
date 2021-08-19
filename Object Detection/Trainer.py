import torch
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
import Config as config
from Models import RetinaUNet
from Losses import BBoxRegLoss, AnchorLoss, SegLoss, Iou_loss
from DataSet import RetinaDataSet
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import warnings
from torch.autograd import Variable as V
from tqdm import tqdm
from utils import Visualizer, features, get_pth, get_ImgPath, data_split_2D, nms, myStr, t2i, read_image_2D
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
        for ii, (img, mask, anchors_labels, bbox_labels, ancLossIdx, bboxLossIdx, segIdx) in enumerate(dataloader):
            segCFM.reset()
            img = V(img).float().cuda()
            mask = V(mask).float().cuda()
            seg, ancOutput, boxOutput = model(img)
            dice += Iou_loss(seg, mask)
            vis.plot("Dice in Iteration:", t2i(Iou_loss(seg, mask)))
            # --------------------Loss Computation--------------
            temp_seg = SegCriterion(img, mask, segIdx)
            if not t.isnan(temp_seg):
                seg_Loss += temp_seg

            for i in range(config.Output_features):
                temp_reg = RegCriterion(V(boxOutput[i]).cuda(), V(bbox_labels[i]).cuda(), V(bboxLossIdx[i]).cuda())
                if not t.isnan(temp_reg):
                    reg_Loss += temp_reg / config.Output_features
                temp_anc = AncCriterion(V(ancOutput[i]).cuda(), V(anchors_labels[i]).cuda())
                if not t.isnan(temp_anc):
                    anc_Loss += temp_anc / config.Output_features

            vis.plot("Follow Seg Loss:", t2i(temp_seg))
            vis.plot("Follow Anchor Loss: ", t2i(temp_anc))
            vis.plot("Follow Box Loss:", t2i(reg_Loss))

            # ---------------Show Bounding Box----------------
            if boxShow and temp_anc / config.Output_features < 0.5 and temp_anc != 0:
                for j in range(config.ValBatchSize):
                    boxes_input = [boxOutput[i][j:j + 1] for i in range(
                        config.Output_features)]  # bbox:[features, batch, 9, h, w, 4], bbox:[batch, features, 9, h, w, 1]
                    anchors_input = [ancOutput[i][j:j + 1] for i in range(
                        config.Output_features)]  # anchor:[features, batch, 9, h, w, 4], anchor:[batch, features, 9, h, w, 1]
                    output_anchcors, output_bboxs = apply_bboxReg_area(bboxMat=boxes_input,
                                                                       img_shape=config.Image_shape,
                                                                       anchors=anchors_input)
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
    model.train()
    return Val_avgLoss


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
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=config.Num_workers)

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
        train_precision, train_sensi, loss_counter = 0, 0, 0

        for ii, (img, mask, anchors_labels, bbox_labels, ancLossIdx, bboxLossIdx, segIdx) in enumerate(
                tqdm(train_dataloader)):
            input = V(img).float().cuda()
            mask = V(mask).float().cuda()
            optimizer.zero_grad()
            seg, ancOutput, boxOutput = model(input)

            box_loss, anc_loss, seg_loss= 0, 0, 0
            temp_seg = SegCriterion(seg, mask, segIdx)
            if not t.isnan(temp_seg):
                seg_loss += temp_seg

            for i in range(config.Output_features):
                temp_reg = RegCriterion(V(boxOutput[i]).cuda(), V(bbox_labels[i]).cuda(), V(bboxLossIdx[i]).cuda())
                if not t.isnan(temp_reg.flatten()):
                    box_loss += temp_reg / config.Output_features
                temp_anc = AncCriterion(V(ancOutput[i]).cuda(), V(anchors_labels[i]).cuda())
                if not t.isnan(temp_anc):
                    anc_loss += temp_anc / config.Output_features

            loss = seg_loss + anc_loss + box_loss
            if loss == 0:
                vis.log("Nan Loss: Seg:{}, Box:{}, Anc:{}".format(myStr(seg_loss), myStr(box_loss), myStr(anc_loss)))
                continue

            loss_counter += t2i(loss)
            loss.backward()
            optimizer.step()

            # ---------------------Train Valuation-------------------------
            train_matrix.genConfusionMat(seg.clone(), mask.clone())
            train_precision += train_matrix.precision()
            train_sensi += train_matrix.sensitive()

            if ii % config.Print_freq == config.Print_freq - 1:
                vis.log('train loss in Segmentation: ' + myStr(seg_loss))
                vis.log('train loss in bounding box regression: ' + myStr(box_loss))
                vis.log('train loss in Anchor Net: ' + myStr(anc_loss))
                vis.plot('Segmentation loss:', t2i(seg_loss))
                vis.plot('Bounding box regression loss', t2i(box_loss))
                vis.plot('Anchor Net Loss: ', t2i(anc_loss))
                vis.plot('Training Loss: ', t2i(loss))
        end_time = time.time()

        # -----------------Validation--------------------
        val_loss = val(model, val_dataloader, vis, boxShow=False)
        avg_loss = loss_counter / len(train_dataloader)
        vis.plot("Avg Train Loss: ", avg_loss)
        epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Mean Precision: {:.5f}, \
                Valid Loss: {:.5f}'.format(
            epoch, avg_loss, train_precision / len(train_dataloader), val_loss.item()))
        print(epoch_str + " Time:" + str(end_time - start_time) + ' lr: {}'.format(lr))

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


if __name__ == "__main__":
    train()
