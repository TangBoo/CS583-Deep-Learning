import torch
from torch import nn
from torchnet import meter
from torch.autograd import Variable as V
from torch import optim
import Losses
from torch.utils.data import DataLoader
from DataSet import UNETDataset, data_split
from model import UNet3d, UNet3d2d, UNet2d
import Config as config
from SegMatrix import SegmentationMetrix
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from torch.optim import lr_scheduler
from utils import Visualizer, get_output, features, get_pth

try:
    import ipdb as pdb
except:
    import pdb
import warnings

warnings.filterwarnings('ignore')


def val(model, dataloader):
    model.eval()
    confusion_matrix = SegmentationMetrix()
    loss = 0
    iou = 0
    pixelAcc = 0
    precision = 0
    sensitive = 0
    criterion = Losses.DiceLoss().to(config.Device)

    with torch.no_grad():
        for ii, (data, label) in enumerate(dataloader):
            confusion_matrix.reset()
            input = V(data.to(config.Device, dtype=torch.float))
            mask = V(label.to(config.Device, dtype=torch.float))
            score = model(input)
            current_loss = criterion(score, mask)
            if torch.isnan(current_loss):
                # pdb.set_trace()
                continue
            loss += current_loss.item()
            confusion_matrix.genConfusionMat(score.clone(), mask.clone())
            iou += confusion_matrix.mIoU()
            pixelAcc += confusion_matrix.pixelAcc()
            precision += confusion_matrix.precision()
            sensitive += confusion_matrix.sensitive()

    model.train()
    return loss / len(dataloader), iou / len(dataloader), pixelAcc / len(dataloader), precision / len(
        dataloader), sensitive / len(dataloader)


def train(**kwargs):
    vis = Visualizer(port=config.vis_port)
    model = nn.DataParallel(UNet3d(num_class=config.Num_class, base=config.Base)).cuda().train()
    handle1 = model.module.register_forward_hook(get_output('encoder1'))
    # handle2 = model.module.register_forward_hook(get_output('encoder2'))
    handle3 = model.module.register_forward_hook(get_output('final'))
    if config.checkpoint:
        if isinstance(model, nn.DataParallel):
            model.module.load(get_pth(model, config.checkpoint))
        else:
            model.load(get_pth(model, config.checkpoint))
        print("Load model successfully")

    train_lst, val_lst = data_split(config.Img_root, config.Train_percent, shuffle=True)
    train_data = UNETDataset(train_lst)
    train_dataloader = DataLoader(train_data, batch_size=config.TrainBatchSize, shuffle=True, num_workers=config.Num_workers)

    val_data = UNETDataset(val_lst)
    val_dataloader = DataLoader(val_data, batch_size=config.ValBatchSize, shuffle=True)

    train_matrix = SegmentationMetrix()
    lr = config.lr

    focalDice_criterion = Losses.FocalDice(alpha=config.Loss_alpha, beta=config.Loss_beta).to(config.Device)
    focal_criterion = Losses.FocalLoss().to(config.Device)
    dice_criterion = Losses.DiceLoss().to(config.Device)
    criterion = dice_criterion

    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=config.Weight_decay)
    # optimizer = optim.SGD(params=model.parameters(), lr=lr)
    # scheduler = lr_scheduler.StepLR(optimizer, 10, 0.1)  # *0.1/epoch
    previous_loss = 0
    prev_ValIou = 0
    patient = config.Patient
    debug_patient = config.Debug_Patient
    for epoch in range(config.Max_epoch):
        train_matrix.reset()
        start_time = datetime.now()
        train_acc = 0
        train_prec = 0
        train_sensi = 0
        loss_counter = 0
        for ii, (data, label) in enumerate(tqdm(train_dataloader)):
            input = V(data.to(config.Device, dtype=torch.float))
            mask = V(label.to(config.Device, dtype=torch.float))

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, mask)

            # ----------------clip gradient-------------
            # nn.utils.clip_grad_norm(model.parameters(), max_norm=20, norm_type=2.0)

            if torch.isnan(loss):
                continue

            loss_counter += loss.item()
            loss.backward()
            try:
                optimizer.step()
            except:
                for param_group in optimizer.param_groups:
                    print(param_group.keys())
                    print([type(value) for value in param_group.values()])
                    print('learning rage', param_group['lr'])
                    print('eps:', param_group['eps'])
                    print('params:', param_group['params'])
                    print('weight_decay:', param_group['weight_dacay'])
                pass

            # train valuation:
            train_matrix.genConfusionMat(score.clone(), label.clone())
            train_acc += train_matrix.pixelAcc()
            train_prec += train_matrix.precision()
            train_sensi += train_matrix.sensitive()

            if ii % config.Print_freq == config.Print_freq - 1:
                vis.log('train loss in epoch:' + criterion.name + ": " + str(loss.item()))
                vis.plot('train loss in epoch', loss.item())
                vis.img_many(features)
                # for name, weight in model.module.named_parameters():
                #     if weight.requires_grad:
                #         print(name, "-grad mean: ", weight.grad.mean(dim=0))
                #         print(name, "-grad min: ", weight.grad.min(dim=0))
                #         print(name, "-grad max: ", weight.grad.max(dim=0))

        model.module.save()
        end_time = datetime.now()

        h, remainder = divmod((end_time - start_time).seconds, 3600)
        m, s = divmod(remainder, 60)

        # ----------------validation-------------------
        val_loss, val_iou, val_pixelAcc, val_precision, val_sensitive = val(model, val_dataloader)
        avg_loss = loss_counter / len(train_dataloader)
        vis.log('val_iou：' + str(val_iou))
        vis.log('val_acc：' + str(val_pixelAcc))
        vis.log('val_precision：' + str(val_precision))
        vis.log('val_sensitive：' + str(val_sensitive))
        vis.plot('val_loss', val_loss)
        vis.plot('val_iou', val_iou)
        vis.plot('val_acc', val_pixelAcc)
        vis.plot('val_precision', val_precision)
        vis.plot('val_sensitive', val_sensitive)
        vis.plot('avg train loss', avg_loss)

        epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean Precision: {:.5f}, \
        Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
            epoch, avg_loss, train_acc / len(train_dataloader), train_prec / len(train_dataloader), val_loss,
            val_pixelAcc, val_iou))
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(epoch_str + time_str + ' lr: {}'.format(lr))

        if avg_loss > previous_loss and previous_loss != 0:
            debug_patient -= 1
            if patient == 0:
                patient = config.Patient
                lr = lr * config.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                patient -= 1

            if debug_patient == 0:
                pdb.set_trace()
                debug_patient = config.Debug_Patient

        previous_loss = avg_loss

        if val_iou > prev_ValIou and prev_ValIou != 0:
            model.module.save()
        prev_ValIou = val_iou
    handle1.remove()
    # handle2.remove()
    handle3.remove()


if __name__ == "__main__":
    train()
