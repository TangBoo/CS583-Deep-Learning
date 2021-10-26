import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np

import torch as t
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import optim
from Realign import getCaselst, getSpace, load4DDcm, load4DImage, Registration, seriesImg_show
from BrainBoneRemove import BoneRemove
import cv2
from IPython import display





def get_ct_value_neighbor_avg(imgs, x, y, d=3):
    # img [time, h, w]
    time, h, w = imgs.shape
    x_start = max(x - d, 0)
    x_end = min(x + d, w)
    y_start = max(y - d, 0)
    y_end = min(y + d, h)
    res = imgs[..., y_start: y_end, x_start: x_end]  # [time, 2d, 2d]
    count = (2 * d) ** 2 - np.sum(res[0, ...] == 0)
    ans = np.sum(res, axis=(1, 2)) / count
    return ans


# @jit(nopython=True)
def get_tissue_signal(img, mask, g_d):
    t, h, w = img.shape
    output = np.zeros((t, h, w))
    for ii in range(h * w):
        y_t_i = int(ii // h)
        x_t_i = int(ii % w)
        if mask[y_t_i, x_t_i] == 0:
            continue
        output[..., y_t_i, x_t_i] = get_ct_value_neighbor_avg(img, x_t_i, y_t_i, g_d)
    return output


def poly_features(x, order=6):
    time, h, w, _ = x.shape
    if not t.is_tensor(x):
        x = t.tensor(x)
    features = t.ones((time, h, w, 1), requires_grad=True)
    for i in range(1, order):
        features = t.cat((features, x ** i), dim=-1)
    return features


def QR_features(x, degree=6):
    # x:[time, h, w, 1]
    time, h, w, _ = x.shape
    poly_x = poly_features(x, degree + 1).permute(1, 2, 0, -1)  # [h, w, time, features]
    ans = t.zeros((h, w, time, degree))
    for i in range(256 * 256):
        row = i // 256
        col = i % 256
        ans[row, col] = t.qr(poly_x[row, col])[0][:, 1:]
    return ans.permute(2, 0, 1, -1)


def upsample(y, factor):
    # y:[time, h, w, 1]
    if not t.is_tensor(y):
        y = t.tensor(y)
    y = y.unsqueeze(0).permute(0, 4, 1, 2, 3)
    expand_y = F.interpolate(y, scale_factor=(factor, 1, 1), mode='trilinear', align_corners=True)
    expand_y = expand_y.permute(0, 2, 3, 4, 1).squeeze(0)
    print(expand_y.shape)
    return expand_y


class MSE(nn.Module):
    def __init__(self, mask):
        super(MSE, self).__init__()
        if not t.is_tensor(mask):
            mask = t.tensor(mask)

        self.mask = mask
        self.loss = t.nn.MSELoss(reduction='sum')

    def forward(self, y, target):
        print(y.shape, target.shape)
        index = t.where(self.mask != 0)
        #         ans = self.loss(y[index], target[index])
        ans = self.loss(y, target)
        return ans


def channel_gradient(mat, degree=1):
    # mat : [time, h, w, 1]
    mat = mat.copy()
    h, w, time = mat.shape
    for j in range(degree):
        time = mat.shape[-1]
        gradient = np.zeros((h, w, time - 1))
        for i in range(1, time):
            gradient[..., i - 1] = mat[..., i] - mat[..., i - 1]
        mat = gradient
    return mat


def dislocation_plus(mat, pluser, sign_mat):
    assert mat.shape[-1] == pluser.shape[-1] + 1
    assert pluser.shape == sign_mat.shape
    h, w, time = mat.shape
    ans = np.zeros_like(mat)
    ans[..., 0] = mat[..., 0]
    for i in range(time - 1):
        ans[..., i + 1] = ans[..., i] + sign_mat[..., i] * pluser[..., i]
    return ans


def resample_index(*args):
    num, sample_num = args
    #     if num < sample_num:
    #         num = sample_num
    if sample_num <= num:
        ans = np.random.choice(int(num), int(sample_num), replace=False)
    else:
        ans = np.random.choice(int(num), int(sample_num), replace=True)
    return t.tensor(ans)


def subsection(mat, mask, group_num=10, Device="cuda:0"):
    if not t.is_tensor(mask):
        mask = t.tensor(mask)
    mat = mat.to(Device)

    time, h, w, _ = mat.shape
    minv = t.min(mat, dim=0)[0]
    maxv = t.max(mat, dim=0)[0]
    step = (((maxv - minv) / group_num) * t.arange(group_num).to(Device)) + minv
    low = step[..., ::2]
    high = step[..., 1::2]
    boaders = t.cat((low.unsqueeze(-1), high.unsqueeze(-1)), dim=-1).permute(-1, -2, 0, 1)
    eps_mat = t.full((1, h, w), 1e-6).to(Device)
    loc_mat = t.zeros((group_num // 2, time, h, w))
    min_loc = t.zeros((group_num // 2, h, w))
    index_mat = t.zeros_like(loc_mat)  # [group_num // 2, time, h, w]
    if mat.shape[-1] == 1:
        mat = mat.squeeze(-1)

    for i in range(group_num // 2):
        condi_left = boaders[0:1, i]
        condi_right = boaders[1:, i]
        loc = t.tensor((mat >= condi_left) * (mat < t.max(condi_right, eps_mat)), dtype=t.float)
        num_sort, index_sort = t.sort(loc, dim=0, descending=True)
        min_loc[i, ...] = t.argmin(num_sort, dim=0)
        index_mat[i] = index_sort
        loc_mat[i] = loc

    sample_num = group_num
    resample_mat = t.zeros((group_num // 2, sample_num, h, w))
    for i in range(group_num // 2):
        temp_min = min_loc[i]
        res = t.zeros((int(sample_num), h, w))
        for j in range(256 * 256):
            row = j // 256
            col = j % 256
            if mask[row, col] == 0:
                continue
            num = temp_min[row, col]
            res[..., row, col] = resample_index(num, sample_num)

        # -------------- Take From Index Sort ---------------------
        resample_mat[i] = res
    sample_loc = t.gather(index_mat, dim=1, index=resample_mat.long()).reshape(-1, 256, 256)
    sample_loc, _ = t.sort(sample_loc, dim=0, descending=False)
    output = t.gather(mat, dim=0, index=sample_loc.long().to(Device))
    return output


class AdaptFilter(nn.Module):
    def __init__(self, shape, degree):
        super(AdaptFilter, self).__init__()
        param_shape = (shape) + (degree,)
        self.weight = t.nn.Parameter(t.zeros(param_shape))

    def forward(self, x):
        x = t.sum(input=t.exp(self.weight * x), dim=-1, keepdims=True)
        return x


def curve_fit(y, mask, degree=6, batch=2, max_epoch=5000, mini_epoch=1000, lr=0.001, lr_decay=1, factor=1, eps=1e-6,
              resample=True, blRemoval=False, group_num=10, Device="cuda:1"):
    """
    y: [time, h, w, 1]
    """
    y_label = y.copy()
    time_axis, h, w, _ = y.shape
    if factor > 1:
        if factor % 2 != 0:
            raise ValueError
        y = upsample(y, factor).to(Device).float()
        time_axis = int(factor * time_axis)
        x_axis = t.arange(0, time_axis)
        x_input = repeat(x_axis, 'c -> c h w d', h=h, w=w, d=1)
    else:
        x_axis = t.arange(0, time_axis)
        x_input = repeat(x_axis, 'c -> c h w d', h=h, w=w, d=1)
        if not t.is_tensor(y):
            y = t.tensor(y).float()

    if not y.is_cuda:
        y = y.float().to(Device)

    if factor > 32 and resample:
        y = subsection(y, mask, group_num).unsqueeze(-1).float().to(Device)
        time_axis = len(y)
        x_axis = t.arange(0, time_axis)
        x_input = repeat(x_axis, 'c -> c h w d', h=h, w=w, d=1)

    # ------------ Normalization -------------
    ymin, _ = t.min(y, dim=0)
    ymax, _ = t.max(y, dim=0)
    y = (y - ymin) / (ymax - ymin + eps)
    poly_x = QR_features(x_input, degree).float().to(Device)
    poly_x = V(poly_x)
    adaptFilter = AdaptFilter((h, w), degree).to(Device)
    optimizer = optim.Adam(adaptFilter.parameters(), lr=lr)
    criterion = t.nn.MSELoss(reduction='mean').to(Device)

    scaler = t.cuda.amp.GradScaler()
    losses = []

    poly_x_train = poly_x.clone()
    y_train = y.clone()
    eps_mat = t.full((time_axis, 256, 256, 1), eps).to(Device)
    for epoch in tqdm(range(max_epoch)):
        rand_indices = t.randperm(time_axis)
        poly_x_train = poly_x_train[rand_indices]
        y_train = y_train[rand_indices]
        step = 0
        for ii in range(time_axis // batch):
            x_bat = poly_x_train[step: step + batch]
            y_bat = y_train[step: step + batch]
            with t.cuda.amp.autocast():
                y_pred = adaptFilter(x_bat)
                loss = criterion(y_pred, y_bat)

            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    adaptFilter.eval()
    time_axis = len(y_label)
    if factor == 1:
        eval_x = poly_x
    else:
        eval_x = t.arange(time_axis).float()
        eval_x = repeat(eval_x, 'c -> c h w d', h=h, w=w, d=1)
        eval_x = QR_features(eval_x, degree).to(Device)
    with t.no_grad():
        predi = t.zeros((time_axis, h, w, 1)).to(Device)
        for tt in range(time_axis):
            predi[tt: tt + 1] = adaptFilter(eval_x[tt: tt + 1])
        predi = predi * (ymax - ymin) + ymin
        predi = predi.squeeze(-1).permute(1, 2, 0)
    predi = predi.detach().cpu().numpy()
    return predi, losses


if __name__ == "__main__":
    png_path = "/data/aiteam_ctp/database/AIS_210713/0713_dst_png"
    dcm_path = "/data/aiteam_ctp/database/AIS_210713/0713_dst"

    case_list_dcm = getCaselst(dcm_path)
    case_list_png = getCaselst(png_path)

    # time_anat_imgs_dcm = load4DDcm(0, case_list_dcm)
    idx = -1
    time_anat_imgs = load4DImage(idx, case_list_png, 0)
    space = getSpace(idx, case_list_dcm)
    output, shifts, rotations = Registration(time_anat_imgs, space=space)
    # print(output.shape)
    brain_mask, vencle_mask = BoneRemove(output[0])
    anat_time_imgs = output.swapaxes(0, 1)
    print(anat_time_imgs.shape)
    input_y = anat_time_imgs[5].copy()
    # seriesImg_show(input_y)
    mask = brain_mask[5]
    input_y = get_tissue_signal(input_y, brain_mask[5], 3)
    input_y = smoothImg(input_y, kernel=3, stride=1, padding=1)
    input_y = np.expand_dims(input_y, axis=-1)


    print(predi.shape)
    # seriesImg_show(predi.transpose(-1, 0, 1))
    h, w = 512, 512
    for i in range(h * w):
        row = i // w
        col = i % w
        if mask[row, col] == 0:
            continue
        tmp = cv2.rectangle(anat_time_imgs[5, 0].copy().astype('uint8'), (col - 2, row - 2),
                            (col + 2, row + 2), (227, 133, 13), 3)
        plt.figure(figsize=(15, 7))
        display.clear_output(True)
        plt.subplot(1, 2, 1)
        plt.plot(predi[row, col], label='predi')
        plt.plot(anat_time_imgs[5, :, row, col], label='original')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.imshow(tmp)
        plt.show()
        plt.pause(0.1)
        plt.close()
