import torch
from torch.utils.data import Dataset, DataLoader
import random
import glob
from pydicom import dcmread

import Config
from Transform import myTransform
import numpy as np
import cv2
import os
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from DataPreprocess import main
import math
from scipy import stats
from einops import repeat


def read_image_3D(case_path, msk_mode='2D'):
    name = case_path.split('/')[-1]
    patient, _, case, slc = name.split('_')
    mask_path = os.path.join(config.Mask_root, "{}_case_{}_{}_AifMask.png".format(patient, case, slc))
    img_lst = glob.glob(case_path + '/*.png')
    img_shape = (len(img_lst), ) + config.Image_shape[1:]
    image = np.zeros(img_shape)

    for idx, path in enumerate(img_lst):
        image[idx,...] = cv2.imread(path, 0)

    idx = np.random.choice(np.arange(config.Time_axis), size=config.Time_axis//2)

    if msk_mode == '3D':
        single_msk = np.expand_dims(cv2.imread(mask_path, 0), 0)
        mask = np.zeros((config.Time_axis,) + config.Mask_shape[1:])
        mask[idx, ...] = repeat(single_msk, '() h w-> b h w', b=len(idx))
    else:
        mask = cv2.imread(mask_path, 0)

    return image, mask


def read_image_2D(case_path):
    name = case_path.split('/')[-1]
    patient, _, case, slc = name.split('/')[-1].split('_')[0:4]
    mask_path = os.path.join(config.Mask_root, "{}_case_{}_{}_AifMask.png".format(patient, case, slc))
    image = cv2.imread(case_path, 0)
    mask = cv2.imread(mask_path, 0)
    return image, mask


def randomIdx(lst):
    return np.random.randint(len(lst))


def show_image(img, mask=None):
    if torch.is_tensor(img):
        img.numpy()
    img = img.reshape(config.Image_shape)
    mask = mask.reshape(config.Mask_shape)
    if img.shape[0] > 1:
        idx = randomIdx(img)
        print(idx)
    else:
        idx = 0
    if mask is not None:
        index = mask != 0
        index = index.squeeze(0)
        img[idx][index] = 0
        mask = mask.squeeze(0)
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img[idx])
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()


def data_split(img_root, ratio, shuffle=False):
    full_list = glob.glob(img_root + '/*_part/*')
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return full_list, []
    if shuffle:
        random.shuffle(full_list)
    train_lst = full_list[:offset]
    val_lst = full_list[offset:]
    return train_lst, val_lst


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


def resize(img, shape=config.Image_shape, mode='nearest'):
    size = img.shape
    if size[1] == config.Time_axis:
        return img
    if len(size) == 4:
        img = img.unsqueeze(0)
    img = F.interpolate(img, size=shape, mode=mode, align_corners=True)
    img = img.squeeze(0)
    return img


class UNETDataset(Dataset):
    def __init__(self, input_lst, Train=True):
        self.images_path = input_lst
        self.isTrain = Train

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img_path = self.images_path[index]
        self.name = img_path
        image, mask = read_image_3D(img_path, msk_mode=config.Msk_mode)
        if self.isTrain:
            image, mask = myTransform(image, mask)
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        resize_shape = (config.Time_axis, ) + config.Image_shape[1:]
        image = resize(image, shape=resize_shape, mode=config.Resize_mode)
        return F.normalize(image, dim=1), F.normalize(mask, dim=1)


class UNET2DDataset(Dataset):
    def __init__(self, img_lst, Train=True):
        super(UNET2DDataset, self).__init__()
        self.images_path = img_lst
        self.isTrain = Train

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img_path = self.images_path[index]
        image, mask = read_image_2D(img_path)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        if self.isTrain:
            image, mask = myTransform(image, mask)
        return image/255, mask/255


def test():
    train_lst, val_lst = data_split(config.Img_root, config.Train_percent, shuffle=True)
    dataset = UNETDataset(train_lst)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for ii, (x, y) in enumerate(train_loader):
        if ii > 1:
            break
        print(x.shape, y.shape)


if __name__ == "__main__":
    test()


