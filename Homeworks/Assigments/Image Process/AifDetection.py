import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import datetime
import skimage.morphology
from skimage import data, filters
import Config as config
from utils import Kmeans, add_box


def ttp_mapping(tissue_signal):
    ttp = np.argmax(tissue_signal, axis=-1)
    return np.expand_dims(ttp, axis=-1)


def cbf_mapping(tissue_signal):
    cbf = np.max(tissue_signal, axis=-1)
    cbf = np.expand_dims(cbf, axis=-1)
    return cbf


def halfRadius_mapping(tissue_signal, cbf_map):
    half_cbf = cbf_map / 2
    radius = np.sum(tissue_signal > half_cbf, axis=-1)
    radius = np.expand_dims(radius, axis=-1)
    return radius


def bat_mapping(tissue_signal):
    shape = config.Img_shape
    bat = np.zeros(shape + (1,))
    end = np.zeros_like(bat)
    for i in range(shape[0] * shape[1]):
        y = int(i // shape[0])
        x = int(i % shape[1])
        idx = np.argwhere(tissue_signal[y, x, :] > 0)
        if len(idx) > 0:
            bat[y, x, :] = idx[0]
            end[y, x, :] = idx[-1]
    return bat, end


def vessel_locating(param_map, mask):
    shape = config.Img_shape
    mask_idx = np.where(mask != 0)
    x_seq = param_map[mask_idx]
    predi = Kmeans(x_seq, cluster=2)
    label_map = predi.labels_
    label = np.argmin(np.bincount(label_map))
    fill = np.zeros_like(label_map)
    fill[np.where(label_map == label)] = 1
    location = np.zeros(shape)
    location[mask_idx] = fill
    return location


def modeNum(param_map, mode='mean'):
    mode = mode.lower()
    if mode == 'mean':
        method = np.mean
    elif mode == 'max':
        method = np.max
    elif mode == 'min':
        method = np.min
    elif mode == 'mode':
        method = lambda x: np.argmax(np.bincount(x.astype('int')))
    else:
        raise ValueError

    param_map = param_map.copy()
    minest = np.min(param_map[param_map != 0])
    maxest = np.max(param_map[param_map != 0])
    param_map[np.where(param_map == minest)] = 0
    param_map[np.where(param_map == maxest)] = 0
    ans = method(param_map[np.where(param_map != 0)])
    return int(ans)


def openOperation(mask):
    dilation_kernal = skimage.morphology.square(2)
    erod_kernel = skimage.morphology.square(2)
    erod = skimage.morphology.erosion(mask, erod_kernel)
    dilation = skimage.morphology.dilation(erod, dilation_kernal)
    return dilation


def aifDetection(tissue_signal, images, mask):
    ttp = ttp_mapping(tissue_signal)
    cbf = cbf_mapping(tissue_signal)
    radius = halfRadius_mapping(tissue_signal, cbf)
    bat, end = bat_mapping(tissue_signal)
    param_map = np.concatenate((bat, radius, cbf, ttp, end), axis=-1)
    vessel_location = vessel_locating(param_map, mask)

    # -----------------Vessel Mask---------------------------
    idx = (modeNum(bat, mode='mode') + modeNum(ttp, mode='mode')) // 2
    Image = images[idx].copy() / 255
    vessel_location_idx = np.where(vessel_location != 0)
    vessel_seq = Image[vessel_location_idx]
    vessel_seq = np.expand_dims(vessel_seq, axis=0)
    vessel_threshold = filters.threshold_otsu(vessel_seq)
    vessel_mask = np.zeros_like(vessel_location)
    vessel_fill = np.zeros_like(vessel_seq)
    vessel_fill[np.where(vessel_seq > vessel_threshold)] = 1
    vessel_mask[vessel_location_idx] = vessel_fill.squeeze(0)

    # ---------------Aif Mask----------------------
    idx = modeNum(bat, mode='mean')
    Image = images[idx].copy() / 255
    aif_location_idx = np.where(vessel_mask == 1)
    aif_seq = Image[aif_location_idx]
    aif_seq = np.expand_dims(aif_seq, axis=0)
    aif_threshold = filters.threshold_otsu(aif_seq)
    aif_mask = np.zeros_like(vessel_location)
    aif_fill = np.zeros_like(aif_seq)
    aif_fill[np.where(aif_seq > aif_threshold)] = 1
    aif_mask[aif_location_idx] = aif_fill.squeeze(0)

    aif_mask = openOperation(aif_mask)
    vessel_mask = openOperation(vessel_mask)
    return vessel_mask, aif_mask, vessel_location, vessel_threshold, aif_threshold, idx


def RGBMapping(img, aif_mask, vessel_mask):
    if len(img.shape) > 2 and img.shape[-1] != 3:
        raise ValueError
    rgb = img.copy().astype('uint8')
    if len(rgb.shape) == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    rgb[..., 0][vessel_mask == 1] = 255
    rgb[..., 0][aif_mask == 1] = 255
    rgb[..., 1][vessel_mask == 1] = 255
    rgb[..., 1][aif_mask == 1] = 48
    rgb[..., 2][vessel_mask == 1] = 0
    rgb[..., 2][aif_mask == 1] = 48
    return rgb


def creatVideo(imgs, aif_threshold, vessel_threshold, vessel_location):
    ans = np.zeros(imgs.shape + (3, )).astype('uint8')
    for i in range(len(imgs)):
        img = imgs[i].copy() / 255
        vessel_idx = np.where(vessel_location != 0)
        vessel_seq = img[vessel_idx]
        vessel_fill = np.zeros_like(vessel_seq)
        vessel_fill[np.where(vessel_seq > vessel_threshold)] = 1
        vessel_mask = np.zeros_like(img)
        vessel_mask[vessel_idx] = vessel_fill
        aif_mask = np.zeros_like(img)
        aif_idx = np.where(vessel_mask == 1)
        aif_seq = img[aif_idx]
        aif_fill = np.zeros_like(aif_seq)
        aif_fill[np.where(aif_seq > aif_threshold)] = 1
        aif_mask[aif_idx] = aif_fill
        rgb = RGBMapping(img * 255, aif_mask, vessel_mask)
        ans[i, ...] = rgb
    return ans


def saveVideo(video, name, path, timepoint, mask):
    size = len(video)
    path = os.path.join(path, '{}.avi'.format(name))
    videoWriter = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), size, config.Img_shape)
    for i in range(len(video)):
        img = video[i]
        if i == timepoint:
            img = add_box(img, mask)
        videoWriter.write(img)
    return


def main(tissue_signal, imgs, mask):
    start = datetime.datetime.now()
    vessel_mask, aif_mask, vessel_location, vessel_threshold, aif_threshold, timepoint = aifDetection(tissue_signal,
                                                                                                      imgs, mask)
    end = datetime.datetime.now()
    print('Time: {}s'.format((end - start).seconds))


