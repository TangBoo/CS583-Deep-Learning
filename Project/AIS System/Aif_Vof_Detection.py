from Realign import ellipse_fitting, getMaxArea
import math
import numpy as np
import cv2
import skimage
from skimage import filters, morphology, feature
from skimage.measure import regionprops, label
from collections import Counter
from scipy import stats
from sklearn.cluster import KMeans as kmeans
from tqdm import tqdm
import matplotlib.pyplot as plt


def otsu(param):
    output = param.copy()
    threshold = filters.threshold_otsu(param[param != 0])
    output[param < threshold] = 0
    return output


def creat_mask(Image, location, threshold):
    location_idx = np.where(location == 1)
    img_seq = Image[location_idx]
    output_mask = np.zeros_like(Image)
    fill = np.zeros_like(img_seq)
    fill[np.where(img_seq > threshold)] = 1
    output_mask[location_idx] = fill
    return output_mask


def openOperation(mask, erod_w=2, dilation_w=2):
    dilation_kernal = skimage.morphology.square(dilation_w)
    erod_kernel = skimage.morphology.square(erod_w)
    erod = skimage.morphology.erosion(mask, erod_kernel)
    dilation = skimage.morphology.dilation(erod, dilation_kernal)
    return dilation


def normalization(param_map):
    minest = np.min(param_map, axis=(0, 1))
    maxest = np.max(param_map, axis=(0, 1))
    param_norm = param_map / (maxest - minest)
    return param_norm


def ttp_mapping(tissue_signal):
    # [h, w, time]
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
    h, w, time = tissue_signal.shape  # [H, W]
    bat = np.zeros((h, w) + (1,))  # [H, W, 1]
    end = np.zeros_like(bat)  # [H, W, 1]
    for i in range(h * w):
        y = int(i // h)
        x = int(i % w)
        idx = np.argwhere(tissue_signal[y, x, :] > 0)
        if len(idx) > 0:
            bat[y, x, :] = idx.min()
            end[y, x, :] = idx.max()
    return bat, end


def computeMode(para_map, mask):
    para_map_ = para_map[mask != 0]
    mode = stats.mode(para_map_)[0][0]
    return mode


def vessel_locating_kmeans(param_map, mask):
    h, w = mask.shape
    # (bat, radius, cbf, ttp, end)
    mask_idx = np.where(mask != 0)
    param_norm = normalization(param_map)
    # --------------Label Selection----------------------
    region = param_norm[..., 2] - param_norm[..., 3]
    region[mask_idx] = np.exp(region[mask_idx])
    union = otsu(region)
    union[union != 0] = 1

    # --------------Normalization----------------
    shape = (h, w)
    x_seq = param_norm[mask_idx]
    predi = Kmeans(x_seq, cluster=2)
    label_map = predi.labels_ + 1
    label_seq = label_map[union[mask_idx] == 1]
    label_ = stats.mode(label_seq)[0][0]
    # ----------------label --------------------------
    fill = np.zeros_like(label_map)
    fill[np.where(label_map == label_)] = 1
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
    elif mode == 'median':
        method = np.median
    elif mode == 'mode':
        method = lambda x: np.argmax(np.bincount(x.astype('int')))
    else:
        raise ValueError

    param_map = param_map.copy()
    ans = method(param_map[np.where(param_map != 0)])
    return ans


def Kmeans(seq, cluster=2):
    seq = seq.reshape(-1, seq.shape[-1])
    y_pred = kmeans(n_clusters=cluster, init='k-means++', n_jobs=-1, precompute_distances=True).fit(seq)
    return y_pred


def aif_filter(aif_mask, param_norm, neighbor=True, outputNum=15):
    # bat, radius, cbf, ttp, end
    sort_conditions = np.sum(param_norm, axis=-1) - 2 * param_norm[..., 2]
    aif_mask = openOperation(aif_mask, 2, 2)
    label_image = label(aif_mask, connectivity=aif_mask.ndim)
    props = regionprops(label_image)
    priQue = []
    for i in range(len(props)):
        row, col = props[i].centroid
        if neighbor:
            priQue.append(np.sum(sort_conditions[int(row) - 1: int(row) + 1, int(col) - 1: int(col) + 1]) / 9)
        else:
            priQue.append(np.sum(sort_conditions[int(row), int(col)]))

    sortIdx = np.argsort(np.array(priQue))
    props = np.array(props)
    props = props[sortIdx]
    output = np.zeros_like(aif_mask)
    aif_location = []
    for i in range(min(outputNum, len(props))):
        row, col = props[i].centroid
        aif_location.append((row, col))
        output[int(row), int(col)] = 1
    return output, aif_location


def vofDetection(mask, meta):
    h, w = mask.shape
    ellipse, info = ellipse_fitting(mask)
    degree = info[-1]
    long_radius = info[1][1]

    center = meta[0]
    angle = meta[-1]
    angle = degree - angle
    if angle > 90:
        angle = 180 - angle
    half_radius = long_radius / 2
    degree = meta[-1]

    x_shift = center[0] * math.tan(math.radians(degree))
    x_top = center[0] + x_shift
    x_bottom = center[0] - x_shift
    line_mask = np.zeros_like(mask)
    cv2.line(line_mask, (int(x_top), 0), (int(x_bottom), h), (255, 0, 0), 1)
    line_mask *= mask

    if degree > 90:
        t_x = int(center[0] + half_radius * math.sin(math.radians(degree)))
        t_y = int(center[1] - half_radius * math.cos(math.radians(degree)))
    else:
        t_x = int(center[0] - half_radius * math.sin(math.radians(degree)))
        t_y = int(center[1] + half_radius * math.cos(math.radians(degree)))
    t_y, t_x = min(h - 1, t_y), min(w - 1, t_x)

    while mask[t_y, t_x] == 0:
        if degree > 90:
            t_x = int(t_x - 5 * math.sin(math.radians(degree)))
            t_y = int(t_y + 5 * math.cos(math.radians(degree)))
        else:
            t_x = int(t_x + 5 * math.sin(math.radians(degree)))
            t_y = int(t_y - 5 * math.cos(math.radians(degree)))
    return t_y, t_x


def aifDetection_Kmeans(tissue_signal, images, mask, output_num=10):
    ttp = ttp_mapping(tissue_signal)
    cbf = cbf_mapping(tissue_signal)
    radius = halfRadius_mapping(tissue_signal, cbf)
    bat, end = bat_mapping(tissue_signal)
    ignore = np.where(bat < 3)
    ttp[ignore] = 0
    cbf[ignore] = 0
    radius[ignore] = 0
    bat[ignore] = 0
    end[ignore] = 0
    param_map = np.concatenate((bat, radius, cbf, ttp, end), axis=-1)
    param_norm = normalization(param_map)
    # (bat, radius, cbf, ttp, end)
    vel_location = vessel_locating_kmeans(param_norm, mask)

    # -----------------Vessel Mask---------------------------
    idx = modeNum(ttp, mode='median')
    idx = int(idx)
    Image = images[idx].copy() / 255
    vessel_location_idx = np.where(vel_location != 0)
    vessel_seq = Image[vessel_location_idx]
    vessel_seq = np.expand_dims(vessel_seq, axis=0)
    vessel_threshold = filters.threshold_otsu(vessel_seq)
    vessel_mask = np.zeros_like(vel_location)
    vessel_fill = np.zeros_like(vessel_seq)
    vessel_fill[np.where(vessel_seq > vessel_threshold)] = 1
    vessel_mask[vessel_location_idx] = vessel_fill.squeeze(0)

    aif_seq = Image[np.where(vessel_mask != 0)]
    aif_seq = np.expand_dims(aif_seq, axis=0)
    aif_threshold = filters.threshold_otsu(aif_seq)
    aif_fill = np.zeros_like(aif_seq)
    aif_fill[np.where(aif_seq > aif_threshold)] = 1
    aif_location = np.zeros_like(vel_location)
    aif_location[np.where(vessel_mask != 0)] = aif_fill

    # ---------------Aif Mask----------------------
    Image = images[idx].copy() / 255
    ttp_mask = creat_mask(Image, aif_location, aif_threshold)

    idx = (modeNum(ttp, mode='median') + modeNum(bat, mode='median')) // 2
    idx = int(idx)
    Image = images[idx].copy() / 255
    aif_mask = creat_mask(Image, aif_location, aif_threshold)
    aif_mask[ttp_mask == 1] = 1
    aif_mask, aif_coord = aif_filter(aif_mask, param_norm, neighbor=True, outputNum=output_num)
    return aif_mask, aif_coord


def compare_avif(mainPeak, anat_time_imgs, brain_mask, final_mask):
    d, time, h, w = anat_time_imgs.shape
    ceil = np.linspace(d // 3, d, 3).astype('int')[1]
    floor = np.linspace(d // 4, d, 4).astype('int')[0]
    slia = anat_time_imgs[:, 0, ...].max(axis=0)
    mask, _ = getMaxArea(slia)
    _, info = ellipse_fitting(mask)
    ans_aif = np.array([])
    ans_vof = np.array([])
    aif_coor = ()
    vof_coor = ()
    max_res = 0
    idx = 0
    for i in tqdm(range(floor, ceil + 1)):
        vof_y, vof_x = vofDetection(brain_mask[i].astype('uint8'), meta=info)
        aif_input = mainPeak[i].transpose(1, 2, 0)
        aif_mask, coor = aifDetection_Kmeans(aif_input, anat_time_imgs[i], final_mask[i], 15)
        vof = mainPeak[i, :, vof_y, vof_x]
        max_y, max_x = 0, 0
        max_cbf = 0
        vof_bat = np.argwhere(vof > 0)[0][0]
        for j in range(len(coor)):
            y, x = int(coor[j][0]), int(coor[j][1])
            tmp_aif = mainPeak[i, :, y, x]
            if len(np.argwhere(tmp_aif > 0)) == 0:
                continue
            else:
                tmp_bat = np.argwhere(tmp_aif > 0)[0][0]
            if max(tmp_aif) - tmp_bat > max_cbf:
                max_y, max_x = y, x
                max_cbf = max(tmp_aif) - tmp_bat

        aif = mainPeak[i, :, max_y, max_x]
        aif_bat = np.argwhere(aif > 0)[0][0]
        aif_vof_res = vof_bat - aif_bat
        if aif_vof_res > max_res:
            max_res = aif_vof_res
            ans_aif = aif
            ans_vof = vof
            aif_coor = (max_y, max_x)
            vof_coor = (vof_y, vof_x)
            idx = i
    return ans_aif, ans_vof, aif_coor, vof_coor, idx


def show_aif_vof(show_aif_vof_img, aif, vof, aif_coor, vof_coor, path, name=None):
    aif_y, aif_x = aif_coor
    vof_y, vof_x = vof_coor
    show_aif_vof_img = ((show_aif_vof_img - show_aif_vof_img.min()) / (
                show_aif_vof_img.max() - show_aif_vof_img.min())) * 255
    show_aif_vof_img = cv2.cvtColor(show_aif_vof_img.astype('uint8'), cv2.COLOR_GRAY2BGR)
    show_aif_vof_img = cv2.rectangle(show_aif_vof_img.astype('uint8'), (aif_x - 2, aif_y - 2),
                                     (aif_x + 2, aif_y + 2), (227, 255, 13), 3)
    show_aif_vof_img = cv2.rectangle(show_aif_vof_img.astype('uint8'), (vof_x - 2, vof_y - 2),
                                     (vof_x + 2, vof_y + 2), (227, 133, 13), 3)
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(show_aif_vof_img)
    plt.subplot(1, 2, 2)
    plt.plot(aif, label='aif')
    plt.plot(vof, label='vof')
    plt.legend()
    plt.savefig(path + '/{}.jpg'.format(name))
    plt.close()
