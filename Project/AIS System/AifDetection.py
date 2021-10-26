# import pdb
#
# import numpy as np
# import os
# import cv2
# import matplotlib.pyplot as plt
# import datetime
# import skimage.morphology
# from skimage import data, filters
# import Config as config
# from utils import Kmeans, add_box
# from skimage.measure import regionprops, label
# from skimage import transform, exposure
# from scipy import stats
#
#
# def ttp_mapping(tissue_signal):
#     ttp = np.argmax(tissue_signal, axis=-1)
#     return np.expand_dims(ttp, axis=-1)
#
#
# def cbf_mapping(tissue_signal):
#     cbf = np.max(tissue_signal, axis=-1)
#     cbf = np.expand_dims(cbf, axis=-1)
#     return cbf
#
#
# def halfRadius_mapping(tissue_signal, cbf_map):
#     half_cbf = cbf_map / 2
#     radius = np.sum(tissue_signal > half_cbf, axis=-1)
#     radius = np.expand_dims(radius, axis=-1)
#     return radius
#
#
# def bat_mapping(tissue_signal):
#     shape = config.Img_shape
#     bat = np.zeros(shape + (1,))
#     end = np.zeros_like(bat)
#     for i in range(shape[0] * shape[1]):
#         y = int(i // shape[0])
#         x = int(i % shape[1])
#         idx = np.argwhere(tissue_signal[y, x, :] > 0)
#         if len(idx) > 0:
#             bat[y, x, :] = idx[0]
#             end[y, x, :] = idx[-1]
#     return bat, end
#
#
# def computeMode(para_map, mask):
#     para_map_ = para_map[mask != 0]
#     mode = stats.mode(para_map_)[0][0]
#     return mode
#
#
# def param_mapping(param, mask, mode='less'):
#     mask_idx = np.where(mask != 0)
#     if param.shape[-1] == 1:
#         param = param.squeeze(-1)
#     param_seq = param[mask_idx]
#
#     if mode == 'less':
#         param_idx = np.where(param_seq < np.median(param_seq))
#     elif mode == 'big':
#         param_idx = np.where(param_seq > np.median(param_seq))
#     else:
#         raise ValueError
#
#     seq_copy = np.zeros_like(param_seq)
#     seq_copy[param_idx] = 1
#     param_img = np.zeros_like(mask)
#     param_img[mask_idx] = seq_copy
#     return param_img
#
#
# def normalization(param_map):
#     minest = np.min(param_map, axis=(0, 1))
#     maxest = np.max(param_map, axis=(0, 1))
#     param_norm = param_map / (maxest - minest)
#     return param_norm
#
#
# def otsu(param):
#     output = param.copy()
#     threshold = filters.threshold_otsu(param[param != 0])
#     output[param < threshold] = 0
#     return output
#
#
# def vessel_location_condition(param_norm, mask):
#     # (bat, radius, cbf, ttp, end)
#     region = 2 * param_norm[..., 2] - np.sum(param_norm, axis=-1)
#     region[mask != 0] = np.exp(region[mask != 0])
#     vessel_region = otsu(region)
#     vessel_region[vessel_region != 0] = 1
#     return vessel_region
#
#
# def vessel_locating_kmeans(param_map, mask):
#     # (bat, radius, cbf, ttp, end)
#     mask_idx = np.where(mask != 0)
#     param_norm = normalization(param_map)
#     # --------------Label Selection----------------------
#     region = param_norm[..., 2] - param_norm[..., 3]
#     region[mask_idx] = np.exp(region[mask_idx])
#     union = otsu(region)
#     union[union != 0] = 1
#     # --------------Normalization----------------
#     shape = config.Img_shape
#     x_seq = param_norm[mask_idx]
#     predi = Kmeans(x_seq, cluster=2)
#     label_map = predi.labels_ + 1
#     label_seq = label_map[union[mask_idx] == 1]
#     label = stats.mode(label_seq)[0][0]
#     # ----------------label --------------------------
#     fill = np.zeros_like(label_map)
#     fill[np.where(label_map == label)] = 1
#     location = np.zeros(shape)
#     location[mask_idx] = fill
#     return location
#
#
# def modeNum(param_map, mode='mean'):
#     mode = mode.lower()
#     if mode == 'mean':
#         method = np.mean
#     elif mode == 'max':
#         method = np.max
#     elif mode == 'min':
#         method = np.min
#     elif mode == 'median':
#         method = np.median
#     elif mode == 'mode':
#         method = lambda x: np.argmax(np.bincount(x.astype('int')))
#     else:
#         raise ValueError
#
#     param_map = param_map.copy()
#     ans = method(param_map[np.where(param_map != 0)])
#     return ans
#
#
# def openOperation(mask, erod_w=2, dilation_w=2):
#     dilation_kernal = skimage.morphology.square(dilation_w)
#     erod_kernel = skimage.morphology.square(erod_w)
#     erod = skimage.morphology.erosion(mask, erod_kernel)
#     dilation = skimage.morphology.dilation(erod, dilation_kernal)
#     return dilation
#
#
# def aif_filter(aif_mask, param_norm, outputNum=6):
#     # bat, radius, cbf, ttp, end
#     sort_conditions = np.sum(param_norm, axis=-1) - 2 * param_norm[..., 2]
#     aif_mask = openOperation(aif_mask, 2, 2)
#     label_image = label(aif_mask, connectivity=aif_mask.ndim)
#     props = regionprops(label_image)
#     priQue = []
#     for i in range(len(props)):
#         row, col = props[i].centroid
#         ymax = int(row + 1)
#         ymin = int(row - 1)
#         xmax = int(col + 1)
#         xmin = int(col - 1)
#         priQue.append(np.sum(sort_conditions[ymin:ymax + 1, xmin:xmax + 1]) / 9)
#     sortIdx = np.argsort(np.array(priQue))
#     props = np.array(props)
#     props = props[sortIdx]
#     output = np.zeros_like(aif_mask)
#     for i in range(min(outputNum, len(props))):
#         row, col = props[i].centroid
#         ymax = int(row + 1)
#         ymin = int(row - 1)
#         xmax = int(col + 1)
#         xmin = int(col - 1)
#         output[ymin:ymax + 1, xmin:xmax + 1] = 1
#     return output
#
#
# def creat_mask(Image, location, threshold):
#     location_idx = np.where(location == 1)
#     img_seq = Image[location_idx]
#     output_mask = np.zeros_like(Image)
#     fill = np.zeros_like(img_seq)
#     fill[np.where(img_seq > threshold)] = 1
#     output_mask[location_idx] = fill
#     return output_mask
#
#
# def aifDetection_Kmeans(tissue_signal, images, mask):
#     ttp = ttp_mapping(tissue_signal)
#     cbf = cbf_mapping(tissue_signal)
#     radius = halfRadius_mapping(tissue_signal, cbf)
#     bat, end = bat_mapping(tissue_signal)
#     param_map = np.concatenate((bat, radius, cbf, ttp, end), axis=-1)
#     param_norm = normalization(param_map)
#     vel_location = vessel_locating_kmeans(param_norm, mask)
#
#     # -----------------Vessel Mask---------------------------
#     idx = modeNum(ttp, mode='median')
#     idx = int(idx)
#     Image = images[idx].copy() / 255
#     vessel_location_idx = np.where(vel_location != 0)
#     vessel_seq = Image[vessel_location_idx]
#     vessel_seq = np.expand_dims(vessel_seq, axis=0)
#     vessel_threshold = filters.threshold_otsu(vessel_seq)
#     vessel_mask = np.zeros_like(vel_location)
#     vessel_fill = np.zeros_like(vessel_seq)
#     vessel_fill[np.where(vessel_seq > vessel_threshold)] = 1
#     vessel_mask[vessel_location_idx] = vessel_fill.squeeze(0)
#
#     aif_seq = Image[np.where(vessel_mask != 0)]
#     aif_seq = np.expand_dims(aif_seq, axis=0)
#     aif_threshold = filters.threshold_otsu(aif_seq)
#     aif_fill = np.zeros_like(aif_seq)
#     aif_fill[np.where(aif_seq > aif_threshold)] = 1
#     aif_location = np.zeros_like(vel_location)
#     aif_location[np.where(vessel_mask != 0)] = aif_fill
#
#     # ---------------Aif Mask----------------------
#     Image = images[idx].copy() / 255
#     ttp_mask = creat_mask(Image, aif_location, aif_threshold)
#
#     idx = (modeNum(ttp, mode='median') + modeNum(bat, mode='median')) // 2
#     idx = int(idx)
#     Image = images[idx].copy() / 255
#     aif_mask = creat_mask(Image, aif_location, aif_threshold)
#     aif_mask[ttp_mask == 1] = 1
#     aif_mask = aif_filter(aif_mask, param_norm, config.Num_Aif)
#     # ------------Compare aif mask with vof mask-----------------
#
#
#     return aif_location, aif_mask, vel_location, vessel_threshold, aif_threshold, idx
#
#
# def aifDetection_Condition(tissue_signal, images, mask):
#     ttp = ttp_mapping(tissue_signal)
#     cbf = cbf_mapping(tissue_signal)
#     radius = halfRadius_mapping(tissue_signal, cbf)
#     bat, end = bat_mapping(tissue_signal)
#     param_map = np.concatenate((bat, radius, cbf, ttp, end), axis=-1)
#     param_norm = normalization(param_map)
#     vessel_mask = vessel_location_condition(param_norm, mask)
#
#     idx = modeNum(ttp, mode='median')
#     idx = int(idx)
#     Image = images[idx].copy() / 255
#     vessel_threshold = np.mean(Image[vessel_mask != 0])
#     aif_location_idx = np.where(vessel_mask == 1)
#     aif_seq = Image[aif_location_idx]
#
#     aif_seq = np.expand_dims(aif_seq, axis=0)
#     aif_threshold = filters.threshold_otsu(aif_seq)
#     aif_location = np.zeros_like(vessel_mask)
#
#     aif_fill = np.zeros_like(aif_seq)
#     aif_fill[np.where(aif_seq > aif_threshold)] = 1
#     aif_location[aif_location_idx] = aif_fill.squeeze(0)
#
#     # -----------------Create Mask---------------------
#     Image = images[idx].copy() / 255
#     ttp_mask = creat_mask(Image, aif_location, aif_threshold)
#     idx = (modeNum(ttp, mode='median') + modeNum(bat, mode='median')) // 2
#     idx = int(idx)
#     Image = images[idx].copy() / 255
#     aif_mask = creat_mask(Image, aif_location, aif_threshold)
#     aif_mask[ttp_mask == 1] = 1
#
#     ttp = ttp_mapping(tissue_signal)
#     cbf = cbf_mapping(tissue_signal)
#     radius = halfRadius_mapping(tissue_signal, cbf)
#     bat, end = bat_mapping(tissue_signal)
#     param_map = np.concatenate((bat, radius, cbf, ttp, end), axis=-1)
#
#     aif_mask = aif_filter(aif_mask, param_map, config.Num_Aif)
#     return aif_location, aif_mask, vessel_threshold, aif_threshold, idx
#
#
# def RGBMapping(img, aif_mask, vessel_mask):
#     if len(img.shape) > 2 and img.shape[-1] != 3:
#         raise ValueError
#     rgb = img.copy().astype('uint8')
#     if len(rgb.shape) == 2:
#         rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
#     rgb[..., 0][vessel_mask == 1] = 255
#     rgb[..., 0][aif_mask == 1] = 255
#     rgb[..., 1][vessel_mask == 1] = 255
#     rgb[..., 1][aif_mask == 1] = 48
#     rgb[..., 2][vessel_mask == 1] = 0
#     rgb[..., 2][aif_mask == 1] = 48
#     return rgb
#
#
# def creatVideo(imgs, aif_threshold, vessel_threshold, vessel_location):
#     ans = np.zeros(imgs.shape + (3,)).astype('uint8')
#     for i in range(len(imgs)):
#         img = imgs[i].copy() / 255
#         vessel_idx = np.where(vessel_location != 0)
#         vessel_seq = img[vessel_idx]
#         vessel_fill = np.zeros_like(vessel_seq)
#         vessel_fill[np.where(vessel_seq > vessel_threshold)] = 1
#         vessel_mask = np.zeros_like(img)
#         vessel_mask[vessel_idx] = vessel_fill
#         aif_mask = np.zeros_like(img)
#         aif_idx = np.where(vessel_mask == 1)
#         aif_seq = img[aif_idx]
#         aif_fill = np.zeros_like(aif_seq)
#         aif_fill[np.where(aif_seq > aif_threshold)] = 1
#         aif_mask[aif_idx] = aif_fill
#         rgb = RGBMapping(img * 255, aif_mask, vessel_mask)
#         ans[i, ...] = rgb
#     return ans
#
#
# def saveVideo(video, name, path, timepoint, mask):
#     size = len(video)
#     path = os.path.join(path, '{}.avi'.format(name))
#     videoWriter = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), size, config.Img_shape)
#     for i in range(len(video)):
#         img = video[i]
#         if i == timepoint:
#             img = add_box(img, mask)
#         videoWriter.write(img)
#     return
#
#
# def saveImage(img, name, path):
#     cv2.imwrite(path + "/{}.png".format(name), img)
#
#
# def main(tissue_signal, imgs, mask):
#     start = datetime.datetime.now()
#     vessel_mask, aif_mask, vessel_location, vessel_threshold, aif_threshold, timepoint = \
#         aifDetection_Kmeans(tissue_signal, imgs, mask)
#     end = datetime.datetime.now()
#     print('Time: {}s'.format((end - start).seconds))
