#!/usr/bin/env python
# coding: utf-8

# In[247]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import datetime
from scipy.linalg import circulant
from scipy.linalg import solve_circulant
from scipy.linalg import solve
from scipy import signal
from scipy.fftpack import fft, ifft
from IPython import display
import pybeads as be
from scipy import sparse
from scipy.sparse.linalg import spsolve
from BaselineRemoval import BaselineRemoval
from matplotlib.pyplot import MultipleLocator 
import copy
import xml.dom.minidom
import json
from scipy.signal import find_peaks
import joblib
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.cluster import KMeans as kmeans
from numpy import trapz
from skimage import data, filters
from skimage import transform, exposure
from skimage.measure import regionprops, label
import skimage
from scipy.signal import detrend
import torch as t
import torch.nn as nn
from torch import tensor
from torch.autograd import Variable as V
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import optim
import os
import itk
from itkwidgets import view
import SimpleITK as sitk
from ipywidgets import interact, fixed
from IPython.display import clear_output
import math
from glob import glob
from PIL import Image
from numba import jit
import torch
svd = np.linalg.svd


# In[248]:


from BrainSegmentation.BrainBoneRemove import seg_brain


# In[249]:


path = "/data/aiteam_ctp/database/AIS_210713/0713_dst_png"


# In[250]:


def getCaselst(root):
    patient_lst = glob(path + '/*/')
    case_lst = []
    for i in range(len(patient_lst)):
        temp_lst = glob(patient_lst[i] + '/*/')
        for j in range(len(temp_lst)):
            if temp_lst[j].endswith('png/'):
                continue
            case_lst.append(temp_lst[j])
    return case_lst

def load4DImage(idx, case_lst):
    slc_lst = glob(case_lst[idx]+'/*/')
    slc_lst = sorted(slc_lst, key=lambda x:x.split('/')[-2].split('_')[0])
    #[d, time, h, w]
    anat_imgs = []
    for i in range(len(slc_lst)):
        time_imgs = []
        time_lst = glob(slc_lst[i] + '/*')
        time_lst = sorted(time_lst, key=lambda x:x.split('/')[-1].split('.')[0].split('img')[-1])
        for j in range(len(time_lst)):
            time_imgs.append(cv2.imread(time_lst[j]))
        anat_imgs.append(time_imgs)
    return np.array(anat_imgs).swapaxes(0, 1)


# In[251]:


def BoneRemove(imgs):
    D, H, W, _ = imgs.shape
    bone_lst, ventricle_lst = seg_brain(imgs, 256, 256, 12)
    bone_masks = [cv2.resize(bone_lst[i], (H, W), interpolation=cv2.INTER_CUBIC) 
                  for i in range(len(imgs))]
    ventricle_masks = [cv2.resize(ventricle_lst[i], (H, W), interpolation=cv2.INTER_CUBIC)
                      for i in range(len(imgs))
                      ]
    return np.array(bone_masks).astype('bool'), np.array(ventricle_masks).astype('bool')


# In[252]:


def seriesImg_show(imgs, time = 0.8):
    plt.figure(figsize=(15, 7))
    for i in range(len(imgs)):
        clear_output(wait = True)
        plt.imshow(imgs[i])
        plt.show()
        plt.pause(time)
        plt.close()
        
def Viewer_3D(imgs):
    itk_img = itk.image_view_from_array(imgs)
    viewer = view(itk_img, rotate=True, axes=True, vmin=0, vmax=255, gradient_opacity=0.9)
    return viewer

def getMaxArea(img):
    label_image = label(img, connectivity=img.ndim)
    props = regionprops(label_image)
    areas = [prop.area for prop in props]
    labels = [prop.label for prop in props]
    index = labels[areas.index(max(areas))]
    ignore_location = np.where(label_image != index)
    img[ignore_location] = 0
    return img, ignore_location

def ellipse_fitting(contour):
    contour = contour.copy()
    contour[contour>0] = 255
    edge = cv2.Canny(contour.astype('uint8'), 0, 1)
    y, x = np.nonzero(edge)
    edge_list = np.array([[x_, y_] for x_, y_ in zip(x, y)])
    
    _ellipse = cv2.fitEllipse(edge_list)
    edge_clone = edge.copy()
    cv2.ellipse(edge_clone, _ellipse, (255, 0, 0), 2)
    edge_clone[edge!=0] = 0
    return edge_clone, _ellipse


def mutualInfo_rigister_3D(fixed, moving, space=(1, 1, 1)):
    '''
    :param fixed: template image
    :param moving: registration image
    :param space: dcm space
    :return: moving img
    '''
    fixed = sitk.GetImageFromArray(fixed)
    fixed.SetSpacing(space)
    moving = sitk.GetImageFromArray(moving)
    moving.SetSpacing(space)
    initial_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    optimized_transform = sitk.Euler3DTransform()
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform, inPlace=False)
    final_transform = registration_method.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                                                  sitk.Cast(moving, sitk.sitkFloat32))
    output = sitk.Resample(moving, final_transform)
    output = sitk.GetArrayFromImage(output)
    return output


def rigid_registration_3D(imgs, slice_num=37, template_shift=(0, 0, 0), template_rotation=(0, 0, 0), isRotate=True,
                          space=(1, 1, 1)):
    '''
    :param imgs: [D, H, w]
    :param slice_num: D
    :param template_shift: the shift of last image on x, y, z
    :param template_rotation: the rotation of last image around x, y, z
    :param isRotate: first image do not rotate, following image will shift on the center of first image
    :param space: dcm space
    :return: imgs
    '''
    ori_imgs = imgs.copy()
    imgs = imgs.copy()
    d, h, w = imgs.shape
    Slia = np.max(imgs, axis=0)
    erod_kernel = np.ones((3, 3), np.uint8)
    Slia = cv2.erode(Slia, erod_kernel)
    Slia, ignore_loc = getMaxArea(Slia)
    for i in range(len(imgs)):
        imgs[i][ignore_loc] = 0

    Sagittal = np.max(imgs, axis=1)
    Coroa = np.max(imgs, axis=2)

    _, Sila_info = ellipse_fitting(Slia)
    _, Sagi_info = ellipse_fitting(Sagittal)
    _, Coroa_info = ellipse_fitting(Coroa)

    (x, y) = Sila_info[0]
    z = Coroa_info[0][1]
    theta_x, theta_y, theta_z = Sagi_info[-1], Coroa_info[-1], Sila_info[-1]

    if not isRotate:
        rotate_out = (theta_x, theta_y, theta_z)
    else:
        theta_x = theta_x - template_rotation[0]
        theta_y = theta_y - template_rotation[1]
        theta_z = theta_z - template_rotation[2]
        rotate_out = (theta_x, theta_y, theta_z)

    offset = (-space[0] * (w // 2 - x), -space[1] * (h // 2 - y), -space[2] * (d // 2 - slice_num // 2))  # (x, y, z)
    # roatation axis: r_x, r_y, r_z, roatation center: z, y, x
    translation = sitk.TranslationTransform(3, offset)
    imgs_sitk = sitk.GetImageFromArray(ori_imgs)  # [z, y, x]->[y, x, z]
    imgs_sitk.SetSpacing(space)
    out = sitk.Resample(imgs_sitk, transform=translation)
    shift_out = (x - template_shift[0], y - template_shift[1], z - template_shift[2])
    return sitk.GetArrayFromImage(out), shift_out, rotate_out


def Registration(time_anat_img, space=(0.412109375, 0.412109375, 1.0)):
    '''
    :param time_anat_img: [time, d, h, w]
    :param space: dcm space
    :return: time_anat_img
    '''
    template_img, template_shift, template_rotate = rigid_registration_3D(time_anat_img[0], template_shift=(0, 0, 0),
                                                                          template_rotation=(0, 0, 0), isRotate=False,
                                                                          space=space)
    t, d, h, w = time_anat_img.shape
    output_shift = {'x': [0, ], 'y': [0, ], 'z': [0, ]}
    output_rotation = {'x': [0, ], 'y': [0, ], 'z': [0, ]}
    output = [template_img]
    for i in tqdm(range(1, t)):
        temp_img, temp_shift, temp_rotate = rigid_registration_3D(time_anat_img[i], template_shift=template_shift,
                                                                  template_rotation=template_rotate, space=space,
                                                                  isRotate=True)
        temp_img = mutualInfo_rigister_3D(template_img, temp_img, space)
        output_shift['x'].append(temp_shift[0])
        output_shift['y'].append(temp_shift[1])
        output_shift['z'].append(temp_shift[2])
        output_rotation['x'].append(temp_rotate[0])
        output_rotation['y'].append(temp_rotate[1])
        output_rotation['z'].append(temp_rotate[2])
        template_img = temp_img
        output.append(temp_img)
    output = np.array(output)
    return output, output_shift, output_rotation


def plot_moveInfor(shifts, rotations):
    x_shift = np.array(shifts['x'])
    y_shift = np.array(shifts['y'])
    z_shift = np.array(shifts['z'])
    x_rotate = np.array(rotations['x'])
    y_rotate = np.array(rotations['y'])
    z_rotate = np.array(rotations['z'])
    x_shift[1:] -= x_shift.mean()
    y_shift[1:] -= y_shift.mean()
    z_shift[1:] -= z_shift.mean()
    x_rotate[1:] -= x_rotate.mean()
    y_rotate[1:] -= y_rotate.mean()
    z_rotate[1:] -= z_rotate.mean()
    plt.figure(figsize=(15, 7))
    plt.subplot(3, 2, 1)
    plt.plot(x_shift)
    plt.axis([0, 30, -10, 10])
    plt.title('x shift')
    plt.subplot(3, 2, 2)
    plt.plot(x_rotate)
    plt.axis([0, 30, -2, 2])
    plt.title('x rotation')
    plt.subplot(3, 2, 3)
    plt.plot(y_shift)
    plt.axis([0, 30, -10, 10])
    plt.title('y shift')
    plt.subplot(3, 2, 4)
    plt.plot(y_rotate)
    plt.axis([0, 30, -2, 2])
    plt.title('y rotation')
    plt.subplot(3, 2, 5)
    plt.plot(z_shift)
    plt.axis([0, 30, -10, 10])
    plt.title('z shift')
    plt.subplot(3, 2, 6)
    plt.plot(z_rotate)
    plt.axis([0, 30, -2, 2])
    plt.title('z rotation')
    plt.show()


# In[253]:


#Data Processing:
import copy
def get_ct_value_neighbor_avg(imgs, x, y, d = 3):
    # img [time, h, w]
    time, h, w = imgs.shape
    x_start = max(x - d, 0)
    x_end = min(x + d, w)
    y_start = max(y - d, 0)
    y_end = min(y + d, h)
    res = imgs[..., y_start : y_end, x_start : x_end] # [time, 2d, 2d]
    count = (2 * d) ** 2 - np.sum(res[0, ...] == 0) 
    return np.sum(res, axis=(1, 2)) / count


def get_ct_value_avg_aif(img, x_start, x_end, y_start, y_end):
    """
    """
    ct_value_sum = 0
    n = 0
    for x_i in range(x_start-4, x_end+4):
        for y_i in range(y_start-4, y_end+4):
            ct_value_sum += img[y_i, x_i]
            n += 1
    
    ct_value_avg = ct_value_sum / n
    return ct_value_avg


# In[254]:


def cal_average_aif_value(img_grey, x_aif, y_aif, w_aif, h_aif):
    """This is to get the manually selected AIF area's average value.

    Args:
        - img_grey (array): gray scale numpy array with shape (y, x).
        - x_aif (int): x start of AIF area.
        - y_aif (int): y start of AIF area.
        - w_aif (int): width of AIF area.
        - h_aif (int): height of AIF area.

    Returns:
        - ct_value_avg (float): average aif pixel value.
    """

    ct_value_sum = 0
    n = 0
    for x_i in range(x_aif, x_aif+w_aif):
        for y_i in range(y_aif, y_aif+h_aif):
            ct_value_sum += img_grey[y_i, x_i]
            n += 1
    
    ct_value_avg = ct_value_sum / n
    return ct_value_avg


# In[255]:


# Signal Pre-Process Helper
def baseline_process(array_src, std_t):
    """
    
    """
    list_src = array_src.tolist()
    list_dst = []
    list_temp = []
    mean_flag, std_flag = 0, 0
    n = 1
    for x in list_src:
        list_temp.append(x)
        mean, std_dev = cal_mean_stddev(list_temp)
        mean_flag, std_flag = mean, std_dev
        n += 1
        if std_flag >= std_t:
            break
        else:
            list_dst.append(x)
    mean, std_dev = cal_mean_stddev(list_dst)
    list_dst_2 = [i - mean + 5 for i in list_src]
    return np.array(list_dst_2)


def cal_mean_stddev(list_data):
    """
    """
    mean_numerator = 0
    for x in list_data:
        mean_numerator += x
    mean = mean_numerator / len(list_data)

    variance_numerator = 0
    for x in list_data:
        variance_numerator += ((x - mean) ** 2)
    std_dev = (variance_numerator / len(list_data)) ** 0.5
    return mean, std_dev


def filter_lp(data, cutoff_l, cutoff_h, ftype):
    """
    low pass filter
    """
    if ftype=="lowpass":
        b, a = signal.butter(9, cutoff_h, 'lowpass')
    elif ftype=="bandpass":
        b, a = signal.butter(7, [cutoff_l, cutoff_h], 'bandpass')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData


# In[256]:


#Signal Process Helper
def svd_denoising(array_1d):
    """
    """
    matrix_cir = circulant(array_1d)
    matrix_cir_sparse = sparse_matrix(matrix_cir)
    u, sigma, vt = np.linalg.svd(matrix_cir_sparse)

    threshold = sigma[2]*0.10

    sigma_denoised = np.zeros([len(array_1d), len(array_1d)])
    for i in range(len(array_1d)):
        if sigma[i] > threshold:
            sigma_denoised[i][i] = sigma[i]
        else:
            sigma_denoised[i][i] = 0

    matrix_cir_sparse_new = np.dot(np.dot(u, sigma_denoised), vt)
    array_1d_new = matrix_cir_sparse_new[:, 0]
    return array_1d_new


def deconv_nonparam_alg_tikhonov_svd(array_ct_value_aif, array_ct_value_tissue, lamdaa):
    """
    Deconvolution-based / Non-parametric / Tikhonov SVD 
    
    """
    a = array_ct_value_aif
    c = array_ct_value_tissue

    a_pad = np.pad(a, (0, len(a)))
    c_pad = np.pad(c, (0, len(a)))

    I = np.identity(2*len(a))

    A = circulant(a_pad)
    A_T = A.T
    
    block_cirMat = np.dot(A_T, A) + (lamdaa ** 2) * I 
    b = solve(block_cirMat, A_T @ c_pad)

    b = b[0:len(a)]
    return b

@jit(nopython=True)
def drive_signal(seq):
    power = np.zeros_like(seq)
    for i in range(len(power) - 1):
        res = seq[i + 1] - seq[i]
        if res > 0:
            power[i] = 1
        elif res == 0:
            power[i] = 0
        else:
            power[i] = -1

    for j in range(1, len(power) - 1):
        res = power[j]
        if res == 0 and power[j - 1] > 0:
            power[j] = 1
        elif res == 0 and power[j - 1] < 0:
            power[j] = -1
    return power

@jit(nopython=True)
def find_mainPeak(seq):
    power = drive_signal(seq)
    sec_power = np.zeros_like(seq)
    for i in range(1, len(power)):
        sec_power[i] = power[i] - power[i - 1]
        
    ans_left = 0
    ans_right = 0
    max_length = 0
    for i in range(len(sec_power)):
        if sec_power[i] != -2:
            continue
        left = i
        right = i
        while left > 0 and sec_power[left] != 2:
            left -= 1
        while right < len(sec_power) - 1 and sec_power[right] != 2:
            right += 1
        if (right - left + 1) >= max_length:
            ans_left = left
            ans_right = right
    return ans_left, ans_right


def find_valley_peak(seq):
    '''
    Input:
        signal sequence
    Output:
        power signal
    '''
    # take the derivative twice for getting movement trend
    fir_p = drive_signal(seq)
    sec_p = drive_signal(fir_p)
    return sec_p


def find_mostRight(seq, target = -1):
    max_idx = 0
    max_len = 0
    temp = 0    
    for i in range(len(seq)):
        if seq[i] == target:
            temp += 1
        else:
            if temp > max_len:
                max_idx = i - 1
                max_len = temp
            temp = 0
    return max_idx


def find_longestLine(seq, target = -1):
    '''
    Input:
        Power signal
    Output:
        start and end valley around peak
    Descirbe:
        Shift windows, O(n)
    '''
    q_p = 0
    s_p = 0
    max_len = 0
    max_r = 0
    while s_p < len(seq):
        if seq[s_p] == target:
            q_p = s_p
            while q_p + 1< len(seq) and seq[q_p + 1] == target:
                q_p += 1 
            if (q_p - s_p) >= max_len:
                max_len = q_p - s_p
                max_r = q_p
            s_p = q_p + 1
        else:
            s_p += 1
    return max(max_r - max_len, 0), max_r


def truncate_tissue(signal_seq):
    '''
    input:
        signal sequence
    output:
        the left and right boundary indexes of peak
    '''
    #peak is -1, valley is 1
    sec_power = find_valley_peak(signal_seq) #power signal
    left_side = np.min(np.where(sec_power == 1))
    peak_p = np.argmax(signal_seq[left_side: ]) + left_side
    
    _, right_side = find_longestLine(sec_power[peak_p: ], target = -1) + peak_p
    
    return left_side, right_side


def truncate_aif(signal_seq):
    sec_power = find_valley_peak(signal_seq) #power signal
    left_side = np.min(np.where(sec_power == 1)) + 1 # get first 1 in power signal
    right_side = find_mostRight(sec_power)
    
    return left_side, right_side


def baseline_correction(signal_seq, name = "tissue"):
    '''
    input :
        signal sequence
    output :
        res : peak of signal
        base_signal: signal sequence without baseline shift
        
    '''
    base_signal = BaselineRemoval(signal_seq).IModPoly(2) #getting off baseline shift
    
    if name == "tissue":
        left_side, right_side = truncate_tissue(base_signal) # get the left and right boundary of peak indexes
    else:
        left_side, right_side = truncate_aif(base_signal)
        
    res = copy.deepcopy(base_signal) 
    
    #--- pick peak ---
    
#     res -= res[left_side - 1]
    
    res[:left_side] = 0
    res[right_side + 1:] = 0
    res[res < 0] = 0
    return res, base_signal

@jit(nopython=True)
def smoothImg(imgs, kernel=3, stride=1, padding = 1):
    if padding != 0:
        if padding >= kernel:
            raise ValueError
        imgs = np.pad(imgs, ((padding, padding), (0, 0), (0, 0)), mode = 'edge')
    totalNum = len(imgs)
    if totalNum < kernel:
        raise ValueError
    
    sliceNum = (totalNum - kernel) // stride + 1
    pivot_ = kernel // 2
    output_shape = (sliceNum, ) + imgs.shape[1:]
    output = np.zeros(output_shape)
    for slid in range(sliceNum):
        pivot = pivot_ + slid * stride
        left = pivot - kernel // 2
        right = pivot + (kernel // 2 if kernel % 2 != 0 else 1)
        output[slid, ...] = np.mean(imgs[left:right + 1], axis = 0)
    return output    


# In[257]:


#Compute IRF:
def sparse_matrix(matrix_array):
    # matrix_array_c = circulant(matrix_array)
    matrix_c_list = matrix_array.tolist()

    matrix_sparse = np.zeros(shape=matrix_array.shape)
    n1 = 1
    for r in matrix_c_list:
        n2 = 1
        for i in r:
            if n2 > n1:
                matrix_sparse[n1-1, n2-1] = 0
            else:
                matrix_sparse[n1-1, n2-1] = i
            n2 += 1
        n1 += 1
    matrix_array_sparse = np.array(matrix_sparse)
    return matrix_array_sparse


def deconv_circulant_matrix_fourier(array_ct_value_aif, array_ct_value_tissue):
    """
    aif * r = tissue, 
    r = solve(aif, tissue)
    """

    # residual_func = solve_circulant(array_ct_value_aif, array_ct_value_tissue, singular='lstsq')
    array_ct_value_aif = svd_denoising(array_ct_value_aif)
    residual_func = solve(circulant(array_ct_value_aif), array_ct_value_tissue)
    # residual_func = ifft(fft(array_ct_value_tissue) / fft(array_ct_value_aif))
    return residual_func


# In[258]:


def sparse_mat(seq, lamda):
    '''
    Tikhonov svd
    input:
        aif signal, superparameter of regularization
    output:
        new aif matrix with lower condition number 
    '''
    if seq.shape[0] == 1:
        seq = seq.squeeze(axis = 0)
    length = len(seq) 
    res = np.zeros((length, length))
    
    for i in range(length):
        res[i, 0] = seq[i]
        if i != 0:
            res[i, 1:i + 1] = res[i - 1, :i]
    new_mat = (res.T @ res + (lamda ** 2) * np.diag([1] * len(res))) * np.linalg.inv(res.T)
    print(np.linalg.cond(new_mat))
    return res

def segmentation(tmaxes):
    '''
    input:
        tmaxes : (256, 256)
    output:
        scale tmaxes for mapping to [0, 255]
    
    '''
    min_val = tmaxes.min()
    max_val = tmaxes.max()
    tmaxes = (tmaxes - min_val) / (max_val - min_val)
    return tmaxes * 255

def joblib_process(sub_f, *args):
    result = []
    with parallel_backend('multiprocessing', n_jobs = -1):
        res = Parallel()(delayed(sub_f)(*[img, args[0], args[1]]) for img in args[0])
    return res

def Kmeans(seq, num_f = 1, cluster = 2):
    seq =seq.reshape(-1, seq.shape[-1])
    y_pred = kmeans(n_clusters = cluster, init = 'k-means++', n_jobs = -1).fit(seq)
    return y_pred

def find_secMin(seq):
    res = [float('inf')] * 2
    ans = 0
    pre_ans = 0
    for i in range(len(seq)):
        if res[0] > seq[i]:
            res[1] = res[0]
            ans = pre_ans
            res[0] = seq[i]
            pre_ans = i
        elif res[1] > seq[i]:
            res[1] = seq[i]
            ans = i
    return ans


# In[259]:


def filter_lp(data, cutoff_l, cutoff_h, ftype):
    """
    low pass filter
    """
    from scipy import signal
    if ftype=="lowpass":
        b, a = signal.butter(9, cutoff_h, 'lowpass')
    elif ftype=="bandpass":
        b, a = signal.butter(7, [cutoff_l, cutoff_h], 'bandpass')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData

def deconvolution(array_ct_value_aif, array_ct_value_tissue, show=False):
    cir_aif = circulant(array_ct_value_aif)
    inv_sig = regulSig(cir_aif)
#     print('Condition Num:',np.linalg.cond(inv_sig))
    irf = inv_sig @ array_ct_value_tissue
    if show:
        return irf, inv_sig[0]
    else:
        return irf

def regulSig(cir_mat):
    size = cir_mat.shape[0]   
    peakposi = np.argmax(cir_mat[:, 0])
    cir_mat = np.linalg.inv(cir_mat)
    
    
    for ii in range(size):
        head = cir_mat[ii][:ii]
        tail = cir_mat[ii][ii:]
        cir_mat[ii,:] = np.r_[tail, head]
    ans = cir_mat.mean(0)
    
    peaks, properties = find_peaks(ans, prominence = 0, width = [0, 2])
    left_bases = properties['left_bases']
    right_bases = properties['right_bases']
    idex = np.argmax(ans[peaks])
    
    left = left_bases[idex] if abs(left_bases[idex] - peakposi) < abs(right_bases[idex - 1] - peakposi) else right_bases[idex - 1]
    right = right_bases[idex] if abs(right_bases[idex] - peakposi) < abs(left_bases[idex + 1] - peakposi) else left_bases[idex + 1]
    
    leftpart = ans[:left]
    rightpart = ans[right:]
    midpart = ans[left : right]
    
    leftpart = cv2.GaussianBlur(leftpart, (1, 3), 0.7)
    rightpart = cv2.GaussianBlur(rightpart, (1, 3), 0.7)
    ans = np.r_[leftpart.squeeze(-1), midpart, rightpart.squeeze(-1)]
    return circulant(ans).T


# In[269]:



def poly_features(x, order=6):
    time, h, w, _ = x.shape
    if not t.is_tensor(x):
        x = t.tensor(x)
    features = t.ones((time, h, w, 1), requires_grad=True)
    for i in range(1, order):
        features = t.cat((features, x ** i), dim=-1)
    return features


def QR_features(x, degree = 6):
    # x:[time, h, w, 1]
    time, h, w, _ = x.shape
    poly_x = poly_features(x, degree + 1).permute(1, 2, 0, -1) # [h, w, time, features]
    ans = t.zeros((h, w, time, degree))
    for i in range(256 * 256):
        row = i // 256
        col = i % 256
        ans[row, col] = t.qr(poly_x[row, col])[0][:, 1:]
    return ans.permute(2, 0, 1, -1)


# In[270]:


@jit(nopython=True)
def find_mainPeak(seq):
    power = drive_signal(seq)
    sec_power = np.zeros_like(seq)
    for i in range(1, len(power)):
        sec_power[i] = power[i] - power[i - 1]
        
    ans_left = 0
    ans_right = 0
    max_length = 0
    for i in range(len(sec_power)):
        if sec_power[i] != -2:
            continue
        left = i
        right = i
        while left > 0 and sec_power[left] != 2:
            left -= 1
        while right < len(sec_power) - 1 and sec_power[right] != 2:
            right += 1
        if (right - left + 1) >= max_length:
            ans_left = left
            ans_right = right
    return ans_left, ans_right


def matMultiplier(A, B):
    aRow, aCol, aH, aW = A.shape
    bRow, bCol, bH, bW = B.shape
#     assert aCol == bRow, "{} != {}".format(aCol, bRow)
#     assert aH == bH and aW == bW, "{}, {}, {}, {}".format(aH, bH, aW, bW)
    ans = t.zeros((aRow, bCol, aH, aW))
    for i in range(aRow):
        a = A[i, :, ...]
        for j in range(bCol):
            b = B[:, j]
            c = a * b
            ans[i, j] = c.sum(0, keepdims=True)
    ans = ans.to(A.device)
    return ans


def curve_fitting(Y, degree=6, Device="cpu", eps=1e-6):
    time, h, w, _ = Y.shape
    x_axis = t.arange(0, time)
    x_input = repeat(x_axis, 'c -> c h w d', h=h, w=w, d=1)
    poly_x = QR_features(x_input, degree).float().to(Device).permute(0, -1, 1, 2)
    y = t.tensor(Y).to(Device).permute(0, -1, 1, 2)
    ymin, _ = t.min(y, dim=0, keepdims=True)
    ymax, _ = t.max(y, dim=0, keepdims=True)
    y = (y - ymin) / (ymax - ymin + eps)
    main_peak_val = None
    predi = AdaptFilter_leastsqr().fit(poly_x, y).predict(poly_x)
    predi = predi * (ymax - ymin) + ymin
    return predi.cpu().detach().numpy()

class AdaptFilter_leastsqr:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.weight = None
        self.ymin, self.ymax = None, None
    
    def fit(self, x, y):
        time, degree, h, w = x.shape # x :[time, degree, h, w]
        x_t = x.permute(1, 0, 2, 3) # [degree, time, 256, 256]
        xTx = matMultiplier(x_t, x) # [degree, degree, 256, 256]
        xTx_inv = t.zeros_like(xTx)
        for i in range(h * w):
            row = i // h
            col = i % w
            try:
                xTx_inv[..., row, col] = t.inverse(xTx[..., row, col])
            except:
                xTx_inv[..., row, col] = t.pinverse(xTx[..., row, col])
                
        profix = matMultiplier(xTx_inv, x_t) # [degree, time, 256, 256]
        self.weight = matMultiplier(profix, y) # [degree, 1, 256, 256]
#         self.weight += 0.001 * t.norm(self.weight, p=1, dim=0)
        return self

    def predict(self, X):
        # X : [time, degree, 256, 256]
        ans = matMultiplier(X, self.weight)
        return ans

def baseline_correction(Y, mask, degree=2, repitition = 10, Device="cpu", eps=1e-6):
    if not t.is_tensor(mask):
        mask = t.tensor(mask)
        
    mask = mask.unsqueeze(0).unsqueeze(0)
    time, _, h, w = Y.shape
    x_axis = t.arange(1, time + 1)
    x_input = repeat(x_axis, 'c -> c h w d', h=h, w=w, d=1)
    poly_x = QR_features(x_input, degree).float().to(Device).permute(0, -1, 1, 2)
    y = t.tensor(Y).to(Device)
    y_label = y.clone()
    ymin, _ = t.min(y, dim=0, keepdims=True)
    ymax, _ = t.max(y, dim=0, keepdims=True)
    y = (y - ymin) / (ymax - ymin + eps)
    main_peak_val = None
    prev_std = 0
    model = AdaptFilter_leastsqr()
    for i in range(repitition):
        predi = model.fit(poly_x, y).predict(poly_x)
        std = t.std((y - predi), axis=0, keepdims=True)
        if i == 0:
            main_peak_index = (y > (predi + std)) # [time, h, w, 1], bool
            main_peak_val = (predi + std)[main_peak_index]

        y = t.minimum(y, (predi + std))
        y_train = y
        if main_peak_val is not None:
            y_train[main_peak_index] = main_peak_val
        condition = t.mean((abs(std - prev_std) / (std + eps))[np.where(mask != 0)])
        if condition < 0.001:
                    break
        prev_std = std
    ans = model.predict(poly_x)
    ans = ans * (ymax - ymin + eps) + ymin
    return ans.cpu().detach().numpy()


# In[271]:


def param_mapping(param, mask, mode = 'less'):
    mask_idx = np.where(mask != 0)
    if param.shape[-1] == 1:
        param = param.squeeze(-1)
    param_seq = param[mask_idx]
    condition = modeNum(param, mode='mean')
    if mode == 'less':
        param_idx = np.where(param_seq < condition)
    elif mode == 'big':
        param_idx = np.where(param_seq > condition)
    else:
        raise ValueError
        
    seq_copy = np.zeros_like(param_seq)
    seq_copy[param_idx] = 1
    param_img = np.zeros_like(mask)
    param_img[mask_idx] = seq_copy
    return param_img

def normalization(param_map):
    minest = np.min(param_map, axis = (0, 1))
    maxest = np.max(param_map, axis = (0, 1))
    param_norm = param_map / (maxest - minest) 
    return param_norm

def otsu(param):
    output = param.copy()
    threshold = filters.threshold_otsu(param[param!=0])
    output[param < threshold] = 0
    return output


def vessel_location_condition(param_norm, mask):
    region = 2 * param_norm[..., 2] - np.sum(param_norm, axis=-1)
    region[mask!=0] = np.exp(region[mask!=0]) 
    plt.figure()
    plt.imshow(region)
    plt.show()
    vessel_region = otsu(region)
    vessel_region[vessel_region!=0] = 1
    plt.figure()
    plt.imshow(vessel_region)
    plt.show()
    return vessel_region


# In[272]:


import skimage
from collections import Counter
from scipy import stats
Img_shape = (256, 256)
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
    h, w, time = tissue_signal.shape # [H, W]
    bat = np.zeros((h, w) + (1,)) # [H, W, 1]
    end = np.zeros_like(bat) # [H, W, 1]
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
    #(bat, radius, cbf, ttp, end)
    mask_idx = np.where(mask != 0)
    param_norm = normalization(param_map)
    # --------------Label Selection----------------------
    region = param_norm[..., 2] - param_norm[..., 3] 
    region[mask_idx] = np.exp(region[mask_idx]) 
    union = otsu(region)
    union[union!=0] = 1

    # --------------Normalization----------------
    shape = Img_shape
    x_seq = param_norm[mask_idx]
    predi = Kmeans(x_seq, cluster=2)
    label_map = predi.labels_ + 1
    label_seq = label_map[union[mask_idx] == 1]
    label = stats.mode(label_seq)[0][0]
    # ----------------label --------------------------
#     label = stats.mode(label_map)[0][0]
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
    elif mode == 'median':
        method = np.median
    elif mode == 'mode':
        method = lambda x: np.argmax(np.bincount(x.astype('int')))
    else:
        raise ValueError

    param_map = param_map.copy()
    ans = method(param_map[np.where(param_map != 0)])
    return ans

def Kmeans(seq, num_f = 1, cluster = 2):
    seq =seq.reshape(-1, seq.shape[-1])
    y_pred = kmeans(n_clusters = cluster, init = 'k-means++', n_jobs = -1, precompute_distances=True).fit(seq)
    return y_pred

def openOperation(mask, erod_w = 2, dilation_w = 2):
    dilation_kernal = skimage.morphology.square(dilation_w)
    erod_kernel = skimage.morphology.square(erod_w)
    erod = skimage.morphology.erosion(mask, erod_kernel)
    dilation = skimage.morphology.dilation(erod, dilation_kernal)
    return dilation


def aif_filter(aif_mask, param_norm, neighbor=True, outputNum=6):
    #bat, radius, cbf, ttp, end
    sort_conditions = np.sum(param_norm, axis=-1) - 2 * param_norm[..., 2]
    aif_mask = openOperation(aif_mask, 2, 2)
    label_image = label(aif_mask, connectivity=aif_mask.ndim)
    props = regionprops(label_image)
    priQue = []
    for i in range(len(props)):
        row, col = props[i].centroid
        if neighbor:
            priQue.append(np.sum(sort_conditions[row - 1 : row + 1, col - 1 : col + 1]) / 9)
        else:
            priQue.append(np.sum(sort_conditions[row, col]) )
        
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


def aifDetection_Kmeans(tissue_signal, images, mask, output_num=10):
    ttp = ttp_mapping(tissue_signal)
    cbf = cbf_mapping(tissue_signal)
    radius = halfRadius_mapping(tissue_signal, cbf)
    bat, end = bat_mapping(tissue_signal)
    param_map = np.concatenate((bat, radius, cbf, ttp, end), axis=-1)
    param_norm = normalization(param_map)
    #(bat, radius, cbf, ttp, end)
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
    aif_seq = np.expand_dims(aif_seq, axis = 0)
    aif_threshold = filters.threshold_otsu(aif_seq)
    aif_fill = np.zeros_like(aif_seq)
    aif_fill[np.where(aif_seq > aif_threshold)] = 1
    aif_location = np.zeros_like(vel_location)
    aif_location[np.where(vessel_mask!=0)] = aif_fill
    
    # ---------------Aif Mask----------------------
    Image = images[idx].copy() / 255
    ttp_mask = creat_mask(Image, aif_location, aif_threshold)
    
    idx = (modeNum(ttp, mode='median') + modeNum(bat, mode='median')) // 2
    idx = int(idx)
    Image = images[idx].copy() / 255
    aif_mask = creat_mask(Image, aif_location, aif_threshold)
    aif_mask[ttp_mask == 1] = 1
    aif_mask, aif_coord = aif_filter(aif_mask, param_norm, output_num)
    return aif_mask, aif_coord

def creat_mask(Image, location, threshold):
    location_idx = np.where(location == 1)
    img_seq = Image[location_idx]
    output_mask = np.zeros_like(Image)
    fill = np.zeros_like(img_seq)
    fill[np.where(img_seq > threshold)] = 1
    output_mask[location_idx] = fill
    return output_mask

def vofDetection(mask):
    h, w = mask.shape
    _, info = ellipse_fitting(mask)
    short_radius = info[1][0]
    long_radius = info[1][1]
    center = info[0]
    degree = info[-1]
    y_shift = (h - center[0]) *math.tan(math.radians(degree))
    x_shift = center[0] * math.tan(math.radians(degree))
    x_top = center[0] + x_shift
    x_bottom = center[0] - x_shift
    line_mask = np.zeros_like(mask)
    cv2.line(line_mask, (int(x_top), 0), (int(x_bottom), 255), (1, 0, 0), 1)
    line_mask *= mask
    half_radius = long_radius / 2 - (long_radius - line_mask.sum()) - 3
    t_x = int(center[0] - half_radius * math.sin(math.radians(degree)))
    t_y = int(center[1] + half_radius * math.cos(math.radians(degree)))
    return t_y, t_x


# In[280]:


def signal_process(tissue_signal, val_mask, moveDown=False, device="cpu"):
    time, h, w = tissue_signal.shape
    ori_signal_mat = np.expand_dims(tissue_signal, axis=-1)
    print('Curve Fitting...')
    simulation = curve_fitting(ori_signal_mat, degree=6, Device=device) # [time, 1, h, w]
    print('Baseline Removal...')
    baseline = baseline_correction(simulation, val_mask, degree=2, repitition=5, Device=device) # [time, 1, h, w]        
    signal_mat = (simulation - baseline).squeeze(1)  # [time, h, w]
    mainPeak_mat = np.zeros_like(tissue_signal)
    sig2show = np.zeros_like(tissue_signal)
    for i in tqdm(range(h * w)):
        row = i // h
        col = i % w
        if val_mask[row, col] == 0:
            continue
        temp = signal_mat[..., row, col].copy()
        left, right = find_mainPeak(temp)
        if moveDown:
            res = max(temp[left], temp[right])
        else:
            res = temp[left]

        signal_mat[..., row, col] -= res
        temp -= res
        temp[temp < 0] = 0
        temp[:left] = 0
        sig2show[..., row, col] = temp.copy()
        temp[right:] = 0
        mainPeak_mat[..., row, col] = temp
    return signal_mat, mainPeak_mat, sig2show


# In[ ]:


def joblib_process(sub_f, *args):
    result = []
    with parallel_backend('multiprocessing', n_jobs = -1):
        res = Parallel()(delayed(sub_f)(*[img, args[0], args[1]]) for img in args[0])
    return res


# In[281]:


class AisSystem:
    def __init__(self, caseIdx, caselst, g_std_t, lamda, SmoothImage = True):
        time_anat_imgs = load4DImage(caseIdx, caselst) #[time, D, H, W]
        print('Brain Bone Remove......')
        self.brain_mask, self.ventricle_mask = BoneRemove(time_anat_imgs[0]) # [D, H, W]
        # [time, d, h, w], rgb->gray
        time_anat_imgs = time_anat_imgs[..., 0] * 0.11 + time_anat_imgs[..., 1] * 0.59 + time_anat_imgs[..., 2] * 0.3
        print('Image Registration......')
        self.time_anat_imgs, self.shifts, self.rotations = Registration(time_anat_imgs)
        self.final_mask = self.brain_mask.copy()
        self.final_mask[self.ventricle_mask==1] = 0
        self.anat_time_imgs = (self.time_anat_imgs * np.expand_dims(self.final_mask, axis=0)).swapaxes(0, 1)
        # [d, t, h, w]
        d, t, h, w = self.anat_time_imgs.shape
        print(self.anat_time_imgs.shape)
        tissue_imgs = []
        signal_imgs=[]
        sig2show_imgs = []
        mainPeak_imgs=[]
        self.vof_signal = []
        self.vof_location = []
        self.aif_signal = []
        self.aif_location = []
        print('Signal Process......')
        for i in tqdm(range(d)):
            print('Smooth Image...')
            if SmoothImage:
                tmp_img = smoothImg(self.anat_time_imgs[i], kernel=3, stride=1, padding=3//2)
            else:
                tmp_img = self.anat_time_imgs[i]
            
            print('Obtain Tissue...')
            if g_std_t > 1:
                tmp_tissue = get_tissue_signal(tmp_img, self.final_mask[i], g_std_t)
            else:
                tmp_tissue = tmp_img
            print('Signal Signal Process...')
            tmp_signal_mat, tmp_mainPeak_mat, tmp_sig2show = signal_process(tmp_tissue, self.final_mask[i])
            print('Vof Detection...')
            vof_y, vof_x = vofDetection(self.brain_mask[i])
            print('Aif Detection...')
            _, (aif_y, aif_x) = aifDetection_Kmeans(tmp_mainPeak_mat, self.anat_time_imgs[i], self.final_mask[i], 15)
            
            tissue_imgs.append(temp_tissue)
            signal_imgs.append(tmp_signal_mat)
            mainPeak_imgs.append(tmp_mainPeak_mat)
            sig2show_imgs.append(tmp_sig2show)
            self.vof_location.append((vof_y, vof_x))
            self.vof_signal.append(mainPeak_mat[..., vof_y, vof_x])
            self.aif_location.append((aif_y, aif_x))
            self.aif_signal.append(mainPeak_mat[..., aif_y, aif_x])
            
        self.tissue_img = np.array(tissue_imgs)
        self.signal_imgs = np.array(signal_imgs)
        self.mainPeak_imgs = np.array(mainPeak_imgs)
        self.sig2show_imgs = np.array(tmp_sig2show)
       
        
    def get_tissue_signal(self, img, mask, g_d):
        t, h, w = img.shape
        output = np.zeros((t, h, w))
        for ii in range(h * w):
            y_t_i = int(ii // h)
            x_t_i = int(ii % w)
            if mask[y_t_i, x_t_i] == 0:
                continue 
            output[..., y_t_i, x_t_i] = get_ct_value_neighbor_avg(img, x_t_i, y_t_i, g_d)
        return output
        
    
    def deconvolution(self, array_ct_value_aif, array_ct_value_tissue):
        cir_aif = circulant(array_ct_value_aif)
        inver_sig = np.linalg.inv(cir_aif)
        sig = inver_sig[0, :]
        peaks, properties = find_peaks(sig, prominence = 0)
        left_bases = properties['left_bases']
        right_bases = properties['right_bases']
        idex = np.argmax(sig[peaks])
        sig[:left_bases[idex]] = 0
        sig[right_bases[idex]:] = 0
        cir_sig = circulant(sig)
        cir_sig = cir_sig.T
        irf = cir_sig @ array_ct_value_tissue
        return irf
        
        
    def compute_irf(self, index):
        '''
        Input: 
            index:the index of pixel in flat image
            isPad: expand time axis for extensive distribution of Tmax
        Output: Tmax of irf
        
        Describe:
                Compute irf using Tikhonov svd
        '''
        y_t_i = int(index // 256)
        x_t_i = int(index % 256)
        
        if self.brain_mask[y_t_i, x_t_i] == 0:
            return 
        
        tissue_lst = [0] * len(self.img_lst)
        for idx, img in enumerate(self.images):
            tissue_lst[idx] = get_ct_value_neighbor_avg(img, x_t_i, y_t_i, self.brain_mask, self.img_mask, g_d)
        
            
        array_ct_value_tissue = np.array(tissue_lst)     
        array_ct_value_tissue_bl = baseline_process(array_ct_value_tissue, g_std_t)
        array_ct_value_tissue_bl_f = filter_lp(array_ct_value_tissue, cutoff_l = None, cutoff_h = 0.41, ftype = "lowpass")

        final_signal_tissue, base_tissue = baseline_correction(array_ct_value_tissue_bl_f)
        residual_func = deconvolution(self.final_signal_aif, final_signal_tissue)
        
        t_idx = np.argmax(residual_func)
        self.irf_img[y_t_i, x_t_i] += t_idx
        self.cbf_img[y_t_i, x_t_i] += np.max(residual_func)
        
       
    
    def compute_irf_time(self, index):
        '''
        Input: 
            index:the index of pixel in flat image
            isPad: expand time axis for extensive distribution of Tmax
        output: Tmax of irf
        
        Describe:
            if the time of peak in tissue is after the time of peak in aif, tmax = tissue_peak_time - aif_peak_time
            else tmax = total_time - (aif_peak_time - tissue_peak_time)
        '''
        y_t_i = int(index // 256)
        x_t_i = int(index % 256)
        
        if self.brain_mask[y_t_i, x_t_i] == 0:
            return
        
        length = len(self.final_signal_aif)
            
        tissue_lst = [0] * length
        for idx, img in enumerate(self.images):
            tissue_lst[idx] = get_ct_value_neighbor_avg(img, x_t_i, y_t_i, self.brain_mask, self.img_mask, g_d)
        
        array_ct_value_tissue = np.array(tissue_lst)     
        array_ct_value_tissue_bl = baseline_process(array_ct_value_tissue, g_std_t)
        array_ct_value_tissue_bl_f = filter_lp(array_ct_value_tissue, cutoff_l = None, cutoff_h = 0.41, ftype = "lowpass")
        final_signal_tissue, base_tissue = baseline_correction(array_ct_value_tissue_bl_f)
        
        
        aif_delay = np.min(np.where(self.final_signal_aif > 0))
        tissue_delay = np.min(np.where(final_signal_tissue > 0))
        
        irf_delay = abs(tissue_delay - aif_delay)
        
        aif_peak = np.argmax(self.final_signal_aif)
        tissue_peak = np.argmax(final_signal_tissue)
        max_aif = np.max(self.final_signal_aif)
        self.cbf_img[y_t_i, x_t_i] = (1 / max_aif) * np.max(final_signal_tissue)
        self.cbv_img[y_t_i, x_t_i] = trapz(final_signal_tissue, np.arange(len(final_signal_tissue)), dx = 1) / trapz(self.final_signal_aif, np.arange(len(self.final_signal_aif)), dx = 1)
        
        if tissue_peak >= aif_peak:
            self.irf_img[y_t_i, x_t_i] += (tissue_peak - aif_peak)

            
        
    def obtain_irf(self, nWorkers = 1, method = compute_irf):
        '''
        nWorkers = -1, using all available cpu resource
        '''
        task_index = []
        for ii in tqdm(range(0, 256 * 256)):
            if self.brain_mask[int(ii // 256), int(ii % 256)] == 0:
                continue
            task_index.append(ii)            
        Parallel(n_jobs = nWorkers, backend = 'threading')([delayed(method)(i) for i in tqdm(task_index)])
        return segmentation(self.irf_img), segmentation(self.cbf_img), segmentation(self.cbv_img)
    
    def obtain_tissue_tmax(self):
        cbf_seq = np.zeros((256 * 256))
        tmax_seq = np.zeros((256 * 256))
        comb_seq = np.zeros((256 * 256))
        for ii in tqdm(range(256 * 256)):
            y = int(ii // 256)
            x = int(ii % 256)
            if self.brain_mask[y, x] == 0:
                continue
                
            tissue_lst = [0] * len(self.img_lst)
            for idx, img in enumerate(self.images):
                tissue_lst[idx] = get_ct_value_neighbor_avg(img, x, y, self.brain_mask, self.img_mask, g_d)

            array_ct_value_tissue = np.array(tissue_lst)     
            array_ct_value_tissue_bl = baseline_process(array_ct_value_tissue, g_std_t)
            array_ct_value_tissue_bl_f = filter_lp(array_ct_value_tissue, cutoff_l = None, cutoff_h = 0.41, ftype = "lowpass")
            final_signal_tissue, base_tissue = baseline_correction(array_ct_value_tissue_bl_f)
            tmax_seq[ii] = np.argmax(final_signal_tissue)
            cbf_seq[ii] = final_signal_tissue.max()
            comb_seq[ii] = tmax_seq[ii] * cbf_seq[ii]
            
        return tmax_seq.reshape((256, 256)), cbf_seq.reshape((256, 256)), comb_seq.reshape((256, 256))
            
    
    def postProcess(self, image, ptype = 'irf'):
        '''
        Input: Image scale in [0, 255]
        Ouput:
               Binary Image
        Discribe:
            1. Get labels from kmeans func that map tmax to [0, 4] or [0, 3].
            2. Get second minimum value and minimum value from labels.
            3. Let all value is not equal to min and sec_min be zero
            #---irf---
            4. Morphological operation.
            5. Let color in output image be uniform.
        '''    
        if ptype == 'irf':
            ncluster = 3
        elif ptype == 'cbf':
            ncluster = 2
            
        predi = Kmeans(image, cluster = ncluster)
        labels = predi.labels_
        num_label = [len(labels[labels == i]) for i in set(labels)]
        color = np.argmin(num_label)
        
        lesion = np.zeros_like(image)
        for idx, v in enumerate(labels):
            lesion[int(idx // 256), int(idx % 256)] = labels[idx] if labels[idx] == color else 0
            
        if ptype == 'cbf':
            res = self.brain_mask.copy()
            res[lesion != 0] = 0
            return res 
        res = image.copy()
        kernel = np.ones((3,3), 'uint8')
        lesion_dialation = cv2.dilate(lesion, kernel, iterations = 1)
        lesion_erode = cv2.erode(lesion_dialation, kernel, iterations = 4)
        lesion_dilation = cv2.dilate(lesion_erode, kernel, iterations = 3)
        res ^= lesion_dilation
        return res
        
        
    def show_R_signal(self):
        
#         final_signal_aif, base_aif = baseline_correction(self.aif_seq, name = "aif")
        
        for x_t_i in range(100, 256):
            for y_t_i in range(100, 256):
                if self.brain_mask[y_t_i, x_t_i] == 0:
                    continue
                
                #---draw contours---
                img = cv2.imread(r"{}/{}".format(self.img_path, self.img_lst[0]), cv2.IMREAD_COLOR)
#                 img = cv2.drawContours(img, self.aif_contours, 0, (153, 0, 153), 1)
#                 img = cv2.drawContours(img, self.mask_contours, 0, (255, 0, 0), 1)
                cv2.rectangle(img, (max(0, x_t_i - 2), max(y_t_i - 2, 0)), 
                              (min(256, x_t_i + 2),min(256, y_t_i + 2)), (227,23,13), 1)
                
                
                #---Tissue TDC---
                
                
                with parallel_backend('threading', n_jobs = -1):
                    tissue_lst = Parallel()(delayed(get_ct_value_neighbor_avg)(*[img, x_t_i, y_t_i, self.brain_mask, self.img_mask, g_d])                                                                  for img in self.images) 
                    
                array_ct_value_tissue = np.array(tissue_lst)  
                array_ct_value_tissue_bl = baseline_process(array_ct_value_tissue, g_std_t)
                array_ct_value_tissue_bl_f = filter_lp(array_ct_value_tissue_bl, cutoff_l = None, cutoff_h = 0.41, ftype = "lowpass")
                final_signal_tissue, base_tissue = baseline_correction(array_ct_value_tissue_bl_f)
                
               
                
                #--- Compute IRF ---
#                 residual_func = deconv_nonparam_alg_tikhonov_svd(self.final_signal_aif, final_signal_tissue, lamdaa = self.lamda)
#                 residual_func_blur, sig = deconvolution(self.final_signal_aif, final_signal_tissue, show=True)
                
                #--- Show Main Peak ---
#                 residual_func_baseline  = BaselineRemoval(residual_func).IModPoly(2)
#                 residual_func_baseline[residual_func_baseline < 0] = 0

#                 peaks, properties = find_peaks(residual_func_baseline, prominence = 0)

#                 left_bases = properties['left_bases']
#                 right_bases = properties['right_bases']
#                 idex = np.argmax(residual_func_baseline[peaks])

#                 residual_func_baseline[:left_bases[idex]] = 0
#                 residual_func_baseline[right_bases[idex]:] = 0
                
                
#                 tissue_t = np.argmax(final_signal_tissue)
#                 aif_t = np.argmax(final_signal_aif)
#                 print("tissue tmax:", tissue_t, "aif tmax:", aif_t, "irf tmax:", tissue_t - aif_t)
                
                #inverse aif
#                 cir_aif = circulant(self.final_signal_aif)
#                 inver_sig = np.linalg.inv(cir_aif)
#                 o_sig = inver_sig[0, :].copy()
                  
    
                
#                 pdb.set_trace()
                #--- Plot Image ---
                plt.figure(figsize=(15, 7))
                
                display.clear_output(wait = True)
                plt.subplot(2, 4, 1)
                plt.imshow(img)
#                 plt.title("{}, x:{}, y:{}".format(x_t_i, y_t_i))
                
#                 plt.subplot(2, 4, 2)
#                 plt.plot(residual_func, label = "irf signal")
# #                 plt.plot(residual_func_blur, label = "irf_blur signal")
#     #             plt.legend()
#                 plt.minorticks_on()
#                 plt.title('IRF')

                plt.subplot(2, 4, 2)
                plt.plot(array_ct_value_tissue_bl, label = "array_tissue_signal_f")
                plt.plot(base_tissue, label = "tissue without baseline shift")
                plt.plot(final_signal_tissue, label = "final_signal_tissue")
                plt.legend()
                plt.title('Tissue TDC')

#                 plt.subplot(2, 4, 4)
#                 plt.plot(final_signal_aif, label = "aif signal")
#                 plt.plot(final_signal_tissue, label = "tissue signal")
#                 plt.plot(sig*100, label = 'reg inv_aif')
#                 plt.legend()
#                 plt.title('AIF & Tissue TDC')

#                 plt.subplot(2, 4, 5)               
#                 plt.plot(self.aif_seq, label = "array_aif_bl_f")
#                 plt.plot(base_aif, label = "aif without baseline shift")
#                 plt.plot(final_signal_aif, label = "final_signal_aif")
#                 plt.legend()
#                 plt.title('AIF TDC')
                
#                 plt.subplot(2, 4, 6)
#                 plt.plot(o_sig*100, label = 'inverse aif')
#                 plt.plot(final_signal_tissue, label = 'tissue')
#                 plt.legend()
#                 plt.title('inverse aif')
                
#                 plt.subplot(2, 4, 7)
#                 plt.plot(residual_func_blur)
# #                 plt.plot(residual_func * 10, label = 'svd irf')
#                 plt.title('irf blur')
                
                plt.show()
                plt.pause(0.8)
                plt.close()


# In[282]:


case_list = getCaselst(path)
idx = 0


# In[ ]:


start = datetime.datetime.now()
sys = AisSystem(0, case_list, g_std_t=1, lamda=0.65, SmoothImage=False)
end = datetime.datetime.now()
(end - start).seconds


# In[193]:


tissue_signal = sys.mainPeak_img.copy().transpose(1, 2, 0)
imgs = sys.images.copy()
mask = sys.brain_mask.copy()
vessel_mask, aif_mask, vel_location,vessel_threshold, aif_threshold, timepoint = aifDetection_Kmeans(tissue_signal, imgs, mask, 15)


# In[194]:


aifs = tissue_signal[np.where(aif_mask!=0)]
ty, tx = vofDetection(mask)
vof = tissue_signal[ty, tx]


# In[195]:


loc = np.where(aif_mask!=0)
loc


# In[217]:


img = imgs[0].copy()
img[ty - 2: ty + 2, tx - 2:tx + 2] = 255
for i in range(len(aifs)):
    res_ttp = np.argmax(aifs[i]) - np.argmax(vof)
    res_cbf = np.max(aifs[i]) - np.max(vof)
    y, x = loc[0][i], loc[1][i]
    img_show = img.copy()
    img_show[y-2:y+2, x-2:x + 2] = 0
#     if res_ttp < 0 and res_cbf > 0:
    if sys.brain_mask[y, x] == 0:
        continue
    print(np.nonzero(aifs[i])[0][0], np.nonzero(vof)[0][0])
    print(max(aifs[i]), max(vof))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(aifs[i], label='aif')
    plt.plot(vof, label='vof')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.imshow(img_show)
    plt.show()


# In[223]:


img_show = imgs.copy()
for i in range(len(imgs)):
    for j in range(len(aifs)):
        y, x = loc[0][j], loc[1][j]
        img_show[i][y-1:y+1, x-1:x+1]=0


# In[221]:


tissue_signal.shape


# In[224]:


# img_show = tissue_signal.transpose(-1, 0, 1)
# img_show = imgs.copy()
seriesImg_show(img_show, .5)


# In[122]:


def vofDetection(mask):
    h, w = mask.shape
    ellipse, info = ellipse_fitting(mask)
    short_radius = info[1][0]
    long_radius = info[1][1]
    center = info[0]
    degree = info[-1]
    x_shift = center[0] * math.tan(math.radians(degree))
    x_top = center[0] + x_shift
    x_bottom = center[0] - x_shift
    line_mask = np.zeros_like(mask)
    cv2.line(line_mask, (int(x_top), 0), (int(x_bottom), 255), (1, 0, 0), 1)
    line_mask *= mask
    half_radius = long_radius / 2 - (long_radius - line_mask.sum()) - 3
    if degree > 90:
        t_x = int(center[0] + half_radius * math.sin(math.radians(degree)))
        t_y = int(center[1] - half_radius * math.cos(math.radians(degree)))
    else:
        t_x = int(center[0] - half_radius * math.sin(math.radians(degree)))
        t_y = int(center[1] + half_radius * math.cos(math.radians(degree)))
    return t_y, t_x


# In[209]:


ellipse, info = ellipse_fitting(mask)
img_show = imgs[20].copy()
# img_show[ellipse!=0] = 255
short_radius = info[1][0]
long_radius = info[1][1]
center = info[0]
degree = info[-1]

x_shift = center[0] * math.tan(math.radians(degree))
x_top = center[0] + x_shift
x_bottom = center[0] - x_shift
y_shift = (255 - center[0]) * math.tan(math.radians(degree))
y_top =  center[1] - y_shift
y_bottom = center[1] + y_shift

# x_top = max(x_top_, x_bottom_)
# x_bottom = min(x_top_, x_bottom_)

print(center, x_top, x_bottom)
line_mask = np.zeros((256, 256))
rigid_img = img_show.copy()
rigid_img[ellipse!=0]=255
cv2.line(rigid_img, (int(x_top), 0), (int(x_bottom), 255), (255, 0, 0), 1)
cv2.line(rigid_img, (0, int(y_top)), (255, int(y_bottom)), (255, 0, 0), 1)
cv2.line(line_mask, (int(x_top), 0), (int(x_bottom), 255), (255, 0, 0), 1)
line_mask *= mask
line_mask[line_mask!=0] = 1
half_radius = long_radius / 2 - (long_radius - line_mask.sum()) - 5
if degree > 90:
    t_x = int(center[0] + half_radius * math.sin(math.radians(degree)))
    t_y = int(center[1] - half_radius * math.cos(math.radians(degree)))
else:
    t_x = int(center[0] - half_radius * math.sin(math.radians(degree)))
    t_y = int(center[1] + half_radius * math.cos(math.radians(degree)))
# img_show[line_mask!=0]= 255 
rigid_img[t_y - 3 : t_y + 3, t_x - 3 : t_x + 3] = 0
plt.figure(figsize=(15, 7))
plt.subplot(1, 3, 1)
plt.imshow(rigid_img)
plt.subplot(1, 3, 2)
plt.imshow(mask)
plt.subplot(1, 3, 3)
plt.imshow(img_show)
plt.show()


# In[210]:


t_y, t_x = vofDetection(mask)
img_show = imgs[0].copy()
img_show[t_y - 1:t_y + 1, t_x - 1 : t_x + 1] = 255
temp = tissue_signal[t_y, t_x, :]
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.imshow(img_show)
plt.subplot(1, 2, 2)
plt.plot(temp)
plt.show()


# In[246]:


img_show = imgs[15].copy()
img_show[aif_mask != 0] = 255
plt.figure(figsize=(15, 7))
plt.subplot(1, 3, 1)
plt.imshow(aif_mask)
plt.subplot(1, 3, 2)
plt.imshow(imgs[0])
plt.subplot(1, 3, 3)
plt.imshow(img_show)
plt.show()


# In[206]:


ori_vof = sys.tissue_signal[..., ty, tx]
plt.figure()
plt.plot(ori_vof)
plt.show()


# In[187]:


mat = np.zeros((44, 256, 256))
mat = mat.transpose(1, 2, 0)
mat.shape


# In[184]:


plt.figure()
plt.imshow(sys.brain_mask)
plt.show()


# In[282]:


import math
imgs = sys.ori_images.copy()
degrees = np.array([])
start = datetime.datetime.now()
for i in range(imgs.shape[0]):
#     brain_mask, bone_mask, contour = removeBone_all(imgs[i])
#     imgs[i][brain_mask == 0] = 0
#     ellipse, info = ellipse_fitting(bone_mask)
#     center = info[0]
#     degree = info[-1]
#     degrees = np.append(degrees, degree)
#     shift = center[0] * math.tan(math.radians(degree))
#     top = center[1] + shift
#     bottom = center[1] - shift
#     ellipse_, _ = ellipse_fitting(imgs[i])
#     ellipse[ellipse_ != 0] = 255
#     rigid_img = np.zeros((256, 256))
    
#     cv2.line(rigid_img, (0, 255 // 2), (255, 255 // 2),(255, 0, 0), 2)
#     cv2.line(rigid_img, (255 // 2, 0), (255 // 2, 255), (255, 0, 0), 2)
#     cv2.line(rigid_img, (int(top), 0), (int(bottom), 255), (255, 0, 0), 1)
#     rigid_img[ellipse!=0] = 255
#     imgs[i][ellipse == 1] = 255

    ans = rigistration(imgs[i])
end = datetime.datetime.now()
print((end - start).seconds)
#     rigid_img_ = np.zeros((256, 256))
#     cv2.line(rigid_img_, (0, 255 // 2), (255, 255 // 2),(255, 0, 0), 2)
#     cv2.line(rigid_img_, (255 // 2, 0), (255 // 2, 255), (255, 0, 0), 2)
#     rigid_img_[ans!=0] = 255
    
#     plt.figure(figsize=(20, 10))
#     display.clear_output(wait = True)
#     plt.subplot(2, 3, 1)
#     plt.imshow(bone_mask)
#     plt.subplot(2, 3, 2)
#     plt.imshow(rigid_img)
#     plt.subplot(2, 3, 3)
#     plt.imshow(imgs[i])
#     plt.subplot(2, 3, 4)
#     plt.ylim(ymin=-1)
#     plt.ylim(ymax=1)
#     plt.plot(degrees - degrees.mean())
#     plt.plot(np.zeros((len(degrees))))
#     plt.subplot(2, 3, 5)
#     plt.imshow(rigid_img_)
#     plt.show()
#     plt.pause(1)
#     plt.close()


# In[96]:


def ellipse_fitting(contour):
    edge = cv2.Canny(contour.astype('uint8'), 0, 1)
    y, x = np.nonzero(edge)
    edge_list = np.array([[x_, y_] for x_, y_ in zip(x, y)])
    _ellipse = cv2.fitEllipse(edge_list)
    edge_clone = edge.copy()
    cv2.ellipse(edge_clone, _ellipse, (255, 0, 0), 2)
    edge_clone[edge!=0] = 0
    return edge_clone, _ellipse


# In[158]:


def rigistration(img):
    h, w = img.shape
    edge_clone, info = ellipse_fitting(img)
    center = info[0]
    degree = info[-1]
    shift_y, shift_x = h // 2 - center[0], w // 2 - center[1]
    shift_mat = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_img = cv2.warpAffine(img, shift_mat, (w, h))
    rotate_mat = cv2.getRotationMatrix2D((w // 2, h//2), degree, 1.0)
    rotated_img = cv2.warpAffine(shifted_img, rotate_mat, (w, h))
    return rotated_img


# In[159]:


rigid_axis = np.zeros((256, 256))
cv2.line(rigid_axis, (0, 255 // 2), (255, 255 // 2),(255, 0, 0), 1)
cv2.line(rigid_axis, (255 // 2, 0), (255 // 2, 255), (255, 0, 0), 1)
ans = rigistration(ellipse)
rigid_axis_ = rigid_axis.copy()
rigid_axis[ans!=0] = 255
rigid_axis_[ellipse!=0] = 255


# In[160]:


plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(rigid_axis)
plt.subplot(1, 2, 2)
plt.imshow(rigid_axis_)
plt.show()


# In[139]:


rotated = cv2.warpAffine(ellipse, rotate_mat, (255, 255))


# In[140]:


epse, info_ = ellipse_fitting(rotated)
info, info_


# In[136]:


plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(imgs[i])
plt.subplot(1, 3, 2)
plt.imshow(rotated)
plt.subplot(1, 3, 3)
plt.imshow(epse)
plt.show()


# In[138]:


center = info[0]
degree = info[-1]
rotate_mat = cv2.getRotationMatrix2D(center, math.radians(degree), 1.0)
rotate_mat, rotate_mat.shape


# In[ ]:


shift_mat = cv2.warpAffine()


# In[204]:


for i in range(100 * 100, 256 * 256):
    row = i // 256
    col = i % 256
    if sys.brain_mask[row, col] == 0:
        continue
    
    plt.figure(figsize=(15, 7))
    display.clear_output(wait = True)
    plt.subplot(1, 2, 1)
    plt.plot(sys.tissue_signal[..., row, col]-100, label='Tissue')
    plt.plot(sys.signal_img[..., row, col], label = 'Signal')
    plt.plot(sys.mainPeak_img[..., row, col], label = 'Main Peak')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(sys.tissue_signal[..., row, col]-100, label='Tissue')
    plt.show()
    plt.pause(2)
    plt.close()


# In[1596]:


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

    sample_num=group_num
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


# In[1413]:


import torch as t
import torch.nn as nn
from torch import tensor
from torch.autograd import Variable as V
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import optim
import os


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


class AdaptFilter(nn.Module):
    def __init__(self, shape, degree):
        super(AdaptFilter, self).__init__()
        param_shape = (shape) + (degree,)
        self.weight = t.nn.Parameter(t.zeros(param_shape))
        t.nn.init.xavier_normal(self.weight, gain=1)
    def forward(self, x):
        weight_decay = t.norm(self.weight, p=1, dim=-1).unsqueeze(0).unsqueeze(-1)
        x = t.sum(input=self.weight * x, dim=-1, keepdims=True) + 0.01 * weight_decay
        return x


def curve_fit(y, mask, degree=6, batch=2, max_epoch=5000, mini_epoch = 1000, lr=0.001, lr_decay=1, factor=1, eps=1e-6, 
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
    
#     if blRemoval:
#         y[-1, ...] = t.exp(y[-1, ...] + eps)
        
    poly_x_train = poly_x.clone()
    y_train = y.clone()
    prev_std = t.zeros((1, 256, 256, 1)).to(Device)
    main_peak_index = np.zeros((1, 256, 256, 1))
    main_peak_val = None
    mask = mask.reshape((1, 256, 256, 1))
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
            
        if blRemoval and epoch != 0 and epoch % (mini_epoch - 1) == 0:
            adaptFilter.eval()
            with t.no_grad():
                predi = t.zeros((time_axis, h, w, 1)).to(Device)
                for tt in range(time_axis):
                    predi[tt : tt + 1] = adaptFilter(poly_x[tt : tt + 1])
                std = t.std((y - predi), axis=0, keepdims=True)
                if epoch == (mini_epoch-1):
                    main_peak_index = (y <= (predi + std)) # [time, h, w, 1], bool
                    main_peak_val = (predi + std)[main_peak_index]

#                 y[1:-1, ...] = t.minimum(y[1:-1,...], (predi + std)[1:-1, ...])
#                 y[-1, ...] = t.maximum((predi + std)[-1, ...], y[-1, ...])
                
                y = t.minimum(y, (predi + std))
                y_train = y
                if main_peak_val is not None:
#                     main_peak_val = t.maximum(main_peak_val, (predi + std)[main_peak_index])
#                     y_train[main_peak_index] = t.exp(main_peak_val + eps)
                    y_train[main_peak_index] = main_peak_val
                condition = t.mean((abs(std - prev_std) / (std + eps_mat))[np.where(mask != 0)])
                if condition < 0.01:
                    break
                prev_std = std
            adaptFilter.train()
        
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
            predi[tt : tt + 1] = adaptFilter(eval_x[tt : tt + 1])
        predi = predi * (ymax - ymin) + ymin
        
    predi = predi.detach().cpu().numpy()
    return predi, losses


# In[34]:


t.cuda.empty_cache()


# In[1523]:


start = datetime.datetime.now()
ans, losses = curve_fit(signal_mat, mask, degree=6, batch=32, max_epoch=4000, lr = 0.01, factor=1)
end = datetime.datetime.now()
(end - start).seconds


# In[1021]:


start = datetime.datetime.now()
curve, losses = curve_fit(ans, mask, degree=2, batch=1, max_epoch=100, mini_epoch=20, lr = 0.01, blRemoval=True)
end = datetime.datetime.now()
(end - start).seconds


# In[1022]:


for i in range(100 * 100, 256 * 256):
    row = i // 256
    col = i % 256
    if sys.brain_mask[row, col] == 0:
        continue
    base_signal = BaselineRemoval(ans[:, row, col, 0]).IModPoly(1)
    left_side, right_side = truncate_tissue(base_signal) # get the left and right boundary of peak indexes
    base_signal_ = base_signal.copy()
    base_signal_[:left_side] = 0
    base_signal_[right_side + 1:] = 0
    base_signal_[base_signal_ < 0] = 0
    
    plt.figure(figsize=(15, 7))
    display.clear_output(wait = True)
    plt.subplot(1, 2, 1)
    plt.plot(sys.signal_taken[row, col] - 80, label='original')
    plt.plot(ans[:,row, col, 0] - 80, label='simulation')
    plt.plot(curve[:, row, col, 0], label='final')
    plt.plot(base_signal, label='package')
    plt.plot(np.zeros(44))
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(curve[:, row, col, 0], label='main peak')
    plt.plot(np.zeros(44))

    plt.show()
    plt.pause(1)
    plt.close()


# In[1558]:


start = datetime.datetime.now()
simulation = curve_fitting(signal_mat)
end = datetime.datetime.now()
(end - start).seconds


# In[1559]:


simulation.shape


# In[1560]:


start = datetime.datetime.now()
baseline = baseline_correction(simulation, mask, degree=2, repitition=5, Device="cpu")
end = datetime.datetime.now()
(end - start).seconds


# In[1561]:


baseline.shape


# In[169]:


from scipy.stats import pearsonr
from tqdm import tqdm
import pdb
def last_square_current_loss(xs, ys, A):
    error = 0.0
    for i in range(len(xs)):
        y1 = 0.0
        for k in range(len(A)):
            y1 += A[k] * xs[i]**k
        error += (ys[i] - y1) ** 2
    return error


def create_Polyfeatures(x, order = 6):
    time, w, h, f = x.shape
    features = np.ones((time, w, h, 1))
    for i in range(1, 6):
        features = np.concatenate((features, x**i), axis=-1)
    return features

def last_square_fit_curve_Gradient(xs, ys, order, iternum=1000, learn_rate=0.001):
    '''
    xs:[256, 256, time]
    ys:[256, 256, time]
    order:6
    '''
    A = [0.0] * (order + 1)
    for r in range(iternum + 1):
        for k in range(len(A)):
            gradient = 0.0
            for i in range(len(xs)):
                y1 = 0.0
                for j in range(len(A)):
                    y1 += A[j] * xs[i]**j
                gradient += -2 * (ys[i] - y1) * xs[i]**k 
            A[k] = A[k] - (learn_rate * gradient)  
        if r % 100 == 0:
            error = last_square_current_loss(xs=xs, ys=ys, A=A)
    return A

def draw_fit_curve(xs, ys, A, order):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fit_xs, fit_ys = np.arange(-1, 1, 0.01), []
    for i in range(0, len(fit_xs)):
        y = 0.0
        for k in range(0, order + 1):
            y += (A[k] * fit_xs[i] ** k)
        fit_ys.append(y)
    ax.scatter(fit_xs, fit_ys, color='g')
    ax.plot(xs, ys, color='m', linestyle='', marker='.')
    plt.show()


# In[170]:


def func(weight, x):
    a1, a2, a3, a4, a5, a6 = weight[:,...]
    return a1*(x**5) + a2*(x**4) + a3*(x**3) + a4*(x**2) + a5*(x) + a6

def error(p, x, y):
    return (func(p, x) - y)**2


# In[174]:


from scipy.optimize import curve_fit, leastsq
x = np.arange(len(signal_taken))
y = signal_taken
p0 = np.zeros((6,))
x.shape, y.shape
param = leastsq(error, p0, args=(x, y))
show_signal = func(param[0], x)


# In[176]:


plt.figure()
plt.scatter(np.arange(len(show_signal)) ,show_signal)
plt.scatter(np.arange(len(signal_taken)), signal_taken)
plt.show()


# In[73]:


base_signal = BaselineRemoval(pred).IModPoly(2)
base_signal_split = BaselineRemoval(show_signal).IModPoly(2)


# In[74]:


plt.figure()
plt.plot(base_signal)
plt.plot(base_signal_split)
plt.show()


# In[295]:


len(signal), len(show_signal)


# In[ ]:


hist = []
for i in range(len(signal) / 25)
    hist = len()


# In[86]:


import seaborn as sns

# signal = expand_signal[..., row, col, 0]
sns.distplot(signal, rug=True)


# In[28]:


plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(np.arange(44), ori_signal)
plt.plot(np.arange(44), curve[::32])
plt.show()


# In[478]:


def find_bbox(mask):
    """
    :param mask:Ground Truth
    :return: [left_bottom_y, left_bottom_x, height, width]
    """
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    stats = stats[stats[:, 4].argsort()]
    return stats[:-1]

def generator_bbox(mask, mode='box', expFactor=1):
    """
    :param mask:
    :param mode: 'center' or 'box'
    :param expFactor: expand the height,width of the anchor box for easier semantic segmentation
    :return: [left_bottom_y, left_bottom_x, right_top_y, right_top_x] or [cnt_y, cnt_x, height, width]
    """
    if mode not in ['center', 'box']:
        raise ValueError

    if mask.shape[0] == 1:
        mask = mask.squeeze(0)

    bbox = find_bbox(mask)
    ans = np.zeros((len(bbox), 4))
    for idx, box in enumerate(bbox):
        ymin = box[0] - 1
        xmin = box[1] - 1
        height = box[2] + 1
        width = box[3] + 1

        if expFactor != 1:
            cnt_y = ymin + height / 2
            cnt_x = xmin + width / 2
            height = expFactor * height
            width = expFactor * width
            ymin = cnt_y - height / 2
            xmin = cnt_x - width / 2

        if mode == 'center':
            ans[idx, :] = [ymin + height / 2, xmin + width / 2, height, width]
        else:
            ans[idx, :] = [ymin + height, xmin, ymin, xmin + width]
    return ans

def bbox2show(bbox, mode='center'):
    """
    :param bbox:[num, 4]
    :param mode: 'center' or 'box'
    :return: center:[cnt_y, cnt_x, height, width] -> [ymin, xmin, height, width] 
        or box:[ymax, xmin, ymin, xmax] -> [ymin, xmin, height, width]
    """
    if mode not in ['center', 'box']:
        raise ValueError

    if isinstance(bbox, list):
        bbox = np.array(bbox)

    ans = np.zeros((len(bbox), 4))
    for idx, box in enumerate(bbox):
        if mode == 'center':
            cnt_y, cnt_x = box[:2]
            height, width = box[2:]
            ans[idx, :] = [int(cnt_y - height / 2), int(cnt_x - width / 2), height, width]
        else:
            ymax, xmin = box[:2]
            ymin, xmax = box[2:]
            height = ymax - ymin
            width = xmax - xmin
            ans[idx, :] = [ymin, xmin, height, width]
    return ans


# In[392]:


def add_box(img, mask):
    bbox = generator_bbox(mask, mode='box', expFactor=1)
    for box in bbox:
        print(box)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255,23,13), 1)
    return img


# In[393]:


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
    print(aif_threshold, vessel_threshold)
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

def add_box(img, mask):
    bbox = generator_bbox(mask, mode='box', expFactor=1.5).astype('int')
    for box in bbox:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (184,134,11), 1)
    return img

def saveVideo(video, name, path, timepoint, mask):
    size = len(video)
    path = os.path.join(path, '{}.avi'.format(name))
    videoWriter = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), size, Img_shape)
    for i in range(len(video)):
        img = video[i]
        if i == timepoint:
            img = add_box(img, mask)
        videoWriter.write(img)
    return


# In[650]:


def param_mapping(param, mask, mode = 'less'):
    mask_idx = np.where(mask != 0)
    if param.shape[-1] == 1:
        param = param.squeeze(-1)
    param_seq = param[mask_idx]
    condition = modeNum(param, mode='mean')
    if mode == 'less':
        param_idx = np.where(param_seq < condition)
    elif mode == 'big':
        param_idx = np.where(param_seq > condition)
    else:
        raise ValueError
        
    seq_copy = np.zeros_like(param_seq)
    seq_copy[param_idx] = 1
    param_img = np.zeros_like(mask)
    param_img[mask_idx] = seq_copy
    return param_img

def normalization(param_map):
    minest = np.min(param_map, axis = (0, 1))
    maxest = np.max(param_map, axis = (0, 1))
    param_norm = param_map / (maxest - minest) 
    return param_norm

def otsu(param):
    output = param.copy()
    threshold = filters.threshold_otsu(param[param!=0])
    output[param < threshold] = 0
    return output


def vessel_location_condition(param_norm, mask):
    region = 2 * param_norm[..., 2] - np.sum(param_norm, axis=-1)
    region[mask!=0] = np.exp(region[mask!=0]) 
    plt.figure()
    plt.imshow(region)
    plt.show()
    vessel_region = otsu(region)
    vessel_region[vessel_region!=0] = 1
    plt.figure()
    plt.imshow(vessel_region)
    plt.show()
    return vessel_region


# In[899]:


import skimage
from collections import Counter
from scipy import stats
Img_shape = (256, 256)
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
    shape = Img_shape
    bat = np.zeros(shape + (1,))
    end = np.zeros_like(bat)
    for i in range(shape[0] * shape[1]):
        y = int(i // shape[0])
        x = int(i % shape[1])
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
    #(bat, radius, cbf, ttp, end)
    mask_idx = np.where(mask != 0)
    param_norm = normalization(param_map)
    # --------------Label Selection----------------------
    region = param_norm[..., 2] - param_norm[..., 3] 
    region[mask_idx] = np.exp(region[mask_idx]) 
    union = otsu(region)
    union[union!=0] = 1
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(region)
    plt.subplot(1, 2, 2)
    plt.imshow(union)
    plt.show()
    # --------------Normalization----------------
    shape = Img_shape
    x_seq = param_norm[mask_idx]
    predi = Kmeans(x_seq, cluster=2)
    label_map = predi.labels_ + 1
    label_seq = label_map[union[mask_idx] == 1]
    label = stats.mode(label_seq)[0][0]
    # ----------------label --------------------------
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
    elif mode == 'median':
        method = np.median
    elif mode == 'mode':
        method = lambda x: np.argmax(np.bincount(x.astype('int')))
    else:
        raise ValueError

    param_map = param_map.copy()
    ans = method(param_map[np.where(param_map != 0)])
    return ans

def Kmeans(seq, num_f = 1, cluster = 2):
    seq =seq.reshape(-1, seq.shape[-1])
    y_pred = kmeans(n_clusters = cluster, init = 'k-means++', n_jobs = -1, precompute_distances=True).fit(seq)
    return y_pred

def openOperation(mask, erod_w = 2, dilation_w = 2):
    dilation_kernal = skimage.morphology.square(dilation_w)
    erod_kernel = skimage.morphology.square(erod_w)
    erod = skimage.morphology.erosion(mask, erod_kernel)
    dilation = skimage.morphology.dilation(erod, dilation_kernal)
    return dilation


def aif_filter(aif_mask, param_norm, outputNum = 6):
    #bat, radius, cbf, ttp, end
    sort_conditions = np.sum(param_norm, axis=-1) - 2 * param_norm[..., 2]
    aif_mask = openOperation(aif_mask, 2, 2)
    label_image = label(aif_mask, connectivity=aif_mask.ndim)
    props = regionprops(label_image)
    priQue = []
    for i in range(len(props)):
        row, col = props[i].centroid
        ymax = int(row + 1)
        ymin = int(row - 1)
        xmax = int(col + 1)
        xmin = int(col - 1)
        priQue.append(np.sum(sort_conditions[ymin:ymax + 1, xmin:xmax + 1]) / 9)
    sortIdx = np.argsort(np.array(priQue))
    props = np.array(props)
    props = props[sortIdx]
    output = np.zeros_like(aif_mask)
    for i in range(min(outputNum, len(props))):
        row, col = props[i].centroid
        ymax = int(row + 1)
        ymin = int(row - 1)
        xmax = int(col + 1)
        xmin = int(col - 1)
        output[ymin:ymax + 1, xmin:xmax + 1] = 1
    return output


def aifDetection_Kmeans(tissue_signal, images, mask):
    ttp = ttp_mapping(tissue_signal)
    cbf = cbf_mapping(tissue_signal)
    radius = halfRadius_mapping(tissue_signal, cbf)
    bat, end = bat_mapping(tissue_signal)
    param_map = np.concatenate((bat, radius, cbf, ttp, end), axis=-1)
    param_norm = normalization(param_map)
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
    aif_seq = np.expand_dims(aif_seq, axis = 0)
    aif_threshold = filters.threshold_otsu(aif_seq)
    aif_fill = np.zeros_like(aif_seq)
    aif_fill[np.where(aif_seq > aif_threshold)] = 1
    aif_location = np.zeros_like(vel_location)
    aif_location[np.where(vessel_mask!=0)] = aif_fill
    
    # ---------------Aif Mask----------------------
    Image = images[idx].copy() / 255
    ttp_mask = creat_mask(Image, aif_location, aif_threshold)
    
    idx = (modeNum(ttp, mode='median') + modeNum(bat, mode='median')) // 2
    idx = int(idx)
    Image = images[idx].copy() / 255
    aif_mask = creat_mask(Image, aif_location, aif_threshold)
    aif_mask[ttp_mask == 1] = 1
    aif_mask = aif_filter(aif_mask, param_norm, 10)
    return aif_location, aif_mask, vel_location, vessel_threshold, aif_threshold, idx 

def creat_mask(Image, location, threshold):
    location_idx = np.where(location == 1)
    img_seq = Image[location_idx]
    output_mask = np.zeros_like(Image)
    fill = np.zeros_like(img_seq)
    fill[np.where(img_seq > threshold)] = 1
    output_mask[location_idx] = fill
    return output_mask


# In[900]:


def aifDetection_Condition(tissue_signal, images, mask):
    
    ttp = ttp_mapping(tissue_signal)
    cbf = cbf_mapping(tissue_signal)
    radius = halfRadius_mapping(tissue_signal, cbf)
    bat, end = bat_mapping(tissue_signal)
    param_map = np.concatenate((bat, radius, cbf, ttp, end), axis=-1)
    param_norm = normalization(param_map)
    vessel_mask = vessel_location_condition(param_norm, mask)
    
    idx = modeNum(ttp, mode='median')
    idx = int(idx)
    Image = images[idx].copy() / 255
    vessel_threshold = np.mean(Image[vessel_mask != 0])
    aif_location_idx = np.where(vessel_mask == 1)
    aif_seq = Image[aif_location_idx]
    
    aif_seq = np.expand_dims(aif_seq, axis=0)
    aif_threshold = filters.threshold_otsu(aif_seq)
    aif_location = np.zeros_like(vessel_mask)
    
    aif_fill = np.zeros_like(aif_seq)
    aif_fill[np.where(aif_seq > aif_threshold)] = 1
    aif_location[aif_location_idx] = aif_fill.squeeze(0)
    
    # -----------------Create Mask---------------------
    Image = images[idx].copy() / 255
    ttp_mask = creat_mask(Image, aif_location, aif_threshold)
    idx = (modeNum(ttp, mode='median') + modeNum(bat, mode='median')) // 2
    idx = int(idx)
    Image = images[idx].copy() / 255
    aif_mask = creat_mask(Image, aif_location, aif_threshold)
    aif_mask[ttp_mask == 1] = 1
    
    ttp = ttp_mapping(tissue_signal)
    cbf = cbf_mapping(tissue_signal)
    radius = halfRadius_mapping(tissue_signal, cbf)
    bat, end = bat_mapping(tissue_signal)
    param_map = np.concatenate((bat, radius, cbf, ttp, end), axis=-1)
    
    aif_mask = aif_filter(aif_mask, param_map, 10)
    return aif_location, aif_mask, vessel_threshold, aif_threshold, idx 


# In[928]:


imgs = sys.images.copy()
tissue_signal = sys.tissue_signal.copy()
mask = sys.brain_mask.copy()


# In[929]:


global_all, timepoint


# In[930]:


# --------------------Kmeans-------------------------
import time
start = datetime.datetime.now()
vessel_mask, aif_mask, vel_location,vessel_threshold, aif_threshold, timepoint = aifDetection_Kmeans(tissue_signal, imgs, mask)
end = datetime.datetime.now()
print('Time: {}s'.format((end - start).seconds))


# In[931]:


print(vel_location.shape)
plt.figure(figsize = (20, 15))
plt.subplot(1, 2, 1)
plt.imshow(RGBMapping(imgs[timepoint], aif_mask, aif_mask))
# plt.subplot(1, 2, 2)
# plt.imshow(vel_location)
# plt.show()


# In[882]:


video = creatVideo(imgs, aif_threshold, vessel_threshold, vessel_mask)
for idx, pic in enumerate(video):
    plt.figure(figsize = (20, 15))
    plt.imshow(pic)
    plt.show()


# In[883]:


saveVideo(video, global_all+'_Kmeans', '', timepoint, aif_mask)
global_all


# In[907]:


# ------------------Conditions-------------------------
start = datetime.datetime.now()
vessel_mask, aif_mask, vessel_threshold, aif_threshold, timepoint = aifDetection_Condition(tissue_signal, imgs, mask)
end = datetime.datetime.now()
print('Time: {}s'.format((end - start).seconds))


# In[908]:


print(vel_location.shape)
plt.figure(figsize = (20, 15))
plt.subplot(1, 2, 1)
plt.imshow(RGBMapping(imgs[timepoint], aif_mask, aif_mask))
plt.subplot(1, 2, 2)
plt.imshow(vessel_mask)
plt.show()


# In[721]:


video = creatVideo(imgs, aif_threshold + 0.1, vessel_threshold, vessel_mask)


# In[722]:


for pic in video:
    plt.figure()
    plt.imshow(pic)
    plt.show()


# In[723]:


saveVideo(video, global_all + '_Condition', '', timepoint, aif_mask)
global_all


# In[209]:


idx = np.where(aif_mask != 0)
signal = tissue_signal[idx]
signal.shape


# In[211]:


signal_avg = np.mean(signal, axis=0)
signal_avg


# In[224]:


tissue_signal.shape


# In[260]:


while(1):
    random_idx = np.random.randint(256*256)
    y, x = random_idx // 256, random_idx % 256
    if mask[y, x] != 0:
        other = tissue_signal[y, x]
        break


# In[261]:


plt.figure()
plt.plot(signal_avg)
plt.plot(other)
plt.show()


# In[156]:


def vessel_locating(param_map, mask):
    shape = Img_shape
    mask_idx = np.where(mask != 0)
    x_seq = param_map[mask_idx]
    predi = Kmeans(x_seq, cluster=2)
    label_map = predi.labels_
    bat = param_map[..., 0]
    mean_bat = modeNum(bat, mode='mean')
    label_idx = np.where(bat[mask != 0] < mean_bat)[:-1]
    label = np.argmax(np.bincount(label_map[label_idx]))
    fill = np.zeros_like(label_map)
    fill[np.where(label_map == label)] = 1
    location = np.zeros(shape)
    location[mask_idx] = fill
    return location


# In[157]:


location = vessel_locating(param_map, mask)


# In[158]:


plt.figure()
plt.imshow(location)
plt.show()


# In[147]:


np.sum(bat == param_map[..., 0], axis = 1)


# In[136]:


mask_idx = np.where(mask != 0)
x_seq = param_map[mask_idx]
predi = Kmeans(x_seq, cluster = 2)
label_map = predi.labels_
location = np.zeros((256, 256))
location[mask_idx] = label_map
plt.figure()
plt.imshow(location)
plt.show()


# In[154]:


mean_bat = modeNum(bat, mode='mean')
label_idx = np.where(bat[mask != 0] < mean_bat)[:-1]
label_seq = label_map[label_idx]
lamfunc = lambda x : np.argmax(np.bincount(x).astype('int'))
lamfunc(label_seq)


# In[130]:


vessel_location = location
# idx = (modeNum(bat, mode='mode') + modeNum(ttp, mode='mode')) // 2
idx = modeNum(ttp, mode='mode')
print('vessel time:',idx)
Image = imgs[idx].copy() / 255
vessel_location_idx = np.where(vessel_location != 0)    
vessel_seq = Image[vessel_location_idx]
vessel_seq = np.expand_dims(vessel_seq, axis=0)
vessel_threshold = filters.threshold_otsu(vessel_seq)
vessel_mask = np.zeros_like(vessel_location)
vessel_fill = np.zeros_like(vessel_seq)
vessel_fill[np.where(vessel_seq > vessel_threshold)] = 1
vessel_mask[vessel_location_idx] = vessel_fill.squeeze(0)
plt.figure()
plt.imshow(vessel_mask)
plt.show()


# In[24]:





# In[21]:





# In[26]:


img = video[timepoint].copy()
img = add_box(img, aif_mask)
plt.figure()
plt.imshow(img)
plt.show()


# In[243]:


videoWriter = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 43, (256, 256))


# In[244]:


for i in range(len(video)):
    img = video[i]
    if i == timepoint:
        img = add_box(img, aif_mask)
        plt.figure()
        plt.imshow(img)
        plt.show()
    videoWriter.write(img)
videoWriter.release()


# In[25]:


video = creatVideo(imgs, aif_threshold, vessel_threshold, vessel_location)
plt.figure()
for i in range(len(video)):
    plt.imshow(video[i])
    plt.show()


# In[169]:


img = imgs[15].copy().astype('uint8')
rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
rgb[..., 0][vessel_mask == 1] = 255
rgb[..., 0][aif_mask == 1] = 255
rgb[..., 1][vessel_mask == 1] = 255
rgb[..., 1][aif_mask == 1] = 48
rgb[..., 2][vessel_mask == 1] = 0
rgb[..., 2][aif_mask == 1] = 48

plt.figure(figsize = (15, 7))
plt.imshow(rgb)
plt.show()
print(rgb.shape)


# In[56]:


from einops import repeat, rearrange
center = predi.cluster_centers_[-1]
centerMat = repeat(center, 'f-> h w f', h = 256, w = 256)

dis_map = np.sum(np.sqrt(np.power(param_map - centerMat, 2)), axis=-1)
plt.figure()
plt.imshow(dis_map)
plt.show()


# In[63]:


img = sys.images[14].copy()
img[np.where(dis_map < 5)] = 0
plt.figure()
plt.imshow(img)
plt.show()


# In[112]:


param_map = np.concatenate((bat, radius, cbf, ttp, end), axis=-1)
# param_map = 
index = np.where(mask == 255)
param_map_ = param_map[index]
predi = Kmeans(param_map_)
label_map = predi.labels_
mapping = np.zeros((255, 255))
mapping[index] = label_map+1
plt.figure()
plt.imshow(mapping)
plt.show()
plt.figure()
plt.imshow(sys.images[15])
plt.show()


# In[119]:


predi.cluster_centers_.shape


# In[113]:


index = np.where(mapping==2.0)
param_map_ = param_map[index]
param_map_.shape
# otsu = filters.threshold_otsu(dist_map[dist_map > middle])


# In[110]:


index = np.where(mapping == 2.0)
param_map_ = param_map[index]
predi = Kmeans(param_map_)
label_map = predi.labels_
mapping = np.zeros((255, 255))
mapping[index] = label_map+1
plt.figure()
plt.imshow(mapping)
plt.show()
plt.figure()
plt.imshow(sys.images[15])
plt.show()


# In[50]:


starttime = datetime.datetime.now()
irf, cbf, _ = sys.obtain_irf(nWorkers = 2, method = sys.compute_irf)
# irf, cbf, conb = sys.obtain_tissue_tmax()
endtime = datetime.datetime.now()
print((endtime - starttime).seconds // 60)


# In[200]:


sys.show_R_signal()


# In[35]:


def show_signal(mat, tissue):
    size = len(mat)
    plt.figure(figsize = (15, 7))
    col_num = 8
    row_num = np.ceil(size / col_num)
    for i in range(size):
        plt.subplot(row_num, col_num, i+1)
        plt.plot(mat[i,:])
        plt.plot(tissue)
    plt.show()


# In[43]:


def regulSig(cir_mat):
    size = cir_mat.shape[0]
#     peakposi = np.argmax(cir_mat[:,0])
    
    cir_mat = np.linalg.inv(cir_mat)
    peakposi = np.argmax(cir_mat[0,:])
    print(peakposi)
    
#     show_signal(cir_mat)
    for ii in range(size):
        head = cir_mat[ii][:ii]
        tail = cir_mat[ii][ii:]
        cir_mat[ii,:] = np.r_[tail, head]
    ans = cir_mat.mean(0)
    
    peaks, properties = find_peaks(ans, prominence = 0, width = [0, 2])
    print(peaks, properties)
    left_bases = properties['left_bases']
    right_bases = properties['right_bases']
    idex = np.argmax(ans[peaks])
    print('idex:', idex)
    print(left_bases[idex], right_bases[idex])
    
    left = left_bases[idex] if abs(left_bases[idex] - peakposi) < abs(right_bases[idex - 1] - peakposi) else right_bases[idex - 1]
    print('left:',abs(left_bases[idex] - peakposi), abs(right_bases[idex - 1] - peakposi))
    right = right_bases[idex] if abs(right_bases[idex] - peakposi) < abs(left_bases[idex + 1] - peakposi) else left_bases[idex + 1]
    print('right',abs(right_bases[idex] - peakposi),  abs(left_bases[idex + 1] - peakposi))
    
    leftpart = ans[:left]
    rightpart = ans[right:]
    print(left, right)
    midpart = ans[left : right]
    
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(leftpart)
    plt.subplot(1, 3, 2)
    plt.plot(midpart)
    plt.subplot(1, 3, 3)
    plt.plot(rightpart)
    plt.show()
    leftpart = cv2.GaussianBlur(leftpart, (1, 3), 0.7)
    rightpart = cv2.GaussianBlur(rightpart, (1, 3), 0.7)
    ans = np.r_[leftpart.squeeze(-1), midpart, rightpart.squeeze(-1)]
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.plot(leftpart)
    plt.subplot(1, 4, 2)
    plt.plot(midpart)
    plt.subplot(1, 4, 3)
    plt.plot(rightpart)
    plt.show()
    plt.figure()
    plt.plot(ans)
    plt.show()
    return circulant(ans).T


# In[44]:


import cv2
aif = sys.final_signal_aif

cir_aif = circulant(aif)
plt.figure()
plt.plot(np.linalg.inv(cir_aif)[0])
plt.show()
cir_aif_inv = regulSig(cir_aif)


# In[174]:


aif = sys.final_signal_aif
gamma = np.zeros_like(aif)
gamma[np.argmax(aif)] = max(aif)


# In[178]:


cir_gamma = circulant(gamma)
cir_gamma_inv = np.linalg.inv(cir_gamma)
print(np.argmax(cir_gamma_inv[0]), np.argmax(gamma))
plt.figure()
plt.plot(cir_gamma_inv[0])
plt.show()


# In[51]:


from skimage import data, filters
from skimage import transform, exposure
irf_img = irf.copy()
cbf_img = cbf.copy() 
# cbv_img = cbv.copy()
brain_mask = sys.brain_mask


# In[52]:


#21, 31, 01
plt.figure(figsize = (15, 7))
plt.subplot(1, 3, 1)
plt.imshow(irf_img)
plt.title('Tmax Image:')
plt.subplot(1, 3, 2)
plt.imshow(cbf_img)
plt.title('CBF Image:')
# plt.subplot(1, 3, 3)
# plt.imshow(cbv_img)
plt.show()


# In[18]:


from skimage import data, filters
from skimage import transform, exposure
irf_img = irf.copy()
cbf_img = cbf.copy() 
# cbv_img = cbv.copy()
brain_mask = sys.brain_mask


# In[19]:


#21, 31, 01
plt.figure(figsize = (15, 7))
plt.subplot(1, 3, 1)
plt.imshow(irf_img)
plt.title('Tmax Image:')
plt.subplot(1, 3, 2)
plt.imshow(cbf_img)
plt.title('CBF Image:')
# plt.subplot(1, 3, 3)
# plt.imshow(cbv_img)
plt.show()


# In[20]:


#22, 31, 01
plt.figure(figsize = (15, 7))
plt.subplot(1, 3, 1)
plt.imshow(irf_img)
plt.title('Tmax Image:')
plt.subplot(1, 3, 2)
plt.imshow(cbf_img)
plt.title('CBF Image:')
# plt.subplot(1, 3, 3)
# plt.imshow(cbv_img)
plt.show()


# In[14]:


plt.figure(figsize = (15, 7))
plt.subplot(1, 3, 1)
plt.imshow(irf_img)
plt.title('Tmax Image:')
plt.subplot(1, 3, 2)
plt.imshow(cbf_img)
plt.title('CBF Image:')
# plt.subplot(1, 3, 3)
# plt.imshow(cbv_img)
plt.show()


# In[50]:


plt.figure(figsize = (15, 7))
plt.subplot(1, 3, 1)
plt.imshow(irf_img)
plt.title('Tmax Image:')
plt.subplot(1, 3, 2)
plt.imshow(cbf_img)
plt.title('CBF Image:')
# plt.subplot(1, 3, 3)
# plt.imshow(cbv_img)
plt.show()


# In[74]:


irf_label, res= postProcess(irf_img, sys.mask_contours, ptype = 'irf', n = 3)


# In[89]:


plt.imshow(res)


# In[70]:


set(res.ravel())
res[res == 1] = 0 


# In[84]:


def get_center(img):
#     h, w = img.size()
    index = np.where(img != 0)
#     print(index)
    size = len(index)
    center_y = int(index[0].mean())
    center_x = int(index[1].mean())
    return (center_y, center_x)
y, x = get_center(res)
# x -= 10
# y -= 20


# In[88]:


mat = irf_label.copy()
mat[y - 5 : y + 5, x - 5: x + 5] = 255
plt.imshow(mat)


# In[87]:


index = np.argmax(irf_img)
y = int(index // 256)
x = int(index % 256)
y,x


# In[82]:


cv2.rectangle(mark, (max(0, x - 2), max(y - 2, 0)),(min(256, x + 2),min(256, y + 2)), (227,23,13), 1)
mark = cv2.drawContours(mark, sys.mask_contours, 0, (255, 0, 0), 1)


# In[219]:


# mark = np.zeros_like(irf_label)
# index = np.argmax(irf_label)
# y = int(index // 256)
# x = int(index % 256)
# y,x


# In[85]:


def extract_cbf(cbf_img, seed, brain_mask):
    '''
    Input:
        1.the cbf image getting from Ais-system
        2.seed is the location of maximal lesion value in irf image
        3.brain_mask is the main matter of brain
    Output:
        lesion core extracted from cbf image
    
    Describe:
        1. Enhence the contrast ratio of Input Image using Clahe.
        2. Calculate the power image at x and y directions, and conbine together.
        3. Enhencing the contrast ratio by expanding distribution between [0, 1] using log correction function, a * log(1 + x).
        4. Split the power image into two parts, basin and ridge, using otsu or cluster.
        5. Locating the target basin area through seed location, the ridge around target basin is contour of lesion in cbf image.
        6. Mapping pixel location into distance from seed location.
        7. Spliting brain into three parts, drop off the part farthest away from target area and select the part closest to seed location.
        
    '''
    h, w = cbf_img.shape
    x, y = seed
    aeh_img = exposure.equalize_adapthist(cbf_img / 255, kernel_size=None, clip_limit=0.01, nbins=256)
    sobel_x = cv2.Sobel(aeh_img, cv2.CV_64F, 1, 0, ksize = 3)
    sobel_y = cv2.Sobel(aeh_img, cv2.CV_64F, 0, 1, ksize = 3)
    sobel_xy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    
    brain_mask = sys.brain_mask
    log_img = exposure.adjust_log(sobel_xy)
    otsu = filters.threshold_otsu(log_img[brain_mask != 0])
    log_img[log_img < otsu] = 0

    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    dist_map = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            if log_img[row, col] == 0:
                continue
            dist_map[row, col] = distance(x, y, col, row)
    
    
    maxV = dist_map.max()
    minV = dist_map.min()
    dist_map[dist_map!=0] = maxV - dist_map[dist_map !=0 ] # pixel inversion
    
    dist_map = ((dist_map - minV) / (maxV - minV)) * 255
    res = dist_map.copy()
    
    maxV = dist_map.max()
    minV = dist_map[dist_map != 0].min()
    
    middle = maxV - 1 * (maxV - minV) // 3
    print(maxV, minV, middle)
    
    otsu = filters.threshold_otsu(dist_map[dist_map > middle])
    dist_map[dist_map < otsu] = 0
    
#     kernel = np.ones((2,2), 'uint8')
#     lesion_dilation = cv2.dilate(dist_map, kernel, iterations = 1)
#     lesion_erode = cv2.erode(lesion_dilation, kernel, iterations = 1)
#     lesion_dilation = cv2.dilate(lesion_erode, kernel, iterations = 1)
    
    
    
    ret,mask = cv2.threshold(dist_map, 127, 255, 0)
    mask = cv2.drawContours(mask, sys.mask_contours, 0, (255, 0, 0), 1)
#     cv2.rectangle(mask, (max(0, x - 2), max(y - 2, 0)), 
#                               (min(256, x + 2),min(256, y + 2)), (227,23,13), 1)
    
    plt.figure(figsize = (15, 7))
    plt.subplot(1, 5, 1)
    plt.imshow(cbf_img)
    plt.title('Input Image')
    plt.subplot(1, 5, 2)
    plt.imshow(sobel_xy)
    plt.title('Sobel Power Image')
    plt.subplot(1, 5, 3)
    plt.imshow(log_img)
    plt.title('Enhance Image with Log')
    plt.subplot(1, 5, 4)
    plt.imshow(res)
    plt.title('Distance Image')
    plt.subplot(1, 5, 5)
    plt.imshow(mask)
    plt.title('Output Image')
    plt.show()
    
    return dist_map

res = extract_cbf(cbf_img, (x, y), brain_mask = sys.brain_mask)
plt.imshow(res)


# In[22]:


h, w = irf_img.shape

aeh_img = exposure.equalize_adapthist(irf_img / 255, kernel_size=None, clip_limit=0.01, nbins=256)
sobel_x = cv2.Sobel(aeh_img, cv2.CV_64F, 1, 0, ksize = 3)
sobel_y = cv2.Sobel(aeh_img, cv2.CV_64F, 0, 1, ksize = 3)
sobel_xy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
plt.figure(figsize = (15, 7))
plt.subplot(1, 4, 1)
plt.imshow(sobel_x)
plt.subplot(1, 4, 2)
plt.imshow(sobel_y)
plt.subplot(1, 4, 3)
plt.imshow(sobel_xy)
plt.subplot(1, 4, 4)
plt.imshow(aeh_img)
plt.show()


# In[61]:


irf_img.shape, cbf_img.shape


# In[63]:


input_img = np.stack((irf_img, cbf_img), axis = 0)


# In[64]:


input_img.shape


# In[68]:


input_img = input_img.reshape((256, 256, 2))


# In[95]:


seq =input_img.reshape(-1, 2)
predi = kmeans(n_clusters = 4, init = 'k-means++', n_jobs = -1).fit(seq)
label_img = predi.labels_


# In[96]:


label_img.resize((256, 256, 1))
label_img.shape


# In[223]:


#Hession Detector

sobel_x = cv2.Sobel(aeh_img, cv2.CV_64F, 1, 0, ksize = 3)
sobel_y = cv2.Sobel(aeh_img, cv2.CV_64F, 0, 1, ksize = 3)
sobel_xy = cv2.Sobel(sobel_x, cv2.CV_64F, 0, 1, ksize = 3)
sobel_yy = cv2.Sobel(sobel_y, cv2.CV_64F, 0, 1, ksize = 3)
sobel_xx = cv2.Sobel(sobel_x, cv2.CV_64F, 1, 0, ksize = 3)
matter = img[brain_mask == 255]
matter_xy = cv2.Laplacian(matter, cv2.CV_64F)
lap = img.copy()
lap[brain_mask == 255] = matter_xy.ravel()

sobel_2xy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
plt.figure(figsize = (15, 7))
plt.subplot(2, 5, 1)
plt.imshow(sobel_x)
plt.title('x driven')
plt.subplot(2, 5, 2)
plt.imshow(sobel_y)
plt.title('y driven')
plt.subplot(2, 5, 3)
plt.imshow(sobel_xy)
plt.title('xy driven')
plt.subplot(2, 5, 4)
plt.imshow(sobel_xx)
plt.title('Sobel xx')
plt.subplot(2, 5, 5)
plt.imshow(sobel_yy)
plt.title('Sobel yy')
plt.subplot(2, 5, 6)
plt.imshow(sobel_2xy)
plt.title('distance x and y power')
plt.subplot(2, 5, 7)
plt.imshow(sobel_x + sobel_y)
plt.title('x + y')
plt.subplot(2, 5, 8)
plt.imshow(sobel_xx + sobel_yy)
plt.title('xx + yy')
plt.subplot(2, 5, 9)
plt.imshow(lap)
plt.title('Laplacian')
plt.show()


# In[224]:


img = sobel_2xy
plt.figure(figsize = (15, 7))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(exposure.adjust_log(img))


# In[225]:


brain_mask = sys.brain_mask
log_img = exposure.adjust_log(img)
otsu = filters.threshold_otsu(log_img[brain_mask != 0])
otsu


# In[226]:


# img = res.copy()
log_img[img < otsu] = 0
plt.imshow(log_img)


# In[134]:


plt.hist(res[res != 0].ravel(), 256, [0, 255])
plt.show()


# In[135]:


show_img = res.copy()
show_img[show_img < otsu] = 0
plt.imshow(show_img)


# In[34]:


ret, img_mask = cv2.threshold(show_img, 127, 255, cv2.THRESH_BINARY)


# In[ ]:


def postProcess(image, mask, ptype = 'irf', n = 4):
        '''
        Input: Image scale in [0, 255]
        Ouput:
               Binary Image
        Discribe:
            1. Get labels from kmeans func that map tmax to [0, 4] or [0, 3].
            2. Get second minimum value and minimum value from labels.
            3. Let all value is not equal to min and sec_min be zero
            #---irf---
            4. Morphological operation.
            5. Let color in output image be uniform.
        '''    

        ncluster = n

        predi = Kmeans(image[mask != 0], cluster = ncluster)
        
        labels = predi.labels_
        label_img = labels.copy().reshape((256, 256))
        num_label = [len(labels[labels == i]) for i in set(labels)]
        color = np.argmin(num_label)
        secColor = find_secMin(num_label)
        lesion = np.zeros_like(image)
        for idx, v in enumerate(labels):
            ele = 0
            if labels[idx] == color:
                ele = 2
            elif labels[idx] == secColor:
                ele = 1
            lesion[int(idx // 256), int(idx % 256)] = ele
            
        kernel = np.ones((3,3), 'uint8')
        lesion_dialation = cv2.dilate(lesion, kernel, iterations = 1)
        lesion_erode = cv2.erode(lesion_dialation, kernel, iterations = 4)
        lesion_dilation = cv2.dilate(lesion_erode, kernel, iterations = 3)
        
        ans_ = cv2.bitwise_and(image, lesion_dilation * 255)
        ans_ = segmentation(ans_)
        ans = ans_.copy()
        ans = cv2.drawContours(lesion*123, mask, 0, (255, 0, 0), 1)
        
        plt.figure(figsize = (15, 7))
        plt.subplot(2, 4, 1)
        plt.imshow(lesion)
        plt.title('Lesion')
        plt.subplot(2, 4, 2)
        plt.imshow(ans)
        plt.title('bitwise_and with Image ')
#         plt.subplot(2, 4, 3)
#         plt.imshow(lesion_dilation)
        plt.subplot(2, 4, 3)
        plt.imshow(label_img)
        plt.title('Label Image')
        plt.show()
        
        return ans_, lesion_dilation


# In[ ]:


import pdb
def removeBone(img):
    img = img.copy()
    edges = skimage.feature.canny(img)
    mask = img.copy()
    mask[edges] = 0
    bone_threshold = filters.threshold_otsu(mask[mask!=0])
    bone_mask = np.zeros_like(img)
    bone_mask[img > bone_threshold] = 1
    erod_kernel = skimage.morphology.square(3)
    bone_mask = skimage.morphology.erosion(bone_mask, erod_kernel)
    bone_mask = skimage.morphology.erosion(bone_mask, erod_kernel)
    
    dilation_kernal = skimage.morphology.square(3)
    bone_mask = skimage.morphology.dilation(bone_mask, dilation_kernal)
    brain_mask = img.copy()
    brain_mask[bone_mask == 1] = 0
    brain_mask[brain_mask!= 0] = 1
    return brain_mask

def removeBone_all(img):
    img = img.copy()
    edges = skimage.feature.canny(img)
    mask = img.copy()
    mask[edges] = 0
    bone_threshold = filters.threshold_otsu(mask[mask!=0])
    bone_mask = np.zeros_like(img)
    bone_mask[img > bone_threshold] = 1
    erod_kernel = skimage.morphology.square(5)
    bone_mask = skimage.morphology.erosion(bone_mask, erod_kernel)
    dilation_kernal = skimage.morphology.square(8)
    bone_mask = skimage.morphology.dilation(bone_mask, dilation_kernal)
    brain_mask = img.copy()
    brain_mask[bone_mask == 1] = 0
    brain_mask[brain_mask!= 0] = 1

    dilation_kernal = skimage.morphology.square(20)
    bone_mask_ = skimage.morphology.dilation(bone_mask, dilation_kernal) 
    bone_mask_[brain_mask != 0] = 2

    bone_mask_[bone_mask == 1] = 0
    bone_mask_[brain_mask == 1] = 0

    erod_kernel = skimage.morphology.square(5)
    bone_mask_ = skimage.morphology.erosion(bone_mask_, erod_kernel)
    dilation_kernal = skimage.morphology.square(20)
    bone_mask_ = skimage.morphology.dilation(bone_mask_, dilation_kernal) 
    brain_mask[bone_mask_ == 1] = 0
    label_image = label(brain_mask, connectivity=mask.ndim)
    props = regionprops(label_image)
    for prop in props:
        if prop.area < 200:
            label_image[label_image == prop.label] = 0
    label_image[label_image!=0] = 1
    return label_image, bone_mask, bone_mask_

