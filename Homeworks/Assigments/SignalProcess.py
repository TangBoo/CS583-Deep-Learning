#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load tmax.py
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

svd = np.linalg.svd


# In[2]:


def gen_mask_aif(path_file, path_mask_dst):
    """This is for generating the bbox area mask of the manual selected AIF area.

    To open labelImg:
    open Anaconda Powershell Prompt.
    cd to labelImg folder.
    python labelImg.py

    Args:
        - path_file (str): path includes the xml file.
        - path_mask_dst (str): ...

    Returns: None
    """

    print("Generating AIF mask...")

    # ---- get path of the xml file ----
    list_files = os.listdir(path_file)
    for f in list_files:
        if f.endswith(".xml"):
            path_xml_dst = r"{}/{}".format(path_file, f)
            break

    DOMTree = xml.dom.minidom.parse(path_xml_dst)
    annotation = DOMTree.documentElement

    # ---- set AIF area -> 255 ----
    mask_aif = np.zeros([256, 256])
    objects = annotation.getElementsByTagName("object")
    for object in objects:
        bndboxs = object.getElementsByTagName("bndbox")
        for bndbox in bndboxs:
            xmin = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
            ymin = int(bndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
            xmax = int(bndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
            ymax = int(bndbox.getElementsByTagName("ymax")[0].childNodes[0].data)
            mask_aif[ymin:ymax, xmin:xmax] = 255

    # ---- saving AIF mask ----
    name_mask_aif = "{}_case_{}_{}_aif.png".format(global_patient, global_case, global_slc)

    if os.path.exists(path_mask_dst) == False:
        os.mkdir(path_mask_dst)
    cv2.imwrite(r"/{}/{}".format(path_mask_dst, name_mask_aif), mask_aif)
    print("AIF mask was done.")


def gen_mask_brain(path_file, path_mask_dst):
    """This is for generating the polygonal area mask of the manual selected brain area.

    To open labelme:
    open Anaconda Powershell Prompt.
    # labelme

    Args:
        - path_file (str): path includes the json file.
        - path_mask_dst (str): ...

    Returns: None
    """

    print("Generating brain mask...")

    # ---- get path of the json file ----
    list_files = os.listdir(path_file)
    for f in list_files:
        if f.endswith(".json"):
            path_json_dst = r"{}/{}".format(path_file, f)
            break

    # parse json file.
    mask_brain = np.zeros([256, 256])
    f = open(path_json_dst, encoding='utf-8')
    setting = json.load(f)
    shapes = setting['shapes']
    contours = []
    for i in range(len(shapes)):
        contour = []
        points = shapes[i]["points"]
        contour.append(points)
        contour_np = np.array(contour)
        c = contour_np.astype(int)
        contours.append(c)

    for contour in contours:
        cv2.drawContours(mask_brain, contour, 0, (255, 255, 255), cv2.FILLED)

    # saving AIF mask.
    name_mask_brain = "{}_case_{}_{}_brain.png".format(global_patient, global_case, global_slc)
    if os.path.exists(path_mask_dst) == False:
        os.mkdir(path_mask_dst)
    cv2.imwrite(r"/{}/{}".format(path_mask_dst, name_mask_brain), mask_brain)
    print("Brain mask was done.")


# In[3]:


# Data Processing:
import copy


def get_ct_value_neighbor_avg(img, x, y, brain_mask, lesion_mask, d=7):
    """
    Input:
        image,
        pixel:[y, x]
        brain_mask : binary brain_mask
        d: radius around pixel
    Output:
        mean of box around pixel
    Discribe:
        Pixels around [x,y] must be covered by brain_mask.
        If [x, y] is in lesion, all pixel around [x, y] cannot be counted;
        If not in lesion, pixels around [x,y] in lesion cannot be count.
    """
    size = img.shape[0]
    x_start = max(x - d, 0)
    x_end = min(x + d, size)
    y_start = max(y - d, 0)
    y_end = min(y + d, size)

    res = np.zeros((2 * d, 2 * d))
    res[...] = img[y_start: y_end, x_start: x_end]

    brain_mask_n = np.zeros_like(res)
    brain_mask_n[...] = brain_mask[y_start: y_end, x_start: x_end]
    res *= brain_mask_n / 255
    if lesion_mask[y, x] == 0:
        res[np.where(lesion_mask[y_start: y_end, x_start: x_end] != 0)] = 0
    elif lesion_mask[y, x] != 0:
        res[np.where(lesion_mask[y_start: y_end, x_start: x_end] == 0)] = 0

    count = (2 * d) ** 2 - len(np.where(res == 0)[1])
    return res.sum() / count


def get_ct_value_avg_aif(img, x_start, x_end, y_start, y_end):
    """
    """
    ct_value_sum = 0
    n = 0
    for x_i in range(x_start - 4, x_end + 4):
        for y_i in range(y_start - 4, y_end + 4):
            ct_value_sum += img[y_i, x_i]
            n += 1

    ct_value_avg = ct_value_sum / n
    return ct_value_avg


# In[4]:


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
    for x_i in range(x_aif, x_aif + w_aif):
        for y_i in range(y_aif, y_aif + h_aif):
            ct_value_sum += img_grey[y_i, x_i]
            n += 1

    ct_value_avg = ct_value_sum / n
    return ct_value_avg


# In[5]:


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
    if ftype == "lowpass":
        b, a = signal.butter(9, cutoff_h, 'lowpass')
    elif ftype == "bandpass":
        b, a = signal.butter(7, [cutoff_l, cutoff_h], 'bandpass')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData


# In[6]:


# Signal Process Helper
def svd_denoising(array_1d):
    """
    """
    matrix_cir = circulant(array_1d)
    matrix_cir_sparse = sparse_matrix(matrix_cir)
    u, sigma, vt = np.linalg.svd(matrix_cir_sparse)

    threshold = sigma[2] * 0.10

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
    """Deconvolution-based / Non-parametric / Tikhonov SVD
    """
    a = array_ct_value_aif
    c = array_ct_value_tissue

    a_pad = np.pad(a, (0, len(a)))
    c_pad = np.pad(c, (0, len(a)))

    I = np.identity(2 * len(a))

    A = circulant(a_pad)
    A_T = A.T

    block_cirMat = np.dot(A_T, A) + (lamdaa ** 2) * I
    b = solve(block_cirMat, A_T @ c_pad)

    b = b[0:len(a)]
    return b


def drive_singal(seq):
    power = np.zeros_like(seq)
    for i in range(len(power) - 1):
        res = seq[i + 1] - seq[i]
        if res > 0:
            power[i] = 1
        elif res == 0:
            power[i] = 0
        else:
            power[i] = -1

    for j in range(len(power) - 1):
        res = power[j]
        if j - 1 < 0:
            continue
        if res == 0 and power[j - 1] > 0:
            power[j] = 1
        elif res == 0 and power[j - 1] < 0:
            power[j] = -1
    return power


def find_valley_peak(seq):
    '''
    Input:
        signal sequence
    Output:
        power signal
    '''
    # take the derivative twice for getting movement trend
    fir_p = drive_singal(seq)
    sec_p = drive_singal(fir_p)
    return sec_p


def find_mostRight(seq, target=-1):
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


def find_longestLine(seq, target=-1):
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
            while q_p + 1 < len(seq) and seq[q_p + 1] == target:
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
    # peak is -1, valley is 1
    sec_power = find_valley_peak(signal_seq)  # power signal
    left_side = np.min(np.where(sec_power == 1))
    peak_p = np.argmax(signal_seq[left_side:]) + left_side

    _, right_side = find_longestLine(sec_power[peak_p:], target=-1) + peak_p

    return left_side, right_side


def truncate_aif(signal_seq):
    sec_power = find_valley_peak(signal_seq)  # power signal
    left_side = np.min(np.where(sec_power == 1)) + 1  # get first 1 in power signal
    right_side = find_mostRight(sec_power)

    return left_side, right_side


def baseline_correction(signal_seq, name="tissue"):
    '''
    input :
        signal sequence
    output :
        res : peak of signal
        base_signal: signal sequence without baseline shift

    '''
    base_signal = BaselineRemoval(signal_seq).IModPoly(2)  # getting off baseline shift

    if name == "tissue":
        left_side, right_side = truncate_tissue(base_signal)  # get the left and right boundary of peak indexes
    else:
        left_side, right_side = truncate_aif(base_signal)

    res = copy.deepcopy(base_signal)

    # --- pick peak ---

    #     res -= res[left_side - 1]

    res[:left_side] = 0
    res[right_side + 1:] = 0
    res[res < 0] = 0
    return res, base_signal


# In[7]:


# Compute IRF:

def sparse_matrix(matrix_array):
    # matrix_array_c = circulant(matrix_array)
    matrix_c_list = matrix_array.tolist()

    matrix_sparse = np.zeros(shape=matrix_array.shape)
    n1 = 1
    for r in matrix_c_list:
        n2 = 1
        for i in r:
            if n2 > n1:
                matrix_sparse[n1 - 1, n2 - 1] = 0
            else:
                matrix_sparse[n1 - 1, n2 - 1] = i
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


# In[8]:


def sparse_mat(seq, lamda):
    '''
    Tikhonov svd
    input:
        aif signal, superparameter of regularization
    output:
        new aif matrix with lower condition number
    '''
    if seq.shape[0] == 1:
        seq = seq.squeeze(axis=0)
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
    with parallel_backend('multiprocessing', n_jobs=-1):
        res = Parallel()(delayed(sub_f)(*[img, args[0], args[1]]) for img in args[0])
    return res


def Kmeans(seq, num_f=1, cluster=2):
    seq = seq.reshape(-1, 1)
    y_pred = kmeans(n_clusters=cluster, init='k-means++', n_jobs=-1).fit(seq)
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


# In[64]:


def postProcess(image, mask, ptype='irf', n=4):
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

    predi = Kmeans(image[mask != 0], cluster=ncluster)

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

    kernel = np.ones((3, 3), 'uint8')
    lesion_dialation = cv2.dilate(lesion, kernel, iterations=1)
    lesion_erode = cv2.erode(lesion_dialation, kernel, iterations=4)
    lesion_dilation = cv2.dilate(lesion_erode, kernel, iterations=3)

    ans_ = cv2.bitwise_and(image, lesion_dilation * 255)
    ans_ = segmentation(ans_)
    ans = ans_.copy()
    ans = cv2.drawContours(lesion * 123, mask, 0, (255, 0, 0), 1)

    plt.figure(figsize=(15, 7))
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


# In[10]:


def filter_lp(data, cutoff_l, cutoff_h, ftype):
    """
    low pass filter
    """
    if ftype == "lowpass":
        b, a = signal.butter(9, cutoff_h, 'lowpass')
    elif ftype == "bandpass":
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
        cir_mat[ii, :] = np.r_[tail, head]
    ans = cir_mat.mean(0)

    peaks, properties = find_peaks(ans, prominence=0, width=[0, 2])
    left_bases = properties['left_bases']
    right_bases = properties['right_bases']
    idex = np.argmax(ans[peaks])

    left = left_bases[idex] if abs(left_bases[idex] - peakposi) < abs(right_bases[idex - 1] - peakposi) else \
    right_bases[idex - 1]
    right = right_bases[idex] if abs(right_bases[idex] - peakposi) < abs(left_bases[idex + 1] - peakposi) else \
    left_bases[idex + 1]

    leftpart = ans[:left]
    rightpart = ans[right:]
    midpart = ans[left: right]

    leftpart = cv2.GaussianBlur(leftpart, (1, 3), 0.7)
    rightpart = cv2.GaussianBlur(rightpart, (1, 3), 0.7)
    ans = np.r_[leftpart.squeeze(-1), midpart, rightpart.squeeze(-1)]
    return circulant(ans).T


# In[30]:


import pdb


# In[41]:


class AisSystem:
    def __init__(self, img_path, brain_aif_mask, mask_path, g_std_t, lamda, **kwargs):

        self.img_path = r"{}/{}".format(img_path, global_patient)
        file_lst = os.listdir(self.img_path)

        self.lamda = lamda

        # --- get image files list---
        img_lst = []
        for f in file_lst:
            if f.endswith(".png"):
                case_i = f.split("_")[2]
                slc_i = f.split("_")[3]
                if case_i == global_case and slc_i == global_slc:
                    img_lst.append(f)

        self.img_lst = sorted(img_lst,
                              key=lambda x: int(x.split('_')[4]))  # Sort image files belonging single slice with time
        self.g_std_t = g_std_t

        # --- Get ground truth, aif mask and brain mask ---
        for f in os.listdir(brain_aif_mask):
            if f.split('_')[-1] == "aif.png":
                self.aif_mask = cv2.imread(r"{}/{}".format(brain_aif_mask, f), 0)
            elif f.split('_')[-1] == 'brain.png':
                self.brain_mask = cv2.imread(r"{}/{}".format(brain_aif_mask, f), 0)
        self.img_mask = cv2.imread(
            r"{}/{}/{}_case_{}_{}_mask.png".format(mask_path, global_patient, global_patient, global_case, global_slc),
            0)
        ret, img_mask_bin = cv2.threshold(self.img_mask, 127, 255, cv2.THRESH_BINARY)
        self.mask_contours, hierarchy = cv2.findContours(img_mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.tissue_signal = np.zeros((256, 256, len(self.img_lst)))
        # --- Prepare original images, cbf images and tmax images ---
        self.images = np.zeros((len(self.img_lst), 256, 256))
        self.cbf_img = np.zeros((256, 256))
        self.irf_img = np.zeros((256, 256))
        self.cbv_img = np.zeros((256, 256))
        self.load_img()  # reading all image into memory
        #         self.tmax = self.get_tissue_signal()
        self.aif_seq, self.aif_contours = self.get_aif_signal()
        self.final_signal_aif, self.base_aif = baseline_correction(self.aif_seq, name="aif")

    def fake_aif_signal(self, peak_t, factor, isPad=False):

        length = len(self.img_lst)
        if isPad:
            length *= 2

        res = np.zeros((length))
        res[peak_t] = factor
        return res

    def get_aif_signal(self):

        ret, img_mask_aif_bin = cv2.threshold(self.aif_mask, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(img_mask_aif_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x_aif, y_aif, w_aif, h_aif = cv2.boundingRect(contours[0])

        # ---- get aif tdc array ----
        with parallel_backend('threading', n_jobs=-1):
            aif_lst = Parallel()(
                delayed(cal_average_aif_value)(*[img, x_aif, y_aif, w_aif, h_aif]) for img in self.images)

        array_ct_value_aif = np.array(aif_lst)
        array_ct_value_aif_bl = baseline_process(array_ct_value_aif, g_std_t)
        array_ct_value_aif_bl_f = filter_lp(array_ct_value_aif_bl, cutoff_l=None, cutoff_h=0.41, ftype="lowpass")
        return array_ct_value_aif_bl_f, contours

    def load_img(self):
        for idx, slc_i in tqdm(enumerate(self.img_lst)):
            self.images[idx, ...] = cv2.imread(r"{}/{}".format(self.img_path, slc_i), 0)

    def read_img(self, index):
        self.images[index, ...] = cv2.imread(r"{}/{}".format(self.img_path, self.img_lst[index]), 0)

    def get_tissue_signal(self):

        tmax = np.zeros((256, 256))
        for ii in tqdm(range(256 * 256)):
            y_t_i = int(ii // 256)
            x_t_i = int(ii % 256)

            if self.brain_mask[y_t_i, x_t_i] == 0:
                continue

            length = len(self.img_lst)
            tissue_lst = [0] * length
            for idx, img in enumerate(self.images):
                tissue_lst[idx] = get_ct_value_neighbor_avg(img, x_t_i, y_t_i, self.brain_mask, self.img_mask, g_d)

            array_ct_value_tissue = np.array(tissue_lst)
            array_ct_value_tissue_bl = baseline_process(array_ct_value_tissue, g_std_t)
            array_ct_value_tissue_bl_f = filter_lp(array_ct_value_tissue, cutoff_l=None, cutoff_h=0.41, ftype="lowpass")
            final_signal_tissue, base_tissue = baseline_correction(array_ct_value_tissue_bl_f)
            tmax[y_t_i, x_t_i] = np.argmax(final_signal_tissue)
        return tmax

    def deconvolution(self, array_ct_value_aif, array_ct_value_tissue):
        cir_aif = circulant(array_ct_value_aif)
        inver_sig = np.linalg.inv(cir_aif)
        sig = inver_sig[0, :]
        peaks, properties = find_peaks(sig, prominence=0)
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
        array_ct_value_tissue_bl_f = filter_lp(array_ct_value_tissue, cutoff_l=None, cutoff_h=0.41, ftype="lowpass")

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
        array_ct_value_tissue_bl_f = filter_lp(array_ct_value_tissue, cutoff_l=None, cutoff_h=0.41, ftype="lowpass")
        final_signal_tissue, base_tissue = baseline_correction(array_ct_value_tissue_bl_f)

        aif_delay = np.min(np.where(self.final_signal_aif > 0))
        tissue_delay = np.min(np.where(final_signal_tissue > 0))

        irf_delay = abs(tissue_delay - aif_delay)

        aif_peak = np.argmax(self.final_signal_aif)
        tissue_peak = np.argmax(final_signal_tissue)
        max_aif = np.max(self.final_signal_aif)
        self.cbf_img[y_t_i, x_t_i] = (1 / max_aif) * np.max(final_signal_tissue)
        self.cbv_img[y_t_i, x_t_i] = trapz(final_signal_tissue, np.arange(len(final_signal_tissue)), dx=1) / trapz(
            self.final_signal_aif, np.arange(len(self.final_signal_aif)), dx=1)

        if tissue_peak >= aif_peak:
            self.irf_img[y_t_i, x_t_i] += (tissue_peak - aif_peak)

    def obtain_irf(self, nWorkers=1, method=compute_irf):
        '''
        nWorkers = -1, using all available cpu resource
        '''
        task_index = []
        for ii in tqdm(range(0, 256 * 256)):
            if self.brain_mask[int(ii // 256), int(ii % 256)] == 0:
                continue
            task_index.append(ii)

        Parallel(n_jobs=nWorkers, backend='threading')([delayed(method)(i) for i in tqdm(task_index)])
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
            array_ct_value_tissue_bl_f = filter_lp(array_ct_value_tissue, cutoff_l=None, cutoff_h=0.41, ftype="lowpass")
            final_signal_tissue, base_tissue = baseline_correction(array_ct_value_tissue_bl_f)
            tmax_seq[ii] = np.argmax(final_signal_tissue)
            cbf_seq[ii] = final_signal_tissue.max()
            comb_seq[ii] = tmax_seq[ii] * cbf_seq[ii]

        return tmax_seq.reshape((256, 256)), cbf_seq.reshape((256, 256)), comb_seq.reshape((256, 256))

    def postProcess(self, image, ptype='irf'):
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

        predi = Kmeans(image, cluster=ncluster)
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
        kernel = np.ones((3, 3), 'uint8')
        lesion_dialation = cv2.dilate(lesion, kernel, iterations=1)
        lesion_erode = cv2.erode(lesion_dialation, kernel, iterations=4)
        lesion_dilation = cv2.dilate(lesion_erode, kernel, iterations=3)
        res ^= lesion_dilation
        return res

    def show_R_signal(self):

        final_signal_aif, base_aif = baseline_correction(self.aif_seq, name="aif")

        for x_t_i in range(100, 256):
            for y_t_i in range(100, 256):
                if self.brain_mask[y_t_i, x_t_i] == 0:
                    continue

                # ---draw contours---
                img = cv2.imread(r"{}/{}".format(self.img_path, self.img_lst[0]), cv2.IMREAD_COLOR)
                img = cv2.drawContours(img, self.aif_contours, 0, (153, 0, 153), 1)
                img = cv2.drawContours(img, self.mask_contours, 0, (255, 0, 0), 1)
                cv2.rectangle(img, (max(0, x_t_i - 2), max(y_t_i - 2, 0)),
                              (min(256, x_t_i + 2), min(256, y_t_i + 2)), (227, 23, 13), 1)

                # ---Tissue TDC---

                with parallel_backend('threading', n_jobs=-1):
                    tissue_lst = Parallel()(
                        delayed(get_ct_value_neighbor_avg)(*[img, x_t_i, y_t_i, self.brain_mask, self.img_mask, g_d])
                        for img in self.images)

                array_ct_value_tissue = np.array(tissue_lst)
                array_ct_value_tissue_bl = baseline_process(array_ct_value_tissue, g_std_t)
                array_ct_value_tissue_bl_f = filter_lp(array_ct_value_tissue_bl, cutoff_l=None, cutoff_h=0.41,
                                                       ftype="lowpass")
                final_signal_tissue, base_tissue = baseline_correction(array_ct_value_tissue_bl_f)

                # --- Compute IRF ---
                residual_func = deconv_nonparam_alg_tikhonov_svd(self.final_signal_aif, final_signal_tissue,
                                                                 lamdaa=self.lamda)
                residual_func_blur, sig = deconvolution(self.final_signal_aif, final_signal_tissue, show=True)

                # --- Show Main Peak ---
                #                 residual_func_baseline  = BaselineRemoval(residual_func).IModPoly(2)
                #                 residual_func_baseline[residual_func_baseline < 0] = 0

                #                 peaks, properties = find_peaks(residual_func_baseline, prominence = 0)

                #                 left_bases = properties['left_bases']
                #                 right_bases = properties['right_bases']
                #                 idex = np.argmax(residual_func_baseline[peaks])

                #                 residual_func_baseline[:left_bases[idex]] = 0
                #                 residual_func_baseline[right_bases[idex]:] = 0

                tissue_t = np.argmax(final_signal_tissue)
                aif_t = np.argmax(final_signal_aif)
                print("tissue tmax:", tissue_t, "aif tmax:", aif_t, "irf tmax:", tissue_t - aif_t)

                # inverse aif
                cir_aif = circulant(self.final_signal_aif)
                inver_sig = np.linalg.inv(cir_aif)
                o_sig = inver_sig[0, :].copy()

                pdb.set_trace()
                # --- Plot Image ---
                plt.figure(figsize=(15, 7))

                display.clear_output(wait=True)
                plt.subplot(2, 4, 1)
                plt.imshow(img)
                #                 plt.title("{}, x:{}, y:{}".format(x_t_i, y_t_i))

                plt.subplot(2, 4, 2)
                plt.plot(residual_func, label="irf signal")
                #                 plt.plot(residual_func_blur, label = "irf_blur signal")
                #             plt.legend()
                plt.minorticks_on()
                plt.title('IRF')

                plt.subplot(2, 4, 3)
                plt.plot(array_ct_value_tissue_bl, label="array_tissue_signal_f")
                plt.plot(base_tissue, label="tissue without baseline shift")
                plt.plot(final_signal_tissue, label="final_signal_tissue")
                plt.legend()
                plt.title('Tissue TDC')

                plt.subplot(2, 4, 4)
                plt.plot(final_signal_aif, label="aif signal")
                plt.plot(final_signal_tissue, label="tissue signal")
                plt.plot(sig * 100, label='reg inv_aif')
                plt.legend()
                plt.title('AIF & Tissue TDC')

                plt.subplot(2, 4, 5)
                plt.plot(self.aif_seq, label="array_aif_bl_f")
                plt.plot(base_aif, label="aif without baseline shift")
                plt.plot(final_signal_aif, label="final_signal_aif")
                plt.legend()
                plt.title('AIF TDC')

                plt.subplot(2, 4, 6)
                plt.plot(o_sig * 100, label='inverse aif')
                plt.plot(final_signal_tissue, label='tissue')
                plt.legend()
                plt.title('inverse aif')

                plt.subplot(2, 4, 7)
                plt.plot(residual_func_blur)
                #                 plt.plot(residual_func * 10, label = 'svd irf')
                plt.title('irf blur')

                plt.show()
                plt.pause(0.8)
                plt.close()


# In[48]:


global_patient = "24"
global_case = "34"
global_slc = "1"

g_d = 7
g_std_t = 0.65

global_all = '{}_case_{}_{}'.format(global_patient, global_case, global_slc)
path_file = r"/home/aiteam_share/database/ISLES2018_brain_aif/{}".format(global_all)

# create your fold for fitting bone and aif mask
bone_aif_mask = r"/home/dxy/ais/aif_mask/{}".format(global_all)
path_image = r"/home/aiteam_share/database/ISLES2018_4D"
path_mask = r"/home/aiteam_share/database/ISLES2018_mask"

gen_mask_aif(path_file, bone_aif_mask)
gen_mask_brain(path_file, bone_aif_mask)

sys = AisSystem(path_image, bone_aif_mask, path_mask, g_std_t=g_std_t, lamda=0.65)

# In[50]:


starttime = datetime.datetime.now()
irf, cbf, _ = sys.obtain_irf(nWorkers=2, method=sys.compute_irf)
# irf, cbf, conb = sys.obtain_tissue_tmax()
endtime = datetime.datetime.now()
print((endtime - starttime).seconds // 60)

# In[49]:


sys.show_R_signal()


# In[35]:


def show_signal(mat, tissue):
    size = len(mat)
    plt.figure(figsize=(15, 7))
    col_num = 8
    row_num = np.ceil(size / col_num)
    for i in range(size):
        plt.subplot(row_num, col_num, i + 1)
        plt.plot(mat[i, :])
        plt.plot(tissue)
    plt.show()


# In[43]:


def regulSig(cir_mat):
    size = cir_mat.shape[0]
    #     peakposi = np.argmax(cir_mat[:,0])

    cir_mat = np.linalg.inv(cir_mat)
    peakposi = np.argmax(cir_mat[0, :])
    print(peakposi)

    #     show_signal(cir_mat)
    for ii in range(size):
        head = cir_mat[ii][:ii]
        tail = cir_mat[ii][ii:]
        cir_mat[ii, :] = np.r_[tail, head]
    ans = cir_mat.mean(0)

    peaks, properties = find_peaks(ans, prominence=0, width=[0, 2])
    print(peaks, properties)
    left_bases = properties['left_bases']
    right_bases = properties['right_bases']
    idex = np.argmax(ans[peaks])
    print('idex:', idex)
    print(left_bases[idex], right_bases[idex])

    left = left_bases[idex] if abs(left_bases[idex] - peakposi) < abs(right_bases[idex - 1] - peakposi) else \
    right_bases[idex - 1]
    print('left:', abs(left_bases[idex] - peakposi), abs(right_bases[idex - 1] - peakposi))
    right = right_bases[idex] if abs(right_bases[idex] - peakposi) < abs(left_bases[idex + 1] - peakposi) else \
    left_bases[idex + 1]
    print('right', abs(right_bases[idex] - peakposi), abs(left_bases[idex + 1] - peakposi))

    leftpart = ans[:left]
    rightpart = ans[right:]
    print(left, right)
    midpart = ans[left: right]

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

# In[144]:


plt.figure()
plt.subplot(1, 3, 1)
plt.plot(aif)
plt.subplot(1, 3, 2)
plt.plot(mat[:, 0])

plt.show()

# In[145]:


plt.figure()
plt.plot(mat_inv[0])
plt.show()

# In[51]:


from skimage import data, filters
from skimage import transform, exposure

irf_img = irf.copy()
cbf_img = cbf.copy()
# cbv_img = cbv.copy()
brain_mask = sys.brain_mask

# In[52]:


# 21, 31, 01
plt.figure(figsize=(15, 7))
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


# 21, 31, 01
plt.figure(figsize=(15, 7))
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


# 22, 31, 01
plt.figure(figsize=(15, 7))
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


plt.figure(figsize=(15, 7))
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


plt.figure(figsize=(15, 7))
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


irf_label, res = postProcess(irf_img, sys.mask_contours, ptype='irf', n=3)

# In[89]:


plt.imshow(res)

# In[79]:


set(irf_img.flatten())

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
mat[y - 5: y + 5, x - 5: x + 5] = 255
plt.imshow(mat)

# In[87]:


index = np.argmax(irf_img)
y = int(index // 256)
x = int(index % 256)
y, x

# In[82]:


cv2.rectangle(mark, (max(0, x - 2), max(y - 2, 0)), (min(256, x + 2), min(256, y + 2)), (227, 23, 13), 1)
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
    sobel_x = cv2.Sobel(aeh_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(aeh_img, cv2.CV_64F, 0, 1, ksize=3)
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
    dist_map[dist_map != 0] = maxV - dist_map[dist_map != 0]  # pixel inversion

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

    ret, mask = cv2.threshold(dist_map, 127, 255, 0)
    mask = cv2.drawContours(mask, sys.mask_contours, 0, (255, 0, 0), 1)
    #     cv2.rectangle(mask, (max(0, x - 2), max(y - 2, 0)),
    #                               (min(256, x + 2),min(256, y + 2)), (227,23,13), 1)

    plt.figure(figsize=(15, 7))
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


res = extract_cbf(cbf_img, (x, y), brain_mask=sys.brain_mask)
plt.imshow(res)

# In[22]:


h, w = irf_img.shape

aeh_img = exposure.equalize_adapthist(irf_img / 255, kernel_size=None, clip_limit=0.01, nbins=256)
sobel_x = cv2.Sobel(aeh_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(aeh_img, cv2.CV_64F, 0, 1, ksize=3)
sobel_xy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
plt.figure(figsize=(15, 7))
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


input_img = np.stack((irf_img, cbf_img), axis=0)

# In[64]:


input_img.shape

# In[68]:


input_img = input_img.reshape((256, 256, 2))

# In[95]:


seq = input_img.reshape(-1, 2)
predi = kmeans(n_clusters=4, init='k-means++', n_jobs=-1).fit(seq)
label_img = predi.labels_

# In[96]:


label_img.resize((256, 256, 1))
label_img.shape

# In[223]:


# Hession Detector

sobel_x = cv2.Sobel(aeh_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(aeh_img, cv2.CV_64F, 0, 1, ksize=3)
sobel_xy = cv2.Sobel(sobel_x, cv2.CV_64F, 0, 1, ksize=3)
sobel_yy = cv2.Sobel(sobel_y, cv2.CV_64F, 0, 1, ksize=3)
sobel_xx = cv2.Sobel(sobel_x, cv2.CV_64F, 1, 0, ksize=3)
matter = img[brain_mask == 255]
matter_xy = cv2.Laplacian(matter, cv2.CV_64F)
lap = img.copy()
lap[brain_mask == 255] = matter_xy.ravel()

sobel_2xy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
plt.figure(figsize=(15, 7))
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
plt.figure(figsize=(15, 7))
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

