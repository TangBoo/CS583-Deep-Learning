# %load tmax.py
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
from scipy.linalg import circulant
from scipy.linalg import solve
from scipy import signal
from BaselineRemoval import BaselineRemoval
import copy
from scipy.signal import find_peaks
from skimage import data, filters
from skimage import transform, exposure


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


def get_center(mask):
    index = np.where(mask != 0)
    center_y = int(index[0].mean())
    center_x = int(index[1].mean())
    return center_y, center_x


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
    # mask = cv2.drawContours(mask, sys.mask_contours, 0, (255, 0, 0), 1)
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

