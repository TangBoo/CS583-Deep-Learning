import numpy as np
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import warnings
from Realign import getCaselst, getSpace, load4DDcm, load4DImage, Registration, seriesImg_show
from utils import smoothImg
from BrainBoneRemove import BoneRemove
from IPython import display
import matplotlib.pyplot as plt
import cv2
warnings.filterwarnings('ignore')


def drive_signal(seq, threshold):
    power = np.zeros_like(seq)
    for i in range(len(power) - 1):
        res = seq[i + 1] - seq[i]
        if res > threshold:
            power[i] = 1
        elif -threshold <= res <= threshold:
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


def find_mainPeak(seq, threshold=0.1, moveUp=True):
    seq = seq.copy()
    power = drive_signal(seq, threshold)
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
        while right < len(sec_power) and sec_power[right] != 2:
            right += 1

        if (right - left + 1) >= max_length:
            ans_left = left
            ans_right = right
            max_length = (ans_right - ans_left + 1)

    left, right = ans_left, ans_right

    if moveUp and seq[left] < 0:
        seq += abs(seq[left])
    seq[seq < 0] = 0
    seq[:left] = 0
    seq[right:] = 0
    return seq


def poly(input_array_for_poly, degree_for_poly):
    input_array_for_poly = np.array(input_array_for_poly)
    X = np.transpose(np.vstack((input_array_for_poly ** k for k in range(degree_for_poly + 1))))
    return np.linalg.qr(X)[0][:, 1:]


def leastsqr_fit(input_array, degree=6):
    lin = LinearRegression()
    input_array_for_poly = np.array(input_array)
    poly_x = poly(list(range(1, 30 + 1)), degree)
    poly_x = np.linalg.qr(poly_x)[0][:, 1:]
    ypred = lin.fit(poly_x, input_array).predict(poly_x)
    return ypred


def baseline_correction(signal, degree, repitition, gradient=0.001):
    lin = LinearRegression()
    poly_x = poly(list(range(1, len(signal) + 1)), degree)
    poly_x = np.linalg.qr(poly_x)[0][:, 1:]
    yold = signal.copy()
    yorig = signal.copy()
    corrected = []
    nrep = 1
    ngradient = 1
    ypred = lin.fit(poly_x, yold).predict(poly_x)
    Previous_Dev = np.std(yorig - ypred)

    # iteration1
    yold = yold[yorig <= (ypred + Previous_Dev)]
    poly_x_updated = poly_x[yorig <= (ypred + Previous_Dev)]
    ypred = ypred[yorig <= (ypred + Previous_Dev)]

    for i in range(2, repitition + 1):
        if i > 2:
            Previous_Dev = DEV
        ypred = lin.fit(poly_x_updated, yold).predict(poly_x_updated)
        DEV = np.std(yold - ypred)

        if np.abs((DEV - Previous_Dev) / DEV) < gradient:
            break
        else:
            for i in range(len(yold)):
                if yold[i] >= ypred[i] + DEV:
                    yold[i] = ypred[i] + DEV
    baseline = lin.predict(poly_x)
    corrected = yorig - baseline
    return baseline


def signal_process(signal_img, brain_mask):
    time, h, w = signal_img.shape
    val_loc = brain_mask != 0
    signal_input = signal_img[:, val_loc].swapaxes(0, 1)
    simulation = np.zeros_like(signal_img)
    with parallel_backend('multiprocessing', n_jobs=-1):
        res = Parallel()(delayed(leastsqr_fit)(*[signal]) for signal in tqdm(signal_input))
    res = np.array(res)
    simulation[:, val_loc] = res.swapaxes(0, 1)

    bl_input = res
    baseline = np.zeros_like(simulation)
    with parallel_backend('multiprocessing', n_jobs=-1):
        res = Parallel()(delayed(baseline_correction)(*[signal, 2, 5]) for signal in tqdm(bl_input))
    baseline[:, val_loc] = np.array(res).swapaxes(0, 1)

    curve_fit = simulation.astype('float') - baseline.astype('float')
    mainPeak_input = curve_fit[:, val_loc].copy()
    mainPeak_output = np.zeros_like(curve_fit)
    with parallel_backend('multiprocessing', n_jobs=-1):
        res = Parallel()(delayed(find_mainPeak)(*[signal]) for signal in tqdm(mainPeak_input.swapaxes(0, 1)))
    mainPeak_output[:, val_loc] = np.array(res).swapaxes(0, 1)
    return curve_fit, mainPeak_output


if __name__ == "__main__":
    png_path = "/data/aiteam_ctp/database/AIS_210713/0713_dst_png"
    dcm_path = "/data/aiteam_ctp/database/AIS_210713/0713_dst"
    case_list_dcm = getCaselst(dcm_path)
    case_list_png = getCaselst(png_path)

    # time_anat_imgs_dcm = load4DDcm(0, case_list_dcm)
    idx = -1
    time_anat_imgs = load4DImage(idx, case_list_png, 0)
    space = getSpace(idx, case_list_dcm)
    time_anat_imgs, shifts, rotations = Registration(time_anat_imgs, space=space)
    # print(output.shape)
    brain_mask, vencle_mask = BoneRemove(time_anat_imgs[0])
    final_mask = brain_mask.copy() ^ vencle_mask
    anat_time_imgs = time_anat_imgs.swapaxes(0, 1)
    # seriesImg_show((anat_time_imgs * np.expand_dims(final_mask, axis=1))[-1])
    signal_input = smoothImg(anat_time_imgs[0], kernel=3, stride=1, padding=3//2)
    simulation, mainPeak = signal_process(signal_input, final_mask[0])
    # seriesImg_show(mainPeak)
    _, h, w = mainPeak.shape
    for i in range(160000, h * w):
        row = int(i // h)
        col = int(i % w)
        if final_mask[15, row, col] == 0:
            continue
        tmp = cv2.rectangle(signal_input[0].copy().astype('uint8'), (col - 2, row - 2),
                            (col + 2, row + 2), (227, 255, 13), 3)

        plt.figure(figsize=(15, 7))
        display.clear_output(True)
        plt.subplot(1, 3, 1)
        plt.plot(signal_input[..., row, col] - 200, label='original')
        plt.plot(simulation[..., row, col], label='simulation')
        plt.plot(mainPeak[..., row, col], label='mainPeak')
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.imshow(tmp)
        plt.subplot(1, 3, 3)
        plt.imshow(final_mask[15])
        plt.show()
        plt.pause(0.5)
        plt.close()

