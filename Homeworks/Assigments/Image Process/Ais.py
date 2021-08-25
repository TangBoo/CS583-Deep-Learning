import pdb
from matplotlib import pyplot as plt
import Config as config
import numpy as np
from Computation import baseline_correction, deconvolution, deconv_nonparam_alg_tikhonov_svd, \
    deconv_circulant_matrix_fourier
from PostProcess import segmentation
from DataProcess import gen_mask_aif, gen_mask_brain
from utils import cal_average_aif_value, cal_mean_stddev, baseline_process, filter_lp, find_secMin, find_bbox, \
    get_ct_value_avg_aif, get_ct_value_neighbor_avg, Kmeans
import os
import cv2
from scipy.linalg import circulant
from IPython import display
from scipy.signal import find_peaks
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from numpy import trapz
from AifDetection import aifDetection, RGBMapping, creatVideo, add_box, saveVideo
import datetime

svd = np.linalg.svd

global_patient = config.global_patient
global_slc = config.global_slc
global_case = config.global_case
g_std_t = config.g_std_t
g_d = config.g_d


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
        self.img_mask = cv2.imread(r"{}/{}/{}_case_{}_{}_mask.png".format(mask_path, global_patient,
                                                                          global_patient, global_case, global_slc), 0)
        ret, img_mask_bin = cv2.threshold(self.img_mask, 127, 255, cv2.THRESH_BINARY)
        self.mask_contours, hierarchy = cv2.findContours(img_mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #         self.tissue_signal = np.zeros((256, 256, len(self.img_lst)))
        # --- Prepare original images, cbf images and tmax images ---
        self.images = np.zeros((len(self.img_lst), 256, 256))
        self.cbf_img = np.zeros((256, 256))
        self.irf_img = np.zeros((256, 256))
        self.cbv_img = np.zeros((256, 256))
        self.load_img()  # reading all image into memory
        self.tissue_signal = self.get_tissue_signal()
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
        output = np.zeros((256, 256, len(self.img_lst)))
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
            # array_ct_value_tissue_bl = baseline_process(array_ct_value_tissue, g_std_t)
            array_ct_value_tissue_bl_f = filter_lp(array_ct_value_tissue, cutoff_l=None, cutoff_h=0.41, ftype="lowpass")
            final_signal_tissue, base_tissue = baseline_correction(array_ct_value_tissue_bl_f)
            output[y_t_i, x_t_i, :] = final_signal_tissue

        return output

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
                        delayed(get_ct_value_neighbor_avg)(*[img, x_t_i, y_t_i, self.brain_mask, self.img_mask, g_d]) \
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


def main():
    global_all = '{}_case_{}_{}'.format(global_patient, global_case, global_slc)
    path_file = r"/home/aiteam_share/database/ISLES2018_brain_aif/{}".format(global_all)

    # create your fold for fitting bone and aif mask
    bone_aif_mask = r"/home/dxy/ais/aif_mask/{}".format(global_all)
    path_image = r"/home/aiteam_share/database/ISLES2018_4D"
    path_mask = r"/home/aiteam_share/database/ISLES2018_mask"
    gen_mask_aif(path_file, bone_aif_mask)
    gen_mask_brain(path_file, bone_aif_mask)
    sys = AisSystem(path_image, bone_aif_mask, path_mask, g_std_t=g_std_t, lamda=0.65)
    imgs = sys.images.copy()
    tissue_signal = sys.tissue_signal.copy()
    mask = sys.brain_mask.copy()

    start = datetime.datetime.now()
    vessel_mask, aif_mask, vessel_location, vessel_threshold, aif_threshold, timepoint = aifDetection(tissue_signal, imgs, mask)
    end = datetime.datetime.now()
    print('Time: {}s'.format((end - start).seconds))
    video = creatVideo(imgs, aif_threshold, vessel_threshold, vessel_location)
    saveVideo(video, 'aifDet', '', timepoint, aif_mask)


if __name__ == "__main__":
    main()
# starttime = datetime.datetime.now()
# irf, cbf, _ = sys.obtain_irf(nWorkers = 2, method = sys.compute_irf)
# # irf, cbf, conb = sys.obtain_tissue_tmax()
# endtime = datetime.datetime.now()
# print((endtime - starttime).seconds // 60)
#sys.show_R_signal()