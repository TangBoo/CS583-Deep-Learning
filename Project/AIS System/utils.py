import numpy as np
from scipy import signal
from sklearn.cluster import KMeans as kmeans
import cv2
from joblib import Parallel, delayed, parallel_backend
from glob import glob

svd = np.linalg.svd


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

    res = img[y_start: y_end, x_start: x_end].copy()

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


def joblib_process(sub_f, *args):
    result = []
    with parallel_backend('multiprocessing', n_jobs=-1):
        res = Parallel()(delayed(sub_f)(*[img, args[0], args[1]]) for img in args[0])
    return res


def Kmeans(seq, cluster=2):
    if isinstance(seq, list):
        raise ValueError
    seq = seq.reshape(-1, seq.shape[-1])
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
    :return: center:[cnt_y, cnt_x, height, width] -> [ymin, xmin, height, width] or box:[ymax, xmin, ymin, xmax] -> [ymin, xmin, height, width]
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


def add_box(img, mask):
    bbox = generator_bbox(mask, mode='box', expFactor=1).astype('int')
    for box in bbox:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (227, 23, 13), 1)
    return img


def smoothImg(imgs, kernel=3, stride=1, padding=1):
    if padding != 0:
        if padding >= kernel:
            raise ValueError
        imgs = np.pad(imgs, ((padding, padding), (0, 0), (0, 0)), mode='edge')
    totalNum = len(imgs)
    if totalNum < kernel:
        raise ValueError

    sliceNum = (totalNum - kernel) // stride + 1
    pivot_ = kernel // 2
    output_shape = (sliceNum,) + imgs.shape[1:]
    output = np.zeros(output_shape)
    for slid in range(sliceNum):
        pivot = pivot_ + slid * stride
        left = pivot - kernel // 2
        right = pivot + (kernel // 2 if kernel % 2 != 0 else 1)
        output[slid, ...] = np.mean(imgs[left:right + 1], axis=0)
    return output


def getCaselst(root):
    patient_lst = glob(root + '/*/')
    case_lst = []
    for i in range(len(patient_lst)):
        temp_lst = glob(patient_lst[i] + '/*/')
        for j in range(len(temp_lst)):
            if temp_lst[j].endswith('png/'):
                continue
            case_lst.append(temp_lst[j])
    return case_lst


def load4DImage(idx, case_lst, channel=0):
    slc_lst = glob(case_lst[idx] + '/*/')
    slc_lst = sorted(slc_lst, key=lambda x: x.split('/')[-2].split('_')[0])
    # [d, time, h, w]
    anat_imgs = []
    for i in range(len(slc_lst)):
        time_imgs = []
        time_lst = glob(slc_lst[i] + '/*')
        time_lst = sorted(time_lst, key=lambda x: int(x.split('/')[-1].split('.')[0].split('img')[-1]))
        for j in range(len(time_lst)):
            time_imgs.append(cv2.imread(time_lst[j], channel))
        anat_imgs.append(time_imgs)
    return np.array(anat_imgs).swapaxes(0, 1)


if __name__ == "__main__":
    x = np.random.rand(30, 1)
    filter_lp(x, 0.7, 0.5, ftype='lowpass')
