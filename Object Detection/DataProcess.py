import glob
import random
import Config as config
import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_json(json_path):
    file = open(json_path, encoding='utf-8')
    setting = json.load(file)
    shapes = setting['shapes']
    for item in shapes:
        if item['label'] == 'brain':
            return False
    return True


def load_json(path_lst):
    json_lst = []
    for f_name in path_lst:
        jsons = glob.glob(f_name + '/*.json')
        lst = []
        for f in jsons:
            if not read_json(f):
                continue
            lst.append(f)
        if len(lst) == 0:
            print(f_name)

        del_i = []
        for i in range(len(lst)):
            if 'aif' in lst[i].split('/')[-1]:
                del_i.append(i)
        lst = [lst[i] for i in range(len(lst)) if i not in del_i]
        json_lst += lst
    return json_lst


def get_ImgPath():
    path_lst = glob.glob(config.Img_root + '/*_part/*')
    name_set = set()
    for f in path_lst:
        if f not in name_set:
            name_set.add(f)
        else:
            print(f)

    for i in reversed(range(4)):
        if i == 1:
            continue
        path_lst = sorted(path_lst, key=lambda x: int(x.split('/')[-1].split('_')[i]))
    json_lst = load_json(path_lst)

    for f in json_lst:
        name = f.split('/')[-1]
        if 'aif' in name:
            print(name)
    imgs_lst = [None] * len(json_lst)
    for ii in range(len(imgs_lst)):
        path = json_lst[ii]
        imgs_lst[ii] = path.split('.')[0] + '.png'
    return imgs_lst


def load_2dPlus(sample_path, json_dict, kernel=3, stride=1):
    def func(x):
        if x.split('.')[-1] != 'png':
            return 1e+6
        return int(x.split('/')[-1].split('_')[-2])

    name = sample_path.split('/')[-1]
    patient, _, case, slc = name.split('_')
    maskIdx = int(json_dict[name])
    lst = glob.glob(sample_path + '/*')
    lst_ = sorted(lst, key=lambda x: func(x))
    cut = -1
    for i in range(len(lst_) - 1, -1, -1):
        if lst_[i].split('.')[-1] != 'png':
            cut = i
        else:
            break
    lst_ = lst_[:cut]
    totalNum = len(lst_)
    if totalNum < kernel:
        raise ValueError

    sliceNum = (totalNum - kernel) // stride + 1
    Data_Container = [{'Image': list(), 'Mask': str(), 'Seg': False} for _ in range(sliceNum)]
    pivot_ = kernel // 2
    for slid in range(sliceNum):
        pivot = pivot_ + slid * stride
        left = pivot - kernel // 2
        right = pivot + (kernel // 2 if kernel % 2 != 0 else 1)
        Data_Container[slid]['Image'] = lst_[left: right + 1]  # kernel//2: kernel//2 + 1
        assert len(Data_Container[slid]['Image']) == kernel, (len(Data_Container[slid]['Image']), kernel)
        Data_Container[slid]['Mask'] = os.path.join(config.Mask_root,
                                                    "{}_case_{}_{}_AifMask.png".format(patient, case, slc))
        if maskIdx in np.arange(left, right + 1):
            Data_Container[slid]['Seg'] = True
    return Data_Container


def get_Path_2dPlus(kernel=3, stride=1):
    path_lst = glob.glob(config.Img_root + '/*_part/*')
    json_lst = load_json(path_lst)
    json_dict = {}
    for file in json_lst:
        name = file.split('/')[-2]
        idx = file.split('/')[-1].split('_')[-2]
        json_dict[name] = idx

    Output_lst = []
    for path in path_lst:
        Output_lst += load_2dPlus(path, json_dict, kernel, stride)
    return Output_lst


def read_2dPlus(path_dict):
    if not isinstance(path_dict, dict):
        raise ValueError

    img_lst = path_dict["Image"]
    mask_path = path_dict["Mask"]
    Seg = path_dict["Seg"]
    image_shape = (len(img_lst), ) + config.Image_shape
    ans = np.zeros(image_shape)
    mask = cv2.imread(mask_path, 0)
    for i in range(len(img_lst)):
        ans[i,...] = cv2.imread(img_lst[i], 0)
    return ans, mask, Seg


def test():
    data_lst = get_Path_2dPlus()
    idx = np.random.randint(len(data_lst))
    image, mask, seg = read_2dPlus(data_lst[idx])
    image = image.transpose((1, 2, 0))
    print(mask.shape)
    plt.figure()
    plt.imshow(image / 255)
    plt.show()


if __name__ == "__main__":
    test()