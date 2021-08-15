import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import glob
from tqdm import tqdm
import os

root = r"/home/aiteam_share/database/ISLES2018_manual_aif"
mask_path = r"/home/aiteam_share/database/ISLES2018_manual_aif/aif_mask"

imgs_lst = []
json_lst_ = []


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


def create_aif_mask(json_lst, mask_path):
    del_lst = []
    name_set = {}
    count = 0
    for sample in tqdm(json_lst):
        file = open(sample, encoding='utf-8')
        setting = json.load(file)
        shapes = setting['shapes']
        contours = []
        mask_aif = np.zeros((256, 256))
        for idx, item in enumerate(shapes):
            if item['label'] == 'aif':
                contour = np.expand_dims(np.array(item["points"]).astype('int'), 0)
                contours.append(contour)

        if len(contours) == 0:
            print(sample)
            continue

        for contour in contours:
            cv2.drawContours(mask_aif, contour, 0, (255, 255, 255), cv2.FILLED)

        name = sample.split('/')[-1].split('.')[0]
        patient, _, case, slc = name.split('_')[0: 4]
        count += 1
        name_aif = "{}_case_{}_{}_AifMask.png".format(patient, case, slc)
        if name_aif not in name_set:
            name_set[name_aif] = name
        else:
            print(name_aif, name_set[name_aif], name)

        if os.path.exists(mask_path) == False:
            os.mkdir(mask_path)

        cv2.imwrite(r"/{}/{}".format(mask_path, name_aif), mask_aif)
    print("The aif mask has created")


def main():
    path_lst = glob.glob('/home/aiteam_share/database/ISLES2018_manual_aif/*_part/*')
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

    create_aif_mask(json_lst, mask_path)
    aif_lst = glob.glob(mask_path + '/*.png')
    print(len(aif_lst) == len(path_lst))

    imgs_lst = [None] * len(json_lst)
    for ii in range(len(imgs_lst)):
        path = json_lst[ii]
        imgs_lst[ii] = path.split('.')[0] + '.png'
    return imgs_lst



if __name__=="__main__":
    imgs_lst = main()
    print(len(imgs_lst))
