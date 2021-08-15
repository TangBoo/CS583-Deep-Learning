import glob
import random
import Config as config
import json


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
