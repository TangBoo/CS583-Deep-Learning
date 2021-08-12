import numpy as np
import pandas as pd
from pydicom import dcmread
from tqdm import tqdm
import glob
import os
from sklearn.cluster import KMeans as kmeans
import matplotlib.pyplot as plt
import warnings
import copy

warnings.filterwarnings('ignore')

root = '/data2/database/siim-covid19-detection'
anchors = [[] for _ in range(3)]
isPlot = False


def save_meta(root):
    file_lst = [os.path.join(root, ii) for ii in os.listdir(root)]
    train_bbox_y = pd.read_csv(file_lst[1])
    train_fold = file_lst[-1]
    train_label_y = pd.read_csv(file_lst[0])
    study_id = train_label_y['id']
    train_label_y['id'] = [name.split('_')[0] for name in study_id]
    train_label_y_ = train_label_y.rename(columns={'id': 'StudyInstanceUID'})
    meta_ = pd.merge(train_bbox_y, train_label_y_, on=['StudyInstanceUID'])
    StudyInstanceUID = meta_['StudyInstanceUID']

    img_path = [glob.glob(os.path.join(train_fold, name) + '/*/*.dcm')[0] for name in StudyInstanceUID]
    meta = pd.concat([meta_, pd.DataFrame(img_path, columns=['img_path'])], axis=1)
    img_shape = []
    drop_rows = []
    img_path = meta["img_path"].tolist()
    for idx, path in enumerate(tqdm(img_path)):
        ds = dcmread(path)
        try:
            img_shape.append(ds.pixel_array.shape)
        except:
            img_shape.append(None)
            drop_rows.append(idx)
    Image_shape = pd.DataFrame(img_shape, columns=['height', 'width'])
    meta = pd.concat([meta, Image_shape], axis=1)
    meta = meta.drop(drop_rows)
    meta.to_csv(os.path.join(root, "meta.csv"))
    return meta


def load_meta(root):
    path = os.path.join(root, 'meta.csv')
    assert os.path.exists(path), "path doesn't exist"
    meta = pd.read_csv(path)
    return meta


def save_anchor(root, anchors):
    file = 'anchors.txt'
    path = os.path.join(root, file)
    with open(path, 'w+') as f:
        for scale in anchors:
            for anchor in scale:
                f.write(str(anchor))
            f.write('\n')
    f.close()


def load_anchor(root):
    file = 'anchors.txt'
    path = os.path.join(root, file)
    assert os.path.exists(path), "anchors doesn't exists"
    with open(path,'r') as f:
        for idx, line in enumerate(f.readlines()):
            for item in line[1:-2].split(']['):
                anchors[idx].append([float(ele) for ele in item.split(',')])
    f.close()


def create_anchors(lst):
    seq = np.array(lst).reshape(-1, 2)
    y_pred = kmeans(n_clusters=9, init='k-means++', n_jobs=-1).fit(seq)
    centers = y_pred.cluster_centers_
    centers = sorted(centers, key=lambda x: x[0] * x[1])

    if isPlot:
        w = []
        h = []
        for i, ele in enumerate(lst):
            if i % 2 == 0:
                w.append(ele)
            else:
                h.append(ele)
        plt.figure()
        plt.scatter(w, h, c=y_pred.labels_)
        plt.show()
    lst = []
    for idx, cnt in enumerate(centers):
        lst.append(cnt.tolist())
        if len(lst) == 3:
            anchors[idx // 3] = lst
            lst = []
    save_anchor(root, anchors)


def main():
    try:
        data = load_meta(root)
    except:
        data = save_meta(root)

    bboxes = [None] * len(data)
    boxes_lst = data['boxes']
    heights, widthes = data['height'].tolist(), data['width'].tolist()
    stati_lst = []
    for idx, boxes in enumerate(boxes_lst):
        lst = []
        h, w = heights[idx], widthes[idx]
        try:
            boxes = boxes[2:-2].split('}, {')
        except:
            boxes = []
        for box in boxes:
            lst.append([float(ele.split(':')[-1]) / float(w if i % 2 == 0 else h) for i, ele in enumerate(box.split(','))])
            stati_lst.extend(lst[-1][2:])
        bboxes[idx] = lst
    data['boxes'] = bboxes
    print("load anchors....")
    try:
        load_anchor(root)
    except:
        create_anchors(stati_lst)
    return data


def get_table():
    table = main()
    try:
        table = table.drop(['Unnamed: 0'], axis=1)
    except:
        pass
    index = table[table['Negative for Pneumonia'] == 1].index.tolist()
    table = table.drop(index)
    table.reset_index(drop=True, inplace=True)
    return table


meta_table = get_table()

