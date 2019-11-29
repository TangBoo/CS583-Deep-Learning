import time
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor as threadPool
import os
import pydicom
import warnings
from keras import backend as k
import matplotlib.pylab as plt
import random
warnings.filterwarnings('ignore')


ia.seed(1)

class DataGenerator:
    def __init__(self,data_path_root,images_path,csv_path,num_thread=None,is_train=False):
        index_array,train_labels = self.read_csv(csv_path)
        self.data_path=data_path_root
        self.image_path=images_path
        self.csv_path=csv_path
        self.num_thread=num_thread
        self.is_train=is_train
        self.train_label=train_labels
        self.index_array=index_array
        self.seq = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Crop(percent=(0, 0.1)),
                iaa.OneOf([iaa.GaussianBlur((0, 0.5)),iaa.AverageBlur(k=(2, 7)),iaa.MedianBlur(k=(3, 11))]),
                iaa.ContrastNormalization((0.75, 1.5)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-25, 25),
                    shear=(-8, 8))], random_order=True)
    def load_image(self,temp_path):
        im = np.array(Image.open(self.image_path+'/'+temp_path+'.png'))
        return im
    def get_img(self,img_paths, img_size):
        p = threadPool()
        X = np.zeros((len(img_paths), img_size, img_size), dtype=np.uint8)
        i = 0
        for future in p.map(self.load_image, img_paths):
            img = np.resize(future, (img_size, img_size))
            X[i, :, :] = img
            i += 1
        p.shutdown(wait=False)
        return X
    def random_pick(self,some_list):
        probabilities = [0.3,0.2,0.2,0.2,0.1]
        x = random.uniform(0,1)
        cumulative_probability = 0.0
        for item, item_probability in zip(some_list, probabilities):
            cumulative_probability += item_probability
            if x < cumulative_probability:break
        return item
    def creat_batch_data(self,x_path,batch_size):
        dia_dic={}
        np.random.shuffle(x_path)
        change_index=self.train_label.loc[x_path]
        change_index.index=range(0,len(change_index.index))        
        keep_prob=change_index.iloc[:,0].map({0:0.18,1:0.9})
        keep=(keep_prob>np.random.rand(len(keep_prob)))
        indices=np.arange(len(x_path))[keep]
        csv=change_index.iloc[indices]
        neg_group=csv['Label'][csv['Label']['any']==0].index
        any_group=csv['Label'][csv['Label']['any']==1].index
        ratio=len(any_group)/len(csv.index)
        posi_num=int(ratio*batch_size)
        dia_dic[1]=csv['Label'][csv['Label']['epidural']>=1].index
        dia_dic[2]=csv['Label'][csv['Label']['intraparenchymal']>=1].index
        dia_dic[3]=csv['Label'][csv['Label']['intraventricular']>=1].index
        dia_dic[4]=csv['Label'][csv['Label']['subarachnoid']>=1].index
        dia_dic[5]=csv['Label'][csv['Label']['subdural']>=1].index
        groups=[]
        for i in range(posi_num):
            item=self.random_pick(dia_dic.keys())
            groups.append(random.choice(dia_dic[item]))
        groups.extend(np.random.choice(neg_group,size=batch_size-len(groups)))
        random.shuffle(groups)
        return groups

    def get_X_batch(self, X_path, batch_size, img_size, is_train=True):
        try:
            if len(X_path) % batch_size != 0:
                raise Exception("batchSize not match the size of data!")
        except Exception as err:
            print(err)
        else:
            while 1:
                indice=self.creat_batch_data(X_path,batch_size)
                X = self.get_img(X_path[indice], img_size)
                Y = self.train_label.loc[X_path[indice]].values
                if is_train:
                    X = (self.seq.augment_images(X))
                    yield np.resize(X / 155, (len(X), img_size, img_size, 1)).astype('float16'), Y.astype('float16')
                else:
                    X = np.resize(X / 155, (batch_size, img_size, img_size, 1))
                    yield X.astype('float16'), Y.astype('float16')

    def get_test_batch(self,X_path,batch_size,img_size):
        try:
            if len(X_path) %batch_size != 0:
                raise Exception("batchSize not match the size of data!")
        except Exception as err:
            print(err)
        else:
            while 1:
                for i in range(0, len(X_path), batch_size):
                    X =(self.get_img(X_path[i:i + batch_size], img_size))
                    yield np.resize(X/155,(len(X),img_size,img_size,1)).astype('float16')
    def read_csv(self, filename):
        df = pd.read_csv(filename)
        df["ImageID"] = df["ID"].str.slice(stop=12)
        df["Diagnosis"] = df["ID"].str.slice(start=13)
        duplicates_to_remove = [
            56346, 56347, 56348, 56349,
            56350, 56351, 1171830, 1171831,
            1171832, 1171833, 1171834, 1171835,
            3705312, 3705313, 3705314, 3705315,
            3705316, 3705317, 3842478, 3842479,
            3842480, 3842481, 3842482, 3842483
        ]
        df = df.drop(index=duplicates_to_remove)
        df = df.reset_index(drop=True)
        df = df.loc[:, ["Label", "Diagnosis", "ImageID"]]
        df = df.set_index(['ImageID', 'Diagnosis']).unstack(level=-1)
        index_array = df.index.values.tolist()
        return index_array, df


# gener=DataGenerator(data_path,images_path=f'{data_path}/stage_2_train_images',csv_path=f'{data_path}/stage_2_train.csv',num_thread=None,is_train=True)



