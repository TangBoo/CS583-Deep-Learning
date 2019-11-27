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
warnings.filterwarnings('ignore')



# data_path= "C:/Users/xiang/Desktop/rsna_stage1_png_224"
# pro_index=f'{data_path}/stage_1_train_images'
# image_index=os.listdir(f'{data_path}/stage_1_train_images')
# image_index=pd.DataFrame(image_index)
# image_index=image_index[0].str.slice(stop=12).values
# csv_path=f'{data_path}/stage_1_train.csv'


ia.seed(1)
def window_img(dcm, width=None, level=None, norm=True):
    pixels = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    # Pad non-square images
    if pixels.shape[0] != pixels.shape[1]:
        (a, b) = pixels.shape
        if a > b:
            padding = ((0, 0), ((a - b) // 2, (a - b) // 2))
        else:
            padding = (((b - a) // 2, (b - a) // 2), (0, 0))
        pixels = np.pad(pixels, padding, mode='constant', constant_values=0)

    if not width:
        width = dcm.WindowWidth
        if type(width) != pydicom.valuerep.DSfloat:
            width = width[0]
    if not level:
        level = dcm.WindowCenter
        if type(level) != pydicom.valuerep.DSfloat:
            level = level[0]
    lower = level - (width / 2)
    upper = level + (width / 2)
    img = np.clip(pixels, lower, upper)
    if norm:
        return (img - lower) / (upper - lower)
    else:
        return img
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
        self.seq=iaa.Sometimes(p=0.5,then_list=[iaa.Fliplr(1),iaa.OneOf([iaa.GaussianBlur((0, 0.5)),iaa.AverageBlur(k=(2, 7)),iaa.MedianBlur(k=(3, 11)),iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))])],random_state=True)
#         self.seq=iaa.Sequential([
# #                                         iaa.Fliplr(0.8),
#                                         iaa.Flipud(1),
#                                         iaa.OneOf([
#                                             iaa.GaussianBlur((0, 0.5)),
#                                             iaa.AverageBlur(k=(2, 7)),
#                                             iaa.MedianBlur(k=(3, 11)),
#                                             iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
#                                         ]),
# #                                         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*155), per_channel=0.5),
#                                     ], random_order=True)

    def load_image(self,temp_path):
        im = np.array(Image.open(self.image_path+'/'+temp_path+'.png'))
        # im_aug = seq.augment_images(images)
        return im
    # def load_image(self,temp_path):
    #     dcm=pydicom.dcmread(self.image_path+'/'+temp_path+'.dcm')
    #     im= window_img(dcm)
    #     # im_aug = seq.augment_images(images)
    #     return im

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

    # def get_img(self,img_paths, img_size):
    #     X = np.zeros((len(img_paths),img_size,img_size),dtype=np.uint8)
    #     i = 0
    #     for img_path in img_paths:
    #         img =np.array(self.load_image(img_path))
    #         # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         img = np.resize(img,(img_size,img_size))
    #         # img=seq.augment_images(images)
    #         X[i,:,:] = img
    #         i += 1
    #     return X

    # def get_X_batch(self, X_path, batch_size, img_size,is_train=True):
    #     try:
    #         if len(X_path)%batch_size!=0:
    #             raise Exception("batchSize not match the size of data!")
    #     except Exception as err:
    #             print(err)
    #     else:
    #         while 1:
    #             np.random.shuffle(X_path)
    #             for i in range(0, len(X_path), batch_size):
    #                 X = self.get_img(X_path[i:i + batch_size], img_size)
    #                 Y = self.train_label.loc[X_path[i:i + batch_size]].values
    #                 if is_train:
    #                     X=(self.seq.augment_images(X))
    #                     yield np.resize(X/155,(len(X),img_size,img_size,1)).astype('float16'), Y.astype('float16')
    #                 else:
    #                     X=np.resize(X/155,(batch_size,img_size,img_size,1))
    #                     yield X.astype('float16'),Y.astype('float16')
    def creat_batch_data(self,x_path,batch_size):
        np.random.shuffle(x_path)
        keep_prob=self.train_label.loc[x_path].iloc[:,0].map({0:0.18,1:0.9})
        keep=(keep_prob>np.random.rand(len(keep_prob)))
        indices=np.arange(len(x_path))[keep]
        # np.random.shuffle(indices)
        rand_int=np.random.randint(len(indices),size=batch_size)
        return indices[rand_int]

    def get_X_batch(self, X_path, batch_size, img_size,is_train=True):
            while 1:
                indice=self.creat_batch_data(X_path,batch_size)
                X = self.get_img(X_path[indice], img_size)
                Y = self.train_label.loc[X_path[indice]].values
                if is_train:
                    X = (self.seq.augment_images(X))
                    yield np.resize(X / 255, (len(X), img_size, img_size, 1)).astype('float16'), Y.astype('float16')
                else:
                    X = np.resize(X / 255, (batch_size, img_size, img_size, 1))
                    yield X.astype('float16'), Y.astype('float16')
#                 X = np.resize(X / 255, (batch_size, img_size, img_size, 1))
#                 yield X.astype('float16'), Y.astype('float16')

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
                    yield np.resize(X/255,(len(X),img_size,img_size,1)).astype('float16')

#     def read_csv(self,csv_path):
#         train_df = pd.read_csv(csv_path).drop_duplicates()
#         train_df['ImageID'] = train_df['ID'].str.slice(stop=12)
#         train_df['Diagnosis'] = train_df['ID'].str.slice(start=13)
#         train_labels = train_df.pivot(index="ImageID", columns="Diagnosis", values="Label")
#         train_labels.drop(['ID_6431af929'], inplace=True)
#         index_array = train_labels.index.values.tolist()
#         return index_array,train_labels
    def read_csv(self,filename):
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
            if 'ID_6431af929' in df.index:
                df.drop(['ID_6431af929'], inplace=True)
            index_array = df.index.values.tolist()
            return index_array,df
# #
# gener=DataGenerator(data_path,images_path=f'{data_path}/stage_1_train_images',csv_path=f'{data_path}/stage_1_train.csv',num_thread=None,is_train=True)
# start=time.time()
# sample=gener.get_X_batch(image_index[0:600000],64,224,True)
# x,y= next(sample)
# print(np.max(x))
# print(np.mean(x))
# print('time: ',time.time()-start)


