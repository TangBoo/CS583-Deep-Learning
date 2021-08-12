import glob

import Config as config
import visdom
import time
import numpy as np
from torch import nn
import torch
import random
features = {}


class Visualizer(object):
    def __init__(self, env=config.Env, port=config.vis_port, **kwargs):
        self.vis = visdom.Visdom(env=env, port=port, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def imgs(self, name, img_, **kwargs):
        #[batch, channel, d, h, w]
        size = img_.shape
        if len(size) == 5:
            idx0 = np.random.randint(0, size[0], size=(1,))
            idx1 = np.random.randint(0, size[1], size=(8,))
            idx2 = np.random.randint(0, size[2], size=(8,))
            img = np.expand_dims(img_.cpu().numpy()[idx0, idx1, idx2, ...], axis=1)
            self.vis.images(img, win=name, nrow=4, opts=dict(title=name))
        else:
            idx0 = np.random.randint(0, size[0], size=(1,))
            idx1 = np.random.randint(0, size[1], size=(8,))
            img = np.expand_dims(img_.cpu().numpy()[idx0, idx1, ...], axis=1)
            self.vis.images(img, win=name, nrow=4, opts=dict(title=name))

    def videos(self, name, imgs_, **kwargs):
        size = imgs_.shape
        idx0, idx1 = 0, 0
        if size[0] > 1:
            idx0 = np.random.randint(0, size[0], size=(1, ))
        if size[1] > 1:
            idx1 = np.random.randint(0, size[1], size=(1, ))
        imgs_ = imgs_.cpu().numpy()[idx0, idx1, ...].swapaxes(1, 0)
        self.vis.video(tensor=imgs_, win=name)

    def img(self, name, img_, **kwargs):
        size = img_.shape
        idx0, idx1, idx2 = 0, 0, 0
        if size[0] > 1:
            idx0 = np.random.randint(0, size[0], size=(1, ))
        if size[1] > 1:
            idx1 = np.random.randint(0, size[1], size=(8, ))
        if size[2] > 1 and size == 5:
            idx2 = np.random.randint(0,size[2], size=(8, ))
        if size == 5:
            self.vis.image(img_.cpu().numpy()[idx0, idx1, idx2, ...], win=name, opts=dict(title=name), **kwargs)
        else:
            self.vis.image(img_.cpu().numpy()[idx0, idx1,...], win=name, opts=dict(title=name), **kwargs)


    def log(self, info, win='log_text'):
        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=name
                      , opts=dict(title=name), update=None if x == 0 else 'append', **kwargs)
        self.index[name] = x + 1

    def img_many(self, d):
        for k, v in d.items():
            self.imgs(k+': '+str(v.shape), v*256)

    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)

    def __getattr__(self, name):
        return getattr(self.vis, name)


def scale(x):
    return (x-x.min())/(x.max() - x.min())


def get_output(name):
    def hook(model, input, output):
        features[name] = output.data
    return hook


def get_pth(model, checkpoint):
    def func(path):
        data_time = path.split('/')[-1].split('_')[1:]
        key = "{}_{}".format(data_time[-2], data_time[-1])
        return key
    if isinstance(model, nn.DataParallel):
        model = model.module
    model_name = model.__str__().split("\n")[0][:-1]
    pth_lst = glob.glob(checkpoint + '/*.pth')
    coll_pth = []
    for item in pth_lst:
        sub_name = item.split('/')[-1].split('_')[0][1:-1].split(' ')[-1][1:-1].split('.')[-1]
        if sub_name == model_name:
            coll_pth.append(item)
    coll_pth.sort(key=func)
    return coll_pth[-1]


def introIndex(target, ratio=3):
    neg_idx = torch.where(target == 0)
    pos_idx = torch.where(target == 1)
    num_pos = len(neg_idx[0])
    num_neg = len(pos_idx[0])
    random_idx = random.sample(range(num_neg), min(num_pos * ratio, num_neg))
    index = tuple(tp[random_idx] for tp in neg_idx)
    idx_mat = torch.zeros_like(target)
    idx_mat[pos_idx] = 1
    idx_mat[index] = 1
    return idx_mat


if __name__ == "__main__":
    pass


