import torch as t
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
import Config as config
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import visdom
import time


class TensorBoard(object):
    def __init__(self):
        self.writer = SummaryWriter(config.TBPath)

    def add_scalar(self, name, value, x_axis):
        self.writer.add_scalar(tag=name, scalar_value=value, global_step=x_axis)

    def add_image(self, name, img):
        self.writer.add_image(tag=name, img_tensor=img, dataformats='CHW')

    def add_images(self, name, imgs):
        grid = tv.utils.make_grid(imgs)
        self.writer.add_image(tag=name, img_tensor=grid)

    # def add_hist(self, name, tensor):


def vis_image(img, ax=None):
    if t.is_tensor(img):
        img = img.numpy()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype('np.uint8'))
    return ax


class Visualizer(object):
    def __init__(self, env="Detection", port=8098, **kwargs):
        self.vis = visdom.Visdom(env=env, port=port, **kwargs)
        self.index = {}
        self.train_log_text = ''
        self.val_log_text = ''
        self.backward_monitor = ''

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def show_anchor(self, name, imgs):
        # img: [b, c, h, w]
        for i in range(imgs.shape[0]):
            img = imgs[i]
            # print(img.shape)
            self.vis.images(img * 255, win=name, opts=dict(title=name))

    def imgs(self, name, img_, **kwargs):
        # [batch, channel, d, h, w]
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
            idx0 = np.random.randint(0, size[0], size=(1,))
        if size[1] > 1:
            idx1 = np.random.randint(0, size[1], size=(1,))
        imgs_ = imgs_.cpu().numpy()[idx0, idx1, ...].swapaxes(1, 0)
        self.vis.video(tensor=imgs_, win=name)

    def img(self, name, img_, **kwargs):
        size = img_.shape
        idx0, idx1, idx2 = 0, 0, 0
        if size[0] > 1:
            idx0 = np.random.randint(0, size[0], size=(1,))
        if size[1] > 1:
            idx1 = np.random.randint(0, size[1], size=(8,))
        if size[2] > 1 and size == 5:
            idx2 = np.random.randint(0, size[2], size=(8,))
        if size == 5:
            self.vis.image(img_.cpu().numpy()[idx0, idx1, idx2, ...], win=name, opts=dict(title=name), **kwargs)
        else:
            self.vis.image(img_.cpu().numpy()[idx0, idx1, ...], win=name, opts=dict(title=name), **kwargs)

    def show_boxes(self, name, img_, box):
        ax = vis_image(img_)
        if len(box) == 0:
            return ax

        for i, bb in enumerate(box):
            xy = (bb[1], bb[0])
            height = bb[2] - bb[0]
            width = bb[3] - bb[1]
            ax.add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor='red', linewidth=1))
        self.img(name, ax)

    def log(self, info, isTrain=True):
        if isTrain:
            self.train_log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
            self.vis.text(self.train_log_text, win="Train log")
        else:
            self.val_log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
            self.vis.text(self.val_log_text, win="Validation Log")

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=name
                      , opts=dict(title=name), update=None if x == 0 else 'append', **kwargs)
        self.index[name] = x + 1

    def img_many(self, d):
        for k, v in d.items():
            self.imgs(k + ': ' + str(v.shape), v * 256)

    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)

    def __getattr__(self, name):
        return getattr(self.vis, name)

    def show_hist(self, name, tensor):
        if isinstance(tensor, list):
            for i in range(len(tensor)):
                if tensor[i] is None:
                    continue
                self.vis.histogram(X=tensor[i].flatten(), win=name + "-feature_{}".format(i), opts=dict(numbins=30, title=name + "-feature_{}".format(i)))
        elif tensor is not None:
            x = tensor.flatten()
            self.vis.histogram(X=x, win=name, opts=dict(numbins=30, title=name))


def test():
    tb = TensorBoard()
    for i in range(100):
        tb.add_scalar('test:', t.rand(1, ), i)


if __name__ == "__main__":
    test()
