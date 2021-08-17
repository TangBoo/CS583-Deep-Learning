import torch
import random
from torchvision import transforms as T
from torchvision.transforms import functional as tf


def rotate(image, mask=None, angle=None):
    if angle is None:
        angle = T.RandomRotation.get_params([-180, 180])
    image = tf.rotate(image, angle)
    if mask is None:
        return image
    mask = tf.rotate(mask, angle)
    return image, mask


def flip(image,mask):
    if random.random() > 0.5:
        image = tf.hflip(image)
        mask = tf.hflip(mask)
    if random.random() < 0.5:
        image = tf.vflip(image)
        mask = tf.vflip(mask)
    return image, mask


def myTransform(img, mask):
    img = torch.tensor(img)
    mask = torch.tensor(mask)
    if random.random() > 0.2:
        img, mask = rotate(img, mask)

    if random.random() > 0.5:
        img, mask = flip(img, mask)

    if random.random() > 0.5:
        img = tf.gaussian_blur(img, [3, 5], sigma=[0.1, 2.0])

    return img, mask


