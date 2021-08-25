import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as kmeans
from utils import find_secMin


def segmentation(tmaxes):
    """
    input:
        tmaxes : (256, 256)
    output:
        scale tmaxes for mapping to [0, 255]

    """
    min_val = tmaxes.min()
    max_val = tmaxes.max()
    tmaxes = (tmaxes - min_val) / (max_val - min_val)
    return tmaxes * 255


def postProcess(image, mask, ptype='irf', n=4):
    """
    Input: Image scale in [0, 255]
    Ouput:
           Binary Image
    Discribe:
        1. Get labels from kmeans func that map tmax to [0, 4] or [0, 3].
        2. Get second minimum value and minimum value from labels.
        3. Let all value is not equal to min and sec_min be zero
        #---irf---
        4. Morphological operation.
        5. Let color in output image be uniform.
    """

    ncluster = n

    predi = kmeans(image[mask != 0], cluster=ncluster)

    labels = predi.labels_
    label_img = labels.copy().reshape((256, 256))
    num_label = [len(labels[labels == i]) for i in set(labels)]
    color = np.argmin(num_label)
    secColor = find_secMin(num_label)
    lesion = np.zeros_like(image)
    for idx, v in enumerate(labels):
        ele = 0
        if labels[idx] == color:
            ele = 2
        elif labels[idx] == secColor:
            ele = 1
        lesion[int(idx // 256), int(idx % 256)] = ele

    kernel = np.ones((3, 3), 'uint8')
    lesion_dialation = cv2.dilate(lesion, kernel, iterations=1)
    lesion_erode = cv2.erode(lesion_dialation, kernel, iterations=4)
    lesion_dilation = cv2.dilate(lesion_erode, kernel, iterations=3)

    ans_ = cv2.bitwise_and(image, lesion_dilation * 255)
    ans_ = segmentation(ans_)
    ans = ans_.copy()
    ans = cv2.drawContours(lesion * 123, mask, 0, (255, 0, 0), 1)

    plt.figure(figsize=(15, 7))
    plt.subplot(2, 4, 1)
    plt.imshow(lesion)
    plt.title('Lesion')
    plt.subplot(2, 4, 2)
    plt.imshow(ans)
    plt.title('bitwise_and with Image ')
    #         plt.subplot(2, 4, 3)
    #         plt.imshow(lesion_dilation)
    plt.subplot(2, 4, 3)
    plt.imshow(label_img)
    plt.title('Label Image')
    plt.show()

    return ans_, lesion_dilation
