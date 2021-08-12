import numpy as np


if __name__ == "__main__":
    shape = (6, 6)

    depth =4
    h, w = shape
    recepField = [2 ** i for i in range(1, depth)]
    scales = [h // j for j in reversed(recepField)]

    cor_mat = np.zeros((((len(scales), ) + shape + (4,))))


    for idx, scale in enumerate(scales):

        cor_mat[idx, :] = [0.5, 0.5, h/(scale+0.5), scale/(w+0.5)]

    print(cor_mat)

