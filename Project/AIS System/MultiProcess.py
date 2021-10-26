import torch as t
from datetime import datetime
import numpy as np
from multiprocessing import Pool
from joblib import Parallel, delayed, parallel_backend
import ray
import warnings
warnings.filterwarnings('ignore')

# ray.init(num_cpus=10)


def resample_index(*args):
    num, sample_num = args
    # print('total num: ', num, 'resample num: ', sample_num)
    if num < sample_num:
        num = sample_num
    return t.randperm(int(num))[:int(sample_num)]


def subsection(mat, group_num=10):
    time, h, w, _ = mat.shape
    minv = t.min(mat, dim=0)[0].int()
    maxv = t.max(mat, dim=0)[0].int() + 1

    step = (((maxv - minv) / group_num) * t.arange(group_num).cuda()) + minv
    low = step[..., ::2]
    high = step[..., 1::2]
    boaders = t.cat((low.unsqueeze(-1), high.unsqueeze(-1)), dim=-1).permute(-1, -2, 0, 1)
    eps_mat = t.full((1, h, w), 1e-6).cuda()
    loc_mat = t.zeros((group_num // 2, time, h, w))
    min_loc = t.zeros((group_num // 2, h, w))
    index_mat = t.zeros_like(loc_mat) # [group_num // 2, time, h, w]
    if mat.shape[-1] == 1:
        mat = mat.squeeze(-1)

    for i in range(group_num // 2):
        condi_left = boaders[0:1, i]
        condi_right = boaders[1:, i]
        loc = t.tensor((mat >= condi_left) * (mat < t.max(condi_right, eps_mat)), dtype=t.float)
        num_sort, index_sort = t.sort(loc, dim=0, descending=True)
        min_loc[i, ...] = t.argmin(num_sort, dim=0)
        index_mat[i] = index_sort
        loc_mat[i] = loc

    loc_sum = loc_mat.sum(1)
    sample_num = loc_sum[loc_sum != 0].min().int().item()
    # loc_mat = loc_mat.numpy()
    resample_mat = t.zeros((group_num // 2, sample_num, h, w))
    for i in range(group_num // 2):
        temp_min = min_loc[i]
        # with parallel_backend('multiprocessing', n_jobs=10):
        #     res = Parallel()(
        #         delayed(resample_index)(*[temp_min[i // 256, i % 256].item(), sample_num]) for i in range(256 * 256))
        # resample_mat[i] = t.tensor(res).reshape((int(sample_num), h, w))
        res = []
        for j in range(256 * 256):
            num = temp_min[j // 256, j % 256]
            res.append(resample_index(num, sample_num))
        # -------------- Take From Index Sort ---------------------
        resample_mat[i] = t.tensor(res).reshape((int(sample_num), h, w))
    print(resample_mat.shape, index_mat.shape)
    sample_loc = t.gather(index_mat, dim=1, index=resample_mat.long()).reshape(-1, 256, 256)
    print(sample_loc.shape)
    sample_loc, _ = t.sort(sample_loc, dim=0, descending=False)
    print(sample_loc.shape, mat.shape)
    print(sample_loc[..., 0, 0])
    # output = t.gather(mat, dim=0, )

    return mat.gather(dim=0, index=sample_loc.long().cuda())
        # res [256 * 256, sample_num]
        # a = t.tensor(np.array(res).reshape((sample_num, h, w)))
        # resample_mat[i] = a
        # print(i, resample_mat[i].shape)


# def random_sample(loc_index):
#     # input: [group, time]
#     # return: [sample_num, ]
#     res = None
#     for i in range(group_num // 2):
#         whe = np.argwhere(loc_index[i] == True).squeeze(-1)
#         if len(whe) == 0:
#             loc = np.random.choice(len(loc_index[i]), sample_num, replace=True)
#         elif len(whe) < sample_num:
#             loc = np.random.choice(whe, sample_num, replace=True)
#         else:
#             loc = np.random.choice(whe, sample_num, replace=False)
#         if i == 0:
#             res = loc
#         else:
#             res = np.hstack((res, loc))
#         res.sort()
#     return res


if __name__ == "__main__":
    time = 1000
    w, h = 256, 256
    mat = t.randn(time, h, w, 1).cuda()
    start = datetime.now()
    signal = subsection(mat, group_num=10)
    end = datetime.now()
    print((end - start).seconds)
    print(signal.shape)
    # loc_mat = loc_mat.reshape(20, 1000, -1)
    # start = datetime.now()
    # with parallel_backend('multiprocessing', n_jobs=10):
    #     result = Parallel()(delayed(random_sample)(loc_mat[..., i].copy()) for i in range(256 * 256))
    # end = datetime.now()
    # print((end - start).seconds)

    # ----------ray-------------
    # start = datetime.now()
    # result = [random_sample.remote(loc_mat[..., i].copy()) for i in range(256 * 256)]
    # end = datetime.now()
    # print((end - start).seconds)

    # loc_list = [loc_mat[..., i] for i in range(256 * 256)]
