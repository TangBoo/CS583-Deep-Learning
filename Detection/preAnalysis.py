from utils import get_ImgPath, read_image_2D, generator_bbox
from sklearn.cluster import KMeans as kmeans
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def KPlus(seq, cluster=2):
    if isinstance(seq, list):
        seq = np.array(seq)
    seq = seq.reshape(-1, 1)
    y_pred = kmeans(n_clusters=cluster, init='k-means++', n_jobs=-1).fit(seq)
    return y_pred


def get_RoiArea(expFactor, Num_features, NumScale=1):
    img_lst = get_ImgPath()
    areas = []
    for i in range(NumScale):
        for path in tqdm(img_lst):
            img, mask = read_image_2D(path)
            bbox = generator_bbox(mask, mode='center', expFactor=expFactor) # [num, cnt_y, cnt_x, h, w]
            for box in bbox:
                height, width = box[2:]
                area = height * width
                areas.append(area)
        expFactor += 1
    predi = KPlus(areas, cluster=Num_features)
    ans = predi.cluster_centers_.squeeze(-1).tolist()
    ans.sort()
    return ans


# if __name__ == "__main__":
#     res = get_RoiArea()
#     print(res)




