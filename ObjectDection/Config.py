import numpy as np
import torch.cuda
from preAnalysis import get_RoiArea

Time_axis = 64
Mask_shape = (1, 256, 256)  # (D, H, W)
Train_percent = 0.9
Val_percent = 0.1
# input_shape = (1, 1) + image_shape #(N, C, D, H, W)
Device = "cuda" if torch.cuda.is_available() else "cpu"
Img_root = r"/home/aiteam_share/database/ISLES2018_manual_aif"
Mask_root = r"/home/aiteam_share/database/ISLES2018_manual_aif/aif_mask"
Num_class = 1
checkpoint = r"/home/aiteam_share/database/ISLES2018_manual_aif/train_3DUnet/checkpoint"
# checkpoint = None

Load_model_path = None
TrainBatchSize = 16
ValBatchSize = 2
Num_workers = 4
# Debug_file = r"/home/aiteam_share/database/ISLES2018_manual_aif/train_3DUnet/Debug"
Debug_file = None
lr = 0.001
lr_decay = 0.95
Weight_decay = 1e-4
Env = "3DUnet"
Max_epoch = 100
Print_freq = 5
Loss_alpha = 0.2
Loss_beta = 0.8
Optim = "adam"
Num_groups = 16

eps = 1e-6
Resize_mode = 'trilinear'
vis_port = 8098
Patient = 10
Debug_Patient = 10
Msk_mode = '3D'

# -------------Object Detection-------------
# Attention: Using receptive field as area size will be good to small bounding box, expBox = 1
# Beside, using predefine area size getting from KMeans will be good to big bounding box, expBox = 2
Image_shape = (256, 256)
Base = 32
HW_ratio = [0.5, 1.0, 2.0]
Area_ratios = [2 ** 0, 2 ** (1 / 2), 2 ** (2 / 3)]
PosThreshold = 0.5
NegThreshold = 0.3
ExpBox = 2
Output_features = 4
LowestNum_Anc = 9

# Receptive Field : k0 = 4, k0 + log(sqrt(w * h) / 256)
# expBox 1: [55.331926863572555, 162.49789029535873, 445.9090909090909]
# expBox 2: [221.32770745, 649.99156118, 1783.63636364]
# expBox3:[497.98734177215255, 1462.4810126582279, 4013.181818181819]
# Areas = [497.98734177215255, 1462.4810126582279, 4013.181818181819]

Depth = 6 # No more than 5, too big receptive field to obtain object
Areas = get_RoiArea(ExpBox, Num_features=Output_features)
# Areas = None
if Areas is not None:
    OutputIndex = np.arange(Depth)[-Output_features - 1: - 1]
else:
    OutputIndex = np.arange(Depth)[-Output_features:]






