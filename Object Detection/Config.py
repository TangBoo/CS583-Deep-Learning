import numpy as np
import torch.cuda
from preAnalysis import get_RoiArea


Time_axis = 64
Mask_shape = (1, 256, 256)  # (D, H, W)
Train_percent = 0.9
Val_percent = 0.1
Device = "cuda" if torch.cuda.is_available() else "cpu"
Img_root = r"/home/aiteam_share/database/ISLES2018_manual_aif"
Mask_root = r"/home/aiteam_share/database/ISLES2018_manual_aif/aif_mask"
Num_class = 1
# checkpoint = r"/home/aiteam_share/database/ISLES2018_manual_aif/train_3DUnet/checkpoint"
checkpoint = None

Load_model_path = None

Num_workers = 4
# Debug_file = r"/home/aiteam_share/database/ISLES2018_manual_aif/train_3DUnet/Debug"
Debug_file = None
lr = 0.001
lr_decay = 0.95
Weight_decay = 1e-4

Max_epoch = 100
Print_freq = 10
Loss_alpha = 0.2
Loss_beta = 0.8
Optim = "adam"
Num_groups = 16

eps = 1e-6
Resize_mode = 'trilinear'
vis_port = 8098
Lr_Patient = 3
Debug_Patient = 1000

# -------------Object Detection-------------
# Attention: Using receptive field as area size will be good to small bounding box, expBox = 1
# Beside, using predefine area size getting from KMeans will be good to big bounding box, expBox = 2
Env = 'Detaction'
ResNetStructure = {'ResNet50': [3, 4, 6, 3],
                   'ResNet101': [3,4, 23, 3],
                   'ResNet152': [3, 8, 36, 3]
                   }
TrainBatchSize = 2
ValBatchSize = 2
Image_shape = (256, 256)
InputChannel = 1
SegChannel = 1
Base = 64
HW_ratio = [0.5, 1.0, 2.0]
Area_ratios = [2 ** 0, 2 ** (1 / 2), 2 ** (2 / 3)]
PosThreshold = 0.5
NegThreshold = 0.3
ExpBox = 2
Data_type = '2DPlus' #'2D'
LowestNum_Anc = 9
# Receptive Field : k0 = 4, k0 + log(sqrt(w * h) / 256)
Depth = 6 # No more than 5, too big receptive field to obtain object
Output_features = 4
BoxScale = 1
if Output_features >= Depth - 1:
    raise ValueError("Depth is not enough for output {} features map".format(Output_features))
Areas = get_RoiArea(ExpBox, Num_features=Output_features, NumScale=BoxScale)
OutputIndex = np.arange(Depth)[-Output_features:]
RecepField = [1] + [2 ** i for i in range(0, Depth)] # [1] is the scale of output layer
if len(OutputIndex) != Output_features:
    raise ValueError






