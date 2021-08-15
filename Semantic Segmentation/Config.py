import torch.cuda


Time_axis = 64
Image_shape = (64, 256, 256) #(D, H, W)
Mask_shape = (1, 256, 256) #(D, H, W)
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
Base = 32
eps = 1e-6
Resize_mode = 'trilinear'
vis_port = 8098
Patient = 3
Debug_Patient = 10
Msk_mode = '3D'



