import os
import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
import skimage
from Realign import getMaxArea
from skimage import filters, morphology, feature

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
path = "/data/aiteam_ctp/database/AIS_210713/0713_dst_png"


def getCaselst(root):
    patient_lst = glob(path + '/*/')
    case_lst = []
    for i in range(len(patient_lst)):
        temp_lst = glob(patient_lst[i] + '/*/')
        for j in range(len(temp_lst)):
            if temp_lst[j].endswith('png/'):
                continue
            case_lst.append(temp_lst[j])
    return case_lst


def load4DImage(idx, case_lst):
    slc_lst = glob(case_lst[idx] + '/*/')
    slc_lst = sorted(slc_lst, key=lambda x: x.split('/')[-2].split('_')[0])
    # [d, time, h, w]
    anat_imgs = []
    for i in range(len(slc_lst)):
        time_imgs = []
        time_lst = glob(slc_lst[i] + '/*')
        time_lst = sorted(time_lst, key=lambda x: x.split('/')[-1].split('.')[0].split('img')[-1])
        for j in range(len(time_lst)):
            time_imgs.append(cv2.imread(time_lst[j]))
        anat_imgs.append(time_imgs)
    return np.array(anat_imgs).swapaxes(0, 1)


# [time, d, h, w]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        mid_channels = int(in_channels / 2)

        self.decoderblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.decoderblock(x)
        return x


class FinalBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(FinalBlock, self).__init__()
        self.finalblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.finalblock(x)
        return x


class ResUNet50(nn.Module):

    def __init__(self, num_class, pretrained=False):
        super().__init__()

        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoder part.
        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            # nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            self.resnet.bn1,
            nn.ReLU(inplace=True),
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

        # Decoder part.
        self.decoder5 = DecoderBlock(in_channels=2048, out_channels=1024)
        self.decoder4 = DecoderBlock(in_channels=1024, out_channels=512)
        self.decoder3 = DecoderBlock(in_channels=512, out_channels=256)
        self.decoder2 = DecoderBlock(in_channels=256, out_channels=64)

        # Final part.
        self.final = FinalBlock(in_channels=256, mid_channels=128, out_channels=num_class)

    def forward(self, x):
        # Encoding, compressive pathway.
        out_encoder1 = self.encoder1(x)
        out_encoder2 = self.encoder2(out_encoder1)
        out_encoder3 = self.encoder3(out_encoder2)
        out_encoder4 = self.encoder4(out_encoder3)
        out_encoder5 = self.encoder5(out_encoder4)
        # Decoding, expansive pathway.
        out_decoder5 = self.decoder5(out_encoder5)
        out_decoder4 = self.decoder4(out_decoder5 + out_encoder4)
        out_decoder3 = self.decoder3(out_decoder4 + out_encoder3)
        out_final = self.final(out_decoder3 + out_encoder2)
        return out_final


class SegBrainDataset(Dataset):
    def __init__(self, img_array, h, w, pretrained):
        self.img_array = img_array
        self.w = w
        self.h = h

        T_list = [transforms.ToTensor()]
        if pretrained:
            T_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.T = transforms.Compose(T_list)

    def __getitem__(self, idx):
        # [D, H, W]
        img = Image.fromarray(self.img_array[idx], mode='RGB')
        img = img.resize((self.w, self.h), resample=Image.LANCZOS)
        img = self.T(img)
        return img

    def __len__(self):
        return len(self.img_array)


def seg_brain(img_arry, h, w, batch=1):
    """
    outputs: (256, 256)
    """
    # ---- setting model ----
    model_b = ResUNet50(1, True).to(device)
    model_b = torch.nn.DataParallel(model_b).to(device)
    model_b_ckpt = None
    if str(device) == "cuda":
        model_b_ckpt = torch.load(r"/home/dxy/ais/BrainSegmentation/checkpoints/segBrain.pth.tar")
    elif str(device) == "cpu":
        model_b_ckpt = torch.load(r"/home/dxy/ais/BrainSegmentation/checkpoints/segBrain.pth.tar", map_location='cpu')

    model_b.load_state_dict(model_b_ckpt)
    model_b.eval()
    for param in model_b.parameters():
        param.requires_grad = False

    model_v = ResUNet50(1, True).to(device)
    model_v = torch.nn.DataParallel(model_v).to(device)
    model_v_ckpt = None

    if str(device) == "cuda":
        model_v_ckpt = torch.load(r"/home/dxy/ais/BrainSegmentation/checkpoints/segVentrc.pth.tar")
    elif str(device) == "cpu":
        model_v_ckpt = torch.load(r"/home/dxy/ais/BrainSegmentation/checkpoints/segVentrc.pth.tar", map_location='cpu')

    model_v.load_state_dict(model_v_ckpt)
    model_v.eval()
    for param in model_v.parameters():
        param.requires_grad = False

    # ---- loading data ----
    dataset = SegBrainDataset(img_arry, h, w, pretrained=True)
    data = DataLoader(dataset=dataset, batch_size=batch, num_workers=12, pin_memory=True)
    output_bone = []
    output_ventricle = []
    # ---- start segmenting brain ----
    for idx_batch, img in enumerate(tqdm(data)):
        # ---- inputs ----
        inputs = img.to(device, dtype=torch.float)

        # ---- fp ----
        outputs_b = model_b(inputs)
        outputs_v = model_v(inputs)
        outputs_b = torch.sigmoid(outputs_b)
        outputs_v = torch.sigmoid(outputs_v)  # shape: (bs, 1 ,256, 256)

        # ---- Brain Bone and Ventricle ----
        pd_b = outputs_b.cpu().data.numpy().squeeze(1)
        pd_v = outputs_v.cpu().data.numpy().squeeze(1)
        bone_mask = np.zeros_like(pd_b)
        bone_mask[pd_b >= 0.5] = 1
        ventricle_mask = np.zeros_like(pd_b)
        ventricle_mask[pd_v >= 0.5] = 1
        output_bone += [bone_mask[i] for i in range(bone_mask.shape[0])]
        output_ventricle += [ventricle_mask[i] for i in range(ventricle_mask.shape[0])]
    return np.array(output_bone), np.array(output_ventricle)


def BoneRemove(imgs):
    D, H, W = imgs.shape
    temp_img = []
    for j in range(D):
        temp_img.append(cv2.cvtColor(imgs[j].astype('uint8'), cv2.COLOR_GRAY2BGR))
    imgs = np.array(temp_img)
    bone_lst, ventricle_lst = seg_brain(imgs, 256, 256, 1)
    bone_masks = [cv2.resize(bone_lst[i], (H, W), interpolation=cv2.INTER_CUBIC)
                  for i in range(len(imgs))]
    ventricle_masks = [cv2.resize(ventricle_lst[i], (H, W), interpolation=cv2.INTER_CUBIC)
                       for i in range(len(imgs))]
    return np.array(bone_masks).astype('bool'), np.array(ventricle_masks).astype('bool')


def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out


def removeBone(img):
    img = img.copy()
    brain_mask = img.copy()
    edges = skimage.feature.canny(img)
    mask = img.copy()
    mask[edges != 0] = 0
    bone_threshold = filters.threshold_otsu(mask[mask != 0])
    bone_mask = np.zeros_like(img)
    bone_mask[img > bone_threshold] = 1
    erod_kernel = skimage.morphology.square(3)
    bone_mask = skimage.morphology.erosion(bone_mask, erod_kernel)
    bone_mask = skimage.morphology.erosion(bone_mask, erod_kernel)
    dilation_kernal = skimage.morphology.square(3)
    bone_mask = skimage.morphology.dilation(bone_mask, dilation_kernal)
    brain_mask[bone_mask == 1] = 0
    brain_mask[brain_mask != 0] = 1
    return brain_mask


def brainBoneRemove_morph(imgs):
    d, h, w = imgs.shape
    slia = imgs.max(axis=0)
    template_img = imgs.copy()
    mask, _ = getMaxArea(slia)
    location = np.where(mask == 0)
    ans = []
    for i in tqdm(range(d)):
        template_img[i][location] = 0
        mask, ignore = getMaxArea(template_img[i])
        mask = FillHole(mask)
        img_show = template_img[i].copy()
        img_show[mask == 0] = 0
        brain_mask = removeBone(img_show)

        edge = cv2.Canny(brain_mask.astype('uint8'), 0, 1)
        dilation_kernal = skimage.morphology.square(3)
        edge = skimage.morphology.dilation(edge, dilation_kernal)
        erod_kernel = skimage.morphology.square(3)
        edge = skimage.morphology.erosion(edge, erod_kernel)
        brain_mask[edge != 0] = 0
        erod_kernel = skimage.morphology.square(2)
        brain_mask = skimage.morphology.erosion(brain_mask, erod_kernel)
        dilation_kernal = skimage.morphology.square(3)
        brain_mask = skimage.morphology.dilation(brain_mask, dilation_kernal)
        brain_mask = FillHole(brain_mask)
        ans.append(brain_mask)
    ans = np.array(ans)
    return ans

# if __name__ == "__main__":
#     anat_time_imgs = load4DImage(0, getCaselst(path))
#     bone_lst, ventricle_lst = seg_brain(anat_time_imgs[0], 256, 256, 12)
#     print(bone_lst.shape)
#     print(anat_time_imgs.shape)
