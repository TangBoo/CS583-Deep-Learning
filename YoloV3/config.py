import numpy as np
import torch.cuda
from torchvision import transforms as T


def scale(img):
    return ((img - img.min()) / (img.max() - img.min()))


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GridSize = [8, 16, 32]
NUM_CLASSES = 3
NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 512
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
transform = T.Compose([T.ToTensor(), T.Resize((IMAGE_SIZE,IMAGE_SIZE)), scale])


