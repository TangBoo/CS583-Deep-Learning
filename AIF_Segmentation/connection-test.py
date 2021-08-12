import utils
import model
import torch
from einops import rearrange, repeat
from torch import nn
from torchsummary import summary

x = torch.rand((1, 1, 64, 128, 128))
x = rearrange(x, 'b c d h w -> b d (c h w)')
b, t, dim = x.shape
x = x.cuda()
qkv = nn.DataParallel(nn.Linear(dim, dim * 3)).cuda()
out = qkv(x)
print(out.shape)



