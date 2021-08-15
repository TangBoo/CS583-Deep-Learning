import torch
import torchvision
import time
import torch.nn as nn
import torch.nn.functional as F
import Config as config
from torch.autograd import Variable as V

import Losses
from utils import Visualizer, get_output, features
from torchsummary import summary
from einops import rearrange, repeat
from utils import get_pth


class Active(nn.Module):
    def __init__(self, inplace=True):
        super(Active, self).__init__()

    @staticmethod
    def forward(x):
        return F.relu(x, inplace=True)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dimension):
        super(EncoderBlock, self).__init__()

        if dimension == "2D":
            self.encoderblock = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, stride=1, padding = 1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, 3, stride=1, padding=1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=out_channels),
                nn.ReLU(inplace=True),
                )
        elif dimension == "3D":
            self.encoderblock = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, 3, stride=1, padding=1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=mid_channels),
                Active(),
                nn.Conv3d(mid_channels, out_channels, 3, stride=1, padding=1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=out_channels),
                Active()
                )

    def forward(self, x):
        x = self.encoderblock(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dimension):
        super(DecoderBlock, self).__init__()

        if dimension == "2D":
            self.decoderblock = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, padding=1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, 3, padding=1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=out_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=out_channels),
                nn.ReLU(inplace=True),
                )
        elif dimension == "3D":
            self.decoderblock = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, 3, padding=1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=mid_channels),
                Active(),
                nn.Conv3d(mid_channels, out_channels, 3, padding=1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=out_channels),
                Active(),
                nn.ConvTranspose3d(out_channels, out_channels, 4, stride=2, padding=1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=out_channels),
                Active()
                )

    def forward(self, x):
        x = self.decoderblock(x)
        return x


class DecoderSpaceBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, factor=2):
        super(DecoderSpaceBlock, self).__init__()
        self.factor = factor
        self.decoderblock = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            nn.GroupNorm(num_groups=config.Num_groups, num_channels=mid_channels),
            Active(),

            nn.Conv3d(mid_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=config.Num_groups, num_channels=out_channels),
            Active(),

            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(1, factor*2, factor*2), stride=(1, factor, factor), padding=(0, factor//2, factor//2)),
            nn.GroupNorm(num_groups=config.Num_groups, num_channels=out_channels),
            Active(),
            nn.MaxPool3d(kernel_size=(factor, 1, 1), stride=(factor, 1, 1))
        )

        self.decoderblock2 = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            nn.GroupNorm(num_groups=config.Num_groups, num_channels=mid_channels),
            Active(),

            nn.ConvTranspose3d(mid_channels, mid_channels, kernel_size=(1, factor * 2, factor * 2),
                               stride=(1, factor, factor), padding=(0, factor // 2, factor // 2)),
            nn.Conv3d(mid_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=config.Num_groups, num_channels=out_channels),
            Active(),
            nn.MaxPool3d(kernel_size=(factor, 1, 1), stride=(factor, 1, 1))
        )

        self.bilinear = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            nn.Upsample(scale_factor=(1, factor, factor), mode='trilinear'),
            nn.Conv3d(mid_channels, out_channels, 1),
            Active()
        )

    def forward(self, x):
        return self.decoderblock(x)


class FinalBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dimension):
        super(FinalBlock, self).__init__()

        if dimension == "2D":
            self.finalblock = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, padding=1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, 1, padding=0)
                )
        elif dimension == "3D":
            self.finalblock = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, 3, padding=1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=mid_channels),
                Active(),
                nn.Conv3d(mid_channels, mid_channels, 3, padding=1),
                nn.GroupNorm(num_groups=config.Num_groups, num_channels=mid_channels),
                Active(),
                nn.Conv3d(mid_channels, out_channels, 1, padding=0),
                )

    def forward(self, x):
        x = self.finalblock(x)
        return x


class SkipConnect(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=True):
        super(SkipConnect, self).__init__()
        self.reduction = reduction
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = Active()
        self.pool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1))

    def forward(self, x):
        x = self.conv3d(x)
        x = self.act(x)
        if self.reduction:
            x = self.pool(x)
        return x


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = r"/home/aiteam_share/database/ISLES2018_manual_aif/train_3DUnet/checkpoint/" + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)


class PAN(nn.Module):
    def __init__(self, base=64):
        super(PAN, self).__init__()
        self.downsample1 = SkipConnect(in_channels=base, out_channels=base, reduction=True)
        self.upsample1 = DecoderSpaceBlock(in_channels=base*2, mid_channels=base, out_channels=base, factor=2)
        self.downsample2 = SkipConnect(in_channels=base, out_channels=base, reduction=True)
        self.upsample2 = DecoderSpaceBlock(in_channels=base*4, mid_channels=base*2, out_channels=base, factor=4)
        self.downsample3 = SkipConnect(in_channels=base, out_channels=base, reduction=True)
        self.upsample3 = DecoderSpaceBlock(in_channels=base*8, mid_channels=base*4, out_channels=base, factor=8)
        self.final = FinalBlock(in_channels=base, mid_channels=base // 2, out_channels=config.Num_class, dimension='3D')

    def forward(self, features):
        x = self.downsample1(features[0]) + self.upsample1(features[1]) #[1, 64, 16, 256, 256]
        x = self.downsample2(x) + self.upsample2(features[2])
        x = self.downsample3(x) + self.upsample3(features[3])
        x = self.final(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, nHead, attn_p=0., proj_p=0., qkv_bias=True):
        '''
        :param dim: the size of flatten feature
        :param nHead: the number of attention heads
        :param attn_p: the dropout probability for qkv linear projections
        :param proj_p: the dropout probability for output
        :param qkv_bias: Are there biases for q, k, v linear projection
        Attrs:
            scale: float
                Normalizing constant for the dot product, prevent large number from dot operation
            qkv:nn.Linear->[batch, time, channel, n_heads, head_dims]
                create query, key, value matrixes using linear projection
            proj:
                project the softmax(query@key), value matrix into output space
            attn_drop, proj_drop:nn.DropOut
        '''
        super(Attention, self).__init__()
        self.n_heads = nHead
        self.dim = dim
        self.head_dim = dim // nHead
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        batch, times, dim = x.shape
        if dim != self.dim:
            raise ValueError
        qkv = self.qkv(x)   #[batch, times + 1, 3 * dim] -> [batch, times + 1, 3, nHead, head_dim]
        qkv = qkv.reshape((batch, times, 3, self.n_heads, self.head_dim))
        qkv = qkv.permute(2, 0, 3, 1, 4)   #[num_mat, batch, n_heads, times, head_dim]
        query, key, value = qkv[0], qkv[1], qkv[2]  #per: [1, batch, n_heads, times, dim // n_heads]
        score = (query @ key.transpose(-2, -1)) * self.scale #[batch, n_heads, times, time]
        score = F.softmax(score, dim=-1) #gate coef: [batch, n_heads, times, times], [0, 1]
        score = self.attn_drop(score)
        weight_avg = score @ value  #[batch, n_heads, times, head_dim]
        weight_avg = weight_avg.transpose(1, 2) #->[batch, times, n_heads, head_dim]
        weight_avg = weight_avg.flatten(2) #->[batch, times, n_heads * head_dim = dim]
        x = self.proj(weight_avg) #[batch, times, dim] * [dim, dim] = [batch, times, dim]
        x = self.proj_drop(x)
        return x


class Vit(nn.Module):
    def __init__(self, img_shape, nheads, attn_drop=0., drop_emb=0.):
        super(Vit, self).__init__()
        self.img_shape = img_shape
        b, c, d, h, w = img_shape
        self.path_size = d + 1
        self.hidden_size = c * h * w
        self.pos_embedding = nn.Parameter(torch.randn(1, self.path_size, self.hidden_size))
        self.cls = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.emb_drop = nn.Dropout(drop_emb)
        self.encoder = Attention(self.hidden_size, nheads, attn_p=attn_drop)
        # self.encoder = nn.TransformerEncoderLayer(self.hidden_size, nhead=nheads, dim_feedforward=self.hidden_size)
        self.to_cls_token = nn.Identity()
        # self.mlp_head = nn.Linear(self.hidden_size, (self.hidden_size // channels))

    def forward(self, x):
        #x : [batch, C, D, H, W]->[batch, D, C*H*W]
        size = x.shape

        if x.shape != self.img_shape:
            raise ValueError
        x = rearrange(x, 'b c d h w->b d (c h w)')
        b, d, _ = x.shape
        cls_token = repeat(self.cls, '() n d -> b n d', b=b)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embedding
        x = self.emb_drop(x)
        x = self.encoder(x)
        x = self.to_cls_token(x[:, 0])
        return x


class UNet3dAttn(BasicModule):
    def __init__(self, num_classes, nheads=8, base=32, attn_drop=0., drop_emb=0.):
        super(UNet3dAttn, self).__init__()
        self.backbone = UNet3d(num_class=num_classes, base=base)
        self.vit = Vit(num_classes, nheads=nheads, channels=1, attn_drop=attn_drop, drop_emb=drop_emb)

    def forward(self, x):
        x = self.backbone(x)
        x = self.vit(x)
        return x


class UNet3d2d(BasicModule):
    def __init__(self, num_class, base=32):
        super().__init__()
        # Encoder part.
        self.encoder1 = EncoderBlock(in_channels=1, mid_channels=base * 1, out_channels=base * 1, dimension="3D")
        self.encoder2 = EncoderBlock(in_channels=base * 1, mid_channels=base * 2, out_channels=base * 2, dimension="3D")
        self.encoder3 = EncoderBlock(in_channels=base * 2, mid_channels=base * 4, out_channels=base * 4, dimension="3D")
        self.encoder4 = EncoderBlock(in_channels=base * 4, mid_channels=base * 8, out_channels=base * 8, dimension="3D")

        # Decoder part.
        self.decoder5 = DecoderBlock(in_channels=base * 8, mid_channels=base * 16, out_channels=base * 16, dimension="3D")
        self.decoder4 = DecoderBlock(in_channels=(8 + 16) * base, mid_channels=base * 8, out_channels=base * 8, dimension="3D")
        self.decoder3 = DecoderBlock(in_channels=(4 + 8) * base, mid_channels=base * 4, out_channels=base * 4, dimension="3D")
        self.decoder2 = DecoderBlock(in_channels=(2 + 4) * base, mid_channels=base * 2, out_channels=base * 2, dimension="3D")
        self.pan = PAN(base=base*2)

    def forward(self, x):
        out_encoder1 = self.encoder1(x)
        out_encoder2 = self.encoder2(F.max_pool3d(out_encoder1, 2, 2))
        out_encoder3 = self.encoder3(F.max_pool3d(out_encoder2, 2, 2))
        out_encoder4 = self.encoder4(F.max_pool3d(out_encoder3, 2, 2))

        # Decoding, expansive pathway.
        out_decoder5 = self.decoder5(F.max_pool3d(out_encoder4, 2, 2))
        out_decoder4 = self.decoder4(torch.cat((out_decoder5, out_encoder4), 1))
        out_decoder3 = self.decoder3(torch.cat((out_decoder4, out_encoder3), 1))
        out_decoder2 = self.decoder2(torch.cat((out_decoder3, out_encoder2), 1))
        input_pan = [out_decoder2, out_decoder3, out_decoder4, out_decoder5]
        final_out = self.pan(input_pan)
        return final_out


class UNet2d(BasicModule):
    def __init__(self, num_class, base=64):
        super().__init__()
        # Encoder part.
        self.encoder1 = EncoderBlock(in_channels=1, mid_channels=base*1, out_channels=base*1, dimension="2D")
        self.encoder2 = EncoderBlock(in_channels=base*1, mid_channels=base*2, out_channels=base*2, dimension="2D")
        self.encoder3 = EncoderBlock(in_channels=base*2, mid_channels=base*4, out_channels=base*4, dimension="2D")
        self.encoder4 = EncoderBlock(in_channels=base*4, mid_channels=base*8, out_channels=base*8, dimension="2D")

        # Decoder part.
        self.decoder5 = DecoderBlock(in_channels=base*8, mid_channels=base*16, out_channels=base*16, dimension="2D")
        self.decoder4 = DecoderBlock(in_channels=(8+16)*base, mid_channels=base*8, out_channels=base*8, dimension="2D")
        self.decoder3 = DecoderBlock(in_channels=(4+8)*base, mid_channels=base*4, out_channels=base*4, dimension="2D")
        self.decoder2 = DecoderBlock(in_channels=(2+4)*base, mid_channels=base*2, out_channels=base*2, dimension="2D")

        # Final part.
        self.final = FinalBlock(in_channels=(1+2)*base, mid_channels=base*1, out_channels=num_class, dimension="2D")

    def forward(self, x):

        # Encoding, compressive pathway.
        out_encoder1 = self.encoder1(x) #[1, 32, 64, 256, 256]
        out_encoder2 = self.encoder2(F.max_pool2d(out_encoder1, 2, 2)) #[1, 64, 32, 256, 256]
        out_encoder3 = self.encoder3(F.max_pool2d(out_encoder2, 2, 2)) #[1, 128]
        out_encoder4 = self.encoder4(F.max_pool2d(out_encoder3, 2, 2)) #[1, 256]

        # Decoding, expansive pathway.
        out_decoder5 = self.decoder5(F.max_pool2d(out_encoder4, 2, 2))
        out_decoder4 = self.decoder4(torch.cat((out_decoder5, out_encoder4), 1))
        out_decoder3 = self.decoder3(torch.cat((out_decoder4, out_encoder3), 1))
        out_decoder2 = self.decoder2(torch.cat((out_decoder3, out_encoder2), 1))
        out_final = self.final(torch.cat((out_decoder2, out_encoder1), 1))
        return out_final


class UNet3d(BasicModule):
    def __init__(self, num_class, base=32):
        super().__init__()

        # Encoder part.
        self.encoder1 = EncoderBlock(in_channels=1, mid_channels=base*1, out_channels=base*1, dimension="3D")
        self.encoder2 = EncoderBlock(in_channels=base*1, mid_channels=base*2, out_channels=base*2, dimension="3D")
        self.encoder3 = EncoderBlock(in_channels=base*2, mid_channels=base*4, out_channels=base*4, dimension="3D")
        self.encoder4 = EncoderBlock(in_channels=base*4, mid_channels=base*8, out_channels=base*8, dimension="3D")

        # Decoder part.
        self.decoder5 = DecoderBlock(in_channels=base*8, mid_channels=base*16, out_channels=base*16, dimension="3D")
        self.decoder4 = DecoderBlock(in_channels=(8+16)*base, mid_channels=base*8, out_channels=base*8, dimension="3D")
        self.decoder3 = DecoderBlock(in_channels=(4+8)*base, mid_channels=base*4, out_channels=base*4, dimension="3D")
        self.decoder2 = DecoderBlock(in_channels=(2+4)*base, mid_channels=base*2, out_channels=base*2, dimension="3D")

        # Final part.
        self.final = FinalBlock(in_channels=(1+2)*base, mid_channels=base*1, out_channels=num_class, dimension="3D")

    def forward(self, x):
        # Encoding, compressive pathway.
        out_encoder1 = self.encoder1(x)
        out_encoder2 = self.encoder2(F.max_pool3d(out_encoder1, 2, 2))
        out_encoder3 = self.encoder3(F.max_pool3d(out_encoder2, 2, 2))
        out_encoder4 = self.encoder4(F.max_pool3d(out_encoder3, 2, 2))

        # Decoding, expansive pathway.
        out_decoder5 = self.decoder5(F.max_pool3d(out_encoder4, 2, 2))
        out_decoder4 = self.decoder4(torch.cat((out_decoder5, out_encoder4), 1))
        out_decoder3 = self.decoder3(torch.cat((out_decoder4, out_encoder3), 1))
        out_decoder2 = self.decoder2(torch.cat((out_decoder3, out_encoder2), 1))
        out_final = self.final(torch.cat((out_decoder2, out_encoder1), 1))

        return out_final



def test():
    config.Time_axis = 64
    base = 16
    #num_classes, nheads=8, base=32, attn_drop=0., drop_emb=0.
    #num_classes, nheads, channels=1, attn_drop=0., drop_emb=0.
    # model = nn.DataParallel(UNet3dAttn(num_classes=config.Num_class, nheads=8, base=base)).cuda()
    model = nn.DataParallel(Vit(img_shape=(1, 128, 32, 16, 16), nheads=8)).cuda()
    # model = nn.DataParallel(nn.Linear(128*32*32, 128*32*32*3)).cuda()
    # model = nn.DataParallel(UNet3d(num_class=config.Num_class, base=16)).cuda()

    # y = torch.rand(1, 128, 32, 32, 32).cuda()
    print(model.module['emb_drop'])


if __name__ == "__main__":
    test()
