import numpy as np
import torch.nn as nn
import torch as t
import Config as config
from einops import rearrange, repeat
from torch.nn import functional as F


class BottleBlock(nn.Module):
    def __init__(self, inChannel, midChannel, stride=1, expfactor=4):
        super(BottleBlock, self).__init__()
        self.norm1 = nn.GroupNorm(config.Num_groups, inChannel)
        self.conv1 = nn.Conv2d(inChannel, midChannel, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.GroupNorm(config.Num_groups, midChannel)
        self.conv2 = nn.Conv2d(midChannel, midChannel, kernel_size=3, stride=stride, padding=1, bias=False)

        self.norm3 = nn.GroupNorm(config.Num_groups, midChannel)
        self.conv3 = nn.Conv2d(midChannel, int(midChannel * expfactor), kernel_size=1, stride=1, bias=False)

        if stride != 1 or inChannel != expfactor * midChannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannel, int(expfactor * midChannel), kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.norm2(out)))
        out = self.conv3(F.relu(self.norm3(out)))
        out += shortcut
        return out


class ESPCN(nn.Module):
    def __init__(self, inChannel, scale_factor):
        super(ESPCN, self).__init__()
        self.Block = nn.Sequential(
            nn.Conv2d(inChannel, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, inChannel * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        x = self.Block(x)
        return self.output(x)


class ResidualConv(nn.Module):
    def __init__(self, input_channel, output_channel, padding=1, down_sample='conv'):
        super(ResidualConv, self).__init__()
        if down_sample not in ['conv', 'pool', 'none']:
            raise ValueError
        self.downSample = down_sample
        stride = 1 if down_sample == 'pool' or down_sample == 'none' else 2
        self.conv_block = nn.Sequential(
            nn.GroupNorm(num_groups=config.Num_groups, num_channels=input_channel),
            nn.ReLU(),
            nn.Conv2d(input_channel, output_channel, stride=stride, kernel_size=3, padding=padding),

            nn.GroupNorm(num_groups=config.Num_groups, num_channels=output_channel),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=config.Num_groups, num_channels=output_channel)
        )

        if stride == 1 and down_sample == 'pool':
            self.pooling = nn.MaxPool2d(stride=2, kernel_size=2)
        else:
            self.pooling = lambda x: x

    def forward(self, x):
        x = self.conv_block(x) + self.conv_skip(x)
        return self.pooling(x)


class BOXRegression(nn.Module):
    def __init__(self, InChannels, OutChannels):
        super(BOXRegression, self).__init__()
        self.num_anchors = len(config.Area_ratios) * len(config.HW_ratio)
        self.Blocks = nn.Sequential(
            nn.Conv2d(InChannels, OutChannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(OutChannels, OutChannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(OutChannels, OutChannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(OutChannels, self.num_anchors * 4, kernel_size=1, stride=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        output = self.Blocks(x)  # eg. b, 9, h, w, 4
        return output.reshape((b, self.num_anchors, h, w, 4))


class AnchorNet(nn.Module):
    def __init__(self, InChannels, OutChannels):
        super(AnchorNet, self).__init__()
        self.num_anchors = len(config.Area_ratios) * len(config.HW_ratio)
        self.Blocks = nn.Sequential(
            nn.Conv2d(InChannels, OutChannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(OutChannels, OutChannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(OutChannels, OutChannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(OutChannels, self.num_anchors, kernel_size=1, stride=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        output = self.Blocks(x)
        return output.reshape((b, self.num_anchors, h, w, 1))


class UpSample(nn.Module):
    def __init__(self, inChannel, stride=2, interpolate=config.Interpolate, smooth=False):
        super(UpSample, self).__init__()
        if interpolate.lower() not in [None, 'none', 'nearest', 'bilinear', 'bicubic', 'pixelshuffle']:
            raise ValueError

        if interpolate.lower() in ['nearest', 'bilinear', 'bicubic']:
            self.upsample = nn.Upsample(scale_factor=stride, mode=interpolate)

        elif interpolate.lower() == 'pixelshuffle':
            self.upsample = ESPCN(inChannel=inChannel, scale_factor=stride)
        else:
            kernel = stride * 2 - stride % 2
            self.upsample = nn.ConvTranspose2d(inChannel, inChannel, kernel_size=kernel, stride=stride, padding=1)

        if smooth:
            self.conv = nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.upsample(x)
        if hasattr(self, "conv"):
            x = self.conv(x)
        return x


class UpBottomPath(nn.Module):
    def __init__(self, interpolate=config.Interpolate, outChannel=1, NumLayer=config.Output_features, IncludeSeg=True):
        super(UpBottomPath, self).__init__()
        self.base = config.Base
        self.numBlocks = NumLayer
        enumChannel = [4 * self.base * (2 ** i) for i in range(NumLayer - 1, -1, -1)] + [self.base]
        enumChannel.reverse()

        self.upSamples = nn.ModuleList([UpSample(256, interpolate=interpolate) for _ in range(NumLayer + 1)])
        self.convLayers = nn.ModuleList([nn.Conv2d(enumChannel[i], 256, 1, 1, 0) for i in range(NumLayer + 1)])
        self.checkerRemove = nn.ModuleList([nn.Conv2d(256, 256, 3, 1, 1) for _ in range(NumLayer + 1)])

        if IncludeSeg:
            self.SegOut = BottleBlock(256, self.base, stride=1, expfactor=outChannel / self.base)

    def forward(self, features):
        if len(features) != self.numBlocks + 1:
            raise ValueError("features length {}, Blocks Length {}, Not Match".format(len(features), self.numBlocks + 1))

        output = []
        x = self.convLayers[-1](features[-1])
        output.append(x)

        for i in range(self.numBlocks-1, -1, -1):
            x_ = self.convLayers[i](features[i])
            x = self.upSamples[i](x) + x_
            x = self.checkerRemove[i](x)
            output.append(x)

        seg = output[-1]
        output.reverse()
        if hasattr(self, "SegOut"):
            x = self.SegOut(seg)
            return x, output
        return output


class BottomUpPath(nn.Module):
    def __init__(self, InLayers=config.Depth, OutLayers=config.Output_features, IncludeTop=True):
        super(BottomUpPath, self).__init__()
        self.InLayer = InLayers
        self.OutLayer = OutLayers
        self.NumLayer = InLayers - 1 if IncludeTop else OutLayers
        self.DownSamples = nn.ModuleList([nn.Conv2d(256, 256, 3, 2, 1) for i in range(self.NumLayer)])
        self.IncludeTop = IncludeTop

    def forward(self, features):

        if self.InLayer != len(features):
            raise ValueError('Number of input layers ,{} is not match InLayer {}'.format(len(features), self.InLayer))

        if self.IncludeTop:
            top = features[0]
            features = features[1:]
        else:
            outIdx = len(features) - self.NumLayer
            top = features[outIdx-1]
            features = features[outIdx:]

        outputs = [top]
        for idx, downLayer in enumerate(self.DownSamples):
            x = outputs[-1]
            x = F.relu(downLayer(x)) + features[idx]
            outputs.append(x)
        return outputs[-self.OutLayer:]


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
                create query, key, value matrices using linear projection
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
        qkv = self.qkv(x)  # [batch, times + 1, 3 * dim] -> [batch, times + 1, 3, nHead, head_dim]
        qkv = qkv.reshape((batch, times, 3, self.n_heads, self.head_dim))
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [num_mat, batch, n_heads, times, head_dim]
        query, key, value = qkv[0], qkv[1], qkv[2]  # per: [1, batch, n_heads, times, dim // n_heads]
        score = (query @ key.transpose(-2, -1)) * self.scale  # [batch, n_heads, times, time]
        score = F.softmax(score, dim=-1)  # gate coef: [batch, n_heads, times, times], [0, 1]
        score = self.attn_drop(score)
        weight_avg = score @ value  # [batch, n_heads, times, head_dim]
        weight_avg = weight_avg.transpose(1, 2)  # ->[batch, times, n_heads, head_dim]
        weight_avg = weight_avg.flatten(2)  # ->[batch, times, n_heads * head_dim = dim]
        x = self.proj(weight_avg)  # [batch, times, dim] * [dim, dim] = [batch, times, dim]
        x = self.proj_drop(x)
        return x


class Vit(nn.Module):
    def __init__(self, img_shape, nheads, attn_drop=0., drop_emb=0.):
        super(Vit, self).__init__()
        self.img_shape = img_shape
        b, c, d, h, w = img_shape
        self.path_size = d + 1
        self.hidden_size = c * h * w
        self.pos_embedding = nn.Parameter(t.randn(1, self.path_size, self.hidden_size))
        self.cls = nn.Parameter(t.randn(1, 1, self.hidden_size))
        self.emb_drop = nn.Dropout(drop_emb)
        self.encoder = Attention(self.hidden_size, nheads, attn_p=attn_drop)
        # self.encoder = nn.TransformerEncoderLayer(self.hidden_size, nhead=nheads, dim_feedforward=self.hidden_size)
        self.to_cls_token = nn.Identity()
        # self.mlp_head = nn.Linear(self.hidden_size, (self.hidden_size // channels))

    def forward(self, x):
        # x : [batch, C, D, H, W]->[batch, D, C*H*W]
        size = x.shape
        if x.shape != self.img_shape:
            raise ValueError
        x = rearrange(x, 'b c d h w->b d (c h w)')
        b, d, _ = x.shape
        cls_token = repeat(self.cls, '() n d -> b n d', b=b)
        x = t.cat((cls_token, x), dim=1)
        x += self.pos_embedding
        x = self.emb_drop(x)
        x = self.encoder(x)
        x = self.to_cls_token(x[:, 0])
        return x


if __name__ == '__main__':
    # model = BottomUpPath(InLayers=config.Depth, OutLayers=4, IncludeTop=True).cuda(0)
    model = UpBottomPath().cuda(0)
    print(model)
    features = [t.rand(1, 64, 256, 256).cuda(), t.rand(1, 256, 128, 128).cuda(), t.rand(1, 512, 64, 64).cuda(), t.rand(1, 1024, 32, 32).cuda(), t.rand(1, 2048, 16, 16).cuda()]
    # features.reverse()
    reg, res = model(features)
    print('reg: ',reg.shape)
    for idx, r in enumerate(res):
        print(idx, r.shape)
