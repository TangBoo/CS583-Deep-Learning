import torch.nn as nn
import torch
import Config as config
from einops import rearrange, repeat
from torch.nn import functional as F


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


class PANet(nn.Module):
    def __init__(self, base, depth, Num_features=config.Output_features, IncludeTop=True):
        super(PANet, self).__init__()
        if Num_features < depth - 3:
            raise ValueError("Depth is not enough for output {} features map".format(Num_features))
        self.includeTop = IncludeTop
        self.depth = depth
        self.base = base
        self.outIdx = [i for i in range(depth - Num_features - 2, depth)][Num_features:]


class Upsample(nn.Module):
    def __init__(self, input_channel, stride, interpolate='none'):
        super(Upsample, self).__init__()
        if interpolate not in ['none', 'bilinear', 'nearest', 'bicubic']:
            raise ValueError
        self.interpolate = interpolate
        if interpolate == 'none':
            kernel = 2 * stride - stride % 2
            self.upsample = nn.ConvTranspose2d(input_channel, input_channel, kernel_size=kernel, stride=stride,
                                               padding=1)
        elif interpolate is not 'none':
            self.upsample = nn.Upsample(scale_factor=stride, mode=interpolate)
        self.conv = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
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


if __name__ == '__main__':
    res_block = ResidualConv(64, 128, 1, 'pool').cuda()
    x = torch.rand((1, 64, 256, 256)).cuda()
    res = res_block(x)
