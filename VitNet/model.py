import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class Active(nn.Module):
    def __init__(self, inplace=True):
        super(Active, self).__init__()

    @staticmethod
    def forward(x):
        return F.relu(x, inplace=True)


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
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
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
    def __init__(self, img_shape, num_classes, nheads, channels=1, attn_drop=0., drop_emb=0.):
        super(Vit, self).__init__()
        self.path_size = img_shape[-3] + 1
        self.hidden_size = channels * img_shape[-1] * img_shape[-2]
        self.pos_embedding = nn.Parameter(torch.randn(1, self.path_size, self.hidden_size))
        self.cls = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.emb_drop = nn.Dropout(drop_emb)
        self.encoder = Attention(self.hidden_size, nheads, attn_p=attn_drop)
        # self.encoder = nn.TransformerEncoderLayer(self.hidden_size, nhead=nheads, dim_feedforward=self.hidden_size)
        self.to_cls_token = nn.Identity()
        # self.mlp_head = nn.Linear(self.hidden_size,  num_classes * (self.hidden_size // channels))

    def forward(self, x):
        #x : [batch, C, D, H, W]->[batch, D, C*H*W]
        x = rearrange(x, 'b c d h w->b d (c h w)')
        b, d, _ = x.shape
        cls_token = repeat(self.cls, '() n d -> b n d', b=b)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embedding
        x = self.emb_drop(x)
        x = self.encoder(x)
        x = self.to_cls_token(x[:, 0])
        return x




def test():
    Time_axis = 64
    base = 16
    #num_classes, nheads=8, base=32, attn_drop=0., drop_emb=0.
    #num_classes, nheads, channels=1, attn_drop=0., drop_emb=0.
    # model = nn.DataParallel(UNet3dAttn(num_classes=config.Num_class, nheads=8, base=base)).cuda()
    encoder = nn.DataParallel(Vit((64, 256, 256), 1, 4, 1)).cuda()
    # encoder = nn.DataParallel(Attention(dim=128*128, nHead=8)).cuda()
    # encoder = nn.DataParallel(nn.TransformerEncoderLayer(d_model=128*128, nhead=8, dim_feedforward=128*128, dropout=0.1)).cuda()
    # img_shape = (2, 1) + config.Image_shape
    # x = V(torch.randn((2, 64, 128*128)).cuda())
    # out = encoder(x)
    # print(out.shape)
    print(encoder)
    # print(summary(model, input_size=(1, config.Time_axis, 256, 256)))
    # print(model)


if __name__ == "__main__":
    test()
