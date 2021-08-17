import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import Config as config


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


class UpSample(nn.Module):
    def __init__(self, inChannel, stride=2, interpolate='none'):
        super(UpSample, self).__init__()
        if interpolate.lower() not in [None, 'none', 'nearest', 'bilinear', 'bicubic', 'pixelshuffle']:
            raise ValueError

        if interpolate.lower() != 'none' and 'pixelshuffle' and interpolate is not None:
            self.upsample = nn.Upsample(scale_factor=stride, mode=interpolate)
        elif interpolate.lower() == 'pixelshuffle':
            self.pixelshuffle = ESPCN(inChannel=inChannel, scale_factor=stride)
        else:
            kernel = stride * 2 - stride % 2
            self.upsample = nn.ConvTranspose2d(inChannel, inChannel, kernel_size=kernel, stride=stride, padding=1)

    def forward(self, x):
        return self.upsample(x)


class UpBottomPath(nn.Module):
    def __init__(self, interpolate='nearest', outChannel=1, NumLayer=4, IncludeSeg=True):
        super(UpBottomPath, self).__init__()
        self.base = config.Base
        self.numBlocks = NumLayer
        enumChannel = [4 * self.base * (2 ** i) for i in range(NumLayer - 1, -1, -1)] + [self.base]
        enumChannel.reverse()
        self.upSamples = nn.ModuleList([UpSample(enumChannel[i], interpolate=interpolate) for i in range(NumLayer + 1)])
        self.convLayers = nn.ModuleList([nn.Conv2d(enumChannel[i], 256, 1, 1, 0) for i in range(NumLayer + 1)])
        self.checkerRemove = nn.ModuleList([nn.Conv2d(256, 256, 3, 1, 1) for i in range(NumLayer + 1)])
        self.IncludeSeg = IncludeSeg

        if IncludeSeg:
            self.SegOut = BottleBlock(256, self.base, stride=1, expfactor=outChannel / self.base)

    def forward(self, features):
        if len(features) != self.numBlocks + 1:
            raise ValueError

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
        if self.IncludeSeg:
            x = self.SegOut(seg)
            return x, output
        return output


class BottomUpPath(nn.Module):
    def __init__(self, InLayers=config.Depth, OutLayers=4, IncludeTop=True):
        super(BottomUpPath, self).__init__()
        self.InLayer = InLayers
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
            # print(F.relu(downLayer(x)).shape, features[idx].shape)
            x = F.relu(downLayer(x)) + features[idx]
            outputs.append(x)
        return outputs[1:]


if __name__ == "__main__":
    model = BottomUpPath(InLayers=config.Depth, OutLayers=4, IncludeTop=True).cuda(0)
    features = [t.rand(1, 256, 256, 256).cuda(), t.rand(1, 256, 128, 128).cuda(), t.rand(1, 256, 64, 64).cuda(), t.rand(1, 256, 32, 32).cuda(), t.rand(1, 256, 16, 16).cuda(), t.rand(1, 256, 8, 8).cuda()]
    features.reverse()
    res = model(features)

    for idx, r in enumerate(res):
        print(idx, r.shape)








        






