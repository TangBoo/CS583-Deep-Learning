import torch as t
import torch.nn as nn
from torch.nn import functional as F
from Layers import BottleBlock, UpBottomPath, BottomUpPath, ResidualConv, UpSample, AnchorNet, BOXRegression
import Config as config
import time
from torchsummary import summary


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        if name is None:
            prefix = r"/home/aiteam_share/database/ISLES2018_manual_aif/train_3DUnet/checkpoint/" + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)


class ResUNet(nn.Module):
    def __init__(self, InChannel, OutChannel, base=config.Base, depth=config.Depth, mode='seg'):
        """
        :param InChannel: :param OutChannel: :param base: :param depth: :param mode: seg or det Describe: As
        for encode part, control the create of encoder using base and base_, base is always two time of base_,
        base as output channel, base_ as input channel. As for decode part, base_ is as input channel, base is as
        output channel, base is always two time of base_. The bridge skip is connection between encoder and decoder,
        it only expand feature maps as channel dimension
        """
        super(ResUNet, self).__init__()
        if mode.lower() not in ['seg', 'det']:
            raise ValueError
        self.mode = mode
        self.depth = depth
        enumChannel = [base * (2 ** i) for i in range(depth)]
        if mode.lower() == 'det':
            self.normChannel = nn.ModuleList([nn.Conv2d(cha, 256, 1, 1, 0) for cha in enumChannel[1:]])

        self.input_layer = nn.Sequential(
            nn.Conv2d(InChannel, base, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=config.Num_groups, num_channels=base),
            nn.ReLU(),
            nn.Conv2d(base, base, kernel_size=3, padding=1)
        )
        self.input_skip = nn.Conv2d(InChannel, base, kernel_size=3, stride=1, padding=1)
        encoder_layers = []

        # base = 32
        for ii in range(depth - 1):
            base_ = base  # 32, 64, 128, 256
            base *= 2  # 64, 128, 256
            encoder_layers.append(ResidualConv(base_, base, padding=1, down_sample='pool'))

        self.encoder_layers = nn.ModuleList(encoder_layers)
        # ------------bridge between encoder and decoder------------------
        base_ = base  # 256
        base *= 2  # 512
        self.bridge_skip = ResidualConv(base_, base, padding=1, down_sample='none')

        # ------------Decoder------------------
        decoder_layers = []
        for ii in range(depth - 1):  # 1024->512, 512->256, 256->128
            decoder_layers.append(nn.Sequential(
                UpSample(base, stride=2, interpolate='none', smooth=True),
                ResidualConv(base, base_, padding=1, down_sample='none')  # 512 +128, 256 + 64, 128 + 32
            ))
            base = base_ + base_ // 2
            base_ //= 2

        self.decoder_layers = nn.ModuleList(decoder_layers)
        # base->1
        self.output = nn.Sequential(
            ResidualConv(base, base_, padding=1, down_sample='none'),
            nn.Conv2d(base_, OutChannel, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.input_layer(x) + self.input_skip(x)
        Xs = [x]
        for layer in self.encoder_layers:
            x = layer(x)
            Xs.append(x)

        x = self.bridge_skip(x)
        proposals = []
        Xs.pop()
        Xs.reverse()
        for idx, layer in enumerate(self.decoder_layers):
            proposals.append(layer(x))
            x = t.cat((Xs[idx], proposals[-1]), 1)
        x = self.output(x)
        if self.mode.lower() == 'seg':
            return x

        proposals.reverse()
        if hasattr(self, 'normChannel'):
            for i in range(len(proposals)):
                proposals[i] = self.normChannel[i](proposals[i])
        return x, proposals


class ResNet(nn.Module):
    def __init__(self, base=config.Base, InChannel=1, numBlocks=config.ResNetStructure['ResNet50']):
        super(ResNet, self).__init__()
        self.base = base
        planes = [base * (2 ** i) for i in range(len(numBlocks))]
        self.InputLayer = nn.Sequential(
            nn.Conv2d(InChannel, base, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(config.Num_groups, base),
            nn.ReLU(),
            nn.Conv2d(base, base, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.layers = nn.ModuleList(
            [self.__make_layer(planes[i], numBlocks[i], stride=2) for i in range(len(numBlocks))])

    def __make_layer(self, midChannel, numBlocks, stride):
        strides = [stride] + [1] * (numBlocks - 1)
        layers = nn.ModuleList()
        for stride in strides:
            layers.append(BottleBlock(self.base, midChannel, stride))
            self.base = midChannel * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.InputLayer(x)
        features = [x]
        for layer in self.layers:
            x = features[-1]
            features.append(layer(x))
        return features


class FPN(nn.Module):
    def __init__(self, base=config.Base, InChannel=1, OutChannel=1, backbone='ResNet50', IncludeSeg=True):
        super(FPN, self).__init__()
        if config.Output_features > 4:
            raise ValueError("Most number of  output layer for FPN is 4 ")

        if backbone not in config.ResNetStructure:
            raise ValueError

        self.IncludeSeg = IncludeSeg
        self.BackBone = ResNet(base=base, InChannel=InChannel, numBlocks=config.ResNetStructure[backbone])
        self.UpBottomPath = UpBottomPath(IncludeSeg=IncludeSeg, outChannel=OutChannel)

    def forward(self, x):
        features = self.BackBone(x)
        return self.UpBottomPath(features)


class PAN(nn.Module):
    def __init__(self, base=config.Base, InChannel=1, OutChannel=1, BackBone='FPN', IncludeTop=True, IncludeSeg=True):
        super(PAN, self).__init__()
        if BackBone.lower() not in ['fpn', 'unet']:
            raise ValueError
        self.IncludeSeg = IncludeSeg
        self.BackBone = FPN(base=base, InChannel=InChannel, OutChannel=OutChannel) if BackBone.lower() == 'fpn' else ResUNet(base=base, InChannel=InChannel, OutChannel=OutChannel, mode='det')
        self.BottomUpPath = BottomUpPath(InLayers=config.Depth-1, IncludeTop=IncludeTop)

    def forward(self, x):
        features = self.BackBone(x)

        if self.IncludeSeg:
            seg, features = features
            return seg, self.BottomUpPath(features)

        return self.BottomUpPath(features)


class RetinaUNet(BasicModule):
    def __init__(self, base=config.Base, InChannel=1, OutChannel=1, OutLayer=config.Output_features, BackBone='fpn', IncludeTop=True, IncludeSeg=True):
        super(RetinaUNet, self).__init__()
        if BackBone.lower() not in ['fpn', 'unet']:
            raise ValueError
        self.IncludeSeg = IncludeSeg
        self.OutLayer = OutLayer
        self.MainNet = PAN(base, InChannel, OutChannel, BackBone=BackBone, IncludeTop=IncludeTop, IncludeSeg=IncludeSeg)
        self.ANCNets = nn.ModuleList([AnchorNet(256, 256) for i in range(self.OutLayer)])
        self.REGNets = nn.ModuleList([BOXRegression(256, 256) for i in range(self.OutLayer)])

    def forward(self, x):
        features = self.MainNet(x)
        seg = None
        if self.IncludeSeg:
            seg, features = features
        BoxOutputs = []
        ANCOutputs = []
        for i in range(self.OutLayer):
            BoxOutputs.append(self.REGNets[i](features[i]))
            ANCOutputs.append(self.ANCNets[i](features[i]))
        return seg, ANCOutputs, BoxOutputs


if __name__ == "__main__":
    model = nn.DataParallel(RetinaUNet(base=64, InChannel=1, OutChannel=1, BackBone='unet')).cuda(0)
    # model = ResUNet(1, 1, 64, 6, mode='Det').cuda(0)

    # summary(model, input_size=(1, 256, 256))
    x = t.randn(2, 1, 256, 256).cuda()
    seg, anchors, boxes = model(x)
    print(seg.shape)
    print(len(anchors))
    print(len(boxes))
    for idx, a in enumerate(zip(anchors, boxes)):
        print(idx, a[0].shape, a[1].shape)